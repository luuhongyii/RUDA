import math

import torch

from . import _C
from ._indexing import _logical
from ._sparse_coo import _input, _make, _sorted, _sparse_output


def _empty(sparse, shape, out):
    dense_dim = sparse.dense_dim()
    sparse_dim = len(shape) - dense_dim
    indices = torch.empty((sparse_dim, 0), dtype=torch.int64, device=sparse.device)
    values = torch.empty((0, *shape[sparse_dim:]), dtype=out.dtype, device=sparse.device)
    return _sparse_output(_make(indices, values, shape, True), out)


def _dense_sparse(dense, sparse, out):
    shape = torch.broadcast_shapes(dense.shape, sparse.shape)
    if not sparse._nnz() or not sparse.numel() or not dense.numel():
        return _empty(sparse, shape, out)
    value = sparse.coalesce() if dense.ndim > sparse.ndim else sparse
    leading = max(0, dense.ndim - sparse.ndim)
    sd = value.sparse_dim() + leading
    trailing = shape[sd:]
    nnz = value._nnz()
    if dense.ndim <= value.dense_dim():
        indices = _logical(value._indices()).clone()
        values = torch.mul(_logical(dense), _logical(value._values()))
    else:
        batches = math.prod(shape[:leading])
        expanded_nnz = nnz * batches
        if expanded_nnz * max(1, sd) > 2**32 - 1:
            raise RuntimeError('sparse product coordinates exceed native U32 indexing')
        indices = torch.empty((sd, expanded_nnz), dtype=torch.int64, device=sparse.device)
        filtered = torch.empty((expanded_nnz, *trailing), dtype=dense.dtype, device=sparse.device)
        _C.scientific_call(512, [_logical(value._indices()), _logical(dense).expand(shape), indices, filtered], [leading], [])
        if leading:
            values = torch.mul(filtered.reshape(batches, nnz, *trailing), _logical(value._values()))
            values = values.reshape(expanded_nnz, *trailing)
        else:
            values = torch.mul(filtered, _logical(value._values()))
    if out._cdata == sparse._cdata and values.dtype != out.dtype:
        raise RuntimeError('in-place dense/sparse multiplication cannot replace values with a different dtype')
    result = _make(indices, values.to(out.dtype), shape, value.is_coalesced())
    return _sparse_output(result, out)


def _scalar_sparse(zero_dim, other, out):
    if zero_dim.is_coalesced():
        scalar, sparse = zero_dim, other
    elif other.ndim == 0 and other.is_coalesced():
        scalar, sparse = other, zero_dim
    else:
        scalar, sparse = zero_dim.coalesce(), other
    if not scalar._nnz():
        return _empty(sparse, torch.broadcast_shapes(scalar.shape, sparse.shape), out)
    return _dense_sparse(_logical(scalar._values()).reshape(()), sparse, out)


def _sparse_sparse(self, other, out):
    if other.ndim == 0:
        return _scalar_sparse(other, self, out)
    if self.ndim == 0:
        return _scalar_sparse(self, other, out)
    if self.shape == other.shape and (not self._nnz() or not other._nnz()):
        return _empty(self, self.shape, out)
    sd = self.sparse_dim()
    if self.ndim != other.ndim or sd != other.sparse_dim() or self.shape[:sd] != other.shape[:sd]:
        raise RuntimeError('COO multiplication requires equal ranks, sparse dimension counts and sparse dimension sizes')
    shape = torch.broadcast_shapes(self.shape, other.shape)
    dtype = torch.result_type(self, other)
    if not torch.can_cast(dtype, out.dtype):
        raise RuntimeError('sparse product cannot cast the common dtype to the output')
    if self.is_coalesced() != other.is_coalesced():
        search, source = (self, other) if self.is_coalesced() else (other, self)
    else:
        search, source = (self, other) if self._nnz() >= other._nnz() else (other, self)
        positions = math.prod(search.shape[:sd])
        if positions and search._nnz() // positions > 50:
            search = search.coalesce()
    indices = _logical(source._indices()).clone()
    values = torch.empty((source._nnz(), *shape[sd:]), dtype=dtype, device=self.device)
    if values.numel():
        identity = search.is_coalesced()
        search_indices = _logical(search._indices())
        order = torch.empty((0,), dtype=torch.int32, device=self.device) if identity else _sorted(search_indices)[0]
        begin = torch.empty((source._nnz(),), dtype=torch.int32, device=self.device)
        end = torch.empty_like(begin)
        _C.scientific_call(513, [search_indices, indices, order, begin, end], [int(identity)], [])
        source_values = _logical(source._values()).to(dtype).expand(source._nnz(), *shape[sd:])
        search_values = _logical(search._values()).to(dtype).expand(search._nnz(), *shape[sd:])
        _C.scientific_call(514, [source_values, search_values, order, begin, end, values], [int(identity)], [])
    return _sparse_output(_make(indices, values.to(out.dtype), shape, source.is_coalesced()), out)


def mul_out(self, other, *, out):
    if self.layout != torch.sparse_coo and other.layout != torch.sparse_coo:
        raise RuntimeError('sparse multiplication requires a COO operand')
    sparse = self if self.layout == torch.sparse_coo else other
    _input(sparse, sparse=True)
    _input(out, sparse=True)
    if out.device != sparse.device:
        raise RuntimeError('sparse multiplication output must use the operand device')
    for value in (self, other):
        if value.layout == torch.sparse_coo:
            _input(value, sparse=True)
        elif value.device.type == 'cpu' and value.ndim == 0:
            continue
        else:
            _input(value, sparse=False)
        if value.device != sparse.device:
            raise RuntimeError('sparse multiplication operands must share a device except CPU scalars')
    if self.layout == torch.sparse_coo and other.layout == torch.sparse_coo:
        return _sparse_sparse(self, other, out)
    dense = other if self.layout == torch.sparse_coo else self
    return _dense_sparse(dense, sparse, out)
