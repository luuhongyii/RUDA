import math
from functools import partial

import torch

from . import _C
from ._indexing import _logical
from ._ops import _numeric_dtypes, _integral_dtypes
from ._ordering import _output
from ._sparse_compressed import _layouts


def _parts(value):
    row = value.layout in (torch.sparse_csr, torch.sparse_bsr)
    return (value.crow_indices(), value.col_indices()) if row else (value.ccol_indices(), value.row_indices())


def _make(reference, pointers, indices, values):
    return torch.ops.aten._sparse_compressed_tensor_unsafe.default(
        pointers, indices, values, list(reference.shape), dtype=values.dtype, layout=reference.layout, device=reference.device)


def _finish(result, out):
    if out is None:
        return result
    if result.dtype != out.dtype:
        result = _make(result, *_parts(result), result.values().to(out.dtype))
    _C.compressed_arithmetic_output(out, result)
    return out


def _coefficient(alpha, dtype, device, subtract=False):
    checked_dtype = torch.complex64 if dtype == torch.complex32 else dtype
    _C.arithmetic_alpha(alpha, checked_dtype, dtype, subtract)
    return torch.scalar_tensor(-alpha if subtract else alpha, dtype=dtype, device=device)


def _union(self, other, dtype, coefficient):
    ap, aj = _parts(self)
    bp, bj = _parts(other)
    batch_shape = tuple(ap.shape[:-1]); batches = math.prod(batch_shape); rows = ap.shape[-1]-1
    if (ap.shape != bp.shape or aj.shape[:-1] != bj.shape[:-1]
            or self.values().shape[len(batch_shape)+1:] != other.values().shape[len(batch_shape)+1:]):
        raise RuntimeError('compressed sparse addition requires matching batch dimensions, blocks and dense payload shapes')
    value_shape = tuple(self.values().shape[len(batch_shape)+1:]); width = math.prod(value_shape)
    if batches*rows*max(width, 1) > 2**32-1:
        raise RuntimeError('compressed sparse addition exceeds native U32 work indexing')
    index_dtype = torch.promote_types(ap.dtype, bp.dtype)
    ap, aj, bp, bj = (_logical(t).to(torch.int64).reshape(batches, t.shape[-1]) for t in (ap, aj, bp, bj))
    counts = torch.zeros((batches, rows+1), dtype=torch.int64, device=self.device)
    if batches and rows:
        _C.scientific_call(525, [ap, aj, bp, bj, counts], [], [])
    pointers = counts.cumsum(-1, dtype=torch.int64)
    del counts
    nnz = int(pointers[0, -1].item()) if batches else 0
    if batches > 1 and not bool(torch.all(pointers[:, -1] == nnz).item()):
        raise RuntimeError('compressed sparse addition requires the same number of stored result entries in every batch')
    if nnz > (2**31-1 if index_dtype == torch.int32 else 2**32-1) or batches*nnz*max(width, 1) > 2**32-1:
        raise RuntimeError('compressed sparse sum exceeds native index/storage limits')
    indices = torch.empty((batches, nnz), dtype=torch.int64, device=self.device)
    values = torch.empty((batches, nnz, width), dtype=dtype, device=self.device)
    if batches and nnz:
        left = _logical(self.values()).to(dtype).reshape(batches, self._nnz(), width)
        right = _logical(other.values()).to(dtype).reshape(batches, other._nnz(), width)
        _C.scientific_call(526, [ap, aj, bp, bj, pointers, left, right, indices, values, coefficient], [], [])
    return _make(self, pointers.to(index_dtype).reshape(batch_shape+(rows+1,)),
                 indices.to(index_dtype).reshape(batch_shape+(nnz,)), values.reshape(batch_shape+(nnz,)+value_shape))


def _dense(self, other, dtype, coefficient, *, out):
    sparse = other if self.layout == torch.strided else self
    dense = self if self.layout == torch.strided else other
    if self.layout == torch.strided:
        result = _logical(dense).to(dtype).clone(memory_format=torch.contiguous_format)
    else:
        result = torch.mul(_logical(dense).to(dtype), coefficient).contiguous()
        coefficient = torch.ones((), dtype=dtype, device=self.device)
    pointers, indices = _parts(sparse)
    batch_shape = tuple(pointers.shape[:-1]); batches = math.prod(batch_shape); axis = len(batch_shape)
    rows, columns = sparse.shape[axis:axis+2]
    width = math.prod(sparse.shape[axis+2:])
    block = tuple(sparse.values().shape[axis+1:axis+3]) if sparse.layout in (torch.sparse_bsr, torch.sparse_bsc) else (1, 1)
    if result.numel() and sparse._nnz():
        pointers, indices = (_logical(t).to(torch.int64).reshape(batches, t.shape[-1]) for t in (pointers, indices))
        values = _logical(sparse.values()).to(dtype).reshape(batches, sparse._nnz(), *block, width)
        column = sparse.layout in (torch.sparse_csc, torch.sparse_bsc)
        _C.scientific_call(527, [pointers, indices, values, result.reshape(batches, rows, columns, width), coefficient], [int(column)], [])
    if out is not None:
        for part in (*_parts(sparse), sparse.values(), dense):
            _C.check_inplace(out, part)
    return _output(result, out, (), cast=True)


def add(self, other, *, alpha=1, out=None, inplace=False, subtract=False):
    supported = _numeric_dtypes + _integral_dtypes + (torch.complex32,)
    for value in (self, other):
        if (value.device.type != torch._C._get_privateuse1_backend_name() or value.dtype not in supported
                or value.layout != torch.strided and value.layout not in _layouts):
            raise RuntimeError('compressed addition requires ordinary numeric RUDA tensors with strided or compressed storage')
    if self.device != other.device or self.shape != other.shape:
        raise RuntimeError('compressed addition requires matching shapes and devices; broadcasting is not supported')
    if subtract and (self.dtype == torch.bool or other.dtype == torch.bool):
        raise RuntimeError('subtraction with a bool operand is not supported')
    sparse_pair = self.layout in _layouts and other.layout in _layouts
    if not sparse_pair and self.layout == other.layout:
        raise RuntimeError('compressed addition requires at least one compressed sparse operand')
    if sparse_pair and (self.layout != other.layout or self.dense_dim() != other.dense_dim()):
        raise RuntimeError('sparse-sparse addition requires matching compressed layouts and dense dimensions')
    dtype = torch.result_type(self, other)
    layout = self.layout if sparse_pair else torch.strided
    if inplace:
        out = self
    if out is not None:
        if out.device != self.device or out.layout != layout or not torch.can_cast(dtype, out.dtype):
            raise RuntimeError('compressed addition output has incompatible dtype, device or layout')
    if sparse_pair and self._cdata == other._cdata:
        operation = torch.sub if subtract else torch.add
        if out is not None and out._cdata == self._cdata:
            operation(self.values(), other.values(), alpha=alpha, out=self.values())
            return out
        return _finish(_make(self, *_parts(self), operation(self.values(), other.values(), alpha=alpha)), out)
    coefficient = _coefficient(alpha, dtype, self.device, subtract)
    if not sparse_pair:
        return _dense(self, other, dtype, coefficient, out=out)
    return _finish(_union(self, other, dtype, coefficient), out)


def register(library):
    for name, subtract in (('add', False), ('sub', True)):
        library.impl(name+'.Tensor', partial(add, subtract=subtract))
        library.impl(name+'.out', partial(add, subtract=subtract))
        library.impl(name+'_.Tensor', partial(add, subtract=subtract, inplace=True))
