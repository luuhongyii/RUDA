import math
import operator

import torch

from . import _C
from ._indexing import _logical
from ._ops import _numeric_dtypes, _integral_dtypes
from ._ordering import _output

_libraries = []


def _input(value, *, sparse):
    layout = torch.sparse_coo if sparse else torch.strided
    if value.layout != layout or value.device.type != torch._C._get_privateuse1_backend_name():
        raise RuntimeError(f'COO operation requires {layout} storage on RUDA')
    if value.dtype not in _numeric_dtypes + _integral_dtypes + (torch.complex32,):
        raise RuntimeError('unsupported COO value dtype')


def _parts(value):
    value = _logical(value)
    return torch.view_as_real(value) if value.is_complex() else value


def _make(indices, values, shape, coalesced):
    sparse_dim = indices.shape[0]
    return torch.ops.aten._sparse_coo_tensor_with_dims_and_tensors.default(
        sparse_dim, len(shape) - sparse_dim, list(shape), indices, values,
        dtype=values.dtype, layout=torch.sparse_coo, device=values.device, is_coalesced=coalesced)


def _sorted(indices):
    nnz = indices.shape[1]
    if nnz > 2**32 - 1:
        raise RuntimeError('COO indices exceed native U32 indexing')
    order = torch.empty((nnz,), device=indices.device, dtype=torch.int32)
    groups = torch.empty_like(order)
    starts = torch.empty_like(order)
    count = torch.zeros((1,), device=indices.device, dtype=torch.int64)
    if nnz:
        _C.scientific_call(502, [indices, order, groups, starts, count], [], [])
    return order, starts, count


def coalesce(self):
    _input(self, sparse=True)
    if self.is_coalesced():
        raise RuntimeError('_coalesce requires an uncoalesced tensor; use coalesce()')
    if self._nnz() < 2:
        return self.clone()._coalesced_(True)
    indices, values = _logical(self._indices()), _logical(self._values())
    order, starts, count = _sorted(indices)
    size = int(count.item())
    new_indices = torch.empty((self.sparse_dim(), size), dtype=torch.int64, device=self.device)
    new_values = torch.empty((size, *values.shape[1:]), dtype=self.dtype, device=self.device)
    _C.scientific_call(503, [indices, _parts(values), order, starts, new_indices, _parts(new_values)], [], [])
    return _make(new_indices, new_values, self.shape, True)


def dense_to_sparse_dim(self, sparse_dim):
    _input(self, sparse=False)
    sparse_dim = operator.index(sparse_dim)
    if not 0 <= sparse_dim <= self.ndim or (self.ndim and sparse_dim == 0):
        raise RuntimeError('dense_to_sparse requires 0 < sparse_dim <= ndim, or sparse_dim=0 for a scalar')
    value = _logical(self)
    rows = math.prod(self.shape[:sparse_dim])
    if rows > 2**32 - 1:
        raise RuntimeError('dense COO block count exceeds native U32 indexing')
    if self.numel() == 0:
        indices = torch.empty((sparse_dim, 0), dtype=torch.int64, device=self.device)
        values = torch.empty((0, *self.shape[sparse_dim:]), dtype=self.dtype, device=self.device)
        return _make(indices, values, self.shape, True)
    prefix = torch.empty((rows,), dtype=torch.int32, device=self.device)
    count = torch.empty((1,), dtype=torch.int64, device=self.device)
    _C.scientific_call(504, [_parts(value), prefix, count], [sparse_dim], [])
    nnz = int(count.item())
    indices = torch.empty((sparse_dim, nnz), dtype=torch.int64, device=self.device)
    values = torch.empty((nnz, *self.shape[sparse_dim:]), dtype=self.dtype, device=self.device)
    if nnz:
        _C.scientific_call(505, [value, prefix, indices, values], [], [])
    return _make(indices, values, self.shape, True)


def dense_to_sparse(self, *, layout=None, blocksize=None, dense_dim=None):
    if layout not in (None, torch.sparse_coo):
        return _C.compressed_from_dense(self, layout, blocksize, dense_dim)
    if blocksize is not None:
        raise RuntimeError('COO conversion does not accept blocksize')
    dense_dim = 0 if dense_dim is None else operator.index(dense_dim)
    return dense_to_sparse_dim(self, self.ndim - dense_dim)


def to_dense(self, dtype=None, masked_grad=None):
    _input(self, sparse=True)
    if dtype is not None:
        raise RuntimeError('dtype argument is not supported by sparse_to_dense')
    value = self.coalesce()
    result = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
    if value._values().numel():
        _C.scientific_call(506, [value._indices(), _logical(value._values()), result], [0], [])
    return result


def _mask_inputs(self, mask, *, sparse):
    _input(self, sparse=sparse)
    _input(mask, sparse=True)
    if self.device != mask.device or self.shape != mask.shape:
        raise RuntimeError('sparse mask operands require matching sizes and devices')
    if sparse and self.sparse_dim() != mask.sparse_dim():
        raise RuntimeError('sparse mask operands require the same number of sparse dimensions')


def sparse_mask(self, mask):
    _mask_inputs(self, mask, sparse=self.layout == torch.sparse_coo)
    if self._cdata == mask._cdata:
        return self
    indices = mask._indices().clone()
    shape = mask._values().shape
    if self.layout == torch.strided:
        values = torch.empty(shape, dtype=self.dtype, device=self.device)
        if values.numel():
            _C.scientific_call(506, [indices, values, _logical(self)], [1], [])
    else:
        dtype = torch.promote_types(self.dtype, mask.dtype)
        source_values = _logical(self._values()).to(dtype)
        values = torch.empty(shape, dtype=dtype, device=self.device)
        if values.numel():
            order, _, _ = _sorted(self._indices())
            _C.scientific_call(507, [self._indices(), _parts(source_values), order, indices,
                                   _parts(values), _parts(values)], [0, 1], [])
        values = values.to(self.dtype)
    return _make(indices, values, self.shape, mask.is_coalesced())


def sparse_mask_projection(self, mask, accumulate_matches=False):
    _mask_inputs(self, mask, sparse=True)
    dtype = torch.promote_types(self.dtype, mask.dtype)
    target_values = _logical(self._values()).to(dtype)
    values = torch.empty_like(target_values)
    if values.numel():
        order, _, _ = _sorted(mask._indices())
        _C.scientific_call(507, [mask._indices(), _parts(target_values), order, self._indices(),
                               _parts(target_values), _parts(values)], [1, int(accumulate_matches)], [])
    return _make(self._indices().clone(), values.to(self.dtype), self.shape, self.is_coalesced())


def _sparse_output(result, out):
    if out.layout != torch.sparse_coo or out.device != result.device or not torch.can_cast(result.dtype, out.dtype):
        raise RuntimeError('sparse output requires a compatible dtype, device and COO layout')
    if out.shape != result.shape or out.sparse_dim() != result.sparse_dim():
        out.sparse_resize_and_clear_(result.shape, result.sparse_dim(), result.dense_dim())
    out.copy_(result)
    return out


def _out(function):
    def implementation(*args, out, **kwargs):
        result = function(*args, **kwargs)
        if result.layout != torch.sparse_coo:
            from ._sparse_compressed import _compressed_output
            return _compressed_output(result, out)
        return _sparse_output(result, out)
    return implementation


def register(dense_library):
    library = torch.library.Library('aten', 'IMPL', 'SparsePrivateUse1')
    dense_library.impl('_to_sparse', dense_to_sparse)
    dense_library.impl('_to_sparse.sparse_dim', dense_to_sparse_dim)
    dense_library.impl('_to_sparse.out', _out(dense_to_sparse))
    dense_library.impl('_to_sparse.sparse_dim_out', _out(dense_to_sparse_dim))
    for name, function in (('_coalesce', coalesce), ('sparse_mask', sparse_mask),
                           ('_sparse_mask_projection', sparse_mask_projection)):
        library.impl(name, function)
        library.impl(name + '.out', _out(function))
    library.impl('_to_dense', to_dense)
    library.impl('_to_dense.out', lambda self, dtype=None, masked_grad=None, *, out:
                 _output(to_dense(self, dtype, masked_grad), out, (), cast=True))
    from ._sparse_algebra import register as register_algebra
    register_algebra(library)
    from ._sparse_reduce import register as register_reduce
    register_reduce(library)
    from ._sparse_softmax import register as register_softmax
    register_softmax(library)
    from ._sparse_indexing import register as register_indexing
    register_indexing(library)
    from ._sparse_compressed import register as register_compressed
    register_compressed(dense_library, library)
    _libraries.append(library)
