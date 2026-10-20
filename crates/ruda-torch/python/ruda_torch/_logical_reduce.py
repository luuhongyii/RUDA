from functools import partial

import torch

from . import _C
from ._indexing import _logical
from ._ops import _registry, _dimension, _storage_dtypes
from ._ordering import _output


def _reduce(a, dim, keepdim, mode, out):
    if a.layout != torch.strided or a.dtype not in _storage_dtypes:
        raise RuntimeError('logical reduction requires a supported strided input')
    if dim is None:
        dimensions = tuple(range(a.ndim))
    else:
        dimensions = tuple(_dimension(axis, a.ndim) for axis in (dim if isinstance(dim, (tuple, list)) else (dim,)))
        if len(set(dimensions)) != len(dimensions):
            raise RuntimeError('duplicate reduction dimension')
        if mode == 2 and not dimensions:
            dimensions = tuple(range(a.ndim))
    if mode == 2:
        dtype = torch.int64
    else:
        dtype = out.dtype if out is not None else torch.uint8 if a.dtype == torch.uint8 else torch.bool
        if dtype not in (torch.bool, torch.uint8):
            raise RuntimeError('all/any output must have bool or uint8 dtype')
    if out is not None and (out.layout != torch.strided or out.device != a.device or out.dtype != dtype):
        raise RuntimeError('logical reduction output dtype/device/layout mismatch')
    shape = tuple(1 if axis in dimensions else size for axis, size in enumerate(a.shape)) if keepdim else tuple(
        size for axis, size in enumerate(a.shape) if axis not in dimensions)
    result = torch.empty(shape, dtype=dtype, device=a.device)
    flags = [int(axis in dimensions) for axis in range(max(a.ndim, 1))]
    _C.scientific_call(478, [_logical(a), result], [mode, *flags], [])
    return _output(result, out, (a,))


def boolean_reduce(a, dim=None, keepdim=False, *, every, out=None):
    return _reduce(a, dim, keepdim, 0 if every else 1, out)


def count_nonzero(a, dim=None, *, out=None):
    return _reduce(a, dim, False, 2, out)


for name, every in (('all', True), ('any', False)):
    for overload in ('out', 'dims_out', 'all_out'):
        _registry.impl(name + '.' + overload, partial(boolean_reduce, every=every))

for name in ('count_nonzero', 'count_nonzero.dim_IntList', 'count_nonzero.out', 'count_nonzero.dim_IntList_out'):
    _registry.impl(name, count_nonzero)
