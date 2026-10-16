import torch

from . import _C
from ._indexing import _logical
from ._ops import _registry, _dimension, _integral_dtypes, _storage_dtypes
from ._ordering import _output


def product_reduce(a, dim=None, keepdim=False, *, dtype=None, out=None):
    if a.layout != torch.strided or a.dtype not in _storage_dtypes:
        raise RuntimeError('prod requires a supported strided input')
    if dim is None or isinstance(dim, (tuple, list)) and not dim:
        dimensions = tuple(range(a.ndim))
    else:
        dimensions = tuple(_dimension(axis, a.ndim) for axis in (dim if isinstance(dim, (tuple, list)) else (dim,)))
        if len(set(dimensions)) != len(dimensions):
            raise RuntimeError('duplicate product reduction dimension')
    if dtype is None:
        if out is not None and dim is not None:
            dtype = out.dtype
        else:
            dtype = torch.int64 if a.dtype in _integral_dtypes else a.dtype
    if dtype not in _storage_dtypes:
        raise RuntimeError('prod output dtype is not supported')
    if out is not None and (out.layout != torch.strided or out.device != a.device or out.dtype != dtype):
        raise RuntimeError('prod output dtype/device/layout mismatch')
    shape = tuple(1 if axis in dimensions else size for axis, size in enumerate(a.shape)) if keepdim else tuple(
        size for axis, size in enumerate(a.shape) if axis not in dimensions)
    result = torch.empty(shape, dtype=dtype, device=a.device)
    value = _logical(a).to(dtype)
    flags = [int(axis in dimensions) for axis in range(max(a.ndim, 1))]
    _C.scientific_call(479, [value, result], flags, [])
    return _output(result, out, (a,))


_registry.impl('prod.out', product_reduce)
_registry.impl('prod.int_out', product_reduce)
