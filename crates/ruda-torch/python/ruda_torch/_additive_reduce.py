import math

import torch

from . import _C
from ._indexing import _logical
from ._ops import _registry, _dimension, _integral_dtypes, _storage_dtypes, fill_
from ._ordering import _output


def _reduce(a, dim, keepdim, dtype, out, *, mean, infer_out):
    if a.layout != torch.strided or a.dtype not in _storage_dtypes:
        raise RuntimeError('sum/mean requires a supported strided input')
    requested = a.dtype if dtype is None else dtype
    if mean and not (requested.is_floating_point or requested.is_complex):
        raise RuntimeError('mean requires floating or complex input or an explicit floating/complex dtype')
    if dim is None or isinstance(dim, (tuple, list)) and not dim:
        dimensions = tuple(range(a.ndim))
    else:
        dimensions = tuple(_dimension(axis, a.ndim) for axis in (dim if isinstance(dim, (tuple, list)) else (dim,)))
        if len(set(dimensions)) != len(dimensions):
            raise RuntimeError('duplicate sum/mean reduction dimension')
    if dtype is None:
        if infer_out and out is not None:
            dtype = out.dtype
        else:
            dtype = torch.int64 if not mean and a.dtype in _integral_dtypes else a.dtype
    if dtype not in _storage_dtypes:
        raise RuntimeError('sum/mean output dtype is not supported')
    if out is not None and (out.layout != torch.strided or out.device != a.device or out.dtype != dtype):
        raise RuntimeError('sum/mean output dtype/device/layout mismatch')
    shape = tuple(1 if axis in dimensions else size for axis, size in enumerate(a.shape)) if keepdim else tuple(
        size for axis, size in enumerate(a.shape) if axis not in dimensions)
    result = torch.empty(shape, dtype=dtype, device=a.device)
    value = _logical(a)
    if not (dtype == torch.float32 and value.dtype in (torch.float16, torch.bfloat16)):
        value = value.to(dtype)
    if not value.numel():
        fill_(result, math.nan if mean else 0)
    else:
        flags = [int(axis in dimensions) for axis in range(max(a.ndim, 1))]
        _C.scientific_call(480, [value, result], [int(mean), *flags], [])
    return _output(result, out, (a,))


def sum_reduce(a, dim=None, keepdim=False, *, dtype=None, out=None):
    return _reduce(a, dim, keepdim, dtype, out, mean=False, infer_out=False)


def sum_dims_out(a, dim=None, keepdim=False, *, dtype=None, out):
    return _reduce(a, dim, keepdim, dtype, out, mean=False, infer_out=True)


def mean_reduce(a, dim=None, keepdim=False, *, dtype=None, out=None):
    return _reduce(a, dim, keepdim, dtype, out, mean=True, infer_out=True)


def mean_all_out(a, *, dtype=None, out):
    if not torch.can_cast(a.dtype, out.dtype):
        raise RuntimeError('mean.dtype_out input cannot be cast to the output dtype')
    return mean_reduce(a, dtype=dtype, out=out)


_registry.impl('sum.out', sum_reduce)
_registry.impl('sum.IntList_out', sum_dims_out)
_registry.impl('mean.out', mean_reduce)
_registry.impl('mean.dtype_out', mean_all_out)
