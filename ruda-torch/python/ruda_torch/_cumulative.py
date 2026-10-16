import math
from functools import partial

import torch

from . import _C
from ._indexing import _logical
from ._ops import _dimension, _integral_dtypes, _numeric_dtypes, _storage_dtypes
from ._ordering import _output


def cumulative(a, dim, *, dtype=None, product=False, inplace=False, out=None):
    axis = _dimension(dim, a.ndim)
    if a.layout != torch.strided:
        raise RuntimeError('cumulative operations require strided input')
    if inplace:
        out = a
    if dtype is None:
        dtype = out.dtype if out is not None else torch.int64 if a.dtype in _integral_dtypes else a.dtype
    if dtype not in _storage_dtypes:
        raise RuntimeError('unsupported cumulative output storage dtype')
    if out is not None:
        if out.layout != torch.strided or out.device != a.device or out.dtype != dtype:
            raise RuntimeError('cumulative output dtype/device/layout mismatch')
        _C.check_inplace(out, a)
    if a.ndim == 0:
        return _output(_logical(a).to(dtype).clone(), out, (a,))
    if a.numel() and (dtype not in _numeric_dtypes + _integral_dtypes or dtype == torch.bool):
        raise RuntimeError('nonempty cumulative operations require floating, standard integer or complex64/128 output')
    if a.numel() > 2**32 - 1:
        raise RuntimeError('cumulative operation exceeds native U32 indexing')
    result = torch.empty(a.shape, dtype=dtype, device=a.device)
    if not a.numel():
        return _output(result, out, (a,))
    outer = math.prod(a.shape[:axis])
    width = a.shape[axis]
    inner = math.prod(a.shape[axis + 1:])
    value = _logical(a).to(dtype).reshape(outer, width, inner)
    target = result.reshape(outer, width, inner)
    complex_input = value.is_complex()
    if complex_input:
        source_parts = torch.view_as_real(value)
        target_parts = torch.view_as_real(target)
        tensors = [source_parts[..., 0], source_parts[..., 1], target_parts[..., 0], target_parts[..., 1]]
    else:
        tensors = [value, value, target, target]
    parallel = axis == a.ndim - 1 or a.numel() == width
    _C.scientific_call(406 if product else 405, tensors, [int(complex_input), int(parallel)], [])
    return _output(result, out, (a,))


def trace(a):
    if a.ndim != 2:
        raise RuntimeError('expected a matrix')
    if a.layout != torch.strided or a.dtype not in _storage_dtypes:
        raise RuntimeError('trace requires a supported strided tensor')
    dtype = torch.int64 if a.dtype in _integral_dtypes else a.dtype
    if min(a.shape) == 0:
        return torch.zeros((), dtype=dtype, device=a.device)
    if a.dtype not in _numeric_dtypes + _integral_dtypes + (torch.complex32,):
        raise RuntimeError('trace requires real floating, standard integer or complex input')
    diagonal = _logical(a).diagonal()
    if a.dtype == torch.complex32:
        from ._scientific import reduce
        return reduce(diagonal)
    return diagonal.sum()


def register(library):
    for name, product in (('cumsum', False), ('cumprod', True)):
        library.impl(name + '.out', partial(cumulative, product=product))
        library.impl(name + '_', partial(cumulative, product=product, inplace=True))
    library.impl('trace', trace)
