from functools import partial

import torch

from . import _C
from ._indexing import _logical, _target, _finish_target, _index_out
from ._ops import _dtypes, _numeric_dtypes, _dimension
from ._ordering import _output
from ._pooling import _nondeterministic


def _size(size, operation):
    if size < 0 or size > 2**32 - 1:
        raise RuntimeError(operation + ' output size exceeds native U32 indexing')
    return size


def embedding_renorm(a, indices, max_norm, norm_type, *, inplace=False, out=None):
    if a.ndim != 2 or a.layout != torch.strided:
        raise RuntimeError('embedding_renorm requires a strided two-dimensional weight')
    if (indices.dtype not in (torch.int32, torch.int64) or indices.layout != torch.strided
            or indices.device != a.device):
        raise RuntimeError('embedding_renorm indices must be int32/int64 on the weight device')
    result = _target(a, inplace)
    if indices.numel():
        if a.dtype not in _dtypes:
            raise RuntimeError('embedding_renorm requires a supported floating-point weight')
        if inplace:
            _C.check_inplace(a, a)
            _C.check_index_output(a, indices, indices)
        source = _logical(indices).reshape(-1)
        wrapped = torch.empty_like(source)
        _C.scientific_call(345, [source, wrapped], [a.shape[0]], [])
        ordered = torch.sort(wrapped).values
        _C.scientific_call(346, [result, ordered], [], [max_norm, norm_type])
    result = _finish_target(a, result, inplace)
    return _index_out(result, out, (a, indices), allow_self=True)


def renorm(a, p, dim, maxnorm, *, inplace=False, out=None):
    if isinstance(p, complex) or isinstance(maxnorm, complex):
        raise RuntimeError('renorm p and maxnorm must be real-valued')
    p, maximum = float(p), float(maxnorm)
    if not p > 0 or not maximum >= 0:
        raise RuntimeError('renorm requires positive p and nonnegative maxnorm')
    if a.ndim < 2 or a.layout != torch.strided or a.dtype not in _numeric_dtypes:
        raise RuntimeError('renorm requires a floating-point or complex strided input with rank >= 2')
    dim = _dimension(dim, a.ndim)
    if inplace:
        _C.check_inplace(a, a)
    value = _logical(a)
    accumulator = torch.float32 if a.dtype in (torch.float16, torch.bfloat16) else a.dtype
    axes = tuple(d for d in range(a.ndim) if d != dim)
    norms = torch.linalg.vector_norm(value, p, axes, keepdim=True, dtype=accumulator)
    factors = torch.empty_like(norms)
    if norms.numel():
        _C.scientific_call(347, [norms, factors], [], [maximum])
    result = value * factors.to(a.dtype)
    if inplace:
        a.copy_(result)
        return a
    return _output(result, out, (a,))


def repeat_interleave(repeats, *, output_size=None, out=None):
    if (repeats.ndim != 1 or repeats.dtype not in (torch.int32, torch.int64)
            or repeats.layout != torch.strided):
        raise RuntimeError('repeat_interleave requires a one-dimensional int32/int64 repeats tensor')
    if not repeats.numel():
        result = repeats.new_empty((0,))
    else:
        value = _logical(repeats)
        prefix = value.cumsum(0, dtype=torch.int64)
        if output_size is None:
            total = int(prefix[-1].item())
            if not bool((value >= 0).all().item()):
                raise RuntimeError('repeat_interleave repeats cannot be negative')
        else:
            total = output_size
        result = repeats.new_empty((_size(total, 'repeat_interleave'),))
        _C.scientific_call(348, [value, prefix, result], [], [])
    return _output(result, out, (repeats,))


def bincount(a, weights=None, minlength=0, *, out=None):
    if weights is not None:
        _nondeterministic('_bincount_cuda')
    if (a.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
            or a.layout != torch.strided or a.ndim != 1):
        raise RuntimeError('bincount requires a one-dimensional nonnegative integral input')
    if minlength < 0:
        raise RuntimeError('bincount minlength must be nonnegative')
    if not a.numel():
        result = torch.zeros((_size(minlength, 'bincount'),), dtype=torch.int64, device=a.device)
    else:
        if weights is not None and (weights.ndim != 1 or weights.shape != a.shape
                                    or weights.device != a.device or weights.layout != torch.strided):
            raise RuntimeError('bincount weights must match input shape and device')
        value = _logical(a)
        minimum, maximum = torch.aminmax(value)
        if a.dtype != torch.uint8 and int(minimum.item()) < 0:
            raise RuntimeError('bincount input cannot contain negative values')
        bins = _size(max(int(maximum.item()) + 1, minlength), 'bincount')
        if weights is None:
            dtype = torch.int64
            samples = torch.empty((0,), dtype=dtype, device=a.device)
        else:
            dtype = torch.float32 if weights.dtype == torch.float32 else torch.float64
            samples = _logical(weights).to(dtype)
        result = torch.zeros((bins,), dtype=dtype, device=a.device)
        _C.scientific_call(349, [value, samples, result], [int(weights is not None)], [])
    return _output(result, out, (a, weights))


def register(library):
    for name, function in (('embedding_renorm', embedding_renorm), ('renorm', renorm)):
        library.impl(name, function)
        library.impl(name + '.out', function)
        library.impl(name + '_', partial(function, inplace=True))
    library.impl('repeat_interleave.Tensor', repeat_interleave)
    library.impl('repeat_interleave.Tensor_out', repeat_interleave)
    library.impl('bincount', bincount)
    library.impl('bincount.out', bincount)
