import math

import torch

from . import _C
from ._random_extra import _reservation


def _inputs(*tensors, wide_only=False):
    first = tensors[0]
    supported = (torch.float32, torch.float64) if wide_only else (
        torch.float16, torch.bfloat16, torch.float32, torch.float64)
    if first.device.type != 'ruda' or first.dtype not in supported:
        raise RuntimeError('distribution requires a supported floating RUDA dtype')
    for tensor in tensors:
        if tensor.layout != torch.strided or tensor.device != first.device or tensor.dtype != first.dtype:
            raise RuntimeError('distribution inputs must have the same dtype and device and strided layout')
    shape = torch.broadcast_shapes(*(tensor.shape for tensor in tensors))
    if math.prod(shape) > 2**32 - 1 or any(size > 2**32 - 1 for size in shape):
        raise RuntimeError('distribution exceeds native U32 indexing')
    return tuple(tensor.resolve_neg().expand(shape) for tensor in tensors), shape


def _sample(op, tensors, generator):
    tensors, shape = _inputs(*tensors)
    result = torch.empty(shape, dtype=tensors[0].dtype, device=tensors[0].device)
    if result.numel():
        with _reservation(generator, result.numel()) as state:
            _C.scientific_call(op, [*tensors, result], state, [])
    return result


def binomial(count, prob, generator=None):
    with torch.no_grad():
        return _sample(425, (count, prob), generator)


def standard_gamma(a, generator=None):
    with torch.no_grad():
        return _sample(426, (a,), generator)


def sample_dirichlet(a, generator=None):
    with torch.no_grad():
        result = _sample(426, (a,), generator)
        total = result.sum(dim=-1, keepdim=True).expand_as(result)
        if result.numel():
            _C.scientific_call(427, [result, total, result], [], [])
        return result


def _gradient(op, tensors):
    tensors, shape = _inputs(*tensors, wide_only=op == 429)
    result = torch.empty(shape, dtype=tensors[0].dtype, device=tensors[0].device)
    if result.numel():
        _C.scientific_call(op, [*tensors, result], [], [])
    return result


def standard_gamma_grad(a, output):
    return _gradient(428, (a, output))


def dirichlet_grad(x, alpha, total):
    return _gradient(429, (x, alpha, total))


def register(library):
    library.impl('binomial', binomial)
    library.impl('_standard_gamma', standard_gamma)
    library.impl('_sample_dirichlet', sample_dirichlet)
    library.impl('_standard_gamma_grad', standard_gamma_grad)
    library.impl('_dirichlet_grad', dirichlet_grad)
