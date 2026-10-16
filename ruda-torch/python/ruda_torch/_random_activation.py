import torch
from torch._prims_common import suggest_memory_format

from . import _C
from ._random_extra import _reservation, _storage


_floating = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _floating_input(a):
    _storage(a)
    if a.dtype not in _floating:
        raise RuntimeError('random activation requires real floating input')


def _dropout(a, keep, mask_dtype, generator):
    _storage(a)
    mask = torch.empty_like(a, dtype=mask_dtype)
    if not a.numel():
        return a.clone(), mask
    _floating_input(a)
    output = torch.empty_like(a)
    with _reservation(generator, (a.numel() + 3) // 4) as state:
        _C.scientific_call(440, [a.resolve_neg(), output, mask], state, [keep])
    return output, mask


def native_dropout(a, p, train):
    _storage(a)
    if train is not None and not train:
        return a.clone(), torch.ones_like(a, dtype=torch.bool)
    if p == 1:
        return torch.zeros_like(a), torch.zeros_like(a, dtype=torch.bool)
    return _dropout(a, 1. - p, torch.bool, None)


def fused_dropout(a, p, generator=None):
    return _dropout(a, p, torch.uint8, generator)


def _masked_scale(a, mask, scale, mask_dtype):
    _floating_input(a)
    _storage(mask)
    if mask.device != a.device or mask.dtype != mask_dtype:
        raise RuntimeError('dropout mask dtype or device does not match')
    x, m = torch.broadcast_tensors(a.resolve_neg(), mask.resolve_neg())
    output = torch.empty_like(a, memory_format=suggest_memory_format(a))
    if output.shape != x.shape:
        _C.prepare_index_output(output, list(x.shape))
    _C.scientific_call(441, [x, m, output], [], [float(scale)])
    return output


def native_dropout_backward(grad_output, mask, scale):
    return _masked_scale(grad_output, mask, scale, torch.bool)


def masked_scale(a, mask, scale):
    return _masked_scale(a, mask, scale, torch.uint8)


def rrelu_with_noise(a, noise, lower=.125, upper=1. / 3., training=False,
                     generator=None, *, out=None):
    _storage(a)
    if out is None:
        out = torch.empty_like(a, memory_format=torch.contiguous_format)
    _storage(out)
    if out.shape != a.shape:
        _C.prepare_index_output(out, list(a.shape))
    if not a.numel():
        return out
    _storage(noise)
    if out.device != a.device or noise.device != a.device:
        raise RuntimeError('RReLU tensors must share a device')
    if not training:
        return torch.ops.aten.leaky_relu.out(a, (float(lower) + float(upper)) / 2., out=out)
    _floating_input(a)
    if noise.dtype != a.dtype or out.dtype != a.dtype:
        raise RuntimeError('training RReLU tensors must have matching dtypes')
    if noise.numel() < a.numel():
        raise RuntimeError('RReLU noise storage is smaller than the input')
    lower, upper = float(lower), float(upper)
    _C.check_inplace(out, a)
    _C.check_inplace(out, noise)
    with torch.no_grad():
        x = a.resolve_neg().contiguous()
        contiguous_noise = noise.resolve_neg().contiguous()
        n = contiguous_noise.view(-1).narrow(0, 0, a.numel()).view(a.shape)
        target = (out if out.is_contiguous() and not out.is_neg()
                  else torch.empty_like(a, memory_format=torch.contiguous_format))
        words = a.numel() * (2 if a.dtype == torch.float64 else 1)
        with _reservation(generator, (words + 3) // 4) as state:
            _C.scientific_call(442, [x, target, n], state, [lower, upper])
        if noise.is_neg() and noise.is_contiguous():
            noise.copy_(contiguous_noise)
        if target is not out:
            out.copy_(target)
    return out


def rrelu_with_noise_(a, noise, lower=.125, upper=1. / 3., training=False, generator=None):
    return rrelu_with_noise(a, noise, lower, upper, training, generator, out=a)


def register(library):
    library.impl('native_dropout', native_dropout)
    library.impl('native_dropout_backward', native_dropout_backward)
    library.impl('_fused_dropout', fused_dropout)
    library.impl('_masked_scale', masked_scale)
    library.impl('rrelu_with_noise', rrelu_with_noise)
    library.impl('rrelu_with_noise.out', rrelu_with_noise)
    library.impl('rrelu_with_noise_', rrelu_with_noise_)
