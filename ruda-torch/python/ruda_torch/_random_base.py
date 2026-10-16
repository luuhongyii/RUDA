import math

import torch

from . import _C
from ._random_extra import _reservation, _storage


_floating = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_bernoulli_dtypes = _floating + (
    torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _target(a):
    _C.check_inplace(a, a)
    return torch.empty_like(a) if a.is_conj() or a.is_neg() else a


def _fill(a, target, op, parameters, generator, probabilities=None):
    words = target.numel() * (2 if target.dtype == torch.float64 and op != 433 else 1)
    tensors = [target] if probabilities is None else [probabilities, target]
    with _reservation(generator, (words + 3) // 4) as state, torch.no_grad():
        _C.scientific_call(op, tensors, state, parameters)
    return a


def uniform_(a, from_=0., to=1., *, generator=None):
    _storage(a)
    if a.is_complex() and not a.numel():
        return a
    target = _target(a)
    real = torch.view_as_real(target) if target.is_complex() else target
    parameters = _C.uniform_parameters(real, float(from_), float(to))
    if not a.numel():
        return a
    with torch.no_grad():
        _fill(a, real, 430, parameters, generator)
        if target is not a:
            a.copy_(target)
    return a


def _check_std(std):
    if isinstance(std, torch.Tensor):
        if std.is_complex() or (std.numel() and not std.is_meta and not std.min().ge(0).item()):
            raise RuntimeError('normal expects non-complex standard deviation with all elements >= 0')
    elif not float(std) >= 0:
        raise RuntimeError('normal expects std >= 0')


def normal_(a, mean=0., std=1., *, generator=None):
    _storage(a)
    mean, std = float(mean), float(std)
    _check_std(std)
    if not a.numel():
        return a
    target = _target(a)
    real = torch.view_as_real(target) if target.is_complex() else target
    if real.dtype not in _floating:
        raise RuntimeError('normal_ requires floating or complex output')
    if target.is_complex():
        std /= math.sqrt(2.0)
    with torch.no_grad():
        _fill(a, real, 431, [mean, std], generator)
        if target is not a:
            a.copy_(target)
    return a


def bernoulli_(a, p=.5, *, generator=None):
    _storage(a)
    p = float(p)
    if not 0 <= p <= 1:
        raise RuntimeError('bernoulli_ expects p in [0, 1]')
    if not a.numel():
        return a
    if a.dtype not in _bernoulli_dtypes:
        raise RuntimeError('bernoulli_ requires real floating, integer or bool output')
    target = _target(a)
    with torch.no_grad():
        _fill(a, target, 432, [p], generator)
        if target is not a:
            a.copy_(target)
    return a


def bernoulli_tensor(a, p, *, generator=None):
    _storage(a)
    if not a.numel():
        return a
    if a.dtype not in _bernoulli_dtypes or not p.is_floating_point():
        raise RuntimeError('bernoulli_ requires real output and floating probabilities')
    target = _target(a)
    with torch.no_grad():
        probabilities = p.to(device=a.device, dtype=torch.float64 if a.dtype == torch.float64 else torch.float32)
        probabilities = probabilities.resolve_neg().expand(a.shape)
        _C.check_inplace(a, probabilities)
        _fill(a, target, 433, [], generator, probabilities)
        if target is not a:
            a.copy_(target)
    return a


def _resize(out, shape):
    _storage(out)
    if out.shape != shape:
        _C.prepare_index_output(out, list(shape))
    return out


def normal_tensor_float(mean, std=1., *, generator=None, out=None):
    _check_std(std)
    if out is None:
        out = torch.empty_like(mean, memory_format=torch.contiguous_format)
    shape = torch.broadcast_shapes(mean.shape, out.shape)
    with torch.no_grad():
        _resize(out, shape)
        normal_(out, 0., std, generator=generator)
        out.add_(mean)
    return out


def normal_float_tensor(mean, std, *, generator=None, out=None):
    _check_std(std)
    if out is None:
        out = torch.empty_like(std, memory_format=torch.contiguous_format)
    with torch.no_grad():
        mean_tensor = torch.full((), mean, dtype=out.dtype, device=out.device)
        _resize(out, std.shape)
        normal_(out, 0., 1., generator=generator)
        out.mul_(std).add_(mean_tensor)
    return out


def normal_tensor_tensor(mean, std, *, generator=None, out=None):
    _check_std(std)
    shape = torch.broadcast_shapes(mean.shape, std.shape)
    if out is None:
        out = torch.empty(shape, dtype=mean.dtype, device=mean.device, layout=mean.layout)
    with torch.no_grad():
        _resize(out, shape)
        normal_(out, 0., 1., generator=generator)
        out.mul_(std).add_(mean)
    return out


def bernoulli_out(p, *, generator=None, out):
    _resize(out, p.shape)
    return bernoulli_tensor(out, p, generator=generator)


def register(library):
    for name, function in (
        ('normal.Tensor_float', normal_tensor_float),
        ('normal.float_Tensor', normal_float_tensor),
        ('normal.Tensor_Tensor', normal_tensor_tensor),
    ):
        library.impl(name, function)
        library.impl(name + '_out', function)
    library.impl('bernoulli.out', bernoulli_out)
