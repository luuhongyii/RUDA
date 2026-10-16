import math

import torch

from . import _C
from ._ops import _typed_value, fill_
from ._ordering import _output, _real


def _input(a):
    if a.layout != torch.strided:
        raise RuntimeError('elementwise extrema require strided inputs')
    return _real(a)


def _launch(a, lower, upper, mode, *, scalar=False):
    if not scalar:
        a, lower, upper = torch.broadcast_tensors(a, lower, upper)
    result = torch.empty(a.shape, dtype=a.dtype, device=a.device)
    _C.scientific_call(481, [a, lower, upper, result], [mode, int(scalar)], [])
    return result


def extreme(a, b, *, maximum, ignore_nan=False, out=None):
    original = a, b
    dtype = torch.result_type(a, b)
    a, b = _input(a), _input(b)
    if a.device != b.device:
        if b.device.type == 'cpu' and b.ndim == 0:
            a, b = b, a
        if a.device.type != 'cpu' or a.ndim != 0:
            raise RuntimeError('ordering operands must be on the same device except a CPU scalar')
        a = a.to(device=b.device, dtype=dtype)
    a, b = a.to(dtype), b.to(dtype)
    result = _launch(a, b, b, (0 if maximum else 1) + (2 if ignore_nan else 0))
    return _output(result, out, original, cast=True)


def _finish(result, a, bounds, inplace, out):
    if inplace:
        if result.shape != a.shape or not torch.can_cast(result.dtype, a.dtype):
            raise RuntimeError('in-place clamp cannot change shape or cast the result dtype')
        out = a
    return _output(result, out, (a, *bounds), cast=True)


def clamp_bound(a, bound, *, maximum, inplace=False, out=None):
    if isinstance(bound, torch.Tensor):
        result = extreme(a, bound, maximum=maximum)
        return _finish(result, a, (bound,), inplace, out)
    return clamp(a, min=bound if maximum else None, max=None if maximum else bound, inplace=inplace, out=out, _single=True)


def clamp(a, min=None, max=None, *, inplace=False, out=None, _single=False):
    bounds = tuple(value for value in (min, max) if value is not None)
    if not bounds:
        raise RuntimeError('clamp requires at least one bound')
    value = _input(a)
    tensor = any(isinstance(bound, torch.Tensor) for bound in bounds)
    if tensor:
        if not all(isinstance(bound, torch.Tensor) for bound in bounds):
            raise TypeError('clamp bounds must both be tensors or both be scalars')
        if any(bound.device != a.device for bound in bounds):
            raise RuntimeError('clamp.Tensor bounds must be on the input device')
        converted = [_input(bound) for bound in bounds]
        if len(bounds) == 1:
            result = extreme(value, converted[0], maximum=min is not None)
        else:
            dtype = _C.extrema_result_type([a, *bounds])
            if dtype == torch.bool:
                raise RuntimeError('clamp with two Tensor bounds does not support Bool output')
            value, lower, upper = (operand.to(dtype) for operand in (value, *converted))
            result = _launch(value, lower, upper, 4)
    else:
        if any(isinstance(bound, complex) for bound in bounds):
            raise RuntimeError('clamp does not support complex bounds')
        dtype = a.dtype
        if not a.is_floating_point():
            for bound in bounds:
                dtype = torch.promote_types(dtype, torch.result_type(a, bound))
        if (inplace or out is a) and dtype != a.dtype:
            raise RuntimeError('in-place scalar clamp cannot promote the input dtype')
        value = value.to(dtype)
        if any(math.isnan(bound) for bound in bounds):
            result = fill_(torch.empty(a.shape, dtype=dtype, device=a.device), bounds[0] if _single else math.nan)
        else:
            if dtype == torch.bool:
                raise RuntimeError('scalar clamp does not support Bool output')
            opmath = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
            lower = _typed_value(bounds[0], opmath, a.device)
            upper = _typed_value(bounds[1], opmath, a.device) if len(bounds) == 2 else lower
            result = _launch(value, lower, upper, 4 if len(bounds) == 2 else 5 if min is not None else 6, scalar=True)
    return _finish(result, a, bounds, inplace, out)
