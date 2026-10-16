import math

import torch

from . import _C
from ._indexing import _logical
from ._ordering import _output


def binary(a, b, *, op, alpha=1, inplace=False, out=None):
    original = (a, b)
    tensors = tuple(value for value in original if isinstance(value, torch.Tensor))
    if op == 1 and any(isinstance(value, bool) or isinstance(value, torch.Tensor) and value.dtype == torch.bool for value in original):
        raise RuntimeError('subtraction with a bool operand is not supported')
    device = next((value.device for value in tensors if value.device.type != 'cpu'), None)
    if device is None:
        raise RuntimeError('complex32 arithmetic requires a GPU tensor')
    for value in tensors:
        if value.layout != torch.strided:
            raise RuntimeError('complex32 arithmetic requires strided inputs')
        if value.device != device and not (value.device.type == 'cpu' and value.ndim == 0):
            raise RuntimeError('operands must share a device; only CPU scalar tensors are permitted')
    dtype = torch.result_type(a, b)
    if dtype not in (torch.complex32, torch.complex64, torch.complex128):
        raise RuntimeError('complex32 arithmetic requires a complex common dtype')
    if op <= 1 and isinstance(alpha, bool):
        raise RuntimeError('boolean alpha is only supported for boolean results')
    coefficient = complex(alpha if op == 0 else -alpha if op == 1 else 1)
    if dtype != torch.complex128:
        limit = torch.finfo(torch.float32).max
        if any(math.isfinite(x) and abs(x) > limit for x in (coefficient.real, coefficient.imag)):
            raise RuntimeError('alpha cannot be converted to complex float32 without overflow')
    scalar = tuple(not isinstance(value, torch.Tensor) or value.device.type == 'cpu' for value in original)
    shape = torch.broadcast_shapes(*(value.shape for value in tensors))
    if inplace:
        if not isinstance(a, torch.Tensor):
            raise RuntimeError('in-place arithmetic requires a tensor destination')
        out = a
        if out.shape != shape or not torch.can_cast(torch.bool if op >= 5 else dtype, out.dtype):
            raise RuntimeError('in-place operation cannot change shape or cast the result dtype')
    if out is not None:
        if out.device != device or out.layout != torch.strided or not torch.can_cast(torch.bool if op >= 5 else dtype, out.dtype):
            raise RuntimeError('invalid complex arithmetic output dtype/device/layout')
        for value in tensors:
            _C.check_inplace(out, value)
    operands = [None if flag else _logical(value).to(dtype).expand(shape) for value, flag in zip(original, scalar)]
    reference = next(value for value in operands if value is not None)
    parts = []
    parameters = [coefficient.real, coefficient.imag]
    for value, operand, flag in zip(original, operands, scalar):
        if flag:
            number = complex(value.item() if isinstance(value, torch.Tensor) else value)
            parameters.extend((number.real, number.imag))
        else:
            parameters.extend((0.0, 0.0))
        pair = torch.view_as_real(reference if flag else operand)
        parts.extend((pair[..., 0], pair[..., 1]))
    result = torch.empty(shape, dtype=dtype, device=device)
    target = torch.view_as_real(result)
    if result.numel() > 2**32 - 1:
        raise RuntimeError('complex arithmetic exceeds native U32 indexing')
    if result.numel():
        _C.scientific_call(396, parts + [target[..., 0], target[..., 1]], [op, int(scalar[0]), int(scalar[1])], parameters)
    if op >= 5:
        result = target[..., 0].to(torch.bool)
    return _output(result, out, tensors, cast=True)


def negate(a, *, inplace=False):
    from ._scientific import execute
    if inplace:
        _C.check_inplace(a, a)
    result = execute(208, a, a, torch.empty_like(a))
    return _output(result, a if inplace else None, (a,))
