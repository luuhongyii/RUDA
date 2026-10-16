import math
from functools import reduce

import torch

from . import _C
from ._indexing import _logical
from ._ops import _integral_dtypes, _numeric_dtypes, _typed_value
from ._ordering import _output


def _common_dtype(tensors):
    dimensional = [value.dtype for value in tensors if value.ndim != 0]
    scalar = [value.dtype for value in tensors if value.ndim == 0]
    if not dimensional or not scalar:
        return reduce(torch.promote_types, dimensional or scalar)
    return torch.result_type(
        torch.empty((0,), dtype=reduce(torch.promote_types, dimensional), device='meta'),
        torch.empty((), dtype=reduce(torch.promote_types, scalar), device='meta'))


def _finish(a, b, c, *, op, dtype, coefficient, scalar_weight=False, out=None, inplace=False, cast=True):
    inputs = (a, b) if scalar_weight else (a, b, c)
    for position, value in enumerate(inputs):
        if value.layout != torch.strided:
            raise RuntimeError('ternary operations require strided tensors')
        cpu_third = position == 2 and (op == 399 or op == 401)
        if value.device != a.device and not (cpu_third and value.device.type == 'cpu' and value.ndim == 0):
            raise RuntimeError('only the third operand of addcmul/lerp may be a CPU scalar tensor')
    if a.device.type == 'cpu':
        raise RuntimeError('RUDA ternary operations require GPU self and tensor1/end inputs')
    shape = torch.broadcast_shapes(*(value.shape for value in inputs))
    if math.prod(shape) > 2**32 - 1:
        raise RuntimeError('ternary operation exceeds native U32 indexing')
    if inplace:
        out = a
        if out.shape != shape:
            raise RuntimeError('in-place ternary operation cannot change the input shape')
    if out is not None:
        if out.device != a.device or out.layout != torch.strided or not (torch.can_cast(dtype, out.dtype) if cast else dtype == out.dtype):
            raise RuntimeError('invalid ternary output dtype/device/layout')
        for value in inputs:
            _C.check_inplace(out, value)
            if out._cdata == value._cdata and out.shape != shape:
                raise RuntimeError('a ternary output that is also an input cannot be resized')
    compute_dtype = (torch.float32 if dtype in (torch.float16, torch.bfloat16)
                     else torch.complex64 if dtype == torch.complex32
                     else torch.int64 if dtype in _integral_dtypes else dtype)
    if compute_dtype == torch.complex64:
        number = complex(coefficient)
        limit = torch.finfo(torch.float32).max
        if any(math.isfinite(part) and abs(part) > limit for part in (number.real, number.imag)):
            raise RuntimeError('ternary coefficient cannot be converted to complex64 without overflow')
    alpha = _typed_value(coefficient, compute_dtype, a.device)
    operands = [_logical(value).to(device=a.device, dtype=dtype).expand(shape) for value in inputs]
    if scalar_weight:
        operands.append(operands[0])
    result = torch.empty_like(operands[0])
    tensors = operands + [alpha, result]
    is_complex = result.is_complex()
    if is_complex:
        parts = [torch.view_as_real(value) for value in tensors]
        tensors = [component for pair in parts for component in (pair[..., 0], pair[..., 1])]
    if result.numel():
        _C.scientific_call(op, tensors, [int(is_complex), int(scalar_weight)], [])
    return _output(result, out, inputs, cast=cast)


def addc(a, tensor1, tensor2, *, value=1, divide=False, inplace=False, out=None):
    inputs = (a, tensor1, tensor2)
    dtype = _common_dtype(inputs)
    if dtype not in _numeric_dtypes + _integral_dtypes or dtype == torch.bool:
        raise RuntimeError('addcmul/addcdiv require supported floating, complex64/128 or integral results')
    if divide and tensor1.dtype in _integral_dtypes and tensor2.dtype in _integral_dtypes:
        raise RuntimeError('integer division with addcdiv is not supported')
    return _finish(*inputs, op=400 if divide else 399, dtype=dtype, coefficient=value, out=out, inplace=inplace)


def lerp(a, end, weight, *, inplace=False, out=None):
    if a.dtype != end.dtype:
        raise RuntimeError('lerp self and end must have the same dtype')
    scalar_weight = not isinstance(weight, torch.Tensor)
    promote_weight = not scalar_weight and weight.ndim == 0
    if not scalar_weight and not promote_weight and weight.dtype != a.dtype:
        raise RuntimeError('non-scalar lerp weight must match the input dtype')
    dtype = _common_dtype((a, end, weight)) if promote_weight else a.dtype
    if dtype not in _numeric_dtypes + (torch.complex32,):
        raise RuntimeError('lerp requires floating or complex results')
    return _finish(a, end, weight, op=401, dtype=dtype,
                   coefficient=weight if scalar_weight else 1, scalar_weight=scalar_weight,
                   out=out, inplace=inplace, cast=scalar_weight or promote_weight)


def register(library):
    library.impl('addcmul.out', addc)
    library.impl('addcdiv.out', lambda a, b, c, *, value=1, out: addc(a, b, c, value=value, divide=True, out=out))
    library.impl('lerp.Scalar_out', lerp)
    library.impl('lerp.Tensor_out', lerp)
