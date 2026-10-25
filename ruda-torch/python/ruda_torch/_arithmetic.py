import torch

from . import _C
from ._indexing import _logical
from ._ops import _numeric_dtypes, _integral_dtypes, _storage_dtypes
from ._ordering import _output


def arithmetic(a, b=None, *, op, alpha=1, inplace=False, out=None):
    unary = op == 3
    if unary:
        b = a
    original = (a, b)
    tensors = tuple(value for value in original if isinstance(value, torch.Tensor))
    device = next((value.device for value in tensors if value.device.type != 'cpu'), None)
    if device is None:
        raise RuntimeError('RUDA arithmetic requires a non-CPU tensor operand')
    supported = _numeric_dtypes + _integral_dtypes + (torch.complex32,)
    for value in tensors:
        if value.layout != torch.strided or value.dtype not in supported:
            raise RuntimeError('arithmetic requires supported strided inputs')
        if value.device != device and not (value.device.type == 'cpu' and value.ndim == 0):
            raise RuntimeError('arithmetic operands must share a device except a CPU scalar tensor')
    if op == 1 and any(isinstance(value, bool) or isinstance(value, torch.Tensor) and value.dtype == torch.bool for value in original):
        raise RuntimeError('subtraction with a bool operand is not supported')
    dtype = a.dtype if unary else torch.result_type(a, b)
    if dtype not in supported or unary and dtype == torch.bool:
        raise RuntimeError('arithmetic does not support the result dtype')
    shape = torch.broadcast_shapes(*(value.shape for value in tensors))
    if inplace:
        if a.shape != shape:
            raise RuntimeError('in-place arithmetic cannot change the input shape')
        out = a
    if out is not None:
        if out.device != device or out.layout != torch.strided or out.dtype not in _storage_dtypes or not torch.can_cast(dtype, out.dtype):
            raise RuntimeError('arithmetic output requires a compatible dtype on the operation device')
        for value in tensors:
            _C.check_inplace(out, value)
    if dtype.is_complex:
        from ._complex32_ops import binary, negate
        if unary:
            return _output(negate(a), out, tensors, cast=True)
        return binary(a, b, op=op, alpha=alpha, out=out)

    coefficient = _C.arithmetic_alpha(alpha, dtype, out.dtype if out is not None else dtype, op == 1) if op <= 1 else 0
    math_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    scalar = tuple(not isinstance(value, torch.Tensor) or value.device.type == 'cpu' for value in original)

    def operand(value, cpu_scalar):
        target = math_dtype if cpu_scalar else dtype
        if not isinstance(value, torch.Tensor):
            return _C.predicate_scalar(value, target, device)
        value = _logical(value).to(target)
        return value.to(device) if cpu_scalar else value.expand(shape)

    left = operand(a, scalar[0])
    right = left if unary else operand(b, scalar[1])
    direct = (out is not None and out.dtype == dtype and out.shape == shape
              and not out.is_neg() and not out.is_conj()
              and all(value.is_contiguous() and value.shape == shape for value in (left, right, out)))
    result = out if direct else torch.empty(shape, dtype=dtype, device=device)
    _C.scientific_call(485, [left, right, result], [2 if unary else 1 if op == 2 else 0, int(scalar[0]), int(scalar[1]), coefficient], [])
    return result if direct else _output(result, out, tensors, cast=True)
