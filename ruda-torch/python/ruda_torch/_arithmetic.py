from functools import lru_cache
import torch

from . import _C
from ._indexing import _logical
from ._ops import _numeric_dtypes, _integral_dtypes, _storage_dtypes
from ._ordering import _output
from ._broadcast import broadcast_shapes


@lru_cache(maxsize=1)
def _direct_available():
    from . import _native
    version = getattr(_native, 'ruda_torch_arithmetic_api_version', None)
    return version is not None and version() >= 2


@lru_cache(maxsize=1)
def _integer_scalar_available():
    from . import _native
    version = getattr(_native, 'ruda_torch_arithmetic_api_version', None)
    return version is not None and version() >= 4


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
    shape = broadcast_shapes(*(value.shape for value in tensors))
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

    if (not unary and dtype in _integral_dtypes and any(scalar) and _integer_scalar_available()
            and all(value.dtype == dtype for value in tensors if value.device == device)
            and all(isinstance(value, torch.Tensor) or type(value) in (bool, int) for value in original)):
        value = original[0] if scalar[0] else original[1]
        if isinstance(value, torch.Tensor):
            value = value.item()
        try:
            immediate = _C.arithmetic_alpha(int(value), dtype, dtype, False)
        except (OverflowError, RuntimeError):
            pass
        else:
            operand = _logical(next(value for value in tensors if value.device == device)).expand(shape)
            destination = (out if out is not None and out.shape == shape and out.dtype == dtype
                           and not out.is_conj() and not out.is_neg() else None)
            result = destination if destination is not None else torch.empty(shape, dtype=dtype, device=device)
            _C.scientific_call(485, [operand, operand, result],
                               [1 if op == 2 else 0, 2 * int(scalar[0]), 2 * int(scalar[1]), coefficient, immediate], [])
            return result if destination is not None else _output(result, out, tensors, cast=True)

    direct = (dtype in (torch.float32, torch.float16, torch.bfloat16) and _direct_available()
              and all(value.device == device and value.dtype in (torch.float32, torch.float16, torch.bfloat16)
                      and (dtype == torch.float32 or value.dtype == dtype)
                      for value in tensors)
              and all(isinstance(value, torch.Tensor) or type(value) in (bool, int, float) for value in original))
    immediate = None
    if direct and not unary and any(scalar):
        try:
            value = original[0] if scalar[0] else original[1]
            immediate = _C.arithmetic_alpha(int(value) if type(value) is bool else value, math_dtype, math_dtype, False)
        except (OverflowError, RuntimeError):
            direct = False

    if direct:
        def expanded(value):
            value = _logical(value)
            return value if value.shape == shape else value.expand(shape)
        anchor = next(value for value in original if isinstance(value, torch.Tensor))
        left = expanded(anchor if scalar[0] else a)
        right = left if unary or scalar[1] else expanded(b)
        if scalar[0]:
            left = right
        params = [2 if unary else 1 if op == 2 else 0, 2 * int(scalar[0]), 2 * int(scalar[1]), coefficient]
        if immediate is not None:
            params.append(immediate)
        destination = (out if out is not None and out.shape == shape and out.dtype == dtype
                       and not out.is_conj() and not out.is_neg() else None)
        result = destination if destination is not None else torch.empty(shape, dtype=dtype, device=device)
        _C.scientific_call(485, [left, right, result], params, [])
        return result if destination is not None else _output(result, out, tensors, cast=True)

    def operand(value, cpu_scalar):
        target = math_dtype if cpu_scalar else dtype
        if not isinstance(value, torch.Tensor):
            return _C.predicate_scalar(value, target, device)
        value = _logical(value)
        if value.dtype != target:
            value = value.to(target)
        return value.to(device) if cpu_scalar else value if value.shape == shape else value.expand(shape)

    left = operand(a, scalar[0])
    right = left if unary else operand(b, scalar[1])
    direct = (out is not None and out.dtype == dtype and out.shape == shape
              and not out.is_neg() and not out.is_conj()
              and all(value.is_contiguous() and value.shape == shape for value in (left, right, out)))
    result = out if direct else torch.empty(shape, dtype=dtype, device=device)
    _C.scientific_call(485, [left, right, result], [2 if unary else 1 if op == 2 else 0, int(scalar[0]), int(scalar[1]), coefficient], [])
    return result if direct else _output(result, out, tensors, cast=True)
