import torch

from . import _C
from ._indexing import _logical
from ._ops import _dtypes, _numeric_dtypes, _integral_dtypes, _storage_dtypes
from ._ordering import _output


def quotient(a, b, *, op, inplace=False, out=None, exact_out=False):
    tensors = tuple(value for value in (a, b) if isinstance(value, torch.Tensor))
    device = next((value.device for value in tensors if value.device.type != 'cpu'), None)
    if device is None:
        raise RuntimeError('RUDA division requires a non-CPU tensor operand')
    supported = _numeric_dtypes + _integral_dtypes + (torch.complex32,)
    for value in tensors:
        if value.layout != torch.strided or value.dtype not in supported:
            raise RuntimeError('division requires supported strided inputs')
        if value.device != device and not (value.device.type == 'cpu' and value.ndim == 0):
            raise RuntimeError('division operands must share a device except a CPU scalar tensor')
    dtype = torch.result_type(a, b)
    if op == 0 and dtype in _integral_dtypes:
        dtype = torch.get_default_dtype()
    if dtype not in supported or op != 0 and (dtype == torch.bool or dtype.is_complex):
        raise RuntimeError('division operation does not support the promoted dtype')
    shape = torch.broadcast_shapes(*(value.shape for value in tensors))
    if inplace:
        if a.shape != shape:
            raise RuntimeError('in-place division cannot change the input shape')
        out = a
    if out is not None:
        if out.device != device or out.layout != torch.strided or out.dtype not in _storage_dtypes or not torch.can_cast(dtype, out.dtype):
            raise RuntimeError('division output requires a compatible dtype on the operation device')
        if exact_out and out.dtype != dtype:
            raise RuntimeError('generated division output requires the functional result dtype')
        for value in tensors:
            _C.check_inplace(out, value)
    if dtype.is_complex:
        from ._complex32_ops import binary
        return binary(a, b, op=3, out=out)

    def operand(value):
        if not isinstance(value, torch.Tensor):
            return _C.predicate_scalar(value, dtype, device).expand(shape)
        value = _logical(value).to(dtype)
        if value.device != device:
            value = value.to(device)
        return value.expand(shape)

    scalar_right = op <= 2 and dtype in _dtypes and (not isinstance(b, torch.Tensor) or b.device.type == 'cpu')
    parameters = _C.division_scalar(b, dtype, op) if scalar_right else (0, 0)
    left = operand(a)
    right = left if scalar_right else operand(b)
    result = torch.empty(shape, dtype=dtype, device=device)
    _C.scientific_call(486, [left, right, result], [op, int(scalar_right), *parameters], [])
    return _output(result, out, tensors, cast=True)


def divide(a, b, *, rounding_mode=None, inplace=False, out=None):
    if rounding_mode not in (None, 'trunc', 'floor'):
        raise RuntimeError("rounding_mode must be None, 'trunc' or 'floor'")
    return quotient(a, b, op=0 if rounding_mode is None else 1 if rounding_mode == 'trunc' else 2,
                    inplace=inplace, out=out, exact_out=out is not None and not isinstance(b, torch.Tensor))


def modulus(a, b, *, floor, inplace=False, out=None):
    return quotient(a, b, op=3 if floor else 4, inplace=inplace, out=out,
                    exact_out=out is not None and not isinstance(a, torch.Tensor))
