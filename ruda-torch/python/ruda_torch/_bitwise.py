from functools import partial

import torch

from . import _C
from ._indexing import _logical
from ._ops import _apply, _integral_dtypes, _logical as _boolean, _typed_value
from ._ordering import _output


def elementwise(a, b=None, *, op, inplace=False, out=None):
    unary = op == 92
    original = (a,) if unary else (a, b)
    tensors = tuple(value for value in original if isinstance(value, torch.Tensor))
    device = next((value.device for value in tensors if value.device.type != 'cpu'), None)
    if device is None:
        raise RuntimeError('RUDA bitwise operations require a GPU tensor')
    for value in tensors:
        if value.layout != torch.strided:
            raise RuntimeError('bitwise operations require strided tensors')
        if value.device != device and not (not unary and value.device.type == 'cpu' and value.ndim == 0):
            raise RuntimeError('operands must share a device; only CPU scalar tensors are permitted')
    dtype = a.dtype if unary else torch.result_type(a, b)
    if dtype not in _integral_dtypes or (op in (397, 398) and dtype == torch.bool):
        raise RuntimeError('bitwise operations require integral results; shifts do not support bool')
    shape = torch.broadcast_shapes(*(value.shape for value in tensors))
    if inplace:
        if not isinstance(a, torch.Tensor):
            raise RuntimeError('in-place bitwise operations require a tensor destination')
        out = a
        if out.shape != shape:
            raise RuntimeError('in-place bitwise operation cannot change the input shape')
    if out is not None:
        valid_dtype = out.dtype == dtype if unary else torch.can_cast(dtype, out.dtype)
        if out.device != device or out.layout != torch.strided or not valid_dtype:
            raise RuntimeError('invalid bitwise output dtype/device/layout')
        for value in tensors:
            _C.check_inplace(out, value)
            if out._cdata == value._cdata and out.shape != shape:
                raise RuntimeError('a bitwise output that is also an input cannot be resized')

    def operand(value):
        if isinstance(value, torch.Tensor):
            return _logical(value).to(device=device, dtype=dtype).expand(shape)
        return _typed_value(value, torch.int64, device).to(dtype).expand(shape)

    left = operand(a)
    right = left if unary else operand(b)
    if left.numel() > 2**32 - 1:
        raise RuntimeError('bitwise operation exceeds native U32 indexing')
    if op in (397, 398):
        result = torch.empty_like(left)
        if result.numel():
            _C.scientific_call(op, [left, right, result], [], [])
    elif dtype == torch.bool:
        result = _boolean(op - 4, left, right)
    else:
        result = _apply(op, left, right)
    return _output(result, out, tensors, cast=not unary)


def register(library):
    library.impl('bitwise_not.out', partial(elementwise, op=92))
    for name, op in (('bitwise_and', 89), ('bitwise_or', 90), ('bitwise_xor', 91)):
        for overload in ('Tensor_out', 'Scalar_out', 'Scalar_Tensor_out'):
            library.impl(name + '.' + overload, partial(elementwise, op=op))
    for name, alias, inplace_alias, op in (
        ('bitwise_left_shift', '__lshift__', '__ilshift__', 397),
        ('bitwise_right_shift', '__rshift__', '__irshift__', 398),
    ):
        for overload in ('Tensor', 'Tensor_Scalar', 'Scalar_Tensor',
                         'Tensor_out', 'Tensor_Scalar_out', 'Scalar_Tensor_out'):
            library.impl(name + '.' + overload, partial(elementwise, op=op))
        for overload in ('Tensor', 'Tensor_Scalar'):
            library.impl(name + '_.' + overload, partial(elementwise, op=op, inplace=True))
        for overload in ('Tensor', 'Scalar'):
            library.impl(alias + '.' + overload, partial(elementwise, op=op))
            library.impl(inplace_alias + '.' + overload, partial(elementwise, op=op, inplace=True))
