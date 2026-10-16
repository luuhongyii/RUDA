import torch

from . import _C
from ._indexing import _logical
from ._ops import _storage_dtypes, _typed_value
from ._ordering import _output


def where(condition, a, b, *, out=None):
    dtype = torch.result_type(a, b)
    if dtype not in _storage_dtypes:
        raise RuntimeError('where requires a supported output dtype')
    operands = tuple(value for value in (condition, a, b) if isinstance(value, torch.Tensor))
    device = next((value.device for value in operands if value.device.type != 'cpu'), condition.device)
    if device.type == 'cpu':
        raise RuntimeError('RUDA where requires a non-CPU operand')
    if out is not None and (out.dtype != dtype or out.device != device or out.layout != torch.strided):
        raise RuntimeError('where output must match the result dtype and device and use strided storage')

    def operand(value, target_dtype):
        if not isinstance(value, torch.Tensor):
            return _typed_value(value, target_dtype, device)
        if value.layout != torch.strided:
            raise RuntimeError('where requires strided inputs')
        if value.device != device and value.ndim != 0:
            raise RuntimeError('where only moves zero-dimensional operands between devices')
        return _logical(value.to(device=device, dtype=target_dtype))

    left, right = operand(a, dtype), operand(b, dtype)
    mask = _C.where_condition(operand(condition, condition.dtype))
    inputs = mask, left, right
    mask, left, right = torch.broadcast_tensors(mask, left, right)
    result = torch.empty(left.shape, dtype=dtype, device=device)
    _C.scientific_call(482, [mask, left, right, result], [], [])
    return _output(result, out, inputs)


def masked_fill(a, mask, value, *, inplace=False):
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise RuntimeError('masked_fill requires a zero-dimensional value tensor')
        if a.device.type == 'cpu':
            raise RuntimeError('RUDA masked_fill requires the input and mask on the RUDA device')
        value = value.to(a.dtype).item() if a.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else value.item()
    if a.layout != torch.strided or a.dtype not in _storage_dtypes:
        raise RuntimeError('masked_fill requires a supported strided input')
    if mask.layout != torch.strided or mask.dtype != torch.bool or mask.device != a.device:
        raise RuntimeError('masked_fill requires a strided boolean mask on the input device')
    if inplace:
        _C.masked_fill_native(a, _logical(mask), a, value, True)
        return a
    shape = torch.broadcast_shapes(a.shape, mask.shape)
    source = _logical(a).expand(shape)
    result = torch.empty(shape, dtype=a.dtype, device=a.device)
    _C.masked_fill_native(source, _logical(mask), result, value, False)
    return result
