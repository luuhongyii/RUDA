import torch

from . import _C
from ._indexing import _logical
from ._ops import _numeric_dtypes, _integral_dtypes, _storage_dtypes
from ._ordering import _output


def power(a, b, *, inplace=False, out=None):
    tensor_base = isinstance(a, torch.Tensor)
    tensor_exponent = isinstance(b, torch.Tensor)
    tensors = tuple(value for value in (a, b) if isinstance(value, torch.Tensor))
    device = next((value.device for value in tensors if value.device.type != 'cpu'), None)
    if device is None:
        raise RuntimeError('RUDA power requires a non-CPU tensor operand')
    for value in tensors:
        if value.layout != torch.strided or value.dtype not in _storage_dtypes:
            raise RuntimeError('power requires supported strided inputs')
        if value.device != device and not (value.device.type == 'cpu' and value.ndim == 0):
            raise RuntimeError('power operands must share a device except a CPU scalar tensor')
    shape = torch.broadcast_shapes(*(value.shape for value in tensors))
    if not tensor_base and a == 1:
        dtype = out.dtype if out is not None else torch.result_type(a, b)
        if dtype not in _storage_dtypes or out is not None and (out.device != device or out.layout != torch.strided):
            raise RuntimeError('power output dtype/device/layout mismatch')
        if out is not None:
            if out.shape != shape:
                _C.prepare_index_output(out, list(shape))
            return out.fill_(1)
        return torch.empty(shape, dtype=dtype, device=device).fill_(1)
    dtype = torch.result_type(a, b)
    parameters = [0, 0, 0]
    if not tensor_exponent:
        parameters = _C.power_scalar(b, dtype, False, a.dtype in _integral_dtypes)
    scalar_identity = not tensor_exponent and parameters[0] in (1, 2)
    if dtype not in _numeric_dtypes + _integral_dtypes + (torch.complex32,) and not scalar_identity:
        raise RuntimeError('power computation dtype is not supported')
    if inplace:
        if a.shape != shape:
            raise RuntimeError('in-place power cannot change the input shape')
        out = a
    if out is not None and (out.device != device or out.layout != torch.strided or out.dtype not in _storage_dtypes or not torch.can_cast(dtype, out.dtype)):
        raise RuntimeError('power output requires a compatible dtype on the operation device')

    def operand(value):
        if not isinstance(value, torch.Tensor):
            return _C.predicate_scalar(value, dtype, device).expand(shape)
        value = _logical(value).to(dtype)
        if value.device != device:
            value = value.to(device)
        return value.expand(shape)

    mode = 0
    scalar_right = not tensor_exponent or b.device.type == 'cpu'
    if scalar_right:
        if tensor_exponent:
            parameters = _C.power_scalar(b, dtype, True, tensor_base and a.dtype in _integral_dtypes)
        mode = 1
    elif tensor_base and a.device.type == 'cpu' and dtype.is_complex:
        parameters = [0, *_C.power_base_log(a, dtype)]
        mode = 2
    elif dtype == torch.bool:
        raise RuntimeError('pow is not implemented for Bool computation')
    left = operand(a) if mode != 2 else operand(b)
    overlap_inputs = (left,) if not tensor_exponent else tensors
    if out is not None:
        for value in overlap_inputs:
            _C.check_inplace(out, value)
    result = torch.empty(shape, dtype=dtype, device=device)
    if parameters[0] == 1:
        result.fill_(1)
    elif parameters[0] == 2:
        result.copy_(left)
    else:
        right = left if mode == 1 else operand(b)
        _C.scientific_call(487, [left, right, result], [mode, *parameters], [])
    return _output(result, out, overlap_inputs, cast=True)
