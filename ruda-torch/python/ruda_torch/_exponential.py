import torch

from . import _C
from ._indexing import _logical
from ._ops import _numeric_dtypes, _integral_dtypes, _storage_dtypes
from ._ordering import _output


def unary(a, *, op, out=None, inplace=False):
    if a.device.type == 'cpu' or a.layout != torch.strided:
        raise RuntimeError('RUDA exponential/logarithmic operations require a non-CPU strided input')
    if a.dtype not in _numeric_dtypes + _integral_dtypes + (torch.complex32,):
        raise RuntimeError('unsupported exponential/logarithmic input dtype')
    if a.dtype == torch.complex32 and op not in (0, 1, 3):
        raise RuntimeError('this exponential/logarithmic operation is not implemented for ComplexHalf')
    dtype = torch.get_default_dtype() if a.dtype in _integral_dtypes else a.dtype
    if inplace:
        out = a
    if out is not None:
        if out.device != a.device or out.layout != torch.strided or out.dtype not in _storage_dtypes or not torch.can_cast(dtype, out.dtype):
            raise RuntimeError('exponential/logarithmic output requires a compatible dtype on the input device')
        _C.check_inplace(out, a)
    value = _logical(a).to(dtype)
    result = torch.empty_like(value)
    _C.scientific_call(489, [value, result], [op], [])
    return _output(result, out, (a,), cast=True)


def backward(grad, saved, *, tanh=False, grad_input=None):
    original = grad, saved
    supported = _numeric_dtypes + _integral_dtypes + (torch.complex32,)
    if grad.device.type == 'cpu' or saved.device != grad.device:
        raise RuntimeError('sigmoid/tanh backward inputs must share a non-CPU device')
    if any(value.layout != torch.strided or value.dtype not in supported for value in original):
        raise RuntimeError('sigmoid/tanh backward requires supported strided inputs')
    common_dtype = torch.result_type(grad, saved)
    dtype = common_dtype if grad_input is None else grad_input.dtype
    if dtype not in _numeric_dtypes + (torch.complex32,):
        raise RuntimeError('sigmoid/tanh backward requires floating or complex computation')
    if grad_input is not None:
        if grad_input.device != grad.device or grad_input.layout != torch.strided or not torch.can_cast(common_dtype, dtype):
            raise RuntimeError('sigmoid/tanh backward output requires a compatible dtype on the input device')
        for value in original:
            _C.check_inplace(grad_input, value)
    left, right = torch.broadcast_tensors(*(_logical(value).to(dtype) for value in original))
    result = torch.empty_like(left)
    _C.scientific_call(490, [left, right, result], [int(tanh)], [])
    return _output(result, grad_input, original)
