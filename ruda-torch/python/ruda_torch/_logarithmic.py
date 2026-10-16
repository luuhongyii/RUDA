import torch

from . import _C
from ._indexing import _logical
from ._ops import _dtypes, _numeric_dtypes, _integral_dtypes, _storage_dtypes
from ._ordering import _output


def _binary(a, b, *, op, eps=None, out=None):
    original = a, b
    if a.device.type == 'cpu' or b.device != a.device:
        raise RuntimeError('logarithmic binary inputs must share a non-CPU device')
    if any(value.layout != torch.strided or value.dtype not in _storage_dtypes for value in original):
        raise RuntimeError('logarithmic binary operations require supported strided inputs')
    common_dtype = torch.result_type(a, b)
    dtype = common_dtype if out is None else out.dtype
    supported = _numeric_dtypes + (torch.complex32,) if op == 0 else _dtypes
    if dtype not in supported:
        raise RuntimeError('unsupported logarithmic computation dtype')
    if dtype.is_complex and any(value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) for value in original):
        raise RuntimeError('complex logaddexp does not support FP8 input conversion')
    if out is not None:
        if out.device != a.device or out.layout != torch.strided or not torch.can_cast(common_dtype, dtype):
            raise RuntimeError('logarithmic output requires a compatible dtype on the input device')
        for value in original:
            _C.check_inplace(out, value)
    parameters = [_C.logit_epsilon(-1.0 if eps is None else float(eps), dtype)] if op == 3 else []
    left, right = torch.broadcast_tensors(*(_logical(value).to(dtype) for value in original))
    result = torch.empty_like(left)
    _C.scientific_call(492, [left, right, result], [op], parameters)
    return _output(result, out, original)


def binary(a, b, *, op, out=None):
    return _binary(a, b, op=op, out=out)


def backward(grad_output, a, eps=None, *, grad_input=None):
    return _binary(grad_output, a, op=3, eps=eps, out=grad_input)


def logit(a, eps=None, *, out=None, inplace=False):
    if a.device.type == 'cpu' or a.layout != torch.strided or a.dtype not in _dtypes + _integral_dtypes:
        raise RuntimeError('logit requires a supported real non-CPU strided input')
    dtype = torch.get_default_dtype() if a.dtype in _integral_dtypes else a.dtype
    if inplace:
        out = a
    if out is not None:
        if out.device != a.device or out.layout != torch.strided or out.dtype not in _storage_dtypes or not torch.can_cast(dtype, out.dtype):
            raise RuntimeError('logit output requires a compatible dtype on the input device')
        _C.check_inplace(out, a)
    epsilon = _C.logit_epsilon(-1.0 if eps is None else float(eps), dtype)
    value = _logical(a).to(dtype)
    result = torch.empty_like(value)
    _C.scientific_call(492, [value, value, result], [2], [epsilon])
    return _output(result, out, (a,), cast=True)
