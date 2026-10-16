import torch

from . import _C
from ._indexing import _logical
from ._ops import _numeric_dtypes, _integral_dtypes, _storage_dtypes
from ._ordering import _output


def unary(a, *, op, out=None, inplace=False):
    if a.device.type == 'cpu' or a.layout != torch.strided:
        raise RuntimeError('RUDA unary operations require a non-CPU strided input')
    if a.dtype not in _numeric_dtypes + _integral_dtypes + (torch.complex32,):
        raise RuntimeError('unsupported unary input dtype')
    if a.is_complex() and op in (1, 3, 4, 5, 6):
        raise RuntimeError('sign and rounding operations do not support complex inputs')
    if a.dtype == torch.bool and 3 <= op <= 6:
        raise RuntimeError('rounding operations do not support Bool input')
    if a.dtype == torch.complex32 and op == 9:
        raise RuntimeError('reciprocal is not implemented for ComplexHalf')
    if inplace:
        if a.is_complex() and op == 0:
            raise RuntimeError('in-place abs is not supported for complex tensors')
        out = a
    floating = op >= 7
    dtype = torch.get_default_dtype() if floating and a.dtype in _integral_dtypes else a.dtype
    complex_abs = op == 0 and a.is_complex()
    if complex_abs:
        dtype = {torch.complex32: torch.float16, torch.complex64: torch.float32, torch.complex128: torch.float64}[a.dtype]
    inputs = (a,)
    if out is not None:
        if out.device != a.device or out.layout != torch.strided or out.dtype not in _storage_dtypes:
            raise RuntimeError('unary output dtype/device/layout mismatch')
        if complex_abs and not out.is_complex():
            if not torch.can_cast(dtype, out.dtype):
                raise RuntimeError('complex abs cannot cast its real result to the requested output dtype')
            inputs = ()
        elif floating:
            if not torch.can_cast(dtype, out.dtype):
                raise RuntimeError('unary floating result cannot be cast to the output dtype')
        elif out.dtype != a.dtype:
            raise RuntimeError('this unary operation requires matching input and output dtypes')
        _C.check_inplace(out, out)
        for value in inputs:
            _C.check_inplace(out, value)
    value = _logical(a)
    if floating:
        value = value.to(dtype)
    result = torch.empty_like(value, dtype=dtype)
    if floating:
        _C.scientific_call(487, [value, value, result], [1, op-4, 0, 0], [])
    else:
        _C.scientific_call(488, [value, result], [op], [])
    return _output(result, out, inputs, cast=floating or complex_abs)
