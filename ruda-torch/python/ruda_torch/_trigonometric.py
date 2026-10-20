import torch

from . import _C
from ._indexing import _logical
from ._ops import _numeric_dtypes, _integral_dtypes, _storage_dtypes
from ._ordering import _output


def unary(a, *, op, out=None, inplace=False):
    if a.device.type == 'cpu' or a.layout != torch.strided:
        raise RuntimeError('RUDA trigonometric operations require a non-CPU strided input')
    if a.dtype not in _numeric_dtypes + _integral_dtypes + (torch.complex32,):
        raise RuntimeError('unsupported trigonometric input dtype')
    dtype = torch.get_default_dtype() if a.dtype in _integral_dtypes else a.dtype
    if inplace:
        out = a
    if out is not None:
        if out.device != a.device or out.layout != torch.strided or out.dtype not in _storage_dtypes or not torch.can_cast(dtype, out.dtype):
            raise RuntimeError('trigonometric output requires a compatible dtype on the input device')
        _C.check_inplace(out, a)
    value = _logical(a).to(dtype)
    result = torch.empty_like(value)
    _C.scientific_call(491, [value, result], [op], [])
    return _output(result, out, (a,), cast=True)
