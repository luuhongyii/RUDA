import torch

from . import _C
from ._indexing import _logical
from ._ops import _storage_dtypes
from ._ordering import _output


def predicate(a, b=None, *, op, inplace=False, out=None):
    unary = op == 88
    logical = op >= 85
    if unary:
        b = a
    if logical and not isinstance(b, torch.Tensor):
        raise TypeError('logical operations require Tensor operands')
    tensors = tuple(value for value in (a, b) if isinstance(value, torch.Tensor))
    device = next((value.device for value in tensors if value.device.type != 'cpu'), None)
    if device is None:
        raise RuntimeError('RUDA predicates require a non-CPU tensor operand')
    if logical and not unary and not inplace and out is None and a.device.type == 'cpu':
        raise RuntimeError('functional logical operations require the first input on the output device')
    for value in tensors:
        if value.layout != torch.strided or value.dtype not in _storage_dtypes:
            raise RuntimeError('predicates require supported strided inputs')
        if value.device != device and not (value.device.type == 'cpu' and value.ndim == 0):
            raise RuntimeError('predicate operands must share a device except a CPU scalar tensor')
    fp8 = (torch.float8_e4m3fn, torch.float8_e5m2)
    if unary:
        dtype = a.dtype
    elif logical and a.dtype != b.dtype and (a.dtype in fp8 or b.dtype in fp8):
        dtype = torch.bool
    else:
        dtype = torch.result_type(a, b)
    if dtype not in _storage_dtypes:
        raise RuntimeError('predicate common dtype is not supported')
    if 80 <= op <= 83 and (dtype.is_complex or dtype in fp8):
        raise RuntimeError('ordered comparisons require a supported real dtype')
    shape = torch.broadcast_shapes(*(value.shape for value in tensors))
    if inplace:
        if a.shape != shape:
            raise RuntimeError('in-place predicate cannot change the input shape')
        out = a
    if out is not None:
        if out.device != device or out.layout != torch.strided or out.dtype not in _storage_dtypes:
            raise RuntimeError('predicate output requires supported storage on the operation device')
        for value in tensors:
            _C.check_inplace(out, value)

    def operand(value):
        if not isinstance(value, torch.Tensor):
            return _C.predicate_scalar(value, dtype, device).expand(shape)
        value = _logical(value).to(dtype)
        if value.device != device:
            value = value.to(device)
        return value.expand(shape)

    left = operand(a)
    right = left if unary else operand(b)
    result = torch.empty(shape, dtype=torch.bool, device=device)
    _C.scientific_call(484, [left, right, result], [op], [])
    return _output(result, out, tensors, cast=True)
