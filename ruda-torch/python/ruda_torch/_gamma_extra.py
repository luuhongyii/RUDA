from functools import partial

import torch

from . import _C
from ._indexing import _logical
from ._ops import _dtypes, _integral_dtypes
from ._ordering import _output


def binary(a, b, *, op, out=None, inplace=False):
    original = (a, b)
    if any(value.layout != torch.strided or value.dtype not in _dtypes + _integral_dtypes for value in original):
        raise RuntimeError('incomplete gamma and zeta require strided real operands')
    dtype = torch.result_type(a, b)
    if op == 382 and dtype in _integral_dtypes:
        dtype = torch.get_default_dtype()
    if dtype not in (torch.float32, torch.float64):
        raise RuntimeError('incomplete gamma and zeta require a float32 or float64 common dtype')
    device = next((value.device for value in original if value.device.type != 'cpu'), None)
    if device is None:
        raise RuntimeError('RUDA special functions require a non-CPU tensor operand')
    for value in original:
        if value.device != device and not (op == 382 and value.device.type == 'cpu' and value.ndim == 0):
            raise RuntimeError('operands must share a device; only zeta accepts CPU scalar tensors')
    left, right = torch.broadcast_tensors(*(_logical(value).to(device=device, dtype=dtype) for value in original))
    if left.numel() > 2**32 - 1:
        raise RuntimeError('special function output exceeds native U32 indexing')
    if inplace:
        out = a
        if out.shape != left.shape or not torch.can_cast(dtype, out.dtype):
            raise RuntimeError('in-place incomplete gamma cannot change shape or cast the result dtype')
    if out is not None:
        for value in original:
            _C.check_inplace(out, value)
    result = torch.empty_like(left)
    if result.numel():
        _C.scientific_call(op, [left, right, result], [], [])
    return _output(result, out, original, cast=True)


@torch.no_grad()
def airy_ai(x, *, out=None):
    if x.layout != torch.strided or x.dtype not in _dtypes + _integral_dtypes:
        raise RuntimeError('Airy Ai requires strided real input')
    dtype = torch.get_default_dtype() if x.dtype in _integral_dtypes else x.dtype
    if dtype not in (torch.float32, torch.float64):
        raise RuntimeError('Airy Ai requires float32 or float64')
    if x.numel() > 2**32 - 1:
        raise RuntimeError('Airy Ai exceeds native U32 indexing')
    if out is not None:
        _C.check_inplace(out, x)
    value = _logical(x).to(dtype)
    result = torch.empty_like(value)
    if result.numel():
        _C.scientific_call(383, [value, value, result], [], [])
    return _output(result, out, (x,), cast=True)


def register(library):
    for name, op in (('igamma', 380), ('igammac', 381)):
        library.impl(name, partial(binary, op=op))
        library.impl(name + '.out', partial(binary, op=op))
        library.impl(name + '_', partial(binary, op=op, inplace=True))
    library.impl('special_zeta', partial(binary, op=382))
    library.impl('special_zeta.out', partial(binary, op=382))
    library.impl('special_airy_ai', airy_ai)
    library.impl('special_airy_ai.out', airy_ai)
