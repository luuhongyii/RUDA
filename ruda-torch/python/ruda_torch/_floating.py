from functools import partial
import operator

import torch
from . import _C
from ._ops import _dtypes, _integral_dtypes, _numeric_dtypes
from ._ordering import _output, _outputs


def _check_out(out, inputs):
    if out is not None:
        for value in inputs:
            if isinstance(value, torch.Tensor):
                _C.check_inplace(out, value)


def frexp(a, *, mantissa=None, exponent=None):
    if a.dtype not in _dtypes:
        raise RuntimeError('frexp requires a floating-point input')
    outputs = None
    if mantissa is not None or exponent is not None:
        if mantissa is None or exponent is None or mantissa.dtype != a.dtype or exponent.dtype != torch.int32:
            raise RuntimeError('frexp requires matching mantissa dtype and int32 exponent')
        if mantissa.device != a.device or exponent.device != a.device:
            raise RuntimeError('frexp inputs and outputs must share a device')
        _check_out(mantissa, (a,))
        _check_out(exponent, (a,))
        outputs = mantissa, exponent
    value = a.resolve_conj().resolve_neg()
    m = torch.empty_like(value)
    e = torch.empty_like(value, dtype=torch.int32)
    _C.scientific_call(258, [value, m, e], [], [])
    return _outputs((m, e), outputs, (a,))


def ldexp(a, b, *, out=None, inplace=False):
    if a.device != b.device:
        raise RuntimeError('ldexp inputs must share a device')
    if a.dtype not in _numeric_dtypes + _integral_dtypes or b.dtype not in _numeric_dtypes + _integral_dtypes:
        raise RuntimeError('unsupported ldexp dtype')
    if inplace:
        out = a
        if torch.broadcast_shapes(a.shape, b.shape) != a.shape:
            raise RuntimeError('in-place ldexp cannot change shape')
    if out is not None and out.dtype in _integral_dtypes:
        raise RuntimeError('ldexp cannot be cast to an integral output')
    _check_out(out, (a, b))
    if a.dtype in _dtypes and b.dtype in _integral_dtypes:
        left, right = torch.broadcast_tensors(a.resolve_neg(), b.to(torch.int32).resolve_neg())
        result = torch.empty_like(left)
        _C.scientific_call(259, [left, right, result], [], [])
    else:
        # Match ATen's non-integral-exponent and non-floating-input path.
        if a.dtype in _integral_dtypes or a.dtype == torch.float32:
            power = torch.pow(2.0, b)
        else:
            power = torch.full((), 2.0, device=a.device, dtype=a.dtype).pow(b)
        result = a * power
    return _output(result, out, (a, b), cast=True)


def signed_neighbor(a, b, *, op, out=None, inplace=False):
    original = (a, b)
    dtype = torch.result_type(a, b)
    if op == 260 and dtype in _integral_dtypes:
        dtype = torch.get_default_dtype()
    if dtype not in _dtypes:
        raise RuntimeError('operation requires a real floating result dtype')
    device = a.device
    if isinstance(b, torch.Tensor) and a.device != b.device:
        if op == 261:
            raise RuntimeError('nextafter inputs must share a device')
        if a.device.type == 'cpu' and a.ndim == 0:
            device = b.device
        elif not (b.device.type == 'cpu' and b.ndim == 0):
            raise RuntimeError('copysign inputs must share a device except CPU scalars')
    if device.type == 'cpu':
        raise RuntimeError('RUDA operation requires a non-CPU input')
    def operand(x):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype).resolve_conj().resolve_neg()
        return torch.scalar_tensor(x, device=device, dtype=dtype)
    left, right = torch.broadcast_tensors(operand(a), operand(b))
    if inplace:
        out = a
        if out.shape != left.shape or not torch.can_cast(dtype, out.dtype):
            raise RuntimeError('in-place operation cannot change shape or cast result dtype')
    _check_out(out, original)
    result = torch.empty_like(left)
    _C.scientific_call(op, [left, right, result], [], [])
    return _output(result, out, original, cast=True)


def fraction_round(a, *, op, decimals=0, out=None, inplace=False):
    if a.dtype not in _dtypes:
        raise RuntimeError('fractional and decimal-rounding kernels require real floating inputs')
    decimals = operator.index(decimals)
    if inplace:
        out = a
    if out is not None and out.dtype != a.dtype:
        raise RuntimeError('frac and decimal round require matching input and output dtypes')
    _check_out(out, (a,))
    value = a.resolve_conj().resolve_neg()
    result = torch.empty_like(value)
    _C.scientific_call(op, [value, result], [decimals], [])
    return _output(result, out, (a,))


def register(library):
    library.impl('frexp.Tensor', frexp)
    library.impl('frexp.Tensor_out', frexp)
    library.impl('ldexp.Tensor', ldexp)
    library.impl('ldexp.out', ldexp)
    library.impl('ldexp_', partial(ldexp, inplace=True))
    for name in ('copysign.Tensor', 'copysign.Scalar', 'copysign.out', 'copysign.Scalar_out'):
        library.impl(name, partial(signed_neighbor, op=260))
    for name in ('copysign_.Tensor', 'copysign_.Scalar'):
        library.impl(name, partial(signed_neighbor, op=260, inplace=True))
    for name in ('nextafter', 'nextafter.out'):
        library.impl(name, partial(signed_neighbor, op=261))
    library.impl('nextafter_', partial(signed_neighbor, op=261, inplace=True))
    for name in ('frac', 'frac.out'):
        library.impl(name, partial(fraction_round, op=262))
    library.impl('frac_', partial(fraction_round, op=262, inplace=True))
    for name in ('round.decimals', 'round.decimals_out'):
        library.impl(name, partial(fraction_round, op=263))
    library.impl('round_.decimals', partial(fraction_round, op=263, inplace=True))
