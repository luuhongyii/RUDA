from functools import partial

import torch

from . import _C
from ._indexing import _logical
from ._ops import _dtypes, _numeric_dtypes, _integral_dtypes
from ._ordering import _output


def _input(a, *, complex_allowed=False, complex_half=False):
    types = (_numeric_dtypes if complex_allowed else _dtypes) + _integral_dtypes
    if complex_half:
        types += (torch.complex32,)
    if a.layout != torch.strided or a.dtype not in types:
        raise RuntimeError('unsupported complex/classification input dtype or layout')


def _out_check(out, inputs, dtype, *, cast=False):
    if out is None:
        return
    if out.layout != torch.strided or out.device != inputs[0].device:
        raise RuntimeError('output must be strided and on the input device')
    if not (torch.can_cast(dtype, out.dtype) if cast else dtype == out.dtype):
        raise RuntimeError('output dtype does not match the operation')
    for value in inputs:
        _C.check_inplace(out, value)


def _launch(op, tensors, integers=(), scalars=()):
    if any(value.numel() > 2**32 - 1 for value in tensors):
        raise RuntimeError('complex/classification operation exceeds native U32 indexing')
    if tensors[0].numel():
        _C.scientific_call(op, tensors, list(integers), list(scalars))


def angle(a, *, out=None):
    _input(a, complex_allowed=True, complex_half=True)
    value = _logical(a)
    if a.is_complex():
        parts = torch.view_as_real(value)
        real, imag = parts[..., 0], parts[..., 1]
    else:
        real = value.to(torch.get_default_dtype()) if a.dtype in _integral_dtypes else value
        imag = real
    if real.dtype not in (torch.float32, torch.float64) and not (a.is_complex() and real.dtype == torch.float16):
        raise RuntimeError('CUDA angle requires float32/float64 real input or supported complex input')
    _out_check(out, (a,), real.dtype, cast=True)
    result = torch.empty_like(real)
    _launch(389, [real, imag, result], [int(a.is_complex())])
    return _output(result, out, (a,), cast=True)


def construct(a, b, *, polar=False, out):
    if a.layout != torch.strided or b.layout != torch.strided:
        raise RuntimeError('complex/polar require strided inputs')
    types = (torch.float32, torch.float64) if polar else (torch.float16, torch.float32, torch.float64)
    if a.dtype != b.dtype or a.dtype not in types:
        raise RuntimeError('complex/polar require matching supported floating input dtypes')
    if a.device != b.device:
        raise RuntimeError('complex/polar inputs must share a device')
    dtype = {torch.float16: torch.complex32, torch.float32: torch.complex64, torch.float64: torch.complex128}[a.dtype]
    _out_check(out, (a, b), dtype)
    left, right = torch.broadcast_tensors(_logical(a), _logical(b))
    result = torch.empty(left.shape, dtype=dtype, device=a.device)
    parts = torch.view_as_real(result)
    _launch(391 if polar else 390, [left, right, parts[..., 0], parts[..., 1]])
    return _output(result, out, (a, b))


def nan_to_num(a, nan=None, posinf=None, neginf=None, *, out):
    _input(a, complex_allowed=True)
    _out_check(out, (a,), a.dtype)
    value = _logical(a)
    result = torch.empty_like(value)
    source = torch.view_as_real(value) if a.is_complex() else value
    target = torch.view_as_real(result) if a.is_complex() else result
    if a.dtype in _integral_dtypes:
        replacements = (0.0, 0.0, 0.0)
    else:
        limits = torch.finfo(source.dtype)
        replacements = (0.0 if nan is None else nan,
                        limits.max if posinf is None else posinf,
                        limits.min if neginf is None else neginf)
    _launch(392, [source, target], scalars=replacements)
    return _output(result, out, (a,))


def classify(a, *, op, out=None):
    _input(a)
    _out_check(out, (a,), torch.bool)
    value = _logical(a)
    result = torch.empty_like(value, dtype=torch.bool)
    _launch(op, [value, result])
    return _output(result, out, (a,))


def register(library):
    library.impl('angle', angle)
    library.impl('angle.out', angle)
    library.impl('complex.out', construct)
    library.impl('polar.out', partial(construct, polar=True))
    library.impl('nan_to_num.out', nan_to_num)
    for name, op in (('signbit', 393), ('isposinf', 394), ('isneginf', 395)):
        library.impl(name, partial(classify, op=op))
        library.impl(name + '.out', partial(classify, op=op))
