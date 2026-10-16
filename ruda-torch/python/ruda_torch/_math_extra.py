from functools import partial

import torch

from . import _C
from ._indexing import _logical
from ._ops import _dimension, _dtypes, _numeric_dtypes, _integral_dtypes
from ._ordering import _output


def binary(a, b, *, op, out=None, inplace=False):
    original = (a, b)
    if any(value.layout != torch.strided or value.dtype not in _dtypes + _integral_dtypes for value in original):
        raise RuntimeError('heaviside/gcd/lcm require strided real operands')
    dtype = torch.result_type(a, b)
    if op == 384:
        if a.dtype != b.dtype or (out is not None and out.dtype != a.dtype):
            raise RuntimeError('heaviside requires identical input and output dtypes')
        dtype = a.dtype
    elif dtype not in _integral_dtypes or dtype == torch.bool:
        raise RuntimeError('gcd/lcm require a non-bool integral common dtype')
    device = next((value.device for value in original if value.device.type != 'cpu'), None)
    if device is None:
        raise RuntimeError('RUDA math operations require a non-CPU operand')
    for value in original:
        if value.device != device and not (op == 384 and value.device.type == 'cpu' and value.ndim == 0):
            raise RuntimeError('operands must share a device; only heaviside accepts CPU scalar tensors')
    left, right = torch.broadcast_tensors(*(_logical(value).to(device=device, dtype=dtype) for value in original))
    if left.numel() > 2**32 - 1:
        raise RuntimeError('math output exceeds native U32 indexing')
    if inplace:
        out = a
        if out.shape != left.shape or not torch.can_cast(dtype, out.dtype):
            raise RuntimeError('in-place operation cannot change shape or cast the result dtype')
    if out is not None:
        for value in original:
            _C.check_inplace(out, value)
    result = torch.empty_like(left)
    if result.numel():
        _C.scientific_call(op, [left, right, result], [], [])
    return _output(result, out, original, cast=op != 384)


def _parts(value):
    if value.is_complex():
        parts = torch.view_as_real(value)
        return [parts[..., 0], parts[..., 1]]
    return [value, value]


def sinc(a, *, out=None, inplace=False):
    if a.layout != torch.strided or a.dtype not in _numeric_dtypes + _integral_dtypes:
        raise RuntimeError('sinc requires a strided numeric input')
    dtype = torch.get_default_dtype() if a.dtype in _integral_dtypes else a.dtype
    if inplace:
        out = a
        if not torch.can_cast(dtype, out.dtype):
            raise RuntimeError('in-place sinc cannot cast floating results to integral storage')
    if out is not None:
        _C.check_inplace(out, a)
    if a.numel() > 2**32 - 1:
        raise RuntimeError('sinc exceeds native U32 indexing')
    value = _logical(a).to(dtype)
    result = torch.empty_like(value)
    if result.numel():
        _C.scientific_call(387, _parts(value) + _parts(result), [int(value.is_complex())], [])
    return _output(result, out, (a,), cast=True)


def cross(a, b, *, dim=-1, out=None):
    if a.ndim == 0 or a.ndim != b.ndim:
        raise RuntimeError('linalg.cross requires nonscalar inputs with the same rank')
    if a.layout != torch.strided or b.layout != torch.strided:
        raise RuntimeError('linalg.cross requires strided input')
    if a.device != b.device or a.dtype != b.dtype:
        raise RuntimeError('linalg.cross requires matching input dtypes and devices')
    if a.dtype not in _numeric_dtypes + _integral_dtypes or a.dtype == torch.bool:
        raise RuntimeError('unsupported linalg.cross dtype')
    axis = _dimension(dim, a.ndim)
    if a.shape[axis] != 3 or b.shape[axis] != 3:
        raise RuntimeError('linalg.cross dimension must have length three in both inputs')
    if out is not None:
        _C.check_index_output(out, a, b)
    left, right = torch.broadcast_tensors(_logical(a), _logical(b))
    if left.numel() > 2**32 - 1:
        raise RuntimeError('linalg.cross exceeds native U32 indexing')
    result = torch.empty(left.shape, dtype=a.dtype, device=a.device)
    if result.numel():
        _C.scientific_call(388, _parts(left) + _parts(right) + _parts(result), [axis, int(a.is_complex())], [])
    return _output(result, out, (a, b))


def register(library):
    library.impl('heaviside', lambda a, values: binary(a, values, op=384))
    library.impl('heaviside.out', lambda a, values, *, out: binary(a, values, op=384, out=out))
    library.impl('heaviside_', lambda a, values: binary(a, values, op=384, inplace=True))
    for name, op in (('gcd', 385), ('lcm', 386)):
        library.impl(name, partial(binary, op=op))
        library.impl(name + '.out', partial(binary, op=op))
        library.impl(name + '_', partial(binary, op=op, inplace=True))
    library.impl('sinc', sinc)
    library.impl('sinc.out', sinc)
    library.impl('sinc_', partial(sinc, inplace=True))
    library.impl('linalg_cross', cross)
    library.impl('linalg_cross.out', cross)
