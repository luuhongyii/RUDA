from __future__ import annotations
import ctypes
import operator
import torch
from . import _C

WIDE = (torch.float64, torch.complex64, torch.complex128)
COMPLEX = (torch.complex64, torch.complex128)
_libraries = []


def initialize(library):
    if not hasattr(library, 'ruda_torch_scientific_api_version') or not hasattr(_C, 'initialize_scientific'):
        return False
    library.ruda_torch_scientific_api_version.restype = ctypes.c_uint32
    if library.ruda_torch_scientific_api_version() != 1 or _C.scientific_api_version != 1:
        raise RuntimeError('RUDA scientific API mismatch; rebuild both native libraries')
    _C.initialize_scientific(ctypes.cast(library.ruda_torch_scientific, ctypes.c_void_p).value)
    return True


def execute(op, a, b, out, scalar=1.0):
    a = a.resolve_conj().resolve_neg()
    b = b.resolve_conj().resolve_neg()
    target = out.resolve_conj().resolve_neg() if out.is_conj() or out.is_neg() else out
    _C.scientific_call(op, [a, b, target], [], [float(scalar)])
    if target is not out:
        out.copy_(target)
    return out


def reduce(a, dim=None, keepdim=False, *, dtype=None, mean=False, product=False):
    if dtype is not None:
        a = a.to(dtype)
    if not mean and not (a.is_floating_point() or a.is_complex()):
        from ._ops import _primitive_reduce
        return _primitive_reduce(98 if product else 99, a, dim, keepdim, dtype=a.dtype)
    if mean and not (a.is_floating_point() or a.is_complex()):
        raise RuntimeError('mean requires a floating or complex dtype')
    dims = tuple(range(a.ndim)) if dim is None else (dim,) if isinstance(dim, int) else tuple(dim)
    if not dims:
        dims = tuple(range(a.ndim))
    normalized = []
    for d in dims:
        d = operator.index(d)
        if d < -max(a.ndim, 1) or d >= max(a.ndim, 1):
            raise IndexError('reduction dimension out of range')
        normalized.append(d % a.ndim if a.ndim else 0)
    if len(set(normalized)) != len(normalized):
        raise RuntimeError('duplicate reduction dimension')
    shape = tuple(1 if i in normalized else n for i, n in enumerate(a.shape))
    out = torch.empty(shape, dtype=a.dtype, device=a.device)
    _C.scientific_call(107 if mean else 98 if product else 6,
                       [a.resolve_conj().resolve_neg(), out], [], [])
    if not keepdim:
        out = out.view(tuple(n for i, n in enumerate(a.shape) if i not in normalized))
    return out


def absolute(a, *, out=None):
    dtype = torch.float64 if a.dtype == torch.complex128 else torch.float32 if a.dtype == torch.complex64 else a.dtype
    if out is None:
        out = torch.empty(a.shape, dtype=dtype, device=a.device)
    if out.shape != a.shape or out.dtype != dtype or out.device != a.device:
        raise RuntimeError('abs output shape/dtype/device mismatch')
    execute(21, a, a, out)
    return out


def conjugate_physical(a, *, out=None):
    if not a.is_complex():
        if out is not None:
            return out.copy_(a)
        return a
    if out is None:
        out = torch.empty_like(a)
    return execute(206, a, a, out)


def sign(a, *, out=None):
    if out is None:
        out = torch.empty_like(a)
    return execute(214 if a.is_complex() else 22, a, a, out)


def triangular(a, diagonal=0, *, upper=False, out=None):
    if a.ndim < 2:
        raise RuntimeError('tril/triu requires at least two dimensions')
    if out is None:
        out = torch.empty_like(a)
    if out.shape != a.shape or out.dtype != a.dtype or out.device != a.device:
        raise RuntimeError('triangular output shape/dtype/device mismatch')
    source = a.resolve_conj().resolve_neg()
    target = torch.empty_like(out) if out.is_conj() or out.is_neg() else out
    _C.scientific_call(201, [source, target], [operator.index(diagonal), int(upper)], [])
    if target is not out:
        out.copy_(target)
    return out


def diagonal(a, offset=0, dim1=0, dim2=1):
    rank = a.ndim
    if rank < 2 or dim1 < -rank or dim1 >= rank or dim2 < -rank or dim2 >= rank:
        raise IndexError('diagonal dimensions out of range')
    dim1, dim2 = dim1 % rank, dim2 % rank
    if dim1 == dim2:
        raise RuntimeError('diagonal dimensions must differ')
    offset = operator.index(offset)
    rows, cols = a.shape[dim1], a.shape[dim2]
    length = max(0, min(rows, cols-offset) if offset >= 0 else min(rows+offset, cols))
    shape = [n for d, n in enumerate(a.shape) if d not in (dim1, dim2)] + [length]
    strides = [n for d, n in enumerate(a.stride()) if d not in (dim1, dim2)] + [a.stride(dim1)+a.stride(dim2)]
    start = a.storage_offset()
    if length:
        start += offset*a.stride(dim2) if offset >= 0 else -offset*a.stride(dim1)
    return torch.as_strided(a, shape, strides, start)


def diag_embed(a, offset=0, dim1=-2, dim2=-1):
    if a.ndim < 1:
        raise RuntimeError('diag_embed requires at least one dimension')
    rank = a.ndim+1
    if dim1 < -rank or dim1 >= rank or dim2 < -rank or dim2 >= rank:
        raise IndexError('diag_embed dimension out of range')
    dim1, dim2 = dim1 % rank, dim2 % rank
    if dim1 == dim2:
        raise RuntimeError('diag_embed dimensions must differ')
    side = a.shape[-1]+abs(operator.index(offset))
    remaining = iter(a.shape[:-1])
    shape = tuple(side if d in (dim1, dim2) else next(remaining) for d in range(rank))
    output = torch.zeros(shape, dtype=a.dtype, device=a.device)
    diagonal(output, offset, dim1, dim2).copy_(a)
    return output


def diag(a, diagonal_offset=0, *, out=None):
    if a.ndim == 1:
        result = diag_embed(a, diagonal_offset)
    elif a.ndim == 2:
        result = diagonal(a, diagonal_offset).clone()
    else:
        raise RuntimeError('diag requires a vector or a matrix')
    if out is not None:
        if out.device != result.device or out.dtype != result.dtype:
            raise RuntimeError('diag output dtype/device mismatch')
        out.resize_(result.shape)
        return out.copy_(result)
    return result


def eye(n, m=None, *, dtype=None, layout=None, device=None, pin_memory=False, out=None):
    n, m = operator.index(n), operator.index(n if m is None else m)
    if n < 0 or m < 0:
        raise RuntimeError('eye dimensions must be nonnegative')
    if layout not in (None, torch.strided) or pin_memory:
        raise RuntimeError('RUDA eye requires strided device storage')
    if out is None:
        out = torch.zeros((n, m), dtype=dtype or torch.get_default_dtype(), device=device or 'ruda')
    else:
        out.resize_((n, m))
        out.zero_()
    diagonal(out).fill_(1)
    return out


def register():
    if _libraries:
        return
    library = torch.library.Library('aten', 'IMPL', 'PrivateUse1')
    entries = {
        'tril': lambda a, diagonal=0: triangular(a, diagonal),
        'tril_': lambda a, diagonal=0: triangular(a, diagonal, out=a),
        'tril.out': lambda a, diagonal=0, *, out: triangular(a, diagonal, out=out),
        'triu': lambda a, diagonal=0: triangular(a, diagonal, upper=True),
        'triu_': lambda a, diagonal=0: triangular(a, diagonal, upper=True, out=a),
        'triu.out': lambda a, diagonal=0, *, out: triangular(a, diagonal, upper=True, out=out),
        'diagonal': diagonal,
        'diag': diag,
        'diag.out': lambda a, diagonal=0, *, out: diag(a, diagonal, out=out),
        'diag_embed': diag_embed,
        'eye': eye,
        'eye.m': eye,
        'eye.out': eye,
        'eye.m_out': eye,
        'conj_physical': conjugate_physical,
        'conj_physical_': lambda a: conjugate_physical(a, out=a),
        'conj_physical.out': conjugate_physical,
        'erf': lambda a: execute(216, a, a, torch.empty_like(a)),
        'erf_': lambda a: execute(216, a, a, a),
        'erf.out': lambda a, *, out: execute(216, a, a, out),
    }
    for name, function in entries.items():
        library.impl(name, function)
    _libraries.append(library)
