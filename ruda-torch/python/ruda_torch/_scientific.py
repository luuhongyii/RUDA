from __future__ import annotations
import ctypes
from functools import partial
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


def unary_math(a, *, op, out=None, inplace=False):
    from ._ops import _numeric_dtypes, _integral_dtypes, _apply
    from ._ordering import _output
    if a.dtype not in _numeric_dtypes + _integral_dtypes:
        raise RuntimeError('unsupported transcendental input dtype')
    if (op == 216 or op >= 223) and a.is_complex():
        raise RuntimeError('error and normal-distribution functions require real tensors')
    dtype = torch.get_default_dtype() if a.dtype in _integral_dtypes else a.dtype
    if inplace:
        out = a
        if not torch.can_cast(dtype, a.dtype):
            raise RuntimeError('floating result cannot be cast to the in-place input dtype')
    if out is not None:
        _C.check_inplace(out, a)
    value = a.to(dtype)
    if op < 216:
        result = _apply(op, value)
    else:
        result = execute(op, value, value, torch.empty_like(value))
    return _output(result, out, (a,), cast=True)


def gamma_math(a, *, op, order=0, out=None, inplace=False):
    from ._ops import _dtypes, _integral_dtypes
    from ._ordering import _output
    order = operator.index(order)
    if order < 0:
        raise RuntimeError('polygamma(n, x) does not support negative n')
    if a.dtype not in _dtypes + _integral_dtypes:
        raise RuntimeError('gamma functions require real tensors')
    dtype = torch.get_default_dtype() if a.dtype in _integral_dtypes else a.dtype
    if inplace:
        out = a
        if not torch.can_cast(dtype, a.dtype):
            raise RuntimeError('floating result cannot be cast to the in-place input dtype')
    if out is not None:
        _C.check_inplace(out, a)
    value = a.to(dtype).resolve_neg()
    result = torch.empty_like(value)
    _C.scientific_call(op, [value, result], [order], [])
    return _output(result, out, (a,), cast=True)


def polygamma(n, a, *, out=None):
    return gamma_math(a, op=230, order=n, out=out)


def bessel_math(a, *, op, out=None, inplace=False):
    if a.is_complex() or (op >= 237 and a.dtype in (torch.float16, torch.bfloat16)):
        raise RuntimeError('unsupported Bessel input dtype')
    if op >= 237:
        with torch.no_grad():
            return gamma_math(a, op=op, out=out, inplace=inplace)
    return gamma_math(a, op=op, out=out, inplace=inplace)


def elementary_math(a, b=None, *, op, eps=None, out=None, inplace=False):
    if b is None and 249 <= op <= 252:
        from ._exponential import unary
        return unary(a, op=op-245, out=out, inplace=inplace)
    if op in (253, 254):
        from ._logarithmic import binary
        return binary(a, b, op=op-253, out=out)
    if op == 255:
        from ._logarithmic import logit as native_logit
        return native_logit(a, eps, out=out, inplace=inplace)
    if op == 256:
        from ._logarithmic import backward
        return backward(a, b, eps, grad_input=out)
    from ._ops import _numeric_dtypes, _integral_dtypes
    from ._ordering import _output
    original = (a,) if b is None else (a, b)
    if any(t.dtype not in _numeric_dtypes + _integral_dtypes for t in original):
        raise RuntimeError('unsupported elementary input dtype')
    dtype = a.dtype if b is None else torch.result_type(a, b)
    if dtype in _integral_dtypes:
        if b is not None:
            raise RuntimeError('binary logarithmic functions require floating or complex inputs')
        dtype = torch.get_default_dtype()
    if op >= 254 and dtype in COMPLEX:
        raise RuntimeError('operation requires real inputs')
    if b is None:
        b = a
    if a.device != b.device:
        raise RuntimeError('elementary operands must share a device')
    if a.device.type == 'cpu':
        raise RuntimeError('elementary RUDA operation requires a non-CPU input')
    left, right = torch.broadcast_tensors(a.to(dtype), b.to(dtype))
    if inplace:
        out = original[0]
        if out.shape != left.shape or not torch.can_cast(dtype, out.dtype):
            raise RuntimeError('in-place operation cannot change shape or cast the result dtype')
    if out is not None:
        for value in original:
            _C.check_inplace(out, value)
    result = execute(op, left, right, torch.empty_like(left), -1.0 if eps is None else float(eps))
    return _output(result, out, original, cast=True)


def logit(a, eps=None, *, out=None, inplace=False):
    return elementary_math(a, op=255, eps=eps, out=out, inplace=inplace)


def logit_backward(grad_output, a, eps=None, *, grad_input=None):
    return elementary_math(grad_output, a, op=256, eps=eps, out=grad_input)


def mvlgamma_out(a, p, *, out):
    from ._ordering import _output
    return _output(torch.mvlgamma(a, p), out, (), cast=True)


def logarithmic_product(a, b, *, op, out=None, inplace=False):
    from ._ops import _dtypes, _integral_dtypes
    from ._ordering import _output
    original = tuple(x for x in (a, b) if isinstance(x, torch.Tensor))
    dtype = torch.result_type(a, b)
    if dtype in _integral_dtypes:
        dtype = torch.get_default_dtype()
    if dtype not in _dtypes:
        raise RuntimeError('xlogy/xlog1py require real inputs')
    device = next((x.device for x in original if x.device.type != 'cpu'), None)
    if device is None:
        raise RuntimeError('xlogy/xlog1py require a non-CPU tensor operand')
    def operand(value):
        if not isinstance(value, torch.Tensor):
            return torch.scalar_tensor(value, dtype=dtype, device=device)
        if value.device != device and not (value.device.type == 'cpu' and value.ndim == 0):
            raise RuntimeError('xlogy/xlog1py operands must share a device')
        return value.to(device=device, dtype=dtype)
    left, right = torch.broadcast_tensors(operand(a), operand(b))
    if inplace:
        out = a
        if out.shape != left.shape or not torch.can_cast(dtype, out.dtype):
            raise RuntimeError('in-place operation cannot change shape or cast the result dtype')
    if out is not None:
        for value in original:
            _C.check_inplace(out, value)
    return _output(execute(op, left, right, torch.empty_like(left)), out, original, cast=True)


def binary_math(a, b, *, op, out=None, inplace=False):
    from ._ops import _dtypes, _integral_dtypes
    from ._ordering import _output
    if a.dtype not in _dtypes + _integral_dtypes or b.dtype not in _dtypes + _integral_dtypes:
        raise RuntimeError('atan2/hypot require real floating or integral tensors')
    dtype = torch.result_type(a, b)
    if op == 222 and dtype not in _dtypes:
        raise RuntimeError('hypot requires floating point inputs')
    if dtype not in _dtypes:
        dtype = torch.get_default_dtype()
    original = a, b
    if a.device != b.device:
        if a.device.type == 'cpu' and a.ndim == 0:
            a = a.to(b.device)
        elif b.device.type == 'cpu' and b.ndim == 0:
            b = b.to(a.device)
        else:
            raise RuntimeError('atan2/hypot operands must share a device')
    left, right = torch.broadcast_tensors(a.to(dtype), b.to(dtype))
    if inplace:
        out = original[0]
        if out.shape != left.shape or not torch.can_cast(dtype, out.dtype):
            raise RuntimeError('in-place operation cannot change shape or cast the result dtype')
    if out is not None:
        for value in original:
            _C.check_inplace(out, value)
    result = execute(op, left, right, torch.empty_like(left))
    return _output(result, out, original, cast=True)


def reduce(a, dim=None, keepdim=False, *, dtype=None, mean=False, product=False):
    if product and not mean:
        from ._product_reduce import product_reduce
        return product_reduce(a, dim, keepdim, dtype=dtype)
    from ._additive_reduce import mean_reduce, sum_reduce
    function = mean_reduce if mean else sum_reduce
    return function(a, dim, keepdim, dtype=dtype)


def absolute(a, *, out=None):
    dtype = {torch.complex32: torch.float16, torch.complex64: torch.float32, torch.complex128: torch.float64}.get(a.dtype, a.dtype)
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
    }
    for name, function in entries.items():
        library.impl(name, function)
    from ._trigonometric import unary as trigonometric
    for name, op in {'sin': 0, 'cos': 1, 'tan': 2, 'asin': 3, 'acos': 4, 'atan': 5,
                     'sinh': 6, 'cosh': 7, 'tanh': 8, 'asinh': 9, 'acosh': 10, 'atanh': 11}.items():
        library.impl(name, partial(trigonometric, op=op))
        library.impl(name + '.out', partial(trigonometric, op=op))
        library.impl(name + '_', partial(trigonometric, op=op, inplace=True))
    for name, op in {'erf': 216, 'erfc': 223, 'erfinv': 225}.items():
        library.impl(name, partial(unary_math, op=op))
        library.impl(name + '.out', partial(unary_math, op=op))
        library.impl(name + '_', partial(unary_math, op=op, inplace=True))
    for name, op in {'special_erfcx': 224, 'special_log_ndtr': 226, 'special_ndtri': 227}.items():
        library.impl(name, partial(unary_math, op=op))
        library.impl(name + '.out', partial(unary_math, op=op))
    for name, op in {'atan2': 221, 'hypot': 222}.items():
        library.impl(name, partial(binary_math, op=op))
        library.impl(name + '.out', partial(binary_math, op=op))
        library.impl(name + '_', partial(binary_math, op=op, inplace=True))
    for name, op in {'lgamma': 228, 'digamma': 229}.items():
        library.impl(name, partial(gamma_math, op=op))
        library.impl(name + '.out', partial(gamma_math, op=op))
        library.impl(name + '_', partial(gamma_math, op=op, inplace=True))
    library.impl('polygamma', polygamma)
    library.impl('polygamma.out', polygamma)
    library.impl('mvlgamma.out', mvlgamma_out)
    for overload in ('Tensor', 'Scalar_Other', 'Scalar_Self', 'OutTensor', 'OutScalar_Other', 'OutScalar_Self'):
        library.impl('xlogy.' + overload, partial(logarithmic_product, op=231))
    for overload in ('Tensor', 'Scalar_Other'):
        library.impl('xlogy_.' + overload, partial(logarithmic_product, op=231, inplace=True))
    for overload in ('', '.other_scalar', '.self_scalar', '.out', '.other_scalar_out', '.self_scalar_out'):
        library.impl('special_xlog1py' + overload, partial(logarithmic_product, op=232))
    for name, op in {'i0': 233, 'special_i0e': 234, 'special_i1': 235, 'special_i1e': 236,
                     'special_bessel_j0': 237, 'special_bessel_j1': 238,
                     'special_bessel_y0': 239, 'special_bessel_y1': 240,
                     'special_modified_bessel_i0': 241, 'special_modified_bessel_i1': 242,
                     'special_modified_bessel_k0': 243, 'special_modified_bessel_k1': 244,
                     'special_scaled_modified_bessel_k0': 245, 'special_scaled_modified_bessel_k1': 246,
                     'special_spherical_bessel_j0': 247}.items():
        library.impl(name, partial(bessel_math, op=op))
        library.impl(name + '.out', partial(bessel_math, op=op))
    library.impl('i0_', partial(bessel_math, op=233, inplace=True))
    for name, op in {'exp2': 249, 'expm1': 250, 'log2': 251, 'log10': 252}.items():
        library.impl(name, partial(elementary_math, op=op))
        library.impl(name + '.out', partial(elementary_math, op=op))
        library.impl(name + '_', partial(elementary_math, op=op, inplace=True))
    for name, op in {'logaddexp': 253, 'logaddexp2': 254, 'special_entr': 257}.items():
        library.impl(name, partial(elementary_math, op=op))
        library.impl(name + '.out', partial(elementary_math, op=op))
    library.impl('logit', logit)
    library.impl('logit.out', logit)
    library.impl('logit_', partial(logit, inplace=True))
    library.impl('logit_backward', logit_backward)
    library.impl('logit_backward.grad_input', logit_backward)
    from ._floating import register as register_floating
    register_floating(library)
    from ._activation import register as register_activation
    register_activation(library)
    from ._loss import register as register_loss
    register_loss(library)
    from ._normalization import register as register_normalization
    register_normalization(library)
    from ._pooling import register as register_pooling
    register_pooling(library)
    from ._padding import register as register_padding
    register_padding(library)
    from ._resize import register as register_resize
    register_resize(library)
    from ._grid import register as register_grid
    register_grid(library)
    from ._rearrange import register as register_rearrange
    register_rearrange(library)
    from ._distance import register as register_distance
    register_distance(library)
    from ._embedding_bag import register as register_embedding_bag
    register_embedding_bag(library)
    from ._indexed_native import register as register_indexed_native
    register_indexed_native(library)
    from ._histogram import register as register_histogram
    register_histogram(library)
    from ._selection import register as register_selection
    register_selection(library)
    from ._ctc import register as register_ctc
    register_ctc(library)
    from ._margin import register as register_margin
    register_margin(library)
    from ._scan_segment import register as register_scan_segment
    register_scan_segment(library)
    from ._polynomial import register as register_polynomial
    register_polynomial(library)
    from ._gamma_extra import register as register_gamma_extra
    register_gamma_extra(library)
    from ._math_extra import register as register_math_extra
    register_math_extra(library)
    from ._complex_classify import register as register_complex_classify
    register_complex_classify(library)
    from ._bitwise import register as register_bitwise
    register_bitwise(library)
    from ._ternary_ops import register as register_ternary
    register_ternary(library)
    from ._membership import register as register_membership
    register_membership(library)
    from ._cumulative import register as register_cumulative
    register_cumulative(library)
    from ._activation_extra import register as register_activation_extra
    register_activation_extra(library)
    from ._triangular_indices import register as register_triangular_indices
    register_triangular_indices(library)
    from ._amp import register as register_amp
    register_amp(library)
    from ._fused_optim import register as register_fused_optim
    register_fused_optim(library)
    from ._rnn_cell import register as register_rnn_cell
    register_rnn_cell(library)
    from ._weight_norm import register as register_weight_norm
    register_weight_norm(library)
    from ._foreach_unary import register as register_foreach_unary
    register_foreach_unary(library)
    from ._linear_combination import register as register_linear_combination
    register_linear_combination(library)
    from ._blas_accumulate import register as register_blas_accumulate
    register_blas_accumulate(library)
    from ._matmul_extra import register as register_matmul_extra
    register_matmul_extra(library)
    from ._grouped_mm import register as register_grouped_mm
    register_grouped_mm(library)
    from ._sparse_coo import register as register_sparse_coo
    register_sparse_coo(library)
    from ._fake_quant_affine import register as register_fake_quant_affine
    register_fake_quant_affine(library)
    _libraries.append(library)
