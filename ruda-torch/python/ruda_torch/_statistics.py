import math
import sys
import warnings

import torch

from . import _C
from ._ops import _registry, _numeric_dtypes, _integral_dtypes, _dimension
from ._indexing import _logical
from ._ordering import _output


def _axes(a, dim):
    dims = tuple(range(a.ndim)) if dim is None else (dim,) if isinstance(dim, int) else tuple(dim)
    if not dims:
        dims = tuple(range(a.ndim))
    axes = tuple(_dimension(axis, a.ndim) for axis in dims)
    if len(set(axes)) != len(axes):
        raise RuntimeError('duplicate reduction dimension')
    return axes if a.ndim else ()


def _shape(a, axes, keepdim):
    return tuple(1 if d in axes else n for d, n in enumerate(a.shape)) if keepdim else tuple(
        n for d, n in enumerate(a.shape) if d not in axes)


def _opmath(dtype):
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def _real_dtype(dtype):
    return {torch.complex32: torch.float16, torch.complex64: torch.float32,
            torch.complex128: torch.float64}.get(dtype, dtype)


def _moments(value):
    rows, length = value.shape
    dtype = _opmath(value.dtype)
    if (getattr(sys.modules.get(__package__), '_statistics_available', False)
            and value.numel() <= 2**32 - 1 and length <= 2**32 - 1):
        mean = torch.empty((rows,), dtype=dtype, device=value.device)
        m2 = torch.empty_like(mean)
        _C.statistics_call(value.contiguous(), mean, m2)
        return mean, m2
    mean = value.to(dtype)
    m2 = mean - mean
    counts = torch.ones((length,), dtype=torch.int64, device=value.device)
    while length > 1:
        pairs = length // 2
        left, right = mean[:, :2*pairs:2], mean[:, 1:2*pairs:2]
        na, nb = counts[:2*pairs:2], counts[1:2*pairs:2]
        total = na + nb
        weight = nb.to(dtype) / total.to(dtype)
        delta = right - left
        merged_mean = left + delta * weight
        merged_m2 = m2[:, :2*pairs:2] + m2[:, 1:2*pairs:2] + delta * delta * na.to(dtype) * weight
        mean = torch.cat((merged_mean, mean[:, -1:]), 1) if length % 2 else merged_mean
        m2 = torch.cat((merged_m2, m2[:, -1:]), 1) if length % 2 else merged_m2
        counts = torch.cat((total, counts[-1:])) if length % 2 else total
        length = mean.shape[1]
    return mean.reshape(rows), m2.reshape(rows)


def variance(a, dim=None, unbiased=True, keepdim=False, *, correction=None, std=False,
             with_mean=False, out=None, out0=None, out1=None):
    supported = _numeric_dtypes + (torch.complex32,)
    if a.dtype not in supported:
        raise RuntimeError('var/std require floating point or complex input')
    axes = _axes(a, dim)
    shape = _shape(a, axes, keepdim)
    length = math.prod(a.shape[d] for d in axes)
    rows = math.prod(a.shape[d] for d in range(a.ndim) if d not in axes)
    correction = (1 if unbiased else 0) if correction is None else correction
    if isinstance(correction, complex):
        if correction.imag != 0:
            raise RuntimeError('correction cannot be converted to a real scalar')
        correction = correction.real
    if not isinstance(correction, (int, float)):
        raise TypeError('correction must be a real scalar')
    correction = float(correction)
    if (length if rows else 0) - correction <= 0:
        warnings.warn('var/std: degrees of freedom is <= 0. Correction should be strictly less than the reduction factor.',
                      UserWarning, stacklevel=2)
    if with_mean:
        variance_dtype = out0.dtype if out0 is not None else _real_dtype(a.dtype)
        mean_dtype = out1.dtype if out1 is not None else a.dtype
        if (variance_dtype not in supported or mean_dtype not in supported
                or variance_dtype != _real_dtype(mean_dtype) or mean_dtype.is_complex != a.is_complex()):
            raise RuntimeError('variance output must be real and match mean precision; mean kind must match input')
        if out0 is not None and out1 is not None:
            _C.check_index_output(out0, out1, out1)
    else:
        variance_dtype = out.dtype if out is not None else _real_dtype(a.dtype)
        if variance_dtype not in supported or (a.numel() and not a.is_complex() and variance_dtype.is_complex):
            raise RuntimeError('var/std output dtype is not supported for this input')
    remaining = tuple(d for d in range(a.ndim) if d not in axes)
    value = _logical(a).permute((*remaining, *axes))
    result = torch.empty(shape, dtype=variance_dtype, device=a.device)
    variance_parts = _components(result.reshape(-1))
    mean = torch.empty(shape, dtype=mean_dtype, device=a.device) if with_mean else None
    mean_parts = _components(mean.reshape(-1)) if with_mean else variance_parts
    _C.scientific_call(460, [*_components(value), *variance_parts, *mean_parts],
                       [len(remaining), int(value.is_complex()), int(std), int(with_mean), int(result.is_complex())], [correction])
    if with_mean:
        return _output(result, out0, (a,)), _output(mean, out1, (a,))
    return _output(result, out, (a,))


def _components(value):
    if value.is_complex():
        parts = torch.view_as_real(value)
        return parts[..., 0], parts[..., 1]
    return value, value


def _nan_reduce(a, axes, keepdim, dtype, *, mean=False):
    remaining = tuple(d for d in range(a.ndim) if d not in axes)
    original = _logical(a).permute((*remaining, *axes))
    value = original.to(dtype)
    result = torch.empty(_shape(a, axes, keepdim), dtype=dtype, device=a.device)
    original_parts = _components(original) if mean else _components(value)
    tensors = [*_components(value), *original_parts, *_components(result.reshape(-1))]
    _C.scientific_call(458 if mean else 457, tensors,
                       [len(remaining), int(value.is_complex()), int(original.is_complex() if mean else value.is_complex())], [])
    return result


def nansum(a, dim=None, keepdim=False, *, dtype=None, out=None):
    axes = _axes(a, dim)
    dtype = dtype if dtype is not None else out.dtype if out is not None else (
        torch.int64 if a.dtype in _integral_dtypes else a.dtype)
    value = _logical(a)
    if a.dtype in _integral_dtypes:
        result = value.sum(axes, keepdim=keepdim, dtype=dtype)
    elif dtype in _integral_dtypes:
        result = torch.nan_to_num(value, nan=0.0).sum(axes, keepdim=keepdim, dtype=dtype)
    else:
        supported = _numeric_dtypes + (torch.complex32,)
        if a.dtype not in supported or dtype not in supported:
            raise RuntimeError('nansum requires a supported floating, complex or integral dtype')
        result = _nan_reduce(value, axes, keepdim, dtype)
    return _output(result, out, (a,))


def nanmean(a, dim=None, keepdim=False, *, dtype=None, out=None):
    supported = _numeric_dtypes + (torch.complex32,)
    if a.dtype not in supported:
        raise RuntimeError('nanmean requires floating point or complex input')
    dtype = dtype if dtype is not None else out.dtype if out is not None else a.dtype
    if dtype not in supported:
        raise RuntimeError('nanmean requires a floating point or complex result dtype')
    axes = _axes(a, dim)
    if out is not None and torch.is_grad_enabled() and a.requires_grad:
        raise RuntimeError('nanmean with out= does not support automatic differentiation')
    result = _nan_reduce(a, axes, keepdim, dtype, mean=True)
    return _output(result, out, (a,))


def _nanmean_autograd(a, dim=None, keepdim=False, *, dtype=None, out=None):
    differentiated = torch.is_grad_enabled() and a.requires_grad
    differentiated = differentiated or torch.autograd.forward_ad.unpack_dual(a).tangent is not None
    if not differentiated:
        return nanmean(a, dim, keepdim, dtype=dtype, out=out)
    if out is not None:
        raise RuntimeError('nanmean with out= does not support automatic differentiation')
    supported = _numeric_dtypes + (torch.complex32,)
    dtype = a.dtype if dtype is None else dtype
    if a.dtype not in supported or dtype not in supported:
        raise RuntimeError('nanmean requires floating point or complex input and result dtype')
    axes = _axes(a, dim)
    value = _logical(a)
    count = (value.detach() == value.detach()).sum(axes, keepdim=keepdim)
    total = torch.nansum(value, dim=axes, keepdim=keepdim, dtype=dtype)
    return total / count


def logsumexp(a, dim, keepdim=False, *, out=None):
    if not isinstance(dim, int) and not dim and a.ndim and a.numel() and not keepdim:
        raise RuntimeError('logsumexp with empty dimensions cannot broadcast the scalar output')
    axes = _axes(a, dim)
    supported = _numeric_dtypes + (torch.complex32,)
    value = _logical(a)
    if a.dtype in _integral_dtypes:
        value = value.to(torch.get_default_dtype())
    if value.dtype not in supported:
        raise RuntimeError('logsumexp requires a supported floating, complex or integral input dtype')
    dtype = value.dtype if out is None else out.dtype
    if dtype not in supported:
        raise RuntimeError('logsumexp requires a floating point or complex output dtype')
    remaining = tuple(d for d in range(a.ndim) if d not in axes)
    value = value.permute((*remaining, *axes))
    result = torch.empty(_shape(a, axes, keepdim), dtype=dtype, device=a.device)
    parts = _components(value)
    _C.scientific_call(459, [*parts, *parts, *_components(result.reshape(-1))],
                       [len(remaining), int(value.is_complex()), int(result.is_complex())], [])
    return _output(result, out, (a,))


def _native_norm(a, order, axes, keepdim, dtype, *, power_sum=False):
    remaining = tuple(d for d in range(a.ndim) if d not in axes)
    value = _logical(a).permute((*remaining, *axes))
    output_dtype = {torch.complex32: torch.float16, torch.complex64: torch.float32,
                    torch.complex128: torch.float64}.get(dtype, dtype)
    result = torch.empty(_shape(a, axes, keepdim), dtype=output_dtype, device=a.device)
    if value.is_complex():
        parts = torch.view_as_real(value)
        real, imag = parts[..., 0], parts[..., 1]
    else:
        real = imag = value
    _C.scientific_call(456 if power_sum else 455, [real, imag, result.reshape(-1)],
                       [len(remaining), int(value.is_complex())], [order])
    return result


def vector_norm(a, ord=2, dim=None, keepdim=False, *, dtype=None, out=None):
    norm_dtypes = _numeric_dtypes + (torch.complex32,)
    if a.dtype not in norm_dtypes:
        raise RuntimeError('linalg.vector_norm requires floating point or complex input')
    if isinstance(ord, complex):
        raise RuntimeError('linalg.vector_norm expects a non-complex order')
    dtype = a.dtype if dtype is None else dtype
    if dtype not in norm_dtypes or dtype.is_complex != a.dtype.is_complex:
        raise RuntimeError('norm dtype must be floating/complex and match the input kind')
    if torch.promote_types(a.dtype, dtype) != dtype:
        raise RuntimeError('norm dtype must not narrow input precision')
    axes = _axes(a, dim)
    length = math.prod(a.shape[d] for d in axes)
    order = float(ord)
    if length == 0 and (order < 0 or order == math.inf):
        raise RuntimeError('norm cannot reduce an empty dimension with this order')
    if a.numel() and length != 1 and dtype == torch.complex32:
        raise RuntimeError('norm reduction is not implemented for complex32; specify complex64 or complex128 dtype')
    result = _native_norm(a, order, axes, keepdim, dtype)
    return _output(result, out, (a,))


def powsum(a, ord=2, dim=None, keepdim=False, *, dtype=None):
    axes = _axes(a, dim)
    shape = _shape(a, axes, keepdim)
    compute_dtype = a.dtype if dtype is None else dtype
    output_dtype = {torch.complex32: torch.float16, torch.complex64: torch.float32,
                    torch.complex128: torch.float64}.get(compute_dtype, compute_dtype)
    supported = a.dtype in _numeric_dtypes + (torch.complex32,) and output_dtype in (
        torch.float16, torch.bfloat16, torch.float32, torch.float64)
    length = math.prod(a.shape[d] for d in axes)
    complex_input = compute_dtype.is_complex if length == 1 and dtype is not None else a.is_complex()
    if not supported or complex_input and output_dtype in (torch.float16, torch.bfloat16):
        dimensions = None if dim is None else [dim] if isinstance(dim, int) else list(dim)
        return _C.powsum_slow(a, ord, dimensions, keepdim, dtype)
    if isinstance(ord, complex) and ord.imag != 0:
        raise RuntimeError('powsum order cannot be converted to a real scalar')
    order = float(ord.real if isinstance(ord, complex) else ord)
    if math.prod(shape) == 0:
        return torch.empty(shape, dtype=output_dtype, device=a.device)
    value = a
    if length == 1 and dtype is not None and dtype.is_complex != a.is_complex():
        value = a.to(dtype)
    return _native_norm(value, order, axes, keepdim, compute_dtype, power_sum=True)


def norm(a, p=2, dim=None, keepdim=False, *, dtype=None, out=None):
    return vector_norm(a, 2 if p is None else p, dim, keepdim, dtype=dtype, out=out)


def _variance_overloads(name, *, std, with_mean):
    def corrected(a, dim=None, *, correction=None, keepdim=False, **outputs):
        return variance(a, dim, keepdim=keepdim, correction=correction, std=std, with_mean=with_mean, **outputs)

    _registry.impl(name + '.correction', corrected)
    _registry.impl(name + '.correction_out', corrected)


for _name in ('var', 'std', 'var_mean', 'std_mean'):
    _variance_overloads(_name, std=_name.startswith('std'), with_mean=_name.endswith('_mean'))

for _name, _function in (('nansum', nansum), ('nanmean', nanmean), ('logsumexp', logsumexp),
                         ('linalg_vector_norm', vector_norm)):
    _registry.impl(_name, _function)
    _registry.impl(_name + '.out', _function)

for _overload in ('Scalar', 'ScalarOpt_dim', 'ScalarOpt_dtype', 'ScalarOpt_dtype_out',
                  'Scalar_out', 'ScalarOpt_dim_dtype', 'dtype_out', 'out'):
    _registry.impl('norm.' + _overload, norm)

if hasattr(torch.ops.aten, 'linalg__powsum'):
    _registry.impl('linalg__powsum', powsum)

_registry.impl('isnan', lambda a: a != a)
_registry.impl('isnan.out', lambda a, *, out: _output(a != a, out, (a,)))
_autograd = torch.library.Library('aten', 'IMPL', 'AutogradPrivateUse1')
_autograd.impl('nanmean', _nanmean_autograd)
_autograd.impl('nanmean.out', _nanmean_autograd)
