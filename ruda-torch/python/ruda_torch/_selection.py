import math

import torch

from . import _C
from ._indexing import _logical
from ._ops import _dimension, _dtypes, _integral_dtypes
from ._ordering import _output, _outputs
from ._pooling import _nondeterministic


def _layout(a):
    if a.layout != torch.strided:
        raise RuntimeError('selection requires strided input')


def _real(a, *, boolean):
    if a.dtype not in _dtypes + _integral_dtypes or (a.dtype == torch.bool and not boolean):
        raise RuntimeError('selection requires a supported real or integral dtype' + ('' if boolean else ' other than bool'))


def _rows(a, dim, keepdim):
    _layout(a)
    axis = _dimension(dim, a.ndim)
    length = a.shape[axis] if a.ndim else 1
    if length == 0:
        raise IndexError('selection reduction dimension must have nonzero size')
    shape = tuple(1 if i == axis else n for i, n in enumerate(a.shape)) if keepdim else tuple(n for i, n in enumerate(a.shape) if i != axis)
    return axis, length, shape


def _limits(a, length):
    if a.numel() > 2**32 - 1 or length > 2**32 - 1 or a.ndim > 32:
        raise RuntimeError('selection exceeds native U32 indexing or rank-32 limits')


def _empty(a, shape):
    return (torch.empty(shape, dtype=a.dtype, device=a.device),
            torch.empty(shape, dtype=torch.int64, device=a.device))


def _source(a, axis, length):
    value = _logical(a)
    if a.ndim:
        value = value.movedim(axis, -1)
    return value.reshape(a.numel() // length, length)


def _select(source, shape, kind, k=1):
    values, indices = _empty(source, (source.shape[0],))
    _C.scientific_call(354, [source, values, indices], [kind, k], [])
    return values.reshape(shape), indices.reshape(shape)


def _write(results, a, values, indices, *, disallow_alias=False):
    outputs = None
    if values is not None or indices is not None:
        if values is None or indices is None:
            raise RuntimeError('selection requires both values and indices outputs')
        if disallow_alias:
            _C.check_index_output(values, a, a)
        outputs = values, indices
    return _outputs(results, outputs, (a,))


def kthvalue(a, k, dim=-1, keepdim=False, *, values=None, indices=None):
    axis, length, shape = _rows(a, dim, keepdim)
    if k < 1 or k > length:
        raise RuntimeError('kthvalue selected number k is out of range')
    if length > 2**31 - 1:
        raise RuntimeError('kthvalue dimension length exceeds int32')
    _limits(a, length)
    if not a.numel():
        result = _empty(a, shape)
    elif a.ndim == 0:
        result = _logical(a).clone(), torch.zeros((), dtype=torch.int64, device=a.device)
    else:
        _real(a, boolean=False)
        result = _select(_source(a, axis, length), shape, 0, k)
    return _write(result, a, values, indices, disallow_alias=True)


def _median_dim(a, dim, keepdim, ignore_nan, values, indices):
    _nondeterministic('median CUDA with indices output')
    axis, length, shape = _rows(a, dim, keepdim)
    _limits(a, length)
    if not a.numel():
        result = _empty(a, shape)
    else:
        _real(a, boolean=False)
        result = _select(_source(a, axis, length), shape, 2 if ignore_nan else 1)
    return _write(result, a, values, indices)


def median_dim(a, dim, keepdim=False, *, values=None, indices=None):
    return _median_dim(a, dim, keepdim, False, values, indices)


def nanmedian_dim(a, dim, keepdim=False, *, values=None, indices=None):
    return _median_dim(a, dim, keepdim, True, values, indices)


def _median_all(a, ignore_nan, out):
    _layout(a)
    if not a.numel():
        result = torch.full((), math.nan, dtype=torch.float32, device=a.device).to(a.dtype)
    else:
        _real(a, boolean=True)
        _limits(a, a.numel())
        result = _select(_logical(a).reshape(1, a.numel()), (), 2 if ignore_nan else 1)[0]
    return _output(result, out, (a,))


def median(a, *, out=None):
    return _median_all(a, False, out)


def nanmedian(a, *, out=None):
    return _median_all(a, True, out)


def mode(a, dim=-1, keepdim=False, *, values=None, indices=None):
    axis, length, shape = _rows(a, dim, keepdim)
    _limits(a, length)
    if not a.numel():
        result = _empty(a, shape)
    elif length == 1:
        result = (_logical(a).clone().reshape(shape),
                  torch.zeros(shape, dtype=torch.int64, device=a.device))
    else:
        _real(a, boolean=True)
        source, order = torch.sort(_source(a, axis, length), dim=1, stable=True)
        selected, positions = _empty(a, (source.shape[0],))
        last = length <= 2048 and a.numel() < 2**31 and sum((n - 1) * s for n, s in zip(a.shape, a.stride())) < 2**31
        _C.scientific_call(355, [source, order, selected, positions], [int(last)], [])
        result = selected.reshape(shape), positions.reshape(shape)
    return _write(result, a, values, indices)


def register(library):
    for name, function in (('median', median), ('median.out', median),
                           ('median.dim', median_dim), ('median.dim_values', median_dim),
                           ('nanmedian', nanmedian), ('nanmedian.out', nanmedian),
                           ('nanmedian.dim', nanmedian_dim), ('nanmedian.dim_values', nanmedian_dim),
                           ('kthvalue', kthvalue), ('kthvalue.values', kthvalue),
                           ('mode', mode), ('mode.values', mode)):
        library.impl(name, function)
