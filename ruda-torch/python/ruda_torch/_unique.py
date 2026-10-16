import math

import torch
from . import _C
from ._ops import _registry, _dimension, _integral_dtypes, _dtypes
from ._indexing import _logical, _index_out


def _unique(a, *, dim=None, consecutive=False, return_inverse=False, return_counts=False):
    if a.dtype not in _dtypes + _integral_dtypes:
        raise RuntimeError('unique requires a supported real or integral dtype')
    if dim is None:
        value = _logical(a)
        length = value.numel()
        width = 1
    else:
        if a.ndim == 0:
            raise IndexError('unique dimension specified for a scalar tensor')
        dim = _dimension(dim, a.ndim)
        if any(size == 0 for axis, size in enumerate(a.shape) if axis != dim):
            raise RuntimeError('unique cannot operate with zero-sized non-selected dimensions')
        value = _logical(a).movedim(dim, 0)
        length = value.shape[0]
        width = math.prod(value.shape[1:])
    if not length:
        result = torch.empty(a.shape if dim is not None else (0,), dtype=a.dtype, device=a.device)
        inverse = torch.empty(a.shape if dim is None and return_inverse else (0,), dtype=torch.int64, device=a.device)
        counts = torch.empty((0,), dtype=torch.int64, device=a.device)
        return result, inverse, counts
    order = torch.empty((0 if consecutive else length,), dtype=torch.int32, device=a.device)
    groups = torch.empty((length,), dtype=torch.int32, device=a.device)
    starts = torch.empty_like(groups)
    total = torch.empty((1,), dtype=torch.int64, device=a.device)
    dimensions = [length, width, int(consecutive)]
    _C.scientific_call(461, [value, order, groups, starts, total], [*dimensions, int(dim is None), 0], [])
    size = int(total.item())
    shape = (size,) if dim is None else (size, *value.shape[1:])
    result = torch.empty(shape, dtype=a.dtype, device=a.device)
    inverse_shape = a.shape if dim is None else (length,)
    inverse = torch.empty(inverse_shape if return_inverse else (0,), dtype=torch.int64, device=a.device)
    counts = torch.empty((size if return_counts else 0,), dtype=torch.int64, device=a.device)
    _C.scientific_call(462, [value, order, groups, starts, result.reshape(-1), inverse.reshape(-1), counts],
                       [*dimensions, int(return_inverse), int(return_counts)], [])
    return result.movedim(0, dim) if dim is not None else result, inverse, counts


def unique(a, sorted=True, return_inverse=False, return_counts=False):
    return _unique(a, return_inverse=return_inverse, return_counts=return_counts)


def unique_dim(a, dim, sorted=True, return_inverse=False, return_counts=False):
    return _unique(a, dim=dim, return_inverse=return_inverse, return_counts=return_counts)


def unique_consecutive(a, return_inverse=False, return_counts=False, dim=None):
    return _unique(a, dim=dim, consecutive=True, return_inverse=return_inverse, return_counts=return_counts)


def _outputs(results, outputs, a):
    for position, out in enumerate(outputs):
        for other in outputs[position + 1:]:
            _C.check_index_output(out, other, other)
    return tuple(_index_out(result, out, (a,)) for result, out in zip(results, outputs))


for name, function in {
    '_unique': lambda a, sorted=True, return_inverse=False: unique(a, sorted, return_inverse)[:2],
    '_unique2': unique,
    'unique_dim': unique_dim,
    'unique_consecutive': unique_consecutive,
    'unique_dim_consecutive': lambda a, dim, return_inverse=False, return_counts=False: unique_consecutive(a, return_inverse, return_counts, dim),
    '_unique.out': lambda a, sorted=True, return_inverse=False, *, out0, out1: _outputs(unique(a, sorted, return_inverse)[:2], (out0, out1), a),
    '_unique2.out': lambda a, sorted=True, return_inverse=False, return_counts=False, *, out0, out1, out2: _outputs(unique(a, sorted, return_inverse, return_counts), (out0, out1, out2), a),
    'unique_dim.out': lambda a, dim, sorted=True, return_inverse=False, return_counts=False, *, out0, out1, out2: _outputs(unique_dim(a, dim, sorted, return_inverse, return_counts), (out0, out1, out2), a),
    'unique_consecutive.out': lambda a, return_inverse=False, return_counts=False, dim=None, *, out0, out1, out2: _outputs(unique_consecutive(a, return_inverse, return_counts, dim), (out0, out1, out2), a),
    'unique_dim_consecutive.out': lambda a, dim, return_inverse=False, return_counts=False, *, out0, out1, out2: _outputs(unique_consecutive(a, return_inverse, return_counts, dim), (out0, out1, out2), a),
}.items():
    _registry.impl(name, function)
