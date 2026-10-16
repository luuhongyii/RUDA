from functools import partial

import torch
from . import _C
from ._ops import _numeric_dtypes, _integral_dtypes
from ._ordering import _output
from ._pooling import _nondeterministic


def _shape(a, padding, rank, reflect):
    pad = tuple(padding)
    if len(pad) != 2 * rank:
        raise RuntimeError(f'padding requires {2 * rank} padding values')
    if a.ndim not in (rank + 1, rank + 2):
        raise RuntimeError('padding requires the expected batched or unbatched input rank')
    nonbatch = a.shape[1:] if a.ndim == rank + 2 else a.shape
    if any(n <= 0 for n in nonbatch):
        raise RuntimeError('padding requires nonempty non-batch input dimensions')
    sizes = list(a.shape)
    for axis in range(rank):
        n = a.shape[-axis - 1]
        left, right = pad[2 * axis:2 * axis + 2]
        if reflect and (left >= n or right >= n):
            raise RuntimeError('reflection padding must be smaller than the input dimension')
        sizes[-axis - 1] = n + left + right
    spatial = sizes[-rank:]
    if any(n < 0 for n in spatial) or not any(n > 0 for n in spatial):
        raise RuntimeError('padding results in an invalid output size')
    return sizes, (*pad, *((0,) * (6 - len(pad))), rank, int(reflect))


def _five(a, rank):
    if a.ndim == rank + 1:
        a = a.unsqueeze(0)
    for _ in range(3 - rank):
        a = a.unsqueeze(2)
    return a


def padding_forward(a, padding, *, rank, reflect, out=None):
    if a.dtype == torch.bool or a.dtype not in _numeric_dtypes + _integral_dtypes:
        raise RuntimeError('padding requires floating, complex or integer input')
    shape, params = _shape(a, padding, rank, reflect)
    result = a.new_empty(shape)
    if result.numel():
        _C.scientific_call(323, [_five(a.resolve_conj().resolve_neg(), rank), _five(result, rank)],
                           (*params, 0), [])
    return _output(result, out, (a,))


def padding_backward(grad_output, a, padding, *, rank, reflect, grad_input=None):
    if a.dtype not in _numeric_dtypes:
        raise RuntimeError('padding backward requires floating or complex input')
    shape, params = _shape(a, padding, rank, reflect)
    if grad_output.dtype != a.dtype or grad_output.device != a.device or tuple(grad_output.shape) != tuple(shape):
        raise RuntimeError('padding gradient must match forward output shape, dtype and device')
    deterministic = reflect and rank == 2 and torch.are_deterministic_algorithms_enabled()
    if not reflect or rank != 2:
        _nondeterministic(('reflection' if reflect else 'replication') + f'_pad{rank}d_backward')
    result = a.new_empty(a.shape)
    if result.numel():
        _C.scientific_call(324, [_five(grad_output.resolve_conj().resolve_neg(), rank), _five(result, rank)],
                           (*params, int(deterministic)), [])
    return _output(result, grad_input, (a, grad_output))


def register(library):
    for rank in (1, 2, 3):
        for name, reflect in (('reflection', True), ('replication', False)):
            forward = partial(padding_forward, rank=rank, reflect=reflect)
            backward = partial(padding_backward, rank=rank, reflect=reflect)
            stem = f'{name}_pad{rank}d'
            library.impl(stem, forward)
            library.impl(stem + '.out', forward)
            library.impl(stem + '_backward', backward)
            library.impl(stem + '_backward.grad_input', backward)
