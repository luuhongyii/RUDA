from functools import partial

import torch
from . import _C
from ._ops import _dtypes
from ._ordering import _output
from ._pooling import _nondeterministic


def _shape(input, grid, interpolation_mode, padding_mode, rank):
    if input.dtype not in _dtypes or grid.dtype != input.dtype:
        raise RuntimeError('grid sampling requires matching real floating input and grid dtypes')
    if input.device != grid.device or input.layout != torch.strided or grid.layout != torch.strided:
        raise RuntimeError('grid sampling requires strided input and grid on the same device')
    if input.ndim != rank + 2 or grid.ndim != rank + 2:
        raise RuntimeError('grid sampling input and grid have an invalid rank')
    if input.shape[0] != grid.shape[0] or grid.shape[-1] != rank:
        raise RuntimeError('grid sampling requires matching batches and one coordinate per spatial dimension')
    if any(n <= 0 for n in input.shape[2:]):
        raise RuntimeError('grid sampling requires nonempty input spatial dimensions')
    if interpolation_mode not in (0, 1, 2) or rank == 3 and interpolation_mode == 2:
        raise RuntimeError('grid sampling interpolation must be bilinear/trilinear, nearest, or 2D bicubic')
    if padding_mode not in (0, 1, 2):
        raise RuntimeError('grid sampling padding must be zeros, border, or reflection')
    return (*input.shape[:2], *grid.shape[1:-1])


def _input_view(value, rank):
    value = value.resolve_neg()
    return value.unsqueeze(2) if rank == 2 else value


def _grid_view(value, rank):
    value = value.resolve_neg()
    return value.unsqueeze(1) if rank == 2 else value


def grid_sampler(input, grid, interpolation_mode, padding_mode, align_corners, *, rank, out=None):
    shape = _shape(input, grid, interpolation_mode, padding_mode, rank)
    result = torch.empty(shape, dtype=input.dtype, device=input.device)
    if result.numel():
        _C.scientific_call(331, [_input_view(input, rank), _grid_view(grid, rank), _input_view(result, rank)],
                           [rank, interpolation_mode, padding_mode, int(align_corners)], [])
    return _output(result, out, (input, grid))


def grid_sampler_backward(grad_output, input, grid, interpolation_mode, padding_mode, align_corners,
                          output_mask, *, rank, out0=None, out1=None):
    shape = _shape(input, grid, interpolation_mode, padding_mode, rank)
    if (tuple(grad_output.shape) != shape or grad_output.dtype != input.dtype
            or grad_output.device != input.device or grad_output.layout != torch.strided):
        raise RuntimeError('grid sampling gradient must match the forward output shape, dtype, and device')
    if len(output_mask) != 2:
        raise RuntimeError('grid sampling backward requires two output mask entries')
    _nondeterministic(f'grid_sampler_{rank}d_backward')
    dx = torch.empty(input.shape, dtype=input.dtype, device=input.device) if output_mask[0] else None
    dgrid = torch.empty(grid.shape, dtype=grid.dtype, device=grid.device)
    dx_view = input.new_empty((0,)) if dx is None else _input_view(dx, rank)
    _C.scientific_call(332, [_input_view(input, rank), _grid_view(grid, rank),
                           _input_view(grad_output, rank), dx_view, _grid_view(dgrid, rank)],
                       [rank, interpolation_mode, padding_mode, int(align_corners), int(output_mask[0])], [])
    if out0 is None and out1 is None:
        return dx, dgrid
    if out0 is None or out1 is None:
        raise RuntimeError('grid sampling backward.out requires both output tensors')
    _C.check_index_output(out0, out1, out1)
    inputs = grad_output, input, grid
    return (out0 if dx is None else _output(dx, out0, inputs), _output(dgrid, out1, inputs))


def register(library):
    for rank in (2, 3):
        name = f'grid_sampler_{rank}d'
        forward = partial(grid_sampler, rank=rank)
        backward = partial(grid_sampler_backward, rank=rank)
        library.impl(name, forward)
        library.impl(name + '.out', forward)
        library.impl(name + '_backward', backward)
        library.impl(name + '_backward.out', backward)
