from functools import partial

import torch
from torch._prims_common import suggest_memory_format
from . import _C
from ._ops import _dtypes
from ._ordering import _output
from ._pooling import _nondeterministic


def _shape(input_size, output_size, rank, *, backward=False):
    source, size = tuple(input_size), tuple(output_size)
    if len(source) != rank + 2 or len(size) != rank:
        raise RuntimeError('resize requires the expected input rank and spatial output size')
    if any(n <= 0 for n in (*source[2:], *size)) or any(n < 0 for n in source[:2]):
        raise RuntimeError('resize requires positive spatial sizes and nonnegative batch/channel sizes')
    if not backward and source[1] == 0:
        raise RuntimeError('resize requires nonempty channels')
    return (*source[:2], *size)


def _five(a, rank):
    for _ in range(3 - rank):
        a = a.unsqueeze(2)
    return a


def _scales(scales, rank):
    return (*((0.0,) * (3 - rank)), *(0.0 if s is None else s for s in scales))


def _forward(a, output_size, scales, rank, mode, align=False, out=None):
    if a.dtype not in _dtypes and not (mode < 2 and a.dtype == torch.uint8):
        raise RuntimeError('resize requires real floating input, or uint8 for nearest modes')
    shape = _shape(a.shape, output_size, rank)
    memory = suggest_memory_format(a) if rank > 1 else torch.contiguous_format
    result = torch.empty(shape, dtype=a.dtype, device=a.device, memory_format=memory)
    if result.numel():
        _C.scientific_call(329 if mode >= 4 else 325 if mode < 2 else 327,
                           [_five(a.resolve_neg(), rank), _five(result, rank)],
                           [rank, mode, int(align)], _scales(scales, rank))
    return _output(result, out, (a,))


def _backward(grad_output, output_size, input_size, scales, rank, mode, align=False, grad_input=None):
    if grad_output.dtype not in _dtypes and not (mode < 2 and grad_output.dtype == torch.uint8):
        raise RuntimeError('resize backward requires real floating input, or uint8 for nearest modes')
    shape = _shape(input_size, output_size, rank, backward=True)
    if tuple(grad_output.shape) != shape:
        raise RuntimeError('resize gradient must match the forward output shape')
    if mode >= 2:
        name = {3: 'bicubic2d', 4: 'bilinear2d_aa', 5: 'bicubic2d_aa'}.get(mode)
        if name is None:
            name = ('linear1d', 'bilinear2d', 'trilinear3d')[rank - 1]
        _nondeterministic('upsample_' + name + '_backward')
    memory = torch.contiguous_format
    if rank == 2 and mode not in (3, 5) or rank == 3 and mode == 2:
        memory = suggest_memory_format(grad_output)
    result = torch.empty(input_size, dtype=grad_output.dtype, device=grad_output.device, memory_format=memory)
    if result.numel():
        _C.scientific_call(330 if mode >= 4 else 326 if mode < 2 else 328,
                           [_five(grad_output.resolve_neg(), rank), _five(result, rank)],
                           [rank, mode, int(align)], _scales(scales, rank))
    return _output(result, grad_input, (grad_output,))


def nearest1d(a, output_size, scales=None, *, mode, out=None):
    return _forward(a, output_size, (scales,), 1, mode, out=out)


def nearest2d(a, output_size, scales_h=None, scales_w=None, *, mode, out=None):
    return _forward(a, output_size, (scales_h, scales_w), 2, mode, out=out)


def nearest3d(a, output_size, scales_d=None, scales_h=None, scales_w=None, *, mode, out=None):
    return _forward(a, output_size, (scales_d, scales_h, scales_w), 3, mode, out=out)


def nearest1d_backward(grad_output, output_size, input_size, scales=None, *, mode, grad_input=None):
    return _backward(grad_output, output_size, input_size, (scales,), 1, mode, grad_input=grad_input)


def nearest2d_backward(grad_output, output_size, input_size, scales_h=None, scales_w=None, *, mode, grad_input=None):
    return _backward(grad_output, output_size, input_size, (scales_h, scales_w), 2, mode, grad_input=grad_input)


def nearest3d_backward(grad_output, output_size, input_size, scales_d=None, scales_h=None, scales_w=None, *, mode, grad_input=None):
    return _backward(grad_output, output_size, input_size, (scales_d, scales_h, scales_w), 3, mode, grad_input=grad_input)


def linear1d(a, output_size, align_corners, scales=None, *, out=None):
    return _forward(a, output_size, (scales,), 1, 2, align_corners, out)


def linear2d(a, output_size, align_corners, scales_h=None, scales_w=None, *, mode, out=None):
    return _forward(a, output_size, (scales_h, scales_w), 2, mode, align_corners, out)


def linear3d(a, output_size, align_corners, scales_d=None, scales_h=None, scales_w=None, *, out=None):
    return _forward(a, output_size, (scales_d, scales_h, scales_w), 3, 2, align_corners, out)


def linear1d_backward(grad_output, output_size, input_size, align_corners, scales=None, *, grad_input=None):
    return _backward(grad_output, output_size, input_size, (scales,), 1, 2, align_corners, grad_input)


def linear2d_backward(grad_output, output_size, input_size, align_corners, scales_h=None, scales_w=None, *, mode, grad_input=None):
    return _backward(grad_output, output_size, input_size, (scales_h, scales_w), 2, mode, align_corners, grad_input)


def linear3d_backward(grad_output, output_size, input_size, align_corners, scales_d=None, scales_h=None, scales_w=None, *, grad_input=None):
    return _backward(grad_output, output_size, input_size, (scales_d, scales_h, scales_w), 3, 2, align_corners, grad_input)


def lanczos2d(a, output_size, align_corners, scales_h=None, scales_w=None, *, out=None):
    shape = _shape(a.shape, output_size, 2)
    if tuple(a.shape) != shape and a.dtype not in (torch.float32, torch.float64, torch.uint8):
        raise RuntimeError('Lanczos resizing requires float32, float64, or uint8 input')
    result = torch.empty(shape, dtype=a.dtype, device=a.device, memory_format=suggest_memory_format(a))
    if tuple(a.shape) == shape:
        result.copy_(a)
    elif result.numel():
        _C.scientific_call(333, [a.resolve_neg(), result], [int(align_corners)],
                           [0.0 if s is None else s for s in (scales_h, scales_w)])
    return _output(result, out, (a,))


def lanczos2d_backward(grad_output, output_size, input_size, align_corners, scales_h=None, scales_w=None, *, grad_input=None):
    shape = _shape(input_size, output_size, 2, backward=True)
    if tuple(grad_output.shape) != shape:
        raise RuntimeError('Lanczos gradient must match the forward output shape')
    if grad_output.dtype not in (torch.float32, torch.float64):
        raise RuntimeError('Lanczos backward requires float32 or float64 input')
    result = torch.empty(input_size, dtype=grad_output.dtype, device=grad_output.device)
    if result.numel():
        _C.scientific_call(334, [grad_output.resolve_neg(), result], [int(align_corners)],
                           [0.0 if s is None else s for s in (scales_h, scales_w)])
    return _output(result, grad_input, (grad_output,))


def register(library):
    entries = []
    for rank, forward, backward in ((1, nearest1d, nearest1d_backward),
                                     (2, nearest2d, nearest2d_backward),
                                     (3, nearest3d, nearest3d_backward)):
        entries.append((f'upsample_nearest{rank}d', partial(forward, mode=0), partial(backward, mode=0)))
        entries.append((f'_upsample_nearest_exact{rank}d', partial(forward, mode=1), partial(backward, mode=1)))
    entries.extend((('upsample_linear1d', linear1d, linear1d_backward),
                    ('upsample_bilinear2d', partial(linear2d, mode=2), partial(linear2d_backward, mode=2)),
                    ('upsample_bicubic2d', partial(linear2d, mode=3), partial(linear2d_backward, mode=3)),
                    ('_upsample_bilinear2d_aa', partial(linear2d, mode=4), partial(linear2d_backward, mode=4)),
                    ('_upsample_bicubic2d_aa', partial(linear2d, mode=5), partial(linear2d_backward, mode=5)),
                    ('upsample_trilinear3d', linear3d, linear3d_backward)))
    if hasattr(torch.ops.aten, '_upsample_lanczos2d_aa'):
        entries.append(('_upsample_lanczos2d_aa', lanczos2d, lanczos2d_backward))
    for name, forward, backward in entries:
        library.impl(name, forward)
        library.impl(name + '.out', forward)
        library.impl(name + '_backward', backward)
        library.impl(name + '_backward.grad_input', backward)
