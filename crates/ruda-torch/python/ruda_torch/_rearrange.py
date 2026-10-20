from functools import partial

import torch
from . import _C
from ._ops import _dimension, _numeric_dtypes, _storage_dtypes
from ._ordering import _output


def _logical(a):
    if a.layout != torch.strided or a.dtype not in _storage_dtypes:
        raise RuntimeError('rearrangement requires a supported strided tensor')
    return a.resolve_conj().resolve_neg()


def _pair(values, name, minimum):
    values = tuple(values)
    if len(values) != 2 or any(v < minimum for v in values):
        raise RuntimeError(f'{name} requires two values >= {minimum}')
    return values


def _windows(spatial, kernel_size, dilation, padding, stride):
    kernel = _pair(kernel_size, 'kernel_size', 1)
    dilation = _pair(dilation, 'dilation', 1)
    padding = _pair(padding, 'padding', 0)
    stride = _pair(stride, 'stride', 1)
    blocks = tuple((n + 2 * p - d * (k - 1) - 1) // s + 1
                   for n, k, d, p, s in zip(spatial, kernel, dilation, padding, stride))
    if any(n <= 0 for n in blocks):
        raise RuntimeError('image unfolding requires positive sliding-window counts')
    return blocks, (*kernel, *dilation, *padding, *stride)


def _image_type(a):
    if a.dtype not in _numeric_dtypes and a.dtype != torch.bool:
        raise RuntimeError('im2col/col2im require floating, complex, or bool input')
    return _logical(a)


def im2col(a, kernel_size, dilation, padding, stride, *, out=None):
    value = _image_type(a)
    if a.ndim not in (3, 4) or any(n <= 0 for n in a.shape[-3:]):
        raise RuntimeError('im2col requires nonempty CHW or batched NCHW input')
    blocks, params = _windows(a.shape[-2:], kernel_size, dilation, padding, stride)
    shape = (*a.shape[:-3], a.shape[-3] * params[0] * params[1], blocks[0] * blocks[1])
    result = a.new_empty(shape)
    if result.numel():
        source = value.unsqueeze(0) if a.ndim == 3 else value
        target = result.unsqueeze(0) if a.ndim == 3 else result
        _C.scientific_call(335, [source, target], params, [])
    return _output(result, out, (a,))


def col2im(a, output_size, kernel_size, dilation, padding, stride, *, out=None):
    value = _image_type(a)
    if a.ndim not in (2, 3) or any(n <= 0 for n in a.shape[-2:]):
        raise RuntimeError('col2im requires nonempty column and block dimensions with an optional batch')
    spatial = _pair(output_size, 'output_size', 1)
    blocks, params = _windows(spatial, kernel_size, dilation, padding, stride)
    kernel = params[0] * params[1]
    if a.shape[-2] % kernel or a.shape[-1] != blocks[0] * blocks[1]:
        raise RuntimeError('col2im column shape does not match the kernel and sliding-window counts')
    result = a.new_empty((*a.shape[:-2], a.shape[-2] // kernel, *spatial))
    if result.numel():
        source = value.unsqueeze(0) if a.ndim == 2 else value
        target = result.unsqueeze(0) if a.ndim == 2 else result
        _C.scientific_call(336, [source, target], params, [])
    return _output(result, out, (a,))


def pixel_shuffle(a, factor, *, inverse=False, out=None):
    if a.ndim < 3 or factor <= 0:
        raise RuntimeError('pixel shuffle requires at least three dimensions and a positive factor')
    channels, height, width = a.shape[-3:]
    square = factor * factor
    if inverse:
        if height % factor or width % factor:
            raise RuntimeError('pixel unshuffle spatial dimensions must be divisible by the factor')
        shape = (*a.shape[:-3], channels * square, height // factor, width // factor)
    else:
        if square > 2**63 - 1 or channels % square:
            raise RuntimeError('pixel shuffle requires a valid squared factor dividing the channels')
        shape = (*a.shape[:-3], channels // square, height * factor, width * factor)
    value = _logical(a)
    result = a.new_empty(shape)
    if result.numel():
        _C.scientific_call(337, [value, result], [int(inverse), factor], [])
    return _output(result, out, (a,))


def channel_shuffle(a, groups, *, out=None):
    if a.ndim <= 2 or groups <= 0 or a.shape[1] % groups:
        raise RuntimeError('channel shuffle requires rank >= 3 and a positive group count dividing channels')
    if not a.numel():
        result = torch.as_strided(a, a.shape, a.stride())
    elif groups == 1 or groups == a.shape[1]:
        result = a.contiguous().view(a.shape)
    else:
        result = a.new_empty(a.shape)
        _C.scientific_call(337, [_logical(a), result], [2, groups], [])
    return _output(result, out, (a,))


def _unfold_shape(input_sizes, dim, size, step):
    shape = list(input_sizes)
    if any(n < 0 for n in shape):
        raise RuntimeError('unfold input dimensions cannot be negative')
    dim = _dimension(dim, len(shape))
    maximum = shape[dim] if shape else 1
    if size < 0 or size > maximum or step <= 0:
        raise RuntimeError('unfold requires 0 <= size <= input dimension and step > 0')
    if shape:
        shape[dim] = (shape[dim] - size) // step + 1
    return (*shape, size), dim


def unfold(a, dimension, size, step):
    shape, dim = _unfold_shape(a.shape, dimension, size, step)
    strides = list(a.stride())
    last_stride = strides[dim] if a.ndim else 1
    if a.ndim:
        strides[dim] *= step
    return torch.as_strided(a, shape, (*strides, last_stride))


def unfold_backward(grad_in, input_sizes, dim, size, step, *, out=None):
    expected, dim = _unfold_shape(input_sizes, dim, size, step)
    value = _logical(grad_in)
    if step >= size:
        value = value.expand(expected)
    elif tuple(value.shape) != expected:
        raise RuntimeError('unfold gradient shape does not match the unfolded input')
    elif grad_in.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise RuntimeError('overlapping unfold backward does not support float8')
    result = grad_in.new_empty(input_sizes)
    if result.numel():
        scalar = len(input_sizes) == 0
        source = value.unsqueeze(0) if scalar else value
        target = result.unsqueeze(0) if scalar else result
        _C.scientific_call(338, [source, target], [dim, size, step, int(scalar)], [])
    return _output(result, out, (grad_in,))


def register(library):
    for name, function in (('im2col', im2col), ('col2im', col2im),
                           ('pixel_shuffle', pixel_shuffle),
                           ('pixel_unshuffle', partial(pixel_shuffle, inverse=True)),
                           ('channel_shuffle', channel_shuffle), ('unfold_backward', unfold_backward)):
        library.impl(name, function)
        library.impl(name + '.out', function)
    library.impl('unfold', unfold)
    library.impl('roll', roll)


def roll(a, shifts, dims=()):
    shifts = (shifts,) if isinstance(shifts, int) else tuple(shifts)
    dims = (dims,) if isinstance(dims, int) else tuple(dims)
    if not shifts:
        raise RuntimeError('roll requires at least one shift')
    flatten = not dims and len(shifts) == 1
    if not flatten and len(shifts) != len(dims):
        raise RuntimeError('roll shifts and dimensions must align')
    value = _logical(a)
    result = torch.empty(a.shape, dtype=a.dtype, device=a.device)
    if not result.numel():
        return result
    if a.dtype not in _numeric_dtypes + (torch.bool, torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.complex32):
        raise RuntimeError('unsupported CUDA roll dtype')
    if flatten:
        normalized = [shifts[0] % a.numel()]
    else:
        if a.ndim == 0:
            raise IndexError('roll dimensions cannot be specified for a scalar tensor')
        normalized = [0] * a.ndim
        for shift, dim in zip(shifts, dims):
            axis = _dimension(dim, a.ndim)
            normalized[axis] = (normalized[axis] + shift) % a.shape[axis]
    _C.scientific_call(404, [value, result], [int(flatten), *normalized], [])
    return result
