from functools import partial
import warnings

import torch
from torch._prims_common import suggest_memory_format
from . import _C
from ._ops import _dtypes, _integral_dtypes
from ._ordering import _output
from ._spatial import _parameters


def _input(a, rank, adaptive, maximum):
    if a.dtype not in _dtypes:
        raise RuntimeError('pooling requires real floating input')
    if a.ndim not in (rank + 1, rank + 2):
        raise RuntimeError('pooling requires a batched or unbatched tensor of the expected rank')
    nonempty = a.shape[-rank:]
    if adaptive and (maximum or rank == 3):
        nonempty = a.shape[1:]
    elif not adaptive:
        nonempty = a.shape[-rank - 1:]
    if any(n <= 0 for n in nonempty):
        raise RuntimeError('pooling input has an empty non-batch dimension')


def _memory(a, rank, adaptive, maximum):
    if a.ndim == rank + 2 and (rank == 2 or maximum and not adaptive):
        return suggest_memory_format(a)
    return torch.contiguous_format


def _five(a, rank):
    if a.ndim == rank + 1:
        a = a.unsqueeze(0)
    return a.unsqueeze(2) if rank == 2 else a


def _regular(a, rank, kernel_size, stride, padding, dilation, ceil_mode, maximum, include_pad=True, divisor=None):
    _input(a, rank, False, maximum)
    kernel = _parameters(kernel_size, rank, 'pooling kernel', minimum=1)
    step = _parameters(stride, rank, 'pooling stride', default=kernel, minimum=1)
    pad = _parameters(padding, rank, 'pooling padding')
    dilate = _parameters(dilation, rank, 'pooling dilation', minimum=1)
    if any(p > k // 2 for p, k in zip(pad, kernel)):
        raise RuntimeError('pooling padding must not exceed half the kernel size')
    if not maximum and rank == 3 and any(n < k for n, k in zip(a.shape[-rank:], kernel)):
        raise RuntimeError('avg_pool3d input spatial dimensions must not be smaller than the kernel')
    if divisor == 0:
        raise RuntimeError('average pooling divisor must be nonzero')
    sizes = []
    for n, k, s, p, d in zip(a.shape[-rank:], kernel, step, pad, dilate):
        size = (n + 2 * p - d * (k - 1) - 1 + (s - 1 if ceil_mode else 0)) // s + 1
        if ceil_mode and (size - 1) * s >= n + p:
            size -= 1
        if size <= 0:
            raise RuntimeError('pooling output spatial dimensions must be positive')
        sizes.append(size)
    if rank == 2:
        kernel, step, pad, dilate = (1, *kernel), (1, *step), (0, *pad), (1, *dilate)
    memory = _memory(a, rank, False, maximum)
    params = (*kernel, *step, *pad, *dilate, 0 if divisor is None else divisor,
              int(include_pad), 0, int(maximum), rank, int(memory != torch.contiguous_format))
    return (*a.shape[:-rank], *sizes), params, memory


def _adaptive(a, rank, output_size, maximum):
    _input(a, rank, True, maximum)
    sizes = tuple(output_size)
    if len(sizes) != rank or any(n < 0 for n in sizes):
        raise RuntimeError('adaptive pooling requires one nonnegative output size per spatial dimension')
    memory = _memory(a, rank, True, maximum)
    params = (1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, int(maximum), rank,
              int(memory != torch.contiguous_format))
    return (*a.shape[:-rank], *sizes), params, memory


def _forward(a, shape, params, memory, out=None, indices=None):
    maximum, rank = bool(params[15]), params[16]
    if maximum and (out is None) != (indices is None):
        raise RuntimeError('both pooling outputs must be provided together')
    value = torch.empty(shape, dtype=a.dtype, device=a.device, memory_format=memory)
    index = (torch.empty(shape, dtype=torch.int64, device=a.device, memory_format=memory)
             if maximum else a.new_empty((0,), dtype=torch.int64))
    if value.numel():
        _C.scientific_call(318, [_five(a.resolve_neg(), rank), _five(value, rank),
                               _five(index, rank) if maximum else index], params, [])
    if maximum:
        if out is not None:
            _C.check_index_output(out, indices, indices)
        return _output(value, out, (a,)), _output(index, indices, (a,))
    return _output(value, out, (a,))


def _nondeterministic(name):
    if torch.are_deterministic_algorithms_enabled():
        message = name + ' does not have a deterministic RUDA implementation'
        if torch.is_deterministic_algorithms_warn_only_enabled():
            warnings.warn(message, stacklevel=3)
        else:
            raise RuntimeError(message)


def _backward(grad, a, shape, params, memory, indices=None, out=None):
    maximum, adaptive, rank = bool(params[15]), bool(params[14]), params[16]
    if grad.dtype != a.dtype or grad.device != a.device or tuple(grad.shape) != tuple(shape):
        raise RuntimeError('pooling gradient must match forward output shape, dtype and device')
    if adaptive and any(n == 0 for n in grad.shape[1:]):
        raise RuntimeError('adaptive pooling gradient must have nonempty non-batch dimensions')
    if maximum and (indices is None or indices.dtype != torch.int64 or indices.device != a.device or tuple(indices.shape) != tuple(shape)):
        raise RuntimeError('pooling indices must be int64 on the input device with the forward output shape')
    if not maximum and (adaptive or rank == 3) or maximum and not adaptive and rank == 3:
        name = ('adaptive_' if adaptive else '') + ('max' if maximum else 'avg') + f'_pool{rank}d_backward'
        _nondeterministic(name)
    result = torch.empty(a.shape, dtype=a.dtype, device=a.device, memory_format=memory)
    index = a.new_empty((0,), dtype=torch.int64) if indices is None else _five(indices.resolve_neg(), rank)
    if result.numel():
        _C.scientific_call(319, [_five(a.resolve_neg(), rank), _five(grad.resolve_neg(), rank), index,
                               _five(result, rank)], params, [])
    return _output(result, out, (a, grad, indices))


def avg_pool(a, kernel_size, stride=(), padding=(0,), ceil_mode=False, count_include_pad=True, divisor_override=None, *, rank, out=None):
    shape, params, memory = _regular(a, rank, kernel_size, stride, padding, (1,), ceil_mode, False, count_include_pad, divisor_override)
    return _forward(a, shape, params, memory, out)


def avg_backward(grad, a, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, *, rank, grad_input=None):
    shape, params, memory = _regular(a, rank, kernel_size, stride, padding, (1,), ceil_mode, False, count_include_pad, divisor_override)
    return _backward(grad, a, shape, params, memory, out=grad_input)


def max_pool(a, kernel_size, stride=(), padding=(0,), dilation=(1,), ceil_mode=False, *, rank, out=None, indices=None):
    shape, params, memory = _regular(a, rank, kernel_size, stride, padding, dilation, ceil_mode, True)
    return _forward(a, shape, params, memory, out, indices)


def max_backward(grad, a, kernel_size, stride, padding, dilation, ceil_mode, indices, *, rank, grad_input=None):
    shape, params, memory = _regular(a, rank, kernel_size, stride, padding, dilation, ceil_mode, True)
    return _backward(grad, a, shape, params, memory, indices, grad_input)


def adaptive_avg(a, output_size, *, rank, out=None):
    shape, params, memory = _adaptive(a, rank, output_size, False)
    return _forward(a, shape, params, memory, out)


def adaptive_avg_backward(grad, a, *, rank, out=None, grad_input=None):
    shape, params, memory = _adaptive(a, rank, grad.shape[-rank:], False)
    return _backward(grad, a, shape, params, memory, out=out if grad_input is None else grad_input)


def adaptive_max(a, output_size, *, rank, out=None, indices=None):
    shape, params, memory = _adaptive(a, rank, output_size, True)
    return _forward(a, shape, params, memory, out, indices)


def adaptive_max_backward(grad, a, indices, *, rank, grad_input=None):
    shape, params, memory = _adaptive(a, rank, grad.shape[-rank:], True)
    return _backward(grad, a, shape, params, memory, indices, grad_input)


def _fractional_shape(a, kernel_size, output_size, rank):
    _input(a, rank, True, True)
    kernel, size = tuple(kernel_size), tuple(output_size)
    if len(kernel) != rank or len(size) != rank:
        raise RuntimeError('fractional pooling requires one kernel and output size per spatial dimension')
    if any(n < 0 for n in size):
        raise RuntimeError('fractional pooling output dimensions must be nonnegative')
    for n, k, o in zip(a.shape[-rank:], kernel, size):
        if rank == 2 and o + k - 1 > n or rank == 3 and (k > n or o + k - 1 >= n):
            raise RuntimeError('fractional pooling kernel and output sizes exceed the input size')
    return (*a.shape[:-rank], *size), ((1, *kernel) if rank == 2 else kernel)


def fractional_max(a, kernel_size, output_size, random_samples, *, rank, output=None, indices=None):
    shape, kernel = _fractional_shape(a, kernel_size, output_size, rank)
    batched = _five(a, rank)
    if random_samples.dtype != a.dtype or random_samples.device != a.device or random_samples.ndim != 3:
        raise RuntimeError('fractional pooling random samples require input dtype/device and three dimensions')
    if (random_samples.shape[0] < batched.shape[0] or random_samples.shape[1] != batched.shape[1]
            or random_samples.shape[2] != rank):
        raise RuntimeError('fractional pooling random samples must have shape [at least N, C, spatial rank]')
    if (output is None) != (indices is None):
        raise RuntimeError('both fractional pooling outputs must be provided together')
    value = a.new_empty(shape)
    index = a.new_empty(shape, dtype=torch.int64)
    if value.numel():
        _C.scientific_call(320, [_five(a.resolve_neg(), rank), random_samples.resolve_neg(),
                               _five(value, rank), _five(index, rank)], [*kernel, rank], [])
    if output is not None:
        _C.check_index_output(output, indices, indices)
    return _output(value, output, (a, random_samples)), _output(index, indices, (a, random_samples))


def fractional_max_backward(grad, a, kernel_size, output_size, indices, *, rank, grad_input=None):
    _input(a, rank, True, True)
    size = tuple(output_size)
    if len(size) != rank or any(n < 0 for n in size):
        raise RuntimeError('fractional pooling backward requires nonnegative spatial output sizes')
    shape = (*a.shape[:-rank], *size)
    if grad.dtype != a.dtype or grad.device != a.device or tuple(grad.shape) != shape:
        raise RuntimeError('fractional pooling gradient must match forward output shape, dtype and device')
    if indices.dtype != torch.int64 or indices.device != a.device or tuple(indices.shape) != shape:
        raise RuntimeError('fractional pooling indices must be int64 with the forward output shape and device')
    _nondeterministic(f'fractional_max_pool{rank}d_backward')
    if a.numel() and not grad.numel():
        raise RuntimeError('fractional pooling backward requires a nonempty output spatial volume')
    result = a.new_empty(a.shape)
    if result.numel():
        _C.scientific_call(321, [_five(grad.resolve_neg(), rank), _five(indices.resolve_neg(), rank),
                               _five(result, rank)], [rank], [])
    return _output(result, grad_input, (a, grad, indices))


def _unpool(a, indices, output_size, rank, out):
    if a.dtype == torch.bool or a.dtype not in _dtypes + _integral_dtypes:
        raise RuntimeError('max unpooling requires real floating or integer input')
    if a.ndim not in (rank + 1, rank + 2) or any(n <= 0 for n in a.shape[1:]):
        raise RuntimeError('max unpooling requires the expected input rank with nonempty non-batch dimensions')
    if indices.dtype != torch.int64 or indices.device != a.device or indices.shape != a.shape:
        raise RuntimeError('max unpooling indices must be int64 and match input shape and device')
    size = tuple(output_size)
    if len(size) != rank or any(n < 0 for n in size):
        raise RuntimeError('max unpooling requires nonnegative spatial output dimensions')
    if out is not None and not out.is_contiguous():
        raise RuntimeError('max unpooling output must be contiguous')
    _nondeterministic(f'max_unpool{rank}d')
    result = a.new_empty((*a.shape[:-rank], *size))
    if result.numel():
        _C.scientific_call(322, [_five(a.resolve_neg(), rank), _five(indices.resolve_neg(), rank),
                               _five(result, rank)], [rank], [])
    return _output(result, out, (a, indices))


def max_unpool2d(a, indices, output_size, *, out=None):
    return _unpool(a, indices, output_size, 2, out)


def max_unpool3d(a, indices, output_size, stride, padding, *, out=None):
    if len(stride) != 3 or len(padding) != 3 or any(s <= 0 for s in stride):
        raise RuntimeError('max_unpool3d requires three positive strides and three padding values')
    return _unpool(a, indices, output_size, 3, out)


def register(library):
    for rank in (2, 3):
        library.impl(f'fractional_max_pool{rank}d', partial(fractional_max, rank=rank))
        library.impl(f'fractional_max_pool{rank}d.output', partial(fractional_max, rank=rank))
        library.impl(f'fractional_max_pool{rank}d_backward', partial(fractional_max_backward, rank=rank))
        library.impl(f'fractional_max_pool{rank}d_backward.grad_input', partial(fractional_max_backward, rank=rank))
        for stem, forward_fn, backward_fn in (('avg_pool', avg_pool, avg_backward),
                                               ('max_pool', max_pool, max_backward),
                                               ('adaptive_max_pool', adaptive_max, adaptive_max_backward)):
            name = f'{stem}{rank}d' + ('_with_indices' if stem == 'max_pool' else '')
            library.impl(name, partial(forward_fn, rank=rank))
            library.impl(name + '.out', partial(forward_fn, rank=rank))
            library.impl(name + '_backward', partial(backward_fn, rank=rank))
            library.impl(name + '_backward.grad_input', partial(backward_fn, rank=rank))
        name = f'_adaptive_avg_pool{rank}d'
        library.impl(name, partial(adaptive_avg, rank=rank))
        library.impl(name + '.out', partial(adaptive_avg, rank=rank))
        library.impl(name[1:] + '.out', partial(adaptive_avg, rank=rank))
        library.impl(name + '_backward', partial(adaptive_avg_backward, rank=rank))
        library.impl(name + '_backward.out', partial(adaptive_avg_backward, rank=rank))
    library.impl('adaptive_avg_pool3d_backward.grad_input', partial(adaptive_avg_backward, rank=3))
    for name, function in (('max_unpool2d', max_unpool2d), ('max_unpool3d', max_unpool3d)):
        library.impl(name, function)
        library.impl(name + '.out', function)
