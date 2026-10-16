import math

import torch

from . import _C
from ._indexing import _logical
from ._ops import _dimension, _dtypes, _numeric_dtypes
from ._ordering import _output


def logcumsumexp(a, dim, *, out=None):
    if a.layout != torch.strided:
        raise RuntimeError('logcumsumexp requires strided input')
    axis = _dimension(dim, a.ndim)
    if a.ndim == 0:
        return _output(_logical(a).clone(), out, (a,))
    result = torch.empty_like(a, memory_format=torch.contiguous_format)
    if not a.numel():
        return _output(result, out, (a,))
    if a.dtype not in _numeric_dtypes:
        raise RuntimeError('logcumsumexp requires real floating or complex input')
    if a.numel() > 2**32 - 1:
        raise RuntimeError('logcumsumexp exceeds native U32 indexing')
    outer, width, inner = math.prod(a.shape[:axis]), a.shape[axis], math.prod(a.shape[axis + 1:])
    value, target = _logical(a).reshape(outer, width, inner), result.reshape(outer, width, inner)
    complex_input = a.is_complex()
    if complex_input:
        source_parts, target_parts = torch.view_as_real(value), torch.view_as_real(target)
        tensors = [source_parts[..., 0], source_parts[..., 1], target_parts[..., 0], target_parts[..., 1]]
    else:
        tensors = [value, value, target, target]
    parallel = axis == a.ndim - 1 or a.numel() == width
    _C.scientific_call(365, tensors, [int(complex_input), int(parallel)], [])
    return _output(result, out, (a,))


def _segment(data, reduce, lengths, indices, offsets, axis, unsafe):
    if data.layout != torch.strided or data.ndim == 0 or data.dtype not in _dtypes:
        raise RuntimeError('segment_reduce requires a nonscalar strided real floating tensor')
    axis = _dimension(axis, data.ndim)
    kinds = {'max': 0, 'amax': 0, 'mean': 1, 'min': 2, 'amin': 2, 'sum': 3, 'prod': 4}
    if reduce not in kinds:
        raise RuntimeError('segment_reduce requires sum, prod, mean, max/amax or min/amin')
    if indices is not None:
        raise RuntimeError('segment_reduce indices-based reduction is not supported')
    descriptor = offsets if offsets is not None else lengths
    if descriptor is None:
        raise RuntimeError('segment_reduce requires lengths or offsets')
    if (descriptor.layout != torch.strided or descriptor.device != data.device
            or descriptor.dtype not in (torch.int32, torch.int64) or descriptor.ndim != axis + 1
            or tuple(descriptor.shape[:-1]) != tuple(data.shape[:axis])):
        raise RuntimeError('segment descriptor must match input leading dimensions/device, use int32/int64, and end at axis')
    count = descriptor.shape[-1] - int(offsets is not None)
    if count < 0:
        raise RuntimeError('segment offsets must include at least one boundary')
    outer, width, inner = math.prod(data.shape[:axis]), data.shape[axis], math.prod(data.shape[axis + 1:])
    shape = (*data.shape[:axis], count, *data.shape[axis + 1:])
    if max(data.numel(), descriptor.numel(), outer * (count + 1), math.prod(shape), width) > 2**32 - 1:
        raise RuntimeError('segment_reduce exceeds native U32 indexing')
    bounds = _logical(descriptor)
    if offsets is None:
        if not unsafe:
            if not bounds.numel():
                raise RuntimeError('segment_reduce cannot validate minimum of empty lengths')
            if bounds.min().item() < 0:
                raise RuntimeError('segment_reduce lengths contain negative values')
            if not torch.all(bounds.sum(dim=-1) == width).item():
                raise RuntimeError('segment_reduce lengths must sum to the reduction dimension size')
        zero = torch.zeros((*bounds.shape[:-1], 1), dtype=bounds.dtype, device=bounds.device)
        bounds = torch.cat((zero, bounds), dim=-1).cumsum(dim=-1, dtype=bounds.dtype)
    return (_logical(data).reshape(outer, width, inner), bounds.reshape(outer, count + 1),
            (outer, count, inner), shape, kinds[reduce])


def _initial(value, data, kind):
    if value is None:
        return (-math.inf, 0., math.inf, 0., 1.)[kind]
    if isinstance(value, complex):
        if value.imag != 0:
            raise RuntimeError('segment_reduce initial must have zero imaginary part')
        value = value.real
    value = float(value)
    if math.isfinite(value) and abs(value) > torch.finfo(data.dtype).max:
        raise RuntimeError('segment_reduce initial overflows the input dtype')
    return value


def segment_reduce(data, reduce, *, lengths=None, indices=None, offsets=None, axis=0, unsafe=False, initial=None, out=None):
    x, bounds, native_shape, shape, kind = _segment(data, reduce, lengths, indices, offsets, axis, unsafe)
    start = _initial(initial, data, kind)
    result = torch.empty(native_shape, dtype=data.dtype, device=data.device)
    if result.numel():
        _C.scientific_call(366, [x, bounds, result], [kind, int(initial is not None), int(data.ndim == 1)], [start])
    return _output(result.reshape(shape), out, (data, lengths, indices, offsets))


def segment_backward(grad, output, data, reduce, *, lengths=None, offsets=None, axis=0, initial=None, out=None):
    x, bounds, native_shape, shape, kind = _segment(data, reduce, lengths, None, offsets, axis, True)
    for value in (grad, output):
        if (value.layout != torch.strided or value.device != data.device or value.dtype != data.dtype
                or tuple(value.shape) != shape):
            raise RuntimeError('segment_reduce backward grad/output shape, dtype or device mismatch')
    start = _initial(initial, data, kind)
    result = torch.zeros(x.shape, dtype=data.dtype, device=data.device)
    if output.numel():
        _C.scientific_call(367, [x, bounds, _logical(output).reshape(native_shape), _logical(grad).reshape(native_shape), result],
                           [kind, int(initial is not None), 0], [start])
    return _output(result.reshape(data.shape), out, (grad, output, data, lengths, offsets))


def register(library):
    for name, function in (('_logcumsumexp', logcumsumexp), ('_logcumsumexp.out', logcumsumexp),
                           ('segment_reduce', segment_reduce), ('segment_reduce.out', segment_reduce),
                           ('_segment_reduce_backward', segment_backward), ('_segment_reduce_backward.out', segment_backward)):
        library.impl(name, function)
