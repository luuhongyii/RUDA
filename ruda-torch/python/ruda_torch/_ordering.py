import math
import sys
import torch
from . import _C
from ._ops import _registry, _dtypes, _integral_dtypes, _dimension, _typed_value, clone
from ._indexing import _logical, gather, index_select


def _native_available(*tensors):
    return (getattr(sys.modules.get(__package__), '_ordering_available', False)
            and all(t.numel() <= 2**32 - 1 and all(n <= 2**32 - 1 for n in t.shape) for t in tensors))


def _real(a):
    if a.dtype not in _dtypes + _integral_dtypes:
        raise RuntimeError('ordering requires a supported real or integral dtype')
    return _logical(a)


def _output(result, out, inputs, *, cast=False):
    if out is None:
        return result
    if out.device != result.device or not (torch.can_cast(result.dtype, out.dtype) if cast else out.dtype == result.dtype):
        raise RuntimeError('output dtype or device does not match the operation')
    for value in inputs:
        if isinstance(value, torch.Tensor):
            _C.check_inplace(out, value)
    if out.shape != result.shape:
        _C.prepare_index_output(out, list(result.shape))
    out.copy_(result)
    return out


def _outputs(results, outputs, inputs):
    if outputs is None:
        return results
    for position, value in enumerate(outputs):
        for other in outputs[position + 1:]:
            _C.check_index_output(value, other, other)
    return tuple(_output(value, out, inputs) for value, out in zip(results, outputs))


def _choose_left(a, b, maximum, *, right_ties=False, ignore_nan=False):
    if right_ties:
        choose = a > b if maximum else a < b
    else:
        choose = a >= b if maximum else a <= b
    if a.is_floating_point():
        left_nan, right_nan = a != a, b != b
        if ignore_nan:
            choose = (choose & ~left_nan) | right_nan
        elif right_ties:
            choose = choose | (left_nan & ~right_nan)
        else:
            choose = choose | left_nan
    return choose


def extreme(a, b, *, maximum, ignore_nan=False, out=None):
    from ._elementwise_extrema import extreme as native_extreme
    return native_extreme(a, b, maximum=maximum, ignore_nan=ignore_nan, out=out)


def clamp(a, min=None, max=None, *, inplace=False, out=None):
    from ._elementwise_extrema import clamp as native_clamp
    return native_clamp(a, min, max, inplace=inplace, out=out)


def clamp_bound(a, bound, *, maximum, inplace=False, out=None):
    from ._elementwise_extrema import clamp_bound as native_clamp_bound
    return native_clamp_bound(a, bound, maximum=maximum, inplace=inplace, out=out)


def _reduce(a, dim, keepdim, maximum, *, indices=False):
    value = _real(a)
    dims = tuple(range(a.ndim)) if dim is None else (dim,) if isinstance(dim, int) else tuple(dim)
    if not dims:
        dims = tuple(range(a.ndim))
    axes = tuple(_dimension(axis, a.ndim) for axis in dims)
    if len(set(axes)) != len(axes):
        raise RuntimeError('duplicate reduction dimension')
    if a.ndim == 0:
        result = clone(value)
        return (result, torch.zeros((), dtype=torch.int64, device=a.device)) if indices else result
    remaining = tuple(axis for axis in range(a.ndim) if axis not in axes)
    length = math.prod(a.shape[axis] for axis in axes)
    if not length:
        raise IndexError('extrema reduction has no identity for an empty dimension')
    shape = tuple(a.shape[axis] for axis in remaining)
    result_shape = tuple(1 if axis in axes else size for axis, size in enumerate(a.shape)) if keepdim else shape
    if not a.numel():
        result = torch.empty(result_shape, dtype=a.dtype, device=a.device)
        return (result, torch.empty(result_shape, dtype=torch.int64, device=a.device)) if indices else result
    value = value.permute((*remaining, *axes)).reshape((*shape, length))
    if _native_available(value) and length <= 2**32 - 1:
        source = value.reshape(-1, length).contiguous()
        output = torch.empty((source.shape[0],), dtype=a.dtype, device=a.device)
        tensors = [source, output]
        if indices:
            positions = torch.empty(output.shape, dtype=torch.int64, device=a.device)
            tensors.append(positions)
        _C.ordering_call(0, tensors, [(4 if maximum else 5) if indices else (0 if maximum else 1), 2, 0])
        result = output.reshape(result_shape)
        return (result, positions.reshape(result_shape)) if indices else result
    positions = torch.arange(length, dtype=torch.int64, device=a.device).expand(value.shape) if indices else None
    while value.shape[-1] > 1:
        pairs = value.shape[-1] // 2
        left, right = value[..., :2*pairs:2], value[..., 1:2*pairs:2]
        choose = _choose_left(left, right, maximum)
        reduced = torch.where(choose, left, right)
        if indices:
            selected = torch.where(choose, positions[..., :2*pairs:2], positions[..., 1:2*pairs:2])
            positions = torch.cat((selected, positions[..., -1:]), -1) if value.shape[-1] % 2 else selected
        value = torch.cat((reduced, value[..., -1:]), -1) if value.shape[-1] % 2 else reduced
    value = value.reshape(result_shape)
    return (value, positions.reshape(result_shape)) if indices else value


def reduce_extreme(a, dim=None, keepdim=False, *, maximum, out=None):
    return _output(_reduce(a, dim, keepdim, maximum), out, (a,))


def extreme_dim(a, dim, keepdim=False, *, maximum, outputs=None):
    return _outputs(_reduce(a, dim, keepdim, maximum, indices=True), outputs, (a,))


def arg_extreme(a, dim=None, keepdim=False, *, maximum, out=None):
    if a.dtype == torch.bool:
        raise RuntimeError('argmax/argmin do not support bool')
    result = _reduce(a, dim, keepdim, maximum, indices=True)[1]
    return _output(result, out, (a,))


def aminmax(a, *, dim=None, keepdim=False, min=None, max=None):
    results = _reduce(a, dim, keepdim, False), _reduce(a, dim, keepdim, True)
    return _outputs(results, None if min is None and max is None else (min, max), (a,))


def cumulative_extreme(a, dim, *, maximum, values=None, indices=None):
    value = _real(a)
    axis = _dimension(dim, a.ndim)
    if a.ndim == 0:
        result = clone(value), torch.zeros((), dtype=torch.int64, device=a.device)
    elif not a.numel():
        result = torch.empty(a.shape, dtype=a.dtype, device=a.device), torch.empty(a.shape, dtype=torch.int64, device=a.device)
    elif _native_available(value):
        moved = value.movedim(axis, -1)
        source = moved.reshape(-1, moved.shape[-1]).contiguous()
        output = torch.empty_like(source)
        positions = torch.empty(source.shape, dtype=torch.int64, device=a.device)
        _C.ordering_call(1, [source, output, positions], [int(maximum)])
        result = (output.reshape(moved.shape).movedim(-1, axis).contiguous(),
                  positions.reshape(moved.shape).movedim(-1, axis).contiguous())
    else:
        value = value.movedim(axis, -1)
        positions = torch.arange(value.shape[-1], dtype=torch.int64, device=a.device).expand(value.shape)
        step = 1
        while step < value.shape[-1]:
            left, right = value[..., :-step], value[..., step:]
            choose = _choose_left(left, right, maximum, right_ties=True)
            positions = torch.cat((positions[..., :step], torch.where(choose, positions[..., :-step], positions[..., step:])), -1)
            value = torch.cat((value[..., :step], torch.where(choose, left, right)), -1)
            step *= 2
        result = clone(value.movedim(-1, axis)), clone(positions.movedim(-1, axis))
    return _outputs(result, None if values is None and indices is None else (values, indices), (a,))


def searchsorted(sequence, values, *, out_int32=False, right=False, side=None, sorter=None, out=None):
    if sequence.ndim == 0:
        raise RuntimeError('searchsorted boundaries must have at least one dimension')
    if side not in (None, 'left', 'right') or side == 'left' and right:
        raise RuntimeError('invalid or conflicting searchsorted side/right arguments')
    right = right or side == 'right'
    original = sequence, values, sorter
    dtype = torch.result_type(sequence, values)
    sequence = _real(sequence).to(dtype)
    if dtype == torch.bool:
        raise RuntimeError('searchsorted does not support bool')
    if isinstance(values, torch.Tensor) and values.device != sequence.device:
        raise RuntimeError('searchsorted inputs must be on the same device')
    values = _real(_typed_value(values, dtype, sequence.device))
    if sequence.ndim != 1 and (values.ndim != sequence.ndim or values.shape[:-1] != sequence.shape[:-1]):
        raise RuntimeError('searchsorted batch dimensions must match')
    length = sequence.shape[-1]
    if out_int32 and length >= 2**31:
        raise RuntimeError('searchsorted boundaries exceed the int32 index range')
    if sorter is not None:
        if sorter.dtype != torch.int64 or sorter.device != sequence.device or sorter.shape != sequence.shape:
            raise RuntimeError('searchsorted sorter must be int64 with the boundaries shape and device')
        sequence = gather(sequence, -1, sorter)
    if _native_available(sequence, values):
        rows = 1 if sequence.ndim == 1 else math.prod(sequence.shape[:-1])
        queries = values.numel() if sequence.ndim == 1 else values.shape[-1]
        source = sequence.reshape(rows, length).contiguous()
        query = values.reshape(rows, queries).contiguous()
        result = torch.empty(query.shape, dtype=torch.int32 if out_int32 else torch.int64, device=sequence.device)
        _C.ordering_call(2, [source, query, result], [int(right)])
        return _output(result.reshape(values.shape), out, original)
    low = torch.zeros(values.shape, dtype=torch.int64, device=sequence.device)
    high = torch.full(values.shape, length, dtype=torch.int64, device=sequence.device)
    if values.numel():
        for _ in range(length.bit_length()):
            active = low < high
            middle = torch.div(low + high, 2, rounding_mode='floor')
            safe = torch.where(middle < length, middle, length - 1)
            boundary = (index_select(sequence, 0, safe.reshape(-1)).reshape(values.shape)
                        if sequence.ndim == 1 else gather(sequence, -1, safe))
            advance = ~(boundary > values) if right else ~(boundary >= values)
            low = torch.where(active & advance, middle + 1, low)
            high = torch.where(active & ~advance, middle, high)
    result = low.to(torch.int32) if out_int32 else low
    return _output(result, out, original)


def bucketize(values, boundaries, *, out_int32=False, right=False, out=None):
    if boundaries.ndim != 1:
        raise RuntimeError('bucketize boundaries must be one-dimensional')
    return searchsorted(boundaries, values, out_int32=out_int32, right=right, out=out)


for name, maximum, ignore_nan in (('maximum', True, False), ('minimum', False, False), ('fmax', True, True), ('fmin', False, True)):
    _registry.impl(name, lambda a, b, maximum=maximum, ignore_nan=ignore_nan: extreme(a, b, maximum=maximum, ignore_nan=ignore_nan))
    _registry.impl(name + '.out', lambda a, b, *, out, maximum=maximum, ignore_nan=ignore_nan: extreme(a, b, maximum=maximum, ignore_nan=ignore_nan, out=out))

for name, maximum in (('max', True), ('min', False)):
    _registry.impl(name, lambda a, maximum=maximum: reduce_extreme(a, maximum=maximum))
    _registry.impl(name + '.unary_out', lambda a, *, out, maximum=maximum: reduce_extreme(a, maximum=maximum, out=out))
    _registry.impl(name + '.other', lambda a, b, maximum=maximum: extreme(a, b, maximum=maximum))
    _registry.impl(name + '.out', lambda a, b, *, out, maximum=maximum: extreme(a, b, maximum=maximum, out=out))
    _registry.impl(name + '.dim', lambda a, dim, keepdim=False, maximum=maximum: extreme_dim(a, dim, keepdim, maximum=maximum))
    _registry.impl('a' + name, lambda a, dim=(), keepdim=False, maximum=maximum: reduce_extreme(a, dim, keepdim, maximum=maximum))
    _registry.impl('a' + name + '.out', lambda a, dim=(), keepdim=False, *, out, maximum=maximum: reduce_extreme(a, dim, keepdim, maximum=maximum, out=out))
    _registry.impl('arg' + name, lambda a, dim=None, keepdim=False, maximum=maximum: arg_extreme(a, dim, keepdim, maximum=maximum))
    _registry.impl('arg' + name + '.out', lambda a, dim=None, keepdim=False, *, out, maximum=maximum: arg_extreme(a, dim, keepdim, maximum=maximum, out=out))
    _registry.impl('cum' + name, lambda a, dim, maximum=maximum: cumulative_extreme(a, dim, maximum=maximum))
    _registry.impl('cum' + name + '.out', lambda a, dim, *, values, indices, maximum=maximum: cumulative_extreme(a, dim, maximum=maximum, values=values, indices=indices))

for name, function in {
    'max.dim_max': lambda a, dim, keepdim=False, *, max, max_values: extreme_dim(a, dim, keepdim, maximum=True, outputs=(max, max_values)),
    'min.dim_min': lambda a, dim, keepdim=False, *, min, min_indices: extreme_dim(a, dim, keepdim, maximum=False, outputs=(min, min_indices)),
    'aminmax': aminmax, 'aminmax.out': aminmax,
    'clamp': clamp, 'clamp.Tensor': clamp, 'clamp.out': clamp, 'clamp.Tensor_out': clamp,
    'clamp_': lambda a, min=None, max=None: clamp(a, min, max, inplace=True),
    'clamp_.Tensor': lambda a, min=None, max=None: clamp(a, min, max, inplace=True),
    'clamp_min': lambda a, min: clamp_bound(a, min, maximum=True), 'clamp_min.Tensor': lambda a, min: clamp_bound(a, min, maximum=True),
    'clamp_max': lambda a, max: clamp_bound(a, max, maximum=False), 'clamp_max.Tensor': lambda a, max: clamp_bound(a, max, maximum=False),
    'clamp_min_': lambda a, min: clamp_bound(a, min, maximum=True, inplace=True), 'clamp_min_.Tensor': lambda a, min: clamp_bound(a, min, maximum=True, inplace=True),
    'clamp_max_': lambda a, max: clamp_bound(a, max, maximum=False, inplace=True), 'clamp_max_.Tensor': lambda a, max: clamp_bound(a, max, maximum=False, inplace=True),
    'clamp_min.out': lambda a, min, *, out: clamp_bound(a, min, maximum=True, out=out), 'clamp_min.Tensor_out': lambda a, min, *, out: clamp_bound(a, min, maximum=True, out=out),
    'clamp_max.out': lambda a, max, *, out: clamp_bound(a, max, maximum=False, out=out), 'clamp_max.Tensor_out': lambda a, max, *, out: clamp_bound(a, max, maximum=False, out=out),
    'searchsorted.Tensor': searchsorted, 'searchsorted.Scalar': searchsorted,
    'searchsorted.Tensor_out': searchsorted, 'searchsorted.Scalar_out': searchsorted,
    'bucketize.Tensor': bucketize, 'bucketize.Scalar': bucketize,
    'bucketize.Tensor_out': bucketize, 'bucketize.Scalar_out': bucketize,
}.items():
    _registry.impl(name, function)
