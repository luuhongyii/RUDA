import warnings
import torch
from . import _C
from ._ops import _registry, _dimension, _out, _typed_value, clone, div, fill_, masked_fill, mul


def _logical(a):
    return a.resolve_conj().resolve_neg() if a.is_conj() or a.is_neg() else a


def _target(a, inplace):
    if not inplace:
        return clone(a)
    return clone(a) if a.is_conj() or a.is_neg() else a


def _finish_target(a, result, inplace):
    if inplace and result is not a:
        a.copy_(result)
        return a
    return result


def _indices(a, index, *, empty_dtype=False):
    if index.device != a.device:
        raise RuntimeError("RUDA indices must be on the same device as the tensor")
    if not (empty_dtype and index.numel() == 0) and index.dtype not in (torch.int32, torch.int64):
        raise RuntimeError("RUDA indices must have dtype int32 or int64")


def _nonempty_rank(a):
    return a.view(1) if a.ndim == 0 else a


def _prefix(a, shape, except_axis=None):
    for axis, size in enumerate(shape):
        if axis != except_axis and a.shape[axis] != size:
            a = a.narrow(axis, 0, size)
    return a


def gather(a, dim, index, *, sparse_grad=False, out=None):
    inputs = (a, index)
    a, index = _logical(a), _logical(index)
    axis = _dimension(dim, a.ndim)
    _indices(a, index, empty_dtype=True)
    result = _out(index.shape, a)
    if index.numel() == 0:
        return _index_out(result, out, inputs)
    value, indices = _nonempty_rank(a), _nonempty_rank(index)
    if value.ndim != indices.ndim or any(indices.shape[d] > value.shape[d] for d in range(value.ndim) if d != axis):
        raise RuntimeError("gather index shape must fit the input outside the indexed dimension")
    value = _prefix(value, indices.shape, axis)
    _C.scientific_call(470, [value, indices, _nonempty_rank(result)], [axis, 0], [])
    return _index_out(result, out, inputs)


def index_select(a, dim, index, *, out=None):
    inputs = (a, index)
    a, index = _logical(a), _logical(index)
    axis = _dimension(dim, a.ndim)
    _indices(a, index)
    if index.ndim > 1:
        raise RuntimeError("index_select index must be a scalar or vector")
    if a.ndim == 0 and index.numel() != 1:
        raise RuntimeError("index_select of a scalar requires exactly one index")
    shape = list(a.shape)
    if shape:
        shape[axis] = index.numel()
    result = _out(shape, a)
    _C.scientific_call(470, [_nonempty_rank(a), _nonempty_rank(index), _nonempty_rank(result)], [axis, 1], [])
    return _index_out(result, out, inputs)


def _expanded_indices(a, indices):
    expanded = []
    for index in indices:
        if index is None:
            expanded.append(None)
        else:
            if index.device != a.device and index.device.type != 'cpu':
                raise RuntimeError("indices must be on CPU or on the indexed tensor's device")
            index = _logical(index).to(a.device)
            if index.dtype in (torch.bool, torch.uint8):
                if index.dtype == torch.uint8:
                    warnings.warn("indexing with dtype torch.uint8 is deprecated; use torch.bool", UserWarning)
                axis = len(expanded)
                if index.ndim == 0:
                    a = a.unsqueeze(axis)
                    expanded.append(_nonzero_positions(index))
                else:
                    if tuple(a.shape[axis:axis + index.ndim]) != tuple(index.shape):
                        raise IndexError("mask shape does not match the indexed dimensions")
                    coordinates = nonzero(index)
                    expanded.extend(coordinates[:, dim] for dim in range(index.ndim))
            elif index.dtype in (torch.int32, torch.int64):
                expanded.append(index)
            else:
                raise IndexError("indices must be int32, int64, bool or uint8 tensors")
        if len(expanded) > a.ndim:
            raise IndexError("too many indices for tensor")
    return a, expanded + [None] * (a.ndim - len(expanded))


def _index_offsets(a, indices, *, clamp=False):
    a, indices = _expanded_indices(a, indices)
    axes = [axis for axis, index in enumerate(indices) if index is not None]
    if not axes:
        shape, broadcast_shape = list(a.shape), ()
    else:
        for axis in axes:
            _indices(a, indices[axis])
        broadcast_shape = torch.broadcast_shapes(*(indices[axis].shape for axis in axes))
        contiguous = axes == list(range(axes[0], axes[-1] + 1))
        remaining = [axis for axis in range(a.ndim) if axis not in axes]
        before = [axis for axis in remaining if axis < axes[0]] if contiguous else []
        after = [axis for axis in remaining if axis not in before]
        shape = [a.shape[axis] for axis in before] + list(broadcast_shape) + [a.shape[axis] for axis in after]
    linear = torch.empty(shape, device=a.device, dtype=torch.int64)
    expanded = [indices[axis].expand(broadcast_shape) for axis in axes]
    _C.scientific_call(475, [linear, *expanded], [a.ndim, len(broadcast_shape), int(clamp), *a.shape, *axes], [])
    return linear


def _index_out(result, out, inputs, *, allow_self=False):
    if out is not None:
        if out.device != result.device or out.dtype != result.dtype:
            raise RuntimeError("index output must match the input device and dtype")
        for position, value in enumerate(inputs):
            if value is not None:
                if allow_self and position == 0:
                    _C.check_inplace(out, value)
                else:
                    _C.check_index_output(out, value, value)
        if out.shape != result.shape:
            _C.prepare_index_output(out, list(result.shape))
        out.copy_(result)
        return out
    return result


def index_tensor(a, indices, *, out=None):
    linear = _index_offsets(a, indices)
    result = _out(linear.shape, a)
    _C.scientific_call(473, [_logical(a), linear, result], [0], [])
    return _index_out(result, out, (a, *indices))


def _integer_indices(indices):
    for index in indices:
        if index is not None and index.dtype not in (torch.int32, torch.int64):
            raise RuntimeError("unsafe index requires int32 or int64 indices")


def unsafe_index(a, indices):
    _integer_indices(indices)
    return index_tensor(a, indices)


def unsafe_masked_index(a, mask, indices, fill):
    _integer_indices(indices)
    linear = _index_offsets(a, indices, clamp=True)
    if a.numel() == 0:
        return fill_(_out(linear.shape, a), fill)
    linear, expanded_mask = _mask(linear, mask)
    result = _out(linear.shape, a)
    scalar = _logical(_typed_value(fill, a.dtype, a.device))
    _C.scientific_call(476, [_logical(a), linear, result, _logical(expanded_mask), scalar], [], [])
    return result


def unsafe_masked_index_put_accumulate(a, mask, indices, values):
    if a.numel() == 0:
        return clone(a)
    _integer_indices(indices)
    if values.dtype != a.dtype:
        raise RuntimeError("index_put values must match the input dtype")
    if values.device != a.device:
        if values.device.type != 'cpu' or values.numel() != 1:
            raise RuntimeError("index_put values must be on the input device (except a CPU scalar)")
        values = values.to(a.device)
    linear = _index_offsets(a, indices, clamp=True)
    values, expanded_mask = _mask(_logical(values), mask)
    while values.ndim > linear.ndim and values.shape[0] == 1:
        values, expanded_mask = values.squeeze(0), expanded_mask.squeeze(0)
    values, expanded_mask = values.expand(linear.shape), expanded_mask.expand(linear.shape)
    result = clone(a)
    _C.scientific_call(477, [values, linear, result, _logical(expanded_mask)], [1, 0], [])
    return result


def _nonzero_prefix(a):
    if a.numel() > 2**32 - 1:
        raise RuntimeError('mask compaction exceeds native U32 indexing')
    prefix = torch.empty((a.numel(),), dtype=torch.int32, device=a.device)
    if not a.numel():
        return prefix, 0
    total = torch.empty((1,), dtype=torch.int64, device=a.device)
    _C.scientific_call(463, [_logical(a), prefix, total], [], [])
    return prefix, int(total.item())


def _nonzero_positions(a):
    a = _logical(a)
    prefix, count = _nonzero_prefix(a)
    positions = torch.empty((count,), dtype=torch.int64, device=a.device)
    if count:
        _C.scientific_call(464, [a, prefix, positions], [a.ndim, 1], [])
    return positions


def nonzero(a, *, out=None):
    value = _logical(a)
    prefix, count = _nonzero_prefix(value)
    result = torch.empty((a.ndim, count), dtype=torch.int64, device=a.device).t()
    if count and a.ndim:
        _C.scientific_call(464, [value, prefix, result], [a.ndim, 0], [])
    return _index_out(result, out, (a,))


def nonzero_numpy(a):
    result = nonzero(_nonempty_rank(a))
    return tuple(result[:, axis] for axis in range(result.shape[1]))


def _mask(a, mask):
    if mask.dtype != torch.bool or mask.device != a.device:
        raise RuntimeError("mask must be boolean and on the input device")
    return torch.broadcast_tensors(a, mask)


def masked_select(a, mask, *, out=None):
    value, expanded = _mask(a, mask)
    prefix, count = _nonzero_prefix(expanded)
    result = torch.empty((count,), dtype=a.dtype, device=a.device)
    if count:
        _C.scientific_call(465, [_logical(value), prefix, result], [count, 0], [])
    return _index_out(result, out, (a, mask))


def _flat_update(result, indices, values, accumulate, *, negative=False):
    if not indices.numel():
        return result
    _C.scientific_call(474, [_logical(values), _logical(indices), result],
                       [int(accumulate), int(negative)], [])
    return result


def masked_scatter(a, mask, source, *, inplace=False, out=None):
    value, expanded = _mask(a, mask)
    if source.dtype != a.dtype or source.device != a.device:
        raise RuntimeError("masked_scatter source must match the input dtype and device")
    if inplace:
        if value.shape != a.shape:
            raise RuntimeError("in-place masked_scatter cannot change the input shape")
        _C.check_index_output(a, source, mask)
    prefix, count = _nonzero_prefix(expanded)
    if source.numel() < count:
        raise RuntimeError("masked_scatter source has fewer elements than the number of true mask entries")
    result = _target(a if inplace else value, inplace)
    if count:
        _C.scientific_call(466, [prefix, _logical(source), result], [count], [])
    result = _finish_target(a, result, inplace)
    return _index_out(result, out, (a, mask, source), allow_self=True)


def masked_scatter_backward(grad, mask, sizes):
    value, expanded = _mask(grad, mask)
    prefix, count = _nonzero_prefix(expanded)
    result = torch.empty(sizes, dtype=grad.dtype, device=grad.device)
    if result.numel() < count:
        raise RuntimeError('masked_scatter_backward output has fewer elements than the number of true mask entries')
    if result.numel():
        _C.scientific_call(465, [_logical(value), prefix, result], [count, 1], [])
    return result


def _flat_indices(a, index):
    if index.dtype != torch.int64 or index.device != a.device:
        raise RuntimeError("flat indices must have dtype int64 on the input device")
    return _logical(index)


def take(a, index, *, out=None):
    indices = _flat_indices(a, index)
    result = _out(index.shape, a)
    _C.scientific_call(473, [_logical(a), indices, result], [1], [])
    return _index_out(result, out, (a, index))


def put(a, index, source, accumulate=False, *, inplace=False, out=None):
    indices = _flat_indices(a, index)
    if source.dtype != a.dtype or source.device != a.device or source.numel() != index.numel():
        raise RuntimeError("put source must match the input dtype/device and number of indices")
    if inplace:
        _C.check_index_output(a, source, index)
    result = _flat_update(_target(a, inplace), indices, source, accumulate, negative=True)
    result = _finish_target(a, result, inplace)
    return _index_out(result, out, (a, index, source), allow_self=True)


def index_copy(a, dim, index, source, *, inplace=False, out=None):
    axis = _dimension(dim, a.ndim)
    if index.dtype != torch.int64 or index.ndim > 1 or index.device != a.device:
        raise RuntimeError("index_copy requires a scalar or vector int64 index on the input device")
    if source.dtype != a.dtype or source.device != a.device:
        raise RuntimeError("index_copy source must match the input dtype and device")
    value, target = _nonempty_rank(_logical(source)), _nonempty_rank(a)
    if value.ndim != target.ndim or value.shape[axis] != index.numel():
        raise RuntimeError("index_copy source dimension must match the number of indices")
    if any(value.shape[d] != target.shape[d] for d in range(target.ndim) if d != axis):
        raise RuntimeError("index_copy source shape must match outside the indexed dimension")
    if inplace:
        _C.check_index_output(a, source, index)
    result = _target(a, inplace)
    _C.scientific_call(472, [value, _nonempty_rank(_logical(index)), _nonempty_rank(result)], [axis, 1], [])
    result = _finish_target(a, result, inplace)
    return _index_out(result, out, (a, index, source), allow_self=True)


def index_fill(a, dim, index, value, *, inplace=False, out=None):
    axis = _dimension(dim, a.ndim)
    if index.dtype != torch.int64 or index.ndim > 1 or index.device != a.device:
        raise RuntimeError("index_fill requires a scalar or vector int64 index on the input device")
    original_value = value if isinstance(value, torch.Tensor) else None
    if not a.is_complex() and (value.is_complex() if original_value is not None else isinstance(value, complex)):
        raise RuntimeError("index_fill cannot convert a complex value to a non-complex dtype")
    if original_value is not None:
        if value.ndim != 0:
            raise RuntimeError("index_fill only supports a zero-dimensional value tensor")
        if value.device.type == 'cpu':
            value = value.to(a.device)
    if inplace:
        _C.check_index_output(a, index, index)
        if original_value is not None:
            _C.check_index_output(a, original_value, index)
    source = _logical(_typed_value(value, a.dtype, a.device))
    result = _target(a, inplace)
    _C.scientific_call(472, [source, _nonempty_rank(_logical(index)), _nonempty_rank(result)], [axis, 2], [])
    result = _finish_target(a, result, inplace)
    return _index_out(result, out, (a, index, original_value), allow_self=True)


def index_put(a, indices, values, accumulate=False, *, inplace=False, out=None, unsafe=False):
    if values.dtype != a.dtype:
        raise RuntimeError("index_put values must match the input dtype")
    original_values = values
    if values.device != a.device:
        if values.device.type != 'cpu' or values.numel() != 1:
            raise RuntimeError("index_put values must be on the input device (except a CPU scalar)")
        values = values.to(a.device)
    if inplace:
        _C.check_index_output(a, original_values, original_values)
        for index in indices:
            if index is not None:
                _C.check_index_output(a, index, index)
    linear = _index_offsets(a, indices)
    values = _logical(values)
    while values.ndim > linear.ndim and values.shape[0] == 1:
        values = values.squeeze(0)
    values = values.expand(linear.shape)
    result = _flat_update(_target(a, inplace), linear, values, accumulate)
    result = _finish_target(a, result, inplace)
    return _index_out(result, out, (a, *indices, original_values), allow_self=True)


def _scatter(a, dim, index, source, *, inplace=False, assign=False):
    axis = _dimension(dim, a.ndim)
    _indices(a, index, empty_dtype=True)
    tensor_source = isinstance(source, torch.Tensor)
    if tensor_source and (source.device != a.device or source.dtype != a.dtype):
        raise RuntimeError("scatter source must match the destination device and dtype")
    if inplace:
        _C.check_index_output(a, source if tensor_source else index, index)
    result = _target(a, inplace)
    if index.numel() == 0:
        return a if inplace else result
    indices, target = _nonempty_rank(_logical(index)), _nonempty_rank(result)
    value = (_nonempty_rank(_logical(source)) if tensor_source else
             _typed_value(source, a.dtype, a.device).expand(indices.shape))
    if indices.ndim != target.ndim or indices.ndim != value.ndim:
        raise RuntimeError("scatter index, source and destination must have matching ranks")
    if any(indices.shape[d] > value.shape[d] or d != axis and indices.shape[d] > target.shape[d]
           for d in range(indices.ndim)):
        raise RuntimeError("scatter index shape exceeds the source or destination")
    value = _prefix(value, indices.shape)
    target = _prefix(target, indices.shape, axis)
    if assign:
        _C.scientific_call(472, [value, indices, target], [axis, 0], [])
    else:
        _C.scientific_call(471, [value, indices, target], [axis, 0], [])
    return _finish_target(a, result, inplace)


def scatter_add(a, dim, index, source, *, inplace=False, out=None):
    result = _scatter(a, dim, index, source, inplace=inplace)
    return _index_out(result, out, (a, index, source), allow_self=True)


def scatter(a, dim, index, source, *, inplace=False):
    return _scatter(a, dim, index, source, inplace=inplace, assign=True)


def index_add(a, dim, index, source, *, alpha=1, inplace=False, out=None):
    axis = _dimension(dim, a.ndim)
    _indices(a, index)
    if index.ndim > 1:
        raise RuntimeError("index_add index must be a scalar or vector")
    if source.device != a.device or source.dtype != a.dtype:
        raise RuntimeError("index_add source must match the destination device and dtype")
    value, target = _nonempty_rank(_logical(source)), _nonempty_rank(a)
    if value.ndim != target.ndim or value.shape[axis] != index.numel():
        raise RuntimeError("index_add source dimension must match the number of indices")
    if any(value.shape[d] != target.shape[d] for d in range(target.ndim) if d != axis):
        raise RuntimeError("index_add source shape must match outside the indexed dimension")
    if inplace:
        _C.check_index_output(a, source, index)
    result = _target(a, inplace)
    scale = _logical(_typed_value(alpha, a.dtype, a.device))
    _C.scientific_call(471, [value, _nonempty_rank(_logical(index)), _nonempty_rank(result), scale], [axis, 1], [])
    result = _finish_target(a, result, inplace)
    return _index_out(result, out, (a, index, source), allow_self=True)


def embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False, *, out=None):
    if weight.ndim != 2:
        raise RuntimeError("embedding weight must be two-dimensional")
    _indices(weight, indices)
    result = torch.empty((*indices.shape, weight.shape[1]), dtype=weight.dtype, device=weight.device)
    _C.scientific_call(468, [_logical(weight), _logical(indices), result], [], [])
    return _index_out(result, out, (weight, indices))


def embedding_dense_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq, *, out=None):
    _indices(grad, indices)
    if num_weights < 0 or grad.ndim != indices.ndim + 1 or grad.shape[:-1] != indices.shape:
        raise RuntimeError("invalid embedding gradient shape")
    if grad.dtype not in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
        raise RuntimeError("embedding gradients require a supported floating dtype")
    width = grad.shape[-1]
    result = torch.empty((num_weights, width), dtype=grad.dtype, device=grad.device)
    _C.scientific_call(469, [_logical(grad), _logical(indices), result], [padding_idx, int(scale_grad_by_freq)], [])
    return _index_out(result, out, (grad, indices))


for name, function in {
    "gather": gather, "gather.out": gather,
    "index_select": index_select, "index_select.out": index_select,
    "index.Tensor": index_tensor,
    "index.Tensor_out": index_tensor,
    "_unsafe_index.Tensor": unsafe_index,
    "_unsafe_index_put": index_put,
    "_unsafe_masked_index": unsafe_masked_index,
    "_unsafe_masked_index_put_accumulate": unsafe_masked_index_put_accumulate,
    "nonzero": nonzero, "nonzero.out": nonzero, "nonzero_numpy": nonzero_numpy,
    "masked_select": masked_select, "masked_select.out": masked_select,
    "masked_scatter": masked_scatter, "masked_scatter.out": masked_scatter,
    "masked_scatter_": lambda a, mask, source: masked_scatter(a, mask, source, inplace=True),
    "masked_scatter_backward": masked_scatter_backward,
    "take": take, "take.out": take,
    "put": put, "put.out": put,
    "put_": lambda a, index, source, accumulate=False: put(a, index, source, accumulate, inplace=True),
    "index_copy": index_copy, "index_copy.out": index_copy,
    "index_copy_": lambda a, dim, index, source: index_copy(a, dim, index, source, inplace=True),
    "index_fill.int_Scalar": index_fill, "index_fill.int_Tensor": index_fill,
    "index_fill.int_Scalar_out": index_fill, "index_fill.int_Tensor_out": index_fill,
    "index_fill_.int_Scalar": lambda a, dim, index, value: index_fill(a, dim, index, value, inplace=True),
    "index_fill_.int_Tensor": lambda a, dim, index, value: index_fill(a, dim, index, value, inplace=True),
    "index_put": index_put, "index_put.out": index_put,
    "index_put_": lambda a, indices, values, accumulate=False: index_put(a, indices, values, accumulate, inplace=True),
    "_index_put_impl_": lambda a, indices, values, accumulate=False, unsafe=False: index_put(a, indices, values, accumulate, inplace=True, unsafe=unsafe),
    "scatter_add": scatter_add, "scatter_add.out": scatter_add,
    "scatter_add_": lambda a, dim, index, source: scatter_add(a, dim, index, source, inplace=True),
    "scatter.src": scatter,
    "scatter.value": scatter,
    "scatter_.src": lambda a, dim, index, source: scatter(a, dim, index, source, inplace=True),
    "scatter_.value": lambda a, dim, index, value: scatter(a, dim, index, value, inplace=True),
    "index_add": index_add, "index_add.out": index_add,
    "index_add_": lambda a, dim, index, source, alpha=1: index_add(a, dim, index, source, alpha=alpha, inplace=True),
    "embedding": embedding, "embedding.out": embedding,
    "embedding_dense_backward": embedding_dense_backward, "embedding_dense_backward.out": embedding_dense_backward,
}.items():
    _registry.impl(name, function)

from . import _unique
