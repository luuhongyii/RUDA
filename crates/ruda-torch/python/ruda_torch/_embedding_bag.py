import torch

from . import _C
from ._ordering import _output, _outputs
from ._pooling import _nondeterministic


def _floating(value):
    if value.layout != torch.strided or value.dtype not in (
            torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise RuntimeError('embedding_bag requires a supported strided floating-point tensor')


def _index(value, reference):
    if (value.layout != torch.strided or value.dtype not in (torch.int32, torch.int64)
            or value.device != reference.device):
        raise RuntimeError('embedding_bag indices and saved indices require int32/int64 on the input device')


def _promote(indices, offsets, reference, *, contiguous=False):
    _index(indices, reference)
    _index(offsets, reference)
    dtype = torch.promote_types(indices.dtype, offsets.dtype)
    indices, offsets = indices.to(dtype), offsets.to(dtype)
    if contiguous and (not indices.is_contiguous() or not offsets.is_contiguous()):
        raise RuntimeError('embedding_bag backward requires contiguous indices and offsets')
    return indices.resolve_neg(), offsets.resolve_neg()


def _mode(mode):
    if mode not in (0, 1, 2):
        raise RuntimeError('embedding_bag mode must be sum (0), mean (1), or max (2)')


def _sample_weights(value, reference, shape, mode):
    if value is None:
        return reference.new_empty((0,))
    if mode != 0:
        raise RuntimeError("embedding_bag per_sample_weights is only supported for mode='sum'")
    if (value.dtype != reference.dtype or value.device != reference.device
            or value.layout != torch.strided or tuple(value.shape) != tuple(shape)):
        raise RuntimeError('embedding_bag per_sample_weights must match indices shape and weight dtype/device')
    return value.resolve_neg()


def embedding_bag(weight, indices, offsets, scale_grad_by_freq=False, mode=0,
                  sparse=False, per_sample_weights=None, include_last_offset=False,
                  padding_idx=-1, *, out0=None, out1=None, out2=None, out3=None):
    _floating(weight)
    _mode(mode)
    if weight.ndim != 2 or indices.ndim not in (1, 2):
        raise RuntimeError('embedding_bag requires a matrix weight and one- or two-dimensional indices')
    if offsets.ndim == 0 or indices.ndim == 1 and offsets.ndim != 1:
        raise RuntimeError('embedding_bag offsets must be one-dimensional for flat indices')
    original = weight, indices, offsets, per_sample_weights
    indices, offsets = _promote(indices, offsets, weight)
    sample_weights = _sample_weights(per_sample_weights, weight, indices.shape, mode)
    count, offset_count = indices.shape[0], offsets.shape[0]
    if include_last_offset and not offset_count:
        raise RuntimeError('include_last_offset requires at least one offset')
    if indices.numel() < count:
        raise RuntimeError('embedding_bag indices storage is smaller than its leading dimension')
    if offsets.numel() < offset_count:
        raise RuntimeError('embedding_bag offsets storage is smaller than its leading dimension')
    bags = offset_count - int(include_last_offset)
    output = weight.new_empty((bags, weight.shape[1]))
    mapping = indices.new_empty((count,))
    sizes = indices.new_empty(offsets.shape)
    maximum = indices.new_empty(output.shape if mode == 2 else (0,))
    flat_indices = indices.contiguous().view(-1)[:count]
    flat_offsets = offsets.contiguous().view(-1)[:offset_count]
    if per_sample_weights is not None and indices.ndim == 2:
        sample_weights = torch.as_strided(sample_weights, (count,), (sample_weights.stride(0),))
    if bags:
        _C.scientific_call(341, [weight.resolve_neg(), flat_indices, flat_offsets,
                               sample_weights, output, mapping,
                               sizes.view(-1)[:offset_count], maximum],
                           [mode, padding_idx, int(scale_grad_by_freq),
                            int(per_sample_weights is not None)], [])
    outputs = None if all(x is None for x in (out0, out1, out2, out3)) else (out0, out1, out2, out3)
    if outputs is not None and any(x is None for x in outputs):
        raise RuntimeError('embedding_bag.out requires all four output tensors')
    return _outputs((output, mapping, sizes, maximum), outputs, original)


def _mapping(indices, offsets, mapping):
    _index(mapping, indices)
    if mapping.ndim != 1:
        raise RuntimeError('embedding_bag offset2bag must be one-dimensional')
    if indices.numel() and not mapping.numel():
        result = indices.new_empty(indices.shape)
        _C.scientific_call(344, [offsets, result], [], [])
        return result
    if mapping.shape != indices.shape:
        raise RuntimeError('embedding_bag offset2bag must have one entry per index')
    if not mapping.is_contiguous():
        raise RuntimeError('embedding_bag backward requires contiguous offset2bag')
    return mapping.to(indices.dtype).resolve_neg()


def dense_backward(grad, indices, offset2bag, bag_size, maximum_indices,
                   num_weights, scale_grad_by_freq, mode, per_sample_weights,
                   padding_idx=-1, *, out=None):
    _floating(grad)
    _mode(mode)
    if grad.ndim != 2 or indices.ndim != 1 or num_weights < 0:
        raise RuntimeError('embedding_bag backward requires a matrix gradient, flat indices and nonnegative row count')
    for value in (indices, offset2bag, bag_size, maximum_indices):
        _index(value, grad)
    sample_weights = _sample_weights(per_sample_weights, grad, indices.shape, mode)
    if mode == 2:
        if grad.dtype == torch.bfloat16:
            raise RuntimeError('embedding_bag max backward does not support bfloat16')
        if maximum_indices.shape != grad.shape:
            raise RuntimeError('embedding_bag maximum_indices must match the gradient shape')
        _nondeterministic('embedding_bag_backward_cuda_max')
        result = grad.new_zeros((num_weights, grad.shape[1]))
        order = indices.new_empty((0,), dtype=torch.int64)
        sorted_indices = indices.resolve_neg()
    else:
        if offset2bag.shape != indices.shape or mode == 1 and bag_size.numel() < grad.shape[0]:
            raise RuntimeError('embedding_bag saved bag mapping or sizes do not match the gradient')
        result = grad.new_empty((num_weights, grad.shape[1]))
        sorted_indices, order = torch.sort(indices.resolve_neg(), stable=True)
    if result.numel():
        _C.scientific_call(342, [grad.resolve_neg(), sorted_indices, order,
                               offset2bag.to(indices.dtype).resolve_neg(),
                               bag_size.to(indices.dtype).resolve_neg(),
                               maximum_indices.to(indices.dtype).resolve_neg(),
                               sample_weights, result],
                           [mode, padding_idx, int(scale_grad_by_freq),
                            int(per_sample_weights is not None)], [])
    return _output(result, out, (grad, indices, offset2bag, bag_size,
                                 maximum_indices, per_sample_weights))


def backward(grad, indices, offsets, offset2bag, bag_size, maximum_indices,
             num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights,
             padding_idx=-1):
    indices, offsets = _promote(indices, offsets, grad, contiguous=True)
    if indices.ndim != 1 or offsets.ndim != 1:
        raise RuntimeError('embedding_bag backward requires flat indices and offsets')
    mapping = _mapping(indices, offsets, offset2bag)
    if sparse:
        return torch.ops.aten._embedding_bag_sparse_backward.default(
            grad, indices, offsets, mapping, bag_size, num_weights,
            scale_grad_by_freq, mode, per_sample_weights, padding_idx)
    return dense_backward(grad, indices, mapping, bag_size, maximum_indices,
                          num_weights, scale_grad_by_freq, mode,
                          per_sample_weights, padding_idx)


def per_sample_backward(grad, weight, indices, offsets, offset2bag, mode,
                        padding_idx=-1, *, out=None):
    _floating(grad)
    _floating(weight)
    if mode != 0:
        raise RuntimeError("embedding_bag per_sample_weights backward requires mode='sum'")
    if (grad.ndim != 2 or weight.ndim != 2 or grad.shape[1] != weight.shape[1]
            or grad.dtype != weight.dtype or grad.device != weight.device):
        raise RuntimeError('embedding_bag gradient and weight must have matching features, dtype and device')
    original = grad, weight, indices, offsets, offset2bag
    indices, offsets = _promote(indices, offsets, grad, contiguous=True)
    if indices.ndim != 1 or offsets.ndim != 1:
        raise RuntimeError('embedding_bag per-sample backward requires flat indices and offsets')
    mapping = _mapping(indices, offsets, offset2bag)
    result = grad.new_empty(indices.shape)
    if result.numel():
        _C.scientific_call(343, [grad.resolve_neg(), weight.resolve_neg(),
                               indices, mapping, result], [0, padding_idx, 0, 0], [])
    return _output(result, out, original)


def register(library):
    for name in ('_embedding_bag', '_embedding_bag_forward_only'):
        library.impl(name, embedding_bag)
        library.impl(name + '.out', embedding_bag)
    library.impl('_embedding_bag_backward', backward)
    for name, function in (('_embedding_bag_dense_backward', dense_backward),
                           ('_embedding_bag_per_sample_weights_backward', per_sample_backward)):
        library.impl(name, function)
        library.impl(name + '.out', function)
