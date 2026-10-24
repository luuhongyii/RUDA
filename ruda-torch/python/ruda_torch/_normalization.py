import math
import torch
from torch._prims_common import suggest_memory_format
from . import _C
from ._ops import _dtypes
from ._ordering import _output


def _acc(a):
    if a.dtype not in _dtypes:
        raise RuntimeError('normalization requires real floating input')
    return torch.float64 if a.dtype == torch.float64 else torch.float32


def _params(a, shape, *values, mixed=False):
    _acc(a)
    defined = [v for v in values if v is not None]
    dtype = defined[0].dtype if defined else a.dtype
    allowed = (a.dtype, torch.float32) if mixed and a.dtype in (torch.float16, torch.bfloat16) else (a.dtype,)
    if dtype not in allowed:
        raise RuntimeError('normalization parameter dtype does not match the input')
    for v in defined:
        if v.device != a.device or v.dtype != dtype or tuple(v.shape) != tuple(shape):
            raise RuntimeError('normalization parameters must have matching dtype, shape and device')
    return dtype


def _parameter(value, a):
    return a.new_empty((0,), dtype=_acc(a)) if value is None else value.resolve_neg().to(_acc(a)).reshape(-1)


def _stats(x, eps, kind, vectorized=False):
    mean, var, inv = (x.new_empty((x.shape[0],), dtype=_acc(x)) for _ in range(3))
    _C.scientific_call(309, [x, mean, var, inv], [kind, int(vectorized)], [float(eps)])
    return mean, var, inv


def _affine(x, mean, inv, weight, bias, kind, channels=0, spatial=1, groups=1):
    result = torch.empty_like(x)
    _C.scientific_call(310, [x, mean, inv, _parameter(weight, x), _parameter(bias, x), result],
                       [kind, channels, spatial, groups, int(weight is not None), int(bias is not None)], [])
    return result


def _saved(value, a, rows, dtype):
    if value.device != a.device or value.dtype != dtype or value.numel() != rows:
        raise RuntimeError('invalid saved normalization statistics')
    return value.resolve_neg().to(_acc(a)).reshape(-1)


def _gradient(grad, a):
    if grad.shape != a.shape or grad.device != a.device or grad.dtype != a.dtype:
        raise RuntimeError('normalization gradient must match input shape, dtype and device')


def _back(x, grad, mean, inv, weight, mask, kind, features, *, channels=0, spatial=1, groups=1, training=True):
    if len(mask) != 3:
        raise RuntimeError('normalization backward requires three output mask entries')
    dx = dw = db = None
    if mask[0]:
        dx = torch.empty_like(x)
        _C.scientific_call(311, [x, grad, mean, inv, _parameter(weight, x), dx],
                           [kind, channels, spatial, groups, int(weight is not None), int(training)], [])
    if mask[1] or mask[2]:
        dw, db = (x.new_empty((features,), dtype=_acc(x)) for _ in range(2))
        _C.scientific_call(312, [x, grad, mean, inv, dw, db], [kind, channels, spatial, groups, 0, 0], [])
    return dx, dw if mask[1] else None, db if mask[2] else None


def _write(values, outputs, inputs):
    if all(v is None for v in outputs):
        return values
    if any(v is None for v in outputs):
        raise RuntimeError('all normalization outputs must be provided together')
    for i, out in enumerate(outputs):
        for other in outputs[i + 1:]:
            _C.check_index_output(out, other, other)
    return tuple(out if value is None else _output(value, out, inputs) for value, out in zip(values, outputs))


def _layer_shape(a, normalized_shape, weight, bias=None):
    shape = tuple(normalized_shape)
    if not shape or len(shape) > a.ndim or tuple(a.shape[-len(shape):]) != shape:
        raise RuntimeError('normalized_shape must match the trailing input dimensions')
    _params(a, shape, weight, bias)
    rows = math.prod(a.shape[:-len(shape)])
    return shape, rows, math.prod(shape), (*a.shape[:-len(shape)], *((1,) * len(shape)))


def _training_norm(a, shape, *parameters):
    from . import _training_available
    return (_training_available and len(shape) == 1 and shape[0] > 0
            and a.dtype in (torch.float32, torch.float16, torch.bfloat16)
            and all(v is None or v.is_contiguous() and not v.is_neg() and not v.is_conj()
                    for v in (a, *parameters)))


def layer_norm(a, normalized_shape, weight, bias, eps, *, out0=None, out1=None, out2=None):
    shape, rows, width, stat_shape = _layer_shape(a, normalized_shape, weight, bias)
    if (_training_norm(a, shape, weight, bias) and math.isfinite(eps)
            and torch.finfo(torch.float32).tiny <= eps <= torch.finfo(torch.float32).max):
        value, mean, inv = _C.training_layer_forward(a, weight, bias, eps)
        return _write((value, mean.reshape(stat_shape), inv.reshape(stat_shape)),
                      (out0, out1, out2), (a, weight, bias))
    x = a.resolve_neg().reshape(rows, width)
    vectorized = a.dtype != torch.float64 and width <= 2**24 and width % 4 == 0 and all(
        v is None or not v.is_contiguous() or v.data_ptr() % (4 * v.element_size()) == 0 for v in (x, weight, bias))
    mean, _, inv = _stats(x, eps, 0, vectorized)
    value = _affine(x, mean, inv, weight, bias, 0)
    return _write((value.reshape(a.shape), mean.reshape(stat_shape), inv.reshape(stat_shape)), (out0, out1, out2), (a, weight, bias))


def layer_backward(grad, a, normalized_shape, mean, rstd, weight, bias, output_mask, *, out0=None, out1=None, out2=None):
    shape, rows, width, _ = _layer_shape(a, normalized_shape, weight, bias)
    _gradient(grad, a)
    if output_mask[1] and weight is None or output_mask[2] and bias is None:
        raise RuntimeError('requested gradient for an undefined affine parameter')
    mu, inv = (_saved(v, a, rows, _acc(a)) for v in (mean, rstd))
    if _training_norm(a, shape, weight, bias):
        values = _C.training_layer_backward(a, weight, bias, grad.resolve_neg().contiguous(),
                                            mu, inv, *output_mask)
        return _write(values, (out0, out1, out2), (a, grad, weight, bias, mean, rstd))
    dx, dw, db = _back(a.resolve_neg().reshape(rows, width), grad.resolve_neg().reshape(rows, width), mu, inv, weight, output_mask, 0, width)
    values = (None if dx is None else dx.reshape(a.shape), None if dw is None else dw.to(weight.dtype).reshape(shape), None if db is None else db.to(bias.dtype).reshape(shape))
    return _write(values, (out0, out1, out2), (a, grad, weight, bias, mean, rstd))


def rms_norm(a, normalized_shape, weight, eps):
    shape, rows, width, stat_shape = _layer_shape(a, normalized_shape, weight)
    epsilon = torch.finfo(_acc(a)).eps if eps is None else float(eps)
    if (_training_norm(a, shape, weight) and math.isfinite(epsilon)
            and torch.finfo(torch.float32).tiny <= epsilon <= torch.finfo(torch.float32).max):
        value, inv = _C.training_rms_forward(a, weight, epsilon)
        return value, inv.reshape(stat_shape)
    x = a.resolve_neg().reshape(rows, width)
    mean, _, inv = _stats(x, epsilon, 3)
    value = _affine(x, mean, inv, weight, None, 3)
    return value.reshape(a.shape), inv.reshape(stat_shape)


def rms_norm_legacy(a, normalized_shape_ndim, weight, eps):
    if not 0 <= normalized_shape_ndim <= a.ndim:
        raise RuntimeError('normalized_shape_ndim must be between zero and the input rank')
    if not a.is_contiguous() or not weight.is_contiguous():
        raise RuntimeError('Expected contiguous input and weight tensors')
    if normalized_shape_ndim == 0:
        value, _ = rms_norm(a.unsqueeze(-1), (1,), weight.reshape(1), eps)
        return value.reshape(a.shape)
    shape = tuple(a.shape[a.ndim - normalized_shape_ndim:])
    value, _ = rms_norm(a, shape, weight, eps)
    return value


def rms_backward(grad, a, normalized_shape, rstd, weight, output_mask):
    shape, rows, width, _ = _layer_shape(a, normalized_shape, weight)
    _gradient(grad, a)
    if len(output_mask) != 2 or output_mask[1] and weight is None:
        raise RuntimeError('invalid RMS normalization gradient mask')
    inv = _saved(rstd, a, rows, _acc(a))
    if _training_norm(a, shape, weight):
        return _C.training_rms_backward(a, weight, grad.resolve_neg().contiguous(), inv, *output_mask)
    mean = torch.zeros_like(inv)
    dx, dw, _ = _back(a.resolve_neg().reshape(rows, width), grad.resolve_neg().reshape(rows, width), mean, inv, weight, (*output_mask, False), 3, width)
    return None if dx is None else dx.reshape(a.shape), None if dw is None else dw.to(weight.dtype).reshape(shape)


def _group_shape(a, weight, bias, N, C, HxW, group):
    if a.ndim < 2 or N != a.shape[0] or C != a.shape[1] or HxW != math.prod(a.shape[2:]) or group <= 0 or C % group:
        raise RuntimeError('invalid group normalization shape or group count')
    if not a.is_contiguous(memory_format=suggest_memory_format(a)):
        raise RuntimeError('native_group_norm requires contiguous input storage')
    _params(a, (C,), weight, bias)
    return N * group, C // group * HxW


def group_norm(a, weight, bias, N, C, HxW, group, eps, *, out0=None, out1=None, out2=None):
    rows, width = _group_shape(a, weight, bias, N, C, HxW, group)
    x = a.resolve_neg().reshape(rows, width)
    mean, _, inv = _stats(x, eps, 1)
    value = _affine(x, mean, inv, weight, bias, 1, C, HxW, group).reshape(a.shape)
    return _write((value, mean.to(a.dtype).reshape(N, group), inv.to(a.dtype).reshape(N, group)), (out0, out1, out2), (a, weight, bias))


def group_backward(grad, a, mean, rstd, weight, N, C, HxW, group, output_mask, *, out0=None, out1=None, out2=None):
    rows, width = _group_shape(a, weight, None, N, C, HxW, group)
    _gradient(grad, a)
    if (output_mask[1] or output_mask[2]) and weight is None:
        raise RuntimeError('requested group affine gradients without a weight')
    mu, inv = (_saved(v, a, rows, a.dtype) for v in (mean, rstd))
    dx, dw, db = _back(a.resolve_neg().reshape(rows, width), grad.resolve_neg().reshape(rows, width), mu, inv, weight, output_mask, 1, C, channels=C, spatial=HxW, groups=group)
    values = (None if dx is None else dx.reshape(a.shape), None if dw is None else dw.to(weight.dtype), None if db is None else db.to(weight.dtype))
    return _write(values, (out0, out1, out2), (a, grad, weight, mean, rstd))


def _batch_shape(a, weight=None, bias=None, running_mean=None, running_var=None):
    if a.ndim < 2 or a.shape[1] == 0:
        raise RuntimeError('batch normalization requires a nonempty channel dimension')
    dtype = _params(a, (a.shape[1],), weight, bias, running_mean, running_var, mixed=True)
    if (running_mean is None) != (running_var is None):
        raise RuntimeError('running_mean and running_var must both be present or absent')
    return a.shape[1], a.numel() // a.shape[1], dtype


def _batch_matrix(a):
    return a.resolve_neg().movedim(1, 0).reshape(a.shape[1], a.numel() // a.shape[1])


def _batch_restore(value, a):
    value = value.reshape(a.shape[1], a.shape[0], *a.shape[2:]).movedim(0, 1)
    result = torch.empty_like(a, memory_format=suggest_memory_format(a))
    return result.copy_(value)


def _batch_channels_last(a):
    return a.numel() < 2**31 and (a.is_contiguous(memory_format=torch.channels_last)
        or a.is_contiguous(memory_format=torch.channels_last_3d)
        or a.is_contiguous() and a.stride(1) == 1)


def _update(mean, var, running_mean, running_var, count, momentum, a):
    if running_mean is None:
        return
    _C.check_index_output(running_mean, a, running_var)
    _C.check_index_output(running_var, a, running_mean)
    correction = count / (count - 1) if count != 1 else float('inf')
    _C.scientific_call(313, [mean, var, running_mean, running_var], [], [float(momentum), correction])


def batch_norm(a, weight, bias, running_mean, running_var, training, momentum, eps, *, out=None, save_mean=None, save_invstd=None):
    channels, count, _ = _batch_shape(a, weight, bias, running_mean, running_var)
    if not training and running_mean is None:
        raise RuntimeError('batch normalization inference requires running statistics')
    x = _batch_matrix(a)
    if training:
        mean, var, inv = _stats(x, eps, 2)
        if count:
            _update(mean, var, running_mean, running_var, count, momentum, a)
    else:
        mean = running_mean.resolve_neg().to(_acc(a)).clone()
        inv = (running_var.resolve_neg().to(_acc(a)) + eps).rsqrt()
    value = _batch_restore(_affine(x, mean, inv, weight, bias, 2, channels), a)
    return _write((value, mean, inv), (out, save_mean, save_invstd), (a, weight, bias, running_mean, running_var))


def batch_backward(grad, a, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask, *, out0=None, out1=None, out2=None):
    channels, count, dtype = _batch_shape(a, weight, None, running_mean, running_var)
    _gradient(grad, a)
    if save_mean is not None and save_invstd is not None and save_mean.numel() and save_invstd.numel():
        mean, inv = (_saved(v, a, channels, _acc(a)) for v in (save_mean, save_invstd))
    elif not train and running_mean is not None:
        mean, inv = running_mean.to(_acc(a)), (running_var.to(_acc(a)) + eps).rsqrt()
    elif count == 0:
        mean = a.new_zeros((channels,), dtype=_acc(a));inv = mean
    else:
        raise RuntimeError('batch normalization backward requires saved or running statistics')
    channels_last_reduce = (train or any(output_mask[1:])) and _batch_channels_last(a) and _batch_channels_last(grad) and (weight is None or weight.is_contiguous()) and mean.is_contiguous() and inv.is_contiguous()
    mask = (output_mask[0], weight is not None, weight is not None) if channels_last_reduce else output_mask
    if count and not channels_last_reduce and (train or any(output_mask[1:])):
        x, dy = _batch_matrix(a), _batch_matrix(grad)
        dx = torch.empty_like(x) if mask[0] else a.new_empty((0,))
        dw, db = (a.new_empty((channels,), dtype=_acc(a)) for _ in range(2))
        spatial = math.prod(a.shape[2:])
        bx = min(512, max(32, 1 << (spatial - 1).bit_length()))
        fused = mask[0] and not _batch_channels_last(a) and a.numel() < 2**31 and grad.numel() < 2**31
        by = 512 // bx if fused else min(1 << max(0, a.shape[0].bit_length() - 1), 16)
        if not fused:
            bx = min(bx, 512 // by)
        _C.scientific_call(314, [x, dy, mean, inv, _parameter(weight, a), dx, dw, db],
                           [spatial, bx, by, int(train), int(weight is not None), int(mask[0])], [])
        dx, dw, db = dx if mask[0] else None, dw if mask[1] else None, db if mask[2] else None
    else:
        dx, dw, db = _back(_batch_matrix(a), _batch_matrix(grad), mean, inv, weight, mask, 2, channels, channels=channels, training=train)
    if channels_last_reduce and weight is None:
        dw, db = (a.new_empty((0,), dtype=_acc(a)) for _ in range(2))
        dtype = _acc(a)
    values = (None if dx is None else _batch_restore(dx, a), None if dw is None else dw.to(dtype), None if db is None else db.to(dtype))
    return _write(values, (out0, out1, out2), (a, grad, weight, running_mean, running_var, save_mean, save_invstd))


def batch_stats(a, eps, *, out0=None, out1=None):
    _batch_shape(a)
    mean, _, inv = _stats(_batch_matrix(a), eps, 4)
    return _write((mean, inv), (out0, out1), (a,))


def batch_elemt(a, weight, bias, mean, invstd, eps, *, out=None):
    channels, _, _ = _batch_shape(a, weight, bias)
    mu, inv = (_saved(v, a, channels, _acc(a)) for v in (mean, invstd))
    value = _batch_restore(_affine(_batch_matrix(a), mu, inv, weight, bias, 2, channels), a)
    return _output(value, out, (a, weight, bias, mean, invstd))


def batch_update_stats(a, running_mean, running_var, momentum, *, out0=None, out1=None):
    _, count, _ = _batch_shape(a, None, None, running_mean, running_var)
    if not count:
        raise RuntimeError('batch_norm_update_stats input must have at least one element')
    mean, var, _ = _stats(_batch_matrix(a), 0., 2)
    _update(mean, var, running_mean, running_var, count, momentum, a)
    return _write((mean, var), (out0, out1), (a, running_mean, running_var))


def batch_gather_counts(a, mean, invstd, running_mean, running_var, momentum, eps, counts, *, out0=None, out1=None):
    _acc(a)
    dtype = a.dtype if running_mean is None else running_mean.dtype
    if dtype not in _dtypes:
        raise RuntimeError('batch normalization running statistics require real floating storage')
    acc = torch.float64 if dtype == torch.float64 else torch.float32
    if mean.ndim != 2 or invstd.shape != mean.shape or mean.dtype != acc or invstd.dtype != acc:
        raise RuntimeError('gathered batch statistics must have matching [ranks, channels] shapes and accumulator dtype')
    ranks, channels = mean.shape
    if counts.shape != (ranks,) or counts.dtype != dtype:
        raise RuntimeError('batch normalization counts must match rank count and running-statistic dtype')
    for value in (mean, invstd, counts, running_mean, running_var):
        if value is not None and value.device != a.device:
            raise RuntimeError('gathered batch statistics must be on the input device')
    for value in (running_mean, running_var):
        if value is not None and (value.dtype != dtype or value.shape != (channels,)):
            raise RuntimeError('running batch statistics must match channel count and dtype')
    inputs = (a, mean, invstd, running_mean, running_var, counts)
    for value, other in ((running_mean, running_var), (running_var, running_mean)):
        if value is not None:
            _C.check_index_output(value, mean, invstd)
            _C.check_index_output(value, counts, counts if other is None else other)
    rm = a.new_empty((0,), dtype=dtype) if running_mean is None else running_mean
    rv = a.new_empty((0,), dtype=dtype) if running_var is None else running_var
    saved_mean, saved_inv = (a.new_empty((channels,), dtype=acc) for _ in range(2))
    span = 0 if not a.numel() else sum((n - 1) * s for n, s in zip(a.shape, a.stride()))
    _C.scientific_call(315, [mean.resolve_neg(), invstd.resolve_neg(), counts.resolve_neg(), rm, rv, saved_mean, saved_inv],
                       [int(a.numel() >= 2**31 or span >= 2**31)], [float(eps), float(momentum)])
    return _write((saved_mean, saved_inv), (out0, out1), inputs)


def batch_gather(a, mean, invstd, running_mean, running_var, momentum, eps, count, *, out0=None, out1=None):
    if mean.ndim != 2:
        raise RuntimeError('gathered batch statistics must have [ranks, channels] shape')
    dtype = a.dtype if running_mean is None else running_mean.dtype
    counts = torch.full((mean.shape[0],), count, dtype=torch.int64, device=a.device).to(dtype)
    return batch_gather_counts(a, mean, invstd, running_mean, running_var, momentum, eps, counts, out0=out0, out1=out1)


def _batch_reduce_config(a, channels_last):
    spatial = math.prod(a.shape[2:])
    floor_power = lambda n: 1 << max(0, n.bit_length() - 1)
    if channels_last:
        count = a.numel() // a.shape[1]
        bx = min(floor_power(a.shape[1]), 32)
        by = min(floor_power((count + 15) // 16), 512 // bx)
        if bx * by != 512:
            bx = min(floor_power(a.shape[1]), 512 // by)
        parts = min((count + by * 16 - 1) // (by * 16), 128)
        parts = 1 if parts < 8 else parts
    else:
        by = min(floor_power(a.shape[0]), 16)
        bx = min(512, max(32, 1 << max(0, (spatial - 1).bit_length())), 512 // by)
        parts = 1
    return spatial, bx, by, parts


def batch_backward_reduce(grad, a, mean, invstd, weight, input_g, weight_g, bias_g, *, out0=None, out1=None, out2=None, out3=None):
    channels, count, dtype = _batch_shape(a, weight)
    _gradient(grad, a)
    if count == 0:
        raise RuntimeError('batch normalization backward reduction requires a nonempty input')
    channels_last = (_batch_channels_last(a) and _batch_channels_last(grad)
                     and (weight is None or weight.is_contiguous()) and mean.is_contiguous() and invstd.is_contiguous())
    mu, inv = (_saved(v, a, channels, _acc(a)) for v in (mean, invstd))
    if not channels_last and weight is None and (weight_g or bias_g):
        raise RuntimeError('requested batch affine gradients without a weight')
    sums_needed = input_g or channels_last
    sum_dy, sum_dy_xmu = (a.new_empty((channels if sums_needed else 0,), dtype=_acc(a)) for _ in range(2))
    weight_needed = weight is not None if channels_last else weight_g
    bias_needed = weight is not None if channels_last else bias_g
    dw = a.new_empty((channels if weight_needed else 0,), dtype=_acc(a))
    db = a.new_empty((channels if bias_needed else 0,), dtype=_acc(a))
    spatial, bx, by, parts = _batch_reduce_config(a, channels_last)
    staging = a.new_empty((2 * channels * parts if channels_last else 0,), dtype=_acc(a))
    _C.scientific_call(316, [_batch_matrix(a), _batch_matrix(grad), mu, inv, sum_dy, sum_dy_xmu, dw, db, staging],
                       [spatial, bx, by, parts, int(channels_last)], [])
    if channels_last and weight is None:
        values = sum_dy, sum_dy_xmu, dw, db
    else:
        values = (sum_dy if sums_needed else None, sum_dy_xmu if sums_needed else None,
                  dw.to(dtype) if weight_needed else None, db.to(dtype) if bias_needed else None)
    return _write(values, (out0, out1, out2, out3), (a, grad, weight, mean, invstd))


def batch_backward_elemt(grad, a, mean, invstd, weight, sum_dy, sum_dy_xmu, count, *, out=None):
    channels, _, _ = _batch_shape(a, weight)
    _gradient(grad, a)
    channels_last = _batch_channels_last(a) and _batch_channels_last(grad)
    mu, inv, dy_sum, dot_sum = (_saved(v, a, channels, _acc(a)) for v in (mean, invstd, sum_dy, sum_dy_xmu))
    if not channels_last and weight is not None and weight.dtype != mean.dtype:
        raise RuntimeError('batch normalization backward element weight must match saved-statistic dtype')
    if count.dtype != torch.int32 or count.device != a.device:
        raise RuntimeError('batch normalization backward counts must be int32 on the input device')
    x = _batch_matrix(a)
    dx = torch.empty_like(x)
    _C.scientific_call(317, [x, _batch_matrix(grad), mu, inv, _parameter(weight, a), dy_sum, dot_sum, count.resolve_neg().reshape(-1), dx],
                       [int(weight is not None)], [])
    if channels_last:
        value = _batch_restore(dx, a)
    else:
        value = dx.reshape(channels, a.shape[0], *a.shape[2:]).movedim(0, 1).contiguous()
    return _output(value, out, (a, grad, mean, invstd, weight, sum_dy, sum_dy_xmu, count))


def _no_stats(a, weight, bias, training, momentum, eps, **kwargs):
    return batch_norm(a, weight, bias, None, None, training, momentum, eps, **kwargs)


def _no_training(a, weight, bias, running_mean, running_var, momentum, eps, *, out0=None, out1=None, out2=None):
    return batch_norm(a, weight, bias, running_mean, running_var, False, momentum, eps, out=out0, save_mean=out1, save_invstd=out2)


def _with_update(a, weight, bias, running_mean, running_var, momentum, eps, *, out=None, save_mean=None, save_invstd=None, reserve=None):
    value, mean, inv = batch_norm(a, weight, bias, running_mean, running_var, True, momentum, eps, out=out, save_mean=save_mean, save_invstd=save_invstd)
    reserved = a.new_empty((0,), dtype=torch.uint8)
    return value, mean, inv, _output(reserved, reserve, ())


def _batch_backward(grad, a, weight, running_mean, running_var, save_mean, save_var, update, eps, output_mask, reserve):
    return batch_backward(grad, a, weight, running_mean, running_var, save_mean, save_var, update, eps, output_mask)


def register(library):
    for name, fn in {'native_layer_norm': layer_norm, 'native_layer_norm_backward': layer_backward,
                     'native_group_norm': group_norm, 'native_group_norm_backward': group_backward,
                     'native_batch_norm_backward': batch_backward, 'batch_norm_stats': batch_stats,
                     'batch_norm_update_stats': batch_update_stats,
                     'batch_norm_gather_stats': batch_gather, 'batch_norm_gather_stats_with_counts': batch_gather_counts,
                     'batch_norm_backward_reduce': batch_backward_reduce, 'batch_norm_backward_elemt': batch_backward_elemt,
                     '_native_batch_norm_legit_no_training': _no_training}.items():
        library.impl(name, fn)
        library.impl(name + '.out', fn)
    for name in ('native_batch_norm', '_native_batch_norm_legit'):
        library.impl(name, batch_norm)
        library.impl(name + '.out', batch_norm)
    library.impl('_native_batch_norm_legit.no_stats', _no_stats)
    library.impl('_native_batch_norm_legit.no_stats_out', _no_stats)
    library.impl('batch_norm_elemt', batch_elemt)
    library.impl('batch_norm_elemt.out', batch_elemt)
    rms_schema = torch.ops.aten._fused_rms_norm.default._schema
    library.impl('_fused_rms_norm', rms_norm_legacy
                 if rms_schema.arguments[1].name == 'normalized_shape_ndim' else rms_norm)
    if hasattr(torch.ops.aten, '_fused_rms_norm_backward'):
        library.impl('_fused_rms_norm_backward', rms_backward)
    library.impl('_batch_norm_with_update', _with_update)
    library.impl('_batch_norm_with_update.out', _with_update)
    library.impl('batch_norm_backward', _batch_backward)
