import torch

from . import _C
from ._indexing import _logical
from ._ops import _dtypes
from ._ordering import _output, _outputs


def _input(a, quant_min, quant_max):
    if a.layout != torch.strided or a.dtype not in _dtypes:
        raise RuntimeError('fake quantization requires a real floating strided input')
    if quant_min > quant_max:
        raise RuntimeError('quant_min must be less than or equal to quant_max')
    if a.numel() > 2**32 - 1 or a.ndim > 32 or any(n > 2**32 - 1 for n in a.shape):
        raise RuntimeError('fake quantization exceeds native U32 indexing or rank-32 limits')


def _pair(a):
    return torch.empty_like(a), torch.empty_like(a, dtype=torch.bool)


def _write(values, output, mask, inputs):
    if (output is None) != (mask is None):
        raise RuntimeError('both fake quantization outputs must be provided together')
    return _outputs(values, None if output is None else (output, mask), inputs)


def _parameter(value, a, dtypes):
    if value.layout != torch.strided or value.device != a.device or value.dtype not in dtypes:
        raise RuntimeError('fake quantization parameter has incompatible dtype, layout or device')
    return _logical(value)


def per_tensor(a, scale, zero_point, quant_min, quant_max, *, out0=None, out1=None):
    _input(a, quant_min, quant_max)
    if not quant_min <= zero_point <= quant_max:
        raise RuntimeError('zero_point must be between quant_min and quant_max')
    values = _pair(a)
    _C.scientific_call(545, [_logical(a), *values], [zero_point, quant_min, quant_max], [scale])
    return _write(values, out0, out1, (a,))


def _tensor_forward(a, scale, zero_point, enabled, quant_min, quant_max, axis, mode):
    values = _pair(a)
    if enabled is None:
        enabled = a.new_empty((0,), dtype=torch.int64)
    _C.scientific_call(546, [_logical(a), scale, zero_point, enabled, *values],
                       [quant_min, quant_max, axis, mode], [])
    return values


def tensor_qparams(a, scale, zero_point, fake_quant_enabled, quant_min, quant_max, *, out0=None, out1=None):
    _input(a, quant_min, quant_max)
    s = _parameter(scale, a, (torch.float32,))
    z = _parameter(zero_point, a, (torch.int32,))
    enabled = _parameter(fake_quant_enabled, a, (torch.int64,))
    if not s.numel() or not z.numel() or not enabled.numel():
        raise RuntimeError('fake quantization tensor parameters must not be empty')
    values = _tensor_forward(a, s.reshape(-1)[:1], z.reshape(-1)[:1].to(torch.int64),
                             enabled.reshape(-1)[:1], quant_min, quant_max, -1, 0)
    return _write(values, out0, out1, (a, scale, zero_point, fake_quant_enabled))


def _channel_shape(a, scale, zero_point, axis):
    if axis < 0 or axis >= a.ndim:
        raise RuntimeError('fake quantization axis must be a nonnegative input dimension')
    if scale.ndim != 1 or zero_point.ndim != 1 or scale.numel() != zero_point.numel() or scale.numel() != a.shape[axis]:
        raise RuntimeError('scale and zero_point must be one-dimensional and match the channel size')


def per_channel(a, scale, zero_point, axis, quant_min, quant_max, *, out0=None, out1=None):
    _input(a, quant_min, quant_max)
    _channel_shape(a, scale, zero_point, axis)
    s = _parameter(scale, a, (torch.float32, torch.bfloat16)).to(torch.float32)
    z = _parameter(zero_point, a, (torch.int32, torch.float32, torch.float16))
    floating = z.is_floating_point()
    if not floating:
        if int(z.min().item()) < quant_min or int(z.max().item()) > quant_max:
            raise RuntimeError('zero_point must be between quant_min and quant_max')
    z = z.to(torch.float32 if floating else torch.int64)
    values = _tensor_forward(a, s, z, None, quant_min, quant_max, axis, 3 if floating else 2)
    return _write(values, out0, out1, (a, scale, zero_point))


def _first(value, a):
    if value.layout != torch.strided or value.is_complex() or value.ndim == 0 or not value.shape[0] or value[0].numel() != 1:
        raise RuntimeError('learnable per-tensor quantization requires a real scalar in parameter[0]')
    return _logical(value[0]).to(device=a.device, dtype=torch.float32).reshape(1)


def learnable_tensor(a, scale, zero_point, quant_min, quant_max, grad_factor=1.0, *, out=None):
    _input(a, quant_min, quant_max)
    s, z = _first(scale, a), _first(zero_point, a)
    z = z.round().clamp(quant_min, quant_max).to(torch.int64)
    zero = int(z[0].item())
    if zero < quant_min or zero > quant_max:
        raise RuntimeError('zero_point must be between quant_min and quant_max')
    values = _tensor_forward(a, s, z, None, quant_min, quant_max, -1, 1)
    return _output(values[0], out, (a, scale, zero_point))


def learnable_channel(a, scale, zero_point, axis, quant_min, quant_max, grad_factor=1.0, *, out=None):
    _input(a, quant_min, quant_max)
    rounded = zero_point.round().clamp(quant_min, quant_max).to(torch.int32)
    value = per_channel(a, scale, rounded, axis, quant_min, quant_max)[0]
    return _output(value, out, (a, scale, zero_point))


def _learnable_inputs(grad, a, scale, zero_point, quant_min, quant_max):
    _input(a, quant_min, quant_max)
    if quant_min > 0 or quant_max < 0:
        raise RuntimeError('learnable quantization gradient requires a quantization range containing zero')
    tensors = (a, grad, scale, zero_point)
    if a.dtype == torch.bfloat16:
        tensors = tuple(v.to(torch.float32) for v in tensors)
    if any(v.dtype != torch.float32 or v.layout != torch.strided for v in tensors):
        raise RuntimeError('learnable fake quantization backward requires Float tensors or BFloat16 input')
    if grad.device != a.device:
        raise RuntimeError('fake quantization gradient must be on the input device')
    return tuple(_logical(v) for v in tensors)


def _learnable_grad(x, grad, scale, zero_point, quant_min, quant_max, axis, grad_factor):
    dx, ds, dz = (torch.empty_like(x) for _ in range(3))
    _C.scientific_call(547, [x, grad, scale, zero_point, dx, ds, dz],
                       [quant_min, quant_max, axis], [grad_factor])
    return dx, ds, dz


def learnable_tensor_backward(grad, a, scale, zero_point, quant_min, quant_max, grad_factor=1.0):
    x, dy, s, z = _learnable_inputs(grad, a, scale, zero_point, quant_min, quant_max)
    if x.numel() != dy.numel():
        raise RuntimeError('fake quantization input and gradient must have the same number of elements')
    s, z = _first(s, a), _first(z, a)
    if not a.numel():
        return a, scale, zero_point
    x, dy = torch.broadcast_tensors(x, dy)
    _input(x, quant_min, quant_max)
    dx, ds, dz = _learnable_grad(x, dy, s, z, quant_min, quant_max, -1, grad_factor)
    return dx, ds.sum().unsqueeze(0).to(scale.device), dz.sum().unsqueeze(0).to(zero_point.device)


def learnable_channel_backward(grad, a, scale, zero_point, axis, quant_min, quant_max, grad_factor=1.0):
    x, dy, s, z = _learnable_inputs(grad, a, scale, zero_point, quant_min, quant_max)
    _channel_shape(x, s, z, axis)
    if x.shape != dy.shape:
        raise RuntimeError('fake quantization input and gradient must have the same shape')
    s = _parameter(s, x, (torch.float32,))
    z = _parameter(z, x, (torch.float32,)).round().clamp(quant_min, quant_max)
    if int(z.min().item()) < quant_min or int(z.max().item()) > quant_max:
        raise RuntimeError('zero_point must be between quant_min and quant_max')
    if not a.numel():
        return a, scale, zero_point
    dx, ds, dz = _learnable_grad(x, dy, s, z, quant_min, quant_max, axis, grad_factor)
    axes = tuple(i for i in range(x.ndim) if i != axis)
    return dx, ds.sum(axes), dz.sum(axes)


def moving_observer(a, observer_on, fake_quant_on, running_min, running_max, scale, zero_point,
                    averaging_const, quant_min, quant_max, ch_axis, per_row_fake_quant=False,
                    symmetric_quant=False, *, out0=None, out1=None):
    _input(a, quant_min, quant_max)
    if ch_axis >= a.ndim or per_row_fake_quant and ch_axis < 0:
        raise RuntimeError('observer channel axis is outside the input dimensions')
    for flag in (observer_on, fake_quant_on):
        if flag.device != a.device or flag.layout != torch.strided or flag.numel() != 1:
            raise RuntimeError('observer flags must be scalar tensors on the input device')
    for value, dtype in ((running_min, a.dtype), (running_max, a.dtype),
                         (scale, torch.float32), (zero_point, torch.int32)):
        _parameter(value, a, (dtype,))
    x = _logical(a)
    size = a.shape[ch_axis] if per_row_fake_quant else 1
    if per_row_fake_quant and not running_min.numel():
        running_min.resize_((size,)).fill_(float('inf'))
        running_max.resize_((size,)).fill_(-float('inf'))
        scale.resize_((size,))
        zero_point.resize_((size,))
    if any(value.numel() != size for value in (running_min, running_max, scale, zero_point)):
        raise RuntimeError('observer state sizes must match the observed channels')
    if per_row_fake_quant:
        if x.ndim != 2:
            axes = list(range(x.ndim))
            axes[0], axes[ch_axis] = axes[ch_axis], axes[0]
            x = x.permute(axes).flatten(1)
        current_min, current_max = torch.aminmax(x, dim=1)
    else:
        current_min, current_max = torch.aminmax(x)
    states = tuple(_logical(value) for value in (running_min, running_max, scale, zero_point))
    obs = _logical(observer_on).to(torch.int64)
    fq = _logical(fake_quant_on).to(torch.int64)
    _C.scientific_call(548, [obs, fq, current_min, current_max, *states],
                       [quant_min, quant_max, int(symmetric_quant)], [averaging_const])
    for original, state in zip((running_min, running_max, scale, zero_point), states):
        if original is not state:
            original.copy_(state)
    if per_row_fake_quant:
        if int(fake_quant_on.item()):
            values = per_channel(a, scale, zero_point, 0, quant_min, quant_max)
        else:
            values = a.clone(), torch.ones_like(a, dtype=torch.bool)
    else:
        values = tensor_qparams(a, scale, zero_point, fq, quant_min, quant_max)
    return _write(values, out0, out1, (a, observer_on, fake_quant_on, running_min, running_max, scale, zero_point))


def register(library):
    for name, fn in (
        ('fake_quantize_per_tensor_affine_cachemask', per_tensor),
        ('_fake_quantize_per_tensor_affine_cachemask_tensor_qparams', tensor_qparams),
        ('fake_quantize_per_channel_affine_cachemask', per_channel),
        ('_fake_quantize_learnable_per_tensor_affine', learnable_tensor),
        ('_fake_quantize_learnable_per_channel_affine', learnable_channel),
        ('_fused_moving_avg_obs_fq_helper', moving_observer),
    ):
        library.impl(name, fn)
        library.impl(name + '.out', fn)
    library.impl('_fake_quantize_learnable_per_tensor_affine_backward', learnable_tensor_backward)
    library.impl('_fake_quantize_learnable_per_channel_affine_backward', learnable_channel_backward)
