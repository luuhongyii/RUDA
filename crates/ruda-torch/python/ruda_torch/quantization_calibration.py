"""Device-resident calibration and affine fake quantization."""
from __future__ import annotations
import math
import torch
from torch import nn
from .packed_quantization import FORMATS
from . import _C


@torch.library.custom_op('ruda::quantization_sort_rows',mutates_args=())
def _sort_rows(x:torch.Tensor)->torch.Tensor:
    out = torch.empty_like(x)
    index = torch.empty(0,dtype=torch.int32,device=x.device)
    _C.packed_dispatch(5,[x,x,x,x,index,out],[8,1,x.shape[1],x.shape[0]])
    return out


@_sort_rows.register_fake
def _sort_rows_fake(x):
    return torch.empty_like(x)


def _maximum(a,b):
    return torch.where(a>b,a,b)


def _minimum(a,b):
    return torch.where(a<b,a,b)


def _clamp(x,low,high):
    return torch.where(x<low,torch.full_like(x,low),torch.where(x>high,torch.full_like(x,high),x))


def _bounds(value):
    low,high = value,value
    while low.shape[-1]>1:
        pairs = low.shape[-1]//2
        lo = _minimum(low[...,:2*pairs:2],low[...,1:2*pairs:2])
        hi = _maximum(high[...,:2*pairs:2],high[...,1:2*pairs:2])
        if low.shape[-1]%2:
            lo,hi = torch.cat((lo,low[...,-1:]),-1),torch.cat((hi,high[...,-1:]),-1)
        low,high = lo,hi
    return low,high


def _params(low, high, format, symmetric):
    if FORMATS[format] >= 14:
        if not symmetric:
            raise ValueError('floating quantization uses symmetric scales; affine offsets are integer-only')
        bound = {'fp4_e2m1': 6., 'fp8_e4m3': 448., 'fp8_e5m2': 57344.}[format]
        return _maximum(low.abs(), high.abs()) / bound, torch.zeros_like(low)
    bits = FORMATS[format]
    if symmetric:
        scales = _maximum(low.abs(), high.abs()) / ((1 << (bits - 1)) - 1)
        offsets = -(1 << (bits - 1)) * scales
    else:
        low, high = _minimum(low, torch.zeros_like(low)), _maximum(high, torch.zeros_like(high))
        scales = (high - low) / ((1 << bits) - 1)
        safe = torch.where(scales == 0, torch.ones_like(scales), scales)
        zeros = _clamp(torch.round(-low / safe), 0, (1 << bits) - 1)
        offsets = -zeros * scales
    return scales, offsets


def _roundtrip(value, scales, offsets, format):
    safe = torch.where(scales == 0, torch.ones_like(scales), scales)
    normalized = (value - offsets) / safe
    code = FORMATS[format]
    if code < 14:
        decoded = _clamp(torch.round(normalized), 0, (1 << code) - 1)
    elif code == 14:
        # Exact E2M1 nearest-even boundaries, including ties at 0.25, 0.75, ...
        magnitude = normalized.abs()
        decoded = torch.zeros_like(magnitude)
        for i, (threshold, upper) in enumerate(((.25, .5), (.75, 1.), (1.25, 1.5), (1.75, 2.), (2.5, 3.), (3.5, 4.), (5., 6.))):
            take = magnitude > threshold if i % 2 == 0 else magnitude >= threshold
            decoded = torch.where(take, torch.full_like(decoded, upper), decoded)
        decoded = torch.where(normalized < 0, -decoded, decoded)
    else:
        bound, dtype = (448., torch.float8_e4m3fn) if code == 18 else (57344., torch.float8_e5m2)
        decoded = _clamp(normalized,-bound,bound).to(dtype).float()
    return decoded * scales + offsets


def calibrate_quantization(x, *, format='int4', group_size=128, symmetric=True,
                          method='minmax', percentile=99.9, mse_steps=80):
    """Return FP32 [rows,ceil(width/group)] scales and additive offsets.

    Percentile drops equal tail fractions; MSE evaluates `mse_steps` ratios
    from 1 down to 1/mse_steps independently for every group, including tails.
    """
    if format not in FORMATS or x.ndim < 1 or x.shape[-1] < 1 or type(group_size) is not int or group_size < 1:
        raise ValueError('invalid calibration format or group geometry')
    if method not in ('minmax', 'percentile', 'mse'):
        raise ValueError('calibration must be minmax, percentile or mse')
    if not 0 < percentile <= 100 or type(mse_steps) is not int or mse_steps < 1:
        raise ValueError('invalid percentile or MSE search size')
    if not x.is_floating_point() or x.dtype in (torch.float8_e4m3fn,torch.float8_e5m2):
        raise ValueError('calibration requires unquantized floating input')
    values = x.detach().float().reshape(-1, x.shape[-1])
    scales, offsets = [], []
    for start in range(0, x.shape[-1], group_size):
        group = values[:, start:start + group_size]
        low, high = _bounds(group)
        if method == 'percentile':
            ordered = _sort_rows(group.contiguous()) if group.device.type=='ruda' else group.sort(dim=1).values
            def quantile(p):
                position = p * (group.shape[1] - 1)
                a, b = math.floor(position), math.ceil(position)
                return ordered[:, a:a+1] * (b-position if a != b else 1.) + (ordered[:, b:b+1] * (position-a) if a != b else 0.)
            tail = (1. - percentile / 100.) / 2.
            low, high = quantile(tail), quantile(1. - tail)
        scale, offset = _params(low, high, format, symmetric)
        if method == 'mse':
            best = ((_roundtrip(group, scale, offset, format) - group) ** 2).sum(dim=1, keepdim=True)
            for step in range(1, mse_steps):
                ratio = 1. - step / mse_steps
                s, o = _params(low * ratio, high * ratio, format, symmetric)
                error = ((_roundtrip(group, s, o, format) - group) ** 2).sum(dim=1, keepdim=True)
                take = error < best
                scale, offset, best = torch.where(take, s, scale), torch.where(take, o, offset), torch.where(take, error, best)
        scales.append(scale)
        offsets.append(offset)
    return torch.cat(scales, 1), torch.cat(offsets, 1)


class QuantizationObserver(nn.Module):
    """Streaming per-tensor/per-channel MinMax or exponential-moving MinMax.

    Running bounds are buffers; state_dict preserves calibration across runs.
    `channel_axis=None` observes a single tensor range, independent of batch size.
    """
    def __init__(self, *, channel_axis=None, averaging_constant=None):
        super().__init__()
        if averaging_constant is not None and not 0 < averaging_constant <= 1:
            raise ValueError('averaging_constant must be in (0,1]')
        self.channel_axis, self.averaging_constant = channel_axis, averaging_constant
        self.register_buffer('min_val', torch.empty(0, dtype=torch.float32))
        self.register_buffer('max_val', torch.empty(0, dtype=torch.float32))

    @torch.no_grad()
    def forward(self, x):
        if not x.numel():
            return x
        value = x.detach().float()
        if self.channel_axis is None:
            low, high = (t.reshape(1) for t in _bounds(value.reshape(1,-1)))
        else:
            axis = self.channel_axis % value.ndim
            value = value.movedim(axis, 0).reshape(value.shape[axis], -1)
            low, high = (t.reshape(-1) for t in _bounds(value))
        if not self.min_val.numel():
            self.min_val, self.max_val = low, high
        else:
            if self.min_val.shape != low.shape or self.min_val.device != low.device:
                raise ValueError('observer channel count/device changed')
            if self.averaging_constant is None:
                self.min_val, self.max_val = _minimum(self.min_val, low), _maximum(self.max_val, high)
            else:
                a = self.averaging_constant
                self.min_val = self.min_val + a * (low - self.min_val)
                self.max_val = self.max_val + a * (high - self.max_val)
        return x

    def calculate_qparams(self, *, format='int8', symmetric=False):
        if not self.min_val.numel():
            raise RuntimeError('observer has not seen any values')
        return _params(self.min_val, self.max_val, format, symmetric)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for name in ('min_val', 'max_val'):
            if prefix + name in state_dict:
                setattr(self, name, torch.empty_like(state_dict[prefix + name], device=getattr(self, name).device))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class _AffineSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero, maximum):
        normalized = x / scale + zero
        quant = _clamp(torch.round(normalized),0,maximum)
        ctx.save_for_backward(normalized, quant, zero)
        ctx.maximum, ctx.scale_shape = maximum, scale.shape
        return (quant - zero) * scale

    @staticmethod
    def backward(ctx, grad):
        value, quant, zero = ctx.saved_tensors
        inside = (value >= 0) & (value <= ctx.maximum)
        dx = grad * inside.to(grad.dtype)
        ds = grad * torch.where(inside, quant - value, quant - zero)
        return dx, ds.sum_to_size(ctx.scale_shape), None, None


def affine_fake_quantize(x, scales, zero_points, *, bits=8):
    """Unsigned affine fake quantization with STE input/scale gradients.

    Zero points are frozen integer-valued metadata, unlike trainable scales.
    Positive scales and integral zero points in [0,2**bits-1] are required.
    """
    if bits not in (2, 3, 4, 5, 6, 8) or x.device != scales.device or x.device != zero_points.device:
        raise ValueError('invalid affine bit width or parameter device')
    return _AffineSTE.apply(x, scales, zero_points.detach(), (1 << bits) - 1)
