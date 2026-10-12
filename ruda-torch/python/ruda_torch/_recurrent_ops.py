"""Shared same-device operations for stateful sequence components."""
from __future__ import annotations

import math
import torch
from ._architecture_ops import positions, work_dtype, precision_context, positive_int


def initial_dt_bias(size, minimum, maximum, device):
    # Parameter initialization only; sequence computation never moves to CPU.
    target = torch.empty(0, device=device).device
    if target.type == 'meta':
        return torch.empty(size, device=target, dtype=torch.float32)
    dt = torch.empty(size, device='cpu', dtype=torch.float64).uniform_(math.log(minimum), math.log(maximum)).exp()
    return (dt + (-(-dt).expm1()).log()).to(device=target, dtype=torch.float32)


def segment_decay(log_decay):
    """exp(sum(g[j+1:i+1])) below the diagonal, without prefix subtraction."""
    length = log_decay.shape[-1]
    ids = positions(length, log_decay.device)
    strict = ids[:, None] > ids[None, :]
    lower = ids[:, None] >= ids[None, :]
    segments = torch.where(strict, log_decay.unsqueeze(-1), torch.zeros_like(log_decay.unsqueeze(-1)))
    return torch.where(lower, segments.cumsum(-2).exp(), torch.zeros_like(segments))


def causal_depthwise_conv1d(x, weight, bias=None, *, state=None, activation=None):
    """[B,T,C] causal convolution; state stores the last K-1 unactivated inputs."""
    if weight.ndim == 3 and weight.shape[1] == 1:
        weight = weight[:, 0, :]
    if x.ndim != 3 or weight.ndim != 2 or weight.shape[0] != x.shape[-1] or weight.shape[1] < 1:
        raise ValueError('causal convolution requires x[B,T,C] and weight[C,K]')
    if x.device != weight.device or x.dtype != weight.dtype:
        raise ValueError('convolution input and weight must share dtype/device')
    if bias is not None and (bias.shape != (x.shape[-1],) or bias.device != x.device or bias.dtype != x.dtype):
        raise ValueError('convolution bias must match channels, dtype and device')
    if activation not in (None, 'silu'):
        raise ValueError('activation must be None or silu')
    history = weight.shape[-1] - 1
    shape = (x.shape[0], x.shape[-1], history)
    if state is None:
        state = x.new_zeros(shape)
    if state.shape != shape or state.device != x.device or state.dtype != x.dtype:
        raise ValueError('convolution state layout, dtype or device mismatch')
    dtype = work_dtype(x)
    with precision_context(x):
        joined = torch.cat((state, x.transpose(1, 2)), -1)
        result = x.to(dtype) * 0
        for tap in range(weight.shape[-1]):
            result = result + joined[..., tap:tap+x.shape[1]].transpose(1, 2).to(dtype) * weight[:, tap].to(dtype)
        if bias is not None:
            result = result + bias.to(dtype)
        if activation == 'silu':
            result = result * result.sigmoid()
    tail = joined[..., -history:].clone() if history else joined[..., :0].clone()
    return result.to(x.dtype), tail


class CausalDepthwiseConv1d(torch.nn.Module):
    def __init__(self, channels, kernel_size=4, *, bias=False, activation='silu', device=None, dtype=None):
        super().__init__()
        positive_int(channels, 'channels')
        positive_int(kernel_size, 'kernel_size')
        if activation not in (None, 'silu'):
            raise ValueError('activation must be None or silu')
        self.weight = torch.nn.Parameter(torch.empty(channels, 1, kernel_size, device=device, dtype=dtype))
        self.bias = torch.nn.Parameter(torch.zeros(channels, device=device, dtype=dtype)) if bias else None
        torch.nn.init.uniform_(self.weight, -kernel_size**-.5, kernel_size**-.5)
        self.activation = activation

    def forward(self, x, state=None):
        return causal_depthwise_conv1d(x, self.weight, self.bias, state=state, activation=self.activation)


def gated_rms_norm(x, gate, weight, eps, *, activation, norm_before_gate, groups=1):
    dtype = work_dtype(x)
    with precision_context(x):
        y, g = x.to(dtype), gate.to(dtype)
        g = g.sigmoid() if activation == 'sigmoid' else g * g.sigmoid()
        if not norm_before_gate:
            y = y * g
        shape = y.shape
        grouped = y.reshape(*shape[:-1], groups, shape[-1] // groups)
        y = (grouped * (grouped.square().mean(-1, keepdim=True) + eps).rsqrt()).reshape(shape)
        y = y * weight.to(dtype)
        if norm_before_gate:
            y = y * g
    return y.to(x.dtype)
