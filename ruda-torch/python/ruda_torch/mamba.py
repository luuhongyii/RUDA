"""Mamba-2 state-space duality and stateful mixers using RUDA device operators."""
from __future__ import annotations

import math
import torch
from torch.utils.checkpoint import checkpoint
from ._architecture_ops import positive_int, positive_float, work_dtype, precision_context
from ._recurrent_ops import segment_decay, CausalDepthwiseConv1d, gated_rms_norm, initial_dt_bias
from .architecture_cache import MambaState


def _ssd_chunk(x, decay, b, c, state):
    factors = segment_decay(decay)
    prefix = decay.cumsum(-1).exp()
    output = ((c @ b.transpose(-1, -2)) * factors) @ x
    output = output + (c @ state.transpose(-1, -2)) * prefix.unsqueeze(-1)
    final = state * prefix[..., -1, None, None] + x.transpose(-1, -2) @ (b * factors[..., -1, :, None])
    return output, final


def ssd_scan(x, dt, a, b, c, *, d=None, z=None, dt_bias=None, dt_softplus=False,
             dt_limit=(0., float('inf')), initial_state=None, chunk_size=64, checkpoint_chunks=True):
    """Mamba-2 scan: X[B,H,T,P], dt[B,H,T], A[H], B/C[B,G,T,N].

    H is divisible by G. State[B,H,P,N] evolves as exp(dt*A)*state +
    dt*outer(X,B); output contracts state with C, optionally adds D*X and
    multiplies by SiLU(z). D is [H] or [H,P]. Returns (output, final_state).
    Training gradients include initial/final state and every supplied parameter.
    """
    positive_int(chunk_size, 'chunk_size')
    if type(dt_softplus) is not bool or type(checkpoint_chunks) is not bool:
        raise TypeError('scan flags must be booleans')
    if x.ndim != 4 or b.ndim != 4 or c.shape != b.shape:
        raise ValueError('SSD requires X[B,H,T,P] and matching B/C[B,G,T,N]')
    batch, heads, length, width = x.shape
    if min(heads, width, b.shape[1], b.shape[-1]) <= 0 or heads % b.shape[1]:
        raise ValueError('SSD groups must divide positive head count')
    if b.shape[0] != batch or b.shape[-2] != length or dt.shape != x.shape[:3] or a.shape != (heads,):
        raise ValueError('SSD batch, length, head or dt shapes differ')
    if d is not None and d.shape not in ((heads,), (heads, width)):
        raise ValueError('SSD D must have shape [heads] or [heads,head_dim]')
    if z is not None and z.shape != x.shape:
        raise ValueError('SSD gate must match X')
    if dt_bias is not None and dt_bias.shape != (heads,):
        raise ValueError('SSD dt_bias must have shape [heads]')
    if (not isinstance(dt_limit, (tuple, list)) or len(dt_limit) != 2 or
            not 0 <= dt_limit[0] < dt_limit[1] or not math.isfinite(dt_limit[0])):
        raise ValueError('invalid SSD dt limits')
    dtype = work_dtype(x)
    tensors = [dt, a, b, c] + [t for t in (d, z, dt_bias, initial_state) if t is not None]
    if any(t.device != x.device or not t.is_floating_point() for t in tensors):
        raise ValueError('all SSD tensors must be floating and on the input device')
    shape = (batch, heads, width, b.shape[-1])
    if initial_state is not None and initial_state.shape != shape:
        raise ValueError('initial SSD state must be [batch,heads,head_dim,state_dim]')
    with precision_context(x):
        xx, bb, cc, step = (t.to(dtype) for t in (x, b, c, dt))
        if dt_bias is not None:
            step = step + dt_bias.to(dtype).reshape(1, heads, 1)
        if dt_softplus:
            step = torch.nn.functional.softplus(step)
        step = step.clamp(min=dt_limit[0], max=dt_limit[1])
        decay = step * a.to(dtype).reshape(1, heads, 1)
        copies = heads // b.shape[1]
        if copies != 1:
            bb, cc = bb.repeat_interleave(copies, 1), cc.repeat_interleave(copies, 1)
        state = xx.new_zeros(shape) if initial_state is None else initial_state.to(dtype)
        output = []
        for start in range(0, length, chunk_size):
            end = min(start + chunk_size, length)
            args = (xx[..., start:end, :] * step[..., start:end, None], decay[..., start:end],
                    bb[..., start:end, :], cc[..., start:end, :], state)
            if checkpoint_chunks and torch.is_grad_enabled():
                y, state = checkpoint(_ssd_chunk, *args, use_reentrant=False, preserve_rng_state=False)
            else:
                y, state = _ssd_chunk(*args)
            output.append(y)
        result = torch.cat(output, -2) if output else xx + (bb.sum() + cc.sum() + step.sum() + a.sum() + state.sum()) * 0
        if d is not None:
            skip = d.to(dtype).reshape(1, heads, 1, 1 if d.ndim == 1 else width)
            result = result + xx * skip
        if z is not None:
            gate = z.to(dtype)
            result = result * gate * gate.sigmoid()
    return result.to(x.dtype), state


class Mamba2(torch.nn.Module):
    """Trainable Mamba-2 mixer with convolution and SSD cache for chunked decode."""
    def __init__(self, hidden_size, *, state_dim=64, expand=2, head_dim=64, groups=1,
                 conv_size=4, bias=False, conv_bias=True, norm_eps=1e-5,
                 norm_before_gate=False, norm_groups=1, dt_min=.001, dt_max=.1,
                 dt_limit=(0., float('inf')), a_init_range=(1., 16.), learnable_initial_state=False,
                 chunk_size=64, checkpoint_chunks=True, device=None, dtype=None):
        super().__init__()
        for name, size in (('hidden_size', hidden_size), ('state_dim', state_dim), ('expand', expand),
                           ('head_dim', head_dim), ('groups', groups), ('norm_groups', norm_groups),
                           ('conv_size', conv_size), ('chunk_size', chunk_size)):
            positive_int(size, name)
        inner = hidden_size * expand
        if inner % head_dim or (inner // head_dim) % groups or inner % norm_groups:
            raise ValueError('Mamba inner width, heads and norm groups must divide evenly')
        if any(type(flag) is not bool for flag in (norm_before_gate, learnable_initial_state, checkpoint_chunks)):
            raise TypeError('Mamba normalization, state and checkpoint flags must be booleans')
        if not 0 < dt_min <= dt_max < float('inf') or not 0 < a_init_range[0] <= a_init_range[1] < float('inf'):
            raise ValueError('invalid Mamba time-step or A initialization range')
        self.hidden_size, self.inner, self.head_dim = hidden_size, inner, head_dim
        self.heads, self.groups, self.state_dim = inner // head_dim, groups, state_dim
        self.chunk_size, self.checkpoint_chunks = chunk_size, checkpoint_chunks
        self.dt_limit, self.norm_before_gate, self.norm_groups = dt_limit, norm_before_gate, norm_groups
        self.norm_eps = positive_float(norm_eps, 'norm_eps')
        options = dict(device=device, dtype=dtype)
        channels = inner + 2 * groups * state_dim
        self.in_proj = torch.nn.Linear(hidden_size, inner + channels + self.heads, bias=bias, **options)
        self.conv1d = CausalDepthwiseConv1d(channels, conv_size, bias=conv_bias, **options)
        self.out_proj = torch.nn.Linear(inner, hidden_size, bias=bias, **options)
        self.norm_weight = torch.nn.Parameter(torch.ones(inner, **options))
        self.A_log = torch.nn.Parameter(torch.empty(self.heads, device=device, dtype=torch.float32).uniform_(*a_init_range).log())
        self.dt_bias = torch.nn.Parameter(initial_dt_bias(self.heads, dt_min, dt_max, device))
        self.D = torch.nn.Parameter(torch.ones(self.heads, device=device, dtype=torch.float32))
        self.initial_state = torch.nn.Parameter(torch.zeros(self.heads, head_dim, state_dim, **options)) if learnable_initial_state else None

    def forward(self, hidden_states, cache=None):
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError('Mamba hidden states must be [batch,tokens,hidden_size]')
        if cache is not None and not isinstance(cache, MambaState):
            raise TypeError('Mamba2 requires a MambaState cache')
        if cache is not None and cache.convolution is None:
            raise ValueError('Mamba2 mixer cache must include convolution history')
        batch, length = hidden_states.shape[:2]
        channels = self.inner + 2 * self.groups * self.state_dim
        projected = self.in_proj(hidden_states)
        z, xbc, dt = projected.split((self.inner, channels, self.heads), -1)
        xbc, conv_state = self.conv1d(xbc, None if cache is None else cache.convolution)
        x, b, c = xbc.split((self.inner, self.groups * self.state_dim, self.groups * self.state_dim), -1)
        x = x.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        b, c = (t.reshape(batch, length, self.groups, self.state_dim).transpose(1, 2) for t in (b, c))
        state = None if self.initial_state is None else self.initial_state.unsqueeze(0).expand(batch, -1, -1, -1)
        if cache is not None:
            state = cache.recurrent
        output, state = ssd_scan(x, dt.transpose(1, 2), -self.A_log.to(work_dtype(x)).exp(), b, c,
                                d=self.D, dt_bias=self.dt_bias, dt_softplus=True, dt_limit=self.dt_limit,
                                initial_state=state, chunk_size=self.chunk_size,
                                checkpoint_chunks=self.checkpoint_chunks)
        output = output.transpose(1, 2).reshape(batch, length, self.inner)
        output = gated_rms_norm(output, z, self.norm_weight, self.norm_eps, activation='silu',
                                norm_before_gate=self.norm_before_gate, groups=self.norm_groups)
        return self.out_proj(output), MambaState(state, conv_state, (0 if cache is None else cache.seen_tokens) + length)
