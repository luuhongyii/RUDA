"""Channel-gated delta attention on the input device, with differentiable state."""
from __future__ import annotations

import math
import torch
from torch.autograd.function import once_differentiable
from torch.utils.checkpoint import checkpoint
from ._architecture_ops import positive_int, positive_float, work_dtype, precision_context, positions
from ._recurrent_ops import segment_decay, CausalDepthwiseConv1d, gated_rms_norm, initial_dt_bias
from .architecture_cache import KDAState
from .sequence_training import solve_triangular


class _NativeKDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, beta, decay, initial, scale, chunk):
        tensors = tuple(t.contiguous() for t in (q, k, v, beta, decay, initial))
        retain = any(ctx.needs_input_grad[:6])
        output, final, boundaries = torch.ops.ruda.kda_forward(*tensors, scale, chunk, retain)
        if retain:
            ctx.save_for_backward(*tensors, boundaries)
        ctx.options = scale, chunk
        ctx.set_materialize_grads(False)
        return output, final

    @staticmethod
    @once_differentiable
    def backward(ctx, dy, dh):
        q, k, v, beta, decay, initial, boundaries = ctx.saved_tensors
        scale, chunk = ctx.options
        dy = torch.zeros_like(v) if dy is None else dy.contiguous()
        dh = torch.zeros_like(initial) if dh is None else dh.contiguous()
        gradients = torch.ops.ruda.kda_backward(q, k, v, beta, decay, initial,
                                               boundaries, dy, dh, scale, chunk)
        return (*gradients, None, None)


def _kda_chunk(q, k, v, g, beta, state, scale):
    length = q.shape[-2]
    if length == 1:
        decayed = state * g[..., 0, :, None].exp()
        residual = beta[..., 0, None] * (v[..., 0, :] - (k[..., 0, :, None] * decayed).sum(-2))
        state = decayed + k[..., 0, :, None] * residual.unsqueeze(-2)
        return ((q[..., 0, :, None] * state).sum(-2) * scale).unsqueeze(-2), state
    # [B,H,K,I,J]; each channel has its own decay, unlike Gated DeltaNet.
    factors = segment_decay(g.transpose(-1, -2)).permute(0, 1, 3, 4, 2)
    carried_keys = factors * k.unsqueeze(-3)
    kk = (k.unsqueeze(-2) * carried_keys).sum(-1)
    qk = (q.unsqueeze(-2) * carried_keys).sum(-1) * scale
    ids = positions(length, q.device)
    strict = ids[:, None] > ids[None, :]
    identity = (ids[:, None] == ids[None, :]).to(q.dtype)
    system = torch.where(strict, beta.unsqueeze(-1) * kk, torch.zeros_like(kk)) + identity
    prefix = g.cumsum(-2).exp()
    rhs = beta.unsqueeze(-1) * (v - (k * prefix) @ state)
    if q.dtype == torch.float64 and q.device.type == 'ruda':
        residual = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
    else:
        residual = solve_triangular(system, rhs, unitriangular=True)
    output = (q * prefix * scale) @ state + qk @ residual
    final = state * prefix[..., -1, :, None] + carried_keys[..., -1, :, :].transpose(-1, -2) @ residual
    return output, final


def kda(query, key, value, log_decay, beta, *, initial_state=None, query_scale=None,
        normalize_qk=False, norm_eps=1e-6, chunk_size=64, checkpoint_chunks=True):
    """KDA with Q/K[B,H,T,K], V[B,HV,T,V], log_decay[B,HV,T,K], beta[B,HV,T].

    HV is a multiple of H. State is [B,HV,K,V]. Each step first multiplies
    state rows by exp(log_decay), then applies the beta-weighted delta update.
    Returns (output, final_state); both carry gradients across chunk boundaries.
    Half inputs accumulate in FP32; explicit FP64 stays FP64. Chunk workspaces are
    bounded by chunk_size, not total sequence length. RUDA FP32 accumulation uses
    native chunk forward/backward and fused one-token decode; its backward saves
    only chunk boundaries and rematerializes chunk intermediates on device.
    checkpoint_chunks controls the composite reference/FP64 path. Native backward
    is first-order. Explicit FP64 retains the same-device composite implementation.
    No CPU numerical fallback.
    """
    positive_int(chunk_size, 'chunk_size')
    positive_float(norm_eps, 'norm_eps')
    if type(normalize_qk) is not bool or type(checkpoint_chunks) is not bool:
        raise TypeError('normalization and checkpoint options must be booleans')
    if query.ndim != 4 or key.shape != query.shape or value.ndim != 4:
        raise ValueError('KDA requires four-dimensional Q/K/V tensors')
    batch, heads, length, width = query.shape
    if min(heads, width, value.shape[1], value.shape[-1]) <= 0 or value.shape[1] % heads:
        raise ValueError('positive value heads must be divisible by query heads')
    if value.shape[0] != batch or value.shape[-2] != length:
        raise ValueError('KDA batch and sequence lengths differ')
    if log_decay.shape != (*value.shape[:3], width) or beta.shape != value.shape[:3]:
        raise ValueError('KDA requires one decay per key channel and one beta per value head/token')
    dtype = work_dtype(query)
    if any(t.device != query.device or not t.is_floating_point() for t in (key, value, log_decay, beta)):
        raise ValueError('all KDA inputs must be floating tensors on the same device')
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError('Q/K/V storage dtypes must match')
    scale = width**-.5 if query_scale is None else float(query_scale)
    if not math.isfinite(scale):
        raise ValueError('query_scale must be finite')
    state_shape = (batch, value.shape[1], width, value.shape[-1])
    if initial_state is not None and (initial_state.shape != state_shape or
            initial_state.device != query.device or not initial_state.is_floating_point()):
        raise ValueError('initial KDA state has incompatible layout or device')
    with precision_context(query):
        q, k, v, g, b = (tensor.to(dtype) for tensor in (query, key, value, log_decay, beta))
        if normalize_qk:
            q = q * (q.square().sum(-1, keepdim=True) + norm_eps).rsqrt()
            k = k * (k.square().sum(-1, keepdim=True) + norm_eps).rsqrt()
        copies = value.shape[1] // heads
        if copies != 1:
            q, k = q.repeat_interleave(copies, 1), k.repeat_interleave(copies, 1)
        state = q.new_zeros(state_shape) if initial_state is None else initial_state.to(dtype)
        if length == 0:
            return (v + (q.sum() + k.sum() + g.sum() + b.sum() + state.sum()) * 0).to(value.dtype), state
        if query.device.type == 'ruda' and dtype == torch.float32:
            from . import _kda_available
            if not _kda_available:
                raise RuntimeError('RUDA KDA API 1 is required; rebuild both native libraries')
            if torch.is_grad_enabled():
                output, state = _NativeKDA.apply(q, k, v, b, g, state, scale, chunk_size)
            else:
                tensors = tuple(t.contiguous() for t in (q, k, v, b, g, state))
                output, state, _ = torch.ops.ruda.kda_forward(*tensors, scale, chunk_size, False)
            return output.to(value.dtype), state
        output = []
        for start in range(0, length, chunk_size):
            end = min(start + chunk_size, length)
            args = (q[..., start:end, :], k[..., start:end, :], v[..., start:end, :],
                    g[..., start:end, :], b[..., start:end], state, scale)
            if checkpoint_chunks and torch.is_grad_enabled():
                y, state = checkpoint(_kda_chunk, *args, use_reentrant=False, preserve_rng_state=False)
            else:
                y, state = _kda_chunk(*args)
            output.append(y)
    return torch.cat(output, -2).to(value.dtype), state


class KimiDeltaAttention(torch.nn.Module):
    """Trainable KDA mixer; returns (hidden_states, KDAState) for prefill or decode."""
    def __init__(self, hidden_size, num_heads, head_dim, *, num_value_heads=None,
                 value_head_dim=None, conv_size=4, conv_bias=False, gate_rank=None,
                 beta_scale=1., norm_eps=1e-5, qk_norm_eps=1e-6, chunk_size=64,
                 checkpoint_chunks=True, device=None, dtype=None):
        super().__init__()
        hv = num_heads if num_value_heads is None else num_value_heads
        vd = head_dim if value_head_dim is None else value_head_dim
        rank = vd if gate_rank is None else gate_rank
        for name, size in (('hidden_size', hidden_size), ('num_heads', num_heads), ('head_dim', head_dim),
                           ('num_value_heads', hv), ('value_head_dim', vd), ('gate_rank', rank), ('chunk_size', chunk_size)):
            positive_int(size, name)
        if hv % num_heads or type(conv_size) is not int or conv_size < 0:
            raise ValueError('invalid KDA value heads or convolution size')
        if type(checkpoint_chunks) is not bool:
            raise TypeError('checkpoint_chunks must be a boolean')
        self.hidden_size, self.num_heads, self.num_value_heads = hidden_size, num_heads, hv
        self.head_dim, self.value_head_dim, self.chunk_size = head_dim, vd, chunk_size
        self.norm_eps, self.qk_norm_eps = positive_float(norm_eps, 'norm_eps'), positive_float(qk_norm_eps, 'qk_norm_eps')
        self.beta_scale = positive_float(beta_scale, 'beta_scale')
        self.checkpoint_chunks = checkpoint_chunks
        options = dict(device=device, dtype=dtype)
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * head_dim, bias=False, **options)
        self.k_proj = torch.nn.Linear(hidden_size, num_heads * head_dim, bias=False, **options)
        self.v_proj = torch.nn.Linear(hidden_size, hv * vd, bias=False, **options)
        self.f_proj = torch.nn.Sequential(torch.nn.Linear(hidden_size, rank, bias=False, **options),
                                         torch.nn.Linear(rank, hv * head_dim, bias=False, **options))
        self.b_proj = torch.nn.Linear(hidden_size, hv, bias=False, **options)
        self.g_proj = torch.nn.Sequential(torch.nn.Linear(hidden_size, rank, bias=False, **options),
                                         torch.nn.Linear(rank, hv * vd, bias=True, **options))
        self.o_norm_weight = torch.nn.Parameter(torch.ones(vd, **options))
        self.o_proj = torch.nn.Linear(hv * vd, hidden_size, bias=False, **options)
        self.convolutions = torch.nn.ModuleList([
            CausalDepthwiseConv1d(channels, conv_size, bias=conv_bias, **options)
            for channels in (num_heads * head_dim, num_heads * head_dim, hv * vd)
        ]) if conv_size else torch.nn.ModuleList()
        self.A_log = torch.nn.Parameter(torch.empty(hv, device=device, dtype=torch.float32).uniform_(1., 16.).log())
        self.dt_bias = torch.nn.Parameter(initial_dt_bias(hv * head_dim, .001, .1, device))

    def forward(self, hidden_states, cache=None):
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError('KDA hidden states must be [batch,tokens,hidden_size]')
        if cache is not None and not isinstance(cache, KDAState):
            raise TypeError('KDA requires a KDAState cache')
        if cache is not None and len(cache.convolution) != len(self.convolutions):
            raise ValueError('KDA convolution cache count differs from the layer')
        batch, length = hidden_states.shape[:2]
        projected = [self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)]
        convolution = []
        for i, tensor in enumerate(projected):
            if self.convolutions:
                projected[i], tail = self.convolutions[i](tensor, None if cache is None else cache.convolution[i])
                convolution.append(tail)
            else:
                projected[i] = tensor * tensor.sigmoid()
        q, k = (tensor.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2) for tensor in projected[:2])
        v = projected[2].reshape(batch, length, self.num_value_heads, self.value_head_dim).transpose(1, 2)
        dtype = work_dtype(hidden_states)
        with precision_context(hidden_states):
            raw_gate = self.f_proj(hidden_states).to(dtype) + self.dt_bias.to(dtype)
            decay = -torch.nn.functional.softplus(raw_gate).reshape(batch, length, self.num_value_heads, self.head_dim)
            decay = decay * self.A_log.to(dtype).exp().reshape(1, 1, -1, 1)
            beta = self.b_proj(hidden_states).to(dtype).sigmoid().transpose(1, 2) * self.beta_scale
        output, state = kda(q, k, v, decay.transpose(1, 2), beta,
                            initial_state=None if cache is None else cache.recurrent, normalize_qk=True,
                            norm_eps=self.qk_norm_eps, chunk_size=self.chunk_size,
                            checkpoint_chunks=self.checkpoint_chunks)
        output = output.transpose(1, 2)
        gate = self.g_proj(hidden_states).reshape(batch, length, self.num_value_heads, self.value_head_dim)
        output = gated_rms_norm(output, gate, self.o_norm_weight, self.norm_eps,
                                activation='sigmoid', norm_before_gate=True)
        output = self.o_proj(output.reshape(batch, length, self.num_value_heads * self.value_head_dim))
        return output, KDAState(state, tuple(convolution), (0 if cache is None else cache.seen_tokens) + length)
