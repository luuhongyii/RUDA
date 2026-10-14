"""Model-independent triangular solves and checkpointed gated-delta training."""
from __future__ import annotations

import math
import torch
from torch.autograd.function import once_differentiable
from torch.utils.checkpoint import checkpoint
from ._architecture_ops import precision_context, work_dtype, positive_int


def triangular_mask(x, diagonal=0, *, upper=False):
    if x.ndim < 2:
        raise ValueError('triangular masking requires matrices')
    return torch.triu(x, diagonal=diagonal) if upper else torch.tril(x, diagonal=diagonal)


def _solve(a, b, upper, unitriangular):
    if a.device.type != 'ruda':
        return torch.linalg.solve_triangular(a, b, upper=upper, unitriangular=unitriangular)
    from .math_backend import _nvidia
    if _nvidia() and a.dtype in (torch.float32,torch.float64):
        from ._vendor_blas import triangular
        return triangular(a,b,upper=upper,unit=unitriangular)
    from . import _sequence_available
    if not _sequence_available:
        raise RuntimeError('RUDA sequence API 1 is required; rebuild both native libraries')
    shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    aa = a.expand(*shape, *a.shape[-2:]).contiguous()
    bb = b.expand(*shape, *b.shape[-2:]).contiguous()
    return torch.ops.ruda.triangular_solve(aa, bb, bool(upper), bool(unitriangular))


class _TriangularSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, upper, unitriangular):
        result = _solve(a, b, upper, unitriangular)
        ctx.save_for_backward(a, result)
        ctx.options = upper, unitriangular, b.shape
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        a, result = ctx.saved_tensors
        upper, unitriangular, bshape = ctx.options
        db = _solve(a.transpose(-1, -2), gradient, not upper, unitriangular)
        da = triangular_mask(-(db @ result.transpose(-1, -2)),
                             1 if upper and unitriangular else -1 if unitriangular else 0,
                             upper=upper)
        return da.sum_to_size(a.shape), db.sum_to_size(bshape), None, None


def solve_triangular(a, b, *, upper=False, left=True, unitriangular=False):
    """First-order batched triangular solve, including broadcast and right solves.

    RUDA uses its native same-device solve. CPU/CUDA are explicit reference
    devices, not recovery paths for a failed RUDA operation.
    """
    if any(type(flag) is not bool for flag in (upper, left, unitriangular)):
        raise TypeError('triangular solve flags must be Python bools')
    if a.ndim < 2 or b.ndim < 2 or a.shape[-1] != a.shape[-2]:
        raise ValueError('A must be square and B must be a matrix')
    if a.device != b.device or a.dtype != b.dtype or not a.is_floating_point():
        raise ValueError('solve operands must share a floating dtype and device')
    if b.shape[-2 if left else -1] != a.shape[-1]:
        raise ValueError('incompatible triangular solve dimensions')
    if not left:
        return _TriangularSolve.apply(a.transpose(-1, -2), b.transpose(-1, -2),
                                      not upper, unitriangular).transpose(-1, -2)
    return _TriangularSolve.apply(a, b, upper, unitriangular)


def _delta_chunk(q, k, v, beta, decay, state, scale):
    length = q.shape[-2]
    index = torch.arange(length, device=q.device, dtype=torch.int64)
    lower = index.unsqueeze(-1) >= index.unsqueeze(0)
    strict = index.unsqueeze(-1) > index.unsqueeze(0)
    diagonal = index.unsqueeze(-1) == index.unsqueeze(0)
    cumulative = decay.cumsum(-1)
    difference = cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)
    # Future positions are masked before exp: negative decays can otherwise
    # overflow in the unused upper triangle and poison its gradients.
    factors = torch.where(lower, difference, torch.zeros_like(difference)).exp()
    kb = k * beta.unsqueeze(-1)
    system = torch.where(strict, (kb @ k.transpose(-1, -2)) * factors,
                         torch.zeros_like(factors)) + diagonal.to(q.dtype)
    rhs = torch.cat((v * beta.unsqueeze(-1), kb * cumulative.exp().unsqueeze(-1)), -1)
    solved = solve_triangular(system, rhs, unitriangular=True)
    u, w = solved[..., :v.shape[-1]], solved[..., v.shape[-1]:]
    residual = u - w @ state
    attention = torch.where(lower, (q * scale @ k.transpose(-1, -2)) * factors,
                            torch.zeros_like(factors))
    output = (q * (cumulative.exp() * scale).unsqueeze(-1)) @ state + attention @ residual
    tail = (cumulative[..., -1:] - cumulative).exp()
    final = state * cumulative[..., -1, None, None].exp() + (k * tail.unsqueeze(-1)).transpose(-1, -2) @ residual
    return output, final


def _delta_scan(q, k, v, beta, decay, state, scale, chunk_size, checkpoint_chunks):
    output = []
    for start in range(0, q.shape[-2], chunk_size):
        end = min(start + chunk_size, q.shape[-2])
        args = (q[..., start:end, :], k[..., start:end, :], v[..., start:end, :],
                beta[..., start:end], decay[..., start:end], state, scale)
        if checkpoint_chunks and torch.is_grad_enabled():
            y, state = checkpoint(_delta_chunk, *args, use_reentrant=False, preserve_rng_state=False)
        else:
            y, state = _delta_chunk(*args)
        output.append(y)
    return torch.cat(output, -2), state


def _delta_chunk_adjoint(q, k, v, beta, decay, state, dy, dh, scale):
    length = q.shape[-2]
    index = torch.arange(length, device=q.device, dtype=torch.int64)
    lower = index[:, None] >= index[None, :]
    strict = index[:, None] > index[None, :]
    diagonal = index[:, None] == index[None, :]
    cumulative = decay.cumsum(-1)
    difference = cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)
    factors = torch.where(lower, difference, torch.zeros_like(difference)).exp()
    prefix = cumulative.exp()
    tail = (cumulative[..., -1:] - cumulative).exp()
    kb = k * beta.unsqueeze(-1)
    kk = kb @ k.transpose(-1, -2)
    system = torch.where(strict, kk * factors, torch.zeros_like(factors)) + diagonal.to(q.dtype)
    rhs = torch.cat((v * beta.unsqueeze(-1), kb * prefix.unsqueeze(-1)), -1)
    solved = solve_triangular(system, rhs, unitriangular=True)
    u, w = solved[..., :v.shape[-1]], solved[..., v.shape[-1]:]
    residual = u - w @ state
    qscaled = q * scale
    qk = qscaled @ k.transpose(-1, -2)
    attention = torch.where(lower, qk * factors, torch.zeros_like(factors))
    qp = q * (prefix * scale).unsqueeze(-1)
    kt = k * tail.unsqueeze(-1)
    final_decay = prefix[..., -1, None, None]

    dqp = dy @ state.transpose(-1, -2)
    da = dy @ residual.transpose(-1, -2)
    dr = attention.transpose(-1, -2) @ dy + kt @ dh
    dkt = residual @ dh.transpose(-1, -2)
    ds = qp.transpose(-1, -2) @ dy + dh * final_decay - w.transpose(-1, -2) @ dr
    dfinal = (dh * state).sum((-2, -1)) * prefix[..., -1]
    dz = torch.cat((dr, -(dr @ state.transpose(-1, -2))), -1)
    drhs = solve_triangular(system.transpose(-1, -2), dz, upper=True, unitriangular=True)
    dsystem = -(drhs @ solved.transpose(-1, -2))
    dvb, dkp = drhs[..., :v.shape[-1]], drhs[..., v.shape[-1]:]
    dv = dvb * beta.unsqueeze(-1)
    db = (dvb * v).sum(-1)
    dkb = dkp * prefix.unsqueeze(-1)
    dp = (dkp * kb).sum(-1)
    dkk = torch.where(strict, dsystem * factors, torch.zeros_like(factors))
    dfactors = torch.where(strict, dsystem * kk, torch.zeros_like(factors))
    dkb = dkb + dkk @ k
    dk = dkk.transpose(-1, -2) @ kb
    dqk = torch.where(lower, da * factors, torch.zeros_like(factors))
    dfactors = dfactors + torch.where(lower, da * qk, torch.zeros_like(factors))
    dq = (dqk @ k) * scale + dqp * (prefix * scale).unsqueeze(-1)
    dk = dk + dqk.transpose(-1, -2) @ qscaled + dkt * tail.unsqueeze(-1)
    dp = dp + (dqp * q).sum(-1) * scale
    dtail = (dkt * k).sum(-1) * tail
    dk = dk + dkb * beta.unsqueeze(-1)
    db = db + (dkb * k).sum(-1)
    ddifference = torch.where(lower, dfactors * factors, torch.zeros_like(factors))
    dc = dp * prefix + ddifference.sum(-1) - ddifference.sum(-2) - dtail
    dlast = dtail.sum(-1) + dfinal
    dc = dc + dlast.unsqueeze(-1) if length == 1 else torch.cat((dc[..., :-1], dc[..., -1:] + dlast.unsqueeze(-1)), -1)
    dg = dc.flip((-1,)).cumsum(-1).flip((-1,))
    return dq, dk, dv, db, dg, ds


def _delta_scan_adjoint(q, k, v, beta, decay, initial, dy, dh, scale, chunk_size):
    starts = range(0, q.shape[-2], chunk_size)
    states = [initial]
    for start in list(starts)[:-1]:
        end = min(start + chunk_size, q.shape[-2])
        _, state = _delta_chunk(q[..., start:end, :], k[..., start:end, :], v[..., start:end, :],
                                beta[..., start:end], decay[..., start:end], states[-1], scale)
        states.append(state)
    chunks = [[], [], [], [], []]
    for start in reversed(starts):
        end = min(start + chunk_size, q.shape[-2])
        gradients = _delta_chunk_adjoint(q[..., start:end, :], k[..., start:end, :], v[..., start:end, :],
            beta[..., start:end], decay[..., start:end], states.pop(), dy[..., start:end, :], dh, scale)
        for group, gradient in zip(chunks, gradients[:5]):
            group.append(gradient)
        dh = gradients[5]
    result = [torch.cat(tuple(reversed(group)), -2 if i < 3 else -1) for i, group in enumerate(chunks)]
    return (*result, dh)


class _NativeDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, beta, decay, initial, scale, chunk_size, checkpoint_chunks):
        ctx.save_for_backward(q, k, v, beta, decay, initial)
        ctx.options = scale, chunk_size, checkpoint_chunks
        return torch.ops.ruda.delta_forward(q.contiguous(), k.contiguous(), v.contiguous(),
                                           beta.contiguous(), decay.contiguous(), initial.contiguous(), scale, chunk_size)

    @staticmethod
    @once_differentiable
    def backward(ctx, gy, gs):
        q, k, v, beta, decay, initial = ctx.saved_tensors
        gy = torch.zeros_like(v) if gy is None else gy
        gs = torch.zeros_like(initial) if gs is None else gs
        gradients = _delta_scan_adjoint(q, k, v, beta, decay, initial, gy, gs, *ctx.options[:2])
        return tuple(g if needed else None for g, needed in zip(gradients, ctx.needs_input_grad[:6])) + (None,) * 3


def gated_delta_rule(query, key, value, beta, log_decay, *, initial_state=None,
                     query_scale=None, normalize_qk=False, norm_eps=1e-6,
                     chunk_size=64, checkpoint_chunks=True, native_forward=True):
    """Differentiable gated delta recurrence with [B,H,T,D] tensor layout.

    S_t = exp(g_t) S_(t-1) + k_t outer
          (beta_t * (v_t - k_t @ (exp(g_t) S_(t-1))))
    y_t = query_scale * q_t @ S_t.

    RUDA reuses ruDNN's native chunk forward by default. The differentiable
    chunk algorithm uses FP32 state/intermediates (FP64 for explicit CPU FP64
    inputs), device GEMMs and a native triangular solve on RUDA.
    The first-order chunk adjoint rematerializes each chunk using same-device operators;
    this is not a standalone fused DeltaNet backward kernel. No state is
    detached between chunks. Both output and final-state gradients propagate.
    The native-forward adjoint retains boundary states, not an autograd tape;
    checkpoint_chunks controls the composite forward path.
    """
    positive_int(chunk_size, 'chunk_size')
    if query.ndim != 4 or key.shape != query.shape or value.ndim != 4 or value.shape[:3] != query.shape[:3]:
        raise ValueError('query/key/value must be matching [B,H,T,D] tensors')
    if min(query.shape[0], query.shape[1], query.shape[-1], value.shape[-1]) < 1:
        raise ValueError('batch, heads and feature dimensions must be positive')
    if beta.shape != query.shape[:3] or log_decay.shape != beta.shape:
        raise ValueError('beta/log_decay must have shape [B,H,T]')
    tensors = (key, value, beta, log_decay)
    if any(t.device != query.device or not t.is_floating_point() for t in tensors):
        raise ValueError('all operands must be floating tensors on one device')
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError('query/key/value storage dtypes must match')
    dtype = work_dtype(query)
    if isinstance(query_scale, (bool, torch.Tensor)) or isinstance(norm_eps, (bool, torch.Tensor)):
        raise TypeError('query_scale and norm_eps must be Python numbers')
    scale = query.shape[-1] ** -.5 if query_scale is None else float(query_scale)
    if not math.isfinite(scale) or not math.isfinite(norm_eps) or norm_eps <= 0:
        raise ValueError('invalid query scale or normalization epsilon')
    shape = (*query.shape[:2], query.shape[-1], value.shape[-1])
    if initial_state is not None and (initial_state.shape != shape or initial_state.device != query.device or not initial_state.is_floating_point()):
        raise ValueError('initial state must have shape [B,H,K,V] on the input device')
    with precision_context(query):
        q, k, v = (t.to(dtype) for t in (query, key, value))
        if normalize_qk:
            q = q * (q.square().sum(-1, keepdim=True) + norm_eps).rsqrt()
            k = k * (k.square().sum(-1, keepdim=True) + norm_eps).rsqrt()
        state = torch.zeros(shape, device=query.device, dtype=dtype) if initial_state is None else initial_state.to(dtype)
        if not query.shape[-2]:
            # Preserve the connected empty-output and identity-state contract.
            return value * 0, state
        args = q, k, v, beta.to(dtype), log_decay.to(dtype), state, scale, chunk_size, checkpoint_chunks
        if query.device.type == 'ruda' and native_forward:
            from . import _sequence_available
            if not _sequence_available:
                raise RuntimeError('RUDA sequence API 1 required for native DeltaNet; rebuild both libraries')
            output, state = _NativeDelta.apply(*args)
        else:
            output, state = _delta_scan(*args)
        return output.to(value.dtype), state
