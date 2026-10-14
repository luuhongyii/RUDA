"""Differentiable context-prefix scans for recurrent attention and convolution."""
from __future__ import annotations

import math
import torch
from torch.utils.checkpoint import checkpoint
from ._architecture_ops import precision_context, work_dtype
from ._expert_exchange import _group, register_group
from ._recurrent_ops import segment_decay
from .sequence_training import solve_triangular


@torch.library.custom_op('ruda::context_shift', mutates_args=())
def _shift(value: torch.Tensor, distance: int, group_key: str) -> torch.Tensor:
    group = _group(group_key)
    source, destination = group.rank - distance, group.rank + distance
    sends, receives = [0] * group.world_size, [0] * group.world_size
    if 0 <= destination < group.world_size:
        sends[destination] = 1
    if 0 <= source < group.world_size:
        receives[source] = 1
    rows = value.contiguous().reshape(1, -1)
    result = group.all_to_all(rows if sum(sends) else rows[:0], sends, receives)
    return result.reshape(value.shape) if sum(receives) else torch.zeros_like(value)


@_shift.register_fake
def _shift_fake(value, distance, group_key):
    return torch.empty_like(value)


def _shift_context(ctx, inputs, output):
    _, ctx.distance, ctx.group_key = inputs


def _shift_gradient(ctx, gradient):
    return _shift(gradient.contiguous(), -ctx.distance, ctx.group_key), None, None


_shift.register_autograd(_shift_gradient, setup_context=_shift_context)


@torch.library.custom_op('ruda::context_initial_state', mutates_args=())
def _initial_state(value: torch.Tensor, group_key: str) -> torch.Tensor:
    result = value.clone()
    _group(group_key).broadcast_(result, 0)
    return result


@_initial_state.register_fake
def _initial_fake(value, group_key):
    return torch.empty_like(value)


def _initial_context(ctx, inputs, output):
    _, ctx.group_key = inputs


@torch.library.custom_op('ruda::context_initial_state_backward', mutates_args=())
def _initial_backward(gradient: torch.Tensor, group_key: str) -> torch.Tensor:
    group = _group(group_key)
    result = gradient.clone()
    group.sum_(result)
    return result if group.rank == 0 else torch.zeros_like(result)


@_initial_backward.register_fake
def _initial_backward_fake(gradient, group_key):
    return torch.empty_like(gradient)


def _initial_gradient(ctx, gradient):
    return _initial_backward(gradient, ctx.group_key), None


_initial_state.register_autograd(_initial_gradient, setup_context=_initial_context)


def _token_metadata(value, documents, valid):
    batch, length = value.shape[:2]
    if documents is None:
        documents = torch.arange(batch, device=value.device, dtype=torch.int64)[:, None].expand(batch, length)
    if valid is None:
        valid = torch.ones((batch, length), device=value.device, dtype=torch.bool)
    if (documents.shape != (batch, length) or documents.dtype != torch.int64 or documents.device != value.device
            or valid.shape != documents.shape or valid.dtype != torch.bool or valid.device != value.device):
        raise ValueError('context recurrence needs device int64 documents and boolean token validity')
    return documents, valid


def context_document_resets(documents, valid, group, *, group_key='context-recurrent'):
    """Compare each live token to the preceding live token, including other ranks."""
    key = register_group(group_key, group)
    batch, length = documents.shape
    ordinal = torch.arange(length, device=documents.device, dtype=torch.int64)[None, :].expand(batch, length)
    if length:
        counts = valid.to(torch.int64).cumsum(-1)
        order = (~valid).to(torch.int64).argsort(dim=-1, stable=True)
        latest = order.gather(1, (counts - 1).clamp_min(0))
        previous = torch.where(counts > 0, latest, torch.full_like(ordinal, -1))
    else:
        previous = ordinal
    padded = torch.cat((torch.zeros((batch, 1), device=documents.device, dtype=torch.int64), documents), -1)
    last_index = previous[:, -1:] if length else torch.full((batch, 1), -1, device=documents.device, dtype=torch.int64)
    local = torch.cat((padded.gather(1, last_index + 1), (last_index >= 0).to(torch.int64)), -1)
    distance = 1
    while distance < group.world_size:
        before = _shift(local, distance, key)
        local = torch.where((local[:, 1:] == 0) & (before[:, 1:] != 0), before, local)
        distance *= 2
    before = _shift(local, 1, key)
    previous = torch.cat((torch.full((batch, 1), -1, device=documents.device, dtype=torch.int64), previous), -1)[:, :length]
    previous_document = torch.where(previous >= 0, padded.gather(1, previous + 1), before[:, :1])
    have_previous = (previous >= 0) | (before[:, 1:] != 0)
    return valid & have_previous & (documents != previous_document)


def _segmented_chunk(q, k, v, beta, decay, initial, reset, scale):
    """Channel-gated WY solve with exact document resets, without infinite decay."""
    length = q.shape[-2]
    if not length:
        anchor = (q.sum() + k.sum() + v.sum() + beta.sum() + decay.sum()) * 0
        return v + anchor + initial.sum() * 0, initial + anchor
    segments = reset.to(torch.int64).cumsum(-1)
    connected = segments[:, :, None] == segments[:, None, :]
    carried = segment_decay(decay.transpose(-1, -2)).permute(0, 1, 3, 4, 2)
    carried = carried * connected[:, None, :, :, None]
    carried_keys = carried * k.unsqueeze(-3)
    kk = (k.unsqueeze(-2) * carried_keys).sum(-1)
    qk = (q.unsqueeze(-2) * carried_keys).sum(-1) * scale
    index = torch.arange(length, device=q.device, dtype=torch.int64)
    strict, diagonal = index[:, None] > index[None, :], index[:, None] == index[None, :]
    system = torch.where(strict, beta.unsqueeze(-1) * kk, torch.zeros_like(kk)) + diagonal.to(q.dtype)
    prefix = decay.cumsum(-2).exp() * (segments == 0)[:, None, :, None]
    rhs = beta.unsqueeze(-1) * (v - (k * prefix) @ initial)
    residual = solve_triangular(system, rhs, unitriangular=True)
    output = (q * prefix * scale) @ initial + qk @ residual
    final = initial * prefix[..., -1, :, None] + carried_keys[..., -1, :, :].transpose(-1, -2) @ residual
    return output, final


def _local_scan(q, k, v, beta, decay, initial, resets, scale, chunk_size, recompute):
    pieces, state = [], initial
    for start in range(0, q.shape[-2], chunk_size):
        end = min(start + chunk_size, q.shape[-2])
        args = (q[..., start:end, :], k[..., start:end, :], v[..., start:end, :],
                beta[..., start:end], decay[..., start:end, :], state, resets[:, start:end], scale)
        if recompute and torch.is_grad_enabled():
            output, state = checkpoint(_segmented_chunk, *args, use_reentrant=False, preserve_rng_state=False)
        else:
            output, state = _segmented_chunk(*args)
        pieces.append(output)
    if not pieces:
        anchor = (q.sum() + k.sum() + v.sum() + beta.sum() + decay.sum()) * 0
        return v + anchor + initial.sum() * 0, initial + anchor
    return torch.cat(pieces, -2), state


def _prefix_affine(transition, update, group, key):
    width = transition.shape[-1]
    packed = torch.cat((transition, update), -1)
    distance = 1
    while distance < group.world_size:
        before = _shift(packed, distance, key)
        a, b = packed[..., :width], packed[..., width:]
        merged = torch.cat((a @ before[..., :width], a @ before[..., width:] + b), -1)
        # A tensor select preserves the collective adjoint on every rank,
        # including ranks whose received block is the zero boundary value.
        choose = torch.tensor(group.rank >= distance, device=packed.device)
        packed = torch.where(choose, merged, packed)
        distance *= 2
    return _shift(packed, 1, key)


def context_parallel_kda(query, key, value, log_decay, beta, group, *,
                         documents=None, valid=None, initial_state=None,
                         query_scale=None, normalize_qk=False, norm_eps=1e-6,
                         chunk_size=64, checkpoint_chunks=True, group_key='context-recurrent'):
    """Exclusive affine prefix over CP ranks, followed by local chunk attention.

    Q/K[B,H,T,K], V[B,HV,T,V], decay[B,HV,T,K], beta[B,HV,T].
    CP ranks own consecutive token intervals; T may differ. Initial state is
    supplied with the same shape/gradient policy on all ranks; rank zero owns
    its value and receives the summed derivative. Returned state is the end of
    this rank's interval, not a detached or replicated global cache.
    """
    if (query.ndim != 4 or key.shape != query.shape or value.ndim != 4
            or value.shape[0] != query.shape[0] or value.shape[2] != query.shape[2]
            or min(query.shape[0], query.shape[1], query.shape[3], value.shape[1], value.shape[3]) < 1
            or value.shape[1] % query.shape[1]):
        raise ValueError('context KDA requires compatible Q/K/V head and token layouts')
    if log_decay.shape != (*value.shape[:3], query.shape[-1]) or beta.shape != value.shape[:3]:
        raise ValueError('context KDA decay/beta layout differs')
    if any(t.device != query.device or not t.is_floating_point() for t in (key, value, log_decay, beta)):
        raise ValueError('context recurrent operands must be floating on one device')
    if key.dtype != query.dtype or value.dtype != query.dtype or query.device.type != group.device_type:
        raise ValueError('context Q/K/V storage must match the communicator device and dtype')
    if type(chunk_size) is not int or chunk_size < 1 or not math.isfinite(norm_eps) or norm_eps <= 0:
        raise ValueError('context chunk size and normalization epsilon must be positive')
    scale = query.shape[-1] ** -.5 if query_scale is None else float(query_scale)
    if not math.isfinite(scale):
        raise ValueError('context query scale must be finite')
    documents, valid = _token_metadata(query.transpose(1, 2), documents, valid)
    state_shape = (query.shape[0], value.shape[1], query.shape[-1], value.shape[-1])
    if initial_state is not None and (initial_state.shape != state_shape or initial_state.device != query.device):
        raise ValueError('context initial state shape/device differs')
    group.validate_training_options((state_shape, str(query.dtype), str(log_decay.dtype), chunk_size,
        scale, normalize_qk, norm_eps, checkpoint_chunks, torch.is_grad_enabled(),
        tuple(t.requires_grad for t in (query, key, value, log_decay, beta)),
        initial_state is None, False if initial_state is None else initial_state.requires_grad))
    communicator = register_group(group_key, group)
    resets = context_document_resets(documents, valid, group, group_key=group_key)
    with precision_context(query):
        dtype = work_dtype(query)
        q, k, v, g, b = (t.to(dtype) for t in (query, key, value, log_decay, beta))
        if normalize_qk:
            q = q * (q.square().sum(-1, keepdim=True) + norm_eps).rsqrt()
            k = k * (k.square().sum(-1, keepdim=True) + norm_eps).rsqrt()
        copies = value.shape[1] // query.shape[1]
        if copies != 1:
            q, k = q.repeat_interleave(copies, 1), k.repeat_interleave(copies, 1)
        keep = valid[:, None, :, None]
        q, k, v, g = (torch.where(keep, t, torch.zeros_like(t)) for t in (q, k, v, g))
        b = torch.where(valid[:, None, :], b, torch.zeros_like(b))
        initial = q.new_zeros(state_shape) if initial_state is None else _initial_state(initial_state.to(dtype), communicator)
        channel = torch.arange(q.shape[-1], device=q.device, dtype=torch.int64)
        eye = (channel[:, None] == channel[None, :]).to(dtype).expand(*state_shape[:2], q.shape[-1], q.shape[-1])
        zero_q = torch.zeros_like(q)
        _, transition = _local_scan(zero_q, k, torch.zeros_like(k), b, g, eye, resets,
                                    1., chunk_size, checkpoint_chunks)
        _, update = _local_scan(zero_q, k, v, b, g, torch.zeros_like(initial), resets,
                                1., chunk_size, checkpoint_chunks)
        before = _prefix_affine(transition, update, group, communicator)
        carried = before[..., :q.shape[-1]] @ initial + before[..., q.shape[-1]:]
        incoming = torch.where(torch.tensor(group.rank != 0, device=q.device), carried, initial)
        output, final = _local_scan(q, k, v, b, g, incoming, resets, scale, chunk_size, checkpoint_chunks)
    return output.to(value.dtype), final


def context_parallel_delta(query, key, value, beta, log_decay, group, **options):
    if log_decay.shape != beta.shape or beta.shape != query.shape[:3]:
        raise ValueError('context DeltaNet needs scalar decay/beta per head and token')
    decay = log_decay.unsqueeze(-1).expand(*log_decay.shape, query.shape[-1])
    return context_parallel_kda(query, key, value, decay, beta, group, **options)


def _tail_merge(left, right, right_count):
    history = left.shape[1]
    positions = torch.arange(history, device=left.device, dtype=torch.int64)[None, :]
    indices = (positions + right_count[:, None]).clamp_max(history - 1)
    suffix = [1] * (left.ndim - 2)
    selected = left.gather(1, indices.reshape(*indices.shape, *suffix).expand(left.shape))
    choose = (positions < history - right_count[:, None]).reshape(*positions.expand(left.shape[:2]).shape, *suffix)
    return torch.where(choose, selected, right)


def context_parallel_causal_conv1d(value, weight, bias, group, *, documents=None, valid=None,
                                   skip_padding=True, activation='silu', group_key='context-convolution'):
    """Exchange only K-1-token halos; keep document and padding boundaries exact."""
    if weight.ndim == 3 and weight.shape[1] == 1:
        weight = weight[:, 0]
    if (value.ndim != 3 or weight.ndim != 2 or weight.shape[0] != value.shape[-1]
            or weight.shape[1] < 1 or weight.dtype != value.dtype or weight.device != value.device):
        raise ValueError('context convolution requires input[B,T,C] and matching weight[C,K]')
    if bias is not None and (bias.shape != value.shape[-1:] or bias.device != value.device or bias.dtype != value.dtype):
        raise ValueError('context convolution bias differs')
    if activation not in (None, 'silu'):
        raise ValueError('context convolution activation must be None or silu')
    documents, valid = _token_metadata(value, documents, valid)
    if not skip_padding:
        valid = torch.ones_like(valid)
    batch, length, channels = value.shape
    group.validate_training_options((batch, channels, tuple(weight.shape), str(value.dtype), skip_padding,
        activation, value.requires_grad, weight.requires_grad, bias is None, torch.is_grad_enabled()))
    key = register_group(group_key, group)
    order = (~valid).to(torch.int64).argsort(dim=1, stable=True)
    inverse = order.argsort(dim=1)
    packed = value.gather(1, order[:, :, None].expand(value.shape))
    docs = documents.gather(1, order)
    counts = valid.sum(-1, dtype=torch.int64)
    ordinal = torch.arange(length, device=value.device, dtype=torch.int64)[None, :]
    live = ordinal < counts[:, None]
    packed = torch.where(live[:, :, None], packed, torch.zeros_like(packed))
    history = weight.shape[-1] - 1
    if history:
        index = counts[:, None] - history + torch.arange(history, device=value.device, dtype=torch.int64)[None, :]
        padded = torch.cat((value.new_zeros(batch, 1, channels), packed), 1)
        padded_docs = torch.cat((torch.zeros(batch, 1, device=docs.device, dtype=docs.dtype), docs), 1)
        tail = padded.gather(1, (index + 1).clamp_min(0)[:, :, None].expand(batch, history, channels))
        tail_docs = padded_docs.gather(1, (index + 1).clamp_min(0))
        retained = counts.clamp_max(history)
        distance = 1
        while distance < group.world_size:
            left = _shift(tail, distance, key)
            left_docs, left_count = _shift(tail_docs, distance, key), _shift(retained, distance, key)
            tail, tail_docs = _tail_merge(left, tail, retained), _tail_merge(left_docs, tail_docs, retained)
            retained = (retained + left_count).clamp_max(history)
            distance *= 2
        incoming = _shift(tail, 1, key)
        incoming_docs, incoming_count = _shift(tail_docs, 1, key), _shift(retained, 1, key)
        incoming_valid = torch.arange(history, device=value.device)[None, :] >= history - incoming_count[:, None]
    else:
        incoming, incoming_docs = value[:, :0], docs[:, :0]
        incoming_valid = live[:, :0]
    joined, joined_docs = torch.cat((incoming, packed), 1), torch.cat((incoming_docs, docs), 1)
    joined_valid = torch.cat((incoming_valid, live), 1)
    boundaries = ((joined_docs[:, 1:] != joined_docs[:, :-1])
                  & joined_valid[:, 1:] & joined_valid[:, :-1]).to(torch.int64)
    segments = torch.cat((torch.zeros((batch, 1), device=value.device, dtype=torch.int64), boundaries.cumsum(-1)), -1)
    current_segments = segments[:, history:history+length]
    dtype = work_dtype(value)
    with precision_context(value):
        output = packed.to(dtype) * 0
        for tap in range(weight.shape[-1]):
            same = ((joined_docs[:, tap:tap+length] == docs) & joined_valid[:, tap:tap+length] & live
                    & (segments[:, tap:tap+length] == current_segments))
            signal = torch.where(same[:, :, None], joined[:, tap:tap+length], torch.zeros_like(packed))
            output = output + signal.to(dtype) * weight[:, tap].to(dtype)
        if bias is not None:
            output = output + bias.to(dtype)
        if activation == 'silu':
            output = output * output.sigmoid()
    output = torch.where(live[:, :, None], output, torch.zeros_like(output)).to(value.dtype)
    return output.gather(1, inverse[:, :, None].expand(value.shape))
