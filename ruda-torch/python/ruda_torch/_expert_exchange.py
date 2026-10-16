"""Differentiable variable-size expert exchanges with device-resident payloads."""
from __future__ import annotations

from threading import RLock
import torch
from torch.autograd.function import once_differentiable

_groups = {}
_lock = RLock()


def register_group(key, group):
    if not isinstance(key, str) or not key:
        raise ValueError('expert exchange needs a stable, nonempty communicator key')
    if _groups.get(key) is group:
        return key
    with _lock:
        previous = _groups.get(key)
        if previous is not None and previous is not group:
            raise ValueError(f'expert communicator key already belongs to another group: {key}')
        _groups[key] = group
    return key


def unregister_group(key, group):
    with _lock:
        if _groups.get(key) is group:
            del _groups[key]


def _group(key):
    with _lock:
        if key not in _groups:
            raise RuntimeError(f'expert communicator is not registered: {key}')
        return _groups[key]


def _counts(tensor, world, total=None):
    if tensor.dtype != torch.int64 or tensor.shape != (world,):
        raise ValueError('exchange counts must be rank-ordered int64[world]')
    counts = tuple(tensor.detach().to('cpu').tolist())
    if any(n < 0 for n in counts) or total is not None and sum(counts) != total:
        raise ValueError('exchange row counts do not describe the payload')
    return counts


@torch.compiler.disable
def backward_participation(rows, trainable, group_key):
    enabled = torch.is_grad_enabled()
    peers = _group(group_key).gather_metadata((enabled, rows.requires_grad or trainable))
    if any(mode != enabled for mode, _ in peers):
        raise ValueError('expert ranks must use the same autograd mode')
    return enabled and any(needed for _, needed in peers)


@torch.library.custom_op('ruda::expert_exchange', mutates_args=())
def exchange(rows: torch.Tensor, ids: torch.Tensor, sends: torch.Tensor,
             group_key: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    group = _group(group_key)
    if rows.ndim != 2 or ids.shape != (rows.shape[0],) or ids.dtype != torch.int64:
        raise ValueError('expert exchange requires row payloads and one int64 expert ID per row')
    if rows.device != ids.device or rows.device != sends.device:
        raise ValueError('expert exchange tensors must share a device')
    send_rows = _counts(sends, group.world_size, rows.shape[0])
    matrix = group.gather_metadata(send_rows)
    if any(len(row) != group.world_size or any(type(n) is not int or n < 0 for n in row) for row in matrix):
        raise ValueError('expert peers supplied an invalid exchange partition')
    receive_rows = tuple(row[group.rank] for row in matrix)
    received = group.all_to_all(rows, send_rows, receive_rows)
    received_ids = group.all_to_all(ids, send_rows, receive_rows)
    counts = torch.tensor(receive_rows, device=sends.device, dtype=torch.int64)
    return received, received_ids, counts


@exchange.register_fake
def _exchange_fake(rows, ids, sends, group_key):
    received = torch.library.get_ctx().new_dynamic_size()
    return (rows.new_empty((received, rows.shape[1])), ids.new_empty((received,)),
            torch.empty_like(sends))


@torch.library.custom_op('ruda::expert_exchange_backward', mutates_args=())
def exchange_backward(gradient: torch.Tensor, sends: torch.Tensor,
                      receives: torch.Tensor, original: torch.Tensor, group_key: str) -> torch.Tensor:
    group = _group(group_key)
    send_rows = _counts(sends, group.world_size, original.shape[0])
    receive_rows = _counts(receives, group.world_size, gradient.shape[0])
    return group.all_to_all(gradient, receive_rows, send_rows)


@exchange_backward.register_fake
def _exchange_backward_fake(gradient, sends, receives, original, group_key):
    return gradient.new_empty(original.shape)


def _exchange_context(ctx, inputs, output):
    rows, _, sends, ctx.group_key = inputs
    received, ids, receives = output
    ctx.save_for_backward(sends, receives, rows, received)
    ctx.mark_non_differentiable(ids, receives)
    ctx.set_materialize_grads(False)


@once_differentiable
def _exchange_gradient(ctx, gradient, _ids, _counts):
    sends, receives, original, received = ctx.saved_tensors
    if gradient is None:
        # Other ranks can still need derivatives from this owner.
        gradient = torch.zeros_like(received)
    return exchange_backward(gradient, sends, receives, original, ctx.group_key), None, None, None


exchange.register_autograd(_exchange_gradient, setup_context=_exchange_context)


@torch.library.custom_op('ruda::expert_return', mutates_args=())
def return_rows(rows: torch.Tensor, sends: torch.Tensor, receives: torch.Tensor,
                original: torch.Tensor, group_key: str) -> torch.Tensor:
    group = _group(group_key)
    send_rows = _counts(sends, group.world_size, original.shape[0])
    receive_rows = _counts(receives, group.world_size, rows.shape[0])
    return group.all_to_all(rows, receive_rows, send_rows)


@return_rows.register_fake
def _return_fake(rows, sends, receives, original, group_key):
    return rows.new_empty((original.shape[0], rows.shape[1]))


@torch.library.custom_op('ruda::expert_return_backward', mutates_args=())
def return_backward(gradient: torch.Tensor, sends: torch.Tensor,
                    receives: torch.Tensor, original: torch.Tensor, group_key: str) -> torch.Tensor:
    group = _group(group_key)
    send_rows = _counts(sends, group.world_size, gradient.shape[0])
    receive_rows = _counts(receives, group.world_size, original.shape[0])
    return group.all_to_all(gradient, send_rows, receive_rows)


@return_backward.register_fake
def _return_backward_fake(gradient, sends, receives, original, group_key):
    return gradient.new_empty(original.shape)


def _return_context(ctx, inputs, output):
    rows, sends, receives, original, ctx.group_key = inputs
    ctx.save_for_backward(rows, sends, receives, original)
    ctx.set_materialize_grads(False)


@once_differentiable
def _return_gradient(ctx, gradient):
    rows, sends, receives, original = ctx.saved_tensors
    if gradient is None:
        gradient = rows.new_zeros((original.shape[0], rows.shape[1]))
    return return_backward(gradient, sends, receives, rows, ctx.group_key), None, None, None, None


return_rows.register_autograd(_return_gradient, setup_context=_return_context)
