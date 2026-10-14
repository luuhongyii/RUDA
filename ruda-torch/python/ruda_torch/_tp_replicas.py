"""Contiguous TP head replicas and their logical projection adjoints."""
from __future__ import annotations

import torch
from ._expert_exchange import _group
from ._collective_ops import group_key


def coordinates(group, replicas):
    if type(replicas) is not int or replicas < 1 or group.world_size % replicas:
        raise ValueError('TP replica count must divide the tensor group')
    return group.rank // replicas, group.world_size // replicas


def shard_coordinates(group, replicas):
    from types import SimpleNamespace
    rank, world = coordinates(group, replicas)
    return SimpleNamespace(rank=rank, world_size=world)


def _exchange_replicas(value, key, replicas, reduce):
    group = _group(key)
    coordinates(group, replicas)
    work = value.detach().float().contiguous().clone() if reduce and value.dtype in (torch.float16, torch.bfloat16) else value.detach().contiguous().clone()
    if replicas == 1 or not value.numel():
        return work.to(value.dtype)
    lane, base = group.rank % replicas, group.rank // replicas * replicas
    shape = work.shape
    work = work.reshape(1, -1)
    steps = []
    step = 1
    while step < replicas:
        steps.append(step)
        if not reduce:
            step *= 2
            continue
        sends, receives = [0] * group.world_size, [0] * group.world_size
        if lane % (2 * step) == step:
            sends[group.rank - step] = 1
        elif lane % (2 * step) == 0 and lane + step < replicas:
            receives[group.rank + step] = 1
        received = group.all_to_all(work[:sum(sends)], sends, receives)
        if sum(receives):
            work = work + received
        step *= 2
    for step in reversed(steps):
        sends, receives = [0] * group.world_size, [0] * group.world_size
        if lane % (2 * step) == 0 and lane + step < replicas:
            sends[base + lane + step] = 1
        elif lane % (2 * step) == step:
            receives[base + lane - step] = 1
        received = group.all_to_all(work[:sum(sends)], sends, receives)
        if sum(receives):
            work = received
    return work.reshape(shape).to(value.dtype)


@torch.library.custom_op('ruda::tp_replica_sum', mutates_args=())
def replica_sum(value: torch.Tensor, key: str, replicas: int) -> torch.Tensor:
    return _exchange_replicas(value, key, replicas, True)


@replica_sum.register_fake
def _sum_fake(value, key, replicas):
    coordinates(_group(key), replicas)
    return torch.empty_like(value)


@torch.library.custom_op('ruda::tp_replica_broadcast', mutates_args=())
def replica_broadcast(value: torch.Tensor, key: str, replicas: int) -> torch.Tensor:
    return _exchange_replicas(value, key, replicas, False)


replica_broadcast.register_fake(_sum_fake)


def _sum_context(ctx, inputs, output):
    _, ctx.key, ctx.replicas = inputs


def _sum_backward(ctx, gradient):
    return replica_sum(gradient, ctx.key, ctx.replicas), None, None


replica_sum.register_autograd(_sum_backward, setup_context=_sum_context)


@torch.library.custom_op('ruda::tp_unique_gather', mutates_args=())
def unique_gather(value: torch.Tensor, key: str, replicas: int) -> torch.Tensor:
    group = _group(key)
    _, shards = coordinates(group, replicas)
    full = group.all_gather(value, axis=-1)
    return full.reshape(*value.shape[:-1], shards, replicas, value.shape[-1])[..., 0, :].reshape(
        *value.shape[:-1], shards * value.shape[-1]).contiguous().clone()


@unique_gather.register_fake
def _gather_fake(value, key, replicas):
    _, shards = coordinates(_group(key), replicas)
    return value.new_empty((*value.shape[:-1], value.shape[-1] * shards))


def _gather_context(ctx, inputs, output):
    value, key, replicas = inputs
    ctx.start = coordinates(_group(key), replicas)[0] * value.shape[-1]
    ctx.width = value.shape[-1]


def _gather_backward(ctx, gradient):
    return gradient.narrow(-1, ctx.start, ctx.width).contiguous(), None, None


unique_gather.register_autograd(_gather_backward, setup_context=_gather_context)


class _GradientScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, replicas):
        ctx.replicas = replicas
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return gradient / ctx.replicas, None


class _ReplicaGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, key, replicas):
        ctx.key, ctx.replicas = key, replicas
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return replica_sum(gradient, ctx.key, ctx.replicas), None, None


def scale_input_gradient(value, replicas):
    return value if replicas == 1 else _GradientScale.apply(value, replicas)


def column_output(value, group, replicas, gather):
    from .parallel_training import gather_from_tensor_parallel
    if replicas == 1:
        return gather_from_tensor_parallel(value, group) if gather else value
    key = group_key(group)
    return unique_gather(value, key, replicas) if gather else _ReplicaGradient.apply(value, key, replicas)


def mark(parameter, replicas):
    if parameter is not None:
        parameter._ruda_tp_sharded = True
        parameter._ruda_tp_replicas = replicas


def count(parameter):
    return getattr(parameter, '_ruda_tp_replicas', 1)


def column_input(module, value):
    from .parallel_training import copy_to_tensor_parallel
    value = scale_input_gradient(value, getattr(module, 'replicas', 1))
    return value if hasattr(module, '_ruda_sequence_axis') else copy_to_tensor_parallel(value, module.group)


def row_output(module, value):
    from .parallel_training import reduce_from_tensor_parallel
    if hasattr(module, '_ruda_sequence_axis'):
        from .sequence_parallel import _reduce_scatter
        return _reduce_scatter(value, module._ruda_sequence_axis, module._ruda_sequence_key)
    return reduce_from_tensor_parallel(value, module.group)


@torch.no_grad()
def initialize_replicas(model, group):
    from .distributed_checkpoint import model_shard_layout
    from .sharded_training import FullyShardedModule
    layout = model_shard_layout(model)
    for path, unit in model.named_modules():
        if isinstance(unit, FullyShardedModule):
            prefix = path + '.' if path else ''
            for name, index, *_ in unit._schema:
                replicas = unit._template_layout.get(name, {}).get('tp_replicas', 1)
                layout.setdefault(prefix + unit._shard_names[index], {})['tp_replicas'] = replicas
            for name, key in unit._buffer_names:
                replicas = unit._template_layout.get(name, {}).get('tp_replicas', 1)
                layout.setdefault(prefix + key, {})['tp_replicas'] = replicas
    key = group_key(group)
    for name, value in (*model.named_parameters(), *model.named_buffers()):
        replicas = layout.get(name, {}).get('tp_replicas', count(value))
        if replicas > 1:
            value.copy_(replica_broadcast(value, key, replicas))
