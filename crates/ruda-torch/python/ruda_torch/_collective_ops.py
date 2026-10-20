"""Captured tensor-parallel primitives with explicit logical replica adjoints."""
from __future__ import annotations

import torch
from ._expert_exchange import _group, register_group


def group_key(group):
    key = getattr(group, '_ruda_compile_collective_key', None)
    if key is None:
        key = register_group(f'collective:{id(group)}', group)
        group._ruda_compile_collective_key = key
    return key


@torch.library.custom_op('ruda::parallel_region', mutates_args=())
def parallel_region(value: torch.Tensor, key: str, kind: int, axis: int) -> torch.Tensor:
    group = _group(key)
    if kind == 0:
        return value.clone()
    if kind == 1:
        return group.sum_(value.detach().clone())
    if kind == 2:
        return group.all_gather(value, axis=axis)
    if kind != 3 or value.ndim == 0 or value.shape[axis] % group.world_size:
        raise ValueError('invalid tensor-parallel region or nondivisible shard dimension')
    size = value.shape[axis] // group.world_size
    return value.narrow(axis, size * group.rank, size).contiguous().clone()


@parallel_region.register_fake
def _parallel_fake(value, key, kind, axis):
    shape = list(value.shape)
    if kind in (2, 3):
        world = _group(key).world_size
        if kind == 2:
            shape[axis] *= world
        else:
            torch._check(shape[axis] % world == 0)
            shape[axis] //= world
    return value.new_empty(shape)


def _parallel_context(ctx, inputs, output):
    _, ctx.key, ctx.kind, ctx.axis = inputs


def _parallel_gradient(ctx, gradient):
    inverse = (1, 0, 3, 2)[ctx.kind]
    return parallel_region(gradient, ctx.key, inverse, ctx.axis), None, None, None


parallel_region.register_autograd(_parallel_gradient, setup_context=_parallel_context)


@torch.library.custom_op('ruda::collective_reduce', mutates_args=())
def collective_reduce(value: torch.Tensor, operation: int, key: str) -> torch.Tensor:
    result = value.detach().clone()
    group = _group(key)
    if operation == 0:
        return group.sum_(result)
    if operation == 1:
        return group.max_(result)
    if operation == 2:
        return group.min_(result)
    raise ValueError('invalid collective reduction')


@collective_reduce.register_fake
def _reduce_fake(value, operation, key):
    return torch.empty_like(value)
