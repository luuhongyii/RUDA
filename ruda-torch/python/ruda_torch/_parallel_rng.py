"""Reserve one checkpointed Philox interval for a logical attention tensor."""
from __future__ import annotations

import math
import torch
from ._expert_exchange import _group


def head_partition(heads, group, global_heads, head_start):
    groups = []
    if group is not None:
        if group.device_type != 'ruda':
            raise ValueError('attention tensor group must communicate RUDA tensors')
        from ._collective_ops import group_key
        groups.append(group_key(group))
        global_heads = heads * group.world_size if global_heads is None else global_heads
        head_start = heads * group.rank if head_start is None else head_start
    else:
        global_heads = heads if global_heads is None else global_heads
        head_start = 0 if head_start is None else head_start
    if type(global_heads) is not int or type(head_start) is not int or global_heads < heads or not 0 <= head_start <= global_heads - heads:
        raise ValueError('invalid global query-head interval')
    if group is None and (global_heads != heads or head_start):
        raise ValueError('head-sharded dropout requires its tensor process group')
    return global_heads, head_start, groups


def module_partition(module, local_heads):
    projection = getattr(module, 'q_proj', getattr(module, 'q_b_proj', None))
    global_heads = getattr(module, '_ruda_global_num_heads', None)
    group = getattr(projection, 'group', None)
    if global_heads is not None:
        group = getattr(module, '_ruda_sequence_group', group)
        return dict(tensor_group=group, global_heads=global_heads,
                    head_start=module._ruda_head_start)
    if group is not None and getattr(projection, 'gather_output', True) is False:
        if getattr(projection, 'replicas', 1) != 1:
            raise ValueError('replicated query-head projection needs explicit global head metadata')
        return dict(tensor_group=group, global_heads=local_heads * group.world_size,
                    head_start=local_heads * group.rank)
    group = getattr(module, '_ruda_sequence_group', group)
    if group is not None:
        return dict(tensor_group=group, global_heads=local_heads, head_start=0)
    return {}


def attention_state(shape, dropout, group_keys, requested):
    if len(shape) != 4 or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError('attention RNG requires global [batch,heads,queries,keys] geometry')
    scores = math.prod(shape)
    if scores >= 1 << 63:
        raise OverflowError('global attention RNG positions exceed int64 metadata')
    if not 0 <= dropout <= 1:
        raise ValueError('invalid attention dropout probability')
    if requested.device.type != 'cpu' or requested.dtype != torch.int64 or requested.shape not in ((0,), (4,)):
        raise ValueError('attention RNG request requires CPU int64[0 or 4]')
    words = requested.tolist()
    if any(type(word) is not int or not 0 <= word <= 0xffffffff for word in words):
        raise ValueError('attention RNG words must be unsigned 32-bit integers')
    groups = []
    for key in group_keys:
        if key:
            group = _group(key)
            if all(group is not other for other in groups):
                groups.append(group)
    blocks = (scores + 3) // 4 if 0 < dropout < 1 else 0
    if not blocks:
        return torch.tensor(words if words else [0, 0, 0, 0], dtype=torch.int64, device='cpu')
    contract = tuple(shape), float(dropout), tuple(words)
    from .random import default_generator
    with default_generator.lock:
        seed, counter = default_generator.seed_value, default_generator.counter
        # Axes must be visited in the same order on a rectangular CP/TP mesh.
        # Root selection and MAX are propagated across axes, not just the last
        # group; every participant advances its saved default-generator state.
        mismatch = False
        for group in groups:
            peers = group.gather_metadata((contract, seed, counter, mismatch))
            mismatch = any(peer[0] != contract or peer[3] for peer in peers)
            seed, counter = peers[0][1], max(peer[2] for peer in peers)
        # Propagate a mismatch discovered on a later axis back to the others
        # before any participant starts a device collective.
        for group in reversed(groups[:-1]):
            mismatch = any(group.gather_metadata(mismatch))
        if mismatch:
            raise ValueError('parallel attention peers disagree on logical RNG geometry or explicit state')
        if words:
            seed = words[0] | words[1] << 32
            counter = words[2] | words[3] << 32
        if counter + blocks >= 1 << 64:
            raise OverflowError('parallel attention Philox counter exhausted')
        if not words:
            default_generator.seed_value = seed
            default_generator.counter = counter + blocks
            words = [seed & 0xffffffff, seed >> 32, counter & 0xffffffff, counter >> 32]
    return torch.tensor(words, dtype=torch.int64, device='cpu')


@torch.library.custom_op('ruda::attention_rng_state', mutates_args=(), tags=(torch.Tag.nondeterministic_seeded,),
    schema='(Tensor anchor, SymInt[] shape, str[] groups, float dropout, Tensor requested) -> Tensor')
def reserve_attention(anchor: torch.Tensor, shape: list[int], groups: list[str], dropout: float,
                      requested: torch.Tensor) -> torch.Tensor:
    return attention_state(tuple(shape), dropout, groups, requested)


@reserve_attention.register_fake
def _reserve_attention_fake(anchor, shape, groups, dropout, requested):
    return torch.empty(4, dtype=torch.int64, device='cpu')
