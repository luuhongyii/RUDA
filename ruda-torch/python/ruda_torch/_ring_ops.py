"""Native ring attention dispatcher and peer circulation, without KV all-gather."""
from __future__ import annotations

import ctypes
from threading import RLock
import torch
from torch.autograd.function import once_differentiable
from ._expert_exchange import _group

_lock = RLock()
_ready = False
_indexed_ready = False


def _native(indexed=False):
    global _ready, _indexed_ready
    from . import _C, _native as library
    with _lock:
        if not _ready:
            if not hasattr(library, 'ruda_torch_ring_attention_api_version') or not hasattr(_C, 'initialize_ring_attention'):
                raise RuntimeError('ring attention requires matching rebuilt native libraries')
            library.ruda_torch_ring_attention_api_version.restype = ctypes.c_uint32
            if library.ruda_torch_ring_attention_api_version() != 1 or _C.ring_attention_api_version != 1:
                raise RuntimeError('RUDA ring attention API mismatch')
            _C.initialize_ring_attention(ctypes.cast(library.ruda_torch_ring_attention, ctypes.c_void_p).value)
            _ready = True
        if indexed and not _indexed_ready:
            version = getattr(library, 'ruda_torch_indexed_ring_attention_api_version', None)
            if version is None or getattr(_C, 'indexed_ring_attention_api_version', 0) != 1:
                raise RuntimeError('globally indexed ring attention requires matching rebuilt native libraries')
            version.restype = ctypes.c_uint32
            if version() != 1:
                raise RuntimeError('RUDA indexed ring attention API mismatch')
            _indexed_ready = True
    return _C


def _rotate(tensors, receive_rows, group):
    if group.world_size == 1:
        return tensors
    sends = [0] * group.world_size
    receives = [0] * group.world_size
    sends[(group.rank + 1) % group.world_size] = tensors[0].shape[0]
    receives[(group.rank - 1) % group.world_size] = receive_rows
    # Sparse peer partitions implement a one-hop ring on either native ruCCL
    # transport. No other peer receives this rank's block.
    return tuple(group.all_to_all(tensor, sends, receives) for tensor in tensors)


def _metadata(position, document, valid, batch, length, start, total, device, query,
              global_heads=0, head_start=0, global_keys=0):
    ordinal = torch.arange(length, device=device, dtype=torch.int64) + start
    ordinal = ordinal.expand(batch, length)
    batches = torch.arange(batch, device=device, dtype=torch.int64)[:, None]
    position = ordinal if position.numel() == 0 else position
    document = batches.expand(batch, length) if document.numel() == 0 else document
    if valid.numel():
        document = torch.where(valid, document, torch.full_like(document, -1))
    if query and global_heads:
        origin = ((batches * global_heads + head_start) * total + ordinal) * global_keys
        stride = torch.full_like(origin, total * global_keys)
        return torch.stack((document, position, origin, stride), -1).reshape(batch * length, 4).contiguous()
    rng_index = ordinal + batches * total if query else ordinal
    return torch.stack((document, position, rng_index), -1).reshape(batch * length, 3).contiguous()


def _call(op, tensors, batch, policy, state):
    global_keys, seed_low, seed_high, counter_low, counter_high = state[:5].tolist()
    causal, left, right, scale, dropout, softcap = policy
    _native(op != 2 and tensors[3].shape[1] == 4).ring_attention(op, tensors, batch, global_keys, causal, left, right,
        scale, dropout, softcap, seed_low | seed_high << 32, counter_low | counter_high << 32)


@torch.library.custom_op('ruda::ring_attention_forward', mutates_args=(),
                         tags=(torch.Tag.nondeterministic_seeded,))
def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            qposition: torch.Tensor, kposition: torch.Tensor, qdocument: torch.Tensor, kdocument: torch.Tensor,
            qvalid: torch.Tensor, kvalid: torch.Tensor, group_key: str,
            causal: bool, left: int, right: int, scale: float, dropout: float, softcap: float,
            requested_rng: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _forward(q, k, v, qposition, kposition, qdocument, kdocument, qvalid, kvalid,
        group_key, causal, left, right, scale, dropout, softcap, requested_rng)


@torch.library.custom_op('ruda::indexed_ring_attention_forward', mutates_args=(),
                         tags=(torch.Tag.nondeterministic_seeded,))
def indexed_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            qposition: torch.Tensor, kposition: torch.Tensor, qdocument: torch.Tensor, kdocument: torch.Tensor,
            qvalid: torch.Tensor, kvalid: torch.Tensor, group_key: str,
            causal: bool, left: int, right: int, scale: float, dropout: float, softcap: float,
            requested_rng: torch.Tensor, global_heads: int, head_start: int, tensor_key: str
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if global_heads <= 0:
        raise ValueError('indexed ring attention requires positive global query heads')
    return _forward(q, k, v, qposition, kposition, qdocument, kdocument, qvalid, kvalid,
        group_key, causal, left, right, scale, dropout, softcap, requested_rng,
        global_heads, head_start, tensor_key)


def _forward(q, k, v, qposition, kposition, qdocument, kdocument, qvalid, kvalid,
             group_key, causal, left, right, scale, dropout, softcap, requested_rng,
             global_heads=0, head_start=0, tensor_key=''):
    _native(bool(global_heads))
    group = _group(group_key)
    batch, queries, heads, dimension = q.shape
    keys, kv_heads, value_dimension = k.shape[1], k.shape[2], v.shape[3]
    policy = causal, left, right, scale, dropout, softcap
    contract = batch, heads, kv_heads, dimension, value_dimension, str(q.dtype), policy, requested_rng.tolist(), global_heads, head_start
    peers = group.gather_metadata((queries, keys, contract))
    if any(other != contract for _, _, other in peers):
        raise ValueError('ring peers disagree on batch, head geometry, dtype or attention policy')
    query_lengths = [item[0] for item in peers]
    key_lengths = [item[1] for item in peers]
    total_queries, total_keys = sum(query_lengths), sum(key_lengths)
    if total_keys > 0xffffffff or not global_heads and batch * total_queries * heads > 0xffffffff:
        raise ValueError('ring global positions exceed native Philox row indexing')
    if global_heads:
        if head_start < 0 or head_start + heads > global_heads:
            raise ValueError('ring local query heads must lie inside the global head interval')
        from ._parallel_rng import attention_state
        words = attention_state((batch, global_heads, total_queries, total_keys), dropout,
            (group_key, tensor_key), requested_rng).tolist()
    qmeta = _metadata(qposition, qdocument, qvalid, batch, queries,
        sum(query_lengths[:group.rank]), total_queries, q.device, True, global_heads, head_start, total_keys)
    kmeta = _metadata(kposition, kdocument, kvalid, batch, keys,
        sum(key_lengths[:group.rank]), total_keys, k.device, False)
    if not global_heads:
        words = requested_rng.tolist()
    if not words:
        local = None
        if group.rank == 0:
            if 0 < dropout < 1:
                from ._varlen_ops import reserve
                local = reserve(batch * total_queries * heads * ((total_keys + 3) // 4))
            else:
                local = [0, 0, 0, 0]
        words = group.gather_metadata(local)[0]
    if len(words) != 4 or any(type(word) is not int or not 0 <= word <= 0xffffffff for word in words):
        raise ValueError('ring dropout state requires four unsigned 32-bit words')
    state = torch.tensor([total_keys, *words, *key_lengths], device='cpu', dtype=torch.int64)
    query = q.reshape(batch * queries, heads, dimension)
    key = k.reshape(batch * keys, kv_heads, dimension)
    value = v.reshape(batch * keys, kv_heads, value_dimension)
    accumulator = torch.zeros((batch * queries, heads, value_dimension), device=q.device, dtype=torch.float32)
    stats = torch.zeros((batch * queries * heads, 2), device=q.device, dtype=torch.float32)
    block = key, value, kmeta
    for step in range(group.world_size):
        current_key, current_value, current_meta = block
        _call(0, [query, current_key, current_value, qmeta, current_meta, accumulator, stats], batch, policy, state)
        if step + 1 < group.world_size:
            owner = (group.rank - step - 1) % group.world_size
            block = _rotate(block, batch * key_lengths[owner], group)
    output = q.new_empty((batch * queries, heads, value_dimension))
    _call(2, [accumulator, stats, output], batch, policy, state)
    return output.reshape(batch, queries, heads, value_dimension), accumulator, stats, qmeta, kmeta, state


@forward.register_fake
def _forward_fake(q, k, v, qposition, kposition, qdocument, kdocument,
                  qvalid, kvalid, group_key, causal, left, right, scale, dropout, softcap, requested_rng):
    batch, queries, heads, _ = q.shape
    width = v.shape[3]
    return (q.new_empty((batch, queries, heads, width)),
        torch.empty((batch * queries, heads, width), device=q.device, dtype=torch.float32),
        torch.empty((batch * queries * heads, 2), device=q.device, dtype=torch.float32),
        torch.empty((batch * queries, 3), device=q.device, dtype=torch.int64),
        torch.empty((k.shape[0] * k.shape[1], 3), device=q.device, dtype=torch.int64),
        torch.empty(5 + _group(group_key).world_size, device='cpu', dtype=torch.int64))


@indexed_forward.register_fake
def _indexed_forward_fake(q, k, v, qposition, kposition, qdocument, kdocument,
                  qvalid, kvalid, group_key, causal, left, right, scale, dropout, softcap,
                  requested_rng, global_heads, head_start, tensor_key):
    result = list(_forward_fake(q, k, v, qposition, kposition, qdocument, kdocument,
        qvalid, kvalid, group_key, causal, left, right, scale, dropout, softcap, requested_rng))
    result[3] = torch.empty((q.shape[0] * q.shape[1], 4), device=q.device, dtype=torch.int64)
    return tuple(result)


@torch.library.custom_op('ruda::ring_attention_backward', mutates_args=())
def backward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gradient: torch.Tensor,
             output: torch.Tensor, stats: torch.Tensor, qmeta: torch.Tensor, kmeta: torch.Tensor,
             state: torch.Tensor, group_key: str, causal: bool, left: int, right: int,
             scale: float, dropout: float, softcap: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    group = _group(group_key)
    batch, queries, heads, dimension = q.shape
    keys, kv_heads, value_dimension = k.shape[1], k.shape[2], v.shape[3]
    query = q.reshape(batch * queries, heads, dimension)
    key = k.reshape(batch * keys, kv_heads, dimension)
    value = v.reshape(batch * keys, kv_heads, value_dimension)
    gradient = gradient.contiguous().reshape(batch * queries, heads, value_dimension)
    dq = torch.zeros(query.shape, device=q.device, dtype=torch.float32)
    dk = torch.zeros(key.shape, device=q.device, dtype=torch.float32)
    dv = torch.zeros(value.shape, device=q.device, dtype=torch.float32)
    key_lengths = state[5:].tolist()
    policy = causal, left, right, scale, dropout, softcap
    block = key, value, kmeta, dk, dv
    for step in range(group.world_size):
        current_key, current_value, current_meta, current_dk, current_dv = block
        _call(1, [query, current_key, current_value, qmeta, current_meta, gradient,
                  output, stats, dq, current_dk, current_dv], batch, policy, state)
        # A complete circulation returns each accumulated dK/dV to its owner.
        owner = (group.rank - step - 1) % group.world_size
        block = _rotate(block, batch * key_lengths[owner], group)
    return dq.to(q.dtype).reshape(q.shape), block[3].to(k.dtype).reshape(k.shape), block[4].to(v.dtype).reshape(v.shape)


@backward.register_fake
def _backward_fake(q, k, v, gradient, output, stats, qmeta, kmeta, state,
                   group_key, causal, left, right, scale, dropout, softcap):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _context(ctx, inputs, result):
    q, k, v = inputs[:3]
    _, output, stats, qmeta, kmeta, state = result
    ctx.save_for_backward(q, k, v, output, stats, qmeta, kmeta, state)
    ctx.options = inputs[9:16]
    ctx.argument_count = len(inputs)
    ctx.mark_non_differentiable(output, stats, qmeta, kmeta, state)


@once_differentiable
def _gradient(ctx, gradient, _output, _stats, _qmeta, _kmeta, _state):
    q, k, v, output, stats, qmeta, kmeta, state = ctx.saved_tensors
    dq, dk, dv = backward(q, k, v, gradient, output, stats, qmeta, kmeta, state, *ctx.options)
    return dq, dk, dv, *(None for _ in range(ctx.argument_count - 3))


forward.register_autograd(_gradient, setup_context=_context)
indexed_forward.register_autograd(_gradient, setup_context=_context)
