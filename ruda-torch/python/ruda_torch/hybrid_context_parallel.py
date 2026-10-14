"""Context-sharded Kimi Linear and Qwen3.5 hybrid model execution."""
from __future__ import annotations

import torch
from torch.nn import functional as F
from ._architecture_ops import precision_context
from .context_recurrent import context_parallel_kda, context_parallel_delta, context_parallel_causal_conv1d


def kimi_recurrent_forward(module, hidden, layout):
    context = layout.context
    group, key = context['group'], context['key']
    docs, valid = context['documents'], context['valid']
    batch, length = hidden.shape[:2]
    projected = [projection(hidden) for projection in (module.q_proj, module.k_proj, module.v_proj)]
    mixed = torch.cat(projected, -1)
    weights = torch.cat([convolution.weight.to(mixed.dtype) for convolution in
                         (module.q_conv1d, module.k_conv1d, module.v_conv1d)], 0)
    mixed = context_parallel_causal_conv1d(mixed, weights, None, group,
        documents=docs, valid=valid, skip_padding=True, activation='silu', group_key=key)
    q, k, v = (value.reshape(batch, length, module.num_heads, module.head_dim)
               for value in mixed.split(module.num_heads * module.head_dim, -1))
    raw_decay = module.f_b_proj(module.f_a_proj(hidden))
    raw_beta = module.b_proj(hidden)
    with precision_context(hidden):
        q = (q.float() * (q.float().square().sum(-1, keepdim=True) + 1e-6).rsqrt()).to(q.dtype)
        k = (k.float() * (k.float().square().sum(-1, keepdim=True) + 1e-6).rsqrt()).to(k.dtype)
        decay = -F.softplus(raw_decay.float() + module.dt_bias.float())
        decay = decay.reshape(batch, length, module.num_heads, module.head_dim) * module.A_log.float().exp()
        beta = raw_beta.float().sigmoid()
    output, _ = context_parallel_kda(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        decay.transpose(1, 2), beta.transpose(1, 2), group, documents=docs, valid=valid,
        normalize_qk=False, group_key=key)
    gate = module.g_b_proj(module.g_a_proj(hidden)).reshape(batch, length, module.num_heads, module.head_dim)
    output = module.o_norm(output.transpose(1, 2), gate).reshape(batch, length, module.num_heads * module.head_dim)
    return module.o_proj(output), None


def kimi_mla_forward(module, hidden, layout):
    from .context_parallel import ring_scaled_dot_product_attention
    from ._parallel_rng import module_partition
    context = layout.context
    batch, length = hidden.shape[:2]
    q = module.q_proj(hidden).reshape(batch, length, module.heads, module.query_dim)
    compressed, shared = module.kv_a_proj_with_mqa(hidden).split((module.rank, module.shared), -1)
    expanded = module.kv_b_proj(module.kv_a_layernorm(compressed))
    k, v = expanded.reshape(batch, length, module.heads, module.nope + module.value_dim).split((module.nope, module.value_dim), -1)
    k = torch.cat((k, shared[:, :, None, :].expand(batch, length, module.heads, module.shared)), -1)
    partition = module_partition(module, q.shape[2])
    sequence_group = getattr(layout, 'sequence_group', None)
    if not partition and sequence_group is not None:
        partition = dict(tensor_group=sequence_group, global_heads=q.shape[2], head_start=0)
    output = ring_scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        context['group'], group_key=context['key'], is_causal=True, scale=module.query_dim ** -.5,
        query_positions=context['positions'], key_positions=context['positions'],
        query_documents=context['documents'], key_documents=context['documents'],
        query_mask=context['valid'], key_mask=context['valid'],
        dropout_p=module.dropout if module.training else 0., **partition)
    return module.o_proj(output.transpose(1, 2).reshape(batch, length, module.heads * module.value_dim)), None


def enable_kimi_context_parallel(model, group, *, group_key='context-ring'):
    from ._expert_exchange import register_group
    from .transformers_kimi import KimiLinearModel
    backbones = [module for module in model.modules() if isinstance(module, KimiLinearModel)]
    if not backbones:
        raise ValueError('Kimi context adapter requires the RUDA Kimi Linear backbone')
    key = register_group(group_key, group)
    for backbone in backbones:
        backbone._ruda_context_group, backbone._ruda_context_key = group, key
    return model


def qwen35_recurrent_forward(module, hidden, attention_mask, metadata):
    if metadata.get('ruda_context_positions') is None or metadata.get('ruda_context_documents') is None:
        raise ValueError('Qwen3.5 context recurrence requires global positions and document IDs')
    valid = metadata.get('ruda_context_valid', attention_mask)
    if valid is not None:
        if valid.shape != hidden.shape[:2] or valid.device != hidden.device:
            raise ValueError('Qwen3.5 context padding mask differs from local tokens')
        hidden = (hidden * valid[:, :, None]).to(hidden.dtype)
    group, key = module._ruda_context_group, module._ruda_context_key
    docs = metadata['ruda_context_documents']
    batch, length = hidden.shape[:2]
    mixed = module.in_proj_qkv(hidden)
    z = module.in_proj_z(hidden).reshape(batch, length, module.num_v_heads, module.head_v_dim)
    beta, a = module.in_proj_b(hidden).sigmoid(), module.in_proj_a(hidden)
    mixed = context_parallel_causal_conv1d(mixed, module.conv1d.weight.to(mixed.dtype),
        None if module.conv1d.bias is None else module.conv1d.bias.to(mixed.dtype), group,
        documents=docs, valid=valid, skip_padding=False, activation='silu', group_key=key)
    q, k, v = mixed.split((module.key_dim, module.key_dim, module.value_dim), -1)
    q, k = (value.reshape(batch, length, module.num_k_heads, module.head_k_dim) for value in (q, k))
    v = v.reshape(batch, length, module.num_v_heads, module.head_v_dim)
    copies = module.num_v_heads // module.num_k_heads
    if copies != 1:
        q, k = q.repeat_interleave(copies, 2), k.repeat_interleave(copies, 2)
    with precision_context(a):
        decay = -module.A_log.float().exp() * F.softplus(a.float() + module.dt_bias.float())
    # Qwen masks the projection input but still advances convolution/recurrence
    # at padding positions. This differs from Kimi's live-token compaction.
    output, _ = context_parallel_delta(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        beta.transpose(1, 2), decay.transpose(1, 2), group, documents=docs,
        normalize_qk=True, norm_eps=1e-6, group_key=key)
    output = output.transpose(1, 2)
    dtype = output.dtype
    with precision_context(output):
        value = output.float()
        value = value * (value.square().mean(-1, keepdim=True) + module.layer_norm_epsilon).rsqrt()
        output = (module.norm.weight * value.to(dtype) * F.silu(z.float())).to(dtype)
    return module.out_proj(output.reshape(batch, length, module.value_dim))


def bind_qwen35_context_recurrence(model, group, key):
    from .transformers_qwen35 import enable_transformers_qwen35
    enable_transformers_qwen35(model)
    for module in model.modules():
        if type(module).__name__ in ('Qwen3_5GatedDeltaNet', 'Qwen3_5MoeGatedDeltaNet'):
            module._ruda_context_group, module._ruda_context_key = group, key
    return model
