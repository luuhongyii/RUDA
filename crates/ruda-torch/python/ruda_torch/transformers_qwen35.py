"""Opt-in RUDA DeltaNet and grouped-expert execution for HF Qwen3.5 models."""
from __future__ import annotations

from types import MethodType
import torch
from torch.nn import functional as F
from ._architecture_ops import precision_context
from ._recurrent_ops import causal_depthwise_conv1d
from .sequence_training import gated_delta_rule


def _delta_forward(module, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    if hasattr(module, '_ruda_context_group'):
        if cache_params is not None:
            raise ValueError('context-sharded Qwen3.5 recurrence does not use a replicated decode cache')
        from .hybrid_context_parallel import qwen35_recurrent_forward
        return qwen35_recurrent_forward(module, hidden_states, attention_mask, kwargs)
    if hidden_states.device.type != 'ruda':
        return module._ruda_original_forward(hidden_states, cache_params=cache_params,
                                              attention_mask=attention_mask, **kwargs)
    boundaries = kwargs.get('cu_seq_lens_q')
    if boundaries is None:
        boundaries = kwargs.get('cu_seqlens')
    if boundaries is not None:
        from .varlen_attention import _boundaries
        if hidden_states.shape[0] != 1 or cache_params is not None:
            raise ValueError('packed Qwen3.5 DeltaNet requires one stream without a decode cache')
        offsets = _boundaries(boundaries.detach().to('cpu'), hidden_states.shape[1])
        outputs = []
        for begin, end in zip(offsets, offsets[1:]):
            if begin != end:
                mask = None if attention_mask is None else attention_mask[:, begin:end]
                outputs.append(_delta_forward(module, hidden_states[:, begin:end], attention_mask=mask))
        return torch.cat(outputs, 1) if outputs else hidden_states * 0
    if attention_mask is not None:
        if attention_mask.shape != hidden_states.shape[:2] or attention_mask.device != hidden_states.device:
            raise ValueError('Qwen3.5 recurrent padding mask must match current tokens')
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)
    batch, length = hidden_states.shape[:2]
    previous = cache_params is not None and cache_params.has_previous_state(module.layer_idx, state_idx=0)
    initial = cache_params.layers[module.layer_idx].recurrent_states[0] if previous else None
    if initial is not None and torch.is_grad_enabled():
        initial = initial.clone()
    mixed = module.in_proj_qkv(hidden_states)
    z = module.in_proj_z(hidden_states).reshape(batch, length, module.num_v_heads, module.head_v_dim)
    beta = module.in_proj_b(hidden_states).sigmoid()
    a = module.in_proj_a(hidden_states)
    if cache_params is not None:
        mixed = cache_params.update_conv_state(mixed.transpose(1, 2), module.layer_idx,
                                               conv_kernel_size=module.conv_kernel_size).transpose(1, 2)
    weight = module.conv1d.weight.to(mixed.dtype)
    bias = None if module.conv1d.bias is None else module.conv1d.bias.to(mixed.dtype)
    mixed, _ = causal_depthwise_conv1d(mixed, weight, bias)
    mixed = F.silu(mixed[:, -length:]) if length else mixed[:, :0]
    query, key, value = mixed.split((module.key_dim, module.key_dim, module.value_dim), -1)
    query = query.reshape(batch, length, module.num_k_heads, module.head_k_dim)
    key = key.reshape(batch, length, module.num_k_heads, module.head_k_dim)
    value = value.reshape(batch, length, module.num_v_heads, module.head_v_dim)
    copies = module.num_v_heads // module.num_k_heads
    if copies != 1:
        query, key = query.repeat_interleave(copies, 2), key.repeat_interleave(copies, 2)
    with precision_context(a):
        decay = -module.A_log.float().exp() * F.softplus(a.float() + module.dt_bias.float())
    output, recurrent = gated_delta_rule(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
        beta.transpose(1, 2), decay.transpose(1, 2), initial_state=initial,
        normalize_qk=True, norm_eps=1e-6, chunk_size=64, native_forward=True)
    if cache_params is not None:
        cache_params.update_recurrent_state(recurrent, module.layer_idx)
    output = output.transpose(1, 2)
    dtype = output.dtype
    with precision_context(output):
        value = output.float()
        value = value * (value.square().mean(-1, keepdim=True) + module.layer_norm_epsilon).rsqrt()
        output = (module.norm.weight * value.to(dtype) * F.silu(z.float())).to(dtype)
    return module.out_proj(output.reshape(batch, length, module.value_dim))


def _experts_forward(module, hidden_states, top_k_index, top_k_weights):
    if hidden_states.device.type != 'ruda':
        return module._ruda_original_forward(hidden_states, top_k_index, top_k_weights)
    from .blas import grouped_gemm
    if hidden_states.ndim != 2 or top_k_index.ndim != 2 or top_k_weights.shape != top_k_index.shape:
        raise ValueError('Qwen3.5 grouped experts require token rows and matching top-k slots')
    tokens, hidden = hidden_states.shape
    slots = top_k_index.shape[1]
    if top_k_index.shape[0] != tokens or slots == 0:
        raise ValueError('expert routing must cover each token with positive top-k slots')
    dtype = (torch.get_autocast_dtype('ruda') if torch.is_autocast_enabled('ruda') else hidden_states.dtype)
    gate_up, down = module.gate_up_proj, module.down_proj
    if (gate_up.dtype not in (torch.float16, torch.bfloat16, torch.float32) or
            down.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
        raise ValueError('native grouped Qwen3.5 experts require floating checkpoint parameters')
    rows = hidden_states[:, None, :].expand(tokens, slots, hidden).reshape(-1, hidden).to(dtype)
    ids = top_k_index.reshape(-1).to(torch.int64)
    gate, up = grouped_gemm(rows.contiguous(), gate_up.to(dtype).contiguous(), ids.contiguous()).split(module.intermediate_dim, -1)
    value = F.silu(gate) * up
    output = grouped_gemm(value.contiguous(), down.to(dtype).contiguous(), ids.contiguous()).reshape(tokens, slots, hidden)
    output = (output * top_k_weights[:, :, None]).to(hidden_states.dtype)
    order = top_k_index.argsort(dim=-1, stable=True)
    output = output.gather(1, order[:, :, None].expand(tokens, slots, hidden))
    result = torch.zeros_like(hidden_states)
    for slot in range(slots):
        result = result + output[:, slot]
    return result


def enable_transformers_qwen35(model):
    """Keep HF parameters/cache API; replace only this instance's RUDA kernels.

    Applies to qwen3_5 and qwen3_5_moe text/backbone/full models with the
    per-layer recurrent-state Cache API. Select RUDA attention separately with
    register_transformers_attention; non-RUDA inputs retain the original methods.
    """
    config = getattr(model, 'config', None)
    text = getattr(config, 'text_config', config)
    if getattr(text, 'model_type', None) not in ('qwen3_5_text', 'qwen3_5_moe_text'):
        raise ValueError('expected an HF Qwen3.5 dense or MoE text configuration')
    replacements = []
    for name, module in model.named_modules():
        kind = type(module).__name__
        if kind in ('Qwen3_5GatedDeltaNet', 'Qwen3_5MoeGatedDeltaNet'):
            required = ('in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a', 'out_proj', 'conv1d',
                        'dt_bias', 'A_log', 'norm', 'layer_idx', 'num_v_heads', 'num_k_heads',
                        'head_k_dim', 'head_v_dim', 'key_dim', 'value_dim', 'conv_kernel_size',
                        'layer_norm_epsilon', 'activation')
            if any(not hasattr(module, attr) for attr in required):
                raise ValueError(f'{name}: unsupported HF Qwen3.5 DeltaNet schema')
            if (module.activation != 'silu' or module.num_k_heads <= 0 or module.num_v_heads % module.num_k_heads
                    or module.key_dim != module.num_k_heads * module.head_k_dim
                    or module.value_dim != module.num_v_heads * module.head_v_dim):
                raise ValueError(f'{name}: unsupported Qwen3.5 recurrent geometry')
            replacements.append((module, _delta_forward))
        elif kind == 'Qwen3_5MoeExperts':
            if any(not hasattr(module, attr) for attr in ('gate_up_proj', 'down_proj', 'num_experts', 'hidden_dim', 'intermediate_dim')):
                raise ValueError(f'{name}: unsupported HF Qwen3.5 expert schema')
            if (tuple(module.gate_up_proj.shape) != (module.num_experts, 2 * module.intermediate_dim, module.hidden_dim)
                    or tuple(module.down_proj.shape) != (module.num_experts, module.hidden_dim, module.intermediate_dim)):
                raise ValueError(f'{name}: stacked expert parameter layout differs')
            replacements.append((module, _experts_forward))
    if not replacements:
        raise ValueError('model contains no supported HF Qwen3.5 recurrent or expert layers')
    for module, replacement in replacements:
        if not hasattr(module, '_ruda_original_forward'):
            module._ruda_original_forward = module.forward
        module.forward = MethodType(replacement, module)
    return model
