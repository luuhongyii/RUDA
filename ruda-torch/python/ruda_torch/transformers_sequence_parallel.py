"""Sequence-sharded decoder residuals with explicit TP region boundaries."""
from __future__ import annotations

from types import MethodType
import copy
import torch
from torch import nn
from .parallel_training import (ColumnParallelLinear, RowParallelLinear, tensor_parallelize,
    gather_from_tensor_parallel, scatter_to_tensor_parallel)
from .sequence_parallel import (SequenceParallelColumnLinear, SequenceParallelRowLinear,
                                gather_sequence_parallel, sequence_projection)


def _input(args, kwargs):
    if 'hidden_states' in kwargs:
        return kwargs['hidden_states'], 'hidden_states'
    if 'hidden' in kwargs:
        return kwargs['hidden'], 'hidden'
    if not args:
        raise ValueError('sequence region requires its hidden-state input')
    return args[0], None


def _replace_input(args, kwargs, value, name):
    return (args, dict(kwargs, **{name: value})) if name is not None else ((value, *args[1:]), kwargs)


def _region_forward(self, *args, **kwargs):
    value, name = _input(args, kwargs)
    if self._ruda_sequence_native_region:
        full = gather_sequence_parallel(value, self._ruda_sequence_group, axis=1,
                                         group_key=self._ruda_sequence_key)
    else:
        # General TP regions already perform their own input-gradient SUMs.
        # Only reconstruct disjoint token slices here, not another SUM.
        full = gather_from_tensor_parallel(value, self._ruda_sequence_group, axis=1)
    args, kwargs = _replace_input(args, kwargs, full, name)
    mask = kwargs.get('attention_mask')
    heads = getattr(self, '_ruda_global_num_heads', None)
    if mask is not None and heads is not None and mask.ndim >= 3 and mask.shape[-3] == heads:
        kwargs = dict(kwargs, attention_mask=mask.narrow(-3, self._ruda_head_start, self._ruda_local_num_heads))
    result = self._ruda_sequence_original_forward(*args, **kwargs)
    if self._ruda_sequence_native_region:
        return result
    if isinstance(result, torch.Tensor):
        return scatter_to_tensor_parallel(result, self._ruda_sequence_group, axis=1)
    if isinstance(result, tuple) and result and isinstance(result[0], torch.Tensor):
        return (scatter_to_tensor_parallel(result[0], self._ruda_sequence_group, axis=1), *result[1:])
    raise TypeError('sequence region output must be hidden states or a tuple beginning with hidden states')


def _bind_region(module, group, key, native):
    if hasattr(module, '_ruda_sequence_original_forward'):
        raise ValueError('decoder region is already sequence-parallel')
    module._ruda_sequence_group, module._ruda_sequence_key = group, key
    module._ruda_sequence_native_region = native
    module._ruda_sequence_original_forward = module.forward
    module.forward = MethodType(_region_forward, module)


def _scatter_entry(module, args, kwargs):
    value, name = _input(args, kwargs)
    value = scatter_to_tensor_parallel(value, module._ruda_sequence_entry_group, axis=1)
    return _replace_input(args, kwargs, value, name)


def _gather_exit(module, args, output):
    return gather_from_tensor_parallel(output, module._ruda_sequence_exit_group, axis=1)


def _recorded_hidden(module, args, output):
    hidden = getattr(output, 'last_hidden_state', output[0] if isinstance(output, tuple) and output else None)
    if not isinstance(hidden, torch.Tensor):
        return output
    group = module._ruda_sequence_group

    def gather(value):
        if (isinstance(value, torch.Tensor) and value.ndim == 3 and value.shape[-1] == hidden.shape[-1]
                and value.shape[1] != hidden.shape[1] and value.shape[1] * group.world_size == hidden.shape[1]):
            return gather_from_tensor_parallel(value, group, axis=1)
        if isinstance(value, tuple):
            return tuple(gather(item) for item in value)
        return value

    if hasattr(output, 'hidden_states') and output.hidden_states is not None:
        return type(output)(**dict(output, hidden_states=gather(output.hidden_states)))
    return gather(output) if isinstance(output, tuple) else output


def _backbone_input(module, args, kwargs):
    use_cache = kwargs.get('use_cache')
    if use_cache is None:
        use_cache = getattr(module.config, 'use_cache', False)
    if kwargs.get('past_key_values') is not None or use_cache:
        raise ValueError('sequence-sharded decoder training/prefill requires use_cache=False')
    value = kwargs.get('input_ids', args[0] if args else None)
    if value is None:
        value = kwargs.get('inputs_embeds')
        length = None if value is None else value.shape[-2]
    else:
        length = value.shape[-1]
    if length is None or length % module._ruda_sequence_group.world_size:
        raise ValueError('local context token length must divide the sequence-parallel group')
    return args, kwargs


def _mark_sequence_norm(module):
    if module is not None:
        for parameter in module.parameters():
            parameter._ruda_sp_replicated = True


def sequence_parallelize_transformers(model, group, *, group_key='model-sequence'):
    """Bind full model endpoints without changing pretrained parameter paths.

    Embedding, masks and RoPE are constructed for the whole local CP interval;
    residuals/norms are sequence-sharded between decoder regions. Dense QKV and
    gated-MLP regions use one input all-gather and output reduce-scatter. More
    complex pre-partitioned TP regions retain their exact internal collectives.
    Final normalized hidden states are gathered before the original LM head,
    including its ordinary vocabulary-parallel/chunked-loss contract.
    """
    from ._expert_exchange import register_group
    key = register_group(group_key, group)
    candidates = [(path, module) for path, module in model.named_modules()
                  if all(hasattr(module, name) for name in ('embed_tokens', 'layers', 'norm'))
                  and isinstance(module.layers, nn.ModuleList)]
    if len(candidates) != 1:
        raise ValueError('select exactly one text decoder backbone for sequence parallelism')
    path, backbone = candidates[0]
    kind = getattr(backbone.config, 'model_type', None)
    if kind not in ('llama', 'qwen2', 'qwen3', 'qwen3_moe', 'deepseek_v3', 'kimi_linear',
                    'qwen3_5_text', 'qwen3_5_moe_text'):
        raise ValueError('unsupported sequence-parallel decoder family')
    if not backbone.layers or hasattr(backbone, '_ruda_sequence_hooks'):
        raise ValueError('sequence decoder must have layers and must not already be bound')
    if kind == 'kimi_linear':
        from .transformers_kimi import tensor_parallelize_kimi
        if not any(isinstance(layer, (ColumnParallelLinear, RowParallelLinear)) for layer in backbone.modules()):
            tensor_parallelize_kimi(backbone, group, vocab_parallel=False)
        backbone._ruda_sequence_group, backbone._ruda_sequence_key = group, key
        backbone._ruda_sequence_hooks = ()
        for layer in backbone.layers:
            _mark_sequence_norm(layer.input_layernorm)
            _mark_sequence_norm(layer.post_attention_layernorm)
        _mark_sequence_norm(backbone.norm)
        return model

    from .parallel_adapters import _ParallelLoRA, _ParallelNF4
    from .packed_parallel import _ParallelPacked

    def partitioned(projection, axis):
        result = (isinstance(projection, ColumnParallelLinear if axis == 0 else RowParallelLinear)
                  or isinstance(projection, (_ParallelLoRA, _ParallelNF4, _ParallelPacked)) and projection.axis == axis)
        if result and projection.group.group is not group.group:
            raise ValueError('existing projection must use the actual sequence/tensor group')
        return result

    prefix = path + '.' if path else ''
    plan, regions, head_geometry = {}, [], []
    for index, layer in enumerate(backbone.layers):
        base = f'{prefix}layers.{index}.'
        attention_name = 'self_attn' if hasattr(layer, 'self_attn') else 'linear_attn'
        attention = getattr(layer, attention_name)
        native_attention = all(hasattr(attention, name) for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'))
        # Qwen3.5's Q projection interleaves a separate gate and requires a
        # family-specific partition plan; do not relabel it as ordinary QKV.
        native_attention = native_attention and kind not in ('qwen3_5_text', 'qwen3_5_moe_text')
        if native_attention:
            config = attention.config
            kv = getattr(config, 'num_key_value_heads', config.num_attention_heads)
            if config.num_attention_heads % group.world_size or (kv % group.world_size and group.world_size % kv):
                raise ValueError('Q heads must be divisible by TP; KV heads must divide TP or be divisible by TP')
            replicas = max(1, group.world_size // kv)
            head_geometry.append((attention, config.num_attention_heads, kv))
            for name in ('q_proj', 'k_proj', 'v_proj'):
                if not partitioned(getattr(attention, name), 0):
                    plan[base + attention_name + '.' + name] = {'kind': 'column', 'gather_output': False,
                        'replicas': replicas if name != 'q_proj' else 1}
                elif getattr(getattr(attention,name),'replicas',1) != (replicas if name != 'q_proj' else 1):
                    raise ValueError('existing QKV shard geometry differs from the requested KV replicas')
            if not partitioned(attention.o_proj, 1):
                plan[base + attention_name + '.o_proj'] = {'kind': 'row', 'input_is_parallel': True}
        regions.append((attention, native_attention, ('q_proj', 'k_proj', 'v_proj'), 'o_proj'))
        mlp_name = 'mlp' if hasattr(layer, 'mlp') else 'block_sparse_moe'
        mlp = getattr(layer, mlp_name)
        native_mlp = all(hasattr(mlp, name) for name in ('gate_proj', 'up_proj', 'down_proj'))
        if native_mlp:
            for name in ('gate_proj', 'up_proj'):
                if not partitioned(getattr(mlp, name), 0):
                    plan[base + mlp_name + '.' + name] = {'kind': 'column', 'gather_output': False}
            if not partitioned(mlp.down_proj, 1):
                plan[base + mlp_name + '.down_proj'] = {'kind': 'row', 'input_is_parallel': True}
        regions.append((mlp, native_mlp, ('gate_proj', 'up_proj'), 'down_proj'))
    if plan:
        tensor_parallelize(model, group, plan)
    for attention, heads, kv in head_geometry:
        local_heads, local_kv = heads // group.world_size, max(1, kv // group.world_size)
        attention._ruda_global_num_heads = heads
        attention._ruda_head_start = group.rank * local_heads
        attention._ruda_local_num_heads = local_heads
        attention.config = copy.copy(attention.config)
        attention.config.num_attention_heads = local_heads
        attention.config.num_key_value_heads = local_kv
        for name, value in (('num_heads', local_heads), ('num_attention_heads', local_heads),
                            ('num_key_value_heads', local_kv), ('num_key_value_groups', local_heads // local_kv)):
            if hasattr(attention, name):setattr(attention, name, value)
    for region, native, columns, row in regions:
        if native:
            for name in columns:
                setattr(region, name, sequence_projection(getattr(region, name), group, axis=0, group_key=key))
            setattr(region, row, sequence_projection(getattr(region, row), group, axis=1, group_key=key))
            _mark_sequence_norm(getattr(region, 'q_norm', None))
            _mark_sequence_norm(getattr(region, 'k_norm', None))
        _bind_region(region, group, key, native)
    for layer in backbone.layers:
        _mark_sequence_norm(layer.input_layernorm)
        _mark_sequence_norm(layer.post_attention_layernorm)
    _mark_sequence_norm(backbone.norm)
    backbone._ruda_sequence_group, backbone._ruda_sequence_key = group, key
    backbone.layers[0]._ruda_sequence_entry_group = group
    backbone.norm._ruda_sequence_exit_group = group
    backbone._ruda_sequence_hooks = (
        backbone.register_forward_pre_hook(_backbone_input, with_kwargs=True),
        backbone.layers[0].register_forward_pre_hook(_scatter_entry, with_kwargs=True),
        backbone.norm.register_forward_hook(_gather_exit),
        backbone.register_forward_hook(_recorded_hidden),
    )
    return model
