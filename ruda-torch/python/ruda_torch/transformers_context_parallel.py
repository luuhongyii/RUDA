"""Rank-local HF causal-LM training with native ring attention and global token loss."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from ._expert_exchange import _group, register_group
from .context_parallel import ring_scaled_dot_product_attention


@torch.library.custom_op('ruda::context_max_position', mutates_args=())
def _context_max_position(positions: torch.Tensor, group_key: str) -> torch.Tensor:
    maximum = (positions.amax().reshape(1) if positions.numel() else
               torch.full((1,), -1, device=positions.device, dtype=positions.dtype))
    maximum = maximum.clone()
    _group(group_key).max_(maximum)
    return maximum


@_context_max_position.register_fake
def _context_max_position_fake(positions, group_key):
    return positions.new_empty((1,))


def _global_rotary_positions(module, args, kwargs):
    positions = kwargs.get('position_ids', args[1] if len(args) > 1 else None)
    if positions is None or positions.ndim not in (2, 3):
        raise ValueError('context rotary update requires global token positions with an optional multimodal axis')
    sentinel = _context_max_position(positions, module._ruda_context_rotary_key)
    extended = torch.cat((positions, sentinel.expand(*positions.shape[:-1], 1)), dim=-1)
    if 'position_ids' in kwargs:
        kwargs = dict(kwargs, position_ids=extended)
    else:
        args = (args[0], extended, *args[2:])
    return args, kwargs


def _local_rotary_result(module, args, kwargs, output):
    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError('context rotary module must return cosine and sine tensors')
    return output[0][..., :-1, :], output[1][..., :-1, :]


def _bind_context_rotary(model, key):
    for module in model.modules():
        kinds = getattr(module, 'rope_type', None)
        kinds = tuple(kinds.values()) if isinstance(kinds, dict) else (kinds,)
        if not any(isinstance(kind, str) and ('dynamic' in kind or kind == 'longrope') for kind in kinds):
            continue
        module._ruda_context_rotary_key = key
        if not hasattr(module, '_ruda_context_rotary_hooks'):
            # Let HF perform its own frequency update/reset and original formula;
            # the extra position only supplies the complete-context maximum.
            module._ruda_context_rotary_hooks = (
                module.register_forward_pre_hook(_global_rotary_positions, with_kwargs=True),
                module.register_forward_hook(_local_rotary_result, with_kwargs=True),
            )


def register_transformers_ring_attention(name='ruda_ring'):
    from transformers.modeling_utils import AttentionInterface, ALL_ATTENTION_FUNCTIONS
    from transformers.masking_utils import AttentionMaskInterface, ALL_MASK_ATTENTION_FUNCTIONS
    if not isinstance(name, str) or not name.startswith('ruda_'):
        raise ValueError('ring attention registration requires a distinct ruda_ name')
    if name in ALL_ATTENTION_FUNCTIONS or name in ALL_MASK_ATTENTION_FUNCTIONS:
        raise ValueError('attention registration name already exists')

    def mask(batch_size, q_length=None, kv_length=None, attention_mask=None, **kwargs):
        if kwargs.get('use_vmap', False):
            raise ValueError('ring attention requires explicit document/position metadata, not a custom mask callback')
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError('context-parallel padding masks must be rank-local two-dimensional tensors')
        return attention_mask

    def attention(module, query, key, value, attention_mask, dropout=0., scaling=None, **kwargs):
        if not hasattr(module, '_ruda_context_group'):
            raise ValueError('bind the context process group before selecting ring attention')
        if kwargs.get('output_attentions', False) or kwargs.get('head_mask') is not None:
            raise ValueError('ring attention does not materialize quadratic attention weights or head masks')
        positions = kwargs.get('ruda_context_positions')
        documents = kwargs.get('ruda_context_documents')
        valid = kwargs.get('ruda_context_valid')
        if positions is None or documents is None:
            raise ValueError('context-parallel HF attention requires explicit global positions and document IDs')
        if query.shape[-2] != key.shape[-2]:
            raise ValueError('HF context training expects matching local query/key intervals; use the ring API for explicit cache positions')
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape != positions.shape:
                raise ValueError('HF ring padding mask must describe this rank, not a global or dense mask')
            padding = attention_mask.to(device=query.device, dtype=torch.bool)
            valid = padding if valid is None else valid & padding
        window = kwargs.get('sliding_window', getattr(module, 'sliding_window', None))
        window = None if window is None else (window - 1, 0)
        causal = kwargs.get('is_causal', getattr(module, 'is_causal', True))
        from ._parallel_rng import module_partition
        result = ring_scaled_dot_product_attention(query, key, value, module._ruda_context_group,
            group_key=module._ruda_context_key, is_causal=causal, scale=scaling,
            dropout_p=dropout, enable_gqa=query.shape[1] != key.shape[1],
            window_size=window, softcap=kwargs.get('softcap'),
            query_positions=positions, key_positions=positions,
            query_documents=documents, key_documents=documents,
            query_mask=valid, key_mask=valid, **module_partition(module, query.shape[1]))
        return result.transpose(1, 2).contiguous(), None

    AttentionMaskInterface.register(name, mask)
    AttentionInterface.register(name, attention)
    return name


def enable_transformers_context_parallel(model, group, *, name='ruda_ring', group_key='context-ring'):
    """Keep the original projections/RoPE/router and replace the attention backend."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    config = getattr(model, 'config', None)
    text_config = getattr(config, 'text_config', config)
    kind = getattr(text_config, 'model_type', None)
    if kind == 'kimi_linear':
        from .hybrid_context_parallel import enable_kimi_context_parallel
        return enable_kimi_context_parallel(model, group, group_key=group_key)
    if kind not in ('llama', 'qwen2', 'qwen3', 'qwen3_moe', 'deepseek_v3', 'qwen3_5_text', 'qwen3_5_moe_text'):
        raise ValueError('this adapter expects a full-attention HF Llama/Qwen/DeepSeek decoder')
    if name not in ALL_ATTENTION_FUNCTIONS:
        register_transformers_ring_attention(name)
    key = register_group(group_key, group)
    if kind in ('qwen3_5_text', 'qwen3_5_moe_text'):
        from .hybrid_context_parallel import bind_qwen35_context_recurrence
        bind_qwen35_context_recurrence(model, group, key)
    selected = []
    for module in model.modules():
        if (hasattr(module, 'layer_idx') and hasattr(module, 'o_proj') and
                (hasattr(module, 'q_proj') or hasattr(module, 'q_a_proj')) and hasattr(module, 'config')):
            selected.append(module)
    if not selected and kind not in ('qwen3_5_text', 'qwen3_5_moe_text'):
        raise ValueError('no compatible HF attention modules found')
    for module in selected:
        module._ruda_context_group, module._ruda_context_key = group, key
        module.config._attn_implementation = name
    config._attn_implementation = name
    text_config._attn_implementation = name
    _bind_context_rotary(model, key)
    return model


def shard_context_batch(input_ids, group, *, labels=None, attention_mask=None,
                        position_ids=None, document_ids=None, ignore_index=-100):
    """Split a logical batch, shifting labels BEFORE the context boundaries.

    This operates on the caller's existing CPU/device token batch. It performs
    no activation, KV or weight gathering. The final token of a document never
    predicts the first token of the next document or padding.
    """
    if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError('context inputs require integer [batch,tokens] IDs')
    batch, length = input_ids.shape
    if batch < 1:
        raise ValueError('context batch must contain at least one sequence')
    labels = input_ids if labels is None else labels
    position_ids = (torch.arange(length, device=input_ids.device, dtype=torch.int64).expand(batch, length)
                    if position_ids is None else position_ids)
    document_ids = (torch.arange(batch, device=input_ids.device, dtype=torch.int64)[:, None].expand(batch, length)
                    if document_ids is None else document_ids)
    valid = (torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None
             else attention_mask.to(dtype=torch.bool))
    for tensor in (labels, position_ids, document_ids, valid):
        if tensor.shape != input_ids.shape or tensor.device != input_ids.device:
            raise ValueError('context token fields must share the complete batch shape and device')
    if (labels.dtype not in (torch.int32, torch.int64) or position_ids.dtype != torch.int64
            or document_ids.dtype != torch.int64):
        raise ValueError('context labels/positions/documents must use integer storage')
    targets = torch.full_like(labels, ignore_index)
    if length > 1:
        same = (document_ids[:, :-1] == document_ids[:, 1:]) & (document_ids[:, :-1] >= 0)
        keep = same & valid[:, :-1] & valid[:, 1:]
        targets[:, :-1] = torch.where(keep, labels[:, 1:], torch.full_like(labels[:, 1:], ignore_index))
    start, end = length * group.rank // group.world_size, length * (group.rank + 1) // group.world_size
    return dict(input_ids=input_ids[:, start:end].contiguous(),
        attention_mask=valid[:, start:end].contiguous(), position_ids=position_ids[:, start:end].contiguous(),
        ruda_context_documents=document_ids[:, start:end].contiguous(),
        context_targets=targets[:, start:end].contiguous())


@torch.library.custom_op('ruda::context_loss_reduce', mutates_args=())
def _reduce_loss(local_sum: torch.Tensor, local_count: torch.Tensor,
                 group_key: str) -> tuple[torch.Tensor, torch.Tensor]:
    group = _group(group_key)
    loss, count = local_sum.detach().clone(), local_count.detach().clone()
    group.sum_(loss)
    group.sum_(count)
    denominator = torch.where(count > 0, count, torch.ones_like(count)).to(loss.dtype)
    return loss / denominator, count


@_reduce_loss.register_fake
def _reduce_loss_fake(local_sum, local_count, group_key):
    return torch.empty_like(local_sum), torch.empty_like(local_count)


def _loss_context(ctx, inputs, output):
    _, count = output
    ctx.save_for_backward(count)
    ctx.mark_non_differentiable(count)


def _loss_gradient(ctx, gradient, _count):
    count, = ctx.saved_tensors
    return gradient / torch.where(count > 0, count, torch.ones_like(count)).to(gradient.dtype), None, None


_reduce_loss.register_autograd(_loss_gradient, setup_context=_loss_context)


@torch.library.custom_op('ruda::context_sum_objective', mutates_args=())
def _sum_objective(value: torch.Tensor, group_key: str) -> torch.Tensor:
    result = value.detach().clone()
    _group(group_key).sum_(result)
    return result


@_sum_objective.register_fake
def _sum_objective_fake(value, group_key):
    return torch.empty_like(value)


def _objective_gradient(ctx, gradient):
    return gradient, None


_sum_objective.register_autograd(_objective_gradient)


def context_parallel_router_loss(gate_logits, group, *, num_experts, top_k,
                                 attention_mask=None, group_key='context-ring'):
    """HF top-k load-balancing objective over the complete context and all layers.

    Every rank differentiates the same replicated objective exactly once; the
    collective's adjoint therefore retains its local probability contribution.
    Counts include every selected slot, with no additional division by top_k.
    """
    if not isinstance(gate_logits, (tuple, list)) or not gate_logits:
        raise ValueError('router loss requires per-layer logits')
    if type(num_experts) is not int or type(top_k) is not int or not 1 <= top_k <= num_experts:
        raise ValueError('invalid router loss expert geometry')
    device = gate_logits[0].device
    probability_sum = torch.zeros(num_experts, dtype=torch.float32, device=device)
    counts = torch.zeros_like(probability_sum)
    rows = torch.zeros((), dtype=torch.float32, device=device)
    for logits in gate_logits:
        if logits.ndim != 2 or logits.shape[1] != num_experts or logits.device != device:
            raise ValueError('per-layer router logits must have shape [local_tokens,experts]')
        valid = (torch.ones(logits.shape[0], device=device, dtype=torch.float32)
                 if attention_mask is None else attention_mask.to(device=device, dtype=torch.float32).reshape(-1))
        if valid.shape[0] != logits.shape[0]:
            raise ValueError('router padding mask and local token count differ')
        probabilities = logits.softmax(-1)
        selected = probabilities.topk(top_k, dim=-1).indices
        counts = counts.scatter_add(0, selected.reshape(-1), valid[:, None].expand(-1, top_k).reshape(-1))
        probability_sum = probability_sum + (probabilities.float() * valid[:, None]).sum(0)
        rows = rows + valid.sum()
    moments = _sum_objective(torch.cat((counts, probability_sum, rows.reshape(1))), register_group(group_key, group))
    denominator = moments[-1].clamp_min(1)
    frequency, probability = moments[:num_experts] / denominator, moments[num_experts:-1] / denominator
    return num_experts * (frequency * probability).sum()


def context_parallel_cross_entropy(logits, targets, group, *, ignore_index=-100,
                                   label_smoothing=0., group_key='context-ring', vocab_group=None):
    if targets.shape != logits.shape[:-1] or targets.device != logits.device:
        raise ValueError('context loss targets must match local logit token positions')
    if targets.dtype not in (torch.int32, torch.int64):
        raise ValueError('context loss requires integer targets')
    if vocab_group is None:
        loss = (F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1).long(),
                    ignore_index=ignore_index, label_smoothing=label_smoothing, reduction='sum')
                if targets.numel() else logits.float().sum() * 0)
    else:
        from .parallel_loss import vocab_parallel_cross_entropy
        loss = vocab_parallel_cross_entropy(logits, targets, vocab_group,
            ignore_index=ignore_index, label_smoothing=label_smoothing, reduction='sum')
    count = (targets != ignore_index).sum(dtype=torch.int64).reshape(())
    return _reduce_loss(loss.reshape(()), count, register_group(group_key, group))[0]


class ContextParallelCausalLM(nn.Module):
    """Rank-local training/prefill wrapper; parameter gradients use a CP replica group."""
    def __init__(self, model, group, *, group_key='context-ring', attention_name='ruda_ring', vocab_group=None):
        super().__init__()
        self.model = enable_transformers_context_parallel(model, group, name=attention_name, group_key=group_key)
        self.config, self.group, self.group_key = model.config, group, group_key
        self.vocab_group = vocab_group

    def forward(self, input_ids=None, *, inputs_embeds=None, attention_mask=None, position_ids=None,
                ruda_context_documents=None, context_targets=None, return_dict=True,
                ignore_index=-100, label_smoothing=0., **kwargs):
        if position_ids is None or ruda_context_documents is None:
            raise ValueError('pass global positions/documents, e.g. from shard_context_batch')
        if kwargs.pop('use_cache', False) or kwargs.get('past_key_values') is not None:
            raise ValueError('context training wrapper has no replicated HF decode cache')
        if 'labels' in kwargs or kwargs.get('logits_to_keep', 0) != 0:
            raise ValueError('supply already-shifted context_targets and retain all local token logits')
        output = self.model(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            position_ids=position_ids, use_cache=False, return_dict=True,
            ruda_context_positions=position_ids, ruda_context_documents=ruda_context_documents,
            ruda_context_valid=None if attention_mask is None else attention_mask.to(torch.bool), **kwargs)
        fields = dict(output)
        text_config = getattr(self.config, 'text_config', self.config)
        router_logits = getattr(output, 'router_logits', None)
        if router_logits is not None:
            fields['aux_loss'] = context_parallel_router_loss(router_logits, self.group,
                num_experts=text_config.num_experts, top_k=text_config.num_experts_per_tok,
                attention_mask=attention_mask, group_key=self.group_key)
        if context_targets is not None:
            loss = context_parallel_cross_entropy(output.logits, context_targets, self.group,
                ignore_index=ignore_index, label_smoothing=label_smoothing,
                group_key=self.group_key, vocab_group=self.vocab_group)
            if fields.get('aux_loss') is not None:
                loss = loss + text_config.router_aux_loss_coef * fields['aux_loss']
            fields['loss'] = loss
        output = type(output)(**fields)
        return output if return_dict else output.to_tuple()
