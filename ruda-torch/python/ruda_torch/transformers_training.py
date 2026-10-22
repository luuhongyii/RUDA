"""Shared floating/adapter training entry for existing causal-LM architectures."""
from __future__ import annotations

import math
from types import MethodType
import torch
from torch import nn

from ._architecture_ops import precision_context
from .causal_finetuning import CausalLMFinetuner


def hf_router_load_balancing_loss(router_logits, *, num_experts, top_k, attention_mask=None):
    """HF softmax top-k objective across all layers, without a token/expert one-hot.

    Counts include every selected slot, matching HF's sum over top-k slots (not
    RUDA RoutingPolicy's separately exposed, k-normalized router objective).
    """
    if (type(num_experts) is not int or type(top_k) is not int
            or not 1 <= top_k <= num_experts):
        raise ValueError('invalid router auxiliary-loss geometry')
    if not isinstance(router_logits, (tuple, list)) or not router_logits:
        raise ValueError('router auxiliary loss requires per-layer router logits')
    device = router_logits[0].device
    probability_sum = torch.zeros(num_experts, dtype=torch.float32, device=device)
    counts = torch.zeros_like(probability_sum)
    rows = torch.zeros((), dtype=torch.float32, device=device)
    with precision_context(router_logits[0]):
        for logits in router_logits:
            if logits.ndim != 2 or logits.shape[-1] != num_experts or logits.device != device:
                raise ValueError('router logits must have shape [tokens,experts] on one device')
            valid = (torch.ones(logits.shape[0], dtype=torch.float32, device=device) if attention_mask is None
                     else attention_mask.to(device=device, dtype=torch.float32).reshape(-1))
            if valid.shape[0] != logits.shape[0]:
                raise ValueError('router padding mask must describe the actual router token rows')
            probabilities = logits.float().softmax(-1)
            selected = probabilities.topk(top_k, dim=-1).indices
            counts = counts.scatter_add(0, selected.reshape(-1), valid[:, None].expand(-1, top_k).reshape(-1))
            probability_sum = probability_sum + (probabilities * valid[:, None]).sum(0)
            rows = rows + valid.sum()
        denominator = rows.clamp_min(1)
        return num_experts * ((counts / denominator) * (probability_sum / denominator)).sum()


class HfRouterAuxiliaryLoss(nn.Module):
    """Configured HF router coefficient, retained by the hidden/head SFT split."""
    def __init__(self, *, num_experts, top_k, coefficient, context_group=None,
                 group_key='context-ring'):
        super().__init__()
        if (type(num_experts) is not int or type(top_k) is not int
                or not 1 <= top_k <= num_experts or not math.isfinite(coefficient) or coefficient < 0):
            raise ValueError('invalid HF router auxiliary-loss configuration')
        self.num_experts, self.top_k, self.coefficient = num_experts, top_k, float(coefficient)
        self.context_group, self.group_key = context_group, group_key

    def loss_options(self):
        return dict(kind='hf-softmax-topk', num_experts=self.num_experts,
                    top_k=self.top_k, coefficient=self.coefficient,
                    context_parallel=self.context_group is not None)

    def forward(self, output, attention_mask):
        logits = getattr(output, 'router_logits', None)
        if not isinstance(logits, (tuple, list)) or not logits:
            raise ValueError('the backbone must return router_logits; enable output_router_logits')
        options = dict(num_experts=self.num_experts, top_k=self.top_k, attention_mask=attention_mask)
        if self.context_group is None:
            loss = hf_router_load_balancing_loss(logits, **options)
        else:
            from .transformers_context_parallel import context_parallel_router_loss
            loss = context_parallel_router_loss(logits, self.context_group, group_key=self.group_key, **options)
        return loss * self.coefficient

    def token_sum(self, auxiliary, local_count):
        if self.context_group is None:
            return auxiliary * local_count
        from .transformers_context_parallel import _sum_objective
        from ._expert_exchange import register_group
        total_count = _sum_objective(local_count.detach(), register_group(self.group_key, self.context_group))
        # Log each shard's token-weighted value, but retain the complete-context
        # coefficient on its local probability derivative before CP gradient sums.
        return auxiliary * local_count + (auxiliary - auxiliary.detach()) * (total_count - local_count)


def _stacked_experts_forward(module, hidden_states, top_k_index, top_k_weights):
    if hidden_states.device.type != 'ruda':
        return module._ruda_original_forward(hidden_states, top_k_index, top_k_weights)
    from .blas import grouped_gemm
    if (hidden_states.ndim != 2 or top_k_index.ndim != 2
            or top_k_weights.shape != top_k_index.shape or top_k_index.shape[0] != hidden_states.shape[0]
            or top_k_index.shape[1] < 1):
        raise ValueError('stacked experts require token rows and matching top-k routing slots')
    gate_up, down = module.gate_up_proj, module.down_proj
    hidden, intermediate = gate_up.shape[2], down.shape[2]
    if hidden_states.shape[1] != hidden:
        raise ValueError('expert hidden width differs from the checkpoint')
    dtype = torch.get_autocast_dtype('ruda') if torch.is_autocast_enabled('ruda') else hidden_states.dtype
    if gate_up.dtype not in (torch.float16, torch.bfloat16, torch.float32) or down.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError('grouped experts require floating parameters')
    tokens, slots = top_k_index.shape
    rows = hidden_states[:, None].expand(tokens, slots, hidden).reshape(-1, hidden).to(dtype)
    ids = top_k_index.reshape(-1).to(torch.int64).contiguous()
    gate, up = grouped_gemm(rows.contiguous(), gate_up.to(dtype).contiguous(), ids).split(intermediate, -1)
    values = module.act_fn(gate) * up
    output = grouped_gemm(values.contiguous(), down.to(dtype).contiguous(), ids).reshape(tokens, slots, hidden)
    output = (output * top_k_weights[:, :, None]).to(hidden_states.dtype)
    order = top_k_index.argsort(dim=-1, stable=True)
    output = output.gather(1, order[:, :, None].expand(tokens, slots, hidden))
    result = torch.zeros_like(hidden_states)
    for slot in range(slots):
        result = result + output[:, slot]
    return result


def enable_transformers_moe(model):
    """Use differentiable RUDA grouped GEMM in HF Qwen3/DeepSeek stacked experts.

    Router selection, correction biases, shared experts and auxiliary objectives
    stay with the original architecture; non-RUDA execution is unchanged.
    """
    replacements = []
    for path, module in model.named_modules():
        if not any(cls.__name__ in ('Qwen3MoeExperts', 'DeepseekV3Experts') for cls in type(module).__mro__):
            continue
        gate_up, down = getattr(module, 'gate_up_proj', None), getattr(module, 'down_proj', None)
        if (not isinstance(gate_up, torch.Tensor) or not isinstance(down, torch.Tensor)
                or gate_up.ndim != 3 or down.ndim != 3 or gate_up.shape[0] != down.shape[0]
                or gate_up.shape[1] != 2 * down.shape[2] or gate_up.shape[2] != down.shape[1]
                or not callable(getattr(module, 'act_fn', None))):
            raise ValueError(f'{path}: unsupported stacked expert parameter schema')
        replacements.append(module)
    for module in replacements:
        if not hasattr(module, '_ruda_original_forward'):
            module._ruda_original_forward = module.forward
        module.forward = MethodType(_stacked_experts_forward, module)
    return model


def prepare_causal_lm(model, *, backbone=None, head=None, ruda_kernels=True,
                      auxiliary_loss='auto', **finetuner_options):
    """Connect an existing causal LM to SFTTrainer/CausalLMTrainer, without copies.

    Uses the model's decoder/output-embedding API. Explicit modules or exact
    paths override those APIs, including text-only views of multimodal models.
    No weights, router policies, LoRA targets or loss coefficients are guessed.
    """
    if not isinstance(model, nn.Module):
        raise TypeError('model must be an existing causal language model')
    if isinstance(backbone, str):
        backbone = model.get_submodule(backbone)
    if isinstance(head, str):
        head = model.get_submodule(head)
    if backbone is None:
        decoder = getattr(model, 'get_decoder', None)
        backbone = decoder() if callable(decoder) else getattr(model, 'base_model', None)
    if head is None:
        output = getattr(model, 'get_output_embeddings', None)
        head = output() if callable(output) else None
    if (not isinstance(backbone, nn.Module) or not isinstance(head, nn.Module)
            or backbone is model or any(module is head for module in backbone.modules())):
        raise ValueError('supply the actual hidden-state backbone and vocabulary projection as modules or exact paths')
    config = getattr(backbone, 'config', getattr(model, 'config', None))
    config = getattr(config, 'text_config', config)
    kind = getattr(config, 'model_type', None)
    options = dict(finetuner_options)
    if auxiliary_loss == 'auto':
        coefficient = getattr(config, 'router_aux_loss_coef', 0.)
        router_outputs = (options.get('backbone_kwargs') or {}).get('output_router_logits',
            getattr(config, 'output_router_logits', False))
        auxiliary_loss = None
        if coefficient and router_outputs:
            if kind not in ('qwen2_moe', 'qwen3_moe', 'qwen3_5_moe_text', 'mixtral'):
                raise ValueError('supply this architecture\'s auxiliary_loss explicitly, or None to disable it')
            auxiliary_loss = HfRouterAuxiliaryLoss(num_experts=getattr(config, 'num_experts',
                getattr(config, 'num_local_experts', None)), top_k=config.num_experts_per_tok,
                coefficient=coefficient)
    if isinstance(auxiliary_loss, HfRouterAuxiliaryLoss):
        options['backbone_kwargs'] = dict(options.get('backbone_kwargs') or {}, output_router_logits=True)
    if ruda_kernels:
        if kind in ('qwen3_5_text', 'qwen3_5_moe_text'):
            from .transformers_qwen35 import enable_transformers_qwen35
            enable_transformers_qwen35(backbone)
        enable_transformers_moe(backbone)
    finetuner = CausalLMFinetuner(backbone, head, auxiliary_loss=auxiliary_loss, **options)
    finetuner.train(model.training)
    return finetuner
