"""Configurable top-k routing, local experts, shared experts and explicit losses."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from ._architecture_ops import positive_int, positive_float, work_dtype, precision_context, stable_topk, positions


@dataclass(frozen=True)
class RoutingPolicy:
    top_k: int
    scoring: str = 'softmax'
    renormalize: bool = False
    scale: float = 1.
    num_groups: int = 1
    topk_groups: int = 1
    group_score_topk: int = 1

    def __post_init__(self):
        for name in ('top_k', 'num_groups', 'topk_groups', 'group_score_topk'):
            positive_int(getattr(self, name), name)
        if self.scoring not in ('softmax', 'sigmoid') or type(self.renormalize) is not bool:
            raise ValueError('invalid router scoring or normalization policy')
        positive_float(self.scale, 'router scale')
        if self.topk_groups > self.num_groups:
            raise ValueError('topk_groups exceeds num_groups')


@dataclass(frozen=True)
class RoutingResult:
    indices: torch.Tensor
    weights: torch.Tensor
    scores: torch.Tensor


def route_experts(logits, policy, *, correction_bias=None):
    """Route [...,experts] with deterministic ties; correction bias affects selection only."""
    if not isinstance(policy, RoutingPolicy) or logits.ndim < 2:
        raise ValueError('routing requires [...,experts] logits and a RoutingPolicy')
    dtype = work_dtype(logits)
    experts = logits.shape[-1]
    if (experts < policy.top_k or experts % policy.num_groups or
            experts // policy.num_groups < policy.group_score_topk or
            policy.topk_groups * (experts // policy.num_groups) < policy.top_k):
        raise ValueError('expert count is incompatible with the routing/group policy')
    if correction_bias is not None and (correction_bias.shape != (experts,) or
            correction_bias.device != logits.device or not correction_bias.is_floating_point()):
        raise ValueError('correction bias must be a same-device floating [experts] tensor')
    with precision_context(logits):
        values = logits.to(dtype)
        scores = values.softmax(-1) if policy.scoring == 'softmax' else values.sigmoid()
        selection = scores.detach()
        if correction_bias is not None:
            selection = selection + correction_bias.detach().to(dtype)
        allowed = None
        if policy.num_groups > 1:
            grouped = selection.reshape(*selection.shape[:-1], policy.num_groups, experts // policy.num_groups)
            group_scores = stable_topk(grouped, policy.group_score_topk)[0].sum(-1)
            group_ids = stable_topk(group_scores, policy.topk_groups)[1]
            all_groups = positions(policy.num_groups, logits.device)
            chosen = (group_ids.unsqueeze(-1) == all_groups).any(-2)
            allowed = chosen.unsqueeze(-1).expand(*grouped.shape).reshape(selection.shape)
        indices = stable_topk(selection, policy.top_k, valid=allowed)[1]
        if logits.device.type == 'ruda' and logits.dtype in (torch.float16, torch.bfloat16, torch.float32) and policy.top_k <= 64:
            from ._router import selected_router_weights
            weights = selected_router_weights(logits.reshape(-1, experts).contiguous(),
                indices.reshape(-1, policy.top_k).contiguous(), scoring=policy.scoring,
                renormalize=policy.renormalize, scale=policy.scale).reshape(indices.shape)
        else:
            weights = scores.gather(-1, indices)
            if policy.renormalize:
                weights = weights / weights.sum(-1, keepdim=True)
            weights = weights * policy.scale
    return RoutingResult(indices, weights, scores)


def _loss_mask(scores, token_mask):
    if token_mask is None:
        return torch.ones(scores.shape[:-1], device=scores.device, dtype=scores.dtype)
    if token_mask.shape != scores.shape[:-1] or token_mask.device != scores.device or token_mask.dtype != torch.bool:
        raise ValueError('token_mask must be boolean and match the router token dimensions')
    return token_mask.to(scores.dtype)


def router_load_balancing_loss(routing, *, token_mask=None, sequence_wise=False):
    """E*sum(mean(score)*selection_frequency); top-k frequency is normalized by k.

    Scores are those of the configured router, including sigmoid when selected.
    sequence_wise averages per-batch-item losses instead of mixing sequences.
    No loss is silently added to the model objective.
    """
    scores, indices = routing.scores, routing.indices
    if indices.shape[:-1] != scores.shape[:-1] or indices.device != scores.device:
        raise ValueError('routing score and index layouts differ')
    if type(sequence_wise) is not bool or (sequence_wise and scores.ndim < 3):
        raise ValueError('sequence-wise loss requires a batch and token axis')
    mask = _loss_mask(scores, token_mask)
    experts, top_k = scores.shape[-1], indices.shape[-1]
    if top_k < 1:
        raise ValueError('load-balancing loss requires at least one selected expert')
    batch = scores.shape[0] if sequence_wise else 1
    probabilities = scores.reshape(batch, -1, experts)
    ids, mask = indices.reshape(batch, -1, top_k), mask.reshape(batch, -1)
    count = mask.sum(-1, keepdim=True)
    denominator = count.clamp(min=1)
    mean_scores = (probabilities * mask.unsqueeze(-1)).sum(1) / denominator
    frequencies = []
    for expert in range(experts):
        frequencies.append(((ids == expert).to(scores.dtype) * mask.unsqueeze(-1)).sum((1, 2)) / (denominator[:, 0] * top_k))
    loss = experts * (mean_scores * torch.stack(frequencies, -1)).sum(-1)
    valid = (count[:, 0] > 0).to(scores.dtype)
    return (loss * valid).sum() / valid.sum().clamp(min=1)


def router_z_loss(logits, *, token_mask=None):
    """Mean squared log-partition over valid tokens, with same-device gradients."""
    dtype = work_dtype(logits)
    if logits.ndim < 2 or logits.shape[-1] == 0:
        raise ValueError('router logits require a positive expert dimension')
    mask = _loss_mask(logits.to(dtype), token_mask)
    with precision_context(logits):
        x = logits.to(dtype)
        maximum = x.amax(-1, keepdim=True)
        partition = maximum.squeeze(-1) + (x - maximum).exp().sum(-1).log()
        return (partition.square() * mask).sum() / mask.sum().clamp(min=1)


def _expert_linear(x, weights, ids):
    if x.device.type == 'ruda':
        from .blas import grouped_gemm
        return grouped_gemm(x.contiguous(), weights.contiguous(), ids.contiguous())
    return torch.bmm(weights.index_select(0, ids), x.unsqueeze(-1)).squeeze(-1)


class SwiGLUExperts(torch.nn.Module):
    """Packed [experts,out,in] parameters; RUDA uses existing grouped GEMM/backward."""
    def __init__(self, num_experts, hidden_size, intermediate_size, *, device=None, dtype=None):
        super().__init__()
        for name, size in (('num_experts', num_experts), ('hidden_size', hidden_size), ('intermediate_size', intermediate_size)):
            positive_int(size, name)
        self.num_experts, self.hidden_size, self.intermediate_size = num_experts, hidden_size, intermediate_size
        self.gate_up = torch.nn.Parameter(torch.empty(num_experts, 2 * intermediate_size, hidden_size, device=device, dtype=dtype))
        self.down = torch.nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size, device=device, dtype=dtype))
        torch.nn.init.uniform_(self.gate_up, -hidden_size**-.5, hidden_size**-.5)
        torch.nn.init.uniform_(self.down, -intermediate_size**-.5, intermediate_size**-.5)

    def forward(self, hidden_states, routing):
        if hidden_states.ndim < 2 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError('expert input width differs from hidden_size')
        if (routing.indices.shape[:-1] != hidden_states.shape[:-1] or routing.weights.shape != routing.indices.shape or
                routing.indices.device != hidden_states.device or routing.weights.device != hidden_states.device or
                routing.scores.shape[-1] != self.num_experts):
            raise ValueError('routing result does not match expert inputs')
        if any(p.device != hidden_states.device or p.dtype != hidden_states.dtype for p in (self.gate_up, self.down)):
            raise ValueError('expert parameters and input must share dtype and device')
        if hidden_states.device.type == 'ruda' and hidden_states.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError('native grouped experts require FP16, BF16 or FP32')
        if hidden_states.numel() == 0:
            return hidden_states + (self.gate_up.sum() + self.down.sum() + routing.weights.sum()) * 0
        slots = routing.indices.shape[-1]
        rows = hidden_states.reshape(-1, self.hidden_size).unsqueeze(1).expand(-1, slots, -1).reshape(-1, self.hidden_size)
        ids = routing.indices.reshape(-1)
        gate, up = _expert_linear(rows, self.gate_up, ids).split(self.intermediate_size, -1)
        activations = (gate.to(work_dtype(gate)) * gate.to(work_dtype(gate)).sigmoid() * up.to(work_dtype(up))).to(gate.dtype)
        output = _expert_linear(activations, self.down, ids).reshape(*routing.indices.shape, self.hidden_size)
        output = (output.to(work_dtype(output)) * routing.weights.unsqueeze(-1)).sum(-2)
        return output.to(hidden_states.dtype)


@dataclass(frozen=True)
class MoEOutput:
    hidden_states: torch.Tensor
    routing: RoutingResult
    router_logits: torch.Tensor


class MixtureOfExperts(torch.nn.Module):
    """Composable router/expert layer; shared branch and its gate are caller-supplied."""
    def __init__(self, router, experts, policy, *, shared_experts=None, shared_gate=None, correction_bias=None):
        super().__init__()
        if not isinstance(policy, RoutingPolicy):
            raise TypeError('supply an explicit RoutingPolicy')
        if shared_gate is not None and shared_experts is None:
            raise ValueError('shared_gate requires a shared_experts branch')
        self.router, self.experts, self.policy = router, experts, policy
        self.shared_experts, self.shared_gate = shared_experts, shared_gate
        self.register_buffer('correction_bias', None if correction_bias is None else correction_bias.detach().clone())

    def forward(self, hidden_states):
        logits = self.router(hidden_states)
        routing = route_experts(logits, self.policy, correction_bias=self.correction_bias)
        output = self.experts(hidden_states, routing)
        if self.shared_experts is not None:
            shared = self.shared_experts(hidden_states)
            if shared.shape != output.shape:
                raise ValueError('shared experts must preserve the hidden-state shape')
            if self.shared_gate is not None:
                gate = self.shared_gate(hidden_states)
                if gate.shape not in (shared.shape, (*shared.shape[:-1], 1)):
                    raise ValueError('shared gate must have one or hidden_size values per token')
                shared = shared * gate.sigmoid()
            output = output + shared
        return MoEOutput(output, routing, logits)
