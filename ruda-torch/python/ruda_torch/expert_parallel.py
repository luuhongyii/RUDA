"""Sparse expert ownership, differentiable token exchange and local grouped GEMMs."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from ._architecture_ops import precision_context
from ._expert_exchange import exchange, return_rows, register_group


@dataclass(frozen=True)
class ExpertPartition:
    experts: int
    rank: int
    world_size: int

    def __post_init__(self):
        if (type(self.experts) is not int or type(self.world_size) is not int or
                type(self.rank) is not int or self.experts < self.world_size or
                self.world_size < 1 or not 0 <= self.rank < self.world_size):
            raise ValueError('expert ownership requires at least one expert per rank')

    @property
    def start(self):
        return self.experts * self.rank // self.world_size

    @property
    def end(self):
        return self.experts * (self.rank + 1) // self.world_size

    @property
    def local_experts(self):
        return self.end - self.start

    def tables(self):
        owners, local = [], []
        for rank in range(self.world_size):
            start = self.experts * rank // self.world_size
            end = self.experts * (rank + 1) // self.world_size
            owners.extend([rank] * (end - start))
            local.extend(range(end - start))
        return owners, local


class ExpertParallelMLP(nn.Module):
    """Local SwiGLU expert parameters and lossless variable-size all-to-all.

    Inputs are rank-local tokens, not replicas of the same token stream. No
    capacity clipping or token dropping is performed. Only per-peer row counts
    cross the host control boundary; token routing and tensor payloads stay on
    the execution device. Build the optimizer after partitioning the model.
    """
    def __init__(self, num_experts, hidden_size, intermediate_size, group, *,
                 group_key, device=None, dtype=None, combine='float32', activation='storage'):
        super().__init__()
        if (type(hidden_size) is not int or hidden_size < 1 or
                type(intermediate_size) is not int or intermediate_size < 1):
            raise ValueError('expert hidden and intermediate dimensions must be positive')
        if combine not in ('float32', 'expert_order') or activation not in ('storage', 'float32'):
            raise ValueError('invalid expert arithmetic ordering')
        self.partition = ExpertPartition(num_experts, group.rank, group.world_size)
        self.group, self.group_key = group, register_group(group_key, group)
        self.num_experts, self.hidden_size = num_experts, hidden_size
        self.intermediate_size = intermediate_size
        self.local_intermediate_size = intermediate_size
        self.tensor_group = None
        self.combine, self.activation = combine, activation
        self.act_fn = F.silu
        self.source_kind = 'stacked'
        self.source_names = ('gate_up_proj', 'down_proj')
        self.source_prefix = None
        self.gate_up_proj = nn.Parameter(torch.empty(self.partition.local_experts,
            2 * intermediate_size, hidden_size, device=device, dtype=dtype))
        self.down_proj = nn.Parameter(torch.empty(self.partition.local_experts,
            hidden_size, intermediate_size, device=device, dtype=dtype))
        self._mark_parameters()
        # Derived buffers are built on CPU and moved at use, including after to_empty(meta).
        self._owners, self._local_ids = self.partition.tables()
        self._owners.append(0)
        self._local_ids.append(-1)

    def _mark_parameters(self):
        for parameter, shape in ((self.gate_up_proj, (self.num_experts, 2 * self.intermediate_size, self.hidden_size)),
                                 (self.down_proj, (self.num_experts, self.hidden_size, self.intermediate_size))):
            parameter._ruda_ep_sharded = True
            parameter._ruda_ep_global_shape = shape
            parameter._ruda_tp_sharded = self.tensor_group is not None

    def shard_tensor_parallel(self, group):
        if self.tensor_group is not None:
            raise ValueError('expert intermediates are already tensor-parallel')
        if self.intermediate_size % group.world_size or group.device_type != self.group.device_type:
            raise ValueError('expert intermediate width must divide the tensor group on the same device type')
        width = self.intermediate_size // group.world_size
        begin, end = width * group.rank, width * (group.rank + 1)
        with torch.no_grad():
            gate = torch.cat((self.gate_up_proj[:, begin:end],
                              self.gate_up_proj[:, self.intermediate_size+begin:self.intermediate_size+end]), dim=1)
            down = self.down_proj[:, :, begin:end].contiguous().clone()
        self.gate_up_proj = nn.Parameter(gate, requires_grad=self.gate_up_proj.requires_grad)
        self.down_proj = nn.Parameter(down, requires_grad=self.down_proj.requires_grad)
        self.tensor_group, self.local_intermediate_size = group, width
        self._mark_parameters()
        return self

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._mark_parameters()
        return result

    @classmethod
    def from_stacked(cls, gate_up, down, group, *, group_key,
                     combine='float32', activation='storage'):
        if (gate_up.ndim != 3 or down.ndim != 3 or gate_up.shape[0] != down.shape[0]
                or gate_up.shape[1] != 2 * down.shape[2] or gate_up.shape[2] != down.shape[1]
                or gate_up.device != down.device or gate_up.dtype != down.dtype):
            raise ValueError('stacked experts require [E,2I,H] and [E,H,I]')
        result = cls(gate_up.shape[0], gate_up.shape[2], down.shape[2], group,
            group_key=group_key, device=gate_up.device, dtype=gate_up.dtype,
            combine=combine, activation=activation)
        start, end = result.partition.start, result.partition.end
        with torch.no_grad():
            result.gate_up_proj.copy_(gate_up[start:end])
            result.down_proj.copy_(down[start:end])
        result.gate_up_proj.requires_grad_(gate_up.requires_grad)
        result.down_proj.requires_grad_(down.requires_grad)
        return result

    @classmethod
    def from_linears(cls, experts, group, *, group_key, names=('gate_proj', 'up_proj', 'down_proj'),
                     combine='float32', activation='storage'):
        if not experts:
            raise ValueError('cannot partition an empty expert bank')
        first = [getattr(experts[0], name) for name in names]
        if any(not isinstance(layer, nn.Linear) or layer.bias is not None for layer in first):
            raise ValueError('expert packing requires bias-free dense SwiGLU projections')
        gate, up, down = [layer.weight for layer in first]
        if gate.shape != up.shape or down.shape != gate.shape[::-1]:
            raise ValueError('expert SwiGLU projection dimensions differ')
        result = cls(len(experts), gate.shape[1], gate.shape[0], group, group_key=group_key,
            device=gate.device, dtype=gate.dtype, combine=combine, activation=activation)
        flags = gate.requires_grad, up.requires_grad, down.requires_grad
        if flags[0] != flags[1]:
            raise ValueError('packed gate and up projections must share requires_grad')
        with torch.no_grad():
            for local, expert_id in enumerate(range(result.partition.start, result.partition.end)):
                layers = [getattr(experts[expert_id], name) for name in names]
                if any(not isinstance(layer, nn.Linear) or layer.bias is not None for layer in layers):
                    raise ValueError('expert packing requires bias-free dense SwiGLU projections')
                weights = [layer.weight for layer in layers]
                for weight, expected, trainable in zip(weights, (gate, up, down), flags):
                    if (weight.shape != expected.shape or weight.dtype != expected.dtype or
                            weight.device != expected.device or weight.requires_grad != trainable):
                        raise ValueError('local experts must have matching geometry, dtype and trainability')
                result.gate_up_proj[local, :result.intermediate_size].copy_(weights[0])
                result.gate_up_proj[local, result.intermediate_size:].copy_(weights[1])
                result.down_proj[local].copy_(weights[2])
        result.gate_up_proj.requires_grad_(flags[0])
        result.down_proj.requires_grad_(flags[2])
        result.source_kind, result.source_names = 'separate', tuple(names)
        return result

    def _local_forward(self, rows, local_ids):
        from .moe import _expert_linear
        if self.tensor_group is not None:
            from .parallel_training import copy_to_tensor_parallel
            rows = copy_to_tensor_parallel(rows, self.tensor_group)
        dtype = (torch.get_autocast_dtype(rows.device.type)
                 if torch.is_autocast_enabled(rows.device.type) else rows.dtype)
        with precision_context(rows):
            valid = local_ids >= 0
            ids = local_ids if rows.device.type == 'ruda' else local_ids.clamp_min(0)
            gate, up = _expert_linear(rows.to(dtype).contiguous(),
                self.gate_up_proj.to(dtype).contiguous(), ids.contiguous()).split(self.local_intermediate_size, -1)
            if self.activation == 'float32':
                value = (self.act_fn(gate.float()) * up.float()).to(dtype)
            else:
                value = self.act_fn(gate) * up
            output = _expert_linear(value.contiguous(), self.down_proj.to(dtype).contiguous(), ids.contiguous())
            if self.tensor_group is not None:
                from .parallel_training import reduce_from_tensor_parallel
                output = reduce_from_tensor_parallel(output, self.tensor_group)
            return torch.where(valid[:, None], output, torch.zeros_like(output))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if (hidden_states.ndim < 2 or hidden_states.shape[-1] != self.hidden_size or
                top_k_index.shape[:-1] != hidden_states.shape[:-1] or
                top_k_weights.shape != top_k_index.shape or top_k_index.shape[-1] < 1):
            raise ValueError('expert inputs and selected routing slots differ')
        if (top_k_index.dtype not in (torch.int32, torch.int64) or
                any(t.device != hidden_states.device for t in (top_k_index, top_k_weights,
                    self.gate_up_proj, self.down_proj)) or hidden_states.device.type != self.group.device_type):
            raise ValueError('expert payloads, routing and parameters must share the communicator device')
        flat = hidden_states.reshape(-1, self.hidden_size)
        ids = top_k_index.reshape(-1).to(torch.int64)
        slots = top_k_index.shape[-1]
        owners = torch.tensor(self._owners, dtype=torch.int64, device=ids.device).index_select(0, ids)
        local = torch.tensor(self._local_ids, dtype=torch.int64, device=ids.device).index_select(0, ids)
        order = owners.argsort(stable=True)
        rows = flat.index_select(0, torch.div(order, slots, rounding_mode='floor'))
        sends = torch.stack([(owners == rank).sum(dtype=torch.int64) for rank in range(self.group.world_size)])
        received, local_ids, receives = exchange(rows, local.index_select(0, order), sends, self.group_key)
        computed = self._local_forward(received, local_ids)
        returned = return_rows(computed, sends, receives, rows, self.group_key)
        output = returned.index_select(0, order.argsort()).reshape(*top_k_index.shape, self.hidden_size)
        weights = torch.where(top_k_index < self.num_experts, top_k_weights, torch.zeros_like(top_k_weights))
        with precision_context(output):
            if self.combine == 'float32':
                result = (output.float() * weights.float().unsqueeze(-1)).sum(-2).to(hidden_states.dtype)
            else:
                output = (output * weights.unsqueeze(-1)).to(hidden_states.dtype)
                expert_order = top_k_index.argsort(dim=-1, stable=True)
                output = output.gather(-2, expert_order.unsqueeze(-1).expand(*expert_order.shape, self.hidden_size))
                result = torch.zeros_like(hidden_states)
                for slot in range(slots):
                    result = result + output[..., slot, :]
        return result


def tensor_parallelize_experts(model, group):
    banks = [module for module in model.modules() if isinstance(module, ExpertParallelMLP)]
    if not banks:
        raise ValueError('partition expert ownership before tensor-parallelizing expert intermediates')
    if any(bank.tensor_group is not None or bank.intermediate_size % group.world_size for bank in banks):
        raise ValueError('all expert banks must have unpartitioned, divisible intermediate widths')
    for bank in banks:
        bank.shard_tensor_parallel(group)
    return model


def expert_parallelize_kimi(model, group, *, group_key='kimi-expert'):
    """Partition RUDA Kimi Linear experts; routers/shared branches stay replicated."""
    from ._kimi_layers import KimiSparseMoeBlock
    replacements = []
    for path, module in model.named_modules():
        if isinstance(module, KimiSparseMoeBlock):
            if isinstance(module.experts, ExpertParallelMLP):
                raise ValueError(f'{path}: experts are already partitioned')
            replacement = ExpertParallelMLP.from_linears(module.experts, group, group_key=group_key,
                names=('w1', 'w3', 'w2'), combine='float32', activation='storage')
            replacement.source_prefix = path + '.experts' if path else 'experts'
            replacements.append((module, replacement))
    if not replacements:
        raise ValueError('model has no RUDA Kimi Linear expert layers')
    for module, replacement in replacements:
        module.experts = replacement
    return model


def expert_parallelize_qwen35(model, group, *, group_key='qwen35-expert'):
    """Partition official Qwen3.5 stacked experts without changing the router."""
    return _parallelize_hf_stacked(model, group, group_key, ('Qwen3_5MoeExperts',))


def expert_parallelize_qwen3(model, group, *, group_key='qwen3-expert'):
    """Partition HF Qwen3 stacked experts, retaining router/auxiliary-loss behavior."""
    return _parallelize_hf_stacked(model, group, group_key, ('Qwen3MoeExperts',))


def expert_parallelize_deepseek(model, group, *, group_key='deepseek-expert'):
    """Partition HF DeepSeek V3 stacked experts; correction and shared experts stay intact."""
    return _parallelize_hf_stacked(model, group, group_key, ('DeepseekV3Experts',))


def _parallelize_hf_stacked(model, group, group_key, class_names):
    from .finetuning import _replace
    replacements = []
    packed = {}
    for path, module in model.named_modules(remove_duplicate=False):
        if type(module).__name__ in class_names:
            replacement = packed.get(id(module))
            if replacement is None:
                replacement = ExpertParallelMLP.from_stacked(module.gate_up_proj, module.down_proj, group,
                    group_key=group_key, combine='expert_order', activation='storage')
                replacement.act_fn = module.act_fn
                replacement.source_kind = 'hf'
                replacement.source_prefix = path
                replacement.train(module.training)
                packed[id(module)] = replacement
            replacements.append((path, replacement))
    if not replacements:
        raise ValueError(f'model has no supported stacked expert layers: {class_names}')
    if replacements[0][0] == '':
        return replacements[0][1]
    _replace(model, replacements)
    return model


class _RoutedExperts(nn.Module):
    def __init__(self, parallel):
        super().__init__()
        self.parallel = parallel
        self.num_experts = parallel.num_experts
        self.hidden_size = parallel.hidden_size
        self.intermediate_size = parallel.intermediate_size

    def forward(self, hidden_states, routing):
        return self.parallel(hidden_states, routing.indices, routing.weights)


def expert_parallelize_moe(model, group, *, group_key='ruda-expert'):
    """Partition RUDA SwiGLUExperts while retaining each RoutingPolicy and loss."""
    from .moe import SwiGLUExperts
    from .finetuning import _replace
    replacements = []
    packed = {}
    for path, module in model.named_modules(remove_duplicate=False):
        if isinstance(module, SwiGLUExperts):
            replacement = packed.get(id(module))
            if replacement is None:
                parallel = ExpertParallelMLP.from_stacked(module.gate_up, module.down, group,
                    group_key=group_key, combine='float32', activation='float32')
                parallel.source_prefix, parallel.source_names = path, ('gate_up', 'down')
                replacement = packed[id(module)] = _RoutedExperts(parallel)
            replacements.append((path, replacement))
    if not replacements:
        raise ValueError('model has no RUDA SwiGLU expert layers')
    if replacements[0][0] == '':
        return replacements[0][1]
    _replace(model, replacements)
    return model
