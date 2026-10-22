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
        self._initialize(num_experts, hidden_size, intermediate_size, group,
                         group_key=group_key, combine=combine, activation=activation)
        self.gate_up_proj = nn.Parameter(torch.empty(self.partition.local_experts,
            2 * intermediate_size, hidden_size, device=device, dtype=dtype))
        self.down_proj = nn.Parameter(torch.empty(self.partition.local_experts,
            hidden_size, intermediate_size, device=device, dtype=dtype))
        self._mark_parameters()

    def _initialize(self, num_experts, hidden_size, intermediate_size, group, *,
                    group_key, combine, activation):
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
        # Derived buffers are built on CPU and moved at use, including after to_empty(meta).
        self._owners, self._local_ids = self.partition.tables()
        self._owners.append(0)
        self._local_ids.append(-1)

    def _mark_parameters(self):
        from ._parameter_lora import parameter_weight
        for field, shape, axis in (('gate_up_proj', (self.num_experts, 2 * self.intermediate_size, self.hidden_size), 1),
                                   ('down_proj', (self.num_experts, self.hidden_size, self.intermediate_size), 2)):
            parameter, adapter = parameter_weight(self, field)
            parameter._ruda_ep_sharded = True
            parameter._ruda_ep_global_shape = shape
            parameter._ruda_tp_sharded = self.tensor_group is not None
            parameter._ruda_parameter_layout = dict(shape=shape, expert_group=self.group,
                tensor_group=self.tensor_group, tensor_axis=axis if self.tensor_group is not None else None,
                transform='expert-gate-up' if field == 'gate_up_proj' else None)
            if adapter is not None:
                adapter.configure_layout(parameter)
                adapter.mark_parameters()

    def shard_tensor_parallel(self, group, *, parameter_cache=None):
        if self.tensor_group is not None:
            raise ValueError('expert intermediates are already tensor-parallel')
        if self.intermediate_size % group.world_size or group.device_type != self.group.device_type:
            raise ValueError('expert intermediate width must divide the tensor group on the same device type')
        width = self.intermediate_size // group.world_size
        begin, end = width * group.rank, width * (group.rank + 1)
        from ._parameter_lora import parameter_weight, partition_parameter_adapter, register_parameter_lora
        from torch.nn.utils import parametrize
        gate_weight, gate_adapter = parameter_weight(self, 'gate_up_proj')
        down_weight, down_adapter = parameter_weight(self, 'down_proj')
        with torch.no_grad():
            gate = torch.cat((gate_weight[:, begin:end],
                              gate_weight[:, self.intermediate_size+begin:self.intermediate_size+end]), dim=1)
            down = down_weight[:, :, begin:end].contiguous().clone()
        for field, adapter in (('gate_up_proj', gate_adapter), ('down_proj', down_adapter)):
            if adapter is not None:
                parametrize.remove_parametrizations(self, field, leave_parametrized=False)
        cache={} if parameter_cache is None else parameter_cache
        for field,value,weight in (('gate_up_proj',gate,gate_weight),('down_proj',down,down_weight)):
            if id(weight) in cache:
                parameter,_,source_field=cache[id(weight)]
                if source_field!=field:raise ValueError('tied expert weights use different tensor partitions')
            else:
                parameter=nn.Parameter(value,requires_grad=weight.requires_grad)
            setattr(self,field,parameter)
        self.tensor_group, self.local_intermediate_size = group, width
        self._mark_parameters()
        for field,weight,adapter in (('gate_up_proj',gate_weight,gate_adapter),('down_proj',down_weight,down_adapter)):
            if adapter is not None:
                local_adapter=(cache[id(weight)][1] if id(weight) in cache else
                               partition_parameter_adapter(adapter,getattr(self,field)))
                if local_adapter is None:raise ValueError('tied expert weights have inconsistent adapters')
                register_parameter_lora(self,field,local_adapter)
            else:
                local_adapter=None
                if id(weight) in cache and cache[id(weight)][1] is not None:
                    raise ValueError('tied expert weights have inconsistent adapters')
            cache[id(weight)]=(parameter_weight(self,field)[0],local_adapter,field)
        return self

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._mark_parameters()
        return result

    @classmethod
    def from_module(cls, module, group, *, group_key, names=('gate_up_proj', 'down_proj'),
                    combine='float32', activation='storage', parameter_cache=None):
        from ._parameter_lora import parameter_weight, partition_parameter_adapter, register_parameter_lora
        sources = [parameter_weight(module, name) for name in names]
        result = cls.from_stacked(sources[0][0], sources[1][0], group, group_key=group_key,
                                  combine=combine, activation=activation)
        order = getattr(module, '_ruda_parameter_order', tuple(module._parameters))
        translated = dict(zip(names, ('gate_up_proj', 'down_proj')))
        result._ruda_parameter_order = tuple(translated[name] for name in order if name in translated)
        cache={} if parameter_cache is None else parameter_cache
        for field, (weight, adapter) in zip(('gate_up_proj', 'down_proj'), sources):
            if id(weight) in cache:
                local_weight,local_adapter,source_field=cache[id(weight)]
                if source_field!=field:raise ValueError('tied expert weights use different ownership layouts')
                setattr(result,field,local_weight)
                if (adapter is None)!=(local_adapter is None):
                    raise ValueError('tied expert weights have inconsistent adapters')
            else:
                local_weight=getattr(result,field)
                local_adapter=None if adapter is None else partition_parameter_adapter(adapter,local_weight)
                cache[id(weight)]=(local_weight,local_adapter,field)
            if adapter is not None:
                register_parameter_lora(result,field,local_adapter)
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
        return _SeparateExpertParallelMLP(experts, group, group_key=group_key,
            names=names, combine=combine, activation=activation)

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
                any(t.device != hidden_states.device for t in (top_k_index, top_k_weights, *self.parameters()))
                or hidden_states.device.type != self.group.device_type):
            raise ValueError('expert payloads, routing and parameters must share the communicator device')
        flat = hidden_states.reshape(-1, self.hidden_size)
        ids = top_k_index.reshape(-1).to(torch.int64)
        slots = top_k_index.shape[-1]
        owners = torch.tensor(self._owners, dtype=torch.int64, device=ids.device).index_select(0, ids)
        local = torch.tensor(self._local_ids, dtype=torch.int64, device=ids.device).index_select(0, ids)
        order = owners.argsort(stable=True)
        rows = flat.index_select(0, torch.div(order, slots, rounding_mode='floor'))
        from ._expert_exchange import backward_participation
        participate = backward_participation(rows, any(p.requires_grad for p in self.parameters()), self.group_key)
        if participate and not rows.requires_grad:
            rows = rows.detach().requires_grad_(True)
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


class _SeparateExpertParallelMLP(ExpertParallelMLP):
    def __init__(self, experts, group, *, group_key, names, combine, activation):
        if not experts or len(names) != 3 or len(set(names)) != 3:
            raise ValueError('separate experts require three distinct projection names')
        projections = [tuple(getattr(expert, name) for name in names) for expert in experts]
        if any(type(layer) is not nn.Linear or layer.bias is not None
               for trio in projections for layer in trio):
            raise ValueError('separate experts require bias-free dense SwiGLU projections')
        gate, up, down = (layer.weight for layer in projections[0])
        if gate.shape != up.shape or down.shape != gate.shape[::-1]:
            raise ValueError('expert SwiGLU projection dimensions differ')
        if any(weight.shape != expected.shape or weight.dtype != expected.dtype or weight.device != expected.device
               for trio in projections for weight, expected in zip((layer.weight for layer in trio), (gate, up, down))):
            raise ValueError('separate experts must share projection geometry, dtype and device')
        self._initialize(len(experts), gate.shape[1], gate.shape[0], group,
            group_key=group_key, combine=combine, activation=activation)
        self.source_kind, self.source_names = 'separate', tuple(names)
        for expert_id in range(self.partition.start, self.partition.end):
            self.add_module(str(expert_id), experts[expert_id])
        self._mark_parameters()

    def projection_items(self):
        for expert_id in range(self.partition.start, self.partition.end):
            expert = self._modules[str(expert_id)]
            for column, name in enumerate(self.source_names):
                yield expert_id, column, name, getattr(expert, name)

    def _mark_parameters(self):
        for expert_id, column, _, layer in self.projection_items():
            parameter = layer.weight
            parameter._ruda_ep_sharded = True
            parameter._ruda_ep_global_shape = ((self.intermediate_size, self.hidden_size)
                if column < 2 else (self.hidden_size, self.intermediate_size))
            parameter._ruda_tp_sharded = self.tensor_group is not None

    def shard_tensor_parallel(self, group):
        if self.tensor_group is not None:
            raise ValueError('expert intermediates are already tensor-parallel')
        if self.intermediate_size % group.world_size or group.device_type != self.group.device_type:
            raise ValueError('expert intermediate width must divide the tensor group on the same device type')
        width = self.intermediate_size // group.world_size
        begin, end = width * group.rank, width * (group.rank + 1)
        replacements = {}
        entries = [(column, layer, layer.weight) for _, column, _, layer in self.projection_items()]
        axes = {}
        for column, _, parameter in entries:
            axis = 0 if column < 2 else 1
            if id(parameter) in axes and axes[id(parameter)] != axis:
                raise ValueError('a tied expert parameter cannot use two different tensor partitions')
            axes[id(parameter)] = axis
        with torch.no_grad():
            for column, layer, parameter in entries:
                axis = 0 if column < 2 else 1
                key = (id(parameter), axis)
                if key not in replacements:
                    value = parameter[begin:end] if axis == 0 else parameter[:, begin:end]
                    replacements[key] = nn.Parameter(value.contiguous().clone(), requires_grad=parameter.requires_grad)
                layer.weight = replacements[key]
                layer.out_features, layer.in_features = layer.weight.shape
        self.tensor_group, self.local_intermediate_size = group, width
        self._mark_parameters()
        return self

    def _local_forward(self, rows, local_ids):
        from .parallel_training import copy_to_tensor_parallel, reduce_from_tensor_parallel
        if self.tensor_group is not None:
            rows = copy_to_tensor_parallel(rows, self.tensor_group)
        dtype = (torch.get_autocast_dtype(rows.device.type)
                 if torch.is_autocast_enabled(rows.device.type) else rows.dtype)
        with precision_context(rows):
            rows = rows.to(dtype)
            if rows.device.type == 'ruda':
                from ._indexed_experts import linear
                banks = [[layer.weight for _, column, _, layer in self.projection_items() if column == part]
                         for part in range(3)]
                gate = linear(rows, banks[0], local_ids)
                up = linear(rows, banks[1], local_ids)
                value = ((self.act_fn(gate.float()) * up.float()).to(dtype)
                         if self.activation == 'float32' else self.act_fn(gate) * up)
                output = linear(value, banks[2], local_ids)
            else:
                output = rows * 0
                for local, expert_id in enumerate(range(self.partition.start, self.partition.end)):
                    selected = (local_ids == local).nonzero().flatten()
                    if selected.numel() == 0:
                        continue
                    expert = self._modules[str(expert_id)]
                    gate, up, down = [getattr(expert, name).weight.to(dtype) for name in self.source_names]
                    value = rows.index_select(0, selected)
                    gate, up = F.linear(value, gate), F.linear(value, up)
                    value = ((self.act_fn(gate.float()) * up.float()).to(dtype)
                             if self.activation == 'float32' else self.act_fn(gate) * up)
                    output = output.index_add(0, selected, F.linear(value, down))
            if self.tensor_group is not None:
                output = reduce_from_tensor_parallel(output, self.tensor_group)
            return torch.where((local_ids >= 0)[:, None], output, torch.zeros_like(output))


def tensor_parallelize_experts(model, group):
    banks = [module for module in model.modules() if isinstance(module, ExpertParallelMLP)]
    if not banks:
        raise ValueError('partition expert ownership before tensor-parallelizing expert intermediates')
    if any(bank.tensor_group is not None or bank.intermediate_size % group.world_size for bank in banks):
        raise ValueError('all expert banks must have unpartitioned, divisible intermediate widths')
    cache={}
    for bank in banks:
        if isinstance(bank,_SeparateExpertParallelMLP):bank.shard_tensor_parallel(group)
        else:bank.shard_tensor_parallel(group,parameter_cache=cache)
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
    parameter_cache = {}
    for path, module in model.named_modules(remove_duplicate=False):
        if any(base.__name__ in class_names for base in type(module).__mro__):
            replacement = packed.get(id(module))
            if replacement is None:
                replacement = ExpertParallelMLP.from_module(module, group,
                    group_key=group_key, combine='expert_order', activation='storage',parameter_cache=parameter_cache)
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
    parameter_cache = {}
    for path, module in model.named_modules(remove_duplicate=False):
        if isinstance(module, SwiGLUExperts):
            replacement = packed.get(id(module))
            if replacement is None:
                parallel = ExpertParallelMLP.from_module(module, group, names=('gate_up', 'down'),
                    group_key=group_key, combine='float32', activation='float32',parameter_cache=parameter_cache)
                parallel.source_prefix, parallel.source_names = path, ('gate_up', 'down')
                replacement = packed[id(module)] = _RoutedExperts(parallel)
            replacements.append((path, replacement))
    if not replacements:
        raise ValueError('model has no RUDA SwiGLU expert layers')
    if replacements[0][0] == '':
        return replacements[0][1]
    _replace(model, replacements)
    return model
