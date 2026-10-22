"""LoRA for explicitly selected matrix and stacked-expert parameters."""
from __future__ import annotations

import math
from collections.abc import Sequence
import torch
from torch import nn
from torch.nn.utils import parametrize


class LoRAParameter(nn.Module):
    """Weight-space LoRA, with independent A/B pairs on a leading expert axis.

    Parametrization keeps the owning architecture's actual forward and routing
    intact. The effective floating weight is materialized on access; this is not
    a quantized expert kernel or input-dropout adapter.
    """
    def __init__(self, weight, *, rank=16, alpha=16., adapter_dtype=torch.float32,
                 use_rslora=False, is_transposed=False):
        super().__init__()
        if (not isinstance(weight, nn.Parameter) or weight.ndim not in (2, 3)
                or weight.dtype not in (torch.float32, torch.float16, torch.bfloat16)
                or weight.device.type == 'meta' or any(size <= 0 for size in weight.shape)):
            raise ValueError('parameter LoRA requires a materialized floating [out,in] or [experts,out,in] parameter')
        if type(rank) is not int or rank <= 0:
            raise ValueError('parameter LoRA rank must be positive')
        layout = getattr(weight, '_ruda_parameter_layout', {})
        if (getattr(weight, '_ruda_tp_sharded', False) or getattr(weight, '_ruda_ep_sharded', False)) and not layout:
            raise ValueError('sharded parameter LoRA requires an explicit logical weight layout')
        if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('parameter LoRA alpha must be positive and finite')
        if adapter_dtype not in (torch.float32, torch.float16, torch.bfloat16) or type(use_rslora) is not bool:
            raise ValueError('invalid parameter LoRA dtype or RSLoRA policy')
        if type(is_transposed) is not bool or (is_transposed and weight.ndim != 3):
            raise ValueError('transposed parameter LoRA requires a 3-D expert parameter')
        self.is_transposed = is_transposed
        self.configure_layout(weight)
        self.rank, self.alpha, self.use_rslora = rank, float(alpha), use_rslora
        self.dropout = 0.
        outputs, inputs = weight.shape[-2:][::-1] if is_transposed else weight.shape[-2:]
        shape = (*weight.shape[:-2], rank, inputs)
        a = torch.empty(shape, dtype=adapter_dtype, device='cpu')
        nn.init.uniform_(a, -self.in_features ** -.5, self.in_features ** -.5)
        self.lora_A = nn.Parameter(a.to(weight.device))
        self.lora_B = nn.Parameter(torch.zeros((*weight.shape[:-2], outputs, rank),
                                              dtype=adapter_dtype, device='cpu').to(weight.device))
        self.mark_parameters()
        axis = parameter_adapter_axis(self, 'lora_A')
        if self.tensor_group is not None and axis is None:
            with torch.no_grad():
                self.tensor_group.broadcast_(self.lora_A)

    def configure_layout(self, weight):
        layout = getattr(weight, '_ruda_parameter_layout', {})
        self.weight_shape = tuple(layout.get('shape', weight.shape))
        self.local_weight_shape = tuple(weight.shape)
        self.expert_group = layout.get('expert_group')
        self.tensor_group = layout.get('tensor_group')
        self.tensor_axis = layout.get('tensor_axis')
        self.tensor_transform = layout.get('transform')
        dimensions = self.weight_shape[-2:]
        self.out_features, self.in_features = dimensions[::-1] if self.is_transposed else dimensions

    def mark_parameters(self):
        for field in ('lora_A', 'lora_B'):
            parameter = getattr(self, field)
            parameter._ruda_ep_sharded = self.expert_group is not None
            parameter._ruda_tp_sharded = parameter_adapter_axis(self, field) is not None

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self.mark_parameters()
        return result

    @property
    def scaling(self):
        return self.alpha / (math.sqrt(self.rank) if self.use_rslora else self.rank)

    def forward(self, weight):
        a, b = self.lora_A, self.lora_B
        if self.tensor_group is not None:
            from .parallel_training import copy_to_tensor_parallel
            if parameter_adapter_axis(self, 'lora_A') is None:
                a = copy_to_tensor_parallel(a, self.tensor_group)
            if parameter_adapter_axis(self, 'lora_B') is None:
                b = copy_to_tensor_parallel(b, self.tensor_group)
        update = b @ a
        if self.is_transposed:
            update = update.transpose(-2, -1)
        return weight + (update * self.scaling).to(weight.dtype)

    def get_extra_state(self):
        state = dict(version=1, weight_shape=self.weight_shape, rank=self.rank,
                     alpha=self.alpha, use_rslora=self.use_rslora)
        if self.is_transposed:
            state['is_transposed'] = True
        return state

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('parameter LoRA checkpoint configuration mismatch')


def parameter_lora_targets(model, paths, linear_paths):
    if paths is None:
        return []
    if (isinstance(paths, str) or not isinstance(paths, Sequence) or not paths
            or any(not isinstance(path, str) or not path for path in paths)
            or len(set(paths)) != len(paths)):
        raise ValueError('target_parameters must contain unique exact parameter names')
    parameters = dict(model.named_parameters(remove_duplicate=False))
    logical = {}
    if __package__ and any(path not in parameters for path in paths):
        from .sharded_adapter_interop import parameter_records
        logical = {name:record['weight'] for name,record in parameter_records(model).items() if record['layer'] is None}
    if any(path not in parameters and path not in logical for path in paths):
        raise ValueError('parameter LoRA target missing or already parametrized')
    identities = {id(parameters[path] if path in parameters else logical[path]) for path in paths}
    targets, seen = [], set()
    for path, weight in parameters.items():
        if id(weight) not in identities:
            continue
        owner_path, _, field = path.rpartition('.')
        owner = model.get_submodule(owner_path) if owner_path else model
        if any(owner_path == linear or owner_path.startswith(linear + '.') for linear in linear_paths):
            raise ValueError('a parameter target overlaps a module LoRA target; select only one')
        if parametrize.is_parametrized(owner, field):
            raise ValueError(f'parameter already parametrized: {path}')
        key = (id(owner), field)
        if key not in seen:
            targets.append((owner, field, weight))
            seen.add(key)
    return targets


def parameter_lora_entries(model):
    """Distinct owning fields, including aliases of the same frozen parameter."""
    for path, owner in model.named_modules():
        if not parametrize.is_parametrized(owner):
            continue
        for field, chain in owner.parametrizations.items():
            if len(chain) == 1 and isinstance(chain[0], LoRAParameter):
                yield path, owner, field, chain[0], chain.original


def adapter_shapes(layer):
    prefix = layer.weight_shape[:-2] if isinstance(layer, LoRAParameter) else ()
    return {'lora_A': (*prefix, layer.rank, layer.in_features),
            'lora_B': (*prefix, layer.out_features, layer.rank)}


def parameter_adapter_metadata(layer):
    state = dict(rank=layer.rank, alpha=layer.alpha, in_features=layer.in_features,
                 out_features=layer.out_features, base='parameter', weight_shape=layer.weight_shape,
                 dropout=0., use_rslora=layer.use_rslora)
    if layer.is_transposed:
        state['is_transposed'] = True
    return state


def parameter_weight(owner, field):
    if parametrize.is_parametrized(owner, field):
        chain = owner.parametrizations[field]
        if len(chain) != 1 or not isinstance(chain[0], LoRAParameter):
            raise ValueError('expert partitioning requires a single RUDA parameter adapter')
        return chain.original, chain[0]
    return getattr(owner, field), None


def register_parameter_lora(owner, field, adapter):
    if not hasattr(owner, '_ruda_parameter_order'):
        owner._ruda_parameter_order = tuple(owner._parameters)
    parametrize.register_parametrization(owner, field, adapter)


def parameter_adapter_axis(layer, field):
    if layer.tensor_group is None or layer.tensor_axis is None:
        return None
    out_axis = len(layer.weight_shape) - (1 if layer.is_transposed else 2)
    if layer.tensor_axis == out_axis:
        return len(layer.weight_shape) - 2 if field == 'lora_B' else None
    return len(layer.weight_shape) - 1 if field == 'lora_A' else None


def _tensor_slice(layer, field, value):
    axis = parameter_adapter_axis(layer, field)
    if axis is None:
        return value
    group = layer.tensor_group
    if layer.tensor_transform == 'expert-gate-up' and field == 'lora_B':
        gate, up = value.chunk(2, dim=axis)
        return torch.cat((gate.chunk(group.world_size, dim=axis)[group.rank],
                          up.chunk(group.world_size, dim=axis)[group.rank]), dim=axis).contiguous()
    return value.chunk(group.world_size, dim=axis)[group.rank].contiguous()


def local_parameter_adapter(layer, field, value):
    if layer.expert_group is not None:
        group = layer.expert_group
        count = layer.weight_shape[0]
        value = value[count * group.rank // group.world_size:count * (group.rank + 1) // group.world_size]
    return _tensor_slice(layer, field, value)


def full_parameter_adapter(layer, field, value=None):
    value = getattr(layer, field).detach() if value is None else value
    axis = parameter_adapter_axis(layer, field)
    if axis is not None:
        if layer.tensor_transform == 'expert-gate-up' and field == 'lora_B':
            gate, up = value.chunk(2, dim=axis)
            value = torch.cat((layer.tensor_group.all_gather(gate.contiguous(), axis=axis),
                               layer.tensor_group.all_gather(up.contiguous(), axis=axis)), dim=axis)
        else:
            value = layer.tensor_group.all_gather(value, axis=axis)
    if layer.expert_group is not None:
        group, count = layer.expert_group, layer.weight_shape[0]
        size = (count + group.world_size - 1) // group.world_size
        padded = value.new_zeros((size, *value.shape[1:]))
        padded[:value.shape[0]].copy_(value)
        gathered = group.all_gather(padded).view(group.world_size, size, *value.shape[1:])
        value = torch.cat([gathered[rank, :count * (rank + 1) // group.world_size - count * rank // group.world_size]
                           for rank in range(group.world_size)], dim=0)
    return value


@torch.no_grad()
def partition_parameter_adapter(source, weight):
    """Refine expert/tensor ownership before constructing the optimizer."""
    target = LoRAParameter(weight, rank=source.rank, alpha=source.alpha,
        adapter_dtype=source.lora_A.dtype, use_rslora=source.use_rslora,
        is_transposed=source.is_transposed).train(source.training)
    if source.weight_shape != target.weight_shape:
        raise ValueError('parameter adapter logical shape changed during partitioning')
    for field in ('lora_A', 'lora_B'):
        value = getattr(source, field).detach()
        if source.expert_group is None and target.expert_group is not None:
            group, count = target.expert_group, target.weight_shape[0]
            value = value[count * group.rank // group.world_size:count * (group.rank + 1) // group.world_size]
        elif source.expert_group is not target.expert_group:
            raise ValueError('changing existing expert ownership requires checkpoint resharding')
        if source.tensor_group is None:
            value = _tensor_slice(target, field, value)
        elif source.tensor_group is not target.tensor_group:
            raise ValueError('changing existing tensor ownership requires checkpoint resharding')
        parameter = getattr(target, field)
        parameter.copy_(value)
        parameter.requires_grad_(getattr(source, field).requires_grad)
        if target.tensor_group is not None and parameter_adapter_axis(target, field) is None:
            target.tensor_group.broadcast_(parameter)
    return target
