"""Sequence-sharded TP projections with all-gather/reduce-scatter adjoints."""
from __future__ import annotations

import torch
from torch.nn import functional as F
from ._expert_exchange import _group, register_group
from .parallel_training import ColumnParallelLinear, RowParallelLinear, scatter_to_tensor_parallel


@torch.library.custom_op('ruda::sequence_all_gather', mutates_args=())
def _gather(value: torch.Tensor, axis: int, group_key: str) -> torch.Tensor:
    return _group(group_key).all_gather(value, axis=axis)


@_gather.register_fake
def _gather_fake(value, axis, group_key):
    shape = list(value.shape)
    shape[axis] *= _group(group_key).world_size
    return value.new_empty(shape)


@torch.library.custom_op('ruda::sequence_reduce_scatter', mutates_args=())
def _reduce_scatter(value: torch.Tensor, axis: int, group_key: str) -> torch.Tensor:
    return _group(group_key).reduce_scatter(value, axis=axis)


@_reduce_scatter.register_fake
def _reduce_scatter_fake(value, axis, group_key):
    shape = list(value.shape)
    world = _group(group_key).world_size
    torch._check(shape[axis] % world == 0)
    shape[axis] //= world
    return value.new_empty(shape)


def _context(ctx, inputs, output):
    _, ctx.axis, ctx.group_key = inputs


def _gather_gradient(ctx, gradient):
    return _reduce_scatter(gradient.contiguous(), ctx.axis, ctx.group_key), None, None


def _scatter_gradient(ctx, gradient):
    return _gather(gradient.contiguous(), ctx.axis, ctx.group_key), None, None


_gather.register_autograd(_gather_gradient, setup_context=_context)
_reduce_scatter.register_autograd(_scatter_gradient, setup_context=_context)


def gather_sequence_parallel(value, group, *, axis=1, group_key='sequence-parallel'):
    """Gather sequence slices; backward sums contributions before scattering."""
    return _gather(value, axis, register_group(group_key, group))


def reduce_scatter_sequence_parallel(value, group, *, axis=1, group_key='sequence-parallel'):
    """Reduce partial TP outputs directly into sequence shards."""
    return _reduce_scatter(value, axis, register_group(group_key, group))


def scatter_sequence_parallel(value, group, *, axis=1):
    """Slice a logically replicated input; backward gathers disjoint gradients."""
    return scatter_to_tensor_parallel(value, group, axis=axis)


class SequenceParallelColumnLinear(ColumnParallelLinear):
    def __init__(self, linear, group, *, sequence_axis=1, group_key='sequence-parallel', gather_input=True):
        if isinstance(linear, ColumnParallelLinear):
            torch.nn.Module.__init__(self)
            if linear.group.group is not group.group:
                raise ValueError('SP column conversion requires the original tensor group')
            for name in ('group', 'in_features', 'out_features', 'local_out_features', 'gather_output', 'float8_recipe'):
                setattr(self, name, getattr(linear, name))
            self.weight, self.bias = linear.weight, linear.bias
            self.replicas = getattr(linear, 'replicas', 1)
            self.gather_output = False
            if hasattr(linear, 'scale_state'):
                self.scale_state = linear.scale_state
            self.train(linear.training)
        else:
            super().__init__(linear, group, gather_output=False)
        self.sequence_axis = sequence_axis
        self.group_key = register_group(group_key, group)
        self.gather_input = gather_input

    def forward(self, value):
        from ._tp_replicas import scale_input_gradient, column_output
        if self.gather_input:
            value = _gather(value, self.sequence_axis, self.group_key)
        replicas = getattr(self, 'replicas', 1)
        value = scale_input_gradient(value, replicas)
        if self.float8_recipe is not None:
            from .float8_parallel import sequence_linear_forward
            output = sequence_linear_forward(self, value, 0)
        else:
            output = F.linear(value, self.weight, self.bias)
        return column_output(output, self.group, replicas, False)


class SequenceParallelRowLinear(RowParallelLinear):
    def __init__(self, linear, group, *, sequence_axis=1, group_key='sequence-parallel'):
        if isinstance(linear, RowParallelLinear):
            torch.nn.Module.__init__(self)
            if linear.group.group is not group.group:
                raise ValueError('SP row conversion requires the original tensor group')
            for name in ('group', 'in_features', 'out_features', 'local_in_features', 'input_is_parallel', 'float8_recipe'):
                setattr(self, name, getattr(linear, name))
            self.weight, self.bias = linear.weight, linear.bias
            self.input_is_parallel = True
            if hasattr(linear, 'scale_state'):
                self.scale_state = linear.scale_state
            self.train(linear.training)
        else:
            super().__init__(linear, group, input_is_parallel=True)
        self.sequence_axis = sequence_axis
        self.group_key = register_group(group_key, group)
        if self.bias is not None:
            self.bias._ruda_sp_replicated = True

    def forward(self, value):
        if self.float8_recipe is not None:
            from .float8_parallel import sequence_linear_forward
            partial = sequence_linear_forward(self, value, 1)
        else:
            partial = F.linear(value, self.weight, None)
        output = _reduce_scatter(partial, self.sequence_axis, self.group_key)
        if self.float8_recipe is not None:
            if self.bias is not None:
                output = output + self.bias.float()
            dtype = torch.get_autocast_dtype('ruda') if torch.is_autocast_enabled('ruda') else value.dtype
            return output.to(dtype)
        return output if self.bias is None else output + self.bias


def sequence_projection(linear, group, *, axis, sequence_axis=1, group_key='sequence-parallel'):
    """Convert an already TP-partitioned region projection without changing names."""
    if isinstance(linear, ColumnParallelLinear) and axis == 0:
        return SequenceParallelColumnLinear(linear, group, sequence_axis=sequence_axis, group_key=group_key, gather_input=False)
    if isinstance(linear, RowParallelLinear) and axis == 1:
        return SequenceParallelRowLinear(linear, group, sequence_axis=sequence_axis, group_key=group_key)
    import copy
    from .parallel_adapters import _ParallelLoRA, _ParallelNF4
    from .packed_parallel import _ParallelPacked
    if not isinstance(linear, (_ParallelLoRA, _ParallelNF4, _ParallelPacked)) or linear.axis != axis:
        raise TypeError('SP region requires a supported TP projection')
    if linear.group.group is not group.group:
        raise ValueError('SP projection must retain the actual tensor group')
    result = copy.copy(linear)
    result._modules = linear._modules.copy()
    result._ruda_sequence_axis = sequence_axis
    result._ruda_sequence_key = register_group(group_key, group)
    if axis == 0:
        result.gather_output = False
    else:
        result.input_is_parallel = True
        if getattr(result, 'bias', None) is not None and result.bias.requires_grad:
            result.bias._ruda_sp_replicated = True
    if isinstance(linear, _ParallelLoRA):
        result.base = sequence_projection(linear.base, group, axis=axis, sequence_axis=sequence_axis, group_key=group_key)
    return result


def sequence_parallelize(model, group, *, column_paths, row_paths, replicated_parameter_names=(),
                         sequence_axis=1, group_key='sequence-parallel'):
    """Apply an explicit projection plan, retaining sequence shards between regions."""
    from .finetuning import _replace
    if (isinstance(column_paths, str) or isinstance(row_paths, str) or
            len(set(column_paths) | set(row_paths)) != len(column_paths) + len(row_paths)):
        raise ValueError('sequence projection paths must be unique and disjoint')
    replacements = []
    for paths, constructor in ((column_paths, SequenceParallelColumnLinear), (row_paths, SequenceParallelRowLinear)):
        for path in paths:
            replacements.append((path, constructor(model.get_submodule(path), group,
                sequence_axis=sequence_axis, group_key=group_key)))
    parameters = dict(model.named_parameters())
    replicated = [parameters[name] for name in replicated_parameter_names]
    _replace(model, replacements)
    for parameter in replicated:
        parameter._ruda_sp_replicated = True
    return model


@torch.no_grad()
def synchronize_sequence_gradients(parameters, group, originals=None):
    parameters = list(parameters)
    originals = parameters if originals is None else list(originals)
    selected = [parameter for parameter, original in zip(parameters, originals, strict=True)
                if getattr(original, '_ruda_sp_replicated', False) and original.requires_grad]
    presence = group.gather_metadata([parameter.grad is not None for parameter in selected])
    if any(len(flags) != len(selected) for flags in presence):
        raise ValueError('sequence-replicated parameter schemas differ')
    for index, parameter in enumerate(selected):
        if not any(flags[index] for flags in presence):
            continue
        value = (torch.zeros_like(parameter, dtype=torch.float32) if parameter.grad is None
                 else parameter.grad.detach().float().contiguous().clone())
        group.sum_(value)
        parameter.grad = value.to(parameter.dtype)


class SequenceGradientOverlap:
    def __init__(self, pending, parameters, group):
        self.pending, self.parameters, self.group = pending, list(parameters), group

    def __getattr__(self, name):
        return getattr(self.pending, name)

    def finish(self, **options):
        result = self.pending.finish(**options)
        synchronize_sequence_gradients(self.parameters, self.group)
        return result
