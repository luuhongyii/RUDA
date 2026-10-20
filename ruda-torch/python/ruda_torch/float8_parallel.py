"""Tensor-parallel FP8 linear training with full-logical-tensor current scales."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.autograd.function import once_differentiable

from .float8_training import Float8Recipe,_FLOATS,_STRATEGIES,_quantize,_encode,_mm,_power_of_two_scale
from ._collective_ops import collective_reduce,group_key


def _sharded_quantize(value,dtype,recipe,group,sharded):
    if not sharded or group.world_size==1:
        return _quantize(value,dtype,recipe.margin,recipe.power_of_two_scale)
    value=value.detach()
    amax=value.float().abs().amax().reshape(1) if value.numel() else torch.zeros(1,device=value.device,dtype=torch.float32)
    finite=torch.isfinite(amax)
    maximum=torch.where(finite,amax,torch.zeros_like(amax))
    invalid=(~finite).to(torch.float32)
    # MAX alone need not propagate NaNs identically on every transport. The
    # separate flag retains the existing whole-tensor nonfinite -> scale 1 rule.
    maximum=collective_reduce(maximum,1,group_key(group))
    invalid=collective_reduce(invalid,1,group_key(group))
    bound=torch.finfo(dtype).max/(2.**recipe.margin)
    scale=torch.where((invalid==0)&(maximum>0),
        (maximum/bound).clamp_min(torch.finfo(torch.float32).tiny),torch.ones_like(maximum)).reshape(())
    if recipe.power_of_two_scale:scale=_power_of_two_scale(scale)
    return _encode(value,scale,dtype),scale


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,weight,group,recipe,axis):
        qx,sx=_sharded_quantize(value,recipe.input_dtype,recipe,group,axis==1)
        qw,sw=_sharded_quantize(weight,recipe.weight_dtype,recipe,group,True)
        output=_mm(qx,qw,sx,sw,False,True,_STRATEGIES[recipe.strategy])
        if axis==1:output=collective_reduce(output,0,group_key(group))
        ctx.save_for_backward(qx,qw,sx,sw)
        ctx.group,ctx.recipe,ctx.axis=group,recipe,axis
        ctx.input_dtype,ctx.weight_dtype=value.dtype,weight.dtype
        ctx.set_materialize_grads(False)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        if gradient is None:return (None,)*5
        qx,qw,sx,sw=ctx.saved_tensors;recipe=ctx.recipe
        qg,sg=_sharded_quantize(gradient.contiguous(),recipe.grad_dtype,recipe,ctx.group,ctx.axis==0)
        dx=dw=None
        if ctx.needs_input_grad[0]:
            dx=_mm(qg,qw,sg,sw,False,False,_STRATEGIES[recipe.strategy])
            if ctx.axis==0:dx=collective_reduce(dx,0,group_key(ctx.group))
            dx=dx.to(ctx.input_dtype)
        if ctx.needs_input_grad[1]:
            dw=_mm(qg,qx,sg,sx,True,False,_STRATEGIES[recipe.strategy]).to(ctx.weight_dtype)
        return dx,dw,None,None,None


def linear_forward(layer,value,axis):
    from .parallel_training import scatter_to_tensor_parallel,gather_from_tensor_parallel
    recipe=layer.float8_recipe
    if not isinstance(recipe,Float8Recipe):raise TypeError('parallel FP8 Linear requires Float8Recipe')
    if value.ndim<1 or value.dtype not in _FLOATS or value.device.type!='ruda':
        raise ValueError('parallel FP8 Linear needs floating RUDA input with a feature axis')
    expected=layer.local_in_features if axis==1 and layer.input_is_parallel else layer.in_features
    if value.shape[-1]!=expected:raise ValueError('parallel FP8 Linear input feature count differs')
    weight=layer.weight;bias=layer.bias
    if weight.device!=value.device or weight.dtype not in _FLOATS or weight.ndim!=2:
        raise ValueError('parallel FP8 Linear needs a floating master weight on the input device')
    width=layer.local_out_features if axis==0 else layer.out_features
    if tuple(weight.shape)!=(width,layer.in_features if axis==0 else layer.local_in_features):
        raise ValueError('parallel FP8 master weight shard geometry changed')
    if bias is not None and (bias.device!=value.device or bias.dtype not in _FLOATS or tuple(bias.shape)!=(width,)):
        raise ValueError('parallel FP8 Linear bias must match the output feature shard')
    output_dtype=torch.get_autocast_dtype('ruda') if torch.is_autocast_enabled('ruda') else value.dtype
    if output_dtype not in _FLOATS:raise ValueError('parallel FP8 Linear output dtype must be floating')
    if axis==1 and not layer.input_is_parallel:value=scatter_to_tensor_parallel(value,layer.group)
    shape=(*value.shape[:-1],width)
    flat=value.reshape(math.prod(value.shape[:-1]),weight.shape[1]).contiguous()
    if recipe.scaling == 'current':
        output=_Linear.apply(flat,weight.contiguous(),layer.group,recipe,axis)
    else:
        from .float8_scaling import recipe_linear
        output=recipe_linear(flat,weight.contiguous(),recipe,layer.scale_state,group=layer.group,axis=axis,reduce=True,observe=layer.training,
                            replicas=getattr(layer,'replicas',1))
    if bias is not None:output=output+bias.float()
    output=output.to(output_dtype).reshape(shape)
    if axis==0:
        from ._tp_replicas import column_output
        output=column_output(output,layer.group,getattr(layer,'replicas',1),layer.gather_output)
    return output


def from_parallel(linear,*,recipe=None):
    """Replace a projection without replacing its existing master Parameters."""
    from .parallel_training import ColumnParallelLinear,RowParallelLinear
    if not isinstance(linear,(ColumnParallelLinear,RowParallelLinear)) or linear.weight.dtype not in _FLOATS:
        raise TypeError('expected a native floating row/column parallel Linear')
    recipe=Float8Recipe() if recipe is None else recipe
    if not isinstance(recipe,Float8Recipe):raise TypeError('recipe must be Float8Recipe')
    result=type(linear).__new__(type(linear));nn.Module.__init__(result)
    for name in ('group','in_features','out_features'):
        setattr(result,name,getattr(linear,name))
    attributes=('local_out_features','gather_output') if isinstance(linear,ColumnParallelLinear) else ('local_in_features','input_is_parallel')
    for name in attributes:setattr(result,name,getattr(linear,name))
    for name in ('sequence_axis', 'group_key', 'gather_input', 'replicas'):
        if hasattr(linear, name):setattr(result, name, getattr(linear, name))
    result.weight,result.bias=linear.weight,linear.bias
    result.float8_recipe=recipe
    from .float8_scaling import make_scale_state
    result.scale_state=make_scale_state(recipe,result.weight,result.group,getattr(linear,'scale_state',None))
    result.train(linear.training)
    return result


class _SequenceLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, weight, group, recipe, axis):
        qx, sx = _sharded_quantize(value, recipe.input_dtype, recipe, group, axis == 1)
        qw, sw = _sharded_quantize(weight, recipe.weight_dtype, recipe, group, True)
        ctx.save_for_backward(qx, qw, sx, sw)
        ctx.group, ctx.recipe, ctx.axis = group, recipe, axis
        ctx.input_dtype, ctx.weight_dtype = value.dtype, weight.dtype
        return _mm(qx, qw, sx, sw, False, True, _STRATEGIES[recipe.strategy])

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        qx, qw, sx, sw = ctx.saved_tensors
        recipe = ctx.recipe
        qg, sg = _sharded_quantize(gradient.contiguous(), recipe.grad_dtype, recipe, ctx.group, ctx.axis == 0)
        dx = (_mm(qg, qw, sg, sw, False, False, _STRATEGIES[recipe.strategy]).to(ctx.input_dtype)
              if ctx.needs_input_grad[0] else None)
        dw = (_mm(qg, qx, sg, sx, True, False, _STRATEGIES[recipe.strategy]).to(ctx.weight_dtype)
              if ctx.needs_input_grad[1] else None)
        return dx, dw, None, None, None


def sequence_linear_forward(layer, value, axis):
    recipe = layer.float8_recipe
    if not isinstance(recipe, Float8Recipe) or value.device.type != 'ruda' or value.dtype not in _FLOATS:
        raise ValueError('sequence-parallel FP8 requires a RUDA floating input and Float8Recipe')
    width = layer.local_out_features if axis == 0 else layer.out_features
    expected = layer.in_features if axis == 0 else layer.local_in_features
    if value.shape[-1] != expected or layer.weight.shape != (width, expected):
        raise ValueError('sequence-parallel FP8 projection geometry differs')
    flat=value.reshape(-1, expected).contiguous()
    if recipe.scaling == 'current':
        output = _SequenceLinear.apply(flat, layer.weight.contiguous(), layer.group, recipe, axis)
    else:
        from .float8_scaling import recipe_linear
        output=recipe_linear(flat,layer.weight.contiguous(),recipe,layer.scale_state,group=layer.group,axis=axis,reduce=False,observe=layer.training,
                            replicas=getattr(layer,'replicas',1))
    output=output.reshape(*value.shape[:-1], width)
    if axis == 0:
        if layer.bias is not None:
            output = output + layer.bias.float()
        dtype = torch.get_autocast_dtype('ruda') if torch.is_autocast_enabled('ruda') else value.dtype
        output = output.to(dtype)
    return output


def recipe_contract(module):
    from .float8_training import Float8Linear
    from .parallel_training import ColumnParallelLinear,RowParallelLinear
    from .sharded_training import FullyShardedModule
    result=[]
    modules=list(module.named_modules())
    for prefix,child in list(modules):
        if isinstance(child,FullyShardedModule):
            modules.extend(('.'.join(part for part in (prefix,name) if part),layer)
                for name,layer in child._template.named_modules())
    for name,child in modules:
        recipe=child.recipe if isinstance(child,Float8Linear) else getattr(child,'float8_recipe',None) if isinstance(child,(ColumnParallelLinear,RowParallelLinear)) else None
        if recipe is not None:
            result.append((name,str(recipe.input_dtype),str(recipe.weight_dtype),str(recipe.grad_dtype),
                           recipe.strategy,recipe.margin,recipe.power_of_two_scale,
                           recipe.scaling,recipe.block_size,recipe.input_block_rows,recipe.weight_block_rows,
                           recipe.grad_block_rows,recipe.amax_history_len,recipe.amax_compute_algo,recipe.scaling_interval))
    return sorted(result)


def validate_recipe_contract(model, saved):
    actual = recipe_contract(model)
    if saved is None:
        if any(entry[7] != 'current' for entry in actual):
            raise ValueError('block/delayed FP8 checkpoint is missing its scaling recipe')
    elif saved != actual:
        raise ValueError('checkpoint FP8 scaling recipe differs')
