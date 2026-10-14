"""Tensor-parallel FP8 linear training with full-logical-tensor current scales."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.autograd.function import once_differentiable

from .float8_training import Float8Recipe,_FLOATS,_STRATEGIES,_quantize,_encode,_mm,_power_of_two_scale


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
    group.max_(maximum)
    group.max_(invalid)
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
        if axis==1:group.sum_(output)
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
            if ctx.axis==0:ctx.group.sum_(dx)
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
    output=_Linear.apply(flat,weight.contiguous(),layer.group,recipe,axis)
    if bias is not None:output=output+bias.float()
    output=output.to(output_dtype).reshape(shape)
    if axis==0 and layer.gather_output:output=gather_from_tensor_parallel(output,layer.group)
    return output


def from_parallel(linear,*,recipe=None):
    """Replace a projection without replacing its existing master Parameters."""
    from .parallel_training import ColumnParallelLinear,RowParallelLinear
    if type(linear) not in (ColumnParallelLinear,RowParallelLinear) or linear.weight.dtype not in _FLOATS:
        raise TypeError('expected a native floating row/column parallel Linear')
    recipe=Float8Recipe() if recipe is None else recipe
    if not isinstance(recipe,Float8Recipe):raise TypeError('recipe must be Float8Recipe')
    result=type(linear).__new__(type(linear));nn.Module.__init__(result)
    for name in ('group','in_features','out_features'):
        setattr(result,name,getattr(linear,name))
    attributes=('local_out_features','gather_output') if isinstance(linear,ColumnParallelLinear) else ('local_in_features','input_is_parallel')
    for name in attributes:setattr(result,name,getattr(linear,name))
    result.weight,result.bias=linear.weight,linear.bias
    result.float8_recipe=recipe
    result.train(linear.training)
    return result


def recipe_contract(module):
    from .float8_training import Float8Linear
    from .parallel_training import ColumnParallelLinear,RowParallelLinear
    result=[]
    for name,child in module.named_modules():
        recipe=child.recipe if isinstance(child,Float8Linear) else getattr(child,'float8_recipe',None) if isinstance(child,(ColumnParallelLinear,RowParallelLinear)) else None
        if recipe is not None:
            result.append((name,str(recipe.input_dtype),str(recipe.weight_dtype),str(recipe.grad_dtype),
                           recipe.strategy,recipe.margin,recipe.power_of_two_scale))
    return result
