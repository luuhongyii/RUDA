"""Device-selected projections over original, separately registered expert weights."""
from __future__ import annotations
import torch
from torch.autograd.function import once_differentiable


def metadata(value, weights, ids, start, gradient=None):
    if not weights or len(weights)>16 or type(start) is not int or not 0<=start<=0xffffffff-len(weights):
        raise ValueError('indexed expert bank requires 1..16 weights and a valid expert offset')
    if value.ndim!=2 or ids.shape!=(value.shape[0],) or weights[0].ndim!=2:
        raise ValueError('indexed experts require [rows,input], [output,input] weights and [rows] IDs')
    if value.shape[1]<1 or weights[0].shape[0]<1 or weights[0].shape[1]!=value.shape[1]:
        raise ValueError('indexed expert projection dimensions differ')
    if value.dtype not in (torch.float32,torch.float16,torch.bfloat16) or ids.dtype not in (torch.int32,torch.int64):
        raise ValueError('indexed expert storage or ID dtype differs')
    if ids.device!=value.device or any(w.shape!=weights[0].shape or w.dtype!=value.dtype or w.device!=value.device for w in weights):
        raise ValueError('indexed expert weights must share geometry, dtype and device')
    if gradient is not None and (gradient.shape!=(value.shape[0],weights[0].shape[0]) or gradient.dtype!=value.dtype or gradient.device!=value.device):
        raise ValueError('indexed expert gradient differs')


@torch.library.custom_op('ruda::indexed_expert_linear',mutates_args=())
def forward(value:torch.Tensor,weights:list[torch.Tensor],ids:torch.Tensor,start:int)->torch.Tensor:
    from . import _C, _indexed_expert_available
    metadata(value,weights,ids,start)
    if not _indexed_expert_available:
        raise RuntimeError('indexed expert API 1 requires the matching RUDA native library and extension')
    return _C.indexed_expert_linear(value,weights,ids,start)


@forward.register_fake
def _fake(value,weights,ids,start):
    metadata(value,weights,ids,start)
    return value.new_empty((value.shape[0],weights[0].shape[0]))


@torch.library.custom_op('ruda::indexed_expert_backward',mutates_args=())
def backward(value:torch.Tensor,weights:list[torch.Tensor],ids:torch.Tensor,gradient:torch.Tensor,start:int)->list[torch.Tensor]:
    from . import _C
    metadata(value,weights,ids,start,gradient)
    return _C.indexed_expert_backward(value,weights,ids,gradient,start)


@backward.register_fake
def _backward_fake(value,weights,ids,gradient,start):
    metadata(value,weights,ids,start,gradient)
    return [torch.empty_like(value),*[torch.empty_like(w) for w in weights]]


@torch.library.custom_op('ruda::indexed_expert_backward_selected',mutates_args=())
def backward_selected(value:torch.Tensor,weights:list[torch.Tensor],ids:torch.Tensor,gradient:torch.Tensor,
                      start:int,need_input:bool,needed:list[bool])->list[torch.Tensor]:
    from . import _C,_indexed_expert_training_available
    metadata(value,weights,ids,start,gradient)
    if not _indexed_expert_training_available:
        raise RuntimeError('indexed expert training API 1 requires matching native components')
    return _C.indexed_expert_backward_selected(value,weights,ids,gradient,start,need_input,needed)


@backward_selected.register_fake
def _selected_fake(value,weights,ids,gradient,start,need_input,needed):
    metadata(value,weights,ids,start,gradient)
    if len(needed)!=len(weights):raise ValueError('indexed expert derivative selection differs')
    return [torch.empty_like(value) if need_input else value.new_empty(0),
            *[torch.empty_like(w) if need else w.new_empty(0) for w,need in zip(weights,needed,strict=True)]]


@torch.library.custom_op('ruda::indexed_expert_activity',mutates_args=())
def activity(ids:torch.Tensor,start:int,count:int)->torch.Tensor:
    from . import _C,_indexed_expert_training_available
    if not _indexed_expert_training_available:
        raise RuntimeError('indexed expert activity requires matching training API 1 components')
    return _C.indexed_expert_activity(ids,start,count)


@activity.register_fake
def _activity_fake(ids,start,count):
    return torch.empty(count,device=ids.device,dtype=torch.int32)


@torch.library.custom_op('ruda::optional_expert_gradient',mutates_args=())
def optional_gradient(gradient:torch.Tensor,active:torch.Tensor,index:int)->torch.Tensor:
    raise RuntimeError('optional expert gradients must be lowered by the RUDA AOT backend before execution')


@optional_gradient.register_fake
def _optional_gradient_fake(gradient,active,index):
    return torch.empty_like(gradient)


class _ExpertParameter(torch.autograd.Function):
    @staticmethod
    def forward(ctx,weight,active,index,host_cache):
        ctx.save_for_backward(active)
        ctx.index,ctx.host_cache=index,host_cache
        ctx.set_materialize_grads(False)
        return weight

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        if gradient is None:return None,None,None,None
        active,=ctx.saved_tensors
        if torch.compiler.is_compiling() or isinstance(gradient,torch._subclasses.fake_tensor.FakeTensor):
            result=optional_gradient(gradient,active,ctx.index)
        else:
            if ctx.host_cache[0] is None:ctx.host_cache[0]=active.to('cpu').tolist()
            result=gradient if ctx.host_cache[0][ctx.index] else None
        return result,None,None,None


def expert_weights(weights,ids,start=0):
    if not torch.is_grad_enabled() or not any(weight.requires_grad for weight in weights):return weights
    active=activity(ids,start,len(weights))
    host_cache=[None]
    return [_ExpertParameter.apply(weight,active,index,host_cache) if weight.requires_grad else weight
            for index,weight in enumerate(weights)]


def _context(ctx,inputs,output):
    value,weights,ids,ctx.start=inputs
    ctx.save_for_backward(value,ids,*weights)
    ctx.set_materialize_grads(False)


@once_differentiable
def _gradient(ctx,gradient):
    value,ids,*weights=ctx.saved_tensors
    if gradient is None:return None,[None]*len(weights),None,None
    needed=list(ctx.needs_input_grad[1])
    gradients=backward_selected(value,weights,ids,gradient.contiguous(),ctx.start,ctx.needs_input_grad[0],needed)
    weight_gradients=[dw if need else None for dw,need in zip(gradients[1:],needed,strict=True)]
    return (gradients[0] if ctx.needs_input_grad[0] else None,
            weight_gradients,None,None)


forward.register_autograd(_gradient,setup_context=_context)


def linear(value,weights,ids,*,track_usage=True,start=0):
    if track_usage:weights=expert_weights(weights,ids,start)
    output=None
    for begin in range(0,len(weights),16):
        bank=[w.to(value.dtype).contiguous() for w in weights[begin:begin+16]]
        part=forward(value.contiguous(),bank,ids.contiguous(),start+begin)
        output=part if output is None else output+part
    return output


def _routed_dropout(value,p,group,sharded):
    if not p:return value
    if p==1:return torch.where(torch.zeros_like(value,dtype=torch.bool),value,torch.zeros_like(value))
    from ._parallel_rng import reserve_attention
    from ._attention_dropout import mask
    from ._collective_ops import group_key
    width=value.shape[-1]
    global_width=width*group.world_size if sharded else width
    start=width*group.rank if sharded else 0
    shape=[1,1,value.shape[0],global_width]
    keys=[] if group is None else [group_key(group)]
    state=reserve_attention(value,shape,keys,float(p),torch.empty(0,device='cpu',dtype=torch.int64))
    multiplier=mask(value.new_empty(0),state,[1,1,value.shape[0],width],shape,[0,0,start],float(p))
    return value*multiplier.reshape(value.shape)


def dense_bank(experts,value,ids,*,names=('w1','w3','w2'),activation=None):
    from torch import nn
    from .parallel_training import ColumnParallelLinear,RowParallelLinear,copy_to_tensor_parallel,reduce_from_tensor_parallel
    from .parallel_adapters import (ColumnParallelLoRALinear,RowParallelLoRALinear,
                                   ColumnParallelNF4Linear,RowParallelNF4Linear)
    from .packed_parallel import ColumnParallelPackedLinear,RowParallelPackedLinear
    from .packed_quantization import PackedLinear
    from .finetuning import LoRALinear,NF4Linear
    from ._architecture_ops import precision_context
    if not experts or value.device.type!='ruda':return None
    layers=[tuple(getattr(expert,name) for name in names) for expert in experts]
    adapters=(LoRALinear,ColumnParallelLoRALinear,RowParallelLoRALinear)
    ordinary=(nn.Linear,NF4Linear,PackedLinear)
    column_types=(ColumnParallelLinear,ColumnParallelNF4Linear,ColumnParallelPackedLinear)
    row_types=(RowParallelLinear,RowParallelNF4Linear,RowParallelPackedLinear)
    bases=[tuple(layer.base if type(layer) in adapters else layer for layer in trio) for trio in layers]
    if any(module._forward_hooks or module._forward_pre_hooks or module._backward_hooks or module._backward_pre_hooks
           for expert in experts for module in expert.modules()):return None
    dense=all(type(base) in ordinary and base.bias is None and type(layer) in ordinary+(LoRALinear,)
              for trio,base_trio in zip(layers,bases,strict=True) for layer,base in zip(trio,base_trio,strict=True))
    parallel=all(type(gate) in column_types and type(up) in column_types and type(down) in row_types
        and not gate.gather_output and not up.gather_output and down.input_is_parallel
        and all(getattr(layer,'bias',None) is None and getattr(layer,'float8_recipe',None) is None for layer in (gate,up,down))
        for gate,up,down in bases)
    if parallel:
        for trio in layers:
            if (any(type(layer) not in column_types+(ColumnParallelLoRALinear,) or layer.gather_output
                    or getattr(layer,'replicas',1)!=1 for layer in trio[:2])
                    or type(trio[2]) not in row_types+(RowParallelLoRALinear,) or not trio[2].input_is_parallel
                    or any(hasattr(layer,'_ruda_sequence_axis') for layer in trio)):
                return None
    if not dense and not parallel:
        from ._indexed_float8_experts import bank as float8_bank
        return float8_bank(experts,value,ids,names=names,activation=activation)
    local_bases=[tuple(getattr(layer,'local',layer) for layer in trio) for trio in bases]
    def shape(layer):
        return tuple(layer.weight.shape) if hasattr(layer,'weight') else (layer.out_features,layer.in_features)
    if any(any(shape(trio[column])!=shape(local_bases[0][column]) for trio in local_bases) for column in range(3)):
        return None
    if any(getattr(layer,'bias',None) is not None for trio in local_bases for layer in trio):return None
    group=bases[0][0].group if parallel else None
    if group is not None and any(layer.group is not group for bank in (bases,layers) for trio in bank for layer in trio):
        raise ValueError('indexed expert tensor shards must share their actual tensor group')
    automatic=torch.is_autocast_enabled(value.device.type)
    dtype=torch.get_autocast_dtype(value.device.type) if automatic else value.dtype
    if dtype not in (torch.float32,torch.float16,torch.bfloat16):return None
    if torch.is_grad_enabled() and not automatic and any(layer.weight.dtype!=dtype
            for trio in local_bases for layer in trio if hasattr(layer,'weight')):
        return None
    policies=[]
    for column in range(3):
        selected=[trio[column] for trio in layers]
        adapted=[type(layer) in adapters for layer in selected]
        if any(adapted) and not all(adapted):return None
        if not any(adapted):
            policies.append(None)
            continue
        first=selected[0]
        policy=(first.lora_A.dtype,first.rank,first.scaling,first.dropout if first.training else 0.)
        if any((layer.lora_A.dtype,layer.rank,layer.scaling,layer.dropout if layer.training else 0.)!=policy
               or layer.lora_A.dtype!=layer.lora_B.dtype for layer in selected):return None
        policies.append(policy)
    act=torch.nn.functional.silu if activation is None else activation

    def projection(rows,column):
        from ._indexed_quantized_experts import linear as quantized_linear
        selected_bases=[trio[column] for trio in local_bases]
        result=None
        begin=0
        while begin<len(selected_bases):
            base_type=type(selected_bases[begin]);end=begin+1
            while end<len(selected_bases) and type(selected_bases[end]) is base_type:end+=1
            bank=selected_bases[begin:end]
            if base_type in (NF4Linear,PackedLinear):
                operand=rows if base_type is PackedLinear else rows.to(dtype)
                part=quantized_linear(operand,bank,ids,start=begin)
            else:
                part=linear(rows.to(dtype),[base.weight for base in bank],ids,start=begin)
            result=part if result is None else result+part
            begin=end
        row_parallel=group is not None and column==2
        if row_parallel:result=reduce_from_tensor_parallel(result,group)
        policy=policies[column]
        if policy is None:return result
        adapter_dtype,rank,scaling,p=policy
        selected=[trio[column] for trio in layers]
        adapted=_routed_dropout(rows.to(adapter_dtype),p,group,row_parallel)
        adapted=adapted.to(dtype if automatic else adapter_dtype)
        weights=expert_weights([layer.lora_A for layer in selected],ids)
        if group is not None and column!=2:
            weights=[copy_to_tensor_parallel(weight,group) for weight in weights]
        hidden=linear(adapted,weights,ids,track_usage=False)
        if row_parallel:hidden=reduce_from_tensor_parallel(hidden,group)
        update=linear(hidden,[layer.lora_B for layer in selected],ids)
        return result+(update*scaling).to(result.dtype)

    with precision_context(value):
        rows=copy_to_tensor_parallel(value,group) if group is not None else value
        gate=projection(rows,0)
        up=projection(rows,1)
        hidden=act(gate)*up
        return projection(hidden,2)
