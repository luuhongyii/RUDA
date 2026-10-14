"""Per-expert FP8 training over separate master parameters and device row IDs."""
from __future__ import annotations
import torch
from torch.autograd.function import once_differentiable
from .float8_training import _FLOATS,_STRATEGIES,_power_of_two_scale


def metadata(value,weights,ids,start):
    if not 1<=len(weights)<=16 or type(start) is not int or not 0<=start<=0xffffffff-len(weights):
        raise ValueError('FP8 expert bank requires 1..16 weights and a uint32 expert interval')
    if value.device.type!='ruda' or value.ndim!=2 or value.shape[1]<1 or value.dtype not in _FLOATS:
        raise ValueError('FP8 expert activations require a floating RUDA matrix')
    if ids.shape!=(value.shape[0],) or ids.dtype not in (torch.int32,torch.int64) or ids.device!=value.device:
        raise ValueError('FP8 expert IDs must match the routed rows')
    shape=weights[0].shape
    if len(shape)!=2 or min(shape)<1:
        raise ValueError('FP8 expert master weights must be nonempty matrices')
    if any(w.shape!=shape or w.dtype not in _FLOATS or w.device!=value.device for w in weights):
        raise ValueError('FP8 expert weights must share shape and device')


@torch.library.custom_op('ruda::indexed_float8_statistics',mutates_args=())
def statistics(value:torch.Tensor,weights:list[torch.Tensor],ids:torch.Tensor,start:int,
               include_weights:bool)->torch.Tensor:
    from . import _C,_indexed_float8_expert_available
    metadata(value,weights,ids,start)
    if not _indexed_float8_expert_available:
        raise RuntimeError('indexed FP8 expert API 1 requires matching rebuilt native components')
    return _C.indexed_float8_statistics(value,weights,ids,start,include_weights)


@statistics.register_fake
def _statistics_fake(value,weights,ids,start,include_weights):
    metadata(value,weights,ids,start)
    return torch.empty((len(weights),2,3),device=value.device,dtype=torch.float32)


def product_metadata(value,weights,ids,gradient,scales,start,mode,formats,strategy,needed):
    metadata(value,weights,ids,start)
    if mode not in (0,1,2) or len(formats)!=2 or any(f not in (18,19) for f in formats) or strategy not in (0,1,2):
        raise ValueError('invalid indexed FP8 product mode, formats or strategy')
    if len(needed)!=len(weights) or any(type(flag) is not bool for flag in needed):
        raise ValueError('FP8 expert derivative selection differs')
    if scales.shape!=(len(weights),2) or scales.dtype!=torch.float32 or scales.device!=value.device:
        raise ValueError('FP8 experts require independent [expert,operand] FP32 scales')
    outputs,inputs=weights[0].shape
    if value.shape[1]!=(outputs if mode==1 else inputs) or gradient.device!=value.device:
        raise ValueError('FP8 expert product dimensions differ')
    if mode==2:
        if gradient.shape!=(value.shape[0],outputs) or gradient.dtype not in _FLOATS:
            raise ValueError('FP8 expert weight gradient must match the projected rows')
        return [(tuple(w.shape) if need else (0,),torch.float32) for w,need in zip(weights,needed,strict=True)]
    if gradient.shape!=(0,) or gradient.dtype!=value.dtype:
        raise ValueError('unused FP8 gradient operand must be empty')
    return [((value.shape[0],outputs if mode==0 else inputs),torch.float32)]


@torch.library.custom_op('ruda::indexed_float8_product',mutates_args=())
def product(value:torch.Tensor,weights:list[torch.Tensor],ids:torch.Tensor,gradient:torch.Tensor,
            scales:torch.Tensor,start:int,mode:int,formats:list[int],strategy:int,
            needed:list[bool])->list[torch.Tensor]:
    from . import _C,_indexed_float8_expert_available
    product_metadata(value,weights,ids,gradient,scales,start,mode,formats,strategy,needed)
    if not _indexed_float8_expert_available:
        raise RuntimeError('indexed FP8 expert API 1 requires matching rebuilt native components')
    return _C.indexed_float8_product(value,weights,ids,gradient,scales,start,mode,formats,strategy,needed)


@product.register_fake
def _product_fake(value,weights,ids,gradient,scales,start,mode,formats,strategy,needed):
    return [torch.empty(shape,device=value.device,dtype=dtype) for shape,dtype in
            product_metadata(value,weights,ids,gradient,scales,start,mode,formats,strategy,needed)]


def _format(dtype):
    return 18 if dtype==torch.float8_e4m3fn else 19


def _logical_statistics(observed,group,reduce):
    if group is not None and reduce:
        from ._collective_ops import collective_reduce,group_key
        observed=collective_reduce(observed.contiguous(),1,group_key(group))
    return observed


def _scale(observed,dtype,recipe):
    maximum,invalid=observed[:,0],observed[:,1]
    bound=torch.finfo(dtype).max/(2.**recipe.margin)
    scale=torch.where((invalid==0)&(maximum>0),
                      (maximum/bound).clamp_min(torch.finfo(torch.float32).tiny),torch.ones_like(maximum))
    return _power_of_two_scale(scale) if recipe.power_of_two_scale else scale


def _observe(states,flags,observed,roles):
    for index,(state,enabled) in enumerate(zip(states,flags,strict=True)):
        if enabled:
            for source,role in enumerate(roles):
                state.pending[role].copy_(torch.maximum(state.pending[role],observed[index,source]))


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,ids,start,recipe,states,group,axis,observe,output_dtype,*weights):
        weights=list(weights)
        if recipe.scaling=='block':
            from ._indexed_block_float8 import multiply
            scales=grad_scale=value.new_empty(0,dtype=torch.float32)
            result=multiply(value,weights,ids,value.new_empty(0),start,0,recipe,group,axis,[False]*len(weights))[0]
        elif recipe.scaling=='current':
            stats=statistics(value,weights,ids,start,True)
            sx=_scale(_logical_statistics(stats[:,0],group,axis==1),recipe.input_dtype,recipe)
            sw=_scale(_logical_statistics(stats[:,1],group,axis>=0),recipe.weight_dtype,recipe)
            scales=torch.stack((sx,sw),dim=1)
            grad_scale=value.new_empty(0,dtype=torch.float32)
        elif recipe.scaling=='delayed':
            snapshot=torch.stack([state.scales for state in states]).clone()
            scales=snapshot[:,:2].contiguous();grad_scale=snapshot[:,2].contiguous()
            if any(observe):_observe(states,observe,statistics(value,weights,ids,start,True),(0,1))
        else:
            raise ValueError('unsupported indexed FP8 scaling recipe')
        if recipe.scaling!='block':
            result=product(value,weights,ids,value.new_empty(0),scales,start,0,
                           [_format(recipe.input_dtype),_format(recipe.weight_dtype)],_STRATEGIES[recipe.strategy],
                           [False]*len(weights))[0]
        if group is not None and axis==1:
            from ._collective_ops import collective_reduce,group_key
            result=collective_reduce(result,0,group_key(group))
        ctx.save_for_backward(value,ids,scales,grad_scale,*weights)
        ctx.start,ctx.recipe,ctx.states,ctx.group,ctx.axis=start,recipe,states,group,axis
        ctx.observe=observe
        ctx.set_materialize_grads(False)
        return result.to(output_dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        value,ids,scales,grad_scale,*weights=ctx.saved_tensors
        if gradient is None:return (None,)*(9+len(weights))
        gradient=gradient.contiguous();recipe=ctx.recipe
        if recipe.scaling=='current':
            stats=statistics(gradient,weights,ids,ctx.start,False)[:,0]
            sg=_scale(_logical_statistics(stats,ctx.group,ctx.axis==0),recipe.grad_dtype,recipe)
        elif recipe.scaling=='delayed':
            sg=grad_scale
            if any(ctx.observe):_observe(ctx.states,ctx.observe,statistics(gradient,weights,ids,ctx.start,False)[:,:1],(2,))
        dx=None
        needed=[bool(ctx.needs_input_grad[9+i]) for i in range(len(weights))]
        if ctx.needs_input_grad[0]:
            if recipe.scaling=='block':
                from ._indexed_block_float8 import multiply
                dx=multiply(gradient,weights,ids,gradient.new_empty(0),ctx.start,1,recipe,ctx.group,ctx.axis,[False]*len(weights))[0]
            else:
                dx=product(gradient,weights,ids,gradient.new_empty(0),torch.stack((sg,scales[:,1]),dim=1),ctx.start,1,
                           [_format(recipe.grad_dtype),_format(recipe.weight_dtype)],_STRATEGIES[recipe.strategy],
                           [False]*len(weights))[0]
            if ctx.group is not None and ctx.axis==0:
                from ._collective_ops import collective_reduce,group_key
                dx=collective_reduce(dx,0,group_key(ctx.group))
            dx=dx.to(value.dtype)
        dw=[None]*len(weights)
        if any(needed):
            if recipe.scaling=='block':
                from ._indexed_block_float8 import multiply
                results=multiply(value,weights,ids,gradient,ctx.start,2,recipe,ctx.group,ctx.axis,needed)
            else:
                results=product(value,weights,ids,gradient,torch.stack((sg,scales[:,0]),dim=1),ctx.start,2,
                                [_format(recipe.grad_dtype),_format(recipe.input_dtype)],_STRATEGIES[recipe.strategy],needed)
            dw=[result.to(weight.dtype) if need else None for result,weight,need in zip(results,weights,needed,strict=True)]
        return (dx,)+(None,)*8+tuple(dw)


def linear(value,layers,ids,*,group=None,axis=-1,output_dtype=None):
    from ._indexed_experts import expert_weights
    from .float8_scaling import Float8ScaleState
    if not layers:raise ValueError('FP8 expert bank is empty')
    recipe=getattr(layers[0],'float8_recipe',None) or layers[0].recipe
    if any((getattr(layer,'float8_recipe',None) or getattr(layer,'recipe',None))!=recipe for layer in layers):
        raise ValueError('each indexed FP8 projection bank requires one scaling recipe')
    states=tuple(getattr(layer,'scale_state',None) for layer in layers)
    if recipe.scaling=='delayed' and any(not isinstance(s,Float8ScaleState) or s.recipe!=recipe for s in states):
        raise ValueError('delayed experts require their original matching persistent scale states')
    flags=tuple(layer.training and torch.is_grad_enabled() for layer in layers)
    weights=expert_weights([layer.weight for layer in layers],ids)
    result=None
    dtype=value.dtype if output_dtype is None else output_dtype
    for begin in range(0,len(weights),16):
        part=_Linear.apply(value.contiguous(),ids.contiguous(),begin,recipe,states[begin:begin+16],group,axis,
                           flags[begin:begin+16],dtype,*weights[begin:begin+16])
        result=part if result is None else result+part
    return result


def bank(experts,value,ids,*,names,activation=None):
    from .float8_training import Float8Linear
    from .parallel_training import ColumnParallelLinear,RowParallelLinear
    from ._architecture_ops import precision_context
    layers=[tuple(getattr(expert,name) for name in names) for expert in experts]
    ordinary=all(type(layer) is Float8Linear for trio in layers for layer in trio)
    parallel=all(type(a) is ColumnParallelLinear and type(b) is ColumnParallelLinear and type(c) is RowParallelLinear
                 for a,b,c in layers)
    if not ordinary and not parallel:return None
    if any(layer.bias is not None or hasattr(layer,'_ruda_sequence_axis') for trio in layers for layer in trio):return None
    if any(module._forward_hooks or module._forward_pre_hooks or module._backward_hooks or module._backward_pre_hooks
           for expert in experts for module in expert.modules()):return None
    group=layers[0][0].group if parallel else None
    if parallel:
        if any(layer.group is not group for trio in layers for layer in trio):raise ValueError('FP8 experts require the actual shared TP group')
        if any(a.gather_output or b.gather_output or not c.input_is_parallel
               or getattr(a,'replicas',1)!=1 or getattr(b,'replicas',1)!=1 for a,b,c in layers):return None
    for column in range(3):
        recipe=getattr(layers[0][column],'float8_recipe',None) if parallel else layers[0][column].recipe
        if recipe is None:return None
        if any((getattr(trio[column],'float8_recipe',None) if parallel else trio[column].recipe)!=recipe
               or trio[column].weight.shape!=layers[0][column].weight.shape for trio in layers):return None
    output_dtype=torch.get_autocast_dtype(value.device.type) if torch.is_autocast_enabled(value.device.type) else value.dtype
    def project(rows,column):
        return linear(rows,[trio[column] for trio in layers],ids,group=group,
                      axis=(1 if column==2 else 0) if parallel else -1,output_dtype=output_dtype)
    act=torch.nn.functional.silu if activation is None else activation
    with precision_context(value):
        return project(act(project(value,0))*project(value,1),2)
