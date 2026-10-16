"""Fixed-address native graphs for multi-output attention and quantized kernels."""
from collections import OrderedDict
import math
import operator
import sys
from threading import RLock

import torch
from torch.fx import Node
from torch.fx.node import map_arg
from ._compile_pointwise import definition as pointwise_definition,describe as pointwise_description
from ._compile_transforms import definition as transform_definition,describe as transform_description
from ._compile_reductions import definition as reduction_definition,describe as reduction_description
from ._compile_scatter import definition as scatter_definition,describe as scatter_description
from ._compile_sort import definition as sort_definition,describe as sort_description,output_layout as sort_output_layout


_OPERATIONS={
    'ruda::dense_attention_forward':(0,9),'ruda::dense_attention_backward':(1,13),
    'ruda::indexed_dense_attention_forward':(35,13),'ruda::indexed_dense_attention_backward':(36,15),
    'ruda::attention_dropout_mask':(37,6),
    'ruda::varlen_attention_forward':(2,13),'ruda::varlen_attention_backward':(3,14),
    'ruda::float8_mm':(4,7),'ruda::float8_quantize':(5,3),
    'ruda::packed_linear':(6,7),'ruda::packed_linear_tensor_core':(6,7),
    'ruda::packed_decode':(7,6),
    'ruda::float8_linear_forward':(8,9),'ruda::float8_current_quantize':(9,4),
    'ruda::kda_forward':(10,9),'ruda::kda_backward':(11,11),
    'ruda::delta_forward':(12,8),'ruda::triangular_solve':(13,4),
    'aten::cumsum':(14,3),'aten::flip':(15,2),
    'aten::cat':(16,2),
    'ruda::router_weights_forward':(21,5),'ruda::router_weights_backward':(22,6),
    'ruda::grouped_gemm':(23,4),'ruda::grouped_gemm_backward':(24,4),
    'ruda::float8_block_quantize':(27,6),'ruda::float8_block_mm':(28,8),
    'ruda::float8_delayed_quantize':(29,4),
    'ruda::float8_delayed_advance':(30,10),
    'ruda::indexed_expert_linear':(31,4),'ruda::indexed_expert_backward':(32,5),
    'ruda::indexed_expert_backward_selected':(38,7),'ruda::indexed_expert_activity':(39,3),
    'ruda::indexed_quantized_expert_linear':(40,6),
    'ruda::indexed_float8_statistics':(41,5),'ruda::indexed_float8_product':(42,10),
    'ruda::indexed_block_float8_statistics':(43,7),'ruda::indexed_block_float8_product':(44,13),
    'ruda::float8_shard_statistics':(33,7),'ruda::float8_shard_quantize':(34,11),
}
_CPU_POSITIONS={0:(),1:(6,),2:(3,4,12),3:(6,),4:(),5:(),6:(),7:(),8:(),9:(),10:(),11:(),12:(),13:(),14:(),15:(),16:(),17:(),18:(),19:(),20:(),21:(),22:(),23:(),24:(),25:(),26:()}
_CPU_POSITIONS.update({27:(),28:(),29:(),30:(),31:(),32:(),33:(),34:()})
_CPU_POSITIONS.update({35:(12,),36:(6,)})
_CPU_POSITIONS[37]=(1,)
_CPU_POSITIONS.update({38:(),39:(),40:(),41:(),42:(),43:(),44:()})


def available(kind=None):
    if kind in (43,44):return (getattr(sys.modules.get(__package__),'_operation_graph_available',False)
                              and getattr(sys.modules.get(__package__),'_indexed_block_float8_expert_available',False))
    if kind in (41,42):return (getattr(sys.modules.get(__package__),'_operation_graph_available',False)
                              and getattr(sys.modules.get(__package__),'_indexed_float8_expert_available',False))
    if kind==40:return (getattr(sys.modules.get(__package__),'_operation_graph_available',False)
                       and getattr(sys.modules.get(__package__),'_indexed_quantized_expert_available',False))
    if kind in (38,39):return (getattr(sys.modules.get(__package__),'_operation_graph_available',False)
                              and getattr(sys.modules.get(__package__),'_indexed_expert_training_available',False))
    if kind==37:return getattr(sys.modules.get(__package__),'_attention_dropout_graph_available',False)
    if kind in (35,36):return getattr(sys.modules.get(__package__),'_indexed_attention_graph_available',False)
    if kind in (33,34):return getattr(sys.modules.get(__package__),'_shard_float8_graph_available',False)
    if kind in (31,32):return getattr(sys.modules.get(__package__),'_indexed_expert_graph_available',False)
    if kind in (29,30):return getattr(sys.modules.get(__package__),'_delayed_float8_graph_available',False)
    if kind in (27,28):return getattr(sys.modules.get(__package__),'_block_float8_graph_available',False)
    if kind==26:return getattr(sys.modules.get(__package__),'_sorting_graph_available',False)
    if kind==25:return getattr(sys.modules.get(__package__),'_scatter_graph_available',False)
    if kind in (23,24):return getattr(sys.modules.get(__package__),'_grouped_graph_available',False)
    if kind in (21,22):return getattr(sys.modules.get(__package__),'_router_graph_available',False)
    if kind==20:return getattr(sys.modules.get(__package__),'_reduction_graph_available',False)
    name='_transform_graph_available' if kind==19 else '_pointwise_graph_available' if kind in (17,18) else '_sequence_graph_available' if kind is not None and kind>=10 else '_operation_graph_available'
    return getattr(sys.modules.get(__package__),name,False)


def arguments(node):
    ordering=sort_definition(node)
    if ordering is not None:return ordering[1]
    scatter=scatter_definition(node)
    if scatter is not None:return scatter[1]
    reduction=reduction_definition(node)
    if reduction is not None:return reduction[1]
    transform=transform_definition(node)
    if transform is not None:return transform[1]
    pointwise=pointwise_definition(node)
    if pointwise is not None:return pointwise[1]
    if node.target is torch.ops.aten.cumsum.default:
        if len(node.args)==2 and set(node.kwargs)<= {'dtype'}:return (*node.args,node.kwargs.get('dtype'))
        return None
    if node.target is torch.ops.aten.cat.default:
        if len(node.args)==1 and set(node.kwargs)<= {'dim'}:return (node.args[0],node.kwargs.get('dim',0))
        if len(node.args)==2 and not node.kwargs:return node.args
        return None
    return node.args if not node.kwargs else None


def operation(node):
    if node.op!='call_function' or not isinstance(node.target,torch._ops.OpOverload):return None
    ordering=sort_definition(node)
    if ordering is not None:return ordering[0]
    scatter=scatter_definition(node)
    if scatter is not None:return scatter[0]
    reduction=reduction_definition(node)
    if reduction is not None:return reduction[0]
    transform=transform_definition(node)
    if transform is not None:return transform[0]
    pointwise=pointwise_definition(node)
    if pointwise is not None:return pointwise[0]
    schema=node.target._schema
    spec=_OPERATIONS.get(schema.name)
    args=arguments(node)
    if spec is None or schema.overload_name or args is None or len(args)!=spec[1]:return None
    if spec[0]==16 and isinstance(args[0],(tuple,list)) and len(args[0])==2:return None
    return spec[0]


def tuple_metadata(node,operations):
    if node.op!='call_function' or node.target is not operator.getitem or len(node.args)!=2 or node.kwargs:return False
    source,index=node.args
    if not isinstance(source,Node) or (source not in operations and source.op!='placeholder') or type(index) is not int:return False
    val=source.meta.get('val')
    return isinstance(val,(tuple,list)) and -len(val)<=index<len(val) and isinstance(node.meta.get('val'),torch.Tensor)


def host_metadata_nodes(gm):
    """CPU inputs are allowed only in the exact boundary/RNG schema positions."""
    copies={torch.ops.aten.clone.default,torch.ops.aten.lift_fresh_copy.default,
            torch.ops.aten.detach.default,torch.ops.aten.alias.default}
    def allowed(node,seen):
        if node in seen:return False
        seen=seen|{node}
        value=node.meta.get('val')
        if not isinstance(value,torch.Tensor) or value.device.type!='cpu' or value.dtype not in (torch.int32,torch.int64) or value.ndim!=1 or not node.users:return False
        for user in node.users:
            if user.op=='call_function' and user.target in copies and len(user.args)==1 and user.args[0] is node:
                if not allowed(user,seen):return False
                continue
            kind=operation(user)
            positions=[i for i,arg in enumerate(user.args) if arg is node]
            if kind is None:
                from ._compile_collectives import definition
                spec=definition(user)
                allowed_positions=({'ruda::ring_attention_forward':(16,), 'ruda::indexed_ring_attention_forward':(16,),
                                    'ruda::ring_attention_backward':(8,), 'ruda::attention_rng_state':(4,)}.get(spec[0],())
                                   if spec is not None else ())
            else:allowed_positions=_CPU_POSITIONS[kind]
            if not positions or any(i not in allowed_positions for i in positions):return False
        return True
    return {node for node in gm.graph.nodes if allowed(node,set())}


def _shape(value):return tuple(value.shape)


def _describe(kind,args,tensor_core):
    if kind in (43,44):
        from ._indexed_block_float8 import describe,product_metadata
        if kind==43:
            value,weights,ids,gradient,start,mode,geometry=args
            specs=describe(value,weights,ids,gradient,start,mode,geometry)
            return [value,ids,*weights,gradient],specs,(start,len(weights),mode,*geometry),(),(),{}
        value,weights,ids,gradient,stats,start,mode,geometry,formats,margin,power,strategy,needed=args
        specs=product_metadata(value,weights,ids,gradient,stats,start,mode,geometry,formats,margin,power,strategy,needed)
        integers=(start,len(weights),mode,*geometry,*formats,margin,int(power),strategy,sum(int(flag)<<i for i,flag in enumerate(needed)))
        return [value,ids,*weights,gradient,*stats],specs,integers,(),(),{}
    if kind in (41,42):
        from ._indexed_float8_experts import metadata,product_metadata
        if kind==41:
            value,weights,ids,start,include_weights=args
            metadata(value,weights,ids,start)
            return [value,ids,*weights],[((len(weights),2,3),torch.float32)],(start,len(weights),int(include_weights)),(),(),{}
        value,weights,ids,gradient,scales,start,mode,formats,strategy,needed=args
        specs=product_metadata(value,weights,ids,gradient,scales,start,mode,formats,strategy,needed)
        integers=(start,len(weights),mode,*formats,strategy,sum(int(flag)<<i for i,flag in enumerate(needed)))
        return [value,ids,*weights,gradient,scales],specs,integers,(),(),{}
    if kind==40:
        from ._indexed_quantized_experts import metadata
        value,buffers,ids,start,layouts,transpose=args
        width=metadata(value,buffers,ids,start,layouts,transpose)
        return [value,ids,*buffers],[((value.shape[0],width),value.dtype)],(start,len(layouts)//6,int(transpose),*layouts),(),(),{}
    if kind==39:
        ids,start,count=args
        if ids.device.type!='ruda' or ids.dtype not in (torch.int32,torch.int64) or ids.ndim!=1:
            raise ValueError('expert activity requires a RUDA integer ID vector')
        if not 0<=start<=0xffffffff-count or not 1<=count<=0xffffffff:
            raise ValueError('expert activity range exceeds uint32')
        return [ids],[((count,),torch.int32)],(start,count),(),(),{}
    if kind==37:
        anchor,rng,shape,global_shape,offsets,p=args
        if len(shape)!=4 or len(global_shape)!=4 or len(offsets)!=3 or anchor.numel()!=0:
            raise ValueError('native dropout tile geometry differs')
        if anchor.device.type!='ruda' or anchor.dtype not in (torch.float16,torch.bfloat16,torch.float32):
            raise ValueError('native dropout tiles require floating RUDA storage')
        if not 0<=p<=1 or shape[0]!=global_shape[0] or any(n<0 for n in (*shape,*global_shape,*offsets)):
            raise ValueError('invalid native dropout tile policy')
        if any(start+local>total for start,local,total in zip(offsets,shape[1:],global_shape[1:])) or math.prod(global_shape)>=1<<63:
            raise ValueError('native dropout tile exceeds global RNG geometry')
        return [anchor],[(tuple(shape),anchor.dtype)],(*global_shape,*offsets),(p,),(),{'rng':rng}
    if kind in (35,36):
        count=9 if kind==35 else 13
        inputs,outputs,integers,scalars,score_shape,extra=_describe(kind-35,args[:count],tensor_core)
        global_heads,head_start=args[count:count+2]
        q,k=args[:2]
        if global_heads<q.shape[1] or not 0<=head_start<=global_heads-q.shape[1]:
            raise ValueError('native indexed attention head interval differs')
        shape=(q.shape[0],global_heads,q.shape[2],k.shape[2])
        if math.prod(shape)>=1<<63:raise ValueError('native indexed attention RNG positions exceed int64')
        if kind==35:extra.update(shape=shape,groups=args[11],requested_rng=args[12])
        return inputs,outputs,(*integers,global_heads,head_start),scalars,score_shape,extra
    if kind==16 and (not isinstance(args[0],(tuple,list)) or not args[0]):raise ValueError('cat requires a nonempty tensor list')
    q=args[0][0] if kind==16 else args[0];extra={};score_shape=()
    if kind in (33,34):
        from .float8_scaling import _shard_geometry
        if kind==33:
            value,*geometry=args
            dtype,margin,power=None,0,False
        else:
            value,statistics,dtype,*geometry,margin,power=args
        if any(type(n) is not int or not 0<=n<=0xffffffff for n in geometry):
            raise ValueError('invalid sharded FP8 graph geometry')
        rows,cols,origin_row,origin_col,global_rows,global_cols=geometry
        shape=_shard_geometry(value,*geometry)
        if rows*cols>0xffffffff-32 or math.prod(shape)>0xffffffff:
            raise ValueError('sharded FP8 graph indexing overflow')
        if ((global_rows+rows-1)//rows)*rows>0xffffffff or ((global_cols+cols-1)//cols)*cols>0xffffffff:
            raise ValueError('sharded FP8 graph block coordinates overflow')
        scalars=()
        if kind==33:
            inputs=[value];integers=tuple(geometry);outputs=[(shape,torch.float32)]
        else:
            if statistics.dtype!=torch.float32 or _shape(statistics)!=shape:
                raise ValueError('sharded FP8 graph statistics geometry differs')
            if dtype not in (torch.float8_e4m3fn,torch.float8_e5m2) or type(margin) is not int or not 0<=margin<=16 or type(power) is not bool:
                raise ValueError('invalid sharded FP8 graph scaling policy')
            height,width=value.shape[0]+origin_row%rows,value.shape[1]+origin_col%cols
            inputs=[value,statistics];integers=(*geometry,margin,int(power))
            outputs=[((height,width),dtype),(((height+rows-1)//rows,(width+cols-1)//cols),torch.float32)]
    elif kind in (31,32,38):
        from ._indexed_experts import metadata
        value,weights,ids=args[:3]
        start=args[4] if kind==38 else args[-1];gradient=args[3] if kind in (32,38) else None
        metadata(value,weights,ids,start,gradient)
        inputs=[value,ids,*weights];integers=(start,len(weights));scalars=()
        if kind in (32,38):
            inputs.append(gradient)
            needed=[True]*(len(weights)+1) if kind==32 else [args[5],*args[6]]
            if len(needed)!=len(weights)+1 or any(type(flag) is not bool for flag in needed):
                raise ValueError('indexed expert derivative selection differs')
            outputs=[(_shape(t) if need else (0,),t.dtype) for t,need in zip((value,*weights),needed,strict=True)]
            if kind==38:integers=(*integers,int(needed[0]),sum(int(flag)<<index for index,flag in enumerate(needed[1:])))
        else:outputs=[((value.shape[0],weights[0].shape[0]),value.dtype)]
    elif kind==30:
        scale,history,observed,step,interval,maximum,power,*limits=args
        if (scale.dtype!=torch.float32 or scale.shape!=(3,) or history.dtype!=torch.float32 or history.ndim!=3
                or history.shape[0]!=3 or history.shape[2]!=2 or not 1<=history.shape[1]<=0xffffffff
                or observed.dtype!=torch.float32 or observed.shape!=(3,3) or step.dtype!=torch.int64 or step.shape!=()):
            raise ValueError('invalid delayed FP8 graph history/state geometry')
        if type(interval) is not int or not 1<=interval<=0x7fffffffffffffff or type(maximum) is not bool or type(power) is not bool:
            raise ValueError('invalid delayed FP8 graph update policy')
        if any(type(n) not in (int,float) or not math.isfinite(n) or n<=0 for n in limits):
            raise ValueError('invalid delayed FP8 graph format maxima')
        inputs=[scale,history,observed,step];integers=(interval,int(maximum),int(power));scalars=tuple(limits)
        outputs=[(_shape(t),t.dtype) for t in inputs]
    elif kind==29:
        value,scale,pending,dtype=args
        if value.dtype not in (torch.float32,torch.float16,torch.bfloat16) or dtype not in (torch.float8_e4m3fn,torch.float8_e5m2):
            raise ValueError('invalid delayed FP8 graph storage dtype')
        if scale.dtype!=torch.float32 or scale.shape!=() or pending.dtype!=torch.float32 or pending.shape!=(3,):
            raise ValueError('delayed FP8 graph requires scalar scale and [amax,invalid,seen] state')
        inputs=[value,scale,pending];integers=(12 if dtype==torch.float8_e4m3fn else 13,);scalars=()
        outputs=[(_shape(value),dtype),((),torch.float32),((3,),torch.float32)]
    elif kind==27:
        value,dtype,rows,cols,margin,power=args
        if value.ndim!=2 or value.dtype not in (torch.float32,torch.float16,torch.bfloat16):raise ValueError('block FP8 graph needs floating matrix storage')
        if dtype not in (torch.float8_e4m3fn,torch.float8_e5m2) or type(rows) is not int or rows<1 or type(cols) is not int or cols<32 or cols%32:
            raise ValueError('invalid block FP8 graph encoding geometry')
        if type(margin) is not int or not 0<=margin<=16 or type(power) is not bool:raise ValueError('invalid block FP8 scaling policy')
        inputs=[value];integers=(12 if dtype==torch.float8_e4m3fn else 13,rows,cols,margin,int(power));scalars=()
        outputs=[(_shape(value),dtype),(((value.shape[0]+rows-1)//rows,(value.shape[1]+cols-1)//cols),torch.float32)]
    elif kind==28:
        a,w,sa,sw,rows_a,rows_w,block_k,strategy=args
        if a.ndim!=2 or w.ndim!=2 or a.shape[1]!=w.shape[1] or any(t.dtype not in (torch.float8_e4m3fn,torch.float8_e5m2) for t in (a,w)):
            raise ValueError('invalid block FP8 graph matrix geometry')
        if any(type(size) is not int or size<1 for size in (rows_a,rows_w,block_k)) or block_k%32 or strategy not in (0,1,2):
            raise ValueError('invalid block FP8 graph block size or strategy')
        for scale,value,rows in ((sa,a,rows_a),(sw,w,rows_w)):
            if scale.dtype!=torch.float32 or scale.shape!=((value.shape[0]+rows-1)//rows,(value.shape[1]+block_k-1)//block_k):
                raise ValueError('block FP8 graph scale geometry differs')
        inputs=[a,w,sa,sw];integers=(rows_a,rows_w,block_k,strategy);scalars=()
        outputs=[((a.shape[0],w.shape[0]),torch.float32)]
    elif kind==26:
        inputs,outputs,integers,scalars,score_shape,extra=sort_description(args)
    elif kind==25:
        inputs,outputs,integers,scalars,score_shape,extra=scatter_description(args)
    elif kind in (23,24):
        from ._grouped_ops import metadata
        metadata(*args[:3],None if kind==23 else args[3]);scalars=()
        if kind==23:
            if args[3] not in (0,1,2):raise ValueError('invalid grouped GEMM strategy')
            inputs=list(args[:3]);integers=(args[3],);outputs=[((q.shape[0],args[1].shape[1]),q.dtype)]
        else:inputs=list(args);integers=();outputs=[(_shape(t),t.dtype) for t in args[:2]]
    elif kind in (21,22):
        from ._router_ops import metadata
        count=2 if kind==21 else 3;inputs=list(args[:count]);scoring,renormalize,scale=args[count:]
        metadata(args[0],args[1],scoring,renormalize,scale,None if kind==21 else args[2])
        integers=(scoring,int(renormalize));scalars=(scale,)
        outputs=[(_shape(args[1]),torch.float32)] if kind==21 else [(_shape(q),q.dtype)]
    elif kind==20:
        inputs,outputs,integers,scalars,score_shape,extra=reduction_description(args)
    elif kind==19:
        inputs,outputs,integers,scalars,score_shape,extra=transform_description(args)
    elif kind>=17:
        inputs,outputs,integers,scalars,score_shape,extra=pointwise_description(kind,args)
    elif kind<=3:
        if q.numel()==0:raise ValueError('native attention graph requires nonempty query rows')
        if kind<=1:
            index=4 if kind==0 else 7
            score_shape,mask_kind,causal,scale,p=args[index:index+5]
            score_shape=tuple(score_shape)
            integers=(mask_kind,int(causal),0 if kind==0 else int(args[12]));scalars=(scale,p)
            inputs=list(args[:4] if kind==0 else args[:6])
            if kind==0:
                outputs=[((*q.shape[:-1],args[2].shape[-1]),q.dtype),((q.shape[0]*q.shape[1]*q.shape[2],2),torch.float32)]
                extra={'scores':math.prod(score_shape),'p':p}
            else:
                outputs=[(_shape(t),t.dtype) for t in args[:3]]
                outputs.append((_shape(args[3]) if args[12] else (1,),torch.float32));extra={'rng':args[6]}
        else:
            index=5 if kind==2 else 7
            causal,lower,left,right,scale,p,softcap=args[index:index+7]
            integers=(int(causal),int(lower),left,right);scalars=(scale,p,softcap)
            if kind==2:
                from ._varlen_ops import _schedule
                records,blocks=_schedule(q,args[1],args[3],args[4])
                inputs=[*args[:3],torch.tensor(records,dtype=torch.int64,device='cpu')]
                outputs=[((q.shape[0],q.shape[1],args[2].shape[-1]),q.dtype),((q.shape[0]*q.shape[1],2),torch.float32)]
                extra={'requested_rng':args[12],'blocks':blocks,'p':p}
            else:
                inputs=list(args[:6]);outputs=[(_shape(t),t.dtype) for t in args[:3]];extra={'rng':args[6]}
    elif kind==4:
        a,b,sa,sb,ta,tb,strategy=args;inputs=[a,b,sa,sb];integers=(int(ta),int(tb),strategy);scalars=()
        outputs=[((a.shape[1 if ta else 0],b.shape[0 if tb else 1]),torch.float32)]
    elif kind==5:
        inputs=list(args[:2]);outputs=[(_shape(q),args[2])];integers=();scalars=()
    elif kind==6:
        layout,transpose=args[5:];inputs=list(args[:5]);integers=(*layout,int(transpose),int(tensor_core));scalars=()
        outputs=[((q.shape[0],layout[2] if transpose else layout[3]),q.dtype)]
    elif kind==7:
        layout,dtype=args[4:];inputs=list(args[:4]);integers=tuple(layout);scalars=();outputs=[((layout[3],layout[2]),dtype)]
    elif kind==8:
        x,w,xf,wf,gf,output_dtype,strategy,margin,power=args;inputs=[x,w];scalars=()
        code={torch.float8_e4m3fn:12,torch.float8_e5m2:13}
        integers=(code[xf],code[wf],strategy,margin,int(power))
        outputs=[((x.shape[0],w.shape[0]),output_dtype),(_shape(x),xf),(_shape(w),wf),((),torch.float32),((),torch.float32)]
    elif kind==9:
        x,dtype,margin,power=args;inputs=[x];scalars=()
        integers=({torch.float8_e4m3fn:12,torch.float8_e5m2:13}[dtype],margin,int(power))
        outputs=[(_shape(x),dtype),((),torch.float32)]
    elif kind in (10,11,12):
        if q.ndim!=4 or q.shape[0]==0:raise ValueError('native sequence graph requires [B,H,T,D] with nonempty batch')
        count=9 if kind==11 else 6
        inputs=list(args[:count]);scale,chunk=args[count:count+2]
        if type(chunk) is not int or chunk<=0 or chunk>2**32-1:raise ValueError('invalid sequence graph chunk size')
        integers=(chunk,);scalars=(scale,)
        if kind==11:outputs=[(_shape(t),t.dtype) for t in args[:6]]
        else:
            outputs=[(_shape(args[2]),args[2].dtype),(_shape(args[5]),args[5].dtype)]
            if kind==10:
                retain=args[8];integers=(chunk,int(retain))
                boundaries=(q.shape[2]+chunk-1)//chunk if retain else 0
                outputs.append(((boundaries,*args[5].shape),torch.float32))
    elif kind==13:
        a,b,upper,unit=args;inputs=[a,b];integers=(int(upper),int(unit));scalars=()
        outputs=[(_shape(b),b.dtype)]
    elif kind==14:
        x,axis,dtype=args;rank=max(x.ndim,1)
        if type(axis) is not int or not -rank<=axis<rank:raise ValueError('invalid cumsum dimension')
        dtype=dtype or (torch.int64 if x.dtype in (torch.bool,torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64) else x.dtype)
        supported=(torch.float32,torch.float16,torch.bfloat16,torch.float64,torch.int64,torch.int32,torch.int16,torch.int8,torch.uint8)
        if dtype not in supported or x.dtype not in (*supported,torch.bool):raise ValueError('unsupported native cumsum dtype')
        inputs=[x];integers=(axis%rank,);scalars=();outputs=[(_shape(x),dtype)]
    elif kind==15:
        x,axes=args;rank=max(x.ndim,1)
        if rank>32 or not isinstance(axes,(tuple,list)) or any(type(a) is not int or not -rank<=a<rank for a in axes):raise ValueError('invalid flip dimensions')
        axes=tuple(a%rank for a in axes)
        if len(set(axes))!=len(axes):raise ValueError('flip dimensions must be unique')
        if x.dtype not in (torch.bool,torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64,
                           torch.float16,torch.bfloat16,torch.float32,torch.float64,torch.float8_e4m3fn,torch.float8_e5m2):raise ValueError('unsupported native flip dtype')
        inputs=[x];integers=(sum(1<<a for a in axes),);scalars=();outputs=[(_shape(x),x.dtype)]
    else:
        tensors,axis=args
        if not isinstance(tensors,(tuple,list)) or not tensors or any(not isinstance(t,torch.Tensor) or t.ndim==0 for t in tensors):raise ValueError('cat requires a nonempty tensor list with rank >= 1')
        reference=next((t for t in tensors if t.shape!=(0,)),tensors[0]);rank=reference.ndim
        if type(axis) is not int or not -rank<=axis<rank:raise ValueError('invalid cat dimension')
        axis%=rank;shape=list(reference.shape);shape[axis]=0
        for t in tensors:
            if t.dtype!=reference.dtype:raise ValueError('native cat requires explicitly promoted input dtypes')
            if t.shape==(0,):continue
            if t.ndim!=rank or any(t.shape[d]!=reference.shape[d] for d in range(rank) if d!=axis):raise ValueError('cat input shapes differ')
            shape[axis]+=t.shape[axis]
        if reference.dtype not in (torch.bool,torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64,
                torch.float16,torch.bfloat16,torch.float32,torch.float64,torch.float8_e4m3fn,torch.float8_e5m2):raise ValueError('unsupported native cat dtype')
        inputs=list(tensors);integers=(axis,);scalars=();outputs=[(tuple(shape),reference.dtype)]
    if 4<=kind<=8 and math.prod(outputs[0][0])==0:raise ValueError('empty quantized operation graph output')
    if any(not isinstance(t,torch.Tensor) for t in inputs):raise ValueError('operation graph inputs must be tensors')
    if q.device.type!='ruda' or q.device.index not in (None,0):raise ValueError('operation graph requires ruda:0')
    for i,t in enumerate(inputs):
        if kind==2 and i==3:continue
        if t.device!=q.device or t.layout!=torch.strided or t.is_conj() or t.is_neg():raise ValueError('operation graph input device/layout differs')
    return inputs,outputs,tuple(integers),tuple(scalars),score_shape,extra


def _state(kind,extra):
    from ._dense_ops import _rng_state,rng_words
    if kind==35:
        from ._parallel_rng import attention_state
        state=attention_state(extra['shape'],extra['p'],extra['groups'],extra['requested_rng'])
    elif kind==0:state=_rng_state(extra['scores'],extra['p'])
    elif kind==2:
        from ._varlen_ops import reserve
        requested=extra['requested_rng']
        if requested.device.type!='cpu' or requested.dtype!=torch.int64 or requested.shape!=(5,):raise ValueError('varlen RNG request must have five CPU int64 words')
        words=requested.tolist()
        words=reserve(extra['blocks']) if 0<extra['p']<1 and words[0]<0 else words[1:]
        state=torch.tensor(words,dtype=torch.int64,device='cpu')
    elif kind in (1,3,36,37):state=extra['rng']
    else:return 0,0,None
    seed,counter=rng_words(state)
    return seed,counter,state


def _weight_bindings(kind,inputs,integers):
    if kind in (31,32,38,41,42,43,44):
        positions=range(2,2+integers[1])
    elif kind==40:
        positions=range(2,len(inputs))
    else:
        return {}
    return {i:inputs[i] for i in positions
            if inputs[i].is_contiguous() and not inputs[i].is_conj() and not inputs[i].is_neg()}


class OperationRegion(torch.nn.Module):
    def __init__(self,node,state,phase,options):
        super().__init__();self.target=node.target;self.kind=operation(node);self.template=arguments(node)
        self.original_args=node.args;self.original_kwargs=node.kwargs
        self.tensor_core=node.target._schema.name=='ruda::packed_linear_tensor_core'
        dependencies=[]
        def dependency(n):
            if n not in dependencies:dependencies.append(n)
            return n
        map_arg(self.template,dependency)
        self.dependencies=tuple(dependencies);self._indices={n:i for i,n in enumerate(dependencies)}
        self.state=state;self.phase=phase;self.options=options
        self._cache=OrderedDict();self._lock=RLock();self._closed=False

    def forward(self,*values):
        from ._compile_native import NativeCoverageError,_stream_id
        with self._lock:
            self.state.check_open()
            if self._closed:raise RuntimeError('native operation region closed')
            args=map_arg(self.template,lambda n:values[self._indices[n]])
            reason=None
            if not available(self.kind):reason='native indexed attention/dropout graph API 1 unavailable' if self.kind in (35,36,37) else 'native sharded FP8 graph API 1 unavailable' if self.kind in (33,34) else 'native delayed FP8 graph API 1 unavailable' if self.kind in (29,30) else 'native block FP8 graph API 1 unavailable' if self.kind in (27,28) else 'native sorting graph API 1 unavailable' if self.kind==26 else 'native scatter graph API 1 unavailable' if self.kind==25 else 'native grouped GEMM graph API 1 unavailable' if self.kind in (23,24) else 'native router graph API 1 unavailable' if self.kind in (21,22) else 'native reduction graph API 1 unavailable' if self.kind==20 else 'native transform graph API 1 unavailable' if self.kind==19 else 'native pointwise graph API 1 unavailable' if self.kind>=17 else 'native sequence graph API 1 unavailable' if self.kind>=10 else 'native operation graph API 1 unavailable'
            else:
                try:inputs,specs,integers,scalars,score_shape,extra=_describe(self.kind,args,self.tensor_core)
                except (ValueError,TypeError,OverflowError) as exc:reason=str(exc)
            if self.kind in (38,39) and not available(self.kind):reason='native indexed expert training API 1 unavailable'
            if self.kind==40 and not available(self.kind):reason='native indexed quantized expert API 1 unavailable'
            if self.kind in (41,42) and not available(self.kind):reason='native indexed FP8 expert API 1 unavailable'
            if self.kind in (43,44) and not available(self.kind):reason='native indexed block FP8 expert API 1 unavailable'
            if reason is not None:
                if self.options.native=='required':raise NativeCoverageError(f'{self.phase}: {reason}')
                self.state.add('region_reference_runs');self.state.reason(self.phase,reason)
                resolve=lambda n:values[self._indices[n]]
                result=self.target(*map_arg(self.original_args,resolve),**map_arg(self.original_kwargs,resolve))
                if self.kind==26:
                    # An internal input view may have been materialized by an
                    # earlier region. Preserve sort's captured output strides.
                    owned=sort_output_layout((result,) if args[1]==2 else result,args[-1],copy=False)
                    return owned[0] if args[1]==2 else owned
                return result
            from . import _C
            key=(_stream_id(),torch.is_inference_mode_enabled(),
                 tuple((_shape(t),t.dtype,t.device) for t in inputs),tuple(specs),integers,scalars,score_shape)
            if self.kind==26:key=(*key,extra['output_strides'])
            borrowed=_weight_bindings(self.kind,inputs,integers)
            if borrowed:
                key=(*key,tuple((i,t.untyped_storage()._cdata,t.data_ptr(),t.storage_offset(),tuple(t.stride()))
                               for i,t in borrowed.items()))
            with torch.no_grad():
                entry=self._cache.get(key)
                if entry is None:
                    reusable={}
                    for previous_key,(_,previous_buffers,_) in self._cache.items():
                        if previous_key[:2]!=key[:2]:continue
                        for i,(source,buffer) in enumerate(zip(inputs,previous_buffers,strict=True)):
                            if self.kind in (31,32,38,40,41,42,43,44) and i>=2 and (self.kind==40 or i<2+integers[1]):continue
                            if _shape(source)==_shape(buffer) and source.dtype==buffer.dtype:
                                reusable[i]=buffer
                    if len(self._cache)>=self.options.cache_size:
                        oldest=next(iter(self._cache));self._cache[oldest][0].close();del self._cache[oldest]
                        self.state.add('native_cache_evictions')
                    device=inputs[0].device
                    buffers=[borrowed[i].detach() if i in borrowed else reusable[i] if i in reusable
                             else torch.empty(t.shape,dtype=t.dtype,device=device) for i,t in enumerate(inputs)]
                    outputs=([torch.empty_strided(shape,stride,dtype=dtype,device=device)
                              for (shape,dtype),stride in zip(specs,extra['output_strides'],strict=True)] if self.kind==26 else
                             [torch.empty(shape,dtype=dtype,device=device) for shape,dtype in specs])
                    graph=_C.NativeOperationGraph(self.kind,[*buffers,*outputs],list(integers),list(scalars),list(score_shape))
                    entry=graph,buffers,outputs;self._cache[key]=entry;self.state.add('native_builds')
                else:self._cache.move_to_end(key);self.state.add('native_cache_hits')
                graph,buffers,outputs=entry
                for i,(dst,src) in enumerate(zip(buffers,inputs,strict=True)):
                    if i not in borrowed:dst.copy_(src)
                seed,counter,rng=_state(self.kind,extra)
                graph.run(seed,counter)
                owned=sort_output_layout(outputs,extra['output_strides'],copy=True) if self.kind==26 else tuple(t.clone() for t in outputs)
                self.state.add('native_replays');self.state.add('native_nodes_executed',graph.kernel_count)
                if self.kind in (0,35):return (*owned,rng)
                if self.kind==2:return (*owned,buffers[3].clone(),rng)
                if self.kind in (32,38,42,44):return list(owned)
                if self.kind==43:return owned
                return owned if self.kind in (1,3,8,9,10,11,12,24,27,29,30,34,36) or self.kind in (20,26) and len(owned)==2 else owned[0]

    def close(self):
        with self._lock:
            if self._closed:return
            first=None
            for key,(graph,_,_) in list(self._cache.items()):
                try:graph.close()
                except Exception as exc:first=first or exc
                else:del self._cache[key]
            if first is not None:raise first
            self._closed=True
