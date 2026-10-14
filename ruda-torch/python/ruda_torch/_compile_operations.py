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
}
_CPU_POSITIONS={0:(),1:(6,),2:(3,4,12),3:(6,),4:(),5:(),6:(),7:(),8:(),9:(),10:(),11:(),12:(),13:(),14:(),15:(),16:(),17:(),18:(),19:(),20:(),21:(),22:(),23:(),24:(),25:(),26:()}


def available(kind=None):
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
    if not isinstance(source,Node) or source not in operations or type(index) is not int:return False
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
            if kind is None:return False
            positions=[i for i,arg in enumerate(user.args) if arg is node]
            if not positions or any(i not in _CPU_POSITIONS[kind] for i in positions):return False
        return True
    return {node for node in gm.graph.nodes if allowed(node,set())}


def _shape(value):return tuple(value.shape)


def _describe(kind,args,tensor_core):
    if kind==16 and (not isinstance(args[0],(tuple,list)) or not args[0]):raise ValueError('cat requires a nonempty tensor list')
    q=args[0][0] if kind==16 else args[0];extra={};score_shape=()
    if kind==26:
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
    if kind==0:state=_rng_state(extra['scores'],extra['p'])
    elif kind==2:
        from ._varlen_ops import reserve
        requested=extra['requested_rng']
        if requested.device.type!='cpu' or requested.dtype!=torch.int64 or requested.shape!=(5,):raise ValueError('varlen RNG request must have five CPU int64 words')
        words=requested.tolist()
        words=reserve(extra['blocks']) if 0<extra['p']<1 and words[0]<0 else words[1:]
        state=torch.tensor(words,dtype=torch.int64,device='cpu')
    elif kind in (1,3):state=extra['rng']
    else:return 0,0,None
    seed,counter=rng_words(state)
    return seed,counter,state


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
        from . import _C
        from ._compile_native import NativeCoverageError,_stream_id
        with self._lock:
            self.state.check_open()
            if self._closed:raise RuntimeError('native operation region closed')
            args=map_arg(self.template,lambda n:values[self._indices[n]])
            reason=None
            if not available(self.kind):reason='native sorting graph API 1 unavailable' if self.kind==26 else 'native scatter graph API 1 unavailable' if self.kind==25 else 'native grouped GEMM graph API 1 unavailable' if self.kind in (23,24) else 'native router graph API 1 unavailable' if self.kind in (21,22) else 'native reduction graph API 1 unavailable' if self.kind==20 else 'native transform graph API 1 unavailable' if self.kind==19 else 'native pointwise graph API 1 unavailable' if self.kind>=17 else 'native sequence graph API 1 unavailable' if self.kind>=10 else 'native operation graph API 1 unavailable'
            else:
                try:inputs,specs,integers,scalars,score_shape,extra=_describe(self.kind,args,self.tensor_core)
                except (ValueError,TypeError,OverflowError) as exc:reason=str(exc)
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
            key=(_stream_id(),torch.is_inference_mode_enabled(),
                 tuple((_shape(t),t.dtype,t.device) for t in inputs),tuple(specs),integers,scalars,score_shape)
            if self.kind==26:key=(*key,extra['output_strides'])
            with torch.no_grad():
                entry=self._cache.get(key)
                if entry is None:
                    reusable={}
                    for previous_key,(_,previous_buffers,_) in self._cache.items():
                        if previous_key[:2]!=key[:2]:continue
                        for i,(source,buffer) in enumerate(zip(inputs,previous_buffers,strict=True)):
                            if _shape(source)==_shape(buffer) and source.dtype==buffer.dtype:
                                reusable[i]=buffer
                    if len(self._cache)>=self.options.cache_size:
                        oldest=next(iter(self._cache));self._cache[oldest][0].close();del self._cache[oldest]
                        self.state.add('native_cache_evictions')
                    device=inputs[0].device
                    buffers=[reusable[i] if i in reusable else torch.empty(t.shape,dtype=t.dtype,device=device) for i,t in enumerate(inputs)]
                    outputs=([torch.empty_strided(shape,stride,dtype=dtype,device=device)
                              for (shape,dtype),stride in zip(specs,extra['output_strides'],strict=True)] if self.kind==26 else
                             [torch.empty(shape,dtype=dtype,device=device) for shape,dtype in specs])
                    graph=_C.NativeOperationGraph(self.kind,[*buffers,*outputs],list(integers),list(scalars),list(score_shape))
                    entry=graph,buffers,outputs;self._cache[key]=entry;self.state.add('native_builds')
                else:self._cache.move_to_end(key);self.state.add('native_cache_hits')
                graph,buffers,outputs=entry
                for dst,src in zip(buffers,inputs,strict=True):dst.copy_(src)
                seed,counter,rng=_state(self.kind,extra)
                graph.run(seed,counter)
                owned=sort_output_layout(outputs,extra['output_strides'],copy=True) if self.kind==26 else tuple(t.clone() for t in outputs)
                self.state.add('native_replays');self.state.add('native_nodes_executed',graph.kernel_count)
                if self.kind==0:return (*owned,rng)
                if self.kind==2:return (*owned,buffers[3].clone(),rng)
                return owned if self.kind in (1,3,8,9,10,11,12,24) or self.kind in (20,26) and len(owned)==2 else owned[0]

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
