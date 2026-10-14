"""ATen ordering/selection regions, including stable ties and output layouts."""
import math
import torch
from torch.fx import Node
from ._compile_pointwise import _bind

_TYPES=(torch.float32,torch.float16,torch.bfloat16,torch.float64,
        torch.int64,torch.int32,torch.int16,torch.int8,torch.uint8,torch.bool)
_SPECS={
    torch.ops.aten.topk.default:(0,('self','k','dim','largest','sorted'),{'dim':-1,'largest':True,'sorted':True}),
    torch.ops.aten.sort.default:(1,('self','dim','descending'),{'dim':-1,'descending':False}),
    torch.ops.aten.sort.stable:(1,('self','stable','dim','descending'),{'dim':-1,'descending':False}),
    torch.ops.aten.argsort.default:(2,('self','dim','descending'),{'dim':-1,'descending':False}),
    torch.ops.aten.argsort.stable:(2,('self','stable','dim','descending'),{'dim':-1,'descending':False}),
}


def definition(node):
    if node.op!='call_function' or node.target not in _SPECS:return None
    mode,names,defaults=_SPECS[node.target];args=_bind(node,names,defaults)
    if args is None or not isinstance(args['self'],Node):return None
    q=args['self'];value=q.meta.get('val');result=node.meta.get('val')
    if not isinstance(value,torch.Tensor) or value.dtype not in _TYPES:return None
    outputs=(result,) if mode==2 else result
    if not isinstance(outputs,(tuple,list)) or len(outputs)!=(1 if mode==2 else 2):return None
    if any(not isinstance(t,torch.Tensor) or t.device!=value.device or t.layout!=torch.strided for t in outputs):return None
    if outputs[-1].dtype!=torch.int64 or mode!=2 and outputs[0].dtype!=value.dtype:return None
    k=args.get('k');descending=args.get('largest',args.get('descending',False));ordered=args.get('sorted',True)
    stable=args.get('stable',False)
    if stable is None:stable=False
    if type(args['dim']) is not int or k is not None and type(k) is not int:return None
    if any(type(v) is not bool for v in (descending,ordered,stable)):return None
    strides=tuple(tuple(t.stride()) for t in outputs)
    if any(type(s) is not int or s<0 for layout in strides for s in layout):return None
    return 26,(q,mode,args['dim'],k,descending,ordered,stable,strides)


def describe(args):
    q,mode,axis,k,descending,ordered,stable,strides=args;shape=tuple(q.shape);rank=max(q.ndim,1)
    if q.dtype not in _TYPES or q.ndim>32 or any(type(d) is not int or not 0<=d<=2**32-1 for d in shape):
        raise ValueError('native sort requires static rank <=32 and supported real dtype')
    if not -rank<=axis<rank:raise ValueError('sort dimension exceeds rank')
    axis%=rank;length=shape[axis] if shape else 1
    k=length if mode!=0 else k
    if type(k) is not int or not 0<=k<=length:raise ValueError('topk selection is out of range')
    if mode==0 and q.dtype==torch.bool:raise ValueError('topk does not support bool')
    output=list(shape)
    if output:output[axis]=k
    output=tuple(output)
    if max(math.prod(output),q.numel())>2**32-1:raise ValueError('sort indexing exceeds U32 range')
    for layout in strides:
        if len(layout)!=len(output):raise ValueError('sort output stride rank differs')
        if math.prod(output):
            extent=1
            for stride,size in sorted((s,d) for s,d in zip(layout,output) if d>1):
                if stride!=extent:raise ValueError('captured sort output is not dense nonoverlapping storage')
                extent*=size
    specs=[(output,torch.int64)] if mode==2 else [(output,q.dtype),(output,torch.int64)]
    return [q],specs,(mode,axis,k,int(descending),int(ordered),int(stable),int(not shape)),(),(),{'output_strides':strides}


def output_layout(values,strides,*,copy):
    result=[]
    for value,layout in zip(values,strides,strict=True):
        if copy or tuple(value.stride())!=layout:
            value=torch.empty_strided(value.shape,layout,dtype=value.dtype,device=value.device).copy_(value)
        result.append(value)
    return tuple(result)
