"""Metadata lowering for functional scatter/index updates and selected-value VJPs."""
import math
import struct
import torch
from torch.fx import Node
from ._compile_pointwise import _bind

_TYPES=(torch.bool,torch.int8,torch.uint8,torch.int16,torch.int32,torch.int64,
        torch.float16,torch.bfloat16,torch.float32,torch.float64)
_SPECS={
    torch.ops.aten.scatter_add.default:(0,('self','dim','index','src'),{}),
    torch.ops.aten.scatter.src:(1,('self','dim','index','src'),{}),
    torch.ops.aten.index_add.default:(2,('self','dim','index','source','alpha'),{'alpha':1}),
    torch.ops.aten.index_copy.default:(3,('self','dim','index','source'),{}),
    torch.ops.aten.value_selecting_reduction_backward.default:(4,('grad','dim','indices','sizes','keepdim'),{}),
    torch.ops.aten.scatter.value:(5,('self','dim','index','value'),{}),
    torch.ops.aten.index_fill.int_Scalar:(6,('self','dim','index','value'),{}),
}


def definition(node):
    if node.op!='call_function' or node.target not in _SPECS:return None
    mode,names,defaults=_SPECS[node.target];args=_bind(node,names,defaults)
    if args is None:return None
    q=args.get('self',args.get('grad'));indices=args.get('index',args.get('indices'))
    source=args.get('src',args.get('source'));scalar=args.get('alpha',args.get('value',1))
    if not isinstance(q,Node) or not isinstance(indices,Node) or mode<=3 and not isinstance(source,Node):return None
    value=q.meta.get('val');result=node.meta.get('val')
    if not isinstance(value,torch.Tensor) or value.dtype not in _TYPES or not isinstance(result,torch.Tensor) or not result.is_contiguous():return None
    if type(args['dim']) is not int or type(scalar) not in (int,float,bool):return None
    shape=args.get('sizes');keepdim=args.get('keepdim',False)
    if type(keepdim) is not bool or shape is not None and (not isinstance(shape,(tuple,list)) or any(type(d) is not int for d in shape)):return None
    return 25,(q,indices,source,mode,args['dim'],scalar,None if shape is None else tuple(shape),keepdim)


def _scalar(value,dtype):
    if dtype==torch.bool:return int(bool(value)),0
    if dtype in (torch.int8,torch.uint8,torch.int16,torch.int32,torch.int64):
        if isinstance(value,float) and not math.isfinite(value):raise ValueError('nonfinite integer update scalar')
        info=torch.iinfo(dtype)
        if not info.min<=value<=info.max:raise ValueError('update scalar exceeds storage dtype')
        value=int(value)
    elif math.isfinite(value) and abs(value)>torch.finfo(dtype).max:
        raise ValueError('update scalar exceeds floating storage dtype')
    if isinstance(value,(int,bool)):
        if not -(2**63)<=value<2**63:raise ValueError('update scalar exceeds signed 64-bit ABI')
        return int(value),0
    return struct.unpack('<q',struct.pack('<d',value))[0],1


def _shape(tensor):
    shape=tuple(tensor.shape)
    if len(shape)>8 or any(type(d) is not int or not 0<=d<=2**32-1 for d in shape) or math.prod(shape)>2**32-1:
        raise ValueError('native scatter requires static rank <=8 and U32 indexing')
    return shape


def describe(args):
    q,indices,source,mode,axis,scalar,shape,keepdim=args
    if q.dtype not in _TYPES or indices.dtype not in (torch.int32,torch.int64):raise ValueError('unsupported update value/index dtype')
    original=_shape(q);index_shape=_shape(indices);insert=0
    if mode==4:
        if shape is None or len(shape)>8 or any(type(d) is not int or not 0<=d<=2**32-1 for d in shape):raise ValueError('invalid reduction gradient output shape')
        output=shape;full=shape or (1,);rank=len(full)
        if not -rank<=axis<rank:raise ValueError('reduction gradient axis exceeds rank')
        axis%=rank
        if index_shape!=original:raise ValueError('selected gradient and indices shapes differ')
        part=list(original)
        if not keepdim and shape:part.insert(axis,1);insert=int(len(shape)>1)
        part=tuple(part) or (1,)
        if len(part)!=rank or any(size>full[d] for d,size in enumerate(part) if d!=axis):raise ValueError('selected gradient shape exceeds destination')
        inputs=[q,indices]
    else:
        output=original;full=original or (1,);rank=len(full)
        if not -rank<=axis<rank:raise ValueError('scatter axis exceeds rank')
        axis%=rank;inputs=[q,indices]
        if mode<=3:
            if source.dtype!=q.dtype:raise ValueError('scatter source must have destination dtype')
            source_shape=_shape(source) or (1,);inputs.append(source)
        if mode in (2,3,6):
            if len(index_shape)>1:raise ValueError('index update needs scalar/vector indices')
            if mode!=6 and (len(source_shape)!=rank or any(size!=(indices.numel() if d==axis else full[d]) for d,size in enumerate(source_shape))):
                raise ValueError('index update source shape differs from destination and index length')
        elif indices.numel()!=0:
            part=index_shape or (1,)
            if len(part)!=rank or any(size>full[d] for d,size in enumerate(part) if d!=axis):raise ValueError('scatter index shape exceeds destination')
            if mode<=1 and (len(source_shape)!=rank or any(a>b for a,b in zip(part,source_shape))):raise ValueError('scatter index shape exceeds source')
    if math.prod(output)>2**32-1:raise ValueError('scatter result exceeds U32 indexing')
    bits,real=(0,0) if mode==5 and indices.numel()==0 else _scalar(scalar,q.dtype)
    return inputs,[(output,q.dtype)],(mode,axis,bits,real,insert),(),(),{}
