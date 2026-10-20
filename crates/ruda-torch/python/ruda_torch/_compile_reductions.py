"""Static metadata for extremum, logical and integer reduction graph regions."""
import math
import torch
from torch.fx import Node
from ._compile_pointwise import _bind


_INTS=(torch.bool,torch.int64,torch.int32,torch.int16,torch.int8,torch.uint8)
_TYPES=(*_INTS,torch.float16,torch.bfloat16,torch.float32,torch.float64)
_SPECS={
    torch.ops.aten.amax.default:(0,('self','dim','keepdim'),{'dim':(),'keepdim':False}),
    torch.ops.aten.amin.default:(1,('self','dim','keepdim'),{'dim':(),'keepdim':False}),
    torch.ops.aten.max.default:(0,('self',),{}),torch.ops.aten.min.default:(1,('self',),{}),
    torch.ops.aten.argmax.default:(2,('self','dim','keepdim'),{'dim':None,'keepdim':False}),
    torch.ops.aten.argmin.default:(3,('self','dim','keepdim'),{'dim':None,'keepdim':False}),
    torch.ops.aten.max.dim:(4,('self','dim','keepdim'),{'keepdim':False}),
    torch.ops.aten.min.dim:(5,('self','dim','keepdim'),{'keepdim':False}),
}
for name,code in (('any',6),('all',7)):
    packet=getattr(torch.ops.aten,name)
    _SPECS[packet.default]=(code,('self',),{})
    _SPECS[packet.dim]=(code,('self','dim','keepdim'),{'keepdim':False})
    _SPECS[packet.dims]=(code,('self','dim','keepdim'),{'dim':None,'keepdim':False})
for name,code,overload in (('sum',8,'dim_IntList'),('prod',9,'dim_int')):
    packet=getattr(torch.ops.aten,name)
    _SPECS[packet.default]=(code,('self','dtype'),{'dtype':None})
    _SPECS[getattr(packet,overload)]=(code,('self','dim','keepdim','dtype'),{'keepdim':False,'dtype':None})


def definition(node):
    if node.op!='call_function' or node.target not in _SPECS:return None
    operation,names,defaults=_SPECS[node.target];args=_bind(node,names,defaults)
    if args is None:return None
    q=args['self'];value=q.meta.get('val') if isinstance(q,Node) else None
    result=node.meta.get('val')
    if operation in (4,5):
        if not isinstance(result,(tuple,list)) or len(result)!=2 or not isinstance(result[1],torch.Tensor) or result[1].dtype!=torch.int64:return None
        result=result[0]
    if not isinstance(value,torch.Tensor) or not isinstance(result,torch.Tensor) or value.dtype not in _TYPES or not result.is_contiguous():return None
    if operation>=8 and (value.dtype not in _INTS or result.dtype not in _INTS or result.dtype==torch.bool):return None
    requested=args.get('dtype')
    if requested is not None and (not isinstance(requested,torch.dtype) or requested!=result.dtype):return None
    dim=args.get('dim');keepdim=args.get('keepdim',False)
    if type(keepdim) is not bool:return None
    if dim is not None and type(dim) is not int and (not isinstance(dim,(list,tuple)) or any(type(d) is not int for d in dim)):return None
    return 20,(q,operation,dim,keepdim,result.dtype)


def describe(args):
    q,operation,dims,keepdim,dtype=args;rank=q.ndim;effective=max(rank,1);shape=tuple(q.shape)
    if q.dtype not in _TYPES or rank>8 or any(type(d) is not int or d<0 or d>2**32-1 for d in shape):
        raise ValueError('native reduction requires static rank <= 8 and supported dtype')
    axes=tuple(range(effective)) if dims is None or operation not in (6,7) and isinstance(dims,(tuple,list)) and not dims else (dims,) if type(dims) is int else tuple(dims)
    if any(type(d) is not int or not -effective<=d<effective for d in axes):raise ValueError('invalid reduction dimensions')
    axes=tuple(d%effective for d in axes)
    if len(set(axes))!=len(axes):raise ValueError('reduction dimensions must be unique')
    if operation in (4,5) and len(axes)!=1:raise ValueError('indexed extrema require one dimension')
    native_shape=shape or (1,)
    length=math.prod(native_shape[d] for d in axes)
    if operation<=5 and length==0:raise ValueError('extremum reduction has no identity for an empty axis')
    if operation in (2,3) and q.dtype==torch.bool:raise ValueError('argmax/argmin do not support bool')
    if operation<8:
        expected=torch.int64 if operation in (2,3) else (torch.uint8 if q.dtype==torch.uint8 else torch.bool) if operation in (6,7) else q.dtype
        if dtype!=expected:raise ValueError('reduction result dtype differs from ATen')
    elif q.dtype not in _INTS or dtype not in _INTS or dtype==torch.bool:raise ValueError('integer reduction requires an integer result')
    output=tuple(1 if d in axes else size for d,size in enumerate(shape)) if keepdim else tuple(size for d,size in enumerate(shape) if d not in axes)
    if max(length,math.prod(output),q.numel())>2**32-1:raise ValueError('native reduction indexing overflow')
    outputs=[(output,dtype)]
    if operation in (4,5):outputs.append((output,torch.int64))
    return [q],outputs,(operation,sum(1<<d for d in axes),int(keepdim)),(),(),{}
