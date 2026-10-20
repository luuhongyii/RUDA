"""Schema-preserving repeat, padding, strided slice and roll graph bindings."""
import math
import struct

import torch
from torch.fx import Node
from ._compile_pointwise import _bind, _i64


_TYPES=(torch.float32,torch.float16,torch.bfloat16,torch.float64,torch.bool,
        torch.int64,torch.int32,torch.int16,torch.int8,torch.uint8,torch.float8_e4m3fn,torch.float8_e5m2)
_SCHEMAS={
    torch.ops.aten.repeat.default:(0,('self','repeats'),{}),
    torch.ops.aten.repeat_interleave.self_int:(1,('self','repeats','dim','output_size'),{'dim':None,'output_size':None}),
    torch.ops.aten.constant_pad_nd.default:(2,('self','pad','value'),{'value':0}),
    torch.ops.aten.slice_copy.Tensor:(3,('self','dim','start','end','step'),{'dim':0,'start':None,'end':None,'step':1}),
    torch.ops.aten.slice.Tensor:(3,('self','dim','start','end','step'),{'dim':0,'start':None,'end':None,'step':1}),
    torch.ops.aten.slice_backward.default:(4,('grad_output','input_sizes','dim','start','end','step'),{}),
    torch.ops.aten.roll.default:(5,('self','shifts','dims'),{'dims':()}),
}


def definition(node):
    if node.op!='call_function' or node.target not in _SCHEMAS:return None
    result=node.meta.get('val')
    if not isinstance(result,torch.Tensor) or result.dtype not in _TYPES:return None
    mode,names,defaults=_SCHEMAS[node.target];args=_bind(node,names,defaults)
    if args is None:return None
    q=args[names[0]];source=q.meta.get('val') if isinstance(q,Node) else None
    if not isinstance(source,torch.Tensor) or source.dtype!=result.dtype:return None
    if mode in (3,4) and source.numel()>0 and result.numel()>0:
        from ._compile_native import _lower
        if _lower(node) is not None:return None
    if node.target is torch.ops.aten.slice.Tensor:
        from ._compile_native import _internal_alias
        if not _internal_alias(node):return None
    elif not result.is_contiguous():return None
    parameters=tuple(args[name] for name in names[1:])
    def concrete(value):
        if isinstance(value,(tuple,list)):return all(concrete(v) for v in value)
        return value is None or type(value) in (int,float,bool)
    if not all(concrete(v) for v in parameters):return None
    return 19,(q,mode,parameters)


def _integers(values):
    if not isinstance(values,(tuple,list)) or any(type(v) is not int for v in values):raise ValueError('transform requires a static integer sequence')
    return tuple(_i64(v) for v in values)


def _axis(value,rank):
    if type(value) is not int or not -rank<=value<rank:raise ValueError('invalid transform dimension')
    return value%rank


def _fill(value,dtype):
    if type(value) not in (int,float,bool):raise ValueError('padding requires a real scalar')
    if dtype==torch.bool:return int(bool(value)),0.
    if dtype in (torch.int64,torch.int32,torch.int16,torch.int8,torch.uint8):
        if type(value) is float and not math.isfinite(value):raise ValueError('nonfinite integer padding value')
        limits=torch.iinfo(dtype)
        if not limits.min<=value<=limits.max:raise ValueError('padding value overflows input dtype')
        return int(value),0.
    if dtype in (torch.float8_e4m3fn,torch.float8_e5m2):raise ValueError('FP8 padding requires explicit scalar encoding')
    converted=struct.unpack('<f',struct.pack('<f',float(value)))[0]
    if dtype!=torch.float32 and math.isfinite(value) and converted!=value:
        raise ValueError('padding value is not exactly representable in the native FP32 scalar ABI')
    if dtype in (torch.float16,torch.bfloat16) and math.isfinite(value) and abs(value)>torch.finfo(dtype).max:
        raise ValueError('padding value overflows input dtype')
    return 0,converted


def describe(args):
    q,mode,parameters=args;shape=tuple(q.shape);rank=q.ndim;scalars=()
    if q.dtype not in _TYPES or rank>8 or any(type(d) is not int or d<0 or d>2**32-1 for d in shape):
        raise ValueError('native transform requires static rank <= 8 and supported dtype')
    if mode==0:
        repeats=_integers(parameters[0])
        if not rank<=len(repeats)<=8 or any(r<0 for r in repeats):raise ValueError('invalid repeat factors')
        output=tuple(d*r for d,r in zip((1,)*(len(repeats)-rank)+shape,repeats,strict=True))
        integers=(mode,*(repeats or (1,)))
    elif mode==1:
        repeats,axis,output_size=parameters
        if type(repeats) is not int or repeats<0:raise ValueError('repeat_interleave requires nonnegative scalar repeats')
        _i64(repeats)
        flatten=axis is None
        if flatten:axis=0;output=(q.numel()*repeats,)
        else:
            axis=_axis(axis,rank);output=list(shape);output[axis]*=repeats;output=tuple(output)
        if output_size is not None and (type(output_size) is not int or output_size!=output[axis]):
            raise ValueError('repeat_interleave output_size differs from the repeated dimension')
        integers=(mode,axis,repeats,int(flatten))
    elif mode==2:
        pad,value=parameters;pad=_integers(pad)
        if len(pad)%2 or len(pad)>2*rank:raise ValueError('invalid constant padding dimensions')
        pairs=[(0,0)]*max(rank,1);output=list(shape)
        for offset in range(len(pad)//2):
            axis=rank-offset-1;before,after=pad[2*offset:2*offset+2];pairs[axis]=(before,after)
            if shape[axis]+min(before,0)+min(after,0)<0:raise ValueError('negative padding exceeds input size')
            output[axis]+=before+after
            if output[axis]<0 or any(p>0 for p in pad) and output[axis]==0:raise ValueError('invalid padded dimension')
        fill,scalar=_fill(value if any(p>0 for p in pad) else 0,q.dtype);scalars=(scalar,)
        integers=(mode,fill,len(pad)//2,*(p for pair in pairs for p in pair));output=tuple(output)
    elif mode in (3,4):
        if mode==4:
            full=_integers(parameters[0]);axis,start,end,step=parameters[1:]
            if len(full)!=rank or any(d<0 or d>2**32-1 for d in full):raise ValueError('invalid slice backward input shape')
        else:full=shape;axis,start,end,step=parameters
        axis=_axis(axis,len(full))
        if type(step) is not int or step<=0:raise ValueError('slice step must be positive')
        _i64(step)
        if start is not None:_i64(start)
        if end is not None:_i64(end)
        start,end,step=slice(start,end,step).indices(full[axis]);part=list(full)
        part[axis]=max(0,(end-start+step-1)//step);part=tuple(part)
        if mode==4 and shape!=part:raise ValueError('slice gradient shape differs from the selected range')
        output=full if mode==4 else part;integers=(mode,axis,start,end,step)
    else:
        shifts,dims=parameters;shifts=_integers(shifts);dims=_integers(dims)
        if not shifts:raise ValueError('roll requires at least one shift')
        if not dims:
            if len(shifts)!=1:raise ValueError('flattened roll requires exactly one shift')
            integers=(mode,1,shifts[0]%max(q.numel(),1))
        else:
            if len(shifts)!=len(dims):raise ValueError('roll shifts and dimensions differ')
            normalized=[0]*rank
            for shift,axis in zip(shifts,dims,strict=True):
                axis=_axis(axis,rank);normalized[axis]=(normalized[axis]+shift)%max(shape[axis],1)
            integers=(mode,0,*normalized)
        output=shape
    if len(output)>8 or any(d<0 or d>2**32-1 for d in output) or math.prod(output)>2**32-1 or q.numel()>2**32-1:
        raise ValueError('native transform indexing overflow')
    return [q],[(output,q.dtype)],integers,scalars,(),{}
