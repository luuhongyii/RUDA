"""Model-independent inference patterns; storage casts remain explicit."""
import math

import torch
from torch.fx import Node


def pack_nibbles(value):
    return value[...,::2]+value[...,1::2]*16


def fuse_nibble_packing(gm):
    from ._compile_pointwise import _bind
    aten=torch.ops.aten
    def slice_input(node,start):
        if not isinstance(node,Node) or node.op!='call_function' or node.target not in (aten.slice.Tensor,aten.slice_copy.Tensor):return None
        args=_bind(node,('self','dim','start','end','step'),{'dim':0,'start':None,'end':None,'step':1})
        if args is None or not isinstance(args['self'],Node):return None
        value=args['self'].meta.get('val')
        if (not isinstance(value,torch.Tensor) or value.dtype!=torch.uint8 or value.ndim==0 or not value.is_contiguous()
                or any(type(d) is not int for d in value.shape) or value.shape[-1]%2):return None
        rank=value.ndim;axis=args['dim']
        if type(axis) is not int or axis not in (-1,rank-1) or type(args['step']) is not int or args['step']!=2:return None
        if args['start'] is not None and type(args['start']) is not int:return None
        if args['start'] not in ((None,0) if start==0 else (1,)):return None
        end=args['end']
        if end is not None and (type(end) is not int or end<value.shape[-1]):return None
        return args['self']
    changed=False
    for output in tuple(gm.graph.nodes):
        if output.op!='call_function' or output.target not in (aten.add.Tensor,aten.add.Scalar):continue
        args=_bind(output,('self','other','alpha'),{'alpha':1})
        result=output.meta.get('val')
        if args is None or args['alpha']!=1 or not isinstance(result,torch.Tensor) or result.dtype!=torch.uint8 or not result.is_contiguous():continue
        for even,multiply in ((args['self'],args['other']),(args['other'],args['self'])):
            source=slice_input(even,0)
            if source is None or not isinstance(multiply,Node) or multiply.op!='call_function' or multiply.target not in (aten.mul.Tensor,aten.mul.Scalar):continue
            factors=_bind(multiply,('self','other'),{})
            if factors is None:continue
            odd=factors['self'] if type(factors['other']) is int and factors['other']==16 else factors['other'] if type(factors['self']) is int and factors['self']==16 else None
            if slice_input(odd,1) is not source:continue
            value=source.meta['val']
            if value.device.type!='ruda' or value.device!=result.device or tuple(result.shape)!=(*value.shape[:-1],value.shape[-1]//2):continue
            safe=True
            for first in (even,odd):
                node=first.next
                while node is not output:
                    if node.op!='call_function' or not isinstance(node.target,torch._ops.OpOverload) or node.target._schema.is_mutable:
                        safe=False;break
                    node=node.next
                if not safe:break
            if not safe:continue
            with gm.graph.inserting_before(output):
                fused=gm.graph.call_function(pack_nibbles,(source,));fused.meta=dict(output.meta)
            output.replace_all_uses_with(fused);gm.graph.erase_node(output)
            for node in (multiply,even,odd):
                if not node.users:gm.graph.erase_node(node)
            changed=True;break
    if changed:gm.recompile()
    return gm


def fuse_normalizations(gm):
    aten=torch.ops.aten
    def call(node,target):return isinstance(node,Node) and node.op=='call_function' and node.target==target
    def args(node,names,defaults=()):
        result=dict(zip(names,node.args));result.update(node.kwargs)
        for key,value in defaults:result.setdefault(key,value)
        return result
    for output in list(gm.graph.nodes):
        if not call(output,aten.mul.Tensor) or len(output.args)!=2 or output.kwargs:continue
        source,inverse=output.args
        if not call(inverse,aten.rsqrt.default):source,inverse=inverse,source
        if not isinstance(source,Node) or not call(inverse,aten.rsqrt.default):continue
        value=source.meta.get('val');result=output.meta.get('val')
        if (not isinstance(value,torch.Tensor) or value.dtype!=torch.float32 or value.ndim==0
                or not value.is_contiguous() or not isinstance(result,torch.Tensor) or result.dtype!=torch.float32):continue
        if len(inverse.args)!=1 or inverse.kwargs:continue
        addition=inverse.args[0]
        if not isinstance(addition,Node) or addition.target not in (aten.add.Tensor,aten.add.Scalar):continue
        added=args(addition,('self','other','alpha'),(('alpha',1),))
        if set(added)!= {'self','other','alpha'} or added['alpha']!=1:continue
        mean,epsilon=added['self'],added['other']
        if (not call(mean,aten.mean.dim) or type(epsilon) not in (int,float) or not math.isfinite(epsilon)
                or not torch.finfo(torch.float32).tiny<=epsilon<=torch.finfo(torch.float32).max):continue
        averaged=args(mean,('self','dim','keepdim','dtype'),(('keepdim',False),('dtype',None)))
        if set(averaged)!= {'self','dim','keepdim','dtype'}:continue
        if averaged['dim'] not in ([-1],(-1,),[value.ndim-1],(value.ndim-1,)) or averaged['keepdim'] is not True:continue
        if averaged['dtype'] not in (None,torch.float32):continue
        square=averaged['self']
        if not (call(square,aten.pow.Tensor_Scalar) and square.args==(source,2) and not square.kwargs
                or call(square,aten.mul.Tensor) and square.args==(source,source) and not square.kwargs):continue
        with gm.graph.inserting_before(output):
            fused=gm.graph.call_function(aten.rms_norm.default,(source,[value.shape[-1]],None,epsilon))
            fused.meta=dict(output.meta)
        output.replace_all_uses_with(fused);gm.graph.erase_node(output)
        for node in (inverse,addition,mean,square):
            if not node.users:gm.graph.erase_node(node)
    gm.recompile()
    return gm
