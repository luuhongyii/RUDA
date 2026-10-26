"""Model-independent inference patterns; storage casts remain explicit."""
import math

import torch
from torch.fx import Node


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
