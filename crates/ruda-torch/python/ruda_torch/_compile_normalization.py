"""Expand AOT normalization tuples into native graph arithmetic and casts."""
import math
import operator

import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.node import map_arg
from torch._subclasses.fake_tensor import FakeTensor


_FORWARD=torch.ops.aten.native_layer_norm.default
_BACKWARD=torch.ops.aten.native_layer_norm_backward.default
_PUBLIC=torch.ops.aten.layer_norm.default
_FLOATS=(torch.float32,torch.float16,torch.bfloat16)


def _value(node):
    return node.meta.get('val') if isinstance(node,Node) else None


def _contract(node):
    if node.op!='call_function' or node.target not in (_FORWARD,_BACKWARD,_PUBLIC):return None
    backward=node.target==_BACKWARD
    names=('grad_out','input','normalized_shape','mean','rstd','weight','bias','output_mask') if backward else ('input','normalized_shape','weight','bias','eps')
    if node.target==_PUBLIC:names=(*names,'cudnn_enable')
    if len(node.args)>len(names) or set(node.kwargs)-set(names):return None
    args=dict(zip(names,node.args))
    if set(args)&set(node.kwargs):return None
    args.update(node.kwargs)
    if not backward:
        args.setdefault('weight',None);args.setdefault('bias',None);args.setdefault('eps',1e-5)
    if node.target==_PUBLIC:
        args.setdefault('cudnn_enable',True)
        if type(args['cudnn_enable']) is not bool:return None
    if set(args)!=set(names):return None
    source=_value(args['input']);shape=args['normalized_shape'];outputs=node.meta.get('val')
    if node.target==_PUBLIC:outputs=(outputs,None,None)
    if not isinstance(source,FakeTensor) or source.device.type!='ruda' or source.device.index not in (None,0):return None
    if source.dtype not in _FLOATS or source.layout!=torch.strided or source.is_conj() or source.is_neg():return None
    if not 1<=source.ndim<=8 or any(type(d) is not int or d<=0 for d in source.shape) or source.numel()>2**32-1:return None
    if not isinstance(shape,(tuple,list)) or not 1<=len(shape)<=source.ndim or any(type(d) is not int or d<=0 for d in shape):return None
    shape=tuple(shape)
    if tuple(source.shape[-len(shape):])!=shape:return None
    if not isinstance(outputs,(tuple,list)) or len(outputs)!=3:return None
    for key in ('weight','bias'):
        if args[key] is not None:
            param=_value(args[key])
            if not isinstance(param,FakeTensor) or param.device!=source.device or param.dtype not in (source.dtype,torch.float32) or tuple(param.shape)!=shape:return None
    stats_shape=tuple(source.shape[:-len(shape)])+(1,)*len(shape)
    expected=(tuple(source.shape),shape,shape) if backward else (tuple(source.shape),stats_shape,stats_shape)
    if backward:
        mask=args['output_mask'];grad=_value(args['grad_out'])
        if not isinstance(mask,(tuple,list)) or len(mask)!=3 or any(type(v) is not bool for v in mask):return None
        if not isinstance(grad,FakeTensor) or grad.device!=source.device or grad.dtype!=source.dtype or grad.shape!=source.shape:return None
        for key in ('mean','rstd'):
            stat=_value(args[key])
            if not isinstance(stat,FakeTensor) or stat.device!=source.device or stat.dtype not in (source.dtype,torch.float32) or tuple(stat.shape)!=stats_shape:return None
        present=(mask[0],mask[1] and args['weight'] is not None,mask[2] and args['bias'] is not None)
    else:
        epsilon=args['eps']
        if type(epsilon) not in (int,float) or not math.isfinite(epsilon) or abs(epsilon)>torch.finfo(torch.float32).max:return None
        present=(True,False,False) if node.target==_PUBLIC else (True,True,True)
    for i,(result,required) in enumerate(zip(outputs,present)):
        if not required:
            if result is not None:return None
        elif (not isinstance(result,FakeTensor) or result.device!=source.device or result.dtype not in _FLOATS
                or tuple(result.shape)!=expected[i] or not result.is_contiguous()):return None
    return args,source,tuple(outputs),stats_shape


class _Writer:
    def __init__(self,graph,source):
        self.graph=graph;self.source=source

    def emit(self,target,args,shape,output_dtype,**kwargs):
        node=self.graph.call_function(target,args,kwargs)
        with self.source.fake_mode:
            node.meta['val']=torch.empty(shape,dtype=output_dtype,device=self.source.device)
        return node

    def cast(self,node,dtype):
        value=_value(node)
        if value.dtype==dtype:return node
        return self.emit(torch.ops.aten._to_copy.default,(node,),tuple(value.shape),dtype,dtype=dtype)

    def binary(self,kind,left,right):
        shape=torch.broadcast_shapes(_value(left).shape,_value(right).shape)
        return self.emit(getattr(torch.ops.aten,kind).Tensor,(left,right),tuple(shape),torch.float32)

    def scalar(self,kind,left,value):
        return self.emit(getattr(torch.ops.aten,kind).Scalar,(left,value),tuple(_value(left).shape),torch.float32)

    def reduction(self,kind,value,axes,keep):
        source=_value(value)
        if not axes:
            return self.emit(torch.ops.aten.clone.default,(value,),tuple(source.shape),source.dtype)
        axes=tuple(axes)
        shape=tuple(1 if d in axes else size for d,size in enumerate(source.shape)) if keep else tuple(size for d,size in enumerate(source.shape) if d not in axes)
        target=torch.ops.aten.mean.dim if kind=='mean' else torch.ops.aten.sum.dim_IntList
        return self.emit(target,(value,axes,keep),shape,torch.float32)

    def result(self,value,metadata):
        value=self.cast(value,metadata.dtype)
        # A returned statistic or gradient is independent storage, including
        # the no-leading-dimensions case and identity dtype conversions.
        return self.emit(torch.ops.aten.clone.default,(value,),tuple(metadata.shape),metadata.dtype)


def _forward(writer,args,source,outputs):
    axes=tuple(range(source.ndim-len(args['normalized_shape']),source.ndim))
    x=writer.cast(args['input'],torch.float32)
    mean=writer.reduction('mean',x,axes,True)
    centered=writer.binary('sub',x,mean)
    variance=writer.reduction('mean',writer.binary('mul',centered,centered),axes,True)
    denominator=writer.scalar('add',variance,args['eps'])
    rstd=writer.emit(torch.ops.aten.rsqrt.default,(denominator,),tuple(_value(denominator).shape),torch.float32)
    out=writer.binary('mul',centered,rstd)
    if args['weight'] is not None:out=writer.binary('mul',out,writer.cast(args['weight'],torch.float32))
    if args['bias'] is not None:out=writer.binary('add',out,writer.cast(args['bias'],torch.float32))
    return tuple(writer.result(value,meta) if meta is not None else None for value,meta in zip((out,mean,rstd),outputs))


def _backward(writer,args,source,outputs):
    axis=source.ndim-len(args['normalized_shape']);axes=tuple(range(axis,source.ndim));leading=tuple(range(axis))
    grad=writer.cast(args['grad_out'],torch.float32)
    x=writer.cast(args['input'],torch.float32)
    mean=writer.cast(args['mean'],torch.float32);rstd=writer.cast(args['rstd'],torch.float32)
    normalized=writer.binary('mul',writer.binary('sub',x,mean),rstd)
    results=[None,None,None]
    if outputs[0] is not None:
        scaled=grad if args['weight'] is None else writer.binary('mul',grad,writer.cast(args['weight'],torch.float32))
        center=writer.reduction('mean',scaled,axes,True)
        projection=writer.reduction('mean',writer.binary('mul',scaled,normalized),axes,True)
        dx=writer.binary('sub',writer.binary('sub',scaled,center),writer.binary('mul',normalized,projection))
        results[0]=writer.result(writer.binary('mul',dx,rstd),outputs[0])
    if outputs[1] is not None:
        results[1]=writer.result(writer.reduction('sum',writer.binary('mul',grad,normalized),leading,False),outputs[1])
    if outputs[2] is not None:
        results[2]=writer.result(writer.reduction('sum',grad,leading,False),outputs[2])
    return tuple(results)


def expand_normalizations(gm):
    contracts={node:contract for node in gm.graph.nodes if (contract:=_contract(node)) is not None}
    if not contracts:return gm
    graph=Graph();env={}
    for node in gm.graph.nodes:
        if node in contracts:
            args,source,outputs,_=contracts[node]
            args=map_arg(args,lambda old:env[old]);writer=_Writer(graph,source)
            result=(_backward if node.target==_BACKWARD else _forward)(writer,args,source,outputs)
            env[node]=result[0] if node.target==_PUBLIC else result
        elif (node.op=='call_function' and node.target is operator.getitem and len(node.args)==2
              and not node.kwargs and isinstance(node.args[0],Node) and node.args[0] in contracts
              and node.args[0].target!=_PUBLIC
              and type(node.args[1]) is int and -3<=node.args[1]<3):
            env[node]=env[node.args[0]][node.args[1]]
        else:
            copied=graph.node_copy(node,lambda old:env[old]);copied.meta=dict(node.meta);env[node]=copied
    return GraphModule(gm,graph)
