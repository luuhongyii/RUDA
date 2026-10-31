"""Exact schema binding for integer and parameterized activation graph regions."""
import struct
import math
import torch
from torch.fx import Node


_INTS=(torch.bool,torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64)
_FLOATS=(torch.float32,torch.float16,torch.bfloat16)
_CODES={torch.bool:3,torch.int64:4,torch.int32:5,torch.int16:6,torch.int8:7,torch.uint8:8}
_INTEGER={}
for name,code,overloads in (
    ('add',0,('Tensor','Scalar')),('sub',1,('Tensor','Scalar')),('mul',2,('Tensor','Scalar')),
    ('floor_divide',3,('default','Scalar')),('fmod',5,('Tensor','Scalar')),('remainder',6,('Tensor','Scalar')),
    ('bitwise_and',7,('Tensor','Scalar')),('bitwise_or',8,('Tensor','Scalar')),('bitwise_xor',9,('Tensor','Scalar')),
    ('eq',14,('Tensor','Scalar')),('ne',15,('Tensor','Scalar')),('lt',16,('Tensor','Scalar')),
    ('le',17,('Tensor','Scalar')),('gt',18,('Tensor','Scalar')),('ge',19,('Tensor','Scalar')),
    ('minimum',20,('default',)),('maximum',21,('default',)),('rsub',22,('Tensor','Scalar')),
):
    for overload in overloads:_INTEGER[getattr(getattr(torch.ops.aten,name),overload)]=(code,False)
for name,code in (('neg',10),('abs',11),('sign',12),('bitwise_not',13)):
    _INTEGER[getattr(torch.ops.aten,name).default]=(code,True)
_DIV=(torch.ops.aten.div.Tensor_mode,torch.ops.aten.div.Scalar_mode)


def _bind(node,names,defaults):
    if len(node.args)>len(names) or set(node.kwargs)-set(names):return None
    args=dict(zip(names,node.args))
    if set(args)&set(node.kwargs):return None
    args.update(node.kwargs)
    for name,value in defaults.items():args.setdefault(name,value)
    return args if set(args)==set(names) else None


def _value(node):return node.meta.get('val') if isinstance(node,Node) else None


def definition(node):
    if node.op!='call_function' or not isinstance(node.target,torch._ops.OpOverload):return None
    target=node.target;result=node.meta.get('val')
    if not isinstance(result,torch.Tensor):return None
    if (target in _INTEGER or target in _DIV) and result.dtype in _INTS:
        code,unary=_INTEGER.get(target,(4,False))
        names=('self',) if unary else ('self','other','alpha') if code in (0,1,22) else ('self','other')
        defaults={'alpha':1} if code in (0,1,22) else {}
        if target in _DIV:names=(*names,'rounding_mode');defaults={'rounding_mode':None}
        args=_bind(node,names,defaults)
        if args is None:return None
        q=args['self'];source=_value(q)
        if not isinstance(source,torch.Tensor) or source.dtype not in _INTS:return None
        if target in _DIV:
            if args['rounding_mode'] not in ('floor','trunc'):return None
            code=3 if args['rounding_mode']=='floor' else 4
        other=None if unary else args['other']
        if isinstance(other,Node):
            value=_value(other)
            if not isinstance(value,torch.Tensor) or value.dtype not in _INTS:return None
        elif not unary and type(other) not in (int,bool):return None
        if result.dtype not in _INTS:return None
        alpha=args.get('alpha',1)
        if type(alpha) not in (int,bool):return None
        return 17,(q,other,code,alpha,result.dtype)
    q=other=None;code=None;p0=p1=0.
    if target in (torch.ops.aten.softplus.default,torch.ops.aten.softplus_backward.default):
        backward=target is torch.ops.aten.softplus_backward.default
        names=('grad_output','self','beta','threshold') if backward else ('self','beta','threshold')
        args=_bind(node,names,{} if backward else {'beta':1.,'threshold':20.})
        if args is None:return None
        q=args['self'];other=args.get('grad_output');code=int(backward);p0=args['beta'];p1=args['threshold']
    elif target in (torch.ops.aten.gelu.default,torch.ops.aten.gelu_backward.default):
        backward=target is torch.ops.aten.gelu_backward.default
        names=('grad_output','self','approximate') if backward else ('self','approximate')
        args=_bind(node,names,{'approximate':'none'})
        if args is None or args['approximate'] not in ('none','tanh'):return None
        q=args['self'];other=args.get('grad_output');code=2+2*int(backward)+int(args['approximate']=='tanh')
    elif target in (torch.ops.aten.leaky_relu.default,torch.ops.aten.leaky_relu_backward.default):
        backward=target is torch.ops.aten.leaky_relu_backward.default
        names=('grad_output','self','negative_slope','self_is_result') if backward else ('self','negative_slope')
        args=_bind(node,names,{} if backward else {'negative_slope':.01})
        if args is None:return None
        q=args['self'];other=args.get('grad_output');p0=args['negative_slope'];code=6+int(backward)
        if backward and (type(args['self_is_result']) is not bool or type(p0) not in (int,float) or args['self_is_result'] and p0<0):return None
    elif target in (torch.ops.aten.clamp.default,torch.ops.aten.clamp_min.default,torch.ops.aten.clamp_max.default):
        if target is torch.ops.aten.clamp.default:
            args=_bind(node,('self','min','max'),{'min':None,'max':None})
        else:
            side='min' if target is torch.ops.aten.clamp_min.default else 'max'
            args=_bind(node,('self',side),{})
            if args is not None:args.setdefault('max' if side=='min' else 'min',None)
        if args is None or args['min'] is None and args['max'] is None:return None
        q=args['self']
        if args['min'] is None:code=9;p0=args['max']
        elif args['max'] is None:code=8;p0=args['min']
        else:code=10;p0=args['min'];p1=args['max']
    elif target is torch.ops.aten.log_sigmoid.default:
        args=_bind(node,('self',),{})
        if args is None:return None
        q=args['self'];code=11
    elif target in (torch.ops.aten.minimum.default,torch.ops.aten.maximum.default):
        args=_bind(node,('self','other'),{})
        if args is None:return None
        q=args['self'];other=args['other'];code=12 if target is torch.ops.aten.minimum.default else 13
    elif target in (torch.ops.aten.mish.default,torch.ops.aten.mish_backward.default):
        backward=target is torch.ops.aten.mish_backward.default
        args=_bind(node,('grad_output','self') if backward else ('self',),{})
        if args is None:return None
        q=args['self'];other=args.get('grad_output');code=14+int(backward)
    if code is None:return None
    source=_value(q)
    if not isinstance(source,torch.Tensor) or source.dtype not in _FLOATS or result.dtype not in _FLOATS:return None
    if other is not None and (not isinstance(_value(other),torch.Tensor) or _value(other).dtype not in _FLOATS):return None
    if type(p0) not in (int,float) or type(p1) not in (int,float):return None
    return 18,(q,other,code,p0,p1,result.dtype)


def _i64(value):
    if type(value) not in (int,bool) or not -(2**63)<=value<2**63:raise ValueError('native integer scalar must fit signed int64')
    return int(value)


def expand_clamps(gm):
    from torch._subclasses.fake_tensor import FakeTensor
    from ._compile_normalization import _Writer
    changed=False
    for node in tuple(gm.graph.nodes):
        if node.op=='call_function' and node.target is torch.ops.aten.clamp.Tensor:
            args=_bind(node,('self','min','max'),{'min':None,'max':None})
            if args is None:continue
            q=args['self'];minimum=args['min'];maximum=args['max']
            source=_value(q);result=node.meta.get('val')
            if (not isinstance(source,FakeTensor) or not isinstance(result,torch.Tensor)
                    or result.device.type!='ruda' or source.device!=result.device
                    or result.dtype not in _FLOATS or source.dtype!=result.dtype or not result.is_contiguous()
                    or any(type(d) is not int for d in (*source.shape,*result.shape))
                    or tuple(source.shape)!=tuple(result.shape) or minimum is None and maximum is None):continue
            bounds=[_value(bound) for bound in (minimum,maximum) if bound is not None]
            if any(not isinstance(bound,torch.Tensor) or bound.dtype!=result.dtype or bound.device!=result.device
                   or any(type(d) is not int for d in bound.shape)
                   or tuple(bound.shape) not in ((),tuple(source.shape)) for bound in bounds):continue
            if minimum is not None and maximum is not None:
                writer=_Writer(gm.graph,source);aten=torch.ops.aten;shape=tuple(source.shape)
                with gm.graph.inserting_before(node):
                    below=writer.emit(aten.lt.Tensor,(q,minimum),shape,torch.bool)
                    lower=writer.emit(aten.where.self,(below,minimum,q),shape,result.dtype)
                    above=writer.emit(aten.lt.Tensor,(maximum,lower),shape,torch.bool)
                    upper=writer.emit(aten.where.self,(above,maximum,lower),shape,result.dtype)
                    for bound in (maximum,minimum):
                        isnan=writer.emit(aten.ne.Tensor,(bound,bound),tuple(_value(bound).shape),torch.bool)
                        upper=writer.emit(aten.where.self,(isnan,bound,upper),shape,result.dtype)
                    isnan=writer.emit(aten.ne.Tensor,(q,q),shape,torch.bool)
                node.target=aten.where.self;node.args=(isnan,q,upper)
            else:
                node.target=torch.ops.aten.minimum.default if minimum is None else torch.ops.aten.maximum.default
                node.args=(q,maximum if minimum is None else minimum)
            node.kwargs={};changed=True;continue
        if node.op!='call_function' or node.target is not torch.ops.aten.clamp.default:continue
        defined=definition(node)
        if defined is None or defined[0]!=18 or defined[1][2]!=10:continue
        q,_,_,minimum,maximum,dtype=defined[1]
        value=node.meta.get('val')
        if value.device.type!='ruda' or not value.is_contiguous() or not all(math.isfinite(v) for v in (minimum,maximum)):continue
        with gm.graph.inserting_before(node):
            lower=gm.graph.call_function(torch.ops.aten.clamp_min.default,(q,minimum))
            lower.meta=dict(node.meta)
        node.target=torch.ops.aten.clamp_max.default;node.args=(lower,maximum);node.kwargs={}
        changed=True
    if changed:gm.recompile()
    return gm


def describe(kind,args):
    q,other,op=args[:3];inputs=[q];shape=tuple(q.shape)
    if kind==17:
        alpha,dtype=args[3:];alpha=_i64(alpha)
        mode=2 if other is None else 0 if isinstance(other,torch.Tensor) else 1
        common=q.dtype if mode==2 else torch.result_type(q,other)
        if common not in _INTS or dtype not in _INTS:raise ValueError('native integer operation requires integral storage')
        if mode==0:
            if other.dtype not in _INTS:raise ValueError('integer operand dtype differs')
            inputs.append(other);shape=tuple(torch.broadcast_shapes(q.shape,other.shape));scalar=0
        else:scalar=0 if mode==2 else _i64(other)
        comparison=14<=op<=19
        if dtype!=(torch.bool if comparison else common):raise ValueError('integer result dtype differs from ATen promotion')
        if common==torch.bool and op not in (0,2,7,8,9,13,14,15,16,17,18,19,20,21):raise ValueError('operation does not support bool')
        if op in (0,1,22) and common!=torch.bool:
            limits=torch.iinfo(common)
            if not limits.min<=alpha<=limits.max:raise ValueError('integer alpha exceeds the promoted dtype')
        integers=(op,mode,scalar,alpha,_CODES[common]);scalars=()
    else:
        p0,p1,dtype=args[3:]
        if q.dtype not in _FLOATS or dtype not in _FLOATS:raise ValueError('activation requires FP32/FP16/BF16')
        scalars=tuple(struct.unpack('<f',struct.pack('<f',float(v)))[0] for v in (p0,p1));integers=(op,)
        if other is not None:
            inputs.append(other)
            if op in (12,13):shape=tuple(torch.broadcast_shapes(q.shape,other.shape))
            elif other.shape!=q.shape or other.dtype!=q.dtype:raise ValueError('activation gradient shape/dtype differs')
        if op not in (12,13) and q.dtype!=dtype:raise ValueError('activation output dtype differs')
    return inputs,[(shape,dtype)],integers,scalars,(),{}
