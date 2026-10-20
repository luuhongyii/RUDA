"""Concretize captured shape expressions from invocation metadata, never hints."""
import builtins
import math
import operator
import struct

import torch
from torch.fx import Graph, GraphModule, Node

_SYMBOLIC=(torch.SymInt,torch.SymFloat,torch.SymBool)
_SCALARS={operator.add,operator.sub,operator.mul,operator.truediv,operator.floordiv,
    operator.mod,operator.pow,operator.neg,operator.pos,operator.abs,operator.lt,operator.le,
    operator.gt,operator.ge,operator.eq,operator.ne,operator.and_,operator.or_,operator.xor,
    operator.not_,operator.lshift,operator.rshift,operator.getitem,builtins.min,builtins.max,
    builtins.int,builtins.float,builtins.bool,math.ceil,math.floor,math.trunc,math.sqrt}
_SCALARS.update(getattr(torch,name) for name in ('sym_int','sym_float','sym_not','sym_min','sym_max',
    'sym_ite','sym_sqrt','_sym_sqrt','_sym_log2') if hasattr(torch,name))
_SIZES={}
for _name,_method in (('sym_size','size'),('sym_stride','stride'),('sym_numel','numel'),
                      ('sym_storage_offset','storage_offset'),('dim','dim')):
    _packet=getattr(torch.ops.aten,_name,None)
    for _overload in ('default','int'):
        _target=getattr(_packet,_overload,None)
        if _target is not None:_SIZES[_target]=_method
_GUARDS={getattr(torch.ops.aten._assert_scalar,'default')}
_METHODS={'size','stride','numel','dim','ndimension','storage_offset'}


class ShapeSpecializationError(ValueError):
    pass


def _map(value,leaf):
    if isinstance(value,tuple):
        values=[_map(v,leaf) for v in value]
        return type(value)(*values) if hasattr(value,'_fields') else torch.Size(values) if isinstance(value,torch.Size) else tuple(values)
    if isinstance(value,list):return [_map(v,leaf) for v in value]
    if isinstance(value,dict):return {k:_map(v,leaf) for k,v in value.items()}
    if isinstance(value,slice):return slice(*(_map(v,leaf) for v in (value.start,value.stop,value.step)))
    return leaf(value)


def _leaves(value):
    result=[]
    def visit(item):
        if isinstance(item,(tuple,list)):
            for child in item:visit(child)
        elif isinstance(item,dict):
            for child in item.values():visit(child)
        elif isinstance(item,slice):
            for child in (item.start,item.stop,item.step):visit(child)
        else:result.append(item)
    visit(value)
    return result


def _float_key(value):return struct.pack('!d',value)


def _equal(a,b):
    if type(a) is float and type(b) is float:return _float_key(a)==_float_key(b)
    return type(a) is type(b) and a==b


def value_key(value):
    if isinstance(value,torch.Tensor):
        if value.layout!=torch.strided:raise ShapeSpecializationError('dynamic native graphs require strided tensors')
        return ('tensor',tuple(value.shape),tuple(value.stride()),value.storage_offset(),value.dtype,value.device,
                value.requires_grad,value.is_conj(),value.is_neg())
    if type(value) in (int,bool,str) or value is None:return (type(value).__name__,value)
    if type(value) is float:return ('float',_float_key(value))
    if isinstance(value,(torch.dtype,torch.device,torch.layout,torch.memory_format)):return (type(value).__name__,str(value))
    if isinstance(value,(tuple,list)):return (type(value).__name__,tuple(value_key(v) for v in value))
    if isinstance(value,dict):return ('dict',tuple((value_key(k),value_key(v)) for k,v in value.items()))
    raise ShapeSpecializationError(f'unsupported dynamic graph input type: {type(value).__name__}')


def attribute(gm,target):
    value=gm
    for part in str(target).split('.'):value=getattr(value,part)
    return value


def shape_guard(node):
    return (node.op=='call_function' and node.target in _GUARDS and len(node.args)==2 and not node.kwargs
            and type(node.args[0]) is bool and isinstance(node.args[1],str))


def needs_specialization(gm):
    for node in gm.graph.nodes:
        if node.op=='call_function' and node.target in _SIZES:return True
        if node.op=='call_method' and node.target in _METHODS:return True
        if node.op=='call_function' and node.target in (builtins.getattr,builtins.len):return True
        for value in _leaves((node.args,node.kwargs,node.meta.get('val'))):
            if isinstance(value,_SYMBOLIC):return True
            if isinstance(value,torch.Tensor) and value.layout==torch.strided:
                if any(isinstance(v,_SYMBOLIC) for v in (*value.shape,*value.stride(),value.storage_offset())):return True
    return False


class _Bindings:
    def __init__(self):
        self.symbols={}
        self.constraints=[]

    def scalar(self,template,actual,label):
        if isinstance(template,_SYMBOLIC):
            kind=int if isinstance(template,torch.SymInt) else bool if isinstance(template,torch.SymBool) else float
            if type(actual) is not kind:raise ShapeSpecializationError(f'{label}: symbolic scalar type changed')
            expression=template.node.expr
            self.constraints.append((expression,actual,kind,label))
            if expression.is_Symbol:self._assign(expression,actual,label)
        elif not _equal(template,actual):
            raise ShapeSpecializationError(f'{label}: captured static metadata changed')

    def _assign(self,symbol,value,label):
        if symbol in self.symbols and not _equal(self.symbols[symbol],value):
            raise ShapeSpecializationError(f'{label}: inconsistent values for {symbol}')
        self.symbols[symbol]=value

    def bind(self,template,actual,label):
        if isinstance(template,torch.Tensor):
            if not isinstance(actual,torch.Tensor) or template.layout!=torch.strided or actual.layout!=torch.strided:
                raise ShapeSpecializationError(f'{label}: tensor layout changed')
            if template.dtype!=actual.dtype or template.device!=actual.device or template.ndim!=actual.ndim:
                raise ShapeSpecializationError(f'{label}: tensor dtype/device/rank changed')
            if template.is_conj()!=actual.is_conj() or template.is_neg()!=actual.is_neg():
                raise ShapeSpecializationError(f'{label}: tensor view bits changed')
            for i,(old,new) in enumerate(zip(template.shape,actual.shape,strict=True)):self.scalar(old,new,f'{label}.shape[{i}]')
            for i,(old,new) in enumerate(zip(template.stride(),actual.stride(),strict=True)):self.scalar(old,new,f'{label}.stride[{i}]')
            self.scalar(template.storage_offset(),actual.storage_offset(),f'{label}.storage_offset')
        elif isinstance(template,(tuple,list)):
            if not isinstance(actual,type(template)) or len(template)!=len(actual):raise ShapeSpecializationError(f'{label}: input structure changed')
            for i,(old,new) in enumerate(zip(template,actual,strict=True)):self.bind(old,new,f'{label}[{i}]')
        elif isinstance(template,dict):
            if not isinstance(actual,dict) or template.keys()!=actual.keys():raise ShapeSpecializationError(f'{label}: input keys changed')
            for key in template:self.bind(template[key],actual[key],f'{label}[{key!r}]')
        else:self.scalar(template,actual,label)

    def _evaluate(self,expression,kind):
        from torch.utils._sympy.interp import sympy_interp
        from torch.utils._sympy.reference import PythonReferenceAnalysis
        if expression.free_symbols-set(self.symbols):raise ShapeSpecializationError(f'unbound shape expression: {expression}')
        try:value=sympy_interp(PythonReferenceAnalysis,self.symbols,expression)
        except (KeyError,NotImplementedError,TypeError,ValueError,ArithmeticError) as exc:
            raise ShapeSpecializationError(f'cannot evaluate shape expression {expression}') from exc
        try:
            if kind is int:
                if type(value) is bool or getattr(value,'is_integer',None) is False or int(value)!=value:
                    raise ShapeSpecializationError(f'nonintegral shape expression: {expression}')
                return int(value)
            if kind is bool:
                import sympy
                if value not in (True,False,sympy.true,sympy.false):raise ShapeSpecializationError(f'nonboolean shape expression: {expression}')
                return bool(value)
            return float(value)
        except (TypeError,ValueError,ArithmeticError) as exc:
            raise ShapeSpecializationError(f'invalid shape expression result: {expression}') from exc

    def solve(self):
        import sympy
        pending=list(self.constraints)
        while pending:
            remaining=[];progress=False
            for expression,actual,kind,label in pending:
                unknown=expression.free_symbols-set(self.symbols)
                if not unknown:
                    if not _equal(self._evaluate(expression,kind),actual):raise ShapeSpecializationError(f'{label}: symbolic shape constraint failed')
                    progress=True;continue
                if kind is int and len(unknown)==1:
                    symbol=next(iter(unknown));reduced=expression.xreplace({k:sympy.sympify(v) for k,v in self.symbols.items()})
                    try:polynomial=sympy.Poly(reduced,symbol)
                    except sympy.PolynomialError:polynomial=None
                    if polynomial is not None and polynomial.degree()==1:
                        coefficient,constant=polynomial.all_coeffs();value=(sympy.Integer(actual)-constant)/coefficient
                        if not value.is_Integer:raise ShapeSpecializationError(f'{label}: nonintegral symbolic dimension')
                        self._assign(symbol,int(value),label);progress=True
                remaining.append((expression,actual,kind,label))
            if not progress:raise ShapeSpecializationError('shape symbols are not uniquely determined by invocation metadata')
            pending=remaining

    def resolve(self,value):
        if not isinstance(value,_SYMBOLIC):return value
        kind=int if isinstance(value,torch.SymInt) else bool if isinstance(value,torch.SymBool) else float
        return self._evaluate(value.node.expr,kind)


def specialize(gm,inputs):
    from torch._subclasses.fake_tensor import FakeTensor,FakeTensorMode,unset_fake_temporarily
    from torch.utils._mode_utils import no_dispatch
    bindings=_Bindings();sources={}
    placeholders=[n for n in gm.graph.nodes if n.op=='placeholder']
    if len(inputs)!=len(placeholders):raise ShapeSpecializationError('dynamic graph input arity changed')
    for node,value in zip(placeholders,inputs,strict=True):
        if 'val' not in node.meta:raise ShapeSpecializationError(f'{node.name}: captured input metadata missing')
        bindings.bind(node.meta['val'],value,node.name);sources[node]=value
    for node in gm.graph.nodes:
        if node.op=='get_attr':
            value=attribute(gm,node.target);sources[node]=value
            if 'val' in node.meta:bindings.bind(node.meta['val'],value,node.name)
    bindings.solve()
    mode=FakeTensorMode(allow_fallback_kernels=False,allow_non_fake_inputs=False,static_shapes=True)
    mode.cache_enabled=False
    tensors={}

    def materialize(value):
        if not isinstance(value,torch.Tensor):return bindings.resolve(value)
        if id(value) in tensors:return tensors[id(value)]
        if value.layout!=torch.strided or value.is_conj() or value.is_neg():
            raise ShapeSpecializationError('symbolic metadata requires an ordinary strided tensor')
        shape=tuple(bindings.resolve(d) for d in value.shape);stride=tuple(bindings.resolve(d) for d in value.stride())
        offset=bindings.resolve(value.storage_offset())
        if any(type(d) is not int or d<0 for d in (*shape,*stride,offset)):
            raise ShapeSpecializationError('negative or nonintegral tensor metadata')
        span=offset+(1+sum((d-1)*s for d,s in zip(shape,stride,strict=True)) if all(shape) else 0)
        with unset_fake_temporarily(),no_dispatch():
            meta=torch.empty((span,),dtype=value.dtype,device='meta').as_strided(shape,stride,offset)
            meta.requires_grad_(value.requires_grad)
            result=FakeTensor(mode,meta,value.device)
        tensors[id(value)]=result
        return result

    graph=Graph();env={};constants={};folded=[]
    def argument(value):
        return constants[value] if isinstance(value,Node) and value in constants else env[value] if isinstance(value,Node) else bindings.resolve(value)
    def tensor_metadata(value):
        if isinstance(value,Node):
            result=value.meta.get('val')
            if not isinstance(result,torch.Tensor):return None
            return result
        return value
    def fold(node,args,kwargs):
        if node.op=='call_function' and node.target in _SIZES:
            tensor=tensor_metadata(args[0])
            if isinstance(tensor,torch.Tensor) and not any(isinstance(v,(Node,torch.Tensor)) for v in _leaves((args[1:],kwargs))):
                return True,getattr(tensor,_SIZES[node.target])(*args[1:],**kwargs)
        if node.op=='call_method' and node.target in _METHODS:
            tensor=tensor_metadata(args[0])
            if isinstance(tensor,torch.Tensor) and not any(isinstance(v,(Node,torch.Tensor)) for v in _leaves((args[1:],kwargs))):
                return True,getattr(tensor,node.target)(*args[1:],**kwargs)
        if node.op=='call_function' and node.target is builtins.getattr and len(args)==2 and not kwargs:
            tensor=tensor_metadata(args[0])
            if isinstance(tensor,torch.Tensor) and args[1] in ('shape','dtype','device','ndim','layout'):
                return True,getattr(tensor,args[1])
        if node.op=='call_function' and node.target is builtins.len and len(args)==1 and not kwargs:
            value=tensor_metadata(args[0])
            if isinstance(value,(torch.Tensor,tuple,list,dict)):return True,len(value)
        if node.op=='call_function' and node.target in _SCALARS:
            if all(type(v) in (int,float,bool,str,type(None)) for v in _leaves((args,kwargs))):
                return True,node.target(*args,**kwargs)
        return False,None

    for node in gm.graph.nodes:
        args=_map(node.args,argument);kwargs=_map(node.kwargs,argument)
        folded_node=False;value=None
        if node.op in ('call_function','call_method'):
            try:folded_node,value=fold(node,args,kwargs)
            except (TypeError,ValueError,ArithmeticError,IndexError) as exc:
                raise ShapeSpecializationError(f'{node.name}: invalid shape metadata operation') from exc
        if folded_node:
            if 'val' in node.meta:bindings.bind(node.meta['val'],value,node.name);bindings.solve()
            constants[node]=value;folded.append(node.name);continue
        cloned=graph.create_node(node.op,node.target,args,kwargs,node.name,node.type)
        cloned.meta=dict(node.meta)
        if 'val' in node.meta:cloned.meta['val']=_map(node.meta['val'],materialize)
        elif node.op=='get_attr':cloned.meta['val']=_map(sources[node],materialize)
        if node.op in ('placeholder','get_attr') and not any(isinstance(v,torch.Tensor) for v in _leaves(sources[node])):
            constants[node]=sources[node]
        env[node]=cloned
    result=GraphModule(gm,graph)
    result.meta=dict(gm.meta)
    return result,tuple(folded)
