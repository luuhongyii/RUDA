"""Keep unused-expert gradients undefined at the AOT backward boundary."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.node import map_arg


@dataclass(frozen=True)
class OptionalGradientOutputs:
    positions: tuple
    count: int
    as_list: bool

    def finish(self, outputs):
        if len(outputs)!=self.count+1:
            raise RuntimeError('optional-gradient backward output contract changed')
        flags=outputs[-1].detach().to('cpu').tolist()
        if len(flags)!=len(self.positions):
            raise RuntimeError('optional-gradient activity count changed')
        result=list(outputs[:-1])
        for index,used in zip(self.positions,flags,strict=True):
            if not used:result[index]=None
        return result if self.as_list else tuple(result)


def extract(gm):
    markers=[node for node in gm.graph.nodes if node.op=='call_function'
             and isinstance(node.target,torch._ops.OpOverload) and node.target._schema.name=='ruda::optional_expert_gradient']
    if not markers:
        return gm,None
    marker=markers[0].target
    from ._compile_native import NativeCoverageError
    graph,env,predicates=Graph(),{},{}
    unary={'aten::alias','aten::detach','aten::clone','aten::_to_copy','aten::to','aten::contiguous',
           'aten::view','aten::_unsafe_view','aten::reshape','aten::t','aten::transpose','aten::permute',
           'aten::squeeze','aten::unsqueeze','aten::expand','aten::sum','aten::neg','aten::slice',
           'aten::narrow','aten::select','aten::as_strided'}

    def predicate(value):
        return predicates.get(value) if isinstance(value,Node) else None

    def merge(kind,left,right):
        if left==right:return left
        if kind=='or' and (left is None or right is None):return None
        if left is None:return right
        if right is None:return left
        return kind,left,right

    for node in gm.graph.nodes:
        if node.op=='output':
            output=node
            break
        if node.op=='call_function' and node.target is marker:
            gradient,active,index=node.args
            if not isinstance(index,int):raise NativeCoverageError('optional expert index must be static')
            state=active.meta.get('val')
            if not isinstance(state,torch.Tensor) or state.dtype!=torch.int32 or state.ndim!=1 or not 0<=index<state.shape[0]:
                raise NativeCoverageError('optional expert gradient requires its captured integer activity vector')
            env[node]=env[gradient]
            predicates[node]=merge('and',predicate(gradient),('flag',active,index))
            continue
        env[node]=graph.node_copy(node,lambda source:env[source]);env[node].meta=dict(node.meta)
        influenced=[arg for arg in node.all_input_nodes if predicate(arg) is not None]
        if not influenced:continue
        name=node.target._schema.name if node.op=='call_function' and isinstance(node.target,torch._ops.OpOverload) else None
        if name in unary:
            predicates[node]=predicate(node.args[0])
        elif name in ('aten::add','aten::sub','aten::mul','aten::div'):
            a,b=node.args[:2]
            if name in ('aten::add','aten::sub'):
                if not isinstance(b,Node) and b==0:predicates[node]=predicate(a)
                elif not isinstance(a,Node) and a==0:predicates[node]=predicate(b)
                else:predicates[node]=merge('or',predicate(a),predicate(b))
            else:predicates[node]=merge('and',predicate(a),predicate(b))
        elif name=='ruda::parallel_region':
            p=predicate(node.args[0])
            predicates[node]=('group',p,node.args[1]) if node.args[2] in (1,2) else p
        elif name=='ruda::tp_replica_sum':
            predicates[node]=('replicas',predicate(node.args[0]),node.args[1],node.args[2])
        else:
            raise NativeCoverageError(f'optional expert gradient needs an explicit activity rule through {node.target}')
    original=output.args[0]
    if not isinstance(original,(tuple,list)):
        raise NativeCoverageError('optional expert gradients require flat AOT backward outputs')
    positions=tuple(index for index,value in enumerate(original) if predicate(value) is not None)
    lowered=[map_arg(value,lambda source:env[source]) for value in original]
    made={}

    def emit(target,args,like,shape):
        value=graph.call_function(target,args)
        reference=like.meta.get('val')
        if not isinstance(reference,torch.Tensor):
            raise NativeCoverageError('optional-gradient metadata lacks its captured tensor shape')
        value.meta['val']=reference.new_empty(shape,dtype=torch.int32)
        return value

    def build(p):
        if p in made:return made[p]
        kind=p[0]
        if kind=='flag':
            source=env[p[1]]
            selected=emit(torch.ops.aten.select.int,(source,0,p[2]),source,())
            result=emit(torch.ops.aten.reshape.default,(selected,[1]),source,(1,))
        elif kind in ('or','and'):
            left,right=build(p[1]),build(p[2])
            operation=torch.ops.aten.maximum.default if kind=='or' else torch.ops.aten.minimum.default
            result=emit(operation,(left,right),left,(1,))
        elif kind=='group':
            source=build(p[1])
            result=emit(torch.ops.ruda.collective_reduce.default,(source,1,p[2]),source,(1,))
        else:
            source=build(p[1])
            cast=graph.call_function(torch.ops.aten._to_copy.default,(source,),{'dtype':torch.float32})
            cast.meta['val']=source.meta['val'].new_empty((1,),dtype=torch.float32)
            summed=graph.call_function(torch.ops.ruda.tp_replica_sum.default,(cast,p[2],p[3]))
            summed.meta['val']=cast.meta['val']
            result=graph.call_function(torch.ops.aten._to_copy.default,(summed,),{'dtype':torch.int32})
            result.meta['val']=source.meta['val']
        made[p]=result
        return result

    if positions:
        flags=[build(predicate(original[index])) for index in positions]
        packed=emit(torch.ops.aten.cat.default,(flags,0),flags[0],(len(flags),))
        graph.output(tuple([*lowered,packed]))
        adapter=OptionalGradientOutputs(positions,len(original),isinstance(original,list))
    else:
        graph.output(lowered if isinstance(original,list) else tuple(lowered))
        adapter=None
    result=GraphModule(gm,graph)
    result.meta=dict(gm.meta)
    return result,adapter
