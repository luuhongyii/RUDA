"""Conservative ATen-to-StaticGraph regions used *after* AOTAutograd.

Tracing never dereferences FakeTensor data. Native graphs are allocated lazily
from real invocation tensors. No operation is retried after a dispatch failure.
"""
from collections import OrderedDict
import math
import operator
from threading import RLock
import sys

import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.node import map_arg

from ._graph_spec import GraphOp, MAX_NODES, MAX_TENSORS, TensorSpec, plan_layout, UNARY_CODES, DTYPE_BYTES, INDEX_CODES, TRANSFORM_CODES, PREDICATE_CODES, FACTORY_CODES, MASK_CODES


class NativeCoverageError(RuntimeError):
    """A required-native graph contains unsupported operations or input metadata."""


# Exact overloads only; no mutations, RNG, alias-changing views or implicit casts.
_TARGETS = {getattr(torch.ops.aten, name).default: name for name in UNARY_CODES
            if name not in ('copy',)}
_TARGETS.update({torch.ops.aten.clone.default: 'copy', torch.ops.aten.add.Tensor: 'add',
    torch.ops.aten.sub.Tensor: 'sub', torch.ops.aten.mul.Tensor: 'mul',
    torch.ops.aten.div.Tensor: 'div', torch.ops.aten.mm.default: 'mm',
    torch.ops.aten.bmm.default: 'bmm', torch.ops.aten.silu_backward.default: 'silu_backward',
    torch.ops.aten.sigmoid_backward.default: 'sigmoid_backward',
    torch.ops.aten.tanh_backward.default: 'tanh_backward',
    torch.ops.aten._softmax.default: 'softmax', torch.ops.aten._log_softmax.default: 'log_softmax',
    torch.ops.aten._softmax_backward_data.default: 'softmax_backward',
    torch.ops.aten._log_softmax_backward_data.default: 'log_softmax_backward',
    torch.ops.aten.threshold_backward.default: 'threshold_backward',
    torch.ops.aten.sum.default: 'sum_all', torch.ops.aten.mean.default: 'mean_all',
    torch.ops.aten.sum.dim_IntList: 'sum_keepdim', torch.ops.aten.mean.dim: 'mean_keepdim',
    torch.ops.aten.add.Scalar: 'add', torch.ops.aten.sub.Scalar: 'sub',
    torch.ops.aten.mul.Scalar: 'mul', torch.ops.aten.div.Scalar: 'div'})
_TARGETS.update({torch.ops.aten.view_copy.default: 'reshape_copy',
    torch.ops.aten.permute_copy.default: 'permute_copy', torch.ops.aten._to_copy.default: 'cast',
    torch.ops.aten.expand_copy.default: 'expand_copy'})
_TARGETS.update({torch.ops.aten.rms_norm.default: 'rms_norm',
    torch.ops.aten.softmax.int: 'softmax_public',torch.ops.aten.log_softmax.int: 'log_softmax_public'})
_TARGETS.update({torch.ops.aten.index_select.default:'index_select',torch.ops.aten.gather.default:'gather',
    torch.ops.aten.embedding.default:'embedding',torch.ops.aten.embedding_dense_backward.default:'embedding_backward',
    torch.ops.aten.gather_backward.default:'gather_backward'})
_TARGETS.update({torch.ops.aten.slice_copy.Tensor:'slice_copy',torch.ops.aten.select_copy.int:'select_copy',
    torch.ops.aten.slice_backward.default:'slice_backward',torch.ops.aten.select_backward.default:'select_backward',
    torch.ops.aten.slice_scatter.default:'slice_scatter',torch.ops.aten.cat.default:'cat'})
_TARGETS.update({torch.ops.aten.masked_fill.Scalar:'masked_fill',torch.ops.aten.where.self:'where'})
_FACTORIES={getattr(torch.ops.aten,name).default:name for name in (
    'full','full_like','zeros','zeros_like','ones','ones_like','new_full','new_zeros','new_ones','scalar_tensor')}
_FACTORIES.update({torch.ops.aten.arange.default:'arange',torch.ops.aten.arange.start:'arange_start',
                   torch.ops.aten.arange.start_step:'arange_start_step'})
_TARGETS.update(_FACTORIES)
for _mask in MASK_CODES:
    _TARGETS[getattr(getattr(torch.ops.aten,_mask),'Tensor' if _mask in ('bitwise_and','bitwise_or','bitwise_xor') else 'default')]=_mask
for _comparison in ('eq','ne','lt','le','gt','ge'):
    _TARGETS[getattr(torch.ops.aten,_comparison).Tensor]=_comparison
    _TARGETS[getattr(torch.ops.aten,_comparison).Scalar]=_comparison+'_scalar'
_TARGETS.update({torch.ops.aten.pow.Tensor_Tensor: 'pow',
    torch.ops.aten.pow.Tensor_Scalar: 'pow',
    torch.ops.aten.floor_divide.default: 'floor_divide',
    torch.ops.aten.floor_divide.Scalar: 'floor_divide',
    torch.ops.aten.fmod.Tensor: 'fmod', torch.ops.aten.fmod.Scalar: 'fmod',
    torch.ops.aten.remainder.Tensor: 'remainder', torch.ops.aten.remainder.Scalar: 'remainder',
    torch.ops.aten.div.Tensor_mode: 'div_mode', torch.ops.aten.div.Scalar_mode: 'div_mode'})
_ALIASES={torch.ops.aten.view.default:'reshape_copy',torch.ops.aten.reshape.default:'reshape_copy',
    torch.ops.aten._unsafe_view.default:'reshape_copy',torch.ops.aten.permute.default:'permute_copy',
    torch.ops.aten.t.default:'transpose_copy',torch.ops.aten.transpose.int:'transpose_copy',
    torch.ops.aten.unsqueeze.default:'reshape_copy',torch.ops.aten.squeeze.dim:'reshape_copy',
    torch.ops.aten.squeeze.default:'reshape_copy',torch.ops.aten.expand.default:'expand_copy'}
_ALIASES.update({torch.ops.aten.slice.Tensor:'slice_copy',torch.ops.aten.select.int:'select_copy'})
_METADATA = {torch.ops.aten.detach.default, torch.ops.aten.alias.default}


def _native_metadata(node):
    if node.op != 'call_function' or node.target not in _METADATA | _ALIASES.keys():
        return False
    if not node.args or not isinstance(node.args[0], Node):
        return False
    source, result = node.args[0].meta.get('val'), node.meta.get('val')
    if not (isinstance(source,torch.Tensor) and isinstance(result,torch.Tensor)
            and source.device.type=='ruda' and source.device==result.device and source.dtype==result.dtype):return False
    if node.target in _METADATA:
        return len(node.args)==1 and not node.kwargs and source.shape==result.shape and source.stride()==result.stride()
    if node.target==torch.ops.aten.reshape.default and not source.is_contiguous():return False
    return True


def _layout_capable():
    return getattr(sys.modules.get(__package__),'_graph_layout_available',False)


def _math_capable():
    return getattr(sys.modules.get(__package__),'_graph_math_available',False)


def _index_capable():
    return getattr(sys.modules.get(__package__),'_graph_index_available',False)


def _factory_capable():
    return getattr(sys.modules.get(__package__),'_graph_factory_available',False)


def _empty_capable():
    return getattr(sys.modules.get(__package__),'_graph_empty_available',False)


def _root_capable():
    return getattr(sys.modules.get(__package__),'_graph_root_available',False)


def _lower_factory(node,kind,args,kwargs,value):
    if not _factory_capable() or not isinstance(value,torch.Tensor) or value.device.type!='ruda' or value.device.index not in (None,0):return None
    if value.layout!=torch.strided or not value.is_contiguous() or any(type(d) is not int or d<(0 if _empty_capable() else 1) for d in value.shape):return None
    if set(kwargs)-{'dtype','layout','device','pin_memory','memory_format','step'}:return None
    if kwargs.get('layout') not in (None,torch.strided) or kwargs.get('pin_memory') not in (None,False):return None
    if kwargs.get('memory_format') not in (None,torch.preserve_format,torch.contiguous_format):return None
    if 'step' in kwargs and kind!='arange_start_step':return None
    based=kind.endswith('_like') or kind.startswith('new_')
    if based:
        if not args or not isinstance(args[0],Node):return None
        anchor=args[0];args=args[1:]
    elif _root_capable():
        anchor=None
    else:
        anchors=[n for n in node.graph.nodes if n.op=='placeholder' and isinstance(n.meta.get('val'),torch.Tensor)
                 and n.meta['val'].device==value.device and all(type(d) is int and d>=(0 if _empty_capable() else 1) for d in n.meta['val'].shape)]
        if not anchors:return None
        anchor=min(anchors,key=lambda n:math.prod(n.meta['val'].shape))
    dtype=str(value.dtype).removeprefix('torch.');step=None
    if kind.startswith('arange'):
        expected={'arange':1,'arange_start':2,'arange_start_step':3}[kind]
        if kind=='arange_start_step' and len(args)==2:args=(*args,kwargs.get('step',1))
        if len(args)!=expected or any(type(v) not in (int,float) or not math.isfinite(v) for v in args):return None
        start,end=(0,args[0]) if kind=='arange' else args[:2]
        step=args[2] if expected==3 else 1
        if step==0 or (step>0 and end<start) or (step<0 and end>start):return None
        if dtype in ('float32','float16','bfloat16'):
            if not _root_capable():return None
            count=math.ceil((float(end)-float(start))/float(step))
            if tuple(value.shape)!=(count,):return None
            return GraphOp('arange_float',node.name,anchor.name if anchor is not None else None,scalar=start,shape=(count,),dtype=dtype,step=step)
        if any(type(v) is not int for v in args):return None
        if step==0 or dtype not in ('int64','int32','int16','int8','uint8'):return None
        count=max(0,-((start-end)//step))
        if tuple(value.shape)!=(count,):return None
        return GraphOp('arange_integer',node.name,anchor.name if anchor is not None else None,scalar=start,shape=tuple(value.shape),dtype=dtype,step=step)
    if kind.endswith('_like'):
        if len(args)!=(1 if kind=='full_like' else 0):return None
        fill=args[0] if args else (1 if kind=='ones_like' else 0)
    elif kind=='scalar_tensor':
        if len(args)!=1:return None
        fill=args[0]
    else:
        if len(args)!=(2 if kind in ('full','new_full') else 1):return None
        if not isinstance(args[0],(tuple,list)) or tuple(args[0])!=tuple(value.shape):return None
        fill=args[1] if len(args)==2 else (1 if kind in ('ones','new_ones') else 0)
    if type(fill) not in (int,float,bool):return None
    return GraphOp('full',node.name,anchor.name if anchor is not None else None,scalar=int(bool(fill)) if dtype=='bool' else int(fill) if type(fill) is bool else fill,shape=tuple(value.shape),dtype=dtype)


def _internal_alias(node, seen=None):
    """Materialize a view only when all consumers are pure, non-aliasing math.

    Returning a view, mutating through it or inspecting its storage/strides stays
    on ordinary device dispatch. No user-observable alias is replaced by a copy.
    """
    seen=set() if seen is None else seen
    if node in seen:return False
    seen.add(node)
    for user in node.users:
        if user.op!='call_function':return False
        if user.target in _ALIASES or _native_metadata(user):
            if not _internal_alias(user,seen.copy()):return False
        elif user.target not in _TARGETS:
            from ._compile_operations import operation
            from ._compile_collectives import definition as collective_definition
            if operation(user) is None and collective_definition(user) is None:return False
    return True


def _rank(node):
    value = node.meta.get('val')
    return value.ndim if isinstance(value, torch.Tensor) else None


def _lower_transform(node,kind,args,kwargs,value):
    if not _index_capable() or not isinstance(value,torch.Tensor):return None
    if kind=='cat':
        if not 1<=len(args)<=2 or set(kwargs)-{'dim'} or not isinstance(args[0],(tuple,list)) or len(args[0])!=2:return None
        if any(not isinstance(n,Node) for n in args[0]):return None
        rank=_rank(args[0][0]);axis=args[1] if len(args)>1 else kwargs.get('dim',0)
        if rank is None or type(axis) is not int or not -rank<=axis<rank:return None
        return GraphOp(kind,node.name,args[0][0].name,args[0][1].name,axis%rank)
    if not args or not isinstance(args[0],Node):return None
    if kind in ('slice_backward','select_backward'):
        required=6 if kind=='slice_backward' else 4
        if len(args)!=required or kwargs or not isinstance(args[1],(tuple,list)) or any(type(d) is not int for d in args[1]):return None
        full=tuple(args[1]);axis=args[2];start=args[3];end=args[4] if required==6 else None;step=args[5] if required==6 else 1
    else:
        scatter=kind=='slice_scatter';offset=2 if scatter else 1
        names=('dim','index') if kind=='select_copy' else ('dim','start','end','step')
        if len(args)>offset+len(names) or set(kwargs)-set(names):return None
        if scatter and (len(args)<2 or not isinstance(args[1],Node)):return None
        source=args[0].meta.get('val')
        if not isinstance(source,torch.Tensor):return None
        full=tuple(source.shape)
        parameters={name:(args[offset+i] if len(args)>offset+i else kwargs.get(name)) for i,name in enumerate(names)}
        axis=parameters['dim'];axis=0 if axis is None else axis
        if kind=='select_copy':start=parameters['index'];end=None;step=1
        else:start=parameters['start'];end=parameters['end'];step=parameters['step'];step=1 if step is None else step
    rank=len(full)
    if type(axis) is not int or not -rank<=axis<rank or type(step) is not int or step!=1:return None
    axis%=rank;length=full[axis]
    if type(length) is not int:return None
    if kind.startswith('select_'):
        if type(start) is not int or not -length<=start<length:return None
        start%=length
    else:
        if start is not None and type(start) is not int or end is not None and type(end) is not int:return None
        start,end,_=slice(start,end,1).indices(length)
        if end<=start and not _empty_capable():return None
    code=start*8+axis
    if code>=2**24:return None
    right=args[1].name if kind=='slice_scatter' else None
    return GraphOp(kind,node.name,args[0].name,right,code,shape=tuple(value.shape))


def _lower(node):
    if node.op != 'call_function' or node.target not in _TARGETS and node.target not in _ALIASES:
        return None
    alias=node.target in _ALIASES
    if alias and (not _layout_capable() or not _internal_alias(node)):
        return None
    value = node.meta.get('val')
    if isinstance(value, torch.Tensor) and not value.is_contiguous() and not alias:
        return None  # Output stride/clone memory-format is observable to users.
    kind, args, kwargs = (_ALIASES[node.target] if alias else _TARGETS[node.target]), node.args, dict(node.kwargs)
    if node.target in _FACTORIES:return _lower_factory(node,kind,args,kwargs,value)
    if kind in TRANSFORM_CODES:return _lower_transform(node,kind,args,kwargs,value)
    if not args or not isinstance(args[0], Node): return None
    left = args[0].name
    if kind in MASK_CODES:
        if not _factory_capable():return None
        if kind in ('tril','triu'):
            if not 1<=len(args)<=2 or set(kwargs)-{'diagonal'}:return None
            diagonal=args[1] if len(args)>1 else kwargs.get('diagonal',0)
            if type(diagonal) is not int:return None
            return GraphOp(kind,node.name,left,scalar=diagonal)
        if kwargs or len(args)!=(1 if kind.endswith('_not') else 2):return None
        if len(args)==2 and not isinstance(args[1],Node):return None
        return GraphOp(kind,node.name,left,None if len(args)==1 else args[1].name)
    if kind in PREDICATE_CODES:
        if not _index_capable() or kwargs:return None
        if kind=='where':
            if len(args)!=3 or any(not isinstance(arg,Node) for arg in args):return None
            return GraphOp(kind,node.name,left,args[1].name,third=args[2].name)
        if kind=='masked_fill':
            if len(args)!=3 or not isinstance(args[1],Node) or type(args[2]) not in (int,float,bool):return None
            return GraphOp(kind,node.name,left,args[1].name,int(args[2]) if type(args[2]) is bool else args[2])
        if len(args)!=2:return None
        if kind.endswith('_scalar'):
            if type(args[1]) not in (int,float,bool):return None
            source=args[0].meta.get('val')
            if not isinstance(source,torch.Tensor) or not source.is_floating_point() and type(args[1]) is float:return None
            return GraphOp(kind,node.name,left,scalar=int(args[1]) if type(args[1]) is bool else args[1])
        if not isinstance(args[1],Node):return None
        return GraphOp(kind,node.name,left,args[1].name)
    if kind in INDEX_CODES:
        if not _index_capable():return None
        if kind=='embedding':
            if not 2<=len(args)<=5 or set(kwargs)-{'padding_idx','scale_grad_by_freq','sparse'} or not isinstance(args[1],Node):return None
            # These parameters affect autograd, not forward lookup. AOT retains
            # padding/frequency/sparse semantics in its separate backward graph.
            return GraphOp(kind,node.name,left,args[1].name)
        if kind=='embedding_backward':
            if len(args)!=5 or kwargs or not isinstance(args[1],Node) or type(args[2]) is not int or type(args[3]) is not int or type(args[4]) is not bool:return None
            if not isinstance(value,torch.Tensor) or tuple(value.shape)!=(args[2],args[0].meta['val'].shape[-1]):return None
            return GraphOp(kind+('_frequency' if args[4] else ''),node.name,left,args[1].name,args[3],shape=tuple(value.shape))
        if kind=='gather_backward':
            if len(args)!=5 or kwargs or not isinstance(args[1],Node) or not isinstance(args[3],Node) or args[4] is not False:return None
            rank=_rank(args[1]);axis=args[2]
            if rank is None or type(axis) is not int or not -rank<=axis<rank:return None
            return GraphOp(kind,node.name,left,args[3].name,axis%rank,shape=tuple(value.shape))
        if not 3<=len(args)<=4 or set(kwargs)-{'sparse_grad'} or not isinstance(args[2],Node):return None
        if kind=='index_select' and (len(args)!=3 or kwargs):return None
        if kind=='gather' and (args[3] if len(args)>3 else kwargs.get('sparse_grad',False)) is not False:return None
        rank=_rank(args[0]);axis=args[1]
        if rank is None or type(axis) is not int or not -rank<=axis<rank:return None
        return GraphOp(kind,node.name,left,args[2].name,axis%rank)
    if kind=='rms_norm':
        if not 2<=len(args)<=4 or set(kwargs)-{'weight','eps'}:return None
        source=args[0].meta.get('val')
        if not isinstance(source,torch.Tensor) or source.ndim==0:return None
        normalized=args[1]
        if not isinstance(normalized,(tuple,list)) or tuple(normalized)!=(source.shape[-1],):return None
        weight=args[2] if len(args)>2 else kwargs.get('weight')
        epsilon=args[3] if len(args)>3 else kwargs.get('eps')
        if epsilon is None:
            if source.dtype not in (torch.float32,torch.float16,torch.bfloat16):return None
            epsilon=torch.finfo(source.dtype).eps
        if type(epsilon) not in (int,float) or not math.isfinite(epsilon) or epsilon<=0:return None
        if weight is not None and not isinstance(weight,Node):return None
        return GraphOp('rms_norm',node.name,left,None if weight is None else weight.name,epsilon)
    if kind in ('softmax_public','log_softmax_public'):
        if not 2<=len(args)<=3 or set(kwargs)-{'dtype'}:return None
        source=args[0].meta.get('val');rank=_rank(args[0]);axis=args[1]
        if not isinstance(source,torch.Tensor) or rank is None or type(axis) is not int or not -rank<=axis<rank:return None
        dtype=args[2] if len(args)>2 else kwargs.get('dtype')
        if dtype not in (None,source.dtype):return None
        return GraphOp(kind.removesuffix('_public'),node.name,left,scalar=axis%rank)
    if kind=='transpose_copy':
        rank=_rank(args[0])
        if rank is None or kwargs:return None
        axes=list(range(rank))
        if node.target==torch.ops.aten.t.default:
            if rank>2 or len(args)!=1:return None
            if rank==2:axes=[1,0]
        else:
            if len(args)!=3 or any(type(d) is not int or not -rank<=d<rank for d in args[1:]):return None
            a,b=args[1]%rank,args[2]%rank
            axes[a],axes[b]=axes[b],axes[a]
        return GraphOp('permute_copy',node.name,left,scalar=sum(d<<(3*i) for i,d in enumerate(axes)),shape=tuple(value.shape))
    if kind in ('reshape_copy','permute_copy','expand_copy','cast'):
        if not _layout_capable():return None
        if not isinstance(value, torch.Tensor) or str(value.dtype).removeprefix('torch.') not in DTYPE_BYTES:
            return None
        if value.dtype not in (torch.float32,torch.float16,torch.bfloat16) and not _index_capable():return None
        if kind == 'cast':
            if len(args) != 1 or set(kwargs)-{'dtype','device','layout','pin_memory','non_blocking','memory_format'}:
                return None
            source = args[0].meta.get('val')
            if not isinstance(source,torch.Tensor) or source.device != value.device or source.layout != value.layout:
                return None
            if kwargs.get('pin_memory',False) or kwargs.get('memory_format') not in (None,torch.preserve_format,torch.contiguous_format):
                return None
            return GraphOp(kind,node.name,left,dtype=str(value.dtype).removeprefix('torch.'))
        if alias and kind=='reshape_copy':
            if kwargs or len(args) not in (1,2):return None
            return GraphOp(kind,node.name,left,shape=tuple(value.shape))
        if kind=='expand_copy' and len(args)==3 and args[2] is False:args=args[:2]
        if len(args) != 2 or kwargs or not isinstance(args[1],(tuple,list)) or any(type(d) is not int for d in args[1]):
            return None
        scalar = 0
        if kind == 'permute_copy':
            rank = _rank(args[0])
            if rank is None or len(args[1]) != rank or any(not -rank<=d<rank for d in args[1]):
                return None
            axes = [d%rank for d in args[1]]
            if len(set(axes)) != rank:
                return None
            scalar = sum(d<<(3*i) for i,d in enumerate(axes))
        return GraphOp(kind,node.name,left,scalar=scalar,shape=tuple(value.shape))
    if kind in ('pow','floor_divide','fmod','remainder','div_mode'):
        if not _math_capable(): return None
        if kind == 'div_mode':
            if len(args) != 2 or set(kwargs)-{'rounding_mode'}: return None
            mode = kwargs.get('rounding_mode')
            if mode not in (None,'floor','trunc'): return None
            if mode is None:
                kind = 'div'
                kwargs = {}
            else:
                kind = 'floor_divide' if mode == 'floor' else 'div_trunc'
                kwargs = {}
        if kind != 'div':
            if len(args) != 2 or kwargs: return None
            source = args[0].meta.get('val')
            if not isinstance(source,torch.Tensor) or source.dtype not in (torch.float32,torch.float16,torch.bfloat16):
                return None
            if isinstance(args[1],Node):
                other = args[1].meta.get('val')
                if not isinstance(other,torch.Tensor) or other.dtype != source.dtype:
                    return None
                return GraphOp(kind,node.name,left,args[1].name)
            scalar = args[1]
            if type(scalar) not in (int,float) or not math.isfinite(scalar): return None
            # Eager integer exponents use the exact i64 power kernel. Do not
            # replace them with an FP32-rounded exponent in a captured graph.
            if kind == 'pow' and type(scalar) is int and scalar not in (0,1,2,3): return None
            return GraphOp(kind+'_scalar',node.name,left,scalar=scalar)
    if kind in ('add', 'sub', 'mul', 'div'):
        if len(args) != 2 or set(kwargs) - ({'alpha'} if kind in ('add','sub') else set()):
            return None
        alpha = kwargs.get('alpha', 1.)
        if type(alpha) not in (int, float) or not math.isfinite(alpha): return None
        if kind == 'sub': alpha = -alpha
        if isinstance(args[1], Node):
            operation = 'add' if kind == 'sub' else kind
            lv,rv = args[0].meta.get('val'),args[1].meta.get('val')
            if isinstance(lv,torch.Tensor) and isinstance(rv,torch.Tensor) and lv.shape != rv.shape:
                if _layout_capable():operation += '_broadcast'
            return GraphOp(operation, node.name, left, args[1].name,
                           alpha if kind in ('add','sub') else 0.)
        if type(args[1]) not in (int,float) or not math.isfinite(args[1]): return None
        scalar = args[1] * alpha if kind in ('add','sub') else args[1]
        return GraphOp(('add' if kind == 'sub' else kind)+'_scalar', node.name, left, scalar=scalar)
    if kind in ('mm', 'bmm', 'silu_backward', 'sigmoid_backward', 'tanh_backward'):
        if len(args) != 2 or not isinstance(args[1], Node) or kwargs: return None
        return GraphOp(kind, node.name, left, args[1].name)
    if kind == 'threshold_backward':
        if len(args) != 3 or not isinstance(args[1], Node) or kwargs: return None
        threshold = args[2]
        if type(threshold) not in (int,float) or not math.isfinite(threshold): return None
        return GraphOp(kind,node.name,left,args[1].name,threshold)
    if kind.startswith(('softmax', 'log_softmax')):
        backward = kind.endswith('_backward')
        if kwargs or len(args) != (4 if backward else 3): return None
        axis = args[2 if backward else 1]; rank = _rank(args[0])
        if type(axis) is not int or rank is None or not -rank <= axis < rank: return None
        if not backward and args[2] is not False: return None  # no half-to-float cast
        if backward:
            value = args[0].meta.get('val')
            if not isinstance(args[1], Node) or args[3] != value.dtype: return None
        return GraphOp(kind, node.name, left, args[1].name if backward else None, axis % rank)
    if kind in ('sum_keepdim', 'mean_keepdim', 'sum_all', 'mean_all'):
        all_dims = kind.endswith('_all')
        if all_dims:
            if len(args) != 1 or set(kwargs)-{'dtype'}: return None
            keepdim = False
        else:
            if not 2 <= len(args) <= 3 or set(kwargs) - {'keepdim','dtype'}: return None
            keepdim = args[2] if len(args) == 3 else kwargs.get('keepdim', False)
        if type(keepdim) is not bool: return None
        if not keepdim and not _layout_capable():return None
        source = args[0].meta.get('val')
        if not isinstance(source,torch.Tensor) or kwargs.get('dtype') not in (None,source.dtype): return None
        rank = _rank(args[0]); dims = list(range(rank)) if all_dims and rank is not None else (args[1] if not all_dims else None)
        if rank is None or not isinstance(dims, (tuple,list)): return None
        if rank == 0:
            if dims and (len(dims)!=1 or dims[0] not in (0,-1)): return None
            return GraphOp('copy',node.name,left)
        dims = list(dims) if dims else list(range(rank))
        if any(type(d) is not int or not -rank <= d < rank for d in dims): return None
        dims = [d % rank for d in dims]
        if len(set(dims)) != len(dims): return None
        operation = kind.removesuffix('_all') if all_dims else kind
        return GraphOp(operation if keepdim else operation.removesuffix('_keepdim'), node.name, left,
                       scalar=sum(1 << d for d in dims))
    if kind == 'copy':
        if len(args) != 1 or set(kwargs) - {'memory_format'}: return None
        if kwargs.get('memory_format') not in (None, torch.preserve_format, torch.contiguous_format): return None
    elif len(args) != 1 or kwargs:
        return None
    return GraphOp(kind, node.name, left)


def _native_available():
    from . import _graph_available
    return _graph_available


def _stream_id():
    from ._streams import current_stream
    return current_stream().stream_id


def _build_graph(inputs, nodes, outputs, device=None):
    from ._graph import StaticGraph
    # AOTAutograd owns differentiation. No hand-written backward bridge is used.
    return StaticGraph(inputs, nodes, outputs=outputs, optimize=True, reuse_workspace=True, device=device)


def _input_spec(value):
    if not isinstance(value, torch.Tensor):
        raise ValueError('native regions require tensor inputs')
    if value.device.type != 'ruda' or value.device.index not in (None, 0):
        raise ValueError('native regions require ruda:0 (no implicit device transfer)')
    if value.layout != torch.strided:
        raise ValueError('native regions require dense strided tensors')
    if value.is_conj() or value.is_neg():
        raise ValueError('native regions do not support unresolved conjugate/negative views')
    return TensorSpec(tuple(value.shape), str(value.dtype).removeprefix('torch.'))


class NativeRegion(torch.nn.Module):
    """A pure region, with bounded fixed-address staging per shape/stream.

    Outputs are cloned before returning: AOT's saved tensors and outstanding
    forwards must never refer to native replay workspace. The reference callable
    is selected BEFORE dispatch and only for known capability/metadata limits.
    """
    def __init__(self, reference, names, nodes, outputs, state, phase, options, device=None):
        super().__init__()
        self.reference = reference
        self.names = tuple(names)
        self.nodes = tuple(nodes)
        self.outputs = tuple(outputs)
        self.state = state
        self.phase = phase
        self.options = options
        self.device=device
        self._cache = OrderedDict()
        self._lock = RLock()
        self._closed = False

    def forward(self, *args):
        with self._lock:
            self.state.check_open()
            if self._closed:
                raise RuntimeError('native region is closed')
            reason = None
            if not _native_available():
                reason = 'native static graph extension unavailable'
            else:
                try:
                    specs = {name: _input_spec(value) for name, value in zip(self.names, args, strict=True)}
                    plan_layout(specs, self.nodes, self.outputs,layout_api=int(_layout_capable()),empty_api=int(_empty_capable()),root_api=int(_root_capable()))
                    if any(node.kind in FACTORY_CODES or node.kind in MASK_CODES for node in self.nodes) and not _factory_capable():
                        raise ValueError('native graph factory extension unavailable')
                    if (any(node.kind in INDEX_CODES or node.kind in TRANSFORM_CODES or node.kind in PREDICATE_CODES for node in self.nodes)
                            or any(spec.dtype not in ('float32','float16','bfloat16') for spec in specs.values())) and not _index_capable():
                        raise ValueError('native graph index extension unavailable')
                    if any(node.kind in ('reshape_copy','permute_copy','cast','expand_copy',
                           'add_broadcast','mul_broadcast','div_broadcast','sum','mean') for node in self.nodes):
                        from . import _graph_layout_available
                        if not _graph_layout_available:
                            raise ValueError('native graph layout extension unavailable')
                    if any(node.kind in ('pow','floor_divide','fmod','remainder','div_trunc',
                           'pow_scalar','floor_divide_scalar','fmod_scalar','remainder_scalar',
                           'div_trunc_scalar') for node in self.nodes) and not _math_capable():
                        raise ValueError('native graph math extension unavailable')
                except (ValueError, TypeError, OverflowError) as exc:
                    reason = str(exc)
            if reason is not None:
                if self.options.native == 'required':
                    raise NativeCoverageError(f'{self.phase}: {reason}')
                self.state.add('region_reference_runs')
                self.state.reason(self.phase, reason)
                return self.reference(*args)

            stream = _stream_id()
            bound = tuple(isinstance(t, torch.nn.Parameter) and self.phase == 'inference' for t in args)
            key = (stream, torch.is_inference_mode_enabled(),
                   tuple((tuple(t.shape), tuple(t.stride()), t.dtype, t.device,
                          t.data_ptr() if direct else None) for t,direct in zip(args,bound,strict=True)))
            entry = self._cache.get(key)
            with torch.no_grad():
                if entry is None:
                    # Reject/evict before allocating: the number of cached native
                    # handles per region never exceeds the documented bound.
                    if len(self._cache) >= self.options.cache_size:
                        old_key = next(iter(self._cache))
                        old_graph, _ = self._cache[old_key]
                        old_graph.close()
                        del self._cache[old_key]
                        self.state.add('native_cache_evictions')
                    buffers = {name: t.detach() if direct else torch.empty_like(t, memory_format=torch.contiguous_format)
                               for name,t,direct in zip(self.names,args,bound,strict=True)}
                    graph = _build_graph(buffers, self.nodes, self.outputs, self.device)
                    entry = graph, buffers
                    self._cache[key] = entry
                    self.state.add('native_builds')
                else:
                    self._cache.move_to_end(key)
                    self.state.add('native_cache_hits')
                graph, buffers = entry
                for buffer,value,direct in zip(buffers.values(),args,bound,strict=True):
                    if not direct:buffer.copy_(value)
                result = graph.replay()
                # Run/clone failures propagate: never execute the region twice.
                owned = tuple(result[name].clone() for name in self.outputs)
                self.state.add('native_replays')
                self.state.add('native_nodes_executed', len(self.nodes))
                return owned

    def close(self):
        with self._lock:
            if not self._closed:
                first_error = None
                for key, (graph, _) in list(self._cache.items()):
                    try:
                        graph.close()
                    except Exception as exc:
                        first_error = first_error or exc
                    else:
                        del self._cache[key]
                if first_error is not None:
                    # Keep failed handles alive so explicit close() can retry.
                    raise first_error
                self._closed = True


def _make_region(gm, nodes, state, phase, options):
    selected = set(nodes)
    dependencies = []
    lowered=[_lower(n) for n in nodes]
    by_name={n.name:n for n in gm.graph.nodes}
    for node in nodes:
        def dependency(arg):
            if arg not in selected and arg not in dependencies:
                dependencies.append(arg)
            return arg
        map_arg((node.args, node.kwargs), dependency)
    for op in lowered:
        if op.kind in FACTORY_CODES and op.left is not None:
            anchor=by_name[op.left]
            if anchor not in selected and anchor not in dependencies:dependencies.append(anchor)
    outputs = [n for n in nodes if any(user not in selected for user in n.users)]
    if not outputs or len(dependencies) + len(nodes) > MAX_TENSORS:
        return None
    graph, env = Graph(), {}
    for node in dependencies:
        env[node] = graph.placeholder(node.name)
    for node in nodes:
        env[node] = graph.node_copy(node, lambda n: env[n])
    graph.output(tuple(env[node] for node in outputs))
    reference = GraphModule(gm, graph)
    region = NativeRegion(reference, [n.name for n in dependencies],
                          lowered, [n.name for n in outputs], state, phase, options,
                          nodes[0].meta['val'].device if not dependencies else None)
    return region, dependencies, outputs


def partition(gm, state, phase, options):
    """Replace consecutive pure eligible operations; keep all others unchanged.

    Conservatively consecutive grouping avoids moving work across mutations or
    RNG calls. Runtime guards also check shape/layout/dtype, not just op names.
    """
    if options.native!='off' and phase=='inference':
        from ._compile_patterns import fuse_normalizations
        gm=fuse_normalizations(gm)
    if options.native!='off' and _layout_capable():
        from ._compile_normalization import expand_normalizations
        gm=expand_normalizations(gm)
    all_nodes = list(gm.graph.nodes)
    calls = [n for n in all_nodes if n.op in ('call_function', 'call_method', 'call_module')]
    from ._compile_operations import OperationRegion,operation,tuple_metadata,host_metadata_nodes
    from ._compile_collectives import make_region as collective_region,definition as collective_definition
    from ._compile_shapes import shape_guard
    operations={n:OperationRegion(n,state,phase,options) for n in calls if options.native!='off' and operation(n) is not None}
    collectives={n:collective_region(n,state,phase,options) for n in calls if options.native!='off' and collective_definition(n) is not None}
    operations.update(collectives)
    candidates = {n for n in calls if n not in operations and _lower(n) is not None}
    host_metadata=host_metadata_nodes(gm).intersection(calls) | {n for n in calls if shape_guard(n)}
    metadata = {n for n in calls if _native_metadata(n) or tuple_metadata(n,operations)} | host_metadata
    if options.native == 'required':
        unsupported = [f'{n.name}: {n.target}' for n in calls if n not in candidates and n not in metadata and n not in operations]
        if unsupported:
            raise NativeCoverageError(f'{phase} is not entirely native: ' + '; '.join(unsupported[:12]))
    groups, pending = [], []
    for node in all_nodes:
        if node in candidates and options.native != 'off':
            pending.append(node)
            if len(pending) == MAX_NODES:
                groups.append(pending)
                pending = []
        elif pending:
            groups.append(pending)
            pending = []
    if pending:
        groups.append(pending)
    minimum = 1 if options.native == 'required' else options.min_native_ops
    regions = []
    for nodes in groups:
        if len(nodes) < minimum:
            continue
        made = _make_region(gm, nodes, state, phase, options)
        if made is None:
            if options.native == 'required':
                raise NativeCoverageError(f'{phase}: region exceeds native tensor limits or has no outputs')
            continue
        regions.append((nodes, made))
    covered = {n for nodes, _ in regions for n in nodes} | operations.keys()
    record = {
        'phase': phase,
        'operator_count': len(calls),
        'native_candidate_ops': len(candidates)+len(operations),
        'planned_native_ops': len(covered),
        'native_metadata_ops': len(metadata-host_metadata),
        'host_metadata_ops': len(host_metadata),
        'native_regions': len(regions)+len(operations),
        'device_dispatch_ops': [str(n.target) for n in calls if n not in covered and n not in metadata],
        'operators': [{'node': n.name, 'target': str(n.target),
                       'execution': 'explicit-host-rng-metadata' if n in collectives and collective_definition(n)[0]=='ruda::attention_rng_state' else
                                    'native-collective-program' if n in collectives else
                                    'guarded-native-operation-graph' if n in operations else
                                    'guarded-native-region' if n in covered else
                                    'explicit-host-metadata' if n in host_metadata else
                                    'native-storage-metadata' if n in metadata else 'device-dispatch'}
                      for n in calls],
    }
    state.record(record)
    if not regions and not operations:
        return gm

    graph, env = Graph(), {}
    starts = {nodes[0]: (nodes, made) for nodes, made in regions}
    root = torch.nn.Module()
    # GraphModule copies referenced parameters/submodules from gm; adding region
    # modules to gm itself would mutate an AOT-owned graph. Use a separate root.
    for name, module in gm.named_children():
        root.add_module(name, module)
    for name, parameter in gm.named_parameters(recurse=False):
        root.register_parameter(name, parameter)
    for name, buffer in gm.named_buffers(recurse=False):
        root.register_buffer(name, buffer)
    for node in all_nodes:
        if node.op == 'get_attr' and '.' not in str(node.target) and not hasattr(root, node.target):
            setattr(root, node.target, getattr(gm, node.target))
    for node in all_nodes:
        if node in operations:
            region=operations[node];name=f'_ruda_native_operation_{len(state.regions)}'
            while hasattr(root,name):
                name = f'{name}_'
            root.add_module(name,region);state.register_region(region)
            env[node]=graph.call_module(name,tuple(env[n] for n in region.dependencies));env[node].meta=dict(node.meta)
        elif node in starts:
            nodes, (region, dependencies, outputs) = starts[node]
            name = f'_ruda_native_region_{len(state.regions)}'
            while hasattr(root, name):
                name += '_'
            root.add_module(name, region)
            state.register_region(region)
            call = graph.call_module(name, tuple(env[n] for n in dependencies))
            for index, output in enumerate(outputs):
                value = graph.call_function(operator.getitem, (call, index))
                value.meta = dict(output.meta)
                env[output] = value
        elif node not in covered:
            env[node] = graph.node_copy(node, lambda n: env[n])
    result = GraphModule(root, graph)
    result.graph.lint()
    result.recompile()
    return result
