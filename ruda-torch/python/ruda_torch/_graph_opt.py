"""Pure graph/lifetime planning. Does not execute tensors or provide a CPU fallback.

Validation runs BEFORE pruning so unused invalid operations are still errors.
Single-consumer SiLU/mul and redundant internal copies are fused. Public results never
share a workspace slot. All options are opt-in pending hardware validation.
"""
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import permutations
import math

from ._graph_spec import GraphOp, Layout, NO_WEIGHT, TensorSpec, plan_layout, DTYPE_BYTES, bmm_layout_variants, MAX_NODES, MAX_TENSORS


@dataclass(frozen=True)
class ExecutionPlan:
    layout: Layout
    nodes: tuple[GraphOp, ...]
    storage_roots: tuple[int, ...]
    original_nodes: int
    original_workspace_bytes: int
    eliminated_outputs: tuple[str, ...]
    fused_activations: tuple[str, ...]

    @property
    def workspace_bytes(self) -> int:
        roots = set(self.storage_roots[self.layout.inputs:])
        return sum(_bytes(self.layout.specs[i]) for i in roots)+self.layout.auxiliary_bytes

    @property
    def workspace_allocations(self) -> int:
        return len(set(self.storage_roots[self.layout.inputs:]))


def _bytes(spec: TensorSpec) -> int:
    return math.prod(spec.shape) * DTYPE_BYTES[spec.dtype]


def _reads(node: GraphOp) -> tuple[str, ...]:
    return tuple(name for name in (node.left,node.right,node.third) if name is not None)


def _prune(nodes: tuple[GraphOp, ...], outputs: tuple[str, ...]):
    needed = set(outputs)
    retained = []
    eliminated = []
    for node in reversed(nodes):
        if node.output not in needed:
            eliminated.append(node.output)
            continue
        retained.append(node)
        needed.update(_reads(node))
    return tuple(reversed(retained)), tuple(reversed(eliminated))


def _fuse(nodes: tuple[GraphOp, ...], outputs: tuple[str, ...]):
    # Count edges, not just consumers: mul(a, a) has two uses and is not fused.
    uses = Counter(name for n in nodes for name in _reads(n))
    producers = {n.output: n for n in nodes}
    public = set(outputs)
    replacements = {}
    removed = set()
    for node in nodes:
        if node.kind != 'mul':
            continue
        for activation, up in ((node.left, node.right), (node.right, node.left)):
            producer = producers.get(activation)
            if (producer is not None and producer.kind == 'silu'
                    and uses[producer.output] == 1 and producer.output not in public):
                replacements[node.output] = GraphOp.silu_mul(node.output, producer.left, up)
                removed.add(producer.output)
                break
    result = tuple(replacements.get(n.output, n) for n in nodes if n.output not in removed)
    # Public intermediate outputs and fan-out retain their original storage cast.
    return result, tuple(n.output for n in nodes if n.output in removed)


def _eliminate_copies(nodes, outputs):
    public=set(outputs);aliases={};retained=[];removed=[]
    def root(name):
        while name in aliases:name=aliases[name]
        return name
    from dataclasses import replace
    for node in nodes:
        node=replace(node,left=root(node.left),right=root(node.right),third=root(node.third))
        if node.kind=='copy' and node.output not in public:
            aliases[node.output]=node.left;removed.append(node.output)
        else:retained.append(node)
    return tuple(retained),tuple(removed)


def _layout_factors(dimensions):
    result=[]
    for size,stride in dimensions:
        if size==1:continue
        if result and result[-1][1]==size*stride:
            previous,_=result.pop();result.append((previous*size,stride))
        else:result.append((size,stride))
    return result


def _reshape_factors(dimensions, shape):
    pending=[factor for axis in dimensions for factor in axis if factor[0]!=1]
    result=[];position=0
    for size in shape:
        axis=[];remaining=size
        while remaining>1:
            if position==len(pending):return None
            extent,stride=pending[position]
            if remaining%extent==0:
                axis.append((extent,stride));position+=1;remaining//=extent
            elif extent%remaining==0:
                tail=extent//remaining;axis.append((remaining,tail*stride))
                pending[position]=(tail,stride);remaining=1
            else:return None
        result.append(axis)
    return result if position==len(pending) else None


def _layout_roots(name, producers, specs):
    target=specs[name].shape
    if len(target)!=3 or 0 in target:return
    chain=[];root=name
    while True:
        node=producers.get(root)
        if node is None or node.kind not in ('copy','reshape_copy','permute_copy','expand_copy','slice_copy',
                                             'repeat_copy','repeat_interleave_copy'):break
        if node.kind=='slice_copy' and (specs[root]!=specs[node.left] or int(node.scalar)>>3):break
        chain.append(node);root=node.left;shape=specs[root].shape
        if not 2<=len(shape)<=8 or 0 in shape:continue
        strides=tuple(math.prod(shape[d+1:]) for d in range(len(shape)))
        dimensions=[[] if size==1 else [(size,stride)] for size,stride in zip(shape,strides)]
        current=shape
        for transform in reversed(chain):
            output_shape=specs[transform.output].shape
            if transform.kind=='reshape_copy':
                dimensions=_reshape_factors(dimensions,output_shape)
                if dimensions is None:break
            elif transform.kind=='permute_copy':
                dimensions=[dimensions[(int(transform.scalar)>>(3*d))&7] for d in range(len(current))]
            elif transform.kind=='expand_copy':
                padding=len(output_shape)-len(current)
                dimensions=[[] for _ in range(padding)]+dimensions
                old=(1,)*padding+current
                dimensions=[[(size,0)] if previous==1 and size>1 else axis
                            for previous,size,axis in zip(old,output_shape,dimensions)]
            elif transform.kind=='repeat_copy':
                padding=len(output_shape)-len(current)
                dimensions=[[] for _ in range(padding)]+dimensions
                old=(1,)*padding+current
                dimensions=[([(size//previous,0)] if size!=previous else [])+axis
                            for previous,size,axis in zip(old,output_shape,dimensions)]
            elif transform.kind=='repeat_interleave_copy':
                axis=int(transform.scalar)
                if axis==-1:
                    dimensions=_reshape_factors(dimensions,(math.prod(current),))
                    if dimensions is None:break
                    current=(math.prod(current),);axis=0
                repeats=output_shape[axis]//current[axis]
                dimensions=[part+([(repeats,0)] if d==axis and repeats!=1 else [])
                            for d,part in enumerate(dimensions)]
            current=output_shape
        if dimensions is None:continue
        mapped=[_layout_factors(axis) for axis in dimensions]
        yield root,shape,strides,mapped


def _grouped_bmm_operand(name, producers, specs):
    target=specs[name].shape;best=(name,(1,2))
    for root,shape,strides,mapped in _layout_roots(name,producers,specs):
        for row in range(len(shape)):
            if shape[row]!=target[1] or mapped[1]!=_layout_factors([(shape[row],strides[row])]):continue
            for column in range(len(shape)):
                if column==row or shape[column]!=target[2] or mapped[2]!=_layout_factors([(shape[column],strides[column])]):continue
                batch_axes=[d for d in range(len(shape)) if d not in (row,column)]
                batches=math.prod(shape[d] for d in batch_axes)
                if target[0]%batches:continue
                expected=[(shape[d],strides[d]) for d in batch_axes]+[(target[0]//batches,0)]
                if mapped[0]==_layout_factors(expected):best=(root,(row,column))
    return best


def _encode_bmm_layout(axes, repeat):
    available=list(range(len(axes)));permutation=0
    for axis in axes:
        position=available.index(axis)
        permutation+=position*math.factorial(len(available)-1);available.pop(position)
    return permutation*(len(axes)-1)+repeat


def _lift_matmul_layout_casts(nodes, outputs, specs):
    from dataclasses import replace
    layouts={'copy','reshape_copy','permute_copy','expand_copy','repeat_copy','repeat_interleave_copy'}
    floating={'float32','float16','bfloat16'}
    fp8={'float8_e4m3fn','float8_e5m2','float8_e8m0fnu'}
    matmuls={'mm','mm_rhs_transposed','mm_lhs_transposed','mm_both_transposed',
             'bmm','bmm_rhs_transposed','bmm_lhs_transposed','bmm_both_transposed'}
    producers={n.output:n for n in nodes};needed=set();visited=set()
    for node in nodes:
        if node.kind not in matmuls:continue
        for name in (node.left,node.right):
            while name not in visited:
                visited.add(name);source=producers.get(name)
                if source is None or source.kind not in layouts|{'cast'}:break
                if source.kind=='cast':needed.add(name)
                name=source.left
    result=[];converted={};names=set(specs)
    def unique(prefix):
        name=prefix;index=0
        while name in names:index+=1;name=f'{prefix}_{index}'
        names.add(name);return name
    for node in nodes:
        if (node.output not in needed or specs[node.output].dtype not in floating
                or specs[node.left].dtype not in floating
                and not (specs[node.output].dtype=='float32' and specs[node.left].dtype in fp8)):
            result.append(node);continue
        chain=[];root=node.left
        while root in producers and producers[root].kind in layouts:
            transform=producers[root];chain.append(transform);root=transform.left
        source_size=math.prod(specs[root].shape);target_size=math.prod(specs[node.left].shape)
        if (not chain or source_size>target_size or source_size==target_size
                and any(transform.kind not in ('copy','reshape_copy','permute_copy') for transform in chain)):
            result.append(node);continue
        key=(root,node.dtype)
        if key not in converted:
            name=unique(node.output+'__layout_cast')
            result.append(GraphOp('cast',name,root,dtype=node.dtype))
            specs[name]=TensorSpec(specs[root].shape,node.dtype);converted[key]=name
        source=converted[key]
        for index,transform in enumerate(reversed(chain)):
            name=node.output if index+1==len(chain) else unique(node.output+'__layout')
            result.append(replace(transform,output=name,left=source))
            specs[name]=TensorSpec(specs[transform.output].shape,node.dtype);source=name
    return _prune(tuple(result),outputs)


def _sink_index_casts(nodes, outputs, specs):
    from dataclasses import replace
    public=set(outputs);uses=Counter(name for node in nodes for name in _reads(node))
    producers={node.output:node for node in nodes};replacements={};removed=set();names=set(specs)
    for node in nodes:
        if node.kind not in ('index_select','embedding') or node.scalar!=0:continue
        cast=producers.get(node.left)
        if (cast is None or cast.kind!='cast' or cast.dtype!='float32'
                or cast.output in public or uses[cast.output]!=1
                or specs[cast.left].dtype not in ('float16','bfloat16')
                or len(specs[cast.left].shape)!=2 or len(specs[node.output].shape)!=2):continue
        name=node.output+'__selected_storage';suffix=0
        while name in names:suffix+=1;name=f'{node.output}__selected_storage_{suffix}'
        names.add(name);specs[name]=TensorSpec(specs[node.output].shape,specs[cast.left].dtype)
        replacements[node.output]=(replace(node,output=name,left=cast.left),GraphOp('cast',node.output,name,dtype='float32'))
        removed.add(cast.output)
    result=[]
    for node in nodes:
        if node.output in removed:continue
        result.extend(replacements.get(node.output,(node,)))
    return tuple(result),tuple(node.output for node in nodes if node.output in removed)


def _schedule_matmul_casts(nodes, outputs):
    public=set(outputs);users={};positions={node.output:i for i,node in enumerate(nodes)}
    matmuls={'mm','mm_rhs_transposed','mm_lhs_transposed','mm_both_transposed',
             'bmm','bmm_rhs_transposed','bmm_lhs_transposed','bmm_both_transposed',
             'bmm_permuted','bmm_grouped','bmm_layout'}
    for node in nodes:
        for name in _reads(node):users.setdefault(name,[]).append(node)
    deferred={};producers={node.output:node for node in nodes};groups={}
    def collect(name):
        node=producers.get(name);consumers=users.get(name,())
        if node is None or name in public or len(consumers)!=1:return []
        if (node.kind=='block_scaled_cast' or node.kind=='cast' and node.dtype=='float32'
                or node.kind in ('index_select','embedding') and node.scalar==0):
            chain=collect(node.left);chain.append(name);deferred[name]=node
            return chain
        return []
    for node in nodes:
        if node.kind in matmuls:
            groups[node.output]=sorted(collect(node.left)+collect(node.right),key=positions.__getitem__)
    result=[]
    for node in nodes:
        if node.output in deferred:continue
        if node.kind in matmuls:
            for name in groups[node.output]:result.append(deferred[name])
        result.append(node)
    return tuple(result)


def _layout_bmm_operands(name, producers, specs):
    target=specs[name].shape
    candidates=[(name,_encode_bmm_layout((0,1,2),1),0)]
    for depth,(root,shape,strides,mapped) in enumerate(_layout_roots(name,producers,specs),1):
        best=None
        for row in range(len(shape)):
            if shape[row]!=target[1] or mapped[1]!=_layout_factors([(shape[row],strides[row])]):continue
            for column in range(len(shape)):
                if column==row or shape[column]!=target[2] or mapped[2]!=_layout_factors([(shape[column],strides[column])]):continue
                batch_axes=[d for d in range(len(shape)) if d not in (row,column)]
                batches=math.prod(shape[d] for d in batch_axes)
                if target[0]%batches:continue
                repeats=target[0]//batches
                active=[d for d in batch_axes if shape[d]>1];singletons=[d for d in batch_axes if shape[d]==1]
                for order in permutations(active):
                    for position in (range(len(order)+1) if repeats!=1 else (len(order),)):
                        expected=[(shape[d],strides[d]) for d in order]
                        expected.insert(position,(repeats,0))
                        if mapped[0]!=_layout_factors(expected):continue
                        code=_encode_bmm_layout((*order,*singletons,row,column),position)
                        if best is None or code<best:best=code
        if best is not None:candidates.append((root,best,depth))
    return candidates


def _fuse_layout_matmul(nodes, outputs, specs):
    producers={n.output:n for n in nodes};replacements={};cache={}
    def choices(name):
        if name not in cache:cache[name]=_layout_bmm_operands(name,producers,specs)
        return cache[name]
    for node in nodes:
        if node.kind!='bmm' or any(0 in specs[name].shape for name in (node.left,node.right,node.output)):continue
        best=None
        for left,left_code,left_depth in choices(node.left):
            variants=bmm_layout_variants(len(specs[left].shape))
            for right,right_code,right_depth in choices(node.right):
                if (left,right)==(node.left,node.right):continue
                code=left_code+variants*right_code
                if code>=2**24:continue
                score=(left_depth+right_depth,-code)
                if best is None or score>best[0]:best=(score,left,right,code)
        if best is not None:
            _,left,right,code=best
            replacements[node.output]=GraphOp('bmm_layout',node.output,left,right,code,shape=specs[node.output].shape)
    if not replacements:return nodes,()
    return _prune(tuple(replacements.get(n.output,n) for n in nodes),outputs)


def _fuse_grouped_matmul(nodes, outputs, specs):
    producers={n.output:n for n in nodes};replacements={}
    for node in nodes:
        if node.kind!='bmm' or any(0 in specs[name].shape for name in (node.left,node.right,node.output)):continue
        left,axes_a=_grouped_bmm_operand(node.left,producers,specs)
        right,axes_b=_grouped_bmm_operand(node.right,producers,specs)
        if (left,right)==(node.left,node.right):continue
        code=sum(axis<<(3*d) for d,axis in enumerate((*axes_a,*axes_b)))
        replacements[node.output]=GraphOp('bmm_grouped',node.output,left,right,code,shape=specs[node.output].shape)
    if not replacements:return nodes,()
    return _prune(tuple(replacements.get(n.output,n) for n in nodes),outputs)


def _fuse_transposed_matmul(nodes, outputs, api):
    from dataclasses import replace
    uses=Counter(name for n in nodes for name in _reads(n))
    producers={n.output:n for n in nodes};public=set(outputs)
    replacements={};removed=set()
    for node in nodes:
        if node.kind=='bmm' and api>=3:
            operands=[node.left,node.right];permutations=[];changed=False
            for side,name in enumerate(operands):
                axes=[0,1,2]
                while True:
                    producer=producers.get(name)
                    if (producer is None or producer.kind!='permute_copy' or producer.shape is None
                            or len(producer.shape)!=3 or uses[name]!=1 or name in public):break
                    permutation=[(int(producer.scalar)>>(3*d))&7 for d in range(3)]
                    axes=[permutation[d] for d in axes]
                    removed.add(name);name=producer.left;changed=True
                operands[side]=name;permutations.append(sum(axis<<(3*d) for d,axis in enumerate(axes)))
            if changed:
                identity=permutations==[136,136]
                replacements[node.output]=replace(node,kind='bmm' if identity else 'bmm_permuted',
                    left=operands[0],right=operands[1],scalar=0. if identity else permutations[0]+(permutations[1]<<9))
            continue
        if node.kind not in ('mm','bmm') or (node.kind=='bmm' and api<2):continue
        rank=3 if node.kind=='bmm' else 2
        permutation=80 if rank==3 else 1
        operands=[node.left,node.right];transposed=[False,False]
        for side,name in enumerate(operands):
            if side==0 and api<2:continue
            producer=producers.get(name)
            if (producer is not None and producer.kind=='permute_copy'
                    and producer.scalar==permutation and producer.shape is not None
                    and len(producer.shape)==rank and uses[name]==1 and name not in public):
                operands[side]=producer.left;transposed[side]=True;removed.add(name)
        if any(transposed):
            suffix='both' if all(transposed) else 'lhs' if transposed[0] else 'rhs'
            replacements[node.output]=replace(node,kind=f'{node.kind}_{suffix}_transposed',
                                               left=operands[0],right=operands[1])
    return (tuple(replacements.get(n.output,n) for n in nodes if n.output not in removed),
            tuple(n.output for n in nodes if n.output in removed))


def _fuse_rotary(nodes, outputs, specs):
    producers={n.output:n for n in nodes};replacements={}
    for node in nodes:
        if node.kind not in ('add','add_broadcast') or node.scalar not in (1.,-1.):continue
        direct=producers.get(node.left);rotated=producers.get(node.right)
        if any(n is None or n.kind not in ('mul','mul_broadcast') for n in (direct,rotated)):continue
        source=direct.left;value=specs[source]
        if (value.dtype not in ('float32','float16','bfloat16') or not value.shape
                or value.shape[-1]==0 or value.shape[-1]%2):continue
        cat=producers.get(rotated.left)
        if cat is None or cat.kind!='cat' or cat.scalar!=len(value.shape)-1:continue
        negative=producers.get(cat.left);lower=producers.get(cat.right)
        if negative is None or negative.kind!='neg':continue
        upper=producers.get(negative.left)
        half=value.shape[-1]//2;axis=len(value.shape)-1
        if any(n is None or n.kind!='slice_copy' or n.left!=source
               or n.shape!=(*value.shape[:-1],half) for n in (lower,upper)):continue
        if lower.scalar!=axis or upper.scalar!=axis+(half<<3):continue
        if any(specs[n.output]!=value for n in (direct,rotated,cat,node)):continue
        if any(specs[name].dtype!=value.dtype for name in (direct.right,rotated.right)):continue
        replacements[node.output]=GraphOp('rotary_add' if node.scalar==1. else 'rotary_sub',
            node.output,source,direct.right,third=rotated.right)
    if not replacements:return nodes,()
    return _prune(tuple(replacements.get(n.output,n) for n in nodes),outputs)


def _fuse_product_reductions(nodes, outputs, specs):
    producers={n.output:n for n in nodes};public=set(outputs)
    uses=Counter(name for node in nodes for name in _reads(node));replacements={}
    for node in nodes:
        if node.kind not in ('sum','sum_keepdim','mean','mean_keepdim'):continue
        product=producers.get(node.left)
        if (product is None or product.kind not in ('mul','mul_broadcast')
                or uses[product.output]!=1 or product.output in public):continue
        spec=specs[product.output]
        if not spec.shape or 0 in spec.shape:continue
        replacements[node.output]=GraphOp('mul_'+node.kind,node.output,product.left,product.right,node.scalar)
    if not replacements:return nodes,()
    return _prune(tuple(replacements.get(n.output,n) for n in nodes),outputs)


def _fuse_abs_extrema(nodes, outputs, specs):
    producers={n.output:n for n in nodes};public=set(outputs)
    uses=Counter(name for node in nodes for name in _reads(node));replacements={}
    for node in nodes:
        if node.kind not in ('amax','amax_keepdim'):continue
        producer=producers.get(node.left)
        if producer is None or uses[node.left]!=1 or node.left in public:continue
        if producer.kind=='abs':
            fused=GraphOp('abs_'+node.kind,node.output,producer.left,scalar=node.scalar)
        elif producer.kind=='abs_'+node.kind:
            inner=int(producer.scalar);outer=int(node.scalar)
            if node.kind.endswith('_keepdim'):mask=inner|outer
            else:
                axes=[d for d in range(len(specs[producer.left].shape)) if not inner&(1<<d)]
                mask=inner|sum(1<<axis for d,axis in enumerate(axes) if outer&(1<<d))
            fused=GraphOp('abs_'+node.kind,node.output,producer.left,scalar=mask)
        else:continue
        replacements[node.output]=fused;producers[node.output]=fused
    if not replacements:return nodes,()
    return _prune(tuple(replacements.get(n.output,n) for n in nodes),outputs)


def _fuse_block_scales(nodes, outputs, specs):
    producers={node.output:node for node in nodes};replacements={}
    storage={'float32','float16','bfloat16','float8_e4m3fn','float8_e5m2','float8_e8m0fnu'}
    candidates=set()
    pending=[name for node in nodes if node.kind in ('mm','mm_rhs_transposed','mm_lhs_transposed','mm_both_transposed',
        'bmm','bmm_rhs_transposed','bmm_lhs_transposed','bmm_both_transposed') for name in (node.left,node.right)]
    while pending:
        name=pending.pop()
        if name in candidates:continue
        candidates.add(name);producer=producers.get(name)
        if producer is not None and producer.kind in ('copy','reshape_copy','permute_copy'):
            pending.append(producer.left)
    def unwrap(name, casts=False):
        while name in producers:
            node=producers[name]
            if node.kind in ('copy','reshape_copy'):
                name=node.left
            elif casts and node.kind=='cast' and specs[name].dtype=='float32':
                name=node.left
                if specs[name].dtype!='float32':break
            else:break
        return name
    def block_extent(dimensions, scale):
        block=1;broadcast=False
        for size,extent in zip(dimensions,scale):
            if size==1:continue
            if extent==1:broadcast=True;block*=size
            elif extent!=size or broadcast:return None
        return block
    for node in nodes:
        if node.output not in candidates:continue
        target=specs[node.output]
        if target.dtype!='float32' or len(target.shape)!=2 or 0 in target.shape:continue
        product=producers.get(unwrap(node.output))
        if product is None or product.kind not in ('mul','mul_broadcast'):continue
        shape=specs[product.output].shape
        split=next((axis for axis in range(len(shape)+1) if math.prod(shape[:axis])==target.shape[0]),None)
        if split is None:continue
        for value,scale in ((product.left,product.right),(product.right,product.left)):
            if specs[value].shape!=shape or specs[value].dtype!='float32' or specs[scale].dtype!='float32':continue
            source=unwrap(value,casts=True);scales=unwrap(scale)
            if specs[source].shape!=target.shape or specs[source].dtype not in storage:continue
            if specs[source].dtype=='float32' and specs[value].shape==target.shape:continue
            scale_shape=(1,)*(len(shape)-len(specs[scale].shape))+specs[scale].shape
            if len(scale_shape)!=len(shape):continue
            rows=block_extent(shape[:split],scale_shape[:split]);columns=block_extent(shape[split:],scale_shape[split:])
            if rows is None or columns is None or rows>2**24:continue
            if math.prod(specs[scales].shape)!=target.shape[0]//rows*(target.shape[1]//columns):continue
            replacements[node.output]=GraphOp('block_scaled_cast',node.output,source,scales,float(rows))
            break
    result=tuple(replacements.get(node.output,node) for node in nodes)
    return _prune(result,outputs)


def _fuse_group_masks(nodes, outputs, specs):
    producers={n.output:n for n in nodes};replacements={}
    for node in nodes:
        if node.kind!='masked_fill':continue
        predicate=producers.get(node.right)
        if predicate is None:continue
        if predicate.kind=='eq_scalar' and predicate.scalar==0.:mask=predicate.left
        elif predicate.kind=='logical_not' or predicate.kind=='bitwise_not' and specs[predicate.left].dtype=='bool':
            mask=predicate.left;cast=producers.get(mask)
            if cast is not None and cast.kind=='cast' and cast.dtype=='bool':mask=cast.left
        else:continue
        flat=producers.get(mask)
        if flat is None or flat.kind!='reshape_copy':continue
        expanded=producers.get(flat.left)
        if expanded is None or expanded.kind!='expand_copy':continue
        unsqueezed=producers.get(expanded.left)
        if unsqueezed is None or unsqueezed.kind!='reshape_copy':continue
        groups=specs[unsqueezed.left];source=specs[node.left];shape=specs[expanded.output].shape
        if (len(source.shape)!=2 or len(groups.shape)!=2 or groups.shape[1]==0
                or source.shape[0]!=groups.shape[0] or source.shape[1]==0 or source.shape[1]%groups.shape[1]
                or specs[unsqueezed.output].shape!=(*groups.shape,1)
                or shape!=(*groups.shape,source.shape[1]//groups.shape[1])
                or specs[flat.output].shape!=source.shape or specs[node.output]!=source):continue
        replacements[node.output]=GraphOp('group_masked_fill',node.output,node.left,unsqueezed.left,node.scalar)
    if not replacements:return nodes,()
    return _prune(tuple(replacements.get(n.output,n) for n in nodes),outputs)


def plan_storage(layout: Layout, *, reuse: bool = False) -> tuple[int, ...]:
    """One tensor per slot; only equal shape/dtype, whole-buffer reuse is allowed.

    Inclusive lifetimes forbid reuse at the node performing the previous value's
    final read. A requested result always receives its OWN allocation. Leaf
    values are also kept to the end, matching the native bridge's conservative
    validation when pruning is disabled.
    """
    if type(reuse) is not bool:
        raise TypeError('reuse must be bool')
    count = len(layout.names)
    roots = list(range(count))
    if not reuse:
        return tuple(roots)
    last = list(range(count))
    consumed = set()
    for i in range(len(layout.scalars)):
        out = layout.inputs + i
        op, a, b = layout.words[3*i:3*i+3]
        for value in ((a,) if op in (161,172) else (a,b,int(layout.scalars[i])) if op in (147,201,202) else (a,b)):
            if value != NO_WEIGHT:
                last[value] = max(last[value], out)
                consumed.add(value)
    protected = set(layout.output_indices)
    for i in range(layout.inputs, count):
        if i not in consumed or i in protected:
            last[i] = count
    # root -> inclusive last use of the current occupant. Only scratch roots
    # are inserted; input and public result allocations never enter the pool.
    pool: dict[int, int] = {}
    for i in range(layout.inputs, count):
        if i in protected:
            continue
        selected = next((root for root, end in pool.items()
                         if end < i and layout.specs[root] == layout.specs[i]), None)
        if selected is None:
            selected = i
        roots[i] = selected
        pool[selected] = last[i]
    return tuple(roots)


def prepare_plan(inputs: Mapping[str, TensorSpec], nodes: Sequence[GraphOp], outputs=None,
                 *, optimize: bool = False, reuse_workspace: bool = False,layout_api=0,empty_api=0,root_api=0,matmul_layout_api=0,rotary_api=0,product_reduce_api=0,abs_extremum_api=0,group_mask_api=0) -> ExecutionPlan:
    if type(optimize) is not bool or type(reuse_workspace) is not bool:
        raise TypeError('optimizer options must be booleans')
    nodes = tuple(nodes)
    original = plan_layout(inputs, nodes, outputs,layout_api=layout_api,empty_api=empty_api,root_api=root_api)
    public_names = tuple(original.names[i] for i in original.output_indices)
    original_count = len(nodes)
    eliminated = fused = ()
    if optimize:
        nodes, eliminated = _prune(nodes, public_names)
        nodes, copies = _eliminate_copies(nodes, public_names)
        eliminated += copies
        nodes, fused = _fuse(nodes, public_names)
        if rotary_api:
            nodes, rotary = _fuse_rotary(nodes, public_names, dict(zip(original.names,original.specs)))
            eliminated += rotary
        if product_reduce_api:
            nodes, products = _fuse_product_reductions(nodes, public_names, dict(zip(original.names,original.specs)))
            eliminated += products
        if abs_extremum_api:
            nodes, extrema = _fuse_abs_extrema(nodes, public_names, dict(zip(original.names,original.specs)))
            eliminated += extrema
        if group_mask_api:
            nodes, masks = _fuse_group_masks(nodes, public_names, dict(zip(original.names,original.specs)))
            eliminated += masks
        if matmul_layout_api:
            if matmul_layout_api>=6:
                nodes, scales = _fuse_block_scales(nodes, public_names, dict(zip(original.names,original.specs)))
                eliminated += scales
            if matmul_layout_api>=5:
                specs=dict(zip(original.names,original.specs))
                candidate=nodes;selected=()
                if matmul_layout_api>=6:
                    candidate, selected = _sink_index_casts(candidate, public_names, specs)
                candidate, casts = _lift_matmul_layout_casts(candidate, public_names, specs)
                candidate, layouts = _fuse_layout_matmul(candidate, public_names, specs)
                if len(candidate)<=MAX_NODES and len(inputs)+len(candidate)<=MAX_TENSORS:
                    nodes=candidate
                    eliminated+=tuple(name for name in (*selected,*casts,*layouts) if name in original.names)
            if matmul_layout_api>=4:
                nodes, grouped = _fuse_grouped_matmul(nodes, public_names, specs if matmul_layout_api>=5 else dict(zip(original.names,original.specs)))
                eliminated += grouped
            nodes, transposes = _fuse_transposed_matmul(nodes, public_names, matmul_layout_api)
            eliminated += transposes
            if matmul_layout_api>=6:
                nodes=_schedule_matmul_casts(nodes,public_names)
    layout = plan_layout(inputs, nodes, public_names,layout_api=layout_api,empty_api=empty_api,root_api=root_api)
    roots = plan_storage(layout, reuse=reuse_workspace)
    return ExecutionPlan(layout, nodes, roots, original_count,
                         original.workspace_bytes, eliminated, fused)
