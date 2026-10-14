"""Pure metadata planner for the native static graph. No tensor execution here."""
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
import math
import struct

NO_WEIGHT = 2**32-1
MAX_NODES = 256
MAX_TENSORS = 512
DTYPE_BYTES = {'float32':4,'float16':2,'bfloat16':2,'bool':1,'int64':8,'int32':4,'int16':2,'int8':1,'uint8':1}
INDEX_CODES = {'index_select':134,'gather':135,'embedding':136,
               'embedding_backward':137,'embedding_backward_frequency':138,'gather_backward':139}
TRANSFORM_CODES = {'slice_copy':140,'select_copy':141,'cat':142,
                   'slice_backward':143,'select_backward':144,'slice_scatter':145}
PREDICATE_CODES = {'masked_fill':146,'where':147,'eq':148,'ne':149,'lt':150,'le':151,'gt':152,'ge':153,
                   'eq_scalar':154,'ne_scalar':155,'lt_scalar':156,'le_scalar':157,'gt_scalar':158,'ge_scalar':159}
FACTORY_CODES = {'full':160,'arange_integer':161,'arange_float':172}
MASK_CODES = {'tril':162,'triu':163,'logical_not':164,'logical_and':165,'logical_or':166,
              'logical_xor':167,'bitwise_not':168,'bitwise_and':169,'bitwise_or':170,'bitwise_xor':171}

@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str

@dataclass(frozen=True)
class GraphOp:
    """One immutable output; names refer to inputs or earlier node outputs.

    copy/silu: right=None, scalar=0.0. add: scalar is alpha. mul: scalar=0.0.
    rms_norm: right is an optional weight, scalar is explicit epsilon.
    silu_mul: round SiLU to the storage dtype before multiplication, like two nodes.
    This plan never infers epsilon from dtype or substitutes other operations.
    """
    kind: str
    output: str
    left: str | None = None
    right: str | None = None
    scalar: float = 0.0
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    third: str | None = None
    step: int | float | None = None

    @classmethod
    def add(cls, output, left, right, *, alpha=1.0): return cls('add',output,left,right,alpha)
    @classmethod
    def mul(cls, output, left, right): return cls('mul',output,left,right)
    @classmethod
    def silu(cls, output, value): return cls('silu',output,value)
    @classmethod
    def silu_mul(cls, output, gate, up): return cls('silu_mul',output,gate,up)
    @classmethod
    def copy(cls, output, value): return cls('copy',output,value)
    @classmethod
    def rms_norm(cls, output, value, weight=None, *, eps=1e-5): return cls('rms_norm',output,value,weight,eps)

# API 3 operators. Codes reuse the production pointwise dispatch where possible.
UNARY_CODES = {'copy': 0, 'relu': 3, 'exp': 9, 'log': 10, 'sqrt': 11,
    'rsqrt': 12, 'sigmoid': 13, 'silu': 14, 'tanh': 17, 'sin': 19,
    'cos': 20, 'abs': 21, 'sign': 22, 'floor': 23, 'ceil': 24, 'trunc': 25,
    'reciprocal': 27, 'log1p': 35, 'sinh': 36, 'cosh': 37, 'asinh': 38,
    'acosh': 39, 'atanh': 40, 'hardsigmoid': 43, 'hardswish': 45, 'neg': 108}
BINARY_CODES = {'add': 1, 'mul': 2, 'threshold_backward': 4, 'div': 8, 'silu_backward': 15,
    'sigmoid_backward': 16, 'tanh_backward': 18, 'silu_mul': 101}
SCALAR_CODES = {'add_scalar': 109, 'mul_scalar': 110, 'div_scalar': 111}
MATH_BINARY_CODES = {'pow': 121, 'floor_divide': 122, 'fmod': 123,
                     'remainder': 124, 'div_trunc': 57}
MATH_SCALAR_CODES = {'pow_scalar': 129, 'floor_divide_scalar': 130,
                     'fmod_scalar': 131, 'remainder_scalar': 132, 'div_trunc_scalar': 133}
CODES = {**UNARY_CODES, **BINARY_CODES, **SCALAR_CODES, 'mm': 7, 'bmm': 30,
    'rms_norm': 100, 'softmax': 102, 'log_softmax': 103, 'sum_keepdim': 104,
    'mean_keepdim': 105, 'softmax_backward': 106, 'log_softmax_backward': 107,
    'reshape_copy': 112, 'permute_copy': 113, 'cast': 114, 'expand_copy': 115,
    'add_broadcast': 116, 'mul_broadcast': 117, 'div_broadcast': 118,
    'sum': 119, 'mean': 120, **MATH_BINARY_CODES, **MATH_SCALAR_CODES, **INDEX_CODES, **TRANSFORM_CODES, **PREDICATE_CODES, **FACTORY_CODES, **MASK_CODES}

@dataclass(frozen=True)
class Layout:
    names: tuple[str,...]
    specs: tuple[TensorSpec,...]
    words: tuple[int,...]
    scalars: tuple[float,...]
    output_indices: tuple[int,...]
    inputs: int

    @property
    def workspace_bytes(self):
        return sum(math.prod(s.shape)*DTYPE_BYTES[s.dtype] for s in self.specs[self.inputs:])+self.auxiliary_bytes

    @property
    def auxiliary_bytes(self):
        size=4 if any(code in INDEX_CODES.values() for code in self.words[::3]) else 0
        for i,code in enumerate(self.words[::3]):
            if code in (137,138,139):
                spec=self.specs[self.inputs+i]
                if spec.dtype!='float32':size+=4*math.prod(spec.shape)
                if code==138:size+=4*spec.shape[0]
        return size


def _broadcast_shape(*shapes):
    rank=max(map(len,shapes));result=[1]*rank
    for shape in shapes:
        for d,size in enumerate((1,)*(rank-len(shape))+shape):
            if result[d]!=1 and size!=1 and result[d]!=size:raise ValueError('incompatible graph broadcast dimensions')
            if result[d]==1:result[d]=size
    return tuple(result)


def _scalar_fits(dtype,value):
    if dtype in ('float32','bool'):return True
    if dtype=='float16':return not math.isfinite(value) or abs(value)<=65504
    if dtype=='bfloat16':return not math.isfinite(value) or abs(value)<=float.fromhex('0x1.fep127')
    bits=DTYPE_BYTES[dtype]*8
    low,high=(0,2**bits-1) if dtype=='uint8' else (-2**(bits-1),2**(bits-1)-1)
    return math.isfinite(value) and value.is_integer() and low<=value<=high


def plan_layout(inputs: Mapping[str,TensorSpec], nodes: Sequence[GraphOp], outputs=None, *, layout_api=0, empty_api=0, root_api=0) -> Layout:
    if layout_api not in (0,1):
        raise ValueError('unsupported graph layout extension version')
    if empty_api not in (0,1):
        raise ValueError('unsupported graph empty-tensor extension version')
    if root_api not in (0,1):
        raise ValueError('unsupported graph factory-root extension version')
    def valid_dimensions(shape):
        return all(type(d) is int and (0 if empty_api else 1)<=d<=2**32-1 for d in shape)
    if not isinstance(inputs,Mapping) or not (0 if root_api else 1)<=len(inputs)<=MAX_TENSORS:
        raise ValueError('static graph needs named inputs')
    nodes=tuple(nodes)
    if not 1<=len(nodes)<=MAX_NODES or len(nodes)+len(inputs)>MAX_TENSORS:
        raise ValueError('static graph accepts 1..256 nodes and at most 512 tensors')
    names=list(inputs); specs=list(inputs.values()); ids={name:i for i,name in enumerate(names)}
    for name,spec in inputs.items():
        if not isinstance(name,str) or not name or not isinstance(spec,TensorSpec):
            raise TypeError('expected nonempty names and TensorSpec values')
        if spec.dtype not in DTYPE_BYTES:
            raise ValueError('unsupported static graph storage dtype')
        if not (0 if layout_api else 1)<=len(spec.shape)<=8 or not valid_dimensions(spec.shape):
            raise ValueError('static graph requires rank 0..8 shapes; zero dimensions require empty-tensor API 1')
        if math.prod(spec.shape)>2**32-1: raise ValueError('32-bit kernel indexing limit')
    words=[]; scalars=[]
    codes=CODES
    for node in nodes:
        if not isinstance(node,GraphOp): raise TypeError('nodes must be GraphOp objects')
        if node.kind not in codes: raise ValueError(f'unsupported graph operator: {node.kind}')
        if not isinstance(node.output,str) or not node.output or node.output in ids:
            raise ValueError('each node requires a new output name; in-place graphs are not supported')
        root=node.kind in FACTORY_CODES and node.left is None and root_api==1
        if not root and node.left not in ids: raise ValueError('node input is missing or refers to a later node')
        a=NO_WEIGHT if root else ids[node.left]; spec=TensorSpec((),node.dtype) if root else specs[a]
        if spec.dtype not in ('float32','float16','bfloat16') and node.kind not in (
                'copy','reshape_copy','permute_copy','cast','expand_copy','index_select','gather','embedding',*TRANSFORM_CODES,*PREDICATE_CODES,*FACTORY_CODES,*MASK_CODES):
            raise ValueError('nonfloating graph storage requires layout/copy/indexing operations')
        if node.third is not None and node.kind!='where':raise ValueError('only where has three tensor inputs')
        if node.step is not None and node.kind not in ('arange_integer','arange_float'):raise ValueError('only arange has a step')
        if isinstance(node.scalar,bool): raise TypeError('graph scalar must be a finite number')
        try: scalar=struct.unpack('<f',struct.pack('<f',float(node.scalar)))[0]
        except (TypeError,ValueError,OverflowError,struct.error) as exc: raise ValueError('invalid FP32 scalar') from exc
        if not math.isfinite(scalar) and node.kind not in (*PREDICATE_CODES,'full'): raise ValueError('graph scalar must be finite in FP32')
        if node.kind in MASK_CODES:
            if node.shape is not None or node.dtype is not None:raise ValueError('mask shape/dtype are inferred')
            b=a
            if node.kind in ('tril','triu'):
                if node.right is not None or len(spec.shape)<2 or scalar!=node.scalar or not scalar.is_integer() or not -(2**31)<=scalar<2**31:
                    raise ValueError('triangular mask requires matrix dimensions and an exact signed diagonal')
            else:
                if scalar!=0.:raise ValueError('logical/bitwise nodes do not take a scalar')
                if node.kind.endswith('_not'):
                    if node.right is not None:raise ValueError('logical/bitwise not is unary')
                else:
                    if node.right not in ids:raise ValueError('logical/bitwise operation needs two inputs')
                    b=ids[node.right]
                    if specs[b].dtype!=spec.dtype:raise ValueError('logical/bitwise operation requires explicit dtype promotion')
                    spec=TensorSpec(_broadcast_shape(spec.shape,specs[b].shape),spec.dtype)
                if node.kind.startswith('logical_'):spec=TensorSpec(spec.shape,'bool')
                elif spec.dtype not in ('bool','int64','int32','int16','int8','uint8'):raise ValueError('bitwise operation needs integer/bool operands')
        elif node.kind in FACTORY_CODES:
            if node.right is not None or node.dtype not in DTYPE_BYTES or not isinstance(node.shape,tuple):
                raise ValueError('factory needs an explicit output shape/dtype and no right operand')
            if len(node.shape)>8 or not valid_dimensions(node.shape):
                raise ValueError('factory requires a supported rank 0..8 shape')
            spec=TensorSpec(node.shape,node.dtype)
            if node.kind=='full' and spec.dtype=='bool':scalar=float(bool(node.scalar))
            if node.kind!='arange_float' and not _scalar_fits(spec.dtype,scalar):raise ValueError('factory scalar exceeds output dtype')
            if node.kind=='full' and spec.dtype in ('float16','bfloat16') and math.isfinite(scalar) and scalar!=node.scalar:
                raise ValueError('low-precision full requires an exact FP32 scalar to avoid double rounding')
            if spec.dtype not in ('float32','float16','bfloat16','bool') and scalar!=node.scalar:
                raise ValueError('integer factory scalar must be exactly representable in FP32')
            b=a
            if node.kind=='arange_integer':
                if len(spec.shape)!=1 or spec.dtype not in ('int64','int32','int16','int8','uint8'):
                    raise ValueError('arange_integer requires an integer vector output')
                if type(node.step) is not int or not -(2**31)<=node.step<2**31 or node.step==0:
                    raise ValueError('arange_integer requires a nonzero signed 32-bit step')
                end=int(scalar)+max(spec.shape[0]-1,0)*node.step
                bits=DTYPE_BYTES[spec.dtype]*8
                low,high=(0,2**bits-1) if spec.dtype=='uint8' else (-2**(bits-1),2**(bits-1)-1)
                if not low<=end<=high:raise ValueError('arange_integer endpoint exceeds output dtype')
                # Opcode 161 carries the signed step in b, not a tensor edge.
                b=node.step & 0xffffffff
            elif node.kind=='arange_float':
                if root_api!=1 or len(spec.shape)!=1 or spec.dtype not in ('float32','float16','bfloat16'):
                    raise ValueError('arange_float requires factory-root API 1 and an FP32/FP16/BF16 vector')
                if type(node.step) not in (int,float) or not math.isfinite(node.step) or node.step==0:
                    raise ValueError('arange_float requires a finite nonzero step')
                try:b=struct.unpack('<I',struct.pack('<f',node.step))[0]
                except (OverflowError,struct.error) as exc:raise ValueError('arange_float step exceeds FP32 accumulator') from exc
        elif node.kind in PREDICATE_CODES:
            if node.shape is not None or node.dtype is not None:raise ValueError('predicate shape/dtype are inferred from inputs')
            if node.kind=='where':
                if spec.dtype!='bool' or node.right not in ids or node.third not in ids or scalar!=0:raise ValueError('where requires bool condition and two values')
                b=ids[node.right];c=ids[node.third]
                if specs[b].dtype!=specs[c].dtype:raise ValueError('where requires explicit dtype promotion')
                spec=TensorSpec(_broadcast_shape(spec.shape,specs[b].shape,specs[c].shape),specs[b].dtype)
                scalar=float(c)
            elif node.kind=='masked_fill':
                if node.right not in ids:raise ValueError('masked_fill requires a boolean mask')
                b=ids[node.right]
                if specs[b].dtype!='bool' or not _scalar_fits(spec.dtype,scalar):raise ValueError('masked_fill mask/scalar dtype differs')
                if spec.dtype not in ('float32','float16','bfloat16','bool') and scalar!=node.scalar:raise ValueError('integer fill must be exactly representable')
                spec=TensorSpec(_broadcast_shape(spec.shape,specs[b].shape),spec.dtype)
            elif node.kind.endswith('_scalar'):
                if node.right is not None or not _scalar_fits(spec.dtype,scalar):raise ValueError('invalid scalar comparison')
                if spec.dtype not in ('float32','float16','bfloat16') and (scalar!=node.scalar or spec.dtype=='bool' and scalar not in (0.,1.)):
                    raise ValueError('comparison integer scalar cannot be rounded')
                b=a;spec=TensorSpec(spec.shape,'bool')
            else:
                if node.right not in ids or scalar!=0:raise ValueError('comparison needs two tensor inputs')
                b=ids[node.right]
                if specs[b].dtype!=spec.dtype:raise ValueError('comparison requires explicit dtype promotion')
                spec=TensorSpec(_broadcast_shape(spec.shape,specs[b].shape),'bool')
        elif node.kind in TRANSFORM_CODES:
            if scalar!=int(scalar) or scalar!=node.scalar or not 0<=scalar<2**24 or node.dtype is not None:
                raise ValueError('invalid exact graph transform parameter')
            code=int(scalar);axis=code if node.kind=='cat' else code&7;start=code>>3
            if node.kind=='cat':
                if node.right not in ids or node.shape is not None:raise ValueError('cat needs two graph inputs')
                b=ids[node.right];right=specs[b]
                if spec.dtype!=right.dtype or len(spec.shape)!=len(right.shape) or not 0<=axis<len(spec.shape):raise ValueError('invalid cat rank/dtype/axis')
                shape=list(spec.shape)
                for d,(x,y) in enumerate(zip(shape,right.shape)):
                    if d==axis:shape[d]+=y
                    elif x!=y:raise ValueError('cat non-axis dimensions differ')
            else:
                shape=node.shape
                if not isinstance(shape,tuple) or len(shape)>8 or not valid_dimensions(shape):raise ValueError('transform requires an explicit output shape')
                b=a
                if node.kind=='slice_scatter':
                    if node.right not in ids:raise ValueError('slice_scatter needs a source tensor')
                    b=ids[node.right];part=specs[b].shape;full=spec.shape
                    if specs[b].dtype!=spec.dtype or shape!=full:raise ValueError('slice_scatter dtype/full shape differs')
                elif node.right is not None:raise ValueError('slice/select transform requires one input')
                elif node.kind.endswith('_backward'):full,part=shape,spec.shape
                else:full,part=spec.shape,shape
                if not 0<=axis<len(full):raise ValueError('transform axis exceeds rank')
                expected=list(full)
                if node.kind.startswith('select_'):
                    if start>=full[axis]:raise ValueError('select index exceeds source')
                    del expected[axis]
                else:
                    if len(full)!=len(part) or start+part[axis]>full[axis]:raise ValueError('slice exceeds source')
                    expected[axis]=part[axis]
                if tuple(expected)!=part:raise ValueError('slice/select output shape differs')
            spec=TensorSpec(tuple(shape),spec.dtype)
        elif node.kind in INDEX_CODES:
            if node.right not in ids:raise ValueError('missing graph indices')
            b=ids[node.right];indices=specs[b]
            if indices.dtype not in ('int32','int64') or scalar!=int(scalar) or scalar!=node.scalar:
                raise ValueError('indexing needs integer indices and an exactly representable integer parameter')
            if node.dtype is not None:raise ValueError('indexing preserves value dtype')
            if node.kind=='embedding':
                if len(spec.shape)!=2 or scalar!=0 or node.shape is not None:raise ValueError('invalid embedding weight/parameter')
                spec=TensorSpec(indices.shape+(spec.shape[1],),spec.dtype)
            elif node.kind.startswith('embedding_backward'):
                shape=node.shape
                if not isinstance(shape,tuple) or len(shape)!=2 or not valid_dimensions(shape):
                    raise ValueError('embedding backward requires a weight output shape')
                if spec.shape!=indices.shape+(shape[1],) or not -1<=scalar<shape[0]:raise ValueError('embedding backward gradient/padding differs')
                spec=TensorSpec(shape,spec.dtype)
            else:
                shape=node.shape if node.kind=='gather_backward' else spec.shape
                if not isinstance(shape,tuple) or not shape or len(shape)>8 or not valid_dimensions(shape) or not 0<=scalar<len(shape):
                    raise ValueError('invalid graph indexing shape/axis')
                axis=int(scalar)
                if node.kind=='index_select':
                    if len(indices.shape)>1 or node.shape is not None:raise ValueError('index_select needs scalar/vector indices')
                    result=list(shape);result[axis]=math.prod(indices.shape)
                else:
                    if len(indices.shape)!=len(shape) or any(d!=axis and v>shape[d] for d,v in enumerate(indices.shape)):
                        raise ValueError('gather indices exceed input non-indexed dimensions')
                    if node.kind=='gather_backward' and spec.shape!=indices.shape:raise ValueError('gather backward gradient differs')
                    if node.kind=='gather' and node.shape is not None:raise ValueError('gather shape is determined by indices')
                    result=shape if node.kind=='gather_backward' else indices.shape
                spec=TensorSpec(tuple(result),spec.dtype)
        elif node.kind in ('reshape_copy', 'permute_copy', 'cast', 'expand_copy'):
            if node.right is not None:
                raise ValueError('layout operators have one input')
            b = a
            if node.kind == 'cast':
                if node.dtype not in DTYPE_BYTES or node.shape is not None or scalar != 0:
                    raise ValueError('cast requires an explicit native dtype and zero scalar')
                spec = TensorSpec(spec.shape, node.dtype)
            else:
                shape = node.shape
                if shape is None or not isinstance(shape, tuple) or len(shape)>8 or not valid_dimensions(shape):
                    raise ValueError('layout nodes require an explicit output shape')
                if node.dtype is not None:
                    raise ValueError('layout operations do not change dtype')
                if node.kind == 'reshape_copy':
                    if math.prod(shape) != math.prod(spec.shape) or scalar != 0:
                        raise ValueError('reshape must preserve element count')
                elif node.kind == 'permute_copy':
                    code = int(scalar)
                    axes = tuple((code >> (3*d)) & 7 for d in range(len(spec.shape)))
                    if code != scalar or code < 0 or code >= 1 << (3*len(spec.shape)) or sorted(axes) != list(range(len(spec.shape))):
                        raise ValueError('invalid packed permutation axes')
                    if shape != tuple(spec.shape[d] for d in axes):
                        raise ValueError('permutation output shape differs')
                else:
                    if len(shape)<len(spec.shape) or scalar != 0 or any(x not in (1,y) for x,y in zip(reversed(spec.shape),reversed(shape))):
                        raise ValueError('expand shape is not broadcast compatible')
                spec = TensorSpec(shape, spec.dtype)
        elif node.kind in ('add_broadcast','mul_broadcast','div_broadcast') or node.kind in MATH_BINARY_CODES:
            if node.right not in ids:
                raise ValueError('missing broadcast input')
            b = ids[node.right]
            right = specs[b]
            if right.dtype != spec.dtype:
                raise ValueError('broadcast arithmetic requires identical dtypes; use an explicit cast')
            shape = []
            left_shape = (1,)*(max(len(spec.shape),len(right.shape))-len(spec.shape))+spec.shape
            right_shape = (1,)*(len(left_shape)-len(right.shape))+right.shape
            for x,y in zip(left_shape,right_shape):
                if x != y and x != 1 and y != 1:
                    raise ValueError('incompatible broadcasting dimensions')
                shape.append(y if x==1 else x)
            if node.kind != 'add_broadcast' and scalar != 0:
                raise ValueError('broadcast multiply/divide requires zero scalar')
            spec = TensorSpec(tuple(shape),spec.dtype)
        elif node.kind in UNARY_CODES:
            if node.right is not None or scalar != 0. or math.copysign(1., scalar) < 0:
                raise ValueError('unary operation requires no right input and canonical zero scalar')
            b = a
        elif node.kind in SCALAR_CODES:
            if node.right is not None:
                raise ValueError('scalar operation has no right tensor')
            b = a
        elif node.kind in MATH_SCALAR_CODES:
            if node.right is not None or node.shape is not None or node.dtype is not None:
                raise ValueError('scalar math requires one input and no layout/dtype change')
            b = a
        elif node.kind in BINARY_CODES:
            if node.right not in ids: raise ValueError('missing second graph input')
            b = ids[node.right]
            if specs[b] != spec: raise ValueError('no broadcasting or mixed-dtype promotion in static graph')
            if node.kind not in ('add','threshold_backward') and (scalar != 0. or math.copysign(1., scalar) < 0):
                raise ValueError('binary scalar must be canonical zero')
        elif node.kind in ('mm', 'bmm'):
            if node.right not in ids: raise ValueError('missing matrix input')
            b = ids[node.right]; right = specs[b]
            rank = 2 if node.kind == 'mm' else 3
            if len(spec.shape) != rank or len(right.shape) != rank or spec.dtype != right.dtype:
                raise ValueError('matmul requires equal-dtype rank-2/rank-3 inputs')
            if spec.shape[-1] != right.shape[-2] or spec.shape[:-2] != right.shape[:-2]:
                raise ValueError('matmul dimensions/batches do not match')
            if scalar != 0. or math.copysign(1., scalar) < 0:
                raise ValueError('matmul scalar must be canonical zero')
            spec = TensorSpec(spec.shape[:-1] + (right.shape[-1],), spec.dtype)
            if math.prod(spec.shape) > 2**32-1: raise ValueError('matmul indexing limit')
        elif node.kind in ('softmax', 'log_softmax', 'softmax_backward', 'log_softmax_backward'):
            if not scalar.is_integer() or not 0 <= scalar < len(spec.shape):
                raise ValueError('softmax axis must be canonical and in range')
            b = a
            if node.kind.endswith('_backward'):
                if node.right not in ids or specs[ids[node.right]] != spec:
                    raise ValueError('softmax backward output/gradient mismatch')
                b = ids[node.right]
            elif node.right is not None:
                raise ValueError('softmax forward has no right tensor')
        elif node.kind in ('sum_keepdim', 'mean_keepdim', 'sum', 'mean'):
            # Bit mask, not a shape guessed from the resulting singleton axes.
            mask = int(scalar)
            if scalar != mask or not 0 < mask < (1 << len(spec.shape)) or node.right is not None:
                raise ValueError('reduction requires a nonempty in-range dimension mask')
            b = a
            if node.kind in ('sum','mean'):
                spec = TensorSpec(tuple(size for d,size in enumerate(spec.shape) if not mask & (1<<d)),spec.dtype)
            else:
                spec = TensorSpec(tuple(1 if mask & (1 << d) else size
                                        for d, size in enumerate(spec.shape)), spec.dtype)
        else:
            if not spec.shape:
                raise ValueError('RMSNorm requires a feature axis')
            if scalar <= 0: raise ValueError('RMSNorm epsilon must remain positive after FP32 conversion')
            if math.prod(spec.shape) and math.prod(spec.shape[:-1]) > (2**32-1)//32: raise ValueError('RMSNorm row grid limit')
            b = NO_WEIGHT
            if node.right is not None:
                if node.right not in ids: raise ValueError('missing RMSNorm weight')
                b = ids[node.right]
                if specs[b] != TensorSpec((spec.shape[-1],), spec.dtype):
                    raise ValueError('RMSNorm needs same-dtype last-axis weight')
        if len(spec.shape)>8 or not valid_dimensions(spec.shape) or math.prod(spec.shape)>2**32-1:
            raise ValueError('graph output exceeds 32-bit indexing')
        words.extend((codes[node.kind],a,b)); scalars.append(scalar)
        ids[node.output]=len(names); names.append(node.output); specs.append(spec)
    if outputs is None: outputs=(nodes[-1].output,)
    if isinstance(outputs,str): raise TypeError('outputs must be a sequence of names, not a string')
    outputs=tuple(outputs)
    if not outputs or len(set(outputs))!=len(outputs) or any(x not in ids or ids[x]<len(inputs) for x in outputs):
        raise ValueError('outputs must be unique produced node names')
    return Layout(tuple(names),tuple(specs),tuple(words),tuple(scalars),tuple(ids[n] for n in outputs),len(inputs))
