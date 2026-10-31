"""Pure metadata planner for the native static graph. No tensor execution here."""
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
import math
import struct

NO_WEIGHT = 2**32-1
MAX_NODES = 256
MAX_TENSORS = 512
FP8_DTYPES = frozenset(('float8_e4m3fn','float8_e5m2','float8_e8m0fnu'))
DTYPE_BYTES = {'float32':4,'float16':2,'bfloat16':2,'bool':1,'int64':8,'int32':4,'int16':2,'int8':1,'uint8':1,
               **dict.fromkeys(FP8_DTYPES,1)}
INDEX_CODES = {'index_select':134,'gather':135,'embedding':136,
               'embedding_backward':137,'embedding_backward_frequency':138,'gather_backward':139}
TRANSFORM_CODES = {'slice_copy':140,'select_copy':141,'cat':142,
                   'slice_backward':143,'select_backward':144,'slice_scatter':145}
PREDICATE_CODES = {'masked_fill':146,'where':147,'eq':148,'ne':149,'lt':150,'le':151,'gt':152,'ge':153,
                   'eq_scalar':154,'ne_scalar':155,'lt_scalar':156,'le_scalar':157,'gt_scalar':158,'ge_scalar':159}
FACTORY_CODES = {'full':160,'arange_integer':161,'arange_float':172}
MASK_CODES = {'tril':162,'triu':163,'logical_not':164,'logical_and':165,'logical_or':166,
              'logical_xor':167,'bitwise_not':168,'bitwise_and':169,'bitwise_or':170,'bitwise_xor':171}

def bmm_layout_variants(rank):
    return math.factorial(rank)*(rank-1) if 2<=rank<=8 else 0


def bmm_layout_axes(rank, code):
    if type(code) is not int or not 0<=code<bmm_layout_variants(rank):
        raise ValueError('invalid BMM layout encoding')
    permutation,repeat=divmod(code,rank-1);available=list(range(rank));axes=[]
    for remaining in range(rank,0,-1):
        position,permutation=divmod(permutation,math.factorial(remaining-1))
        axes.append(available.pop(position))
    return tuple(axes),repeat

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
ACTIVATION_CODES = {'gelu':174,'gelu_tanh':175,'gelu_backward':176,'gelu_tanh_backward':177,
    'leaky_relu':178,'leaky_relu_backward':179,'clamp_min':180,'clamp_max':181,
    'log_sigmoid':182,'minimum':183,'maximum':184,'mish':185,'mish_backward':186,
    'softplus':187,'softplus_backward':188}
EXTREMUM_CODES = {'amax_keepdim':189,'amin_keepdim':190,'amax':191,'amin':192}
ABS_EXTREMUM_CODES = {'abs_amax_keepdim':208,'abs_amax':209}
MATMUL_CODES = {'mm':7,'bmm':30,'mm_rhs_transposed':173,'bmm_rhs_transposed':196,
    'bmm_lhs_transposed':197,'bmm_both_transposed':198,'mm_lhs_transposed':199,'mm_both_transposed':200}
ROTARY_CODES = {'rotary_add':201,'rotary_sub':202}
PRODUCT_REDUCE_CODES = {'mul_sum_keepdim':203,'mul_sum':204,'mul_mean_keepdim':205,'mul_mean':206}
STRIDED_TRANSFORM_CODES = {'slice_strided_copy':212,'slice_strided_backward':213,'slice_strided_scatter':214,
    'repeat_copy':215,'repeat_interleave_copy':216}
CODES = {**UNARY_CODES, **BINARY_CODES, **SCALAR_CODES, **MATMUL_CODES,
    **ROTARY_CODES, **PRODUCT_REDUCE_CODES, **ABS_EXTREMUM_CODES, **STRIDED_TRANSFORM_CODES,
    'bmm_permuted':207,'group_masked_fill':210,'bmm_grouped':211,'bmm_layout':217,'block_scaled_cast':218,
    'pack_nibbles':193, 'segment_sum':194, 'segment_sum_exp':195,
    'rms_norm': 100, 'softmax': 102, 'log_softmax': 103, 'sum_keepdim': 104,
    'mean_keepdim': 105, 'softmax_backward': 106, 'log_softmax_backward': 107,
    'reshape_copy': 112, 'permute_copy': 113, 'cast': 114, 'expand_copy': 115,
    'add_broadcast': 116, 'mul_broadcast': 117, 'div_broadcast': 118,
    'sum': 119, 'mean': 120, **MATH_BINARY_CODES, **MATH_SCALAR_CODES, **INDEX_CODES, **TRANSFORM_CODES, **PREDICATE_CODES, **FACTORY_CODES, **MASK_CODES, **ACTIVATION_CODES, **EXTREMUM_CODES}

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
        if (spec.dtype in FP8_DTYPES or node.dtype in FP8_DTYPES
                or any(name in ids and specs[ids[name]].dtype in FP8_DTYPES for name in (node.right,node.third))):
            if node.kind not in ('copy','reshape_copy','permute_copy','cast','expand_copy','block_scaled_cast'):
                raise ValueError('FP8 graph storage requires explicit cast or layout operations')
        if spec.dtype not in ('float32','float16','bfloat16') and node.kind not in (
                'copy','reshape_copy','permute_copy','cast','expand_copy','block_scaled_cast','pack_nibbles','index_select','gather','embedding',*STRIDED_TRANSFORM_CODES,*TRANSFORM_CODES,*PREDICATE_CODES,*FACTORY_CODES,*MASK_CODES):
            raise ValueError('nonfloating graph storage requires layout/copy/indexing operations')
        if node.third is not None and node.kind not in ('where',*ROTARY_CODES):raise ValueError('operator does not accept a third tensor input')
        if node.step is not None and node.kind not in ('arange_integer','arange_float'):raise ValueError('only arange has a step')
        if isinstance(node.scalar,bool): raise TypeError('graph scalar must be a finite number')
        try: scalar=struct.unpack('<f',struct.pack('<f',float(node.scalar)))[0]
        except (TypeError,ValueError,OverflowError,struct.error) as exc: raise ValueError('invalid FP32 scalar') from exc
        if not math.isfinite(scalar) and node.kind not in (*PREDICATE_CODES,'full','group_masked_fill'): raise ValueError('graph scalar must be finite in FP32')
        if node.kind=='block_scaled_cast':
            if (node.right not in ids or node.shape is not None or node.dtype is not None
                    or len(spec.shape)!=2 or 0 in spec.shape
                    or spec.dtype not in ('float32','float16','bfloat16',*FP8_DTYPES)
                    or scalar!=node.scalar or not scalar.is_integer() or not 1<=scalar<=2**24):
                raise ValueError('invalid block-scaled cast input or row block')
            b=ids[node.right];scale=specs[b];rows=int(scalar);count=math.prod(scale.shape)
            if (scale.dtype!='float32' or spec.shape[0]%rows or count==0
                    or count%(spec.shape[0]//rows) or spec.shape[1]%(count//(spec.shape[0]//rows))):
                raise ValueError('invalid block-scaled cast scale grid')
            spec=TensorSpec(spec.shape,'float32')
        elif node.kind in STRIDED_TRANSFORM_CODES:
            shape=node.shape;b=a
            if (not isinstance(shape,tuple) or len(shape)>8 or not valid_dimensions(shape)
                    or node.dtype is not None or scalar!=node.scalar or not scalar.is_integer()):
                raise ValueError('invalid strided transform metadata')
            if node.kind=='repeat_copy':
                if node.right is not None or scalar!=0 or math.copysign(1.,scalar)<0 or len(shape)<len(spec.shape):raise ValueError('invalid repeat rank/scalar')
                source=(1,)*(len(shape)-len(spec.shape))+spec.shape
                if any((size!=0 if old==0 else size%old!=0) for old,size in zip(source,shape)):
                    raise ValueError('repeat dimensions must be multiples of the input')
            elif node.kind=='repeat_interleave_copy':
                if node.right is not None or not -1<=scalar<len(spec.shape):raise ValueError('invalid interleave axis')
                if scalar==-1:
                    size=math.prod(spec.shape)
                    if len(shape)!=1 or (shape[0]!=0 if size==0 else shape[0]%size!=0):raise ValueError('invalid flat interleave shape')
                else:
                    axis=int(scalar)
                    if len(shape)!=len(spec.shape) or any(d!=axis and size!=spec.shape[d] for d,size in enumerate(shape)):
                        raise ValueError('invalid interleave dimensions')
                    if (shape[axis]!=0 if spec.shape[axis]==0 else shape[axis]%spec.shape[axis]!=0):
                        raise ValueError('invalid interleave extent')
            else:
                if not 0<=scalar<2**24:raise ValueError('invalid strided slice encoding')
                code=int(scalar);axis=code&7;start=(code>>3)&2047;step=code>>14
                if step==0:raise ValueError('strided slice requires a positive step')
                if node.kind=='slice_strided_scatter':
                    if node.right not in ids:raise ValueError('missing strided scatter input')
                    b=ids[node.right];part=specs[b].shape;full=spec.shape
                    if specs[b].dtype!=spec.dtype or shape!=full:raise ValueError('invalid strided scatter output/dtype')
                elif node.right is not None:raise ValueError('strided slice is unary')
                elif node.kind=='slice_strided_backward':full,part=shape,spec.shape
                else:full,part=spec.shape,shape
                if (len(full)!=len(part) or not 0<=axis<len(full) or start>full[axis]
                        or any(d!=axis and size!=part[d] for d,size in enumerate(full))
                        or part[axis] and start+(part[axis]-1)*step>=full[axis]):
                    raise ValueError('strided slice exceeds source')
            spec=TensorSpec(shape,spec.dtype)
        elif node.kind=='group_masked_fill':
            if node.right not in ids or node.shape is not None or node.dtype is not None or len(spec.shape)!=2:raise ValueError('group masking needs a matrix and group indicators')
            b=ids[node.right];mask=specs[b]
            if (len(mask.shape)!=2 or mask.shape[0]!=spec.shape[0] or mask.shape[1]==0
                    or spec.shape[1]==0 or spec.shape[1]%mask.shape[1] or not _scalar_fits(spec.dtype,scalar)):
                raise ValueError('invalid group mask shape or fill scalar')
        elif node.kind in ('bmm_grouped','bmm_layout'):
            if node.right not in ids or node.dtype is not None or node.shape is None or len(node.shape)!=3:
                raise ValueError('grouped BMM requires two inputs and a rank-3 output shape')
            b=ids[node.right];right=specs[b]
            if (not valid_dimensions(node.shape) or 0 in node.shape or not 2<=len(spec.shape)<=8
                    or not 2<=len(right.shape)<=8 or 0 in spec.shape or 0 in right.shape or spec.dtype!=right.dtype
                    or scalar!=node.scalar or not scalar.is_integer() or not 0<=scalar<(2**24 if node.kind=='bmm_layout' else 4096)):
                raise ValueError('invalid grouped BMM dtype/shape/axes')
            if node.kind=='bmm_layout':
                variants=bmm_layout_variants(len(spec.shape))
                right_code,left_code=divmod(int(scalar),variants)
                left_axes,_=bmm_layout_axes(len(spec.shape),left_code)
                right_axes,_=bmm_layout_axes(len(right.shape),right_code)
                ar,ac=left_axes[-2:];br,bc=right_axes[-2:]
            else:
                ar,ac,br,bc=tuple((int(scalar)>>(3*d))&7 for d in range(4))
            if ar==ac or br==bc or max(ar,ac)>=len(spec.shape) or max(br,bc)>=len(right.shape):
                raise ValueError('invalid grouped BMM matrix axes')
            if (spec.shape[ac]!=right.shape[br] or node.shape[1:]!=(spec.shape[ar],right.shape[bc])
                    or node.shape[0]*spec.shape[ar]*spec.shape[ac]>2**32-1
                    or node.shape[0]*right.shape[br]*right.shape[bc]>2**32-1
                    or any(node.shape[0]%math.prod(s.shape[d] for d in range(len(s.shape)) if d not in pair)
                           for s,pair in ((spec,(ar,ac)),(right,(br,bc))))):
                raise ValueError('invalid grouped BMM dimensions/batch repeats')
            spec=TensorSpec(node.shape,spec.dtype)
        elif node.kind=='bmm_permuted':
            if node.right not in ids or node.shape is not None or node.dtype is not None:raise ValueError('permuted BMM needs two matrix inputs')
            b=ids[node.right];right=specs[b]
            if len(spec.shape)!=3 or len(right.shape)!=3 or spec.dtype!=right.dtype:raise ValueError('permuted BMM needs equal-dtype rank-3 inputs')
            if scalar!=node.scalar or not scalar.is_integer() or not 0<=scalar<2**18:raise ValueError('invalid BMM permutations')
            code=int(scalar);axes=[tuple((code>>(side*9+d*3))&7 for d in range(3)) for side in range(2)]
            if any(sorted(p)!=[0,1,2] for p in axes):raise ValueError('invalid BMM permutation axes')
            left_shape=tuple(spec.shape[d] for d in axes[0]);right_shape=tuple(right.shape[d] for d in axes[1])
            if left_shape[0]!=right_shape[0] or left_shape[2]!=right_shape[1]:raise ValueError('permuted BMM dimensions do not match')
            spec=TensorSpec((left_shape[0],left_shape[1],right_shape[2]),spec.dtype)
        elif node.kind in PRODUCT_REDUCE_CODES:
            if node.right not in ids or node.shape is not None or node.dtype is not None:raise ValueError('product reduction needs two tensor inputs')
            b=ids[node.right]
            if specs[b].dtype!=spec.dtype:raise ValueError('product reduction requires explicit dtype promotion')
            shape=_broadcast_shape(spec.shape,specs[b].shape);mask=int(scalar)
            if (not shape or 0 in shape or math.prod(shape)>2**32-1 or scalar!=mask
                    or not 0<mask<(1<<len(shape))):raise ValueError('invalid product reduction dimensions/mask')
            if node.kind.endswith('_keepdim'):shape=tuple(1 if mask&(1<<d) else size for d,size in enumerate(shape))
            else:shape=tuple(size for d,size in enumerate(shape) if not mask&(1<<d))
            spec=TensorSpec(shape,spec.dtype)
        elif node.kind in ROTARY_CODES:
            if (node.right not in ids or node.third not in ids or scalar!=0 or math.copysign(1.,scalar)<0
                    or node.shape is not None or node.dtype is not None or not spec.shape
                    or spec.shape[-1]==0 or spec.shape[-1]%2):raise ValueError('rotary requires an even feature axis and three tensor inputs')
            b=ids[node.right];c=ids[node.third]
            if specs[b].dtype!=spec.dtype or specs[c].dtype!=spec.dtype:raise ValueError('rotary requires explicit dtype promotion')
            if _broadcast_shape(spec.shape,specs[b].shape,specs[c].shape)!=spec.shape:raise ValueError('rotary coefficients cannot expand the input shape')
            scalar=float(c)
        elif node.kind in MASK_CODES:
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
        elif node.kind in ('segment_sum','segment_sum_exp'):
            if (spec.dtype!='float32' or not 1<=len(spec.shape)<=7 or spec.shape[-1]==0 or node.right is not None
                    or node.shape is not None or node.dtype is not None or scalar!=0 or math.copysign(1.,scalar)<0):
                raise ValueError('segment scan requires FP32 storage and a nonempty last dimension')
            b=a;spec=TensorSpec((*spec.shape,spec.shape[-1]),spec.dtype)
        elif node.kind=='pack_nibbles':
            if (spec.dtype!='uint8' or not spec.shape or spec.shape[-1]%2 or node.right is not None
                    or node.shape is not None or node.dtype is not None or scalar!=0 or math.copysign(1.,scalar)<0):
                raise ValueError('nibble packing requires uint8 storage with an even last dimension')
            b=a;spec=TensorSpec((*spec.shape[:-1],spec.shape[-1]//2),spec.dtype)
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
        elif node.kind in ACTIVATION_CODES:
            if node.shape is not None or node.dtype is not None:
                raise ValueError('activation shape/dtype are inferred')
            binary=node.kind.endswith('_backward') or node.kind in ('minimum','maximum')
            parameter=node.kind in ('leaky_relu','leaky_relu_backward','clamp_min','clamp_max','softplus','softplus_backward')
            if not parameter and (scalar!=0. or math.copysign(1.,scalar)<0):
                raise ValueError('activation requires canonical zero scalar')
            b=a
            if binary:
                if node.right not in ids:raise ValueError('activation requires a second tensor')
                b=ids[node.right]
                if specs[b].dtype!=spec.dtype:raise ValueError('activation requires explicit dtype promotion')
                if node.kind in ('minimum','maximum'):
                    spec=TensorSpec(_broadcast_shape(spec.shape,specs[b].shape),spec.dtype)
                elif specs[b]!=spec:raise ValueError('activation gradient shape/dtype mismatch')
            elif node.right is not None:raise ValueError('unary activation has no second tensor')
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
        elif node.kind in MATMUL_CODES:
            if node.right not in ids: raise ValueError('missing matrix input')
            b = ids[node.right]; right = specs[b]
            rank = 3 if node.kind.startswith('bmm') else 2
            if len(spec.shape) != rank or len(right.shape) != rank or spec.dtype != right.dtype:
                raise ValueError('matmul requires equal-dtype rank-2/rank-3 inputs')
            lhs_transposed = '_lhs_' in node.kind or '_both_' in node.kind
            rhs_transposed = '_rhs_' in node.kind or '_both_' in node.kind
            if spec.shape[-2 if lhs_transposed else -1] != right.shape[-1 if rhs_transposed else -2] or spec.shape[:-2] != right.shape[:-2]:
                raise ValueError('matmul dimensions/batches do not match')
            if scalar != 0. or math.copysign(1., scalar) < 0:
                raise ValueError('matmul scalar must be canonical zero')
            spec = TensorSpec(spec.shape[:-2] + (spec.shape[-1 if lhs_transposed else -2],
                                                right.shape[-2 if rhs_transposed else -1]), spec.dtype)
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
        elif node.kind in ('sum_keepdim', 'mean_keepdim', 'sum', 'mean', *EXTREMUM_CODES, *ABS_EXTREMUM_CODES):
            # Bit mask, not a shape guessed from the resulting singleton axes.
            mask = int(scalar)
            if scalar != mask or not 0 < mask < (1 << len(spec.shape)) or node.right is not None:
                raise ValueError('reduction requires a nonempty in-range dimension mask')
            b = a
            if node.kind in (*EXTREMUM_CODES,*ABS_EXTREMUM_CODES) and any(size==0 and mask & (1<<d) for d,size in enumerate(spec.shape)):
                raise ValueError('extremum reduction has no identity for an empty axis')
            if node.kind in ('sum','mean','amax','amin','abs_amax'):
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
