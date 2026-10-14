"""Native packed INT2/3/4/5/6/8, FP4 and FP8 tensors and frozen Linear layers."""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
import torch
from torch import nn
from . import _C

FORMATS = {'int2': 2, 'int3': 3, 'int4': 4, 'int5': 5, 'int6': 6,
           'int8': 8, 'fp4_e2m1': 14, 'fp8_e4m3': 18, 'fp8_e5m2': 19}
_FLOATS = (torch.float32, torch.float16, torch.bfloat16)


def _bits(format):
    code = FORMATS[format]
    return 4 if code == 14 else 8 if code >= 18 else code


def _layout(format, group_size, shape):
    if format not in FORMATS or not shape or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError('invalid packed format or shape')
    if shape[-1] < 1 or type(group_size) is not int or group_size < 1:
        raise ValueError('packed width and group_size must be positive')
    values = [FORMATS[format], group_size, shape[-1], math.prod(shape[:-1])]
    if any(n > 0xffffffff for n in values) or math.prod(shape) > 0xffffffff:
        raise ValueError('packed geometry exceeds U32 indexing')
    return values


@torch.library.custom_op('ruda::packed_linear', mutates_args=())
def _linear(x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor,
            offsets: torch.Tensor, group_index: torch.Tensor, layout: list[int],
            transpose: bool) -> torch.Tensor:
    width = layout[2] if transpose else layout[3]
    out = torch.empty((x.shape[0], width), device=x.device, dtype=x.dtype)
    _C.packed_dispatch(3 if transpose else 2, [x, codes, scales, offsets, group_index, out], layout)
    return out


@_linear.register_fake
def _linear_fake(x, codes, scales, offsets, group_index, layout, transpose):
    return x.new_empty((x.shape[0], layout[2] if transpose else layout[3]))


def _linear_context(ctx, inputs, output):
    x, codes, scales, offsets, index, layout, transpose = inputs
    ctx.save_for_backward(codes, scales, offsets, index)
    ctx.layout, ctx.transpose = layout, transpose


def _linear_backward(ctx, grad):
    codes, scales, offsets, index = ctx.saved_tensors
    dx = _linear(grad.contiguous(), codes, scales, offsets, index, ctx.layout, not ctx.transpose)
    return dx, None, None, None, None, None, None


_linear.register_autograd(_linear_backward, setup_context=_linear_context)


@torch.library.custom_op('ruda::packed_linear_tensor_core', mutates_args=())
def _tensor_core_linear(x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor,
                        offsets: torch.Tensor, group_index: torch.Tensor, layout: list[int],
                        transpose: bool) -> torch.Tensor:
    import ctypes
    from . import _native
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError('packed Tensor Core Linear requires FP16/BF16 activations')
    if not hasattr(_native, 'ruda_torch_packed_matmul_api_version') or getattr(_C, 'packed_matmul_api_version', None) != 1:
        raise RuntimeError('packed Tensor Core Linear requires matching rebuilt native libraries')
    _native.ruda_torch_packed_matmul_api_version.restype = ctypes.c_uint32
    if _native.ruda_torch_packed_matmul_api_version() != 1:
        raise RuntimeError('packed matmul API mismatch')
    width = layout[2] if transpose else layout[3]
    out = x.new_empty((x.shape[0], width))
    _C.packed_dispatch(7 if transpose else 6, [x, codes, scales, offsets, group_index, out], layout)
    return out


@_tensor_core_linear.register_fake
def _tensor_core_fake(x, codes, scales, offsets, group_index, layout, transpose):
    return x.new_empty((x.shape[0], layout[2] if transpose else layout[3]))


def _tensor_core_backward(ctx, grad):
    codes, scales, offsets, index = ctx.saved_tensors
    dx = _tensor_core_linear(grad.contiguous(), codes, scales, offsets, index, ctx.layout, not ctx.transpose)
    return dx, None, None, None, None, None, None


_tensor_core_linear.register_autograd(_tensor_core_backward, setup_context=_linear_context)


@torch.library.custom_op('ruda::packed_decode', mutates_args=())
def _decode(codes: torch.Tensor, scales: torch.Tensor, offsets: torch.Tensor,
            group_index: torch.Tensor, layout: list[int], dtype: torch.dtype) -> torch.Tensor:
    out = torch.empty((layout[3], layout[2]), device=codes.device, dtype=dtype)
    _C.packed_dispatch(1, [codes, codes, scales, offsets, group_index, out], layout)
    return out


@_decode.register_fake
def _decode_fake(codes, scales, offsets, group_index, layout, dtype):
    return torch.empty((layout[3], layout[2]), device=codes.device, dtype=dtype)


@torch.library.custom_op('ruda::packed_encode', mutates_args=())
def _encode(x: torch.Tensor, scales: torch.Tensor, offsets: torch.Tensor,
            layout: list[int]) -> torch.Tensor:
    bits = 4 if layout[0] == 14 else 8 if layout[0] >= 18 else layout[0]
    out = torch.empty((layout[3], (layout[2] * bits + 7) // 8), device=x.device, dtype=torch.uint8)
    index = torch.empty(0, device=x.device, dtype=torch.int32)
    _C.packed_dispatch(0, [x, index, scales, offsets, index, out], layout)
    return out


@_encode.register_fake
def _encode_fake(x, scales, offsets, layout):
    bits = 4 if layout[0] == 14 else 8 if layout[0] >= 18 else layout[0]
    return torch.empty((layout[3], (layout[2] * bits + 7) // 8), device=x.device, dtype=torch.uint8)


@torch.library.custom_op('ruda::packed_embedding', mutates_args=())
def _embedding(index: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor,
               offsets: torch.Tensor, group_index: torch.Tensor, layout: list[int],
               dtype: torch.dtype) -> torch.Tensor:
    out = torch.empty((index.numel(),layout[2]),device=index.device,dtype=dtype)
    _C.packed_dispatch(4,[index,codes,scales,offsets,group_index,out],layout)
    return out


@_embedding.register_fake
def _embedding_fake(index,codes,scales,offsets,group_index,layout,dtype):
    return torch.empty((index.numel(),layout[2]),device=index.device,dtype=dtype)


@dataclass(frozen=True)
class PackedQuantizedTensor:
    """Byte codes and per-row/group FP32 scale/offset, without a dense shadow.

    Integer codes are unsigned; symmetric integers use a negative offset.
    `group_index` optionally maps each input column to a group (GPTQ act-order).
    An explicit map may retain a larger parameter table for an input shard.
    Frozen metadata are serialized explicitly by PackedLinear.state_dict().
    """
    codes: torch.Tensor
    scales: torch.Tensor
    offsets: torch.Tensor
    shape: tuple[int, ...]
    format: str
    group_size: int
    group_index: torch.Tensor | None = None

    def __post_init__(self):
        layout = _layout(self.format, self.group_size, self.shape)
        rows, width = layout[3], layout[2]
        groups = (width + self.group_size - 1) // self.group_size
        if self.group_index is None:
            object.__setattr__(self, 'group_index', torch.empty(0, dtype=torch.int32, device=self.codes.device))
        if self.group_index.dtype != torch.int32 or self.group_index.shape not in ((0,), (width,)):
            raise ValueError('group_index must be empty or I32 [input_features]')
        if self.group_index.numel() and self.scales.ndim == 2:
            groups = self.scales.shape[1]
        if groups < 1 or groups > 0xffffffff or rows * groups > 0xffffffff:
            raise ValueError('packed parameter table exceeds U32 indexing')
        if groups != (width + self.group_size - 1) // self.group_size and self.codes.device.type == 'ruda':
            if not getattr(sys.modules.get(__package__), '_packed_indexed_groups_available', False):
                raise RuntimeError('packed TP parameter tables require rebuilt indexed-groups API 1 libraries')
        if self.codes.dtype != torch.uint8 or tuple(self.codes.shape) != (rows, (width * _bits(self.format) + 7) // 8):
            raise ValueError('packed code byte geometry differs from logical shape')
        for value in (self.scales, self.offsets):
            if value.dtype != torch.float32 or tuple(value.shape) != (rows, groups):
                raise ValueError('packed scales and offsets must be FP32 [rows,groups]')
        for value in (self.codes, self.scales, self.offsets, self.group_index):
            if value.device != self.codes.device or not value.is_contiguous() or value.requires_grad:
                raise ValueError('packed tensors require contiguous, frozen, same-device operands')
        if self.group_index.numel() and self.group_index.device.type == 'cpu':
            if (self.group_index < 0).any() or (self.group_index >= groups).any():
                raise ValueError('group_index points outside the parameter groups')

    def to(self, device):
        return PackedQuantizedTensor(*(t.to(device) for t in (self.codes, self.scales, self.offsets)),
                                     self.shape, self.format, self.group_size, self.group_index.to(device))

    def dequantize(self, dtype=torch.float32):
        if dtype not in _FLOATS or self.codes.device.type != 'ruda':
            raise ValueError('native dequantization requires RUDA and FP32/FP16/BF16 output')
        return _decode(self.codes, self.scales, self.offsets, self.group_index,
                       _layout(self.format, self.group_size, self.shape), dtype).reshape(self.shape)


def quantize_packed(x, *, format='int4', group_size=128, symmetric=True,
                    calibration='minmax', percentile=99.9, mse_steps=80,
                    scales=None, zero_points=None, offsets=None):
    """PTQ on the RUDA device. Explicit affine zero points or offsets are accepted.

    Calibration never copies tensor values to CPU. Percentile uses exact sorted
    order statistics with linear interpolation, MSE searches clipping ratios.
    """
    from .quantization_calibration import calibrate_quantization
    if x.device.type != 'ruda' or x.dtype not in _FLOATS:
        raise ValueError('packed quantization requires a RUDA FP32/FP16/BF16 tensor')
    layout = _layout(format, group_size, tuple(x.shape))
    if scales is None:
        if offsets is not None or zero_points is not None:
            raise ValueError('explicit offsets/zero points require explicit scales')
        scales, offsets = calibrate_quantization(x, format=format, group_size=group_size,
            symmetric=symmetric, method=calibration, percentile=percentile, mse_steps=mse_steps)
    else:
        if zero_points is not None and offsets is not None:
            raise ValueError('supply zero_points or offsets, not both')
        if zero_points is not None:
            offsets = -zero_points.to(device=x.device, dtype=torch.float32) * scales
        elif offsets is None:
            offsets = torch.zeros_like(scales)
    scales, offsets = scales.detach().float().contiguous(), offsets.detach().float().contiguous()
    expected = (layout[3], (layout[2] + group_size - 1) // group_size)
    if scales.shape != expected or offsets.shape != expected or scales.device != x.device or offsets.device != x.device:
        raise ValueError('explicit parameters must be [rows,groups] on the input device')
    codes = _encode(x.detach().reshape(layout[3], layout[2]).contiguous(), scales, offsets, layout)
    return PackedQuantizedTensor(codes, scales, offsets, tuple(x.shape), format, group_size)


class PackedLinear(nn.Module):
    """Packed frozen base with native input gradients; LoRA may wrap this module.

    tensor_core decodes weight tiles into the FP16/BF16 activation dtype and
    accumulates in FP32. scalar retains FP32 decoded-weight arithmetic.
    """
    def __init__(self, weight: PackedQuantizedTensor, bias=None, *, strategy='scalar'):
        super().__init__()
        if strategy not in ('scalar', 'tensor_core'):
            raise ValueError('packed Linear strategy must be scalar or tensor_core')
        self.strategy = strategy
        if len(weight.shape) != 2:
            raise ValueError('PackedLinear requires [out,in] weights')
        self.out_features, self.in_features = weight.shape
        self.format, self.group_size = weight.format, weight.group_size
        for name in ('codes', 'scales', 'offsets', 'group_index'):
            self.register_buffer(name, getattr(weight, name))
        if bias is not None and (bias.shape != (self.out_features,) or bias.dtype not in _FLOATS or bias.device != weight.codes.device):
            raise ValueError('packed bias must be floating [out_features] on the weight device')
        self.register_buffer('bias', None if bias is None else bias.detach())

    @property
    def packed_weight(self):
        return PackedQuantizedTensor(self.codes, self.scales, self.offsets,
            (self.out_features, self.in_features), self.format, self.group_size, self.group_index)

    def _apply(self, fn, recurse=True):
        # Module.to(dtype=...) changes compute bias, not packed parameter precision.
        metadata = {name: getattr(self, name) for name in ('scales', 'offsets')}
        super()._apply(fn, recurse)
        for name, original in metadata.items():
            setattr(self, name, original.to(device=getattr(self, name).device, dtype=torch.float32))
        self.packed_weight
        return self

    def get_extra_state(self):
        state = {'version': 1, 'shape': (self.out_features, self.in_features),
                 'format': self.format, 'group_size': self.group_size}
        if self.strategy != 'scalar':
            state.update(version=2, strategy=self.strategy)
        if self.scales.shape[1] != (self.in_features + self.group_size - 1) // self.group_size:
            state.update(version=3, parameter_groups=self.scales.shape[1])
        return state

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('packed state layout differs from the constructed module')

    @classmethod
    def from_linear(cls, linear, *, strategy='scalar', **kwargs):
        return cls(quantize_packed(linear.weight, **kwargs), linear.bias, strategy=strategy)

    def forward(self, x):
        if x.device.type != 'ruda' or x.device != self.codes.device or x.dtype not in _FLOATS or x.shape[-1] != self.in_features:
            raise ValueError('packed Linear input device/dtype/width mismatch')
        shape = x.shape[:-1] + (self.out_features,)
        multiply = _tensor_core_linear if self.strategy == 'tensor_core' else _linear
        result = multiply(x.reshape(-1, self.in_features).contiguous(), self.codes, self.scales,
            self.offsets, self.group_index, _layout(self.format, self.group_size, (self.out_features, self.in_features)), False).reshape(shape)
        return result if self.bias is None else result + self.bias.to(dtype=x.dtype)


class PackedEmbedding(PackedLinear):
    """Frozen embedding table, decoding only requested rows on the RUDA device."""
    def __init__(self,weight,*,padding_idx=None,output_dtype=torch.float16):
        super().__init__(weight)
        if output_dtype not in _FLOATS:
            raise ValueError('embedding output requires FP32/FP16/BF16')
        self.num_embeddings,self.embedding_dim = weight.shape
        self.padding_idx = padding_idx
        self.register_buffer('_output_type',torch.empty(0,device=weight.codes.device,dtype=output_dtype),persistent=False)

    @property
    def output_dtype(self):
        return self._output_type.dtype

    def get_extra_state(self):
        state = super().get_extra_state()
        state.update(padding_idx=self.padding_idx,output_dtype=str(self.output_dtype))
        return state

    def forward(self,index):
        if index.device != self.codes.device or index.device.type != 'ruda' or index.dtype not in (torch.int32,torch.int64):
            raise ValueError('packed embedding needs integer indices on its RUDA device')
        result = _embedding(index.long().reshape(-1).contiguous(),self.codes,self.scales,self.offsets,self.group_index,
            _layout(self.format,self.group_size,(self.num_embeddings,self.embedding_dim)),self.output_dtype)
        return result.reshape(*index.shape,self.embedding_dim)


def scaled_float8_mm(a, b, scale_a, scale_b, *, out_dtype=torch.float16, bias=None, strategy='auto'):
    """Scaled FP8 GEMM; operands are FP8 [M,K] and [K,N].

    Scales are scalar or A [M,1], B [1,N]. auto uses native FP8 instructions
    or an exact FP16 Tensor Core carrier; scalar retains the packed dot kernel.
    native requires FP8 hardware; float16 explicitly selects the carrier.
    """
    supported = (torch.float8_e4m3fn, torch.float8_e5m2)
    if a.dtype not in supported or b.dtype not in supported or a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError('scaled FP8 GEMM requires conformable FP8 matrices')
    if a.device.type != 'ruda' or b.device != a.device or out_dtype not in _FLOATS:
        raise ValueError('scaled FP8 GEMM requires matching RUDA devices and a floating output dtype')
    if scale_a.numel() != 1 and scale_a.shape != (a.shape[0], 1):
        raise ValueError('scale_a must be scalar or [M,1]')
    if scale_b.numel() != 1 and scale_b.shape != (1, b.shape[1]):
        raise ValueError('scale_b must be scalar or [1,N]')
    if scale_a.device != a.device or scale_b.device != a.device:
        raise ValueError('FP8 scales must be on the operand device')
    if scale_a.dtype != torch.float32 or scale_b.dtype != torch.float32:
        raise ValueError('FP8 scale metadata must be FP32')
    if bias is not None and (bias.shape != (b.shape[1],) or bias.device != a.device or bias.dtype not in _FLOATS):
        raise ValueError('FP8 GEMM bias must be a floating [N] tensor on the operand device')
    if strategy not in ('auto', 'native', 'float16', 'scalar'):
        raise ValueError('FP8 GEMM strategy must be auto, native, float16 or scalar')
    if strategy != 'scalar':
        from .float8_training import _mm, _STRATEGIES
        result = _mm(a, b, scale_a, scale_b, False, False, _STRATEGIES[strategy])
        if bias is not None:
            result = result + bias.float()
        return result.to(out_dtype)
    weight = b.t().contiguous().view(torch.uint8)
    scales = scale_b.float().reshape(-1, 1).expand(b.shape[1], 1).contiguous()
    offsets = torch.zeros_like(scales)
    index = torch.empty(0, dtype=torch.int32, device=b.device)
    fmt = 'fp8_e4m3' if b.dtype == torch.float8_e4m3fn else 'fp8_e5m2'
    result = _linear(a.float().contiguous(), weight, scales, offsets, index,
                     _layout(fmt, b.shape[0], (b.shape[1], b.shape[0])), False)
    result = result * scale_a.float()
    if bias is not None:
        result = result + bias.float()
    return result.to(out_dtype)


def _aten_scaled_mm(a,b,scale_a,scale_b,bias=None,scale_result=None,out_dtype=None,use_fast_accum=False):
    dtype = a.dtype if out_dtype is None else out_dtype
    if dtype not in _FLOATS:
        raise NotImplementedError('RUDA _scaled_mm currently requires explicit FP16/BF16/FP32 output')
    if scale_result is not None:
        raise ValueError('scale_result applies to FP8 output, not a floating accumulation output')
    # FP32 accumulation is used for either precision preference.
    return scaled_float8_mm(a,b,scale_a,scale_b,out_dtype=dtype,bias=bias)


def _aten_scaled_mm_out(a,b,scale_a,scale_b,bias=None,scale_result=None,out_dtype=None,use_fast_accum=False,*,out):
    result = _aten_scaled_mm(a,b,scale_a,scale_b,bias,scale_result,out.dtype if out_dtype is None else out_dtype,use_fast_accum)
    if out.dtype != result.dtype or out.device != result.device:
        raise ValueError('scaled_mm out dtype/device mismatch')
    _C.prepare_index_output(out,list(result.shape))
    return out.copy_(result)


_aten_quantization = torch.library.Library('aten','IMPL','PrivateUse1')
_aten_quantization.impl('_scaled_mm',_aten_scaled_mm)
_aten_quantization.impl('_scaled_mm.out',_aten_scaled_mm_out)
