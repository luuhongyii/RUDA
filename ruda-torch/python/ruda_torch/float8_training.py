"""FP8 tensor-core GEMMs with current, block and delayed training scales."""
from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass
from threading import RLock

import torch
from torch import nn
from torch.autograd.function import once_differentiable

_FLOATS = (torch.float32, torch.float16, torch.bfloat16)
_FP8 = (torch.float8_e4m3fn, torch.float8_e5m2)
_STRATEGIES = {'auto': 0, 'native': 1, 'float16': 2}
_lock = RLock()
_ready = False


def _native():
    global _ready
    from . import _C, _native as library
    with _lock:
        if not _ready:
            if not hasattr(library, 'ruda_torch_float8_api_version') or not hasattr(_C, 'initialize_float8'):
                raise RuntimeError('FP8 tensor cores require matching rebuilt RUDA native libraries')
            library.ruda_torch_float8_api_version.restype = ctypes.c_uint32
            if library.ruda_torch_float8_api_version() != 1 or _C.float8_api_version != 1:
                raise RuntimeError('RUDA FP8 API mismatch')
            _C.initialize_float8(ctypes.cast(library.ruda_torch_float8, ctypes.c_void_p).value)
            _ready = True
    return _C


@torch.library.custom_op('ruda::float8_mm', mutates_args=())
def _mm(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor,
        transpose_a: bool, transpose_b: bool, strategy: int) -> torch.Tensor:
    if a.dtype not in _FP8 or b.dtype not in _FP8 or a.ndim != 2 or b.ndim != 2:
        raise ValueError('FP8 GEMM requires two FP8 matrices')
    m, k = a.shape[::-1] if transpose_a else a.shape
    kb, n = b.shape[::-1] if transpose_b else b.shape
    if k != kb or strategy not in (0, 1, 2):
        raise ValueError('invalid FP8 GEMM geometry/strategy')
    for scale, shape in ((scale_a, (m, 1)), (scale_b, (1, n))):
        if scale.dtype != torch.float32 or (scale.numel() != 1 and scale.shape != shape):
            raise ValueError('FP8 GEMM scale must be FP32 scalar or per row/column')
    if a.device.type != 'ruda' or any(t.device != a.device for t in (b, scale_a, scale_b)):
        raise ValueError('FP8 GEMM operands must share a RUDA device')
    out = torch.empty((m, n), device=a.device, dtype=torch.float32)
    _native().float8_dispatch(0, [a.contiguous(), b.contiguous(), scale_a.contiguous(), scale_b.contiguous(), out],
                             transpose_a, transpose_b, strategy)
    return out


@_mm.register_fake
def _mm_fake(a, b, scale_a, scale_b, transpose_a, transpose_b, strategy):
    return torch.empty((a.shape[1 if transpose_a else 0], b.shape[0 if transpose_b else 1]),
                       device=a.device, dtype=torch.float32)


@torch.library.custom_op('ruda::float8_quantize', mutates_args=())
def _encode(x: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if x.device.type != 'ruda' or x.dtype not in _FLOATS or dtype not in _FP8:
        raise ValueError('FP8 encoding requires a RUDA FP32/FP16/BF16 source')
    if scale.device != x.device or scale.dtype != torch.float32 or scale.numel() != 1:
        raise ValueError('FP8 encoding scale must be an FP32 scalar on the source device')
    out = torch.empty(x.shape, device=x.device, dtype=dtype)
    _native().float8_dispatch(1, [x.contiguous(), scale.contiguous(), out], False, False, 0)
    return out


@_encode.register_fake
def _encode_fake(x, scale, dtype):
    return torch.empty(x.shape, device=x.device, dtype=dtype)


def _power_of_two_scale(scale):
    bits = scale.view(torch.int32)
    exponent = bits & 0x7f800000
    increment = ((bits & 0x007fffff) != 0).to(torch.int32) * 0x00800000
    return (exponent + increment).view(torch.float32)


@torch.library.custom_op('ruda::float8_current_quantize',mutates_args=())
def _current_encode(x:torch.Tensor,dtype:torch.dtype,margin:int,power_of_two:bool)->tuple[torch.Tensor,torch.Tensor]:
    maximum = torch.finfo(dtype).max / (2. ** margin)
    amax = x.detach().float().abs().amax().reshape(()) if x.numel() else torch.zeros((), device=x.device)
    scale = torch.where((amax > 0) & torch.isfinite(amax),
                        (amax / maximum).clamp_min(torch.finfo(torch.float32).tiny), torch.ones_like(amax))
    if power_of_two:
        scale = _power_of_two_scale(scale)
    return _encode(x, scale, dtype), scale


@_current_encode.register_fake
def _current_encode_fake(x,dtype,margin,power_of_two):
    return torch.empty(x.shape,device=x.device,dtype=dtype),torch.empty((),device=x.device,dtype=torch.float32)


def _quantize(x,dtype,margin,power_of_two):
    return _current_encode(x.detach(),dtype,margin,power_of_two)


@dataclass(frozen=True)
class Float8Recipe:
    """Scaling policy for forward, input-gradient and weight-gradient GEMMs.

    auto selects native FP8 MMA or an exact FP16 carrier on older tensor cores.
    native requires FP8 instructions; float16 explicitly chooses the carrier.
    Optimizer parameters and their gradients are never stored in FP8.
    """
    input_dtype: torch.dtype = torch.float8_e4m3fn
    weight_dtype: torch.dtype = torch.float8_e4m3fn
    grad_dtype: torch.dtype = torch.float8_e5m2
    strategy: str = 'auto'
    margin: int = 0
    power_of_two_scale: bool = False
    scaling: str = 'current'
    block_size: int = 128
    input_block_rows: int = 1
    weight_block_rows: int = 128
    grad_block_rows: int = 1
    amax_history_len: int = 16
    amax_compute_algo: str = 'max'
    scaling_interval: int = 1

    def __post_init__(self):
        if any(dtype not in _FP8 for dtype in (self.input_dtype, self.weight_dtype, self.grad_dtype)):
            raise ValueError('FP8 recipe formats must be E4M3FN or E5M2')
        if self.strategy not in _STRATEGIES:
            raise ValueError('FP8 strategy must be auto, native or float16')
        if type(self.margin) is not int or not 0 <= self.margin <= 16:
            raise ValueError('FP8 scaling margin must be an integer in [0,16]')
        if type(self.power_of_two_scale) is not bool:
            raise TypeError('power_of_two_scale must be boolean')
        if self.scaling not in ('current', 'block', 'delayed'):
            raise ValueError('FP8 scaling must be current, block or delayed')
        for value in (self.block_size, self.input_block_rows, self.weight_block_rows,
                      self.grad_block_rows, self.amax_history_len, self.scaling_interval):
            if type(value) is not int or value < 1:
                raise ValueError('FP8 block dimensions, history and interval must be positive integers')
        if self.block_size % 32 or any(rows * self.block_size > 2**32 - 33
                for rows in (self.input_block_rows, self.weight_block_rows, self.grad_block_rows)):
            raise ValueError('FP8 K block must be a multiple of 32 with U32-addressable block geometry')
        if self.amax_compute_algo not in ('max', 'most_recent'):
            raise ValueError('delayed amax_compute_algo must be max or most_recent')


@torch.library.custom_op('ruda::float8_linear_forward', mutates_args=())
def _linear(x: torch.Tensor, weight: torch.Tensor, input_format: torch.dtype,
            weight_format: torch.dtype, grad_format: torch.dtype, output_dtype: torch.dtype,
            strategy: int, margin: int, power_of_two: bool
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    qx, sx = _quantize(x, input_format, margin, power_of_two)
    qw, sw = _quantize(weight, weight_format, margin, power_of_two)
    out = _mm(qx, qw, sx, sw, False, True, strategy).to(output_dtype)
    return out, qx, qw, sx, sw


@_linear.register_fake
def _linear_fake(x, weight, input_format, weight_format, grad_format, output_dtype, strategy, margin, power_of_two):
    return (torch.empty((x.shape[0], weight.shape[0]), dtype=output_dtype, device=x.device),
            torch.empty(x.shape, dtype=input_format, device=x.device),
            torch.empty(weight.shape, dtype=weight_format, device=weight.device),
            torch.empty((), dtype=torch.float32, device=x.device),
            torch.empty((), dtype=torch.float32, device=weight.device))


def _linear_context(ctx, inputs, output):
    x, weight, _, _, grad_format, _, strategy, margin, power_of_two = inputs
    _, qx, qw, sx, sw = output
    ctx.save_for_backward(qx, qw, sx, sw)
    ctx.input_dtype, ctx.weight_dtype = x.dtype, weight.dtype
    ctx.options = grad_format, strategy, margin, power_of_two
    ctx.mark_non_differentiable(qx, qw, sx, sw)
    ctx.set_materialize_grads(False)


@once_differentiable
def _linear_backward(ctx, grad, _qx, _qw, _sx, _sw):
    if grad is None:
        return (None,) * 9
    qx, qw, sx, sw = ctx.saved_tensors
    grad_format, strategy, margin, power_of_two = ctx.options
    qg, sg = _quantize(grad.contiguous(), grad_format, margin, power_of_two)
    dx = _mm(qg, qw, sg, sw, False, False, strategy).to(ctx.input_dtype) if ctx.needs_input_grad[0] else None
    dw = _mm(qg, qx, sg, sx, True, False, strategy).to(ctx.weight_dtype) if ctx.needs_input_grad[1] else None
    return dx, dw, None, None, None, None, None, None, None


_linear.register_autograd(_linear_backward, setup_context=_linear_context)


def float8_linear(x, weight, bias=None, *, recipe=None, output_dtype=None, scale_state=None, observe_scaling=True):
    """Trainable FP8 Linear with FP32 accumulation; no CPU tensor transfers."""
    recipe = Float8Recipe() if recipe is None else recipe
    if not isinstance(recipe, Float8Recipe):
        raise TypeError('recipe must be Float8Recipe')
    if x.ndim < 1 or weight.ndim != 2 or x.shape[-1] != weight.shape[1]:
        raise ValueError('FP8 Linear requires [...,in] input and [out,in] weights')
    if x.device.type != 'ruda' or weight.device != x.device or x.dtype not in _FLOATS or weight.dtype not in _FLOATS:
        raise ValueError('FP8 Linear needs matching RUDA FP32/FP16/BF16 input and master weight')
    if output_dtype is None:
        output_dtype = torch.get_autocast_dtype('ruda') if torch.is_autocast_enabled('ruda') else x.dtype
    if output_dtype not in _FLOATS:
        raise ValueError('FP8 Linear output must be FP32/FP16/BF16')
    if bias is not None and (bias.shape != (weight.shape[0],) or bias.device != x.device or bias.dtype not in _FLOATS):
        raise ValueError('FP8 Linear bias must be floating [out] on the input device')
    shape = x.shape[:-1] + (weight.shape[0],)
    flat = x.reshape(math.prod(x.shape[:-1]), weight.shape[1]).contiguous()
    # Bias is added to the FP32 accumulator before the output conversion.
    if recipe.scaling == 'current':
        out = _linear(flat, weight.contiguous(), recipe.input_dtype, recipe.weight_dtype, recipe.grad_dtype,
                      torch.float32, _STRATEGIES[recipe.strategy], recipe.margin, recipe.power_of_two_scale)[0]
    else:
        from .float8_scaling import recipe_linear
        out = recipe_linear(flat, weight.contiguous(), recipe, scale_state, observe=observe_scaling)
    if bias is not None:
        out = out + bias.float()
    return out.to(output_dtype).reshape(shape)


class Float8Linear(nn.Module):
    """Trainable master parameters plus FP8 forward, input and weight gradients.

    Master parameter keys are unchanged. Delayed scaling adds persistent scale,
    history and pending-observation buffers, advanced at optimizer boundaries.
    """
    def __init__(self, in_features, out_features, bias=True, *, device='ruda:0',
                 dtype=torch.float32, recipe=None):
        super().__init__()
        if any(type(n) is not int or n < 1 for n in (in_features, out_features)) or dtype not in _FLOATS:
            raise ValueError('invalid FP8 Linear dimensions/master dtype')
        self.in_features, self.out_features = in_features, out_features
        self.recipe = Float8Recipe() if recipe is None else recipe
        if not isinstance(self.recipe, Float8Recipe):
            raise TypeError('recipe must be Float8Recipe')
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype)) if bias else None
        from .float8_scaling import make_scale_state
        self.scale_state = make_scale_state(self.recipe, self.weight)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.uniform_(self.bias, -1 / math.sqrt(self.in_features), 1 / math.sqrt(self.in_features))

    @classmethod
    def from_linear(cls, linear, *, recipe=None):
        if not isinstance(linear, nn.Linear) or linear.weight.dtype not in _FLOATS:
            raise TypeError('expected a floating torch.nn.Linear master module')
        # Preserve parameter identities, optimizer references and tied weights.
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result.in_features, result.out_features = linear.in_features, linear.out_features
        result.recipe = Float8Recipe() if recipe is None else recipe
        if not isinstance(result.recipe, Float8Recipe):
            raise TypeError('recipe must be Float8Recipe')
        result.weight, result.bias = linear.weight, linear.bias
        from .float8_scaling import make_scale_state
        result.scale_state = make_scale_state(result.recipe, result.weight)
        result.train(linear.training)
        return result

    def forward(self, x):
        return float8_linear(x, self.weight, self.bias, recipe=self.recipe, scale_state=self.scale_state, observe_scaling=self.training)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, recipe={self.recipe!r}'


def convert_to_float8_training(model, *, target_modules, recipe=None):
    """Replace explicitly selected Linear modules, preserving master/tied parameters."""
    from .finetuning import _selected, _replace
    from .parallel_training import ColumnParallelLinear,RowParallelLinear
    from .float8_parallel import from_parallel
    selected = _selected(model, target_modules, (nn.Linear,ColumnParallelLinear,RowParallelLinear))
    replacements, shared = [], {}
    for name, module in selected:
        if id(module) not in shared:
            shared[id(module)] = (Float8Linear.from_linear(module,recipe=recipe) if isinstance(module,nn.Linear)
                                  else from_parallel(module,recipe=recipe))
        replacements.append((name, shared[id(module)]))
    _replace(model, replacements)
    return model
