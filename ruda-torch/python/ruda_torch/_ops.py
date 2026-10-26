import math
import numbers
from functools import partial
import torch
from torch._prims_common import suggest_memory_format
from . import _C
from .math_backend import _nvidia

_registry = torch.library.Library("aten", "IMPL", "PrivateUse1")
_native_eager_ops = frozenset(getattr(_C, 'native_eager_ops', ()))
if _native_eager_ops:
    from . import _native
    _arithmetic_version = getattr(_native, 'ruda_torch_arithmetic_api_version', None)
    _C.initialize_eager_arithmetic(_arithmetic_version is not None and _arithmetic_version() >= 2)
    if hasattr(_C, 'initialize_eager_cat'):
        _C.initialize_eager_cat(_arithmetic_version is not None and _arithmetic_version() >= 3)
_dtypes = (torch.float32, torch.float16, torch.bfloat16, torch.float64)
_numeric_dtypes = _dtypes + (torch.complex64, torch.complex128)
_wide_dtypes = (torch.float64, torch.complex64, torch.complex128)
_integral_dtypes = (torch.bool, torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8)
_storage_dtypes = _numeric_dtypes + _integral_dtypes + (torch.float8_e4m3fn, torch.float8_e5m2, torch.complex32)

def _has_complex32(*values):
    return any(isinstance(value, torch.Tensor) and value.dtype == torch.complex32 for value in values)

def _out(shape, tensor):
    return torch.empty(shape, device=tensor.device, dtype=tensor.dtype)

def _apply(op, a, b=None, *, out=None, scalar=0.0):
    if b is None:
        b = a
    a, b = torch.broadcast_tensors(a, b)
    if out is None:
        out = _out(a.shape, a)
    if out.shape != a.shape:
        raise RuntimeError("RUDA output shape mismatch")
    if a.dtype in _wide_dtypes or b.dtype in _wide_dtypes or out.dtype in _wide_dtypes or _has_complex32(a, b, out):
        from ._scientific import execute
        execute(op, a, b, out, scalar)
    else:
        _C.execute(op, a, b, out, scalar)
    return out

def _scalar(value, tensor):
    if isinstance(value, torch.Tensor):
        if value.device != tensor.device or value.dtype != tensor.dtype:
            raise NotImplementedError("RUDA requires matching GPU devices and tensor dtypes")
        return value
    result = torch.empty((), device=tensor.device, dtype=tensor.dtype if tensor.dtype in _wide_dtypes else torch.float32)
    return fill_(result, value)

def fill_(tensor, value):
    _C.fill(tensor, value)
    return tensor

def _binary(op, a, b, *, scalar=0.0, inplace=False):
    if _has_complex32(a, b):
        from ._complex32_ops import binary
        if op == 50:
            return binary(b, a, op=1, alpha=scalar, out=a if inplace else None)
        return binary(a, b, op={1: 0, 2: 2, 8: 3}[op], alpha=scalar if op == 1 else 1, inplace=inplace)
    if op in (1, 2) and (a.dtype not in _numeric_dtypes or isinstance(b, torch.Tensor) and b.dtype not in _numeric_dtypes):
        return _integral_arithmetic(op, a, b, alpha=scalar if op == 1 else 1, inplace=inplace)
    if a.dtype not in _numeric_dtypes:
        raise RuntimeError("RUDA arithmetic requires a supported floating dtype")
    dtype = a.dtype
    if isinstance(b, torch.Tensor):
        if b.device != a.device or b.dtype not in _numeric_dtypes:
            raise RuntimeError("RUDA arithmetic inputs must use supported floating dtypes on the same device")
        dtype = torch.promote_types(a.dtype, b.dtype)
    else:
        if isinstance(b, complex) and not a.is_complex():
            dtype = torch.result_type(a, b)
            b = _typed_value(b, dtype, a.device)
        else:
            b = _scalar(b, a)
    left, right = torch.broadcast_tensors(a, b)
    if inplace:
        if not torch.can_cast(dtype, a.dtype):
            raise RuntimeError('in-place operation cannot cast the promoted result dtype')
        if a.shape != left.shape:
            raise RuntimeError("RUDA in-place output cannot change shape through broadcasting")
        result = a
    else:
        result = torch.empty(left.shape, device=a.device, dtype=dtype)
    if left.dtype in _wide_dtypes or right.dtype in _wide_dtypes or result.dtype in _wide_dtypes:
        from ._scientific import execute
        execute(op, left, right, result, scalar)
    else:
        _C.execute(op, left, right, result, float(scalar))
    return result


def add(a, b, alpha=1):
    from ._arithmetic import arithmetic
    return arithmetic(a, b, op=0, alpha=alpha)

def add_(a, b, alpha=1):
    from ._arithmetic import arithmetic
    return arithmetic(a, b, op=0, alpha=alpha, inplace=True)

def mul(a, b):
    from ._arithmetic import arithmetic
    return arithmetic(a, b, op=2)

def mul_(a, b):
    from ._arithmetic import arithmetic
    return arithmetic(a, b, op=2, inplace=True)


def mul_out(a, b, *, out):
    from ._arithmetic import arithmetic
    return arithmetic(a, b, op=2, out=out)

def _basic_binary_out(a, b, alpha=1, *, op, rounding_mode=None, out):
    if op <= 2:
        if not isinstance(b, torch.Tensor) and out.dtype != torch.result_type(a, b):
            raise RuntimeError('Scalar_out requires the functional result dtype')
        from ._arithmetic import arithmetic
        return arithmetic(a, b, op=op, alpha=alpha, out=out)
    from ._division import divide
    return divide(a, b, rounding_mode=rounding_mode, out=out)


def _neg_out(a, *, out):
    from ._arithmetic import arithmetic
    return arithmetic(a, op=3, out=out)

def _division_op(rounding_mode):
    if rounding_mode is None:
        return 8
    if rounding_mode == "trunc":
        return 57
    if rounding_mode == "floor":
        _require_math()
        return 122
    raise RuntimeError("rounding_mode must be None, 'trunc' or 'floor'")


def div(a, b, *, rounding_mode=None, out=None):
    from ._division import divide
    return divide(a, b, rounding_mode=rounding_mode, out=out)

def div_(a, b, *, rounding_mode=None):
    from ._division import divide
    return divide(a, b, rounding_mode=rounding_mode, inplace=True)

def mm(a, b):
    _scalar(b, a)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise RuntimeError("RUDA mm requires compatible two-dimensional tensors")
    if _nvidia():
        from ._vendor_blas import gemm
        return gemm(a,b)
    if a.dtype in _wide_dtypes:
        from .blas import aten_matmul
        return aten_matmul(a,b)
    out = _out((a.shape[0], b.shape[1]), a)
    _C.execute(7, a, b, out, 0.0)
    return out

def addmm(bias, a, b, *, beta=1, alpha=1):
    _scalar(b, a)
    _scalar(bias, a)
    if a.dtype not in _numeric_dtypes:
        raise RuntimeError("RUDA addmm requires a supported floating dtype")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise RuntimeError("RUDA addmm requires compatible two-dimensional tensors")
    shape = (a.shape[0], b.shape[1])
    # Validate broadcasting even when beta=0, but the device kernel will not
    # read bias values in that case (including NaN or infinity).
    expanded_bias = bias.expand(shape)
    if _nvidia():
        from ._vendor_blas import gemm
        return gemm(a,b,expanded_bias,alpha=alpha,beta=beta)
    if a.dtype in _wide_dtypes:
        from .blas import gemm
        return gemm(a,b,expanded_bias,alpha=alpha,beta=beta)
    result = _out(shape, a)
    _C.addmm(expanded_bias, a, b, result, float(alpha), float(beta))
    return result

def bmm(a, b):
    _scalar(b, a)
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0] or a.shape[2] != b.shape[1]:
        raise RuntimeError("RUDA bmm requires compatible three-dimensional batches")
    if _nvidia():
        from ._vendor_blas import gemm
        return gemm(a,b)
    if a.dtype in _wide_dtypes:
        from .blas import aten_matmul
        return aten_matmul(a,b)
    out = _out((a.shape[0], a.shape[1], b.shape[2]), a)
    _C.execute(30, a, b, out, 0.0)
    return out

def _dimension(dim, rank):
    if dim < -max(rank, 1) or dim >= max(rank, 1):
        raise IndexError("dimension out of range")
    return dim % rank if rank else 0

def softmax(a, dim, half_to_float=False, *, logarithmic=False, out=None):
    if a.dtype not in _dtypes:
        raise RuntimeError('softmax requires real floating input')
    if half_to_float:
        if a.dtype != torch.float16:
            raise RuntimeError("half_to_float requires float16 input")
    dim = _dimension(dim, a.ndim)
    x = a.resolve_neg().reshape((1,)) if a.ndim == 0 else a.resolve_neg()
    value = torch.empty(x.shape, device=a.device, dtype=torch.float32 if half_to_float else a.dtype)
    if a.dtype == torch.float64:
        _C.scientific_call(306 if logarithmic else 305, [x, x, value], [dim], [])
    else:
        _C.execute(33 if logarithmic else 31, x, x, value, float(dim))
    from ._ordering import _output
    return _output(value.reshape(a.shape), out, (a,))

def softmax_backward(grad, output, dim, input_dtype, *, logarithmic=False, out=None):
    if input_dtype not in _dtypes or grad.shape != output.shape:
        raise RuntimeError("RUDA softmax backward requires supported dtypes and matching shapes")
    dim = _dimension(dim, output.ndim)
    if grad.dtype != output.dtype or grad.device != output.device:
        raise RuntimeError('softmax gradient and output must share dtype and device')
    if (input_dtype == torch.float64) != (output.dtype == torch.float64):
        raise RuntimeError('float64 softmax backward requires float64 output and input')
    shape = output.shape or (1,)
    g, y = grad.resolve_neg().reshape(shape), output.resolve_neg().reshape(shape)
    value = torch.empty(shape, device=output.device, dtype=input_dtype)
    if input_dtype == torch.float64:
        _C.scientific_call(308 if logarithmic else 307, [g, y, value], [dim], [])
    else:
        _C.execute(34 if logarithmic else 32, g, y, value, float(dim))
    from ._ordering import _output
    return _output(value.reshape(output.shape), out, (grad, output))

def sum_dim(a, dim=None, keepdim=False, *, dtype=None):
    from ._additive_reduce import sum_reduce
    return sum_reduce(a, dim, keepdim, dtype=dtype)

def mean_dim(a, dim=None, keepdim=False, *, dtype=None):
    from ._additive_reduce import mean_reduce
    return mean_reduce(a, dim, keepdim, dtype=dtype)

def _norm_axes(a, normalized_shape, weight, bias):
    normalized_shape = tuple(normalized_shape)
    if not normalized_shape or len(normalized_shape) > a.ndim or tuple(a.shape[-len(normalized_shape):]) != normalized_shape:
        raise RuntimeError("normalized_shape must match the trailing input dimensions")
    for value in (weight, bias):
        if value is not None and (tuple(value.shape) != normalized_shape or value.device != a.device or value.dtype != a.dtype):
            raise RuntimeError("LayerNorm affine parameters must match normalized_shape, dtype and device")
    return tuple(range(a.ndim - len(normalized_shape), a.ndim))

def layer_norm(a, normalized_shape, weight=None, bias=None, eps=1e-5):
    axes = _norm_axes(a, normalized_shape, weight, bias)
    # Transformer inference overwhelmingly normalizes the last feature axis.
    # Use one fused native GPU kernel for that common case; keep the existing
    # composition for multi-axis normalization and non-contiguous last axes.
    if (a.dtype in (torch.float32, torch.float16, torch.bfloat16)
            and len(axes) == 1 and axes[0] == a.ndim - 1 and a.is_contiguous() and a.shape[-1] > 0
            and (weight is None or weight.is_contiguous()) and (bias is None or bias.is_contiguous())):
        stats_shape = list(a.shape)
        stats_shape[-1] = 1
        # Training needs FP32 saved statistics: keeping low-precision mean/rstd
        # would lose information and the old backward materialized full FP32
        # copies. The separately negotiated training API supplies those stats
        # and fused first-order backward without changing inference behavior.
        needs_grad = a.requires_grad or (weight is not None and weight.requires_grad) or (bias is not None and bias.requires_grad)
        if needs_grad:
            from . import _training_available
            if _training_available:
                result, mean, rstd = _C.training_layer_forward(a, weight, bias, eps)
                return result, mean.view(stats_shape), rstd.view(stats_shape)
        result = torch.empty_like(a)
        mean = torch.empty(stats_shape, device=a.device, dtype=a.dtype)
        rstd = torch.empty(stats_shape, device=a.device, dtype=a.dtype)
        _C.layer_norm(a, weight, bias, result, mean, rstd, eps)
        return result, mean, rstd
    if a.dtype not in (torch.float32, torch.float64):
        result, mean, rstd = layer_norm(a.float(), normalized_shape,
            weight.float() if weight is not None else None, bias.float() if bias is not None else None, eps)
        return result.to(a.dtype), mean, rstd
    mean = mean_dim(a, axes, True)
    centered = a - mean
    variance = mean_dim(centered * centered, axes, True)
    rstd = torch.rsqrt(variance + eps)
    result = centered * rstd
    if weight is not None:
        result = result * weight
    if bias is not None:
        result = result + bias
    return result, mean, rstd

def rms_norm(a, normalized_shape, weight=None, eps=None):
    axes = _norm_axes(a, normalized_shape, weight, None)
    epsilon = torch.finfo(a.dtype).eps if eps is None else float(eps)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise RuntimeError("RMSNorm epsilon must be finite and positive")
    if (a.dtype in (torch.float32, torch.float16, torch.bfloat16)
            and len(axes) == 1 and axes[0] == a.ndim - 1 and a.is_contiguous() and a.shape[-1] > 0
            and (weight is None or weight.is_contiguous())):
        result = torch.empty_like(a)
        _C.rms_norm(a, weight, result, epsilon)
        return result
    # Portable device fallback for uncommon multi-axis/view cases. Storage remains
    # on RUDA; the last-axis common path above is a single fused GPU kernel.
    square = a * a
    variance = mean_dim(square, axes, True)
    result = a * torch.rsqrt(variance + epsilon)
    if weight is not None:
        result = result * weight
    return result

def layer_norm_backward(grad, a, normalized_shape, mean, rstd, weight, bias, output_mask):
    axes = _norm_axes(a, normalized_shape, weight, bias)
    if (len(axes) == 1 and axes[0] == a.ndim - 1 and a.is_contiguous() and grad.is_contiguous()
            and a.shape[-1] > 0 and mean.dtype == torch.float32 and rstd.dtype == torch.float32
            and mean.is_contiguous() and rstd.is_contiguous()
            and (weight is None or weight.is_contiguous()) and (bias is None or bias.is_contiguous())):
        from . import _training_available
        if _training_available:
            return _C.training_layer_backward(a, weight, bias, grad, mean.reshape(-1), rstd.reshape(-1),
                bool(output_mask[0]), bool(output_mask[1]), bool(output_mask[2]))
    if a.dtype != torch.float32:
        results = layer_norm_backward(grad.float(), a.float(), normalized_shape, mean.float(), rstd.float(),
            weight.float() if weight is not None else None, bias.float() if bias is not None else None, output_mask)
        return tuple(value.to(a.dtype) if value is not None else None for value in results)
    leading = tuple(range(a.ndim - len(axes)))
    normalized = (a - mean) * rstd
    dx = dw = db = None
    if output_mask[0]:
        scaled_grad = grad if weight is None else grad * weight
        dx = (scaled_grad - mean_dim(scaled_grad, axes, True)
              - normalized * mean_dim(scaled_grad * normalized, axes, True)) * rstd
    if output_mask[1] and weight is not None:
        dw = sum_dim(grad * normalized, leading) if leading else clone(grad * normalized)
    if output_mask[2] and bias is not None:
        db = sum_dim(grad, leading) if leading else clone(grad)
    return dx, dw, db

def pow_scalar(a, exponent, *, out=None):
    from ._power import power
    return power(a, exponent, out=out)


def _require_math():
    from . import _math_available
    if not _math_available:
        raise RuntimeError('this operation requires the RUDA native math extension; rebuild the native library')


def pow_tensor(a, exponent, *, out=None):
    from ._power import power
    return power(a, exponent, out=out)


def pow_scalar_base(base, exponent, *, out=None):
    from ._power import power
    return power(base, exponent, out=out)


def _math_out(result, out):
    if out.device != result.device or not torch.can_cast(result.dtype, out.dtype):
        raise RuntimeError('RUDA math output must share the device and allow result dtype casting')
    _C.prepare_index_output(out, list(result.shape))
    out.copy_(result)
    return out


def _power_inplace(a, exponent):
    from ._power import power
    return power(a, exponent, inplace=True)


def _floating_modulus(a, b, *, floor, inplace=False, out=None):
    from ._division import modulus
    return modulus(a, b, floor=floor, inplace=inplace, out=out)


def _scalar_modulus(a, b, *, floor, out=None):
    from ._division import modulus
    return modulus(a, b, floor=floor, out=out)


def _typed_division(a, b, *, rounding_mode, inplace=False):
    if inplace:
        _C.check_inplace(a, b if isinstance(b, torch.Tensor) else a)
    if rounding_mode == 'floor':
        _require_math()
    left, right, dtype = _primitive_operands(a, b)
    if dtype == torch.bool or dtype in (torch.complex32, torch.complex64, torch.complex128):
        raise RuntimeError('RUDA rounded division is not defined for bool or complex results')
    if dtype in _dtypes:
        op = 122 if rounding_mode == 'floor' else 57
    else:
        _require_math()
        op = 126 if rounding_mode == 'floor' else 125
    result = _apply(op, left, right)
    return _primitive_result(result, a, inplace)


def _true_division(a, b, *, inplace=False):
    if inplace:
        _C.check_inplace(a, b if isinstance(b, torch.Tensor) else a)
    left, right, dtype = _primitive_operands(a, b)
    if dtype not in _numeric_dtypes:
        dtype = torch.get_default_dtype()
        if dtype not in _dtypes:
            raise RuntimeError('RUDA true division does not support the selected default floating dtype')
        left, right = left.to(dtype), right.to(dtype)
    if inplace and (left.shape != a.shape or not torch.can_cast(dtype, a.dtype)):
        raise RuntimeError('RUDA in-place true division cannot change shape or cast the floating result to the input dtype')
    return _primitive_result(_apply(8, left, right), a, inplace)

def activation_backward(grad, output, *, tanh=False, grad_input=None):
    from ._exponential import backward
    return backward(grad, output, tanh=tanh, grad_input=grad_input)

def clone(a, *, memory_format=torch.preserve_format):
    if (a.is_conj() or a.is_neg()) and not torch._is_zerotensor(a):
        return _C.scientific_resolve(a)
    out = torch.empty_like(a, memory_format=memory_format)
    return out.copy_(a)


def leaky_relu(a, negative_slope=0.01):
    return _apply(41, a, scalar=float(negative_slope))


def leaky_relu_(a, negative_slope=0.01):
    return _apply(41, a, out=a, scalar=float(negative_slope))


def leaky_relu_backward(grad, a, negative_slope, self_is_result):
    if self_is_result and negative_slope < 0:
        raise RuntimeError("in-place leaky_relu backward does not support a negative slope")
    return _apply(42, grad, a, scalar=float(negative_slope))


def log_sigmoid_forward(a):
    return _apply(47, a), _out((0,), a)


def log_sigmoid_backward(grad, a, buffer):
    return _apply(48, grad, a)


def addcmul(a, tensor1, tensor2, *, value=1):
    from ._ternary_ops import addc
    return addc(a, tensor1, tensor2, value=value)


def addcmul_(a, tensor1, tensor2, *, value=1):
    from ._ternary_ops import addc
    return addc(a, tensor1, tensor2, value=value, inplace=True)


def addcdiv(a, tensor1, tensor2, *, value=1):
    from ._ternary_ops import addc
    return addc(a, tensor1, tensor2, value=value, divide=True)


def addcdiv_(a, tensor1, tensor2, *, value=1):
    from ._ternary_ops import addc
    return addc(a, tensor1, tensor2, value=value, divide=True, inplace=True)


def lerp(a, end, weight, *, inplace=False):
    from ._ternary_ops import lerp as native_lerp
    return native_lerp(a, end, weight, inplace=inplace)


def cat(tensors, dim=0):
    tensors = tuple(tensors)
    if not tensors:
        raise RuntimeError("cat expects a non-empty tensor list")
    first = tensors[0]
    dtype = first.dtype
    memory_format = None
    for tensor in tensors:
        if tensor.device != first.device or tensor.dtype not in _storage_dtypes or tensor.layout != torch.strided:
            raise RuntimeError("RUDA cat requires supported strided tensors on the same device")
        if tensor.ndim == 0:
            raise RuntimeError("zero-dimensional tensors cannot be concatenated")
        dtype = torch.promote_types(dtype, tensor.dtype)
        current_format = suggest_memory_format(tensor)
        if memory_format is None:
            memory_format = current_format
        elif memory_format != current_format:
            memory_format = torch.contiguous_format
    inputs = tuple(tensor for tensor in tensors if tensor.ndim != 1 or tensor.shape[0] != 0)
    if not inputs:
        return torch.empty((0,), device=first.device, dtype=dtype)
    dim = _dimension(dim, inputs[0].ndim)
    shape = list(inputs[0].shape)
    shape[dim] = 0
    for tensor in inputs:
        if tensor.ndim != len(shape) or any(
            tensor.shape[axis] != shape[axis] for axis in range(len(shape)) if axis != dim
        ):
            raise RuntimeError("cat tensor sizes must match except in the concatenation dimension")
        shape[dim] += tensor.shape[dim]
    result = torch.empty(shape, device=first.device, dtype=dtype, memory_format=memory_format)
    cursor = 0
    for tensor in inputs:
        if tensor.numel() != 0:
            destination = torch.as_strided(result, tensor.shape, result.stride(), cursor * result.stride(dim))
            destination.copy_(tensor)
        cursor += tensor.shape[dim]
    return result


def stack(tensors, dim=0):
    tensors = tuple(tensors)
    if not tensors:
        raise RuntimeError("stack expects a non-empty tensor list")
    shape = tensors[0].shape
    if any(tensor.shape != shape for tensor in tensors):
        raise RuntimeError("stack expects every tensor to have the same shape")
    dim = _dimension(dim, len(shape) + 1)
    if dim < len(shape):
        output_shape = list(shape)
        output_shape.insert(dim, len(tensors))
        return cat(tensors, dim).view(output_shape)
    return cat(tuple(tensor.unsqueeze(dim) for tensor in tensors), dim)


def mv(matrix, vector):
    from ._blas_accumulate import mv as native_mv
    return native_mv(matrix, vector)


def dot(a, b):
    from ._dot import dot as native_dot
    return native_dot(a, b)


def vdot(a, b):
    from ._dot import dot as native_dot
    return native_dot(a, b, conjugate=True)


def _basic_unary(a, *, op, inplace=False, out=None):
    from ._unary_basic import unary
    return unary(a, op=op, inplace=inplace, out=out)


def _exponential_unary(a, *, op, inplace=False, out=None):
    from ._exponential import unary
    return unary(a, op=op, inplace=inplace, out=out)


def absolute(a, *, inplace=False, out=None):
    return _basic_unary(a, op=0, inplace=inplace, out=out)


def sgn(a, *, inplace=False, out=None):
    return _basic_unary(a, op=2, inplace=inplace, out=out)


def outer(a, b):
    return torch.ops.aten.outer.default(a, b)


def rsub(a, b, alpha=1):
    from ._arithmetic import arithmetic
    return arithmetic(b, a, op=1, alpha=alpha)


def constant_pad_nd(a, pad, value=0):
    pad = tuple(pad)
    if len(pad) % 2 or len(pad) > 2 * a.ndim:
        raise RuntimeError("padding must contain pairs for no more than the input rank")
    cropped = a
    for axis in range(a.ndim - len(pad) // 2, a.ndim):
        index = 2 * (a.ndim - axis - 1)
        left, right = pad[index:index + 2]
        if left < 0:
            cropped = cropped.narrow(axis, -left, cropped.shape[axis] + left)
        if right < 0:
            cropped = cropped.narrow(axis, 0, cropped.shape[axis] + right)
    if all(amount <= 0 for amount in pad):
        return clone(cropped)
    shape = list(a.shape)
    for axis in range(a.ndim - len(pad) // 2, a.ndim):
        index = 2 * (a.ndim - axis - 1)
        shape[axis] += pad[index] + pad[index + 1]
        if shape[axis] < 0:
            raise RuntimeError("negative padding results in a negative output size")
    result = torch.empty(shape, dtype=a.dtype, device=a.device, memory_format=suggest_memory_format(a))
    fill_(result, value)
    if cropped.numel() != 0:
        destination = result
        for axis in range(a.ndim - len(pad) // 2, a.ndim):
            index = 2 * (a.ndim - axis - 1)
            left, right = pad[index:index + 2]
            if left > 0:
                destination = destination.narrow(axis, left, destination.shape[axis] - left)
            if right > 0:
                destination = destination.narrow(axis, 0, destination.shape[axis] - right)
        destination.copy_(cropped)
    return result


def addmv(bias, matrix, vector, *, beta=1, alpha=1):
    from ._blas_accumulate import addmv as native_addmv
    return native_addmv(bias, matrix, vector, beta=beta, alpha=alpha)


def addbmm(bias, a, b, *, beta=1, alpha=1):
    from ._blas_accumulate import batched
    return batched(bias, a, b, beta=beta, alpha=alpha, reduce_batch=True)


def baddbmm(bias, a, b, *, beta=1, alpha=1):
    from ._blas_accumulate import batched
    return batched(bias, a, b, beta=beta, alpha=alpha)


def hardtanh(a, min_val=-1, max_val=1, *, inplace=False):
    return _apply(51, a, _scalar(max_val, a), scalar=float(min_val), out=a if inplace else None)


def hardtanh_backward(grad, a, min_val, max_val):
    lower = _apply(4, grad, a, scalar=float(min_val))
    return _apply(52, lower, a, scalar=float(max_val))


def softplus(a, beta=1, threshold=20):
    return _apply(53, a, _scalar(threshold, a), scalar=float(beta))


def softplus_backward(grad, a, beta, threshold):
    _scalar(grad, a)
    return _ternary(54, grad, a, _scalar(threshold, a), value=beta, dtype=a.dtype)


def mse_loss(a, target, reduction=1):
    _scalar(target, a)
    if reduction not in (0, 1, 2):
        raise RuntimeError("MSE reduction must be none, mean or sum")
    if a.dtype != torch.float32:
        return mse_loss(a.float(), target.float(), reduction).to(a.dtype)
    loss = _apply(55, a, target)
    if reduction == 0:
        return loss
    return mean_dim(loss) if reduction == 1 else sum_dim(loss)


def mse_loss_backward(grad, a, target, reduction):
    _scalar(target, a)
    _scalar(grad, a)
    if reduction not in (0, 1, 2):
        raise RuntimeError("MSE reduction must be none, mean or sum")
    norm = 2.0
    if reduction == 1:
        norm = 2.0 / a.numel() if a.numel() else float("inf")
    return _ternary(56, grad, a, target, value=norm, dtype=a.dtype)


def fill_tensor_(a, value):
    if value.ndim != 0:
        raise RuntimeError("fill_ only supports a zero-dimensional value tensor")
    return a.copy_(value)


def fill(a, value):
    result = torch.empty_like(a, memory_format=torch.preserve_format)
    if isinstance(value, torch.Tensor):
        return fill_tensor_(result, value)
    return fill_(result, value)


def hardshrink(a, lambd=0.5):
    boundary = _scalar(lambd, a).to(a.dtype)
    return _apply(58, a, boundary)


def softshrink(a, lambd=0.5):
    if not 0 <= lambd <= torch.finfo(a.dtype).max:
        raise RuntimeError("softshrink lambda must be nonnegative and within the input dtype range")
    boundary = _scalar(lambd, a).to(a.dtype)
    return _apply(59, a, boundary)


def shrink_backward(grad, a, lambd):
    _scalar(grad, a)
    boundary = _scalar(lambd, a).to(a.dtype)
    return _ternary(60, grad, a, boundary, dtype=a.dtype)


def threshold(a, threshold, value, *, inplace=False):
    boundary = _scalar(threshold, a).to(a.dtype)
    replacement = _scalar(value, a).to(a.dtype)
    result = _ternary(61, replacement, a, boundary, dtype=a.dtype)
    if inplace:
        a.copy_(result)
        return a
    return result


def prelu_kernel(a, weight):
    _scalar(weight, a)
    return _apply(62, a, weight)


def threshold_backward(grad, a, threshold):
    _scalar(grad, a)
    boundary = _scalar(threshold, a).to(a.dtype)
    return _ternary(73, grad, a, boundary, dtype=a.dtype)


def prelu_kernel_backward(grad, a, weight):
    _scalar(weight, a)
    _scalar(grad, a)
    a, weight, grad = torch.broadcast_tensors(a, weight, grad)
    grad_input = _ternary(63, grad, a, weight, dtype=a.dtype)
    grad_weight = _apply(64, a, grad)
    return grad_input, grad_weight


def _piecewise_loss(a, target, reduction, boundary, *, huber):
    if reduction not in (0, 1, 2):
        raise RuntimeError("loss reduction must be none, mean or sum")
    if huber:
        if not boundary > 0:
            raise RuntimeError("huber_loss delta must be positive")
    elif not boundary >= 0:
        raise RuntimeError("smooth_l1_loss beta must be nonnegative")
    if a.device != target.device or a.dtype not in _dtypes or target.dtype not in _dtypes:
        raise RuntimeError("RUDA loss inputs must use supported floating dtypes on the same device")
    dtype = torch.promote_types(a.dtype, target.dtype)
    loss = _apply(66 if huber else 65, a.to(dtype), target.to(dtype), scalar=float(boundary))
    if reduction == 0:
        return loss
    return mean_dim(loss) if reduction == 1 else sum_dim(loss)


def smooth_l1_loss(a, target, reduction=1, beta=1.0):
    return _piecewise_loss(a, target, reduction, beta, huber=False)


def huber_loss(a, target, reduction=1, delta=1.0):
    return _piecewise_loss(a, target, reduction, delta, huber=True)


def _piecewise_loss_backward(grad, a, target, reduction, boundary, *, huber):
    _scalar(target, a)
    _scalar(grad, a)
    if reduction not in (0, 1, 2):
        raise RuntimeError("loss reduction must be none, mean or sum")
    norm = 1.0
    if reduction == 1:
        norm = 1.0 / a.numel() if a.numel() else float("inf")
    normalization = _scalar(norm, a).to(a.dtype)
    difference = add(a, target, alpha=-1)
    return _ternary(68 if huber else 67, normalization, difference, grad,
                    value=boundary, dtype=a.dtype)


def smooth_l1_loss_backward(grad, a, target, reduction, beta):
    return _piecewise_loss_backward(grad, a, target, reduction, beta, huber=False)


def huber_loss_backward(grad, a, target, reduction, delta):
    return _piecewise_loss_backward(grad, a, target, reduction, delta, huber=True)


def _glu_halves(a, dim):
    if a.ndim == 0:
        raise RuntimeError("glu does not support zero-dimensional tensors")
    dim = _dimension(dim, a.ndim)
    if a.shape[dim] % 2:
        raise RuntimeError("glu requires an even size along the split dimension")
    width = a.shape[dim] // 2
    return a.narrow(dim, 0, width), a.narrow(dim, width, width), dim


def glu(a, dim=-1):
    from ._activation_extra import glu as native_glu
    return native_glu(a, dim)


def glu_backward(grad, a, dim):
    from ._activation_extra import glu_backward as native_glu_backward
    return native_glu_backward(grad, a, dim)


def _adaptive_avg_pool(a, output_size, spatial_dims):
    output_size = tuple(output_size)
    if a.ndim not in (spatial_dims + 1, spatial_dims + 2):
        raise RuntimeError("adaptive average pooling input rank does not match its spatial dimensions")
    if len(output_size) != spatial_dims or any(size < 0 for size in output_size):
        raise RuntimeError("adaptive average pooling requires nonnegative spatial output sizes")
    if any(size == 0 for size in a.shape[-spatial_dims:]):
        raise RuntimeError("adaptive average pooling requires nonempty input spatial dimensions")
    if spatial_dims == 3 and any(size == 0 for size in a.shape[1:]):
        raise RuntimeError("adaptive_avg_pool3d requires nonempty non-batch dimensions")
    shape = tuple(a.shape[:-spatial_dims]) + output_size
    result = torch.empty(shape, device=a.device, dtype=a.dtype, memory_format=suggest_memory_format(a))
    if result.numel() != 0:
        _C.execute(74 if spatial_dims == 2 else 76, a, a, result, 0.0)
    return result


def adaptive_avg_pool2d(a, output_size):
    return _adaptive_avg_pool(a, output_size, 2)


def adaptive_avg_pool3d(a, output_size):
    return _adaptive_avg_pool(a, output_size, 3)


def _adaptive_avg_pool_backward(grad, a, spatial_dims):
    _scalar(grad, a)
    if a.ndim not in (spatial_dims + 1, spatial_dims + 2) or grad.ndim != a.ndim:
        raise RuntimeError("adaptive average pooling gradient rank must match the input")
    if grad.shape[:-spatial_dims] != a.shape[:-spatial_dims]:
        raise RuntimeError("adaptive average pooling gradient batch and channel dimensions must match the input")
    if any(size == 0 for size in grad.shape[1:]):
        raise RuntimeError("adaptive average pooling requires nonempty gradient non-batch dimensions")
    result = torch.empty_like(a, memory_format=torch.contiguous_format)
    if result.numel() != 0:
        _C.execute(75 if spatial_dims == 2 else 77, grad, grad, result, 0.0)
    return result


def adaptive_avg_pool2d_backward(grad, a):
    return _adaptive_avg_pool_backward(grad, a, 2)


def adaptive_avg_pool3d_backward(grad, a):
    return _adaptive_avg_pool_backward(grad, a, 3)


def _typed_value(value, dtype, device):
    if dtype not in _storage_dtypes:
        raise RuntimeError("RUDA does not support the promoted dtype")
    if isinstance(value, torch.Tensor):
        if value.device != device:
            raise RuntimeError("RUDA operands must be on the same device")
        return value.to(dtype)
    result = torch.empty((), dtype=dtype, device=device)
    return fill_(result, value)


def arange(start, end=None, step=1, *, dtype=None, layout=None, device=None,
           pin_memory=False, out=None):
    if end is None:
        start, end = 0, start
    if layout not in (None, torch.strided) or pin_memory:
        raise RuntimeError("RUDA arange requires strided device storage")
    if out is None:
        dtype = dtype or (torch.int64 if all(isinstance(v, numbers.Integral)
                                           for v in (start, end, step)) else torch.get_default_dtype())
        out = torch.empty(0, dtype=dtype, device=device or 'ruda')
    if out.dtype not in _storage_dtypes or out.dtype == torch.bool or out.is_complex():
        raise RuntimeError("RUDA arange requires a supported numeric dtype")
    if not all(math.isfinite(v) for v in (start, end, step)):
        raise RuntimeError("arange requires finite bounds and step")
    if step == 0:
        raise RuntimeError("step must be nonzero")
    if (step > 0 and end < start) or (step < 0 and end > start):
        raise RuntimeError("upper bound and larger bound inconsistent with step sign")
    bounds = (start, end, step)
    integral = out.dtype not in _dtypes
    if integral:
        start, end, step = int(start), int(end), int(step)
        if any(v < -(1 << 63) or v >= (1 << 63) for v in (start, end, step)):
            raise RuntimeError("arange scalar exceeds int64 range")
    else:
        start, end, step = float(start), float(end), float(step)
    if integral and out.dtype == torch.int64:
        if step == 0:
            raise RuntimeError("step must be nonzero after int64 conversion")
        distance = abs(end - start)
        length = (distance + abs(step) - 1) // abs(step)
    else:
        first, last, increment = map(float, bounds)
        length = math.ceil((last - first) / increment)
    if length < 0 or length >= (1 << 63):
        raise RuntimeError("invalid size, possible overflow")
    if out.numel() != length:
        _C.prepare_index_output(out, [length])
    if length:
        accumulator = torch.int64 if integral else torch.float64 if out.dtype == torch.float64 else torch.float32
        first = _typed_value(start, accumulator, out.device)
        increment = _typed_value(step, accumulator, out.device)
        _C.execute(108, first, increment, out, 0.0)
    return out


def _comparison(op, a, b, *, inplace=False, out=None):
    from ._predicates import predicate
    return predicate(a, b, op=op, inplace=inplace, out=out)


def _logical(op, a, b=None, *, inplace=False, out=None):
    from ._predicates import predicate
    return predicate(a, b, op=op, inplace=inplace, out=out)


def where(condition, a, b, *, out=None):
    from ._conditional_select import where as native_where
    return native_where(condition, a, b, out=out)


def masked_fill(a, mask, value, *, inplace=False):
    from ._conditional_select import masked_fill as native_masked_fill
    return native_masked_fill(a, mask, value, inplace=inplace)


def _primitive_operands(a, b):
    dtype = torch.result_type(a, b)
    left = _typed_value(a, dtype, a.device)
    right = (_typed_value(b, torch.int64, a.device).to(dtype)
             if isinstance(b, int) and dtype not in _dtypes else _typed_value(b, dtype, a.device))
    return (*torch.broadcast_tensors(left, right), dtype)


def _primitive_result(result, a, inplace):
    if inplace:
        if result.shape != a.shape or not torch.can_cast(result.dtype, a.dtype):
            raise RuntimeError("RUDA in-place result cannot change shape or cast to the input dtype")
        a.copy_(result)
        return a
    return result


def _integral_arithmetic(op, a, b, *, alpha=1, inplace=False):
    if inplace:
        _C.check_inplace(a, b if isinstance(b, torch.Tensor) else a)
    left, right, dtype = _primitive_operands(a, b)
    if op != 2:
        if isinstance(alpha, bool) and dtype != torch.bool:
            raise RuntimeError("Boolean alpha only supported for Boolean results")
        if dtype not in _numeric_dtypes and not isinstance(alpha, int):
            raise RuntimeError("For integral input tensors alpha must be integral")
    if inplace and (left.shape != a.shape or not torch.can_cast(dtype, a.dtype)):
        raise RuntimeError("RUDA in-place result cannot change shape or cast to the input dtype")
    if dtype in _numeric_dtypes:
        result = _binary(1 if op == 94 else op, left, right,
                         scalar=-alpha if op == 94 else alpha if op == 1 else 0)
    elif dtype == torch.bool:
        if op == 1 and not alpha:
            result = clone(left)
        else:
            result = _logical(86 if op == 1 else 85, left, right)
    else:
        if op != 2 and alpha != 1:
            factor = _typed_value(alpha, torch.int64, a.device).to(dtype)
            right = _apply(95, right, factor)
        result = _apply(93 if op == 1 else 94 if op == 94 else 95, left, right)
    return _primitive_result(result, a, inplace)


def sub(a, b, alpha=1, *, inplace=False):
    from ._arithmetic import arithmetic
    return arithmetic(a, b, op=1, alpha=alpha, inplace=inplace)


def neg(a, *, inplace=False):
    from ._arithmetic import arithmetic
    return arithmetic(a, op=3, inplace=inplace)


def bitwise(op, a, b=None, *, inplace=False):
    from ._bitwise import elementwise
    return elementwise(a, b, op=op, inplace=inplace)


def flip(a, dims):
    dimensions = tuple(_dimension(dim, a.ndim) for dim in dims)
    if len(set(dimensions)) != len(dimensions):
        raise RuntimeError("duplicate flip dimension")
    if not dimensions or a.ndim == 0:
        return clone(a)
    result = a
    for dim in dimensions:
        output = _out(a.shape, a)
        _apply(97, result, result, out=output, scalar=float(dim))
        result = output
    return result if a.is_contiguous() else torch.empty_like(a, memory_format=torch.preserve_format).copy_(result)


def _primitive_reduce(op, a, dim=None, keepdim=False, *, dtype=None):
    dtype = dtype or (a.dtype if a.dtype in _dtypes else torch.int64)
    if dtype not in _storage_dtypes or dtype == torch.bool:
        raise NotImplementedError("RUDA primitive reduction requires a supported numeric output dtype")
    if dim is None or isinstance(dim, (tuple, list)) and not dim:
        dimensions = tuple(range(a.ndim))
    else:
        dimensions = tuple(_dimension(d, a.ndim) for d in (dim if isinstance(dim, (tuple, list)) else (dim,)))
        if len(set(dimensions)) != len(dimensions):
            raise RuntimeError("duplicate reduction dimension")
    result = a.to(dtype)
    if dtype in (torch.float16, torch.bfloat16):
        result = result.float()
    if a.ndim == 0:
        return clone(result).to(dtype)
    for axis in dimensions:
        shape = list(result.shape)
        shape[axis] = 1
        output = _out(shape, result)
        if result.shape[axis] == 0:
            fill_(output, 1 if op == 98 else 0)
        else:
            _C.execute(op, result, result, output, float(axis))
        result = output
    if not keepdim:
        result = result.view(tuple(size for axis, size in enumerate(a.shape) if axis not in dimensions))
    return result.to(dtype)


def prod(a, dim=None, keepdim=False, *, dtype=None):
    from ._product_reduce import product_reduce
    return product_reduce(a, dim, keepdim, dtype=dtype)


def _boolean_reduce(a, dim=None, keepdim=False, *, every):
    from ._logical_reduce import boolean_reduce
    return boolean_reduce(a, dim, keepdim, every=every)


def _cumulative(op, a, dim, *, dtype=None):
    from ._cumulative import cumulative
    return cumulative(a, dim, dtype=dtype, product=op == 101)


for name, function in {
    "fill_.Scalar": fill_, "zero_": lambda a: fill_(a, 0),
    "fill_.Tensor": fill_tensor_, "fill.Scalar": fill, "fill.Tensor": fill,
    "zero": lambda a: fill(a, 0),
    "where.self": where, "where.self_out": where,
    "masked_fill.Scalar": masked_fill, "masked_fill.Tensor": masked_fill,
    "masked_fill_.Scalar": lambda a, mask, value: masked_fill(a, mask, value, inplace=True),
    "masked_fill_.Tensor": lambda a, mask, value: masked_fill(a, mask, value, inplace=True),
    "logical_and": lambda a, b: _logical(85, a, b),
    "logical_or": lambda a, b: _logical(86, a, b),
    "logical_xor": lambda a, b: _logical(87, a, b),
    "logical_not": lambda a: _logical(88, a),
    "logical_and.out": lambda a, b, *, out: _logical(85, a, b, out=out),
    "logical_or.out": lambda a, b, *, out: _logical(86, a, b, out=out),
    "logical_xor.out": lambda a, b, *, out: _logical(87, a, b, out=out),
    "logical_not.out": lambda a, *, out: _logical(88, a, out=out),
    "logical_and_": lambda a, b: _logical(85, a, b, inplace=True),
    "logical_or_": lambda a, b: _logical(86, a, b, inplace=True),
    "logical_xor_": lambda a, b: _logical(87, a, b, inplace=True),
    "logical_not_": lambda a: _logical(88, a, inplace=True),
    "add.Tensor": add, "add.Scalar": add, "add_.Tensor": add_, "add_.Scalar": add_,
    "add.out": partial(_basic_binary_out, op=0), "add.Scalar_out": partial(_basic_binary_out, op=0),
    "mul.Tensor": mul, "mul.Scalar": mul, "mul_.Tensor": mul_, "mul_.Scalar": mul_,
    "mul.out": mul_out,
    "mul.Scalar_out": partial(_basic_binary_out, op=2),
    "addcmul": addcmul, "addcmul_": addcmul_, "addcdiv": addcdiv, "addcdiv_": addcdiv_,
    "lerp.Scalar": lerp, "lerp.Tensor": lerp,
    "lerp_.Scalar": lambda a, end, weight: lerp(a, end, weight, inplace=True),
    "lerp_.Tensor": lambda a, end, weight: lerp(a, end, weight, inplace=True),
    "div.Tensor": div, "div.Scalar": div, "div.Tensor_mode": div, "div.Scalar_mode": div,
    "div_.Tensor": div_, "div_.Scalar": div_, "div_.Tensor_mode": div_, "div_.Scalar_mode": div_,
    "div.out": partial(_basic_binary_out, op=3), "div.Scalar_out": partial(_basic_binary_out, op=3),
    "div.out_mode": partial(_basic_binary_out, op=3), "div.Scalar_mode_out": partial(_basic_binary_out, op=3),
    "sub.Tensor": sub, "sub.Scalar": sub,
    "sub_.Tensor": lambda a, b, alpha=1: sub(a, b, alpha=alpha, inplace=True),
    "sub_.Scalar": lambda a, b, alpha=1: sub(a, b, alpha=alpha, inplace=True),
    "sub.out": partial(_basic_binary_out, op=1), "sub.Scalar_out": partial(_basic_binary_out, op=1),
    "rsub.Tensor": rsub, "rsub.Scalar": rsub,
    "neg": neg,
    "neg_": lambda a: neg(a, inplace=True),
    "neg.out": _neg_out,
    "mm": mm, "addmm": addmm, "bmm": bmm,
    "mv": mv, "dot": dot, "outer": outer, "cat": cat, "stack": stack,
    "cat.out": lambda tensors, dim=0, *, out: _math_out(cat(tensors, dim), out),
    "ger": outer, "vdot": vdot,
    "addmv": addmv, "addbmm": addbmm, "baddbmm": baddbmm,
    "constant_pad_nd": constant_pad_nd,
    "relu": lambda a: _apply(3, a),
    "relu_": lambda a: _apply(3, a, out=a),
    "sum": lambda a, dtype=None: sum_dim(a, dtype=dtype), "sum.dim_IntList": sum_dim,
    "mean": lambda a, dtype=None: mean_dim(a, dtype=dtype), "mean.dim": mean_dim,
    "exp": partial(_exponential_unary, op=0), "log": partial(_exponential_unary, op=1),
    "exp_": partial(_exponential_unary, op=0, inplace=True), "log_": partial(_exponential_unary, op=1, inplace=True),
    "exp.out": partial(_exponential_unary, op=0), "log.out": partial(_exponential_unary, op=1),
    "sqrt": partial(_basic_unary, op=7), "rsqrt": partial(_basic_unary, op=8),
    "sqrt_": partial(_basic_unary, op=7, inplace=True), "rsqrt_": partial(_basic_unary, op=8, inplace=True),
    "sqrt.out": partial(_basic_unary, op=7), "rsqrt.out": partial(_basic_unary, op=8),
    "sigmoid": partial(_exponential_unary, op=3), "sigmoid_backward": activation_backward,
    "sigmoid_backward.grad_input": activation_backward,
    "sigmoid_": partial(_exponential_unary, op=3, inplace=True),
    "sigmoid.out": partial(_exponential_unary, op=3),
    "tanh_backward": partial(activation_backward, tanh=True),
    "tanh_backward.grad_input": partial(activation_backward, tanh=True),
    "abs": absolute, "abs_": lambda a: absolute(a, inplace=True),
    "abs.out": absolute,
    "sgn.out": sgn,
    "sign": partial(_basic_unary, op=1), "sign_": partial(_basic_unary, op=1, inplace=True),
    "sign.out": partial(_basic_unary, op=1),
    "sgn": sgn, "sgn_": lambda a: sgn(a, inplace=True),
    "floor": partial(_basic_unary, op=3), "floor_": partial(_basic_unary, op=3, inplace=True),
    "floor.out": partial(_basic_unary, op=3),
    "ceil": partial(_basic_unary, op=4), "ceil_": partial(_basic_unary, op=4, inplace=True),
    "ceil.out": partial(_basic_unary, op=4),
    "trunc": partial(_basic_unary, op=5), "trunc_": partial(_basic_unary, op=5, inplace=True),
    "trunc.out": partial(_basic_unary, op=5),
    "round": partial(_basic_unary, op=6), "round_": partial(_basic_unary, op=6, inplace=True),
    "round.out": partial(_basic_unary, op=6),
    "reciprocal": partial(_basic_unary, op=9), "reciprocal_": partial(_basic_unary, op=9, inplace=True),
    "reciprocal.out": partial(_basic_unary, op=9),
    "log1p": partial(_exponential_unary, op=2), "log1p_": partial(_exponential_unary, op=2, inplace=True),
    "log1p.out": partial(_exponential_unary, op=2),
    "glu": glu, "glu_backward": glu_backward,
    "_softmax": softmax, "_softmax_backward_data": softmax_backward,
    "_softmax.out": softmax,
    "_softmax_backward_data.out": lambda grad, y, dim, dtype, *, grad_input: softmax_backward(grad, y, dim, dtype, out=grad_input),
    "_log_softmax": lambda a, dim, half_to_float=False, **kw: softmax(a, dim, half_to_float, logarithmic=True, **kw),
    "_log_softmax.out": lambda a, dim, half_to_float=False, **kw: softmax(a, dim, half_to_float, logarithmic=True, **kw),
    "_log_softmax_backward_data": lambda grad, y, dim, dtype, **kw: softmax_backward(grad, y, dim, dtype, logarithmic=True, **kw),
    "_log_softmax_backward_data.out": lambda grad, y, dim, dtype, **kw: softmax_backward(grad, y, dim, dtype, logarithmic=True, **kw),
    "pow.Tensor_Scalar": pow_scalar, "pow.Tensor_Tensor": pow_tensor, "pow.Scalar": pow_scalar_base,
    "pow_.Scalar": _power_inplace, "pow_.Tensor": _power_inplace,
    "pow.Tensor_Scalar_out": pow_scalar,
    "pow.Tensor_Tensor_out": pow_tensor,
    "pow.Scalar_out": pow_scalar_base,
    "floor_divide": lambda a, b: div(a, b, rounding_mode='floor'),
    "floor_divide.Scalar": lambda a, b: div(a, b, rounding_mode='floor'),
    "floor_divide_.Tensor": lambda a, b: div_(a, b, rounding_mode='floor'),
    "floor_divide_.Scalar": lambda a, b: div_(a, b, rounding_mode='floor'),
    "floor_divide.out": lambda a, b, *, out: div(a, b, rounding_mode='floor', out=out),
    "floor_divide.Scalar_out": lambda a, b, *, out: div(a, b, rounding_mode='floor', out=out),
    "remainder.Tensor": lambda a, b: _floating_modulus(a, b, floor=True),
    "remainder.Scalar": lambda a, b: _floating_modulus(a, b, floor=True),
    "remainder.Scalar_Tensor": lambda a, b: _scalar_modulus(a, b, floor=True),
    "remainder_.Tensor": lambda a, b: _floating_modulus(a, b, floor=True, inplace=True),
    "remainder_.Scalar": lambda a, b: _floating_modulus(a, b, floor=True, inplace=True),
    "remainder.Tensor_out": lambda a, b, *, out: _floating_modulus(a, b, floor=True, out=out),
    "remainder.Scalar_out": lambda a, b, *, out: _floating_modulus(a, b, floor=True, out=out),
    "remainder.Scalar_Tensor_out": lambda a, b, *, out: _scalar_modulus(a, b, floor=True, out=out),
    "fmod.Tensor": lambda a, b: _floating_modulus(a, b, floor=False),
    "fmod.Scalar": lambda a, b: _floating_modulus(a, b, floor=False),
    "fmod_.Tensor": lambda a, b: _floating_modulus(a, b, floor=False, inplace=True),
    "fmod_.Scalar": lambda a, b: _floating_modulus(a, b, floor=False, inplace=True),
    "fmod.Tensor_out": lambda a, b, *, out: _floating_modulus(a, b, floor=False, out=out),
    "fmod.Scalar_out": lambda a, b, *, out: _floating_modulus(a, b, floor=False, out=out),
    "clone": clone,
    "flip": flip, "prod": prod, "prod.dim_int": prod,
    "all": lambda a: _boolean_reduce(a, every=True),
    "all.dim": lambda a, dim, keepdim=False: _boolean_reduce(a, dim, keepdim, every=True),
    "all.dims": lambda a, dim=None, keepdim=False: _boolean_reduce(a, dim, keepdim, every=True),
    "any": lambda a: _boolean_reduce(a, every=False),
    "any.dim": lambda a, dim, keepdim=False: _boolean_reduce(a, dim, keepdim, every=False),
    "any.dims": lambda a, dim=None, keepdim=False: _boolean_reduce(a, dim, keepdim, every=False),
    "cumsum": lambda a, dim, dtype=None: _cumulative(100, a, dim, dtype=dtype),
    "cumprod": lambda a, dim, dtype=None: _cumulative(101, a, dim, dtype=dtype),
    "bitwise_not": lambda a: bitwise(92, a),
    "bitwise_not_": lambda a: bitwise(92, a, inplace=True),
}.items():
    if name not in _native_eager_ops:
        _registry.impl(name, function)


for name, op in (("eq", 78), ("ne", 79), ("lt", 80), ("le", 81), ("gt", 82), ("ge", 83)):
    for overload in ("Tensor", "Scalar"):
        _registry.impl(f"{name}.{overload}", lambda a, b, op=op: _comparison(op, a, b))
        _registry.impl(f"{name}_.{overload}", lambda a, b, op=op: _comparison(op, a, b, inplace=True))
        _registry.impl(f"{name}.{overload}_out", lambda a, b, *, out, op=op: _comparison(op, a, b, out=out))

for name, op in (("bitwise_and", 89), ("bitwise_or", 90), ("bitwise_xor", 91)):
    for overload in ("Tensor", "Scalar"):
        _registry.impl(f"{name}.{overload}", lambda a, b, op=op: bitwise(op, a, b))
        _registry.impl(f"{name}_.{overload}", lambda a, b, op=op: bitwise(op, a, b, inplace=True))
    _registry.impl(f"{name}.Scalar_Tensor", lambda a, b, op=op: bitwise(op, b, a))

_registry.impl("arange", lambda end, **kwargs: arange(0, end, **kwargs))
_registry.impl("arange.start", arange)
_registry.impl("arange.start_step", arange)
_registry.impl("arange.out", lambda end, *, out: arange(0, end, out=out))
_registry.impl("arange.start_out", arange)

from . import _indexing
from . import _ordering
from . import _scatter_reduce
from . import _statistics
from . import _spatial
from . import _logical_reduce
from . import _product_reduce
from . import _additive_reduce
