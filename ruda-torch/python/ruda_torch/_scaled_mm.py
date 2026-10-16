import torch

from . import _C
from ._indexing import _logical
from ._ordering import _output
from .math_backend import _nvidia

_fp8 = (torch.float8_e4m3fn, torch.float8_e5m2)
_fp4 = torch.float4_e2m1fn_x2
_outputs = (torch.float16, torch.bfloat16, torch.float32, *_fp8)


def _mx_scales(a, b, sa, sb):
    for scale, outer, inner in ((sa, a.shape[0], a.shape[1]), (sb, b.shape[1], b.shape[0])):
        count = ((outer + 127) // 128 * 128) * ((inner + 127) // 128 * 4)
        if scale.dtype != torch.float8_e8m0fnu or not scale.is_contiguous() or scale.numel() != count:
            raise RuntimeError(f'MXFP8 requires a contiguous E8M0 scale tensor with {count} elements in SWIZZLE_32_4_4 layout')


def _scaling(a, b, sa, sb):
    if a.dtype == b.dtype == _fp4:
        from ._scaled_fp4 import scales
        scales(a, b, sa, sb, 16)
        return 7, 7
    if sa.dtype == sb.dtype == torch.float8_e8m0fnu:
        _mx_scales(a, b, sa, sb)
        return 6, 6
    def matches(tensor, scale, mode):
        if scale.dtype != torch.float32:
            return False
        rows, inner = tensor.shape
        if mode == 0:
            return scale.numel() == 1
        if mode == 1:
            return tuple(scale.shape) == (rows, 1) and scale.is_contiguous()
        if scale.ndim != 2:
            return False
        columns = (inner + 127) // 128
        if mode == 2:
            shape, strides = (rows, columns), (1, rows)
        else:
            shape, strides = ((rows + 127) // 128, columns), (columns, 1)
        return tuple(scale.shape) == shape and all(size <= 1 or actual == expected
                                                  for size, actual, expected in zip(shape, scale.stride(), strides))
    transposed_b, transposed_scale_b = b.t(), sb.t()
    for ma, mb in ((0, 0), (1, 1), (3, 2), (2, 3), (2, 2)):
        if matches(a, sa, ma) and matches(transposed_b, transposed_scale_b, mb):
            return ma, mb
    raise NotImplementedError('_scaled_mm requires paired FP32 tensor/row/block128 scales, E8M0 block32 FP8 scales or E4M3 block16 FP4 scales')


def scaled_mm(a, b, scale_a, scale_b, bias=None, scale_result=None, out_dtype=None, use_fast_accum=False, *, out=None):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise RuntimeError('_scaled_mm requires compatible two-dimensional matrices')
    if not (a.dtype in _fp8 and b.dtype in _fp8 or a.dtype == b.dtype == _fp4):
        raise NotImplementedError('_scaled_mm requires FP8 matrices or two packed E2M1 matrices')
    if a.dtype == b.dtype == torch.float8_e5m2:
        raise RuntimeError('_scaled_mm does not support two E5M2 operands')
    if a.device.type != torch._C._get_privateuse1_backend_name():
        raise RuntimeError('_scaled_mm requires RUDA tensors')
    tensors = (a, b, scale_a, scale_b, *((bias,) if bias is not None else ()),
               *((scale_result,) if scale_result is not None else ()))
    for tensor in tensors:
        if tensor.device != a.device or tensor.layout != torch.strided:
            raise RuntimeError('_scaled_mm requires strided tensors on the same RUDA device')
    ma, mb = _scaling(a, b, scale_a, scale_b)
    m, k, n = a.shape[0], a.shape[1], b.shape[1]
    if k % 16 or n % 16:
        raise RuntimeError('_scaled_mm requires K and N divisible by 16')
    if a.dtype == _fp4 and use_fast_accum:
        raise RuntimeError('_scaled_mm does not support fast accumulation for packed FP4')
    dtype = out.dtype if out is not None else a.dtype if out_dtype is None else out_dtype
    if out_dtype is not None and dtype != out_dtype:
        raise RuntimeError('_scaled_mm out_dtype must match the output tensor dtype')
    empty_fp4 = a.dtype == dtype == _fp4 and k != 0 and (m == 0 or n == 0)
    if dtype not in _outputs and not empty_fp4:
        raise NotImplementedError('_scaled_mm output must be FP16, BF16, FP32, E4M3FN or E5M2')
    if a.dtype == _fp4 and dtype not in (torch.float16, torch.bfloat16, torch.float32) and not empty_fp4:
        raise NotImplementedError('FP4 _scaled_mm output must be FP16/BF16/FP32')
    if scale_result is not None and (scale_result.dtype != torch.float32 or scale_result.numel() != 1):
        raise RuntimeError('_scaled_mm scale_result must be an FP32 singleton')
    if bias is not None:
        if bias.numel() != n or bias.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError('_scaled_mm bias must contain N FP16/BF16 values')
        if dtype == torch.float32 or dtype in (torch.float16, torch.bfloat16) and bias.dtype != dtype:
            raise RuntimeError('_scaled_mm bias must match FP16/BF16 output; FP32 output does not support bias')
    if out is not None:
        if out.device != a.device or out.layout != torch.strided:
            raise RuntimeError('_scaled_mm output must be strided and on the input device')
        for tensor in tensors:
            _C.check_inplace(out, tensor)
    if m != 0 and k != 0 and n != 0 and 2 <= ma <= 5 and use_fast_accum:
        raise RuntimeError('_scaled_mm does not support fast accumulation with 128-element block scales')
    result = _run(a, b, scale_a, scale_b, bias, dtype, ma, mb, use_fast_accum)
    return _output(result, out, tensors)


def _run(a, b, scale_a, scale_b, bias, dtype, ma, mb, use_fast_accum):
    if a.dtype == _fp4:
        from ._scaled_fp4 import run
        return run(a, b, scale_a, scale_b, bias, dtype, 16)
    m, k, n = a.shape[0], a.shape[1], b.shape[1]
    left, right, sa, sb = map(_logical, (a, b, scale_a, scale_b))
    bias_value = (_logical(bias).reshape(-1).contiguous() if bias is not None else sa if sa.dtype == torch.float32
                  else torch.empty(0, dtype=torch.float32, device=a.device))
    if _nvidia() and m != 0 and k != 0 and n != 0:
        from ._vendor_blas import scaled_mm as vendor_scaled_mm
        result = vendor_scaled_mm(left, right, sa, sb, bias_value if bias is not None else None,
                                  dtype, ma, mb, use_fast_accum)
    else:
        result = torch.empty((m, n), dtype=dtype, device=a.device)
        _C.scientific_call(499, [left, right, sa, sb, bias_value, result],
                           [ma, mb, int(use_fast_accum and k != 0 and m != 0 and n != 0), int(bias is not None)], [])
    return result
