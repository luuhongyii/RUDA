import torch

from . import _C
from ._ordering import _output
from ._scaled_mm import _fp8, _fp4, _outputs, _run, _mx_scales


def _recipe(a, b, scale_a, recipe_a, scale_b, recipe_b):
    if any(len(values) != 1 for values in (scale_a, recipe_a, scale_b, recipe_b)):
        raise NotImplementedError('_scaled_mm_v2 currently supports one FP32/E8M0 scale and one recipe per operand')
    ra, rb = recipe_a[0] & 255, recipe_b[0] & 255
    sa, sb = scale_a[0], scale_b[0]
    if (ra, rb) == (3, 3) and sa.dtype == sb.dtype == torch.float8_e8m0fnu:
        if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
            raise ValueError('_scaled_mm_v2 MXFP8 recipes require E4M3FN operands')
        _mx_scales(a, b, sa, sb)
        return 6, 6, sa, sb
    if sa.dtype != torch.float32 or sb.dtype != torch.float32:
        raise NotImplementedError('_scaled_mm_v2 FP8 inputs require paired FP32 recipes or E8M0 block32 recipes')
    m, k, n = a.shape[0], a.shape[1], b.shape[1]
    if (ra, rb) == (0, 0):
        if sa.numel() != 1 or sb.numel() != 1:
            raise ValueError('tensorwise scales must each contain one FP32 value')
        return 0, 0, sa, sb
    if (ra, rb) == (1, 1):
        if sa.ndim < 2 or sa.shape[0] != m or sa.shape[1] != 1 or sa.numel() != m:
            raise ValueError('rowwise scale_a must have M rows and singleton remaining dimensions')
        if sb.ndim < 2 or sb.numel() != n or sb.stride(1) != 1:
            raise ValueError('rowwise scale_b must contain N values with stride(1) equal to one')
        return 1, 1, sa.as_strided((m, 1), (1, 1)), sb.as_strided((1, n), (n, 1))
    if (ra, rb) not in ((4, 4), (4, 5), (5, 4)):
        raise NotImplementedError('_scaled_mm_v2 requires paired tensor/row or 1x128/128x128 recipes')
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        raise ValueError('_scaled_mm_v2 block128 recipes require E4M3FN operands')
    groups = (k + 127) // 128
    for scale, recipe, outer in ((sa, ra, m), (sb, rb, n)):
        rows, columns = (outer, groups) if recipe == 4 else ((groups + 3) // 4 * 4, (outer + 127) // 128)
        if scale.ndim != 2 or tuple(scale.shape) != (rows, columns):
            raise ValueError(f'block128 scale must have shape {(rows, columns)}')
        if scale.stride(0) != 1 or not (scale.stride(1) == rows or columns == 1 and scale.stride(1) == 1):
            raise ValueError(f'block128 scale must have strides (1, {rows}), except a singleton second dimension')
    return ra, rb, sa, sb


def scaled_mm_v2(a, b, scale_a, recipe_a, swizzle_a, scale_b, recipe_b, swizzle_b,
                 bias, out_dtype, contraction_dim=(), use_fast_accum=False, *, out=None):
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError('_scaled_mm_v2 requires matrix operands')
    m, k, n = a.shape[0], a.shape[1], b.shape[1]
    dtype = out.dtype if out is not None else a.dtype if out_dtype is None else out_dtype
    if m == 0 or k == 0 or n == 0:
        result = out
        if result is None:
            result = torch.empty((m, n), dtype=dtype, device=a.device)
        elif tuple(result.shape) != (m, n):
            if result.device.type == torch._C._get_privateuse1_backend_name():
                _C.prepare_index_output(result, [m, n])
            else:
                result.resize_((m, n))
        if k == 0:
            result.zero_()
        return result
    if contraction_dim:
        if len(contraction_dim) != 2:
            raise ValueError('contraction_dim must have exactly two elements')
        da, db = contraction_dim
        if not -2 <= da < 2 or not -2 <= db < 2:
            raise IndexError('_scaled_mm_v2 contraction dimension is outside a matrix')
        if a.shape[da] != b.shape[db]:
            raise ValueError('_scaled_mm_v2 contraction dimensions do not match')
    if k != b.shape[0]:
        raise ValueError('_scaled_mm_v2 stored matrix contraction dimensions must match')
    multiplier = 2 if a.dtype == _fp4 else 1
    if k * multiplier % 16 or b.shape[0] * multiplier % 16 or n % 16:
        raise ValueError('_scaled_mm_v2 requires K and N divisible by 16')
    if not (a.dtype in _fp8 and b.dtype in _fp8 or a.dtype == b.dtype == _fp4):
        raise NotImplementedError('_scaled_mm_v2 requires FP8 matrices or two packed E2M1 matrices')
    if a.dtype == _fp4 and use_fast_accum:
        raise ValueError('_scaled_mm_v2 does not support fast accumulation for packed FP4')
    if a.dtype == b.dtype == torch.float8_e5m2:
        raise ValueError('_scaled_mm_v2 does not support two E5M2 operands')
    if dtype not in _outputs:
        raise NotImplementedError('_scaled_mm_v2 output must be FP16/BF16/FP32/E4M3FN/E5M2')
    if out_dtype is not None and dtype != out_dtype:
        raise ValueError('_scaled_mm_v2 out_dtype must match out.dtype')
    if a.device.type != torch._C._get_privateuse1_backend_name():
        raise ValueError('_scaled_mm_v2 requires RUDA tensors')
    tensors = (a, b, *scale_a, *scale_b, *((bias,) if bias is not None else ()))
    for tensor in tensors:
        if tensor.device != a.device or tensor.layout != torch.strided:
            raise ValueError('_scaled_mm_v2 inputs must be strided tensors on the same RUDA device')
    if bias is not None:
        if bias.numel() != n or bias.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError('_scaled_mm_v2 bias must contain N FP16/BF16 values')
        if dtype == torch.float32 or dtype in (torch.float16, torch.bfloat16) and bias.dtype != dtype:
            raise ValueError('_scaled_mm_v2 bias must match FP16/BF16 output; FP32 output does not support bias')
    if out is not None:
        if out.device != a.device or out.layout != torch.strided:
            raise ValueError('_scaled_mm_v2 output must be strided and on the input device')
        for tensor in tensors:
            _C.check_inplace(out, tensor)
    if a.dtype == _fp4:
        from ._scaled_fp4 import v2
        result = v2(a, b, scale_a, recipe_a, swizzle_a, scale_b, recipe_b, swizzle_b, bias, dtype, out_dtype)
        return _output(result, out, tensors)
    ma, mb, sa, sb = _recipe(a, b, scale_a, recipe_a, scale_b, recipe_b)
    if ma in (4, 5) and use_fast_accum:
        raise ValueError('_scaled_mm_v2 block128 recipes do not support fast accumulation')
    if ma == 6:
        if len(swizzle_a) != 1 or len(swizzle_b) != 1 or swizzle_a[0] & 255 != 1 or swizzle_b[0] & 255 != 1:
            raise ValueError('_scaled_mm_v2 MXFP8 requires one SWIZZLE_32_4_4 value per operand')
        if dtype != (torch.bfloat16 if out_dtype is None else out_dtype):
            raise ValueError('_scaled_mm_v2 MXFP8 output must match out_dtype, or BF16 when out_dtype is omitted')
        use_fast_accum = False
    result = _run(a, b, sa, sb, bias, dtype, ma, mb, use_fast_accum)
    return _output(result, out, tensors)
