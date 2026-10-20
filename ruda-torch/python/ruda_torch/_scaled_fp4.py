import torch

from . import _C
from ._indexing import _logical
from .math_backend import _nvidia


def scales(a, b, sa, sb, group):
    dtype = torch.float8_e4m3fn if group == 16 else torch.float8_e8m0fnu
    for scale, outer, packed_k in ((sa, a.shape[0], a.shape[1]), (sb, b.shape[1], b.shape[0])):
        columns = (packed_k * 2 + group * 4 - 1) // (group * 4) * 4
        count = (outer + 127) // 128 * 128 * columns
        if scale.dtype != dtype or not scale.is_contiguous() or scale.numel() != count:
            raise ValueError(f'FP4 block{group} requires a contiguous {dtype} scale with {count} SWIZZLE_32_4_4 elements')


def run(a, b, sa, sb, bias, dtype, group, ga=None, gb=None):
    m, packed_k, n = a.shape[0], a.shape[1], b.shape[1]
    if m == 0 or packed_k == 0 or n == 0:
        result = torch.empty((m, n), dtype=dtype, device=a.device)
        return result.zero_() if packed_k == 0 else result
    left, right, sa, sb = map(_logical, (a, b, sa, sb))
    global_scale = ga is not None
    if global_scale:
        ga, gb = _logical(ga).reshape(-1), _logical(gb).reshape(-1)
    bias_value = _logical(bias).reshape(-1).contiguous() if bias is not None else None
    if _nvidia():
        if group == 32:
            raise NotImplementedError('explicit NVIDIA MXFP4 requires a block32 FP4 implementation; cuBLASLt only supports block16 FP4')
        from ._vendor_blas import scaled_mm
        alpha = ga * gb if global_scale else None
        return scaled_mm(left, right, sa, sb, bias_value, dtype, 7, 7, False, alpha=alpha)
    dummy = torch.empty(0, dtype=torch.float32, device=a.device)
    result = torch.empty((m, n), dtype=dtype, device=a.device)
    _C.scientific_call(500, [left, right, sa, sb, ga if global_scale else dummy,
                           gb if global_scale else dummy, bias_value if bias_value is not None else dummy, result],
                       [group, int(global_scale), int(bias is not None)], [])
    return result


def v2(a, b, scale_a, recipe_a, swizzle_a, scale_b, recipe_b, swizzle_b, bias, dtype, out_dtype):
    ra, rb = tuple(value & 255 for value in recipe_a), tuple(value & 255 for value in recipe_b)
    if ra != rb or ra not in ((2,), (2, 0), (3,)) or len(scale_a) != len(ra) or len(scale_b) != len(rb):
        raise ValueError('FP4 requires paired block16 E4M3 scales, optional tensorwise FP32 scales, or block32 E8M0 scales')
    if len(swizzle_a) != len(ra) or len(swizzle_b) != len(rb):
        raise ValueError('FP4 swizzle list lengths must match their scale recipes')
    if swizzle_a[0] & 255 != 1 or swizzle_b[0] & 255 != 1:
        raise ValueError('FP4 block scales require SWIZZLE_32_4_4')
    group = 32 if ra == (3,) else 16
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError('FP4 scaled_mm output must be FP16/BF16/FP32')
    sa, sb = scale_a[0], scale_b[0]
    scales(a, b, sa, sb, group)
    ga = gb = None
    if len(ra) == 2:
        ga, gb = scale_a[1], scale_b[1]
        if any(scale.dtype != torch.float32 or scale.numel() != 1 or scale.ndim > 1 for scale in (ga, gb)):
            raise ValueError('NVFP4 global scales must be FP32 scalars or one-element vectors')
    if group == 32:
        if dtype != (torch.bfloat16 if out_dtype is None else out_dtype):
            raise ValueError('MXFP4 output must match out_dtype, or BF16 when out_dtype is omitted')
        if dtype != torch.bfloat16:
            raise NotImplementedError('MXFP4 requires BF16 output')
        bias = None
    return run(a, b, sa, sb, bias, dtype, group, ga, gb)
