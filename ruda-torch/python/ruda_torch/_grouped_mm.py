import torch

from . import _C
from ._indexing import _logical
from .math_backend import _nvidia


def _transposed(tensor):
    rows, columns = tensor.shape[-2:]
    row_stride, column_stride = tensor.stride()[-2:]
    alignment = 16 // tensor.element_size()
    if tensor.data_ptr() % 16:
        raise ValueError('grouped_mm requires 16-byte-aligned matrix pointers')
    if row_stride == 1 and column_stride >= max(1, rows):
        if column_stride % alignment:
            raise ValueError('grouped_mm leading strides must be multiples of 16 bytes')
        return True
    if column_stride == 1 and row_stride >= max(1, columns):
        if row_stride % alignment:
            raise ValueError('grouped_mm leading strides must be multiples of 16 bytes')
        return False
    raise ValueError('grouped_mm requires row-major or column-major matrix strides')


def _structure(a, b, offs, bias, scaled=False, contraction_dim=()):
    if a.ndim not in (2, 3) or b.ndim not in (2, 3):
        raise ValueError('grouped_mm operands must be 2D or 3D')
    if a.device.type != torch._C._get_privateuse1_backend_name():
        raise ValueError('grouped_mm requires RUDA tensors')
    for tensor in (a, b, *((offs,) if offs is not None else ())):
        if tensor.device != a.device or tensor.layout != torch.strided:
            raise ValueError('grouped_mm tensors must be strided and on the same RUDA device')
    if bias is not None:
        raise ValueError('grouped_mm does not support bias')
    if (offs is not None) != (a.ndim == 2 or b.ndim == 2):
        raise ValueError('offsets are required exactly when at least one matrix is 2D')
    if offs is not None and (offs.ndim != 1 or offs.dtype != torch.int32):
        raise ValueError('grouped_mm offsets must be a 1D int32 tensor')
    if a.ndim != 2 or b.ndim != 2:
        if contraction_dim:
            if len(contraction_dim) != 2:
                raise ValueError('contraction_dim must contain two elements')
            da, db = contraction_dim[0], b.size(contraction_dim[1])
            if a.size(da) != b.size(db):
                raise ValueError('grouped_mm contraction dimensions do not match')
            if da != -1 or db != -2:
                raise ValueError('grouped_mm currently requires contraction dimensions (-1, -2)')
        elif a.shape[-1] != b.shape[-2]:
            raise ValueError('grouped_mm contraction dimensions do not match')
    ta, tb = _transposed(a), _transposed(b)
    if scaled:
        if ta or not tb:
            raise ValueError('scaled_grouped_mm requires row-major A and column-major B')
        if a.shape[-1] % 16 or b.shape[-2] % 16 or b.shape[-1] % 16:
            raise ValueError('scaled_grouped_mm stored K dimensions and N must be divisible by 16')
    groups = a.shape[0] if offs is None else offs.numel()
    if a.ndim == 3 and a.shape[0] != groups or b.ndim == 3 and b.shape[0] != groups:
        raise ValueError('grouped_mm batch sizes must match the group count')
    mode = 0 if a.ndim == b.ndim == 3 else 1 if b.ndim == 3 else 2 if a.ndim == 3 else 3
    shape = (groups, a.shape[-2], b.shape[-1]) if mode in (0, 3) else (a.shape[-2], b.shape[-1])
    return mode, groups, shape


def _scales(a, b, sa, sb, mode, groups, kind):
    dtype = torch.float32 if kind == 1 else torch.float8_e4m3fn if kind == 4 else torch.float8_e8m0fnu
    for tensor in (sa, sb):
        if tensor.device != a.device or tensor.layout != torch.strided or tensor.dtype != dtype:
            raise ValueError(f'grouped_mm requires {dtype} scales on the matrix device')
    if kind == 1:
        multiplier = groups if mode == 3 else 1
        for matrix, scale, dim in ((a, sa, 0), (b, sb, 1)):
            if matrix.ndim == 2:
                if scale.ndim != 1 or scale.numel() != matrix.shape[dim] * multiplier or not scale.is_contiguous():
                    raise ValueError('rowwise 2D grouped_mm scales must be contiguous vectors with one entry per row/column and K group')
            elif scale.ndim != 2 or tuple(scale.shape) != (groups, matrix.shape[dim + 1]) or scale.stride(1) != 1:
                raise ValueError('rowwise 3D grouped_mm scales must have shape (G, rows/columns) and unit last stride')
        return
    if mode not in (1, 3):
        raise ValueError('block-scaled grouped_mm only supports 2D x 2D and 2D x 3D')
    if not a.is_contiguous() or not b.transpose(-2, -1).is_contiguous():
        raise ValueError('block-scaled grouped_mm requires contiguous A and contiguous transposed B')
    block = 16 if kind == 4 else 32
    pack = 2 if kind in (3, 4) else 1
    for matrix, scale, dim in ((a, sa, 0), (b, sb, 1)):
        if scale.ndim != 2 or not scale.is_contiguous():
            raise ValueError('block-scaled grouped_mm scales must be contiguous 2D SWIZZLE_32_4_4 buffers')
        if matrix.ndim == 2:
            if scale.shape[0] < matrix.shape[dim]:
                raise ValueError('block-scaled grouped_mm scale leading dimension is too small')
        else:
            columns = (matrix.shape[1] * pack // block + 3) // 4 * 4
            count = (matrix.shape[2] + 127) // 128 * 128 * columns
            if tuple(scale.shape) != (groups, count):
                raise ValueError(f'3D block-scaled grouped_mm scales must have shape {(groups, count)}')


def _run(a, b, offs, mode, groups, shape, dtype, kind, fast, sa=None, sb=None, ga=None, gb=None):
    if _nvidia():
        raise NotImplementedError('explicit NVIDIA grouped_mm is not implemented; select the native RUDA math backend')
    alignment = 4 if dtype == torch.float32 else 8
    padded = (shape[-1] + alignment - 1) // alignment * alignment
    strides = (shape[-2] * padded, padded, 1) if len(shape) == 3 else (padded, 1)
    result = torch.empty_strided(shape, strides, dtype=dtype, device=a.device)
    dummy = torch.empty(0, dtype=torch.float32, device=a.device)
    alpha = dummy
    if kind == 4:
        for scale in (ga, gb):
            if scale.dtype != torch.float32 or scale.device != a.device or scale.layout != torch.strided:
                raise ValueError('NVFP4 grouped_mm global scales must be FP32 tensors on the matrix device')
        combined = _logical(ga) * _logical(gb)
        if combined.numel() < groups:
            raise ValueError('NVFP4 global scale product must contain at least one value per group')
        alpha = combined.as_strided((groups,), (1,))
    if result.numel() == 0 or groups == 0:
        return result
    offsets = torch.empty(0, dtype=torch.int32, device=a.device) if offs is None else _logical(offs).as_strided((groups,), (1,))
    _C.scientific_call(501, [_logical(a), _logical(b), offsets,
                           _logical(sa) if sa is not None else dummy,
                           _logical(sb) if sb is not None else dummy, alpha, result],
                       [mode, kind, int(fast)], [])
    return result


def grouped_mm(a, b, offs=None, bias=None, out_dtype=None):
    mode, groups, shape = _structure(a, b, offs, bias)
    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32) or b.dtype != a.dtype:
        raise ValueError('grouped_mm requires matching FP16, BF16 or FP32 inputs')
    dtype = a.dtype if out_dtype is None else out_dtype
    if dtype != a.dtype:
        raise ValueError('grouped_mm output dtype must match A')
    return _run(a, b, offs, mode, groups, shape, dtype, 0, False)


def scaled_grouped_mm(a, b, scale_a, scale_b, offs=None, bias=None, scale_result=None,
                      out_dtype=None, use_fast_accum=False):
    mode, groups, shape = _structure(a, b, offs, bias, True)
    if scale_result is not None:
        raise ValueError('scaled_grouped_mm does not support scale_result')
    if a.dtype != torch.float8_e4m3fn or b.dtype != a.dtype:
        raise ValueError('scaled_grouped_mm requires E4M3FN matrices')
    dtype = torch.bfloat16 if out_dtype is None else out_dtype
    if dtype != torch.bfloat16:
        raise ValueError('scaled_grouped_mm requires BF16 output')
    kind = 1 if scale_a.dtype == torch.float32 else 2
    _scales(a, b, scale_a, scale_b, mode, groups, kind)
    return _run(a, b, offs, mode, groups, shape, dtype, kind, use_fast_accum if kind == 1 else False, scale_a, scale_b)


def scaled_grouped_mm_v2(a, b, scale_a, recipe_a, swizzle_a, scale_b, recipe_b, swizzle_b,
                         offs=None, bias=None, out_dtype=None, contraction_dim=(), use_fast_accum=False):
    mode, groups, shape = _structure(a, b, offs, bias, True, contraction_dim)
    dtype = torch.bfloat16 if out_dtype is None else out_dtype
    if dtype != torch.bfloat16:
        raise ValueError('scaled_grouped_mm_v2 requires BF16 output')
    ra, rb = tuple(value & 255 for value in recipe_a), tuple(value & 255 for value in recipe_b)
    if ra != rb or len(scale_a) != len(ra) or len(scale_b) != len(rb):
        raise ValueError('scaled_grouped_mm_v2 requires matched scale recipes')
    ga = gb = None
    if a.dtype == b.dtype == torch.float8_e4m3fn and ra in ((1,), (3,)):
        kind = 1 if ra == (1,) else 2
        if kind == 2 and (len(swizzle_a) != 1 or len(swizzle_b) != 1 or swizzle_a[0] & 255 != 1 or swizzle_b[0] & 255 != 1):
            raise ValueError('MXFP8 grouped_mm requires SWIZZLE_32_4_4 for both scales')
    elif a.dtype == b.dtype == getattr(torch, 'float4_e2m1fn_x2', None) and ra in ((3,), (2, 0)):
        kind = 3 if ra == (3,) else 4
        if kind == 4:
            ga, gb = scale_a[1], scale_b[1]
    else:
        raise NotImplementedError('scaled_grouped_mm_v2 supports FP8 rowwise, MXFP8, MXFP4 and two-level NVFP4 recipes')
    sa, sb = scale_a[0], scale_b[0]
    _scales(a, b, sa, sb, mode, groups, kind)
    return _run(a, b, offs, mode, groups, shape, dtype, kind, use_fast_accum if kind == 1 else False, sa, sb, ga, gb)


def register(library):
    library.impl('_grouped_mm', grouped_mm)
    library.impl('_scaled_grouped_mm', scaled_grouped_mm)
    if hasattr(torch.ops.aten, '_scaled_grouped_mm_v2'):
        library.impl('_scaled_grouped_mm_v2', scaled_grouped_mm_v2)
