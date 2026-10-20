"""Direct cuBLAS on RUDA allocations, retaining device-side layout conversion."""
import ctypes as ct
from math import prod
import torch
from ._vendor_runtime import invoke,column,empty_column,integer

_types = {torch.float32: 0, torch.float64: 1, torch.float16: 2,
          torch.bfloat16: 14, torch.complex64: 4, torch.complex128: 5}


def dot(a, b, *, conjugate=False):
    if (a.dtype not in _types or a.dtype != b.dtype or a.device != b.device
            or a.ndim != 1 or b.ndim != 1 or a.shape != b.shape):
        raise ValueError('cuBLAS dot requires matching vector operands')
    out = torch.empty((), dtype=a.dtype, device=a.device)
    if a.numel() == 0:
        return out.zero_()
    n = integer(a.numel())
    incx = integer(a.stride(0)) if n != 1 else 1
    incy = integer(b.stride(0)) if n != 1 else 1
    invoke(15, [a, b, out], [n, incx, incy, _types[a.dtype], int(conjugate and a.is_complex())])
    return out


def int_mm(a, b):
    if a.dtype != torch.int8 or b.dtype != torch.int8 or a.device != b.device or a.ndim != 2 or b.ndim != 2:
        raise ValueError('cuBLASLt integer GEMM requires matching int8 matrices')
    m, k, n = a.shape[0], a.shape[1], b.shape[1]
    if b.shape[0] != k:
        raise ValueError('cuBLASLt integer GEMM contraction dimensions differ')
    result = torch.empty((m, n), dtype=torch.int32, device=a.device)
    invoke(16, [a.contiguous(), b.contiguous(), result], [integer(m), integer(n), integer(k)])
    return result


def scaled_mm(a, b, scale_a, scale_b, bias, dtype, mode_a, mode_b, fast, *, alpha=None):
    types = {**_types, torch.float8_e4m3fn: 28, torch.float8_e5m2: 29, torch.float4_e2m1fn_x2: 33}
    m, k, n = a.shape[0], a.shape[1], b.shape[1]
    if a.dtype == torch.float4_e2m1fn_x2:
        k *= 2
    left, right = a.contiguous(), column(b)
    scale_a = column(scale_a) if mode_a in (2, 4, 5) else scale_a.contiguous()
    scale_b = column(scale_b) if mode_b in (3, 4, 5) else scale_b.contiguous()
    result = torch.empty((m, n), dtype=dtype, device=a.device)
    tensors = [left, right, scale_a, scale_b, result]
    if bias is not None:
        tensors.append(bias)
    if alpha is not None:
        tensors.extend((alpha.reshape(1).contiguous(), torch.zeros(1, dtype=torch.float32, device=a.device)))
    bias_type = types[bias.dtype] if bias is not None else 2 if dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else types[dtype]
    invoke(17, tensors, [integer(m), integer(n), integer(k), types[a.dtype], types[b.dtype], types[dtype],
                         bias_type, mode_a, mode_b, int(fast), int(bias is not None), int(alpha is not None)])
    return result


def gemm(a, b, c=None, *, alpha=1, beta=0, ta=False, tb=False, ca=False, cb=False,
         math=0, strategy=0, out_dtype=None):
    if strategy != 0:
        raise ValueError('explicit RUDA kernel strategies require math_backend("ruda"); NVIDIA selects its own algorithm')
    if a.device != b.device or a.dtype != b.dtype or a.dtype not in _types or a.ndim < 2 or b.ndim < 2:
        raise ValueError('cuBLAS GEMM requires matching RUDA matrix operands')
    if math not in (0, 1):
        raise ValueError('invalid GEMM math policy')
    alpha,beta = complex(alpha),complex(beta)
    if not a.is_complex() and (alpha.imag or beta.imag):
        raise ValueError('real cuBLAS operands require real alpha/beta')
    if a.dtype not in (torch.float64,torch.complex128):
        alpha = complex(ct.c_float(alpha.real).value,ct.c_float(alpha.imag).value)
        beta = complex(ct.c_float(beta.real).value,ct.c_float(beta.imag).value)
    if ta:
        a = a.transpose(-2, -1)
    if tb:
        b = b.transpose(-2, -1)
    if ca and a.is_complex():
        a = a.conj()
    if cb and b.is_complex():
        b = b.conj()
    m, k, n = a.shape[-2], a.shape[-1], b.shape[-1]
    if b.shape[-2] != k:
        raise ValueError('cuBLAS GEMM contraction dimensions differ')
    batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    shape = (*batch, m, n)
    out_dtype = a.dtype if out_dtype is None else out_dtype
    if out_dtype not in (a.dtype, torch.float32) or out_dtype != a.dtype and a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError('unsupported cuBLAS output precision')
    if c is not None and (c.device != a.device or c.dtype != out_dtype or tuple(c.shape) != shape):
        raise ValueError('GEMM C must match result shape, device and dtype')
    if c is None and complex(beta) != 0:
        raise ValueError('nonzero beta requires C')
    out = empty_column(shape, a, out_dtype)
    batches = prod(batch)
    if out.numel() == 0:
        return out.contiguous()
    if k == 0 or complex(alpha) == 0:
        if c is None or complex(beta) == 0:
            out.zero_()
        else:
            out.copy_(c*(beta if a.is_complex() else beta.real))
        return out.contiguous()
    a = column(a.expand(*batch, m, k))
    b = column(b.expand(*batch, k, n))
    if c is not None and complex(beta) != 0:
        out.copy_(c)
    dtype, output_type = _types[a.dtype], _types[out_dtype]
    compute = 71 if a.dtype in (torch.float64, torch.complex128) else (77 if math else 69)
    # FP16/BF16 GEMM retains FP32 accumulation without pedantic disabling TC.
    if a.dtype in (torch.float16, torch.bfloat16):
        compute = 68
    m, n, k, batches = map(integer, (m, n, k, batches))

    invoke(0,[a,b,out],[m,n,k,batches,dtype,output_type,compute],
           [alpha.real,alpha.imag,beta.real,beta.imag])
    return out.contiguous()


def triangular(a, b, left=True, upper=True, transpose=False, unit=False, alpha=1.):
    if a.dtype not in (torch.float32, torch.float64) or a.dtype != b.dtype or a.device != b.device:
        raise ValueError('cuBLAS TRSM requires matching real F32/F64 matrices')
    if a.ndim < 2 or b.ndim < 2 or a.shape[-2] != a.shape[-1] or a.shape[-1] != b.shape[-2 if left else -1]:
        raise ValueError('incompatible cuBLAS TRSM dimensions')
    batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    order, m, n = a.shape[-1], b.shape[-2], b.shape[-1]
    out = empty_column((*batch, m, n), b)
    if not out.numel():
        return out.contiguous()
    if alpha == 0:
        out.zero_()
        return out.contiguous()
    out.copy_(b.expand(*batch,m,n))
    a = column(a.expand(*batch,order,order))
    width, batches = a.element_size(), prod(batch)
    order,m,n = map(integer,(order,m,n))

    invoke(1,[a,out],[order,m,n,integer(batches),width,int(left),int(upper),int(transpose),int(unit)],[float(alpha)])
    return out.contiguous()
