"""Native RUDA GEMM, fused half GEMM, expert GEMM and triangular solves.

Complex GEMM uses split-component GPU arithmetic internally and standard
interleaved complex PyTorch storage externally. These are operation APIs, not
a cuBLAS ABI clone. Gradients are first order; no CPU numerical fallback exists.
"""
from __future__ import annotations
import torch
from . import _C

__all__ = ["gemm", "fused_gemm", "grouped_gemm", "triangular_solve"]

def _tensor(value, name):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if value.device.type != "ruda" or value.device.index not in (None, 0):
        raise ValueError(f"{name} requires native ruda:0 storage")
    if value.is_conj():
        value = value.resolve_conj()
    if value.is_neg():
        value = value.resolve_neg()
    return value

def _scalar(value):
    if isinstance(value, torch.Tensor):
        raise TypeError("BLAS alpha/beta must be host scalar parameters, not tensors")
    return complex(value)

def _first_order():
    if torch.is_grad_enabled():
        raise RuntimeError("RUDA BLAS supports first-order differentiation only")

def _raw_gemm(a, b, c=None, *, alpha=1, beta=0, ta=False, tb=False, ca=False, cb=False, math=0, strategy=0):
    alpha, beta = _scalar(alpha), _scalar(beta)
    return _C.blas_gemm(a, b, c, alpha.real, alpha.imag, beta.real, beta.imag,
                        bool(ta), bool(tb), bool(ca), bool(cb), math, strategy)

def _reduce(tensor, shape, conjugate=False):
    return _C.blas_reduce_to(tensor, list(shape), bool(conjugate))

def _scale(tensor, value):
    value = _scalar(value)
    return _C.blas_scale(tensor, value.real, value.imag)

class _Gemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, c, alpha, beta, ta, tb, ca, cb, math, strategy):
        ctx.save_for_backward(a, b)
        ctx.options = (alpha, beta, ta, tb, ca, cb, math, strategy)
        ctx.c_shape = None if c is None else tuple(c.shape)
        ctx.set_materialize_grads(False)
        return _raw_gemm(a, b, c, alpha=alpha, beta=beta, ta=ta, tb=tb, ca=ca, cb=cb, math=math, strategy=strategy)

    @staticmethod
    def backward(ctx, grad):
        _first_order()
        if grad is None:
            return (None,) * 11
        a, b = ctx.saved_tensors
        alpha, beta, ta, tb, ca, cb, math, strategy = ctx.options
        da = db = dc = None
        if ctx.needs_input_grad[0]:
            da = _raw_gemm(grad, b, alpha=alpha.conjugate(), tb=not tb, cb=not cb, math=math, strategy=strategy)
            if ta:
                da = da.transpose(-2, -1)
            da = _reduce(da, a.shape, ca)
        if ctx.needs_input_grad[1]:
            db = _raw_gemm(a, grad, alpha=alpha.conjugate(), ta=not ta, ca=not ca, math=math, strategy=strategy)
            if tb:
                db = db.transpose(-2, -1)
            db = _reduce(db, b.shape, cb)
        if ctx.c_shape is not None and ctx.needs_input_grad[2]:
            dc = _reduce(_scale(grad, beta.conjugate()), ctx.c_shape)
        return da, db, dc, None, None, None, None, None, None, None, None

def gemm(a, b, c=None, *, alpha=1, beta=0, transpose_a=False, transpose_b=False,
         conjugate_a=False, conjugate_b=False, f32_math="strict", strategy="auto"):
    """alpha*op(a)@op(b)+beta*c, batch-broadcast F32/F64/C64/C128 matrices.

    Transpose and conjugate are independent flags. C has exact result shape;
    beta != 0 requires C. Scalar parameters follow operand real precision.
    Complex arithmetic retains component precision and currently uses the
    native scalar GPU GEMM. 'auto' and 'scalar' never invoke a host GEMM.
    """
    a, b = _tensor(a, "a"), _tensor(b, "b")
    if c is not None:
        c = _tensor(c, "c")
    if a.dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
        raise ValueError("gemm requires FP32/FP64/complex64/complex128")
    if strategy not in ("auto", "scalar") or f32_math not in ("strict", "tf32"):
        raise ValueError("strategy must be auto/scalar and f32_math strict/tf32")
    return _Gemm.apply(a, b, c, _scalar(alpha), _scalar(beta), bool(transpose_a), bool(transpose_b),
                       bool(conjugate_a), bool(conjugate_b), int(f32_math == "tf32"), int(strategy == "scalar"))

class _Fused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, bias, activation, strategy, out_dtype):
        ctx.save_for_backward(a, b, bias)
        ctx.options = activation, strategy
        ctx.set_materialize_grads(False)
        return _C.blas_fused(a, b, bias, activation, strategy, int(out_dtype == torch.float32))

    @staticmethod
    def backward(ctx, grad):
        _first_order()
        if grad is None:
            return (None,) * 6
        a, b, bias = ctx.saved_tensors
        activation, strategy = ctx.options
        # Recompute FP32 pre-activation with the same GEMM path, without storing
        # an extra pre-activation matrix in the fused forward.
        z = _C.blas_fused(a, b, bias, 0, strategy, 1)
        dz = grad.to(torch.float32)
        if activation == 1:
            dz = dz * (z > 0).to(torch.float32)
        elif activation == 2:
            t = (0.7978845608028654 * (z + 0.044715 * z * z * z)).tanh()
            dz = dz * (0.5 * (1 + t) + 0.5 * z * (1 - t*t) * 0.7978845608028654 * (1 + 0.134145*z*z))
        da = db = dbias = None
        if ctx.needs_input_grad[0]:
            da = _raw_gemm(dz, b.to(torch.float32), tb=True).to(a.dtype)
        if ctx.needs_input_grad[1]:
            db = _raw_gemm(a.to(torch.float32), dz, ta=True).to(b.dtype)
        if ctx.needs_input_grad[2]:
            dbias = _reduce(dz, bias.shape).to(bias.dtype)
        return da, db, dbias, None, None, None

def fused_gemm(a, b, bias, *, activation="none", strategy="auto", out_dtype=None):
    """FP16/BF16 [M,K]@[K,N]+bias[N], fused before one output rounding.

    GELU is the cuBLASLt tanh formula. Bias/output may use FP32 or input storage;
    explicit tensor_core errors on unsupported hardware rather than falling
    back. Auto may use a supported GPU scalar kernel. No CPU fallback.
    """
    a, b, bias = _tensor(a, "a"), _tensor(b, "b"), _tensor(bias, "bias")
    acts = {"none": 0, "relu": 1, "gelu": 2}
    strategies = {"auto": 0, "scalar": 1, "tensor_core": 2}
    if activation not in acts or strategy not in strategies:
        raise ValueError("invalid fused GEMM activation or strategy")
    if out_dtype is not None and out_dtype not in (a.dtype, torch.float32):
        raise ValueError("fused GEMM output must use input storage or FP32")
    return _Fused.apply(a, b, bias, acts[activation], strategies[strategy], a.dtype if out_dtype is None else out_dtype)

class _Grouped(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, ids, strategy):
        ctx.save_for_backward(x, weights, ids)
        ctx.set_materialize_grads(False)
        return _C.blas_grouped(x, weights, ids, strategy)

    @staticmethod
    def backward(ctx, grad):
        _first_order()
        if grad is None:
            return None, None, None, None
        x, weights, ids = ctx.saved_tensors
        dx, dw = _C.blas_grouped_backward(x, weights, ids, grad)
        return dx if ctx.needs_input_grad[0] else None, dw if ctx.needs_input_grad[1] else None, None, None

def grouped_gemm(x, weights, row_experts, *, strategy="auto"):
    """[M,K] times selected [E,N,K] experts, arbitrary row order, int32/int64 IDs.

    Negative/out-of-range IDs are padding: zero outputs and zero gradient
    contribution. Forward auto/tensor_core uses device regrouping plus CMMA;
    explicit unsupported TensorCore is an error. First-order gradients are
    deterministic GPU FP32 reductions rounded once to parameter storage.
    """
    strategies = {"auto": 0, "scalar": 1, "tensor_core": 2}
    if strategy not in strategies:
        raise ValueError("invalid grouped GEMM strategy")
    return _Grouped.apply(_tensor(x, "x"), _tensor(weights, "weights"),
                           _tensor(row_experts, "row_experts"), strategies[strategy])

class _Triangular(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, left, upper, transpose, unit, alpha):
        y = _C.blas_triangular(a, b, left, upper, transpose, unit, alpha)
        ctx.save_for_backward(a, y)
        ctx.options = left, upper, transpose, unit, alpha
        ctx.b_shape = tuple(b.shape)
        ctx.set_materialize_grads(False)
        return y

    @staticmethod
    def backward(ctx, grad):
        _first_order()
        if grad is None:
            return (None,) * 7
        a, y = ctx.saved_tensors
        left, upper, transpose, unit, alpha = ctx.options
        if alpha == 0:
            return (torch.zeros_like(a) if ctx.needs_input_grad[0] else None,
                    torch.zeros(ctx.b_shape, dtype=a.dtype, device=a.device) if ctx.needs_input_grad[1] else None,
                    None, None, None, None, None)
        h = _C.blas_triangular(a, grad, left, upper, not transpose, unit, 1.0)
        da = db = None
        if ctx.needs_input_grad[0]:
            da = _raw_gemm(h, y, alpha=-1, tb=True) if left else _raw_gemm(y, h, alpha=-1, ta=True)
            if transpose:
                da = da.transpose(-2, -1)
            da = torch.triu(da, diagonal=int(unit)) if upper else torch.tril(da, diagonal=-int(unit))
            da = _reduce(da, a.shape)
        if ctx.needs_input_grad[1]:
            db = _reduce(_scale(h, alpha), ctx.b_shape)
        return da, db, None, None, None, None, None

def triangular_solve(a, b, *, left=True, upper=True, transpose=False, unit_diagonal=False, alpha=1.0):
    """F32/F64 op(A) X = alpha B or X op(A) = alpha B, broadcast batches.

    Opposite-triangle and unit-diagonal values are not read. A zero nonunit
    pivot follows IEEE division semantics. Gradients ignore the unused triangle
    and, for unit_diagonal, the diagonal. This API returns X, not a legacy tuple.
    """
    if isinstance(alpha, torch.Tensor) or isinstance(alpha, complex):
        raise TypeError("triangular solve alpha must be a real host scalar")
    return _Triangular.apply(_tensor(a, "a"), _tensor(b, "b"), bool(left), bool(upper),
                              bool(transpose), bool(unit_diagonal), float(alpha))

def aten_matmul(a, b):
    """ATen integration adapter for F32/F64/complex vector or matrix matmul."""
    if a.ndim == 0 or b.ndim == 0:
        raise ValueError("matmul operands must be at least one-dimensional")
    av, bv = a.ndim == 1, b.ndim == 1
    result = gemm(a.unsqueeze(-2) if av else a, b.unsqueeze(-1) if bv else b)
    if av:
        result = result.squeeze(-2)
    if bv:
        result = result.squeeze(-1)
    return result

def aten_solve_triangular(a, b, *, upper, left=True, unitriangular=False):
    return triangular_solve(a, b, upper=upper, left=left, unit_diagonal=unitriangular)

class _Addmm(_Gemm):
    @staticmethod
    def forward(ctx, a, b, c, alpha, beta, ta, tb, ca, cb, math, strategy):
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("addmm matrix operands must be two-dimensional")
        shape = (a.shape[0], b.shape[1])
        output = _Gemm.forward(ctx, a, b, c.expand(shape), alpha, beta, ta, tb, ca, cb, math, strategy)
        ctx.c_shape = tuple(c.shape)
        return output

def aten_addmm(c, a, b, *, beta=1, alpha=1):
    """ATen addmm adapter with broadcast C and native broadcast VJP reduction."""
    return _Addmm.apply(_tensor(a, "a"), _tensor(b, "b"), _tensor(c, "input"),
                         _scalar(alpha), _scalar(beta), False, False, False, False, 0, 0)
