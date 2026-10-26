"""Fused HF half-RoPE for Q and K, with strided inputs and first-order backward."""
from typing import Optional
import torch


@torch.library.custom_op('ruda::rotary_qk_forward', mutates_args=())
def _forward(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
              sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from . import _C
    oq, ok = q.new_empty(q.shape), k.new_empty(k.shape)
    _C.scientific_call(652, [q, k, cos, sin, oq, ok], [], [])
    return oq, ok


@_forward.register_fake
def _forward_fake(q, k, cos, sin):
    return q.new_empty(q.shape), k.new_empty(k.shape)


@torch.library.custom_op('ruda::rotary_qk_backward', mutates_args=(),
    schema='(Tensor? dq, Tensor? dk, Tensor cos, Tensor sin, SymInt[] q_shape, SymInt[] k_shape) -> (Tensor?, Tensor?)')
def _backward(dq: Optional[torch.Tensor], dk: Optional[torch.Tensor], cos: torch.Tensor,
               sin: torch.Tensor, q_shape: list[int], k_shape: list[int]
               ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    from . import _C
    iq = cos.expand(q_shape) if dq is None else dq
    ik = cos.expand(k_shape) if dk is None else dk
    oq = None if dq is None else dq.new_empty(q_shape)
    ok = None if dk is None else dk.new_empty(k_shape)
    _C.scientific_call(653, [iq, ik, cos, sin, iq if oq is None else oq, ik if ok is None else ok],
                       [int(dq is not None), int(dk is not None)], [])
    return oq, ok


@_backward.register_fake
def _backward_fake(dq, dk, cos, sin, q_shape, k_shape):
    return (None if dq is None else dq.new_empty(q_shape),
            None if dk is None else dk.new_empty(k_shape))


def _setup(ctx, inputs, output):
    q, k, cos, sin = inputs
    ctx.q_shape, ctx.k_shape = list(q.shape), list(k.shape)
    ctx.save_for_backward(cos, sin)
    ctx.set_materialize_grads(False)


def _gradient(ctx, dq, dk):
    if not ctx.needs_input_grad[0]:
        dq = None
    if not ctx.needs_input_grad[1]:
        dk = None
    if dq is None and dk is None:
        return None, None, None, None
    cos, sin = ctx.saved_tensors
    oq, ok = _backward(dq, dk, cos, sin, ctx.q_shape, ctx.k_shape)
    return oq, ok, None, None


_forward.register_autograd(_gradient, setup_context=_setup)


class _Rotary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        output = _forward(q, k, cos, sin)
        _setup(ctx, (q, k, cos, sin), output)
        return output

    @staticmethod
    def backward(ctx, dq, dk):
        return _gradient(ctx, dq, dk)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """HF half-rotation, not interleaved RoPE; cosine/sine are fixed coefficients.

    Input [B,H,T,D] may be transposed. Q/K share B,T,D but can have different
    head counts. Output storage is contiguous. Multiply/cast/add boundaries
    match the unfused low-precision expression, including its backward.
    """
    from . import _training_fusion_available
    if not _training_fusion_available:
        raise RuntimeError('fused RoPE requires rebuilding both native libraries')
    if (type(unsqueeze_dim) is not int or unsqueeze_dim != 1 or q.ndim != 4 or k.ndim != 4
            or q.device.type != 'ruda' or q.dtype not in (torch.float32, torch.float16, torch.bfloat16)
            or q.shape[3] == 0 or q.shape[3] % 2
            or (q.shape[0], q.shape[2], q.shape[3]) != (k.shape[0], k.shape[2], k.shape[3])
            or any(value.device != q.device or value.dtype != q.dtype or value.is_neg() or value.is_conj()
                   for value in (q, k, cos, sin))):
        raise ValueError('fused half-RoPE requires same-device/dtype Q/K [B,H,T,D] with even D')
    for coefficient in (cos, sin):
        if (coefficient.requires_grad or coefficient.ndim != 3
                or coefficient.shape[0] not in (1, q.shape[0])
                or coefficient.shape[1] not in (1, q.shape[2])
                or coefficient.shape[2] != q.shape[3]):
            raise ValueError('RoPE coefficients must be fixed [B or 1,T or 1,D] tensors')
    forward = (_Rotary.apply if not torch.compiler.is_compiling() and torch.is_grad_enabled()
               and (q.requires_grad or k.requires_grad) else _forward)
    return forward(q, k, cos.unsqueeze(1), sin.unsqueeze(1))
