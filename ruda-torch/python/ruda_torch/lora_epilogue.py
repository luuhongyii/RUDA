"""Preserve FP32 scale -> low-precision cast -> residual-add boundaries in one kernel."""
import torch


@torch.library.custom_op('ruda::lora_epilogue_forward', mutates_args=())
def _forward(base: torch.Tensor, update: torch.Tensor, scale: float) -> torch.Tensor:
    from . import _C
    if any(value.is_neg() or value.is_conj() for value in (base, update)):
        raise ValueError('expected resolved RUDA base and adapter update')
    result = base.new_empty(base.shape)
    _C.scientific_call(654, [base, update, result], [], [scale])
    return result


@_forward.register_fake
def _forward_fake(base, update, scale):
    return base.new_empty(base.shape)


@torch.library.custom_op('ruda::lora_epilogue_backward', mutates_args=())
def _backward(grad: torch.Tensor, scale: float) -> torch.Tensor:
    from . import _C
    result = grad.new_empty(grad.shape, dtype=torch.float32)
    _C.scientific_call(655, [grad, result], [], [scale])
    return result


@_backward.register_fake
def _backward_fake(grad, scale):
    return grad.new_empty(grad.shape, dtype=torch.float32)


def _setup(ctx, inputs, output):
    ctx.scale = inputs[2]
    ctx.set_materialize_grads(False)


def _gradient(ctx, grad):
    if grad is None:
        return None, None, None
    grad = grad.resolve_neg()
    update_grad = None
    if ctx.needs_input_grad[1]:
        if torch.is_grad_enabled():
            update_grad = grad.to(torch.float32) * ctx.scale
        else:
            update_grad = _backward(grad, ctx.scale)
    return (grad if ctx.needs_input_grad[0] else None,
            update_grad, None)


_forward.register_autograd(_gradient, setup_context=_setup)


def lora_epilogue(base, update, scale):
    """Same-shape FP16/BF16 base plus a scaled FP32 adapter update; no saved activations.

    FP32 bases and low-precision adapters keep their existing composition.
    This kernel deliberately does not reassociate the residual addition with
    the FP32 scale: the intervening FP16/BF16 rounding is part of the contract.
    """
    from . import _training_fusion_available
    if not _training_fusion_available:
        raise RuntimeError('LoRA epilogue fusion requires rebuilding both native libraries')
    if (base.device.type != 'ruda' or base.dtype not in (torch.float16, torch.bfloat16)
            or update.device != base.device or update.dtype != torch.float32 or update.shape != base.shape):
        raise ValueError('expected same-shape RUDA FP16/BF16 base and FP32 adapter update')
    if type(scale) not in (int, float):
        raise TypeError('LoRA scale must be a Python number')
    return _forward(base, update, float(scale))


def rank16_residual_eligible(base, rank, weight):
    from . import _lora_projection_available
    return (_lora_projection_available and base.device.type == 'ruda'
            and base.dtype in (torch.float16, torch.bfloat16) and base.ndim > 0
            and rank.dtype == weight.dtype == torch.float32
            and rank.device == weight.device == base.device
            and rank.shape == (*base.shape[:-1], 16)
            and weight.shape == (base.shape[-1], 16)
            and base.shape[-1] >= 128 and base.shape[-1] % 128 == 0
            and 0 < rank.numel() // 16 <= 16 * 65535 and rank.numel() % 256 == 0
            and all(t.is_contiguous() and t.numel() <= 2**32-1
                    and not t.is_neg() and not t.is_conj() for t in (base, rank, weight))
            and weight.storage_offset() % 4 == 0)


@torch.library.custom_op('ruda::lora_rank16_residual', mutates_args=())
def _rank16_residual(base: torch.Tensor, rank: torch.Tensor, weight: torch.Tensor,
                     scale: float) -> torch.Tensor:
    from . import _C
    if not rank16_residual_eligible(base, rank, weight):
        raise ValueError('unsupported RUDA rank-16 residual projection geometry')
    result = base.new_empty(base.shape)
    _C.scientific_call(656, [base, rank, weight, result], [], [scale])
    return result


@_rank16_residual.register_fake
def _rank16_residual_fake(base, rank, weight, scale):
    return base.new_empty(base.shape)


def _rank16_setup(ctx, inputs, output):
    _, rank, weight, ctx.scale = inputs
    ctx.save_for_backward(rank, weight)
    ctx.set_materialize_grads(False)


def _rank16_gradient(ctx, grad):
    if grad is None:
        return None, None, None, None
    rank, weight = ctx.saved_tensors
    grad = grad.resolve_neg()
    rank_grad = weight_grad = None
    if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
        scaled = grad.float() * ctx.scale if torch.is_grad_enabled() else _backward(grad, ctx.scale)
        flat = scaled.reshape(-1, weight.shape[0])
        if ctx.needs_input_grad[1]:
            rank_grad = (flat @ weight).reshape(rank.shape)
        if ctx.needs_input_grad[2]:
            weight_grad = flat.t() @ rank.reshape(-1, 16)
    return grad if ctx.needs_input_grad[0] else None, rank_grad, weight_grad, None


_rank16_residual.register_autograd(_rank16_gradient, setup_context=_rank16_setup)
