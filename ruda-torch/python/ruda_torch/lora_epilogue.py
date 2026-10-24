"""Preserve FP32 scale -> low-precision cast -> residual-add boundaries in one kernel."""
import torch


@torch.library.custom_op('ruda::lora_epilogue_forward', mutates_args=())
def _forward(base: torch.Tensor, update: torch.Tensor, scale: float) -> torch.Tensor:
    from . import _C
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
            or update.device != base.device or update.dtype != torch.float32 or update.shape != base.shape
            or any(value.is_neg() or value.is_conj() for value in (base, update))):
        raise ValueError('expected same-shape RUDA FP16/BF16 base and FP32 adapter update')
    if type(scale) not in (int, float):
        raise TypeError('LoRA scale must be a Python number')
    return _forward(base, update, float(scale))
