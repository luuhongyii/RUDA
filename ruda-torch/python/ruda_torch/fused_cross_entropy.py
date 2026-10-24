"""FP32 full-vocabulary loss without a materialized log-probability tensor."""
import torch


@torch.library.custom_op('ruda::cross_entropy_forward', mutates_args=())
def _forward(logits: torch.Tensor, targets: torch.Tensor, ignore: int) -> tuple[torch.Tensor, torch.Tensor]:
    from . import _C
    losses = logits.new_empty((logits.shape[0],), dtype=torch.float32)
    stats = logits.new_empty((logits.shape[0], 2), dtype=torch.float32)
    _C.scientific_call(650, [logits, targets, losses, stats], [ignore], [])
    return losses, stats


@_forward.register_fake
def _forward_fake(logits, targets, ignore):
    return (logits.new_empty((logits.shape[0],), dtype=torch.float32),
            logits.new_empty((logits.shape[0], 2), dtype=torch.float32))


@torch.library.custom_op('ruda::cross_entropy_backward', mutates_args=())
def _backward(logits: torch.Tensor, targets: torch.Tensor, stats: torch.Tensor,
              grad: torch.Tensor, ignore: int) -> torch.Tensor:
    from . import _C
    result = torch.empty_like(logits)
    _C.scientific_call(651, [logits, targets, stats, grad, result], [ignore], [])
    return result


@_backward.register_fake
def _backward_fake(logits, targets, stats, grad, ignore):
    return torch.empty_like(logits)


def _setup(ctx, inputs, output):
    logits, targets, ctx.ignore = inputs
    ctx.save_for_backward(logits, targets, output[1])
    ctx.mark_non_differentiable(output[1])


def _gradient(ctx, grad, ignored_stats):
    if grad is None:
        return None, None, None
    logits, targets, stats = ctx.saved_tensors
    return _backward(logits, targets, stats, grad.contiguous(), ctx.ignore), None, None


_forward.register_autograd(_gradient, setup_context=_setup)


def cross_entropy(logits, targets, *, ignore_index=-100, reduction='mean'):
    """First-order index-target CE; mean of all-ignored rows is a differentiable zero.

    FP32 max/log-sum statistics are stored separately to avoid catastrophic
    cancellation for large common logits. Bounds errors remain explicit.
    """
    from . import _training_fusion_available
    if not _training_fusion_available:
        raise RuntimeError('fused cross entropy requires rebuilding both native libraries')
    if (logits.device.type != 'ruda' or logits.ndim != 2 or logits.shape[1] < 1
            or logits.dtype not in (torch.float32, torch.float16, torch.bfloat16)
            or targets.device != logits.device or targets.shape != logits.shape[:1]
            or targets.dtype not in (torch.int32, torch.int64)):
        raise ValueError('expected RUDA floating [tokens,vocabulary] logits and integer [tokens] targets')
    if type(ignore_index) is not int or not -(1 << 63) <= ignore_index < (1 << 63):
        raise ValueError('ignore_index must fit int64')
    if reduction not in ('none', 'sum', 'mean'):
        raise ValueError('invalid loss reduction')
    losses, _ = _forward(logits.resolve_neg().contiguous(),
                         targets.resolve_neg().to(torch.int64).contiguous(), ignore_index)
    if reduction == 'none':
        return losses
    total = losses.sum()
    if reduction == 'sum':
        return total
    return total / (targets != ignore_index).sum().clamp_min(1).to(torch.float32)
