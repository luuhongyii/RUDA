"""Native channel-gated delta operators and shape-only compiler contracts."""
import torch


@torch.library.custom_op('ruda::kda_forward', mutates_args=())
def kda_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                beta: torch.Tensor, decay: torch.Tensor, initial: torch.Tensor,
                scale: float, chunk: int, retain: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from . import _C
    return _C.kda_forward(q, k, v, beta, decay, initial, scale, chunk, retain)


@kda_forward.register_fake
def _forward_fake(q, k, v, beta, decay, initial, scale, chunk, retain):
    count = (q.shape[-2] + chunk - 1) // chunk if retain else 0
    return (torch.empty_like(v), torch.empty_like(initial), initial.new_empty((count, *initial.shape)))


@torch.library.custom_op('ruda::kda_backward', mutates_args=())
def kda_backward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 beta: torch.Tensor, decay: torch.Tensor, initial: torch.Tensor,
                 boundaries: torch.Tensor, dy: torch.Tensor, dh: torch.Tensor,
                 scale: float, chunk: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                  torch.Tensor, torch.Tensor, torch.Tensor]:
    from . import _C
    return _C.kda_backward(q, k, v, beta, decay, initial, boundaries, dy, dh, scale, chunk)


@kda_backward.register_fake
def _backward_fake(q, k, v, beta, decay, initial, boundaries, dy, dh, scale, chunk):
    return tuple(torch.empty_like(t) for t in (q, k, v, beta, decay, initial))
