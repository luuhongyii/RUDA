"""Opaque fused primitives keep native kernels intact through AOTAutograd."""
from typing import Optional
import torch


@torch.library.custom_op('ruda::training_rms_forward', mutates_args=())
def rms_forward(x: torch.Tensor, weight: Optional[torch.Tensor], epsilon: float,
                hf_rounding: bool) -> tuple[torch.Tensor, torch.Tensor]:
    from . import _C
    function = _C.training_hf_rms_forward if hf_rounding else _C.training_rms_forward
    return function(x, weight, epsilon)


@rms_forward.register_fake
def _rms_forward_fake(x, weight, epsilon, hf_rounding):
    return torch.empty_like(x), x.new_empty((x.numel() // x.shape[-1],), dtype=torch.float32)


@torch.library.custom_op('ruda::training_rms_backward', mutates_args=(),
    schema='(Tensor x, Tensor? weight, Tensor dy, Tensor stats, bool need_x, bool need_w, bool hf_rounding) -> (Tensor?, Tensor?)')
def rms_backward(x: torch.Tensor, weight: Optional[torch.Tensor], dy: torch.Tensor,
                 stats: torch.Tensor, need_x: bool, need_w: bool, hf_rounding: bool
                 ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    from . import _C
    function = _C.training_hf_rms_backward if hf_rounding else _C.training_rms_backward
    return function(x, weight, dy, stats, need_x, need_w)


@rms_backward.register_fake
def _rms_backward_fake(x, weight, dy, stats, need_x, need_w, hf_rounding):
    return torch.empty_like(x) if need_x else None, torch.empty_like(weight) if need_w else None


@torch.library.custom_op('ruda::training_silu_forward', mutates_args=())
def silu_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    from . import _C
    return _C.training_silu_forward(gate, up)


@silu_forward.register_fake
def _silu_forward_fake(gate, up):
    return torch.empty_like(gate)


@torch.library.custom_op('ruda::training_silu_backward', mutates_args=(),
    schema='(Tensor gate, Tensor up, Tensor dy, bool need_gate, bool need_up) -> (Tensor?, Tensor?)')
def silu_backward(gate: torch.Tensor, up: torch.Tensor, dy: torch.Tensor,
                  need_gate: bool, need_up: bool) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    from . import _C
    return _C.training_silu_backward(gate, up, dy, need_gate, need_up)


@silu_backward.register_fake
def _silu_backward_fake(gate, up, dy, need_gate, need_up):
    return torch.empty_like(gate) if need_gate else None, torch.empty_like(up) if need_up else None


@torch.library.custom_op('ruda::training_layer_forward', mutates_args=())
def layer_forward(x: torch.Tensor, weight: Optional[torch.Tensor], bias: Optional[torch.Tensor],
                  epsilon: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from . import _C
    return _C.training_layer_forward(x, weight, bias, epsilon)


@layer_forward.register_fake
def _layer_forward_fake(x, weight, bias, epsilon):
    shape = (x.numel() // x.shape[-1],)
    return (torch.empty_like(x), x.new_empty(shape, dtype=torch.float32),
            x.new_empty(shape, dtype=torch.float32))


@torch.library.custom_op('ruda::training_layer_backward', mutates_args=(),
    schema='(Tensor x, Tensor? weight, Tensor? bias, Tensor dy, Tensor mean, Tensor stats, bool need_x, bool need_w, bool need_b) -> (Tensor?, Tensor?, Tensor?)')
def layer_backward(x: torch.Tensor, weight: Optional[torch.Tensor], bias: Optional[torch.Tensor],
                   dy: torch.Tensor, mean: torch.Tensor, stats: torch.Tensor,
                   need_x: bool, need_w: bool, need_b: bool
                   ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    from . import _C
    return _C.training_layer_backward(x, weight, bias, dy, mean, stats, need_x, need_w, need_b)


@layer_backward.register_fake
def _layer_backward_fake(x, weight, bias, dy, mean, stats, need_x, need_w, need_b):
    return (torch.empty_like(x) if need_x else None, torch.empty_like(weight) if need_w else None,
            torch.empty_like(bias) if need_b else None)
