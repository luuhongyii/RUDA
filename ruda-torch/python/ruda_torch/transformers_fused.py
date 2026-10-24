"""Per-model HF dense-training fusion; CPU/CUDA and state-dict paths are unchanged."""
from types import MethodType

import torch
from torch import nn


_NORMS = {'Qwen2RMSNorm', 'Qwen3RMSNorm', 'LlamaRMSNorm', 'MistralRMSNorm'}
_MLPS = {'Qwen2MLP', 'Qwen3MLP', 'LlamaMLP', 'MistralMLP'}
_attention_name = None


def _rms_forward(module, hidden_states):
    if (hidden_states.device.type != 'ruda' or hidden_states.dtype != module.weight.dtype
            or hidden_states.dtype not in (torch.float32, torch.float16, torch.bfloat16)):
        return module._ruda_fused_original_forward(hidden_states)
    from . import _training_fusion_available
    from .training import rms_norm
    x = hidden_states.contiguous()
    if _training_fusion_available:
        return rms_norm(x, module.weight, eps=module.variance_epsilon, round_normalized=True)
    # An older native ABI can still fuse the FP32 normalization, retaining
    # HF's explicit activation cast and affine/gradient storage boundaries.
    return module.weight * rms_norm(x, eps=module.variance_epsilon)


def _mlp_forward(module, hidden_states):
    if hidden_states.device.type != 'ruda':
        return module._ruda_fused_original_forward(hidden_states)
    from .training import silu_mul
    gate = module.gate_proj(hidden_states)
    up = module.up_proj(hidden_states)
    return module.down_proj(silu_mul(gate.contiguous(), up.contiguous()))


def enable_transformers_fused(model, *, attention=None):
    """Connect existing RUDA RMSNorm/SwiGLU forward AND backward implementations.

    Exact HF dense modules only; custom forwards/activations, tensor-parallel
    variants and unknown architectures are left intact. Parameter identities,
    LoRA projections, hooks, checkpoint paths and non-RUDA execution survive.
    """
    if not isinstance(model, nn.Module):
        raise TypeError('model must be an existing torch module')
    if attention not in (None, 'native'):
        raise ValueError('attention must be None or native')
    if attention is not None:
        config = getattr(model, 'config', None)
        if getattr(config, 'model_type', None) not in ('qwen2', 'qwen3', 'llama', 'mistral'):
            raise ValueError('native dense attention requires a supported HF dense architecture')
        setter = getattr(model, 'set_attn_implementation', None)
        if not callable(setter):
            raise ValueError('this Transformers version lacks per-model attention selection')
        global _attention_name
        if _attention_name is None:
            from .transformers_attention import register_transformers_attention
            _attention_name = register_transformers_attention('ruda_fused_training', implementation='native')
        setter(_attention_name)
    for module in model.modules():
        cls = type(module)
        if not cls.__module__.startswith('transformers.models.'):
            continue
        if hasattr(module, '_ruda_fused_original_forward'):
            continue
        if getattr(module.forward, '__func__', None) is not cls.forward:
            continue
        replacement = None
        if cls.__name__ in _NORMS:
            weight = getattr(module, 'weight', None)
            if isinstance(weight, torch.Tensor) and weight.ndim == 1 and hasattr(module, 'variance_epsilon'):
                replacement = _rms_forward
        elif cls.__name__ in _MLPS:
            config = getattr(module, 'config', None)
            activation = getattr(module, 'act_fn', None)
            silu = (type(activation) is nn.SiLU and not activation.inplace
                    or type(activation).__name__ == 'SiLUActivation'
                    and type(activation).__module__ == 'transformers.activations')
            if (silu and getattr(config, 'hidden_act', None) == 'silu'
                    and getattr(config, 'pretraining_tp', 1) == 1
                    and all(isinstance(getattr(module, name, None), nn.Module)
                            for name in ('gate_proj', 'up_proj', 'down_proj'))):
                replacement = _mlp_forward
        if replacement is not None:
            module._ruda_fused_original_forward = module.forward
            module.forward = MethodType(replacement, module)
    return model
