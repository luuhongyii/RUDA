"""Per-model HF inference/training fusion; CPU/CUDA and state dicts are unchanged."""
from types import FunctionType, MethodType

import torch
from torch import nn


_FAMILIES = ('Qwen2', 'Qwen3', 'Llama', 'Mistral', 'Qwen3Moe', 'Granite', 'Mixtral')
_NORMS = {family + 'RMSNorm' for family in _FAMILIES}
_MLPS = {family + 'MLP': ('gate_proj', 'up_proj', 'down_proj')
         for family in _FAMILIES if family != 'Mixtral'}
_MLPS['MixtralBlockSparseTop2MLP'] = ('w1', 'w3', 'w2')
_ATTENTIONS = {family + suffix for family in _FAMILIES
               for suffix in ('Attention', 'SdpaAttention', 'FlashAttention2')}
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


def _plain_silu(activation):
    cls = type(activation)
    supported = (cls is nn.SiLU and not activation.inplace
                 or cls.__name__ == 'SiLUActivation' and cls.__module__ == 'transformers.activations')
    if not supported or getattr(activation.forward, '__func__', None) is not cls.forward:
        return False
    from torch.nn.modules import module
    return not any((activation._forward_hooks, activation._forward_pre_hooks,
                    activation._backward_hooks, activation._backward_pre_hooks,
                    module._global_forward_hooks, module._global_forward_pre_hooks,
                    module._global_backward_hooks, module._global_backward_pre_hooks))


def _activation_mul(activation, gate, up):
    if (gate.device.type == 'ruda' and gate.dtype in (torch.float32, torch.float16, torch.bfloat16)
            and gate.device == up.device and gate.dtype == up.dtype and gate.shape == up.shape
            and _plain_silu(activation)):
        from .training import silu_mul
        return silu_mul(gate.contiguous(), up.contiguous())
    return activation(gate) * up


def _mlp_forward(module, hidden_states):
    if hidden_states.device.type != 'ruda' or not _plain_silu(module.act_fn):
        return module._ruda_fused_original_forward(hidden_states)
    gate_name, up_name, down_name = _MLPS[type(module).__name__]
    gate = getattr(module, gate_name)(hidden_states)
    up = getattr(module, up_name)(hidden_states)
    return getattr(module, down_name)(_activation_mul(module.act_fn, gate, up))


def _with_rotary(original):
    positional = original.__globals__.get('apply_rotary_pos_emb')
    if not callable(positional):
        return None

    def rotary(q, k, cos, sin, *args, **kwargs):
        dimension = args[0] if args else kwargs.get('unsqueeze_dim', 1)
        supported_call = (len(args) <= 1 and not (args and 'unsqueeze_dim' in kwargs)
                          and not set(kwargs) - {'unsqueeze_dim'}
                          and type(dimension) is int and dimension == 1)
        if (supported_call and q.device.type == 'ruda' and q.ndim == 4 and k.ndim == 4
                and q.dtype in (torch.float32, torch.float16, torch.bfloat16)
                and q.shape[3] > 0 and q.shape[3] % 2 == 0
                and (q.shape[0], q.shape[2], q.shape[3]) == (k.shape[0], k.shape[2], k.shape[3])
                and all(value.device == q.device and value.dtype == q.dtype
                        and not value.is_neg() and not value.is_conj() for value in (q, k, cos, sin))
                and all(not value.requires_grad and value.ndim == 3
                        and value.shape[0] in (1, q.shape[0]) and value.shape[1] in (1, q.shape[2])
                        and value.shape[2] == q.shape[3] for value in (cos, sin))):
            from . import _training_fusion_available
            if _training_fusion_available:
                from .rotary import apply_rotary_pos_emb
                return apply_rotary_pos_emb(q, k, cos, sin)
        return positional(q, k, cos, sin, *args, **kwargs)

    # Copy this method's global namespace, not the HF module's namespace.
    # Other models, including the CUDA reference, keep their original function.
    namespace = dict(original.__globals__)
    namespace['apply_rotary_pos_emb'] = rotary
    function = FunctionType(original.__code__, namespace, original.__name__,
                            original.__defaults__, original.__closure__)
    function.__kwdefaults__ = original.__kwdefaults__
    function.__annotations__ = dict(original.__annotations__)
    function.__dict__.update(original.__dict__)
    function.__qualname__ = original.__qualname__
    return function


def enable_transformers_fused(model, *, attention=None):
    """Connect existing RUDA RMSNorm/SwiGLU forward AND backward implementations.

    Exact HF dense/expert modules only; custom forwards/activations, tensor-parallel
    variants and unknown architectures are left intact. Parameter identities,
    LoRA projections, hooks, checkpoint paths and non-RUDA execution survive.
    """
    if not isinstance(model, nn.Module):
        raise TypeError('model must be an existing torch module')
    if attention not in (None, 'native'):
        raise ValueError('attention must be None or native')
    if attention is not None:
        config = getattr(model, 'config', None)
        if getattr(config, 'model_type', None) not in ('qwen2', 'qwen3', 'llama', 'mistral',
                                                      'qwen3_moe', 'granite', 'mixtral'):
            raise ValueError('native dense attention requires a supported HF dense architecture')
        setter = getattr(model, 'set_attn_implementation', None)
        if not callable(setter):
            raise ValueError('this Transformers version lacks per-model attention selection')
        global _attention_name
        if _attention_name is None:
            from .transformers_attention import register_transformers_attention
            _attention_name = register_transformers_attention('ruda_fused_training', implementation='native')
        setter(_attention_name)
    attention_functions = {}
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
            if (_plain_silu(activation) and getattr(config, 'hidden_act', 'silu') == 'silu'
                    and getattr(config, 'pretraining_tp', 1) == 1
                    and all(isinstance(getattr(module, name, None), nn.Module)
                            for name in _MLPS[cls.__name__])):
                replacement = _mlp_forward
        elif cls.__name__ in _ATTENTIONS:
            original = module.forward.__func__
            if original not in attention_functions:
                attention_functions[original] = _with_rotary(original)
            replacement = attention_functions[original]
        if replacement is not None:
            module._ruda_fused_original_forward = module.forward
            module.forward = MethodType(replacement, module)
    return model
