"""Instance-local Qwen2/Qwen2.5 inference fusions, preserving BF16 rounding."""
from __future__ import annotations

import ctypes
import inspect
from types import MethodType

import torch
from torch.nn import functional as F
from threading import RLock

_native_ready = False
_native_lock = RLock()


def _linear(module, bias):
    return (type(module) is torch.nn.Linear and getattr(module.forward,'__func__',None) is torch.nn.Linear.forward
        and not module._forward_hooks and not module._forward_pre_hooks
        and (module.bias is not None) == bias)


def _standard_layers(model):
    if not _linear(model.lm_head,False) or type(model.model.embed_tokens) is not torch.nn.Embedding:
        return False
    return all(all(_linear(p,True) for p in (layer.self_attn.q_proj,layer.self_attn.k_proj,layer.self_attn.v_proj))
        and all(_linear(p,False) for p in (layer.self_attn.o_proj,layer.mlp.gate_proj,layer.mlp.up_proj,layer.mlp.down_proj))
        for layer in model.model.layers)


def _initialize_native():
    global _native_ready
    from . import _C, _native
    with _native_lock:
        if not _native_ready:
            version = getattr(_native, 'ruda_torch_qwen2_inference_api_version', None)
            if version is None or not hasattr(_C, 'initialize_qwen_graph'):
                raise RuntimeError('Qwen2 inference requires rebuilding both Rust and C++ libraries')
            version.restype = ctypes.c_uint32
            if version() != 3 or _C.qwen_graph_api_version != 1:
                raise RuntimeError('Qwen2 inference API mismatch')
            _C.initialize_qwen_graph(ctypes.cast(_native.ruda_torch_qwen2_graph,ctypes.c_void_p).value,
                                     ctypes.cast(_native.ruda_torch_qwen2_flash,ctypes.c_void_p).value)
            _native_ready = True
    return _C


def _packed_attention(module):
    projections = (module.q_proj, module.k_proj, module.v_proj)
    key = tuple((p.weight.data_ptr(),p.weight._version,p.weight.dtype,p.weight.device,
                 p.bias.data_ptr(),p.bias._version) for p in projections)
    if getattr(module, '_ruda_projection_key', None) != key:
        with torch.inference_mode(False), torch.no_grad():
            module._ruda_qkv_weight = torch.cat([p.weight.detach() for p in projections])
            module._ruda_qkv_bias = torch.cat([p.bias.detach() for p in projections])
        module._ruda_projection_key = key
    return module._ruda_qkv_weight, module._ruda_qkv_bias


def _packed_mlp(module):
    projections = (module.gate_proj,module.up_proj)
    key = tuple((p.weight.data_ptr(),p.weight._version,p.weight.dtype,p.weight.device) for p in projections)
    if getattr(module, '_ruda_projection_key', None) != key:
        with torch.inference_mode(False), torch.no_grad():
            module._ruda_gate_up_weight = torch.cat([p.weight.detach() for p in projections])
        module._ruda_projection_key = key
    return module._ruda_gate_up_weight


def _frequency(module):
    source = module.inv_freq
    key = (source.data_ptr(),source._version,source.dtype,source.device)
    if getattr(module,'_ruda_frequency_key',None) != key:
        with torch.inference_mode(False), torch.no_grad():
            module._ruda_frequency = source.detach().float()
        module._ruda_frequency_key = key
    return module._ruda_frequency


def _eligible(module, x):
    return (x.device.type == 'ruda' and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and x.ndim == 3 and x.shape[0] == 1
            and not module.training and not torch.is_grad_enabled())


def _rms_forward(self, hidden_states):
    if (not _eligible(self, hidden_states) or not hidden_states.is_contiguous()
            or self.weight.dtype != hidden_states.dtype or not self.weight.is_contiguous()):
        return self._ruda_qwen2_original_forward(hidden_states)
    from . import _C
    result = torch.empty_like(hidden_states)
    _C.execute(109, hidden_states, self.weight, result, self.variance_epsilon)
    return result


def _mlp_forward(self, x):
    if not _eligible(self, x) or not all(_linear(p,False) for p in (self.gate_proj,self.up_proj,self.down_proj)):
        return self._ruda_qwen2_original_forward(x)
    from . import _C
    packed = F.linear(x, _packed_mlp(self))
    intermediate = x.new_empty((*x.shape[:-1], self.intermediate_size))
    _C.execute(111, packed, packed, intermediate, 0.)
    return self.down_proj(intermediate)


class _RotaryPair(tuple):
    def __new__(cls, cos, sin, packed):
        pair = super().__new__(cls, (cos, sin))
        pair.packed = packed
        return pair


def _rotary_forward(self, x, position_ids):
    cos, sin = self._ruda_qwen2_original_forward(x, position_ids)
    if not _eligible(self, x):
        return cos, sin
    return _RotaryPair(cos, sin, torch.stack((cos, sin)))


def _attention_forward(self, hidden_states, position_embeddings, attention_mask,
                       past_key_values=None, **kwargs):
    if not _eligible(self, hidden_states):
        return self._ruda_qwen2_original_forward(hidden_states, position_embeddings, attention_mask,
                                                 past_key_values=past_key_values, **kwargs)
    from . import _C
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    if all(_linear(p,True) for p in (self.q_proj,self.k_proj,self.v_proj)):
        weight, bias = _packed_attention(self)
        qkv = F.linear(hidden_states, weight, bias)
        sizes = (self.q_proj.out_features,self.k_proj.out_features,self.v_proj.out_features)
        query,key,value = (part.reshape(hidden_shape).transpose(1,2) for part in qkv.split(sizes,dim=-1))
    else:
        query,key,value = (p(hidden_states).view(hidden_shape).transpose(1,2) for p in (self.q_proj,self.k_proj,self.v_proj))
    if isinstance(position_embeddings, _RotaryPair):
        trig = position_embeddings.packed
    else:
        trig = torch.stack(position_embeddings)
    rotated_query = torch.empty(query.shape, device=query.device, dtype=query.dtype)
    rotated_key = torch.empty(key.shape, device=key.device, dtype=key.dtype)
    _C.execute(110, query, trig, rotated_query, 0.)
    _C.execute(110, key, trig, rotated_key, 0.)
    if past_key_values is not None:
        rotated_key, value = past_key_values.update(rotated_key, value, self.layer_idx)
    if (self.config._attn_implementation.startswith('ruda_') and attention_mask is None
            and self.sliding_window is None and query.dtype in (torch.float16,torch.bfloat16)
            and query.shape[-2] >= 16 and query.shape[-2] == rotated_key.shape[-2]
            and self.scaling == self.head_dim**-.5 and not kwargs.get('output_attentions',False)
            and kwargs.get('head_mask') is None and kwargs.get('softcap') is None):
        output = torch.empty_like(rotated_query)
        _initialize_native().qwen_flash([rotated_query.contiguous(),rotated_key.contiguous(),value.contiguous(),output],
                                       bool(self.is_causal),self.scaling)
        return self.o_proj(output.transpose(1,2).reshape(*input_shape,-1).contiguous()), None
    attention = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation,
                                                     eager_attention_forward)
    output, weights = attention(self, rotated_query, rotated_key, value, attention_mask,
        dropout=0., scaling=self.scaling, sliding_window=self.sliding_window, **kwargs)
    output = output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(output), weights


def _model_forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                   inputs_embeds=None, labels=None, use_cache=None, logits_to_keep=0, **kwargs):
    from ._qwen_cache import Qwen2InferenceCache
    candidate = input_ids if input_ids is not None else inputs_embeds
    enabled = ((input_ids is None) != (inputs_embeds is None)
        and candidate.device.type == 'ruda' and candidate.ndim == (2 if input_ids is not None else 3) and candidate.shape[0] == 1
        and not self.training and not torch.is_grad_enabled() and labels is None
        and self.model.embed_tokens.weight.dtype in (torch.float16,torch.bfloat16)
        and self.config._attn_implementation.startswith('ruda_') and self.config.hidden_act == 'silu'
        and getattr(self.model.rotary_emb,'rope_type',None) == 'default'
        and not getattr(self.model,'has_sliding_layers',False) and _standard_layers(self))
    if enabled and past_key_values is None and (use_cache if use_cache is not None else self.config.use_cache):
        past_key_values = Qwen2InferenceCache(len(self.model.layers),self._ruda_cache_capacity)
    options_ok = (not kwargs.get('output_hidden_states',getattr(self.config,'output_hidden_states',False))
        and not kwargs.get('output_attentions',getattr(self.config,'output_attentions',False))
        and kwargs.get('return_dict',True) is not False
        and set(kwargs) <= {'output_hidden_states','output_attentions','return_dict'}
        and kwargs.get('head_mask') is None and kwargs.get('softcap') is None
        and not any(m._forward_hooks or m._forward_pre_hooks for m in self.modules()))
    if (enabled and options_ok and candidate.shape[1] == 1 and isinstance(past_key_values,Qwen2InferenceCache)
            and attention_mask is None and position_ids is None and use_cache is not False
            and isinstance(logits_to_keep,int) and logits_to_keep >= 0):
        length = past_key_values.prepare_decode()
        if length is not None and length > 0:
            from transformers.modeling_outputs import CausalLMOutputWithPast
            tensors = [self.model.norm.weight,self.lm_head.weight,_frequency(self.model.rotary_emb)]
            key = []
            for layer,cache in zip(self.model.layers,past_key_values.layers):
                qkv,bias = _packed_attention(layer.self_attn)
                gate_up = _packed_mlp(layer.mlp)
                tensors.extend([layer.input_layernorm.weight,layer.post_attention_layernorm.weight,
                    qkv,bias,layer.self_attn.o_proj.weight,gate_up,layer.mlp.down_proj.weight,
                    cache.key_storage,cache.value_storage])
            key = (_initialize_native().stream_command(0),tuple(
                (t.data_ptr(),tuple(t.shape),t._version if i<3 or (i-3)%9<7 else None) for i,t in enumerate(tensors)))
            if (past_key_values._ruda_graph_key != key or past_key_values._ruda_graph_length != length):
                past_key_values.close_graph()
                logits = self.lm_head.weight.new_empty((1,1,self.lm_head.out_features))
                bindings = [t.detach() for t in tensors[:3]]+[logits]+[t.detach() for t in tensors[3:]]
                config = self.config
                dimension = self.model.layers[0].self_attn.head_dim
                dims = [len(self.model.layers),config.hidden_size,config.intermediate_size,
                    config.num_attention_heads,config.num_key_value_heads,dimension,past_key_values.layers[0].capacity]
                past_key_values._ruda_graph = _initialize_native().NativeQwenGraph(bindings,dims,
                    [config.rms_norm_eps,float(self.model.rotary_emb.attention_scaling)],length)
                past_key_values._ruda_graph_key,past_key_values._ruda_graph_length = key,length
                past_key_values._ruda_graph_logits = logits
            hidden = self.model.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
            past_key_values._ruda_graph.run(hidden.contiguous(),length)
            past_key_values.publish_decode(length+1)
            return CausalLMOutputWithPast(logits=past_key_values._ruda_graph_logits.clone(),past_key_values=past_key_values)
    return self._ruda_qwen2_model_original_forward(input_ids=input_ids,attention_mask=attention_mask,
        position_ids=position_ids,past_key_values=past_key_values,inputs_embeds=inputs_embeds,labels=labels,
        use_cache=use_cache,logits_to_keep=logits_to_keep,**kwargs)


def enable_transformers_qwen2(model, *, cache_capacity=None):
    """Fuse single-batch HF Qwen2 inference; other batches/devices and training stay unchanged."""
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Attention, Qwen2MLP, Qwen2RMSNorm, Qwen2RotaryEmbedding)
    if getattr(getattr(model, 'config', None), 'model_type', None) != 'qwen2':
        raise ValueError('expected a Transformers Qwen2/Qwen2.5 model')
    _initialize_native()
    capacity = getattr(model.config,'max_position_embeddings',None) if cache_capacity is None else cache_capacity
    if type(capacity) is not int or capacity <= 0:
        raise ValueError('cache_capacity must be a positive integer')
    replacements = []
    for module in model.modules():
        if hasattr(module, '_ruda_qwen2_original_forward'):
            continue
        if type(module) is Qwen2RMSNorm:
            replacement = _rms_forward
        elif type(module) is Qwen2MLP and module.config.hidden_act == 'silu':
            replacement = _mlp_forward
        elif type(module) is Qwen2RotaryEmbedding:
            replacement = _rotary_forward
        elif type(module) is Qwen2Attention:
            parameters = inspect.signature(module.forward).parameters
            if not {'hidden_states', 'position_embeddings', 'attention_mask', 'past_key_values'} <= parameters.keys():
                raise ValueError('unsupported Transformers Qwen2Attention forward signature')
            replacement = _attention_forward
        else:
            continue
        replacements.append((module, replacement))
    for module, replacement in replacements:
        module._ruda_qwen2_original_forward = module.forward
        module.forward = MethodType(replacement, module)
        if (type(module) is Qwen2Attention and module.q_proj.weight.device.type == 'ruda'
                and all(_linear(p,True) for p in (module.q_proj,module.k_proj,module.v_proj))):
            _packed_attention(module)
        elif (type(module) is Qwen2MLP and module.gate_proj.weight.device.type == 'ruda' and module.config.hidden_act == 'silu'
                and all(_linear(p,False) for p in (module.gate_proj,module.up_proj,module.down_proj))):
            _packed_mlp(module)
    if hasattr(model,'lm_head') and hasattr(getattr(model,'model',None),'layers'):
        model._ruda_cache_capacity = capacity
        if not hasattr(model,'_ruda_qwen2_model_original_forward'):
            model._ruda_qwen2_model_original_forward = model.forward
            model.forward = MethodType(_model_forward,model)
    return model
