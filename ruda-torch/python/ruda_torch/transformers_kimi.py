"""HF-compatible Kimi Linear checkpoint training and generation on RUDA.

Import this optional module explicitly, or call register_transformers_kimi().
No FLA, Triton, FlashAttention, or remote model-code import is required.
"""
from __future__ import annotations

import torch
from torch import nn
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ._kimi_config import KimiLinearConfig
from ._kimi_layers import (TokenLayout, KimiDecoderLayer, KimiRMSNorm,
                           KimiGatedRMSNorm, KimiShortConvolution, KimiDeltaAttention,
                           KimiMoEGate)
from ._architecture_ops import positions


class KimiLinearCache:
    is_compileable = False
    is_sliding = False

    def __init__(self, layers, seen_tokens=0, valid=None):
        self.layers = tuple(layers)
        self.seen_tokens = seen_tokens
        self.valid = valid

    def __len__(self):
        return len(self.layers)

    def get_seq_length(self, layer_idx=0):
        return self.seen_tokens

    def get_max_cache_shape(self):
        return None

    def get_mask_sizes(self, cache_position, layer_idx):
        return self.seen_tokens + cache_position.shape[0], 0

    def reorder_cache(self, beam_idx):
        self.layers = tuple(None if layer is None else layer.reorder(beam_idx) for layer in self.layers)
        if self.valid is not None:
            self.valid = self.valid.index_select(0, beam_idx.to(self.valid.device))

    def batch_select_indices(self, indices):
        self.reorder_cache(indices)

    def batch_repeat_interleave(self, repeats):
        def repeat(layer):
            return None if layer is None else layer._map(lambda value: value.repeat_interleave(repeats, 0))
        self.layers = tuple(repeat(layer) for layer in self.layers)
        if self.valid is not None:
            self.valid = self.valid.repeat_interleave(repeats, 0)

    def detach(self):
        return KimiLinearCache(tuple(None if layer is None else layer.detach() for layer in self.layers),
                               self.seen_tokens, None if self.valid is None else self.valid.detach())


class KimiPreTrainedModel(PreTrainedModel):
    config_class = KimiLinearConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['KimiDecoderLayer']
    _skip_keys_device_placement = 'past_key_values'
    _is_stateful = True

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        options = dict(gradient_checkpointing_kwargs or {})
        options.setdefault('use_reentrant', False)
        return super().gradient_checkpointing_enable(gradient_checkpointing_kwargs=options)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0., std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, (KimiRMSNorm, KimiGatedRMSNorm)):
            nn.init.ones_(module.weight)
        elif isinstance(module, KimiShortConvolution):
            nn.init.uniform_(module.weight, -module.window ** -.5, module.window ** -.5)
        elif isinstance(module, KimiDeltaAttention):
            with torch.no_grad():
                module.A_log.uniform_(1., 16.).log_()
        elif isinstance(module, KimiMoEGate):
            nn.init.kaiming_uniform_(module.weight, a=5 ** .5)


class KimiLinearModel(KimiPreTrainedModel):
    def __init__(self, config):
        config.validate_architecture()
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([KimiDecoderLayer(config, index) for index in range(config.num_hidden_layers)])
        self.norm = KimiRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, cache_position=None,
                use_cache=None, output_hidden_states=None, output_attentions=None,
                return_dict=None, cu_seqlens=None, max_seqlen=None,
                ruda_context_positions=None, ruda_context_documents=None, ruda_context_valid=None, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError('supply exactly one of input_ids and inputs_embeds')
        if output_attentions or (output_attentions is None and self.config.output_attentions):
            raise ValueError('RUDA fused Kimi attention does not materialize quadratic attention weights')
        if kwargs:
            raise TypeError(f'unsupported Kimi model arguments: {tuple(kwargs)}')
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        packed_flat = hidden.ndim == 2 and cu_seqlens is not None
        if packed_flat:
            hidden = hidden.unsqueeze(0)
        if hidden.ndim != 3 or hidden.shape[-1] != self.config.hidden_size or hidden.shape[0] == 0:
            raise ValueError('Kimi inputs must have [batch,tokens,hidden_size] geometry')
        batch, length = hidden.shape[:2]
        use_cache = (self.config.use_cache and not self.training and cu_seqlens is None) if use_cache is None else use_cache
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        if past_key_values is not None:
            if not isinstance(past_key_values, KimiLinearCache) or len(past_key_values) != len(self.layers):
                raise ValueError('Kimi requires its matching hybrid cache')
        seen = 0 if past_key_values is None else past_key_values.get_seq_length()
        if cache_position is not None and (cache_position.ndim != 1 or cache_position.shape[0] != length):
            raise ValueError('cache_position must describe the current token interval')
        documents = None
        context_group = getattr(self, '_ruda_context_group', None)
        if context_group is not None:
            if past_key_values is not None or use_cache or cu_seqlens is not None:
                raise ValueError('Kimi context shards use document metadata, not a replicated decode/packed cache')
            if (ruda_context_positions is None or ruda_context_documents is None
                    or ruda_context_positions.shape != (batch, length) or ruda_context_documents.shape != (batch, length)):
                raise ValueError('Kimi context shards require local global positions and document IDs')
            valid = ruda_context_valid
            if attention_mask is not None:
                mask = attention_mask.to(device=hidden.device, dtype=torch.bool)
                valid = mask if valid is None else valid & mask
            if valid is None:
                valid = torch.ones((batch, length), device=hidden.device, dtype=torch.bool)
            layout = TokenLayout(batch, length, hidden.device, valid)
            layout.context = {'group': context_group, 'key': self._ruda_context_key,
                              'positions': ruda_context_positions, 'documents': ruda_context_documents, 'valid': valid}
            attention = None
        elif any(value is not None for value in (ruda_context_positions, ruda_context_documents, ruda_context_valid)):
            raise ValueError('enable the Kimi context process group before passing context metadata')
        elif cu_seqlens is not None:
            from .varlen_attention import _boundaries
            if batch != 1 or attention_mask is not None or past_key_values is not None or use_cache:
                raise ValueError('packed Kimi documents require one flat stream, no padding mask and no cache')
            documents = _boundaries(cu_seqlens, length)
            maximum = max(b - a for a, b in zip(documents, documents[1:]))
            if max_seqlen is not None and max_seqlen != maximum:
                raise ValueError('max_seqlen differs from the actual packed documents')
            layout = TokenLayout(batch, length, hidden.device)
            query_indices = key_indices = positions(length, hidden.device)
            attention = (query_indices, key_indices, cu_seqlens, cu_seqlens)
            valid = None
        else:
            if max_seqlen is not None:
                raise ValueError('max_seqlen requires cu_seqlens')
            valid = None
            previous_valid = None if past_key_values is None else past_key_values.valid
            if attention_mask is not None:
                if attention_mask.ndim != 2 or attention_mask.shape[0] != batch:
                    raise ValueError('Kimi attention_mask must be a two-dimensional padding mask')
                valid = attention_mask.to(device=hidden.device, dtype=torch.bool)
                if valid.shape[1] == length and seen:
                    prefix = (torch.ones(batch, seen, dtype=torch.bool, device=hidden.device)
                              if previous_valid is None else previous_valid.to(hidden.device))
                    valid = torch.cat((prefix, valid), 1)
                elif valid.shape[1] != seen + length:
                    raise ValueError('Kimi mask must describe current tokens or the full cached sequence')
            elif previous_valid is not None:
                valid = torch.cat((previous_valid.to(hidden.device),
                                   torch.ones(batch, length, device=hidden.device, dtype=torch.bool)), 1)
            current_valid = None if valid is None else valid[:, seen:]
            layout = TokenLayout(batch, length, hidden.device, current_valid)
            keys = TokenLayout(batch, seen + length, hidden.device, valid)
            query_indices, cu_q = layout.indices(hidden.device)
            key_indices, cu_k = keys.indices(hidden.device)
            attention = query_indices, key_indices, cu_q, cu_k
        history = () if output_hidden_states else None
        updated = []
        sequence_group = getattr(self, '_ruda_sequence_group', None)
        if sequence_group is not None:
            from .parallel_training import scatter_to_tensor_parallel
            if use_cache or past_key_values is not None or packed_flat or length % sequence_group.world_size:
                raise ValueError('Kimi SP uses cacheless [batch,tokens] inputs divisible by its tensor group')
            layout.sequence_group = sequence_group
            hidden = scatter_to_tensor_parallel(hidden, sequence_group, axis=1)
        for index, layer in enumerate(self.layers):
            if output_hidden_states:
                recorded = hidden
                if sequence_group is not None:
                    from .parallel_training import gather_from_tensor_parallel
                    recorded = gather_from_tensor_parallel(recorded, sequence_group, axis=1)
                history += (recorded.squeeze(0) if packed_flat else recorded,)
            state = None if past_key_values is None else past_key_values.layers[index]
            if self.gradient_checkpointing and self.training:
                hidden, state = self._gradient_checkpointing_func(layer.__call__, hidden, layout,
                                                                 attention, state, documents)
            else:
                hidden, state = layer(hidden, layout, attention, state, documents)
            if use_cache:
                updated.append(state)
        hidden = self.norm(hidden)
        if sequence_group is not None:
            from .parallel_training import gather_from_tensor_parallel
            hidden = gather_from_tensor_parallel(hidden, sequence_group, axis=1)
        if packed_flat:
            hidden = hidden.squeeze(0)
        if output_hidden_states:
            history += (hidden,)
        cache = KimiLinearCache(updated, seen + length, valid) if use_cache else None
        result = BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=cache,
                                         hidden_states=history)
        return result if return_dict else result.to_tuple()


class KimiLinearForCausalLM(KimiPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ['lm_head.weight']

    def __init__(self, config):
        super().__init__(config)
        self.model = KimiLinearModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def get_decoder(self):
        return self.model

    def set_decoder(self, value):
        self.model = value

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None,
                cache_position=None, logits_to_keep=0, generation_mode=False,
                cu_seqlens=None, max_seqlen=None, **loss_kwargs):
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        context = {name: loss_kwargs.pop(name) for name in
                   ('ruda_context_positions', 'ruda_context_documents', 'ruda_context_valid') if name in loss_kwargs}
        result = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                            return_dict=True, cache_position=cache_position, cu_seqlens=cu_seqlens,
                            max_seqlen=max_seqlen, **context)
        hidden = result.last_hidden_state
        if generation_mode:
            logits_to_keep = 1
        if type(logits_to_keep) is not int or logits_to_keep < 0:
            raise ValueError('logits_to_keep must be a nonnegative integer')
        if labels is not None and logits_to_keep:
            raise ValueError('training labels require all token logits')
        logits = self.lm_head(hidden[..., -logits_to_keep:, :] if logits_to_keep else hidden)
        loss = None
        if labels is not None:
            if labels.shape != hidden.shape[:-1]:
                raise ValueError('Kimi labels must match the input token shape')
            if cu_seqlens is not None:
                from .varlen_attention import _boundaries
                boundaries = _boundaries(cu_seqlens, hidden.shape[-2])
                starts = torch.tensor(boundaries[:-1], device=labels.device, dtype=torch.int64)
                token = positions(labels.shape[-1], labels.device)
                starts = (token[:, None] == starts[None, :]).any(-1)
                labels = labels.masked_fill(starts, -100)
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size, **loss_kwargs)
        elif loss_kwargs:
            raise TypeError(f'loss arguments require labels: {tuple(loss_kwargs)}')
        output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=result.past_key_values,
                                        hidden_states=result.hidden_states)
        return output if return_dict else output.to_tuple()

    @classmethod
    def _supports_default_dynamic_cache(cls):
        return False

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None,
                                      inputs_embeds=None, cache_position=None, use_cache=True, **kwargs):
        seen = 0 if past_key_values is None else past_key_values.get_seq_length()
        if seen:
            if cache_position is not None:
                input_ids = input_ids[:, -cache_position.shape[0]:]
            elif input_ids.shape[1] > seen:
                input_ids = input_ids[:, seen:]
            else:
                input_ids = input_ids[:, -1:]
        data = {'input_ids': input_ids.contiguous()}
        if inputs_embeds is not None and not seen:
            data = {'inputs_embeds': inputs_embeds}
        data.update(past_key_values=past_key_values, attention_mask=attention_mask,
                    cache_position=cache_position, use_cache=use_cache, logits_to_keep=1)
        for name in ('output_hidden_states', 'output_attentions'):
            if name in kwargs:
                data[name] = kwargs[name]
        return data

    def _reorder_cache(self, past_key_values, beam_idx):
        past_key_values.reorder_cache(beam_idx)
        return past_key_values


def register_transformers_kimi(*, exist_ok=False):
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
    AutoConfig.register('kimi_linear', KimiLinearConfig, exist_ok=exist_ok)
    AutoModel.register(KimiLinearConfig, KimiLinearModel, exist_ok=exist_ok)
    AutoModelForCausalLM.register(KimiLinearConfig, KimiLinearForCausalLM, exist_ok=exist_ok)
    return KimiLinearForCausalLM


def tensor_parallelize_kimi(model, group, *, vocab_parallel=True):
    """Shard FFN intermediates and mixer weights; retain replicated hybrid state."""
    from .parallel_training import tensor_parallelize
    if not isinstance(model, (KimiLinearModel, KimiLinearForCausalLM)):
        raise TypeError('expected a RUDA Kimi Linear model')
    prefix = 'model.' if isinstance(model, KimiLinearForCausalLM) else ''
    backbone = model.model if prefix else model
    plan = {}
    if vocab_parallel:
        plan[prefix + 'embed_tokens'] = 'embedding'
        if prefix:
            plan['lm_head'] = 'column'
    elif prefix and not model.config.tie_word_embeddings:
        plan['lm_head'] = 'row'
    for index, layer in enumerate(backbone.layers):
        path = prefix + f'layers.{index}.'
        attention = path + 'self_attn.'
        if layer.is_linear_attn:
            for name in ('q_proj', 'k_proj', 'v_proj', 'f_b_proj', 'g_b_proj'):
                plan[attention + name] = 'column'
            for name in ('f_a_proj', 'g_a_proj', 'b_proj', 'o_proj'):
                plan[attention + name] = 'row'
        else:
            plan[attention + 'q_proj'] = 'column'
            plan[attention + 'kv_b_proj'] = 'column'
            plan[attention + 'kv_a_proj_with_mqa'] = 'row'
            plan[attention + 'o_proj'] = 'row'
        def mlp(base, expert=False):
            gate, up, down = ('w1', 'w3', 'w2') if expert else ('gate_proj', 'up_proj', 'down_proj')
            plan[base + gate] = {'kind': 'column', 'gather_output': False}
            plan[base + up] = {'kind': 'column', 'gather_output': False}
            plan[base + down] = {'kind': 'row', 'input_is_parallel': True}
        if hasattr(layer, 'block_sparse_moe'):
            from .expert_parallel import ExpertParallelMLP
            if isinstance(layer.block_sparse_moe.experts, ExpertParallelMLP):
                bank = layer.block_sparse_moe.experts
                if bank.tensor_group is None:
                    bank.shard_tensor_parallel(group)
                elif bank.tensor_group.group is not group.group:
                    raise ValueError('Kimi expert tensor shards belong to another group')
            else:
                for expert in range(len(layer.block_sparse_moe.experts)):
                    mlp(path + f'block_sparse_moe.experts.{expert}.', True)
            if hasattr(layer.block_sparse_moe, 'shared_experts'):
                mlp(path + 'block_sparse_moe.shared_experts.')
        else:
            mlp(path + 'mlp.')
    return tensor_parallelize(model, group, plan)
