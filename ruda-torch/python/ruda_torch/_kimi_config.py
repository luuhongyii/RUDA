"""Official Kimi Linear configuration and checkpoint geometry."""
from __future__ import annotations

from transformers import PretrainedConfig


class KimiLinearConfig(PretrainedConfig):
    model_type = 'kimi_linear'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(self, vocab_size=163840, hidden_size=4096, head_dim=None,
                 intermediate_size=11008, num_hidden_layers=32, num_attention_heads=32,
                 num_key_value_heads=None, hidden_act='silu', initializer_range=.02,
                 rms_norm_eps=1e-6, use_cache=True, pad_token_id=0, bos_token_id=1,
                 eos_token_id=2, rope_theta=10000., rope_scaling=None,
                 tie_word_embeddings=False, moe_intermediate_size=None,
                 moe_renormalize=True, moe_router_activation_func='sigmoid',
                 num_experts=None, num_experts_per_token=None, num_shared_experts=0,
                 routed_scaling_factor=1., first_k_dense_replace=0, moe_layer_freq=1,
                 use_grouped_topk=True, num_expert_group=1, topk_group=1,
                 q_lora_rank=None, kv_lora_rank=None, qk_nope_head_dim=None,
                 qk_rope_head_dim=None, v_head_dim=None, mla_use_nope=False,
                 num_nextn_predict_layers=0, linear_attn_config=None, **kwargs):
        values = dict(locals())
        values.pop('self')
        values.pop('kwargs')
        values.pop('__class__', None)
        for name, value in values.items():
            setattr(self, name, value)
        self.head_dim = hidden_size // num_attention_heads if head_dim is None else head_dim
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        source_type = kwargs.pop('model_type', self.model_type)
        if source_type != self.model_type:
            raise ValueError('Kimi Linear requires model_type=kimi_linear')
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id,
                         eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)

    def is_kda_layer(self, index):
        return self.linear_attn_config is not None and index + 1 in self.linear_attn_config['kda_layers']

    def is_moe_layer(self, index):
        return (self.num_experts is not None and index >= self.first_k_dense_replace
                and index % self.moe_layer_freq == 0)

    def validate_architecture(self):
        from ._architecture_ops import positive_int, positive_float
        for name in ('vocab_size', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
                     'num_attention_heads', 'num_key_value_heads', 'moe_layer_freq'):
            positive_int(getattr(self, name), name)
        positive_float(self.rms_norm_eps, 'rms_norm_eps')
        if self.hidden_act != 'silu':
            raise ValueError('Kimi Linear checkpoint requires silu feed-forward activation')
        if self.num_nextn_predict_layers != 0:
            raise ValueError('Kimi Linear MTP layers are not represented by the base causal model')
        if self.linear_attn_config is not None:
            config = self.linear_attn_config
            kda, full = config['kda_layers'], config['full_attn_layers']
            if (any(type(i) is not int for i in (*kda, *full)) or
                    len(set(kda)) != len(kda) or len(set(full)) != len(full) or
                    set(kda) & set(full) or set(kda) | set(full) != set(range(1, self.num_hidden_layers + 1))):
                raise ValueError('Kimi layer lists must partition the one-based decoder layers')
            if kda:
                for name in ('head_dim', 'num_heads', 'short_conv_kernel_size'):
                    positive_int(config[name], name)
        if any(not self.is_kda_layer(i) for i in range(self.num_hidden_layers)):
            if not self.mla_use_nope or self.q_lora_rank is not None:
                raise ValueError('Kimi Linear MLA requires NoPE and a direct Q projection')
            for name in ('kv_lora_rank', 'qk_nope_head_dim', 'v_head_dim'):
                positive_int(getattr(self, name), name)
            if type(self.qk_rope_head_dim) is not int or self.qk_rope_head_dim < 0:
                raise ValueError('qk_rope_head_dim must be nonnegative')
            if self.num_key_value_heads != self.num_attention_heads:
                raise ValueError('expanded Kimi MLA has one KV head per query head')
        if type(self.first_k_dense_replace) is not int or self.first_k_dense_replace < 0:
            raise ValueError('first_k_dense_replace must be nonnegative')
        if self.num_experts is not None:
            for name in ('num_experts', 'num_experts_per_token', 'moe_intermediate_size',
                         'num_expert_group', 'topk_group'):
                positive_int(getattr(self, name), name)
            if (self.num_experts % self.num_expert_group or
                    self.num_experts // self.num_expert_group < 2 or
                    self.topk_group > self.num_expert_group or
                    self.num_experts_per_token > self.topk_group * (self.num_experts // self.num_expert_group)):
                raise ValueError('invalid grouped top-2 Kimi router geometry')
            if self.moe_router_activation_func not in ('sigmoid', 'softmax'):
                raise ValueError('unsupported Kimi router scoring function')
            if self.num_shared_experts is not None and (type(self.num_shared_experts) is not int or self.num_shared_experts < 0):
                raise ValueError('num_shared_experts must be nonnegative or None')
            positive_float(self.routed_scaling_factor, 'routed_scaling_factor')
