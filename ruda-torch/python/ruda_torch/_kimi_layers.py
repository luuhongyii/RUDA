"""Checkpoint-shaped Kimi Linear layers using RUDA differentiable primitives."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from ._architecture_ops import positions, precision_context
from ._recurrent_ops import causal_depthwise_conv1d, gated_rms_norm
from .architecture_cache import KDAState, KVCache
from .kda import kda
from .varlen_attention import varlen_scaled_dot_product_attention


class TokenLayout:
    def __init__(self, batch, length, device, valid=None):
        self.batch, self.length = batch, length
        self.valid = valid
        if valid is None:
            self.order = self.inverse = self.lengths = self.packed_valid = None
        else:
            if valid.shape != (batch, length) or valid.device != device or valid.dtype != torch.bool:
                raise ValueError('Kimi padding metadata must match the token batch')
            self.order = (~valid).to(torch.int64).argsort(dim=-1, stable=True)
            self.inverse = self.order.argsort(dim=-1)
            self.lengths = valid.to(torch.int64).sum(-1)
            self.packed_valid = positions(length, device)[None, :] < self.lengths[:, None]
        self._counts = None

    def pack(self, value):
        if self.order is None:
            return value
        index = self.order.reshape(self.batch, self.length, *([1] * (value.ndim - 2))).expand(value.shape)
        packed = value.gather(1, index)
        valid = self.packed_valid.reshape(self.batch, self.length, *([1] * (value.ndim - 2)))
        return torch.where(valid, packed, torch.zeros_like(packed))

    def unpack(self, value):
        if self.inverse is None:
            return value
        index = self.inverse.reshape(self.batch, self.length, *([1] * (value.ndim - 2))).expand(value.shape)
        return value.gather(1, index)

    def counts(self):
        if self._counts is None:
            self._counts = ([self.length] * self.batch if self.lengths is None else
                            self.lengths.detach().to('cpu').tolist())
        return self._counts

    def indices(self, device):
        counts = self.counts()
        if self.order is None:
            index = positions(self.batch * self.length, device)
        else:
            index = torch.cat([self.order[row, :count] + row * self.length
                               for row, count in enumerate(counts)])
        boundaries = [0]
        for count in counts:
            boundaries.append(boundaries[-1] + count)
        return index, torch.tensor(boundaries, dtype=torch.int64, device='cpu')


class KimiRMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.variance_epsilon = eps

    def forward(self, value):
        with precision_context(value):
            work = value.float()
            work = work * (work.square().mean(-1, keepdim=True) + self.variance_epsilon).rsqrt()
        return self.weight * work.to(value.dtype)


class KimiGatedRMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, value, gate):
        return gated_rms_norm(value, gate, self.weight, self.eps,
                              activation='sigmoid', norm_before_gate=True)


class KimiShortConvolution(nn.Module):
    def __init__(self, width, window):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width, 1, window))
        self.window = window

    def forward(self, value, state, layout):
        batch, length, width = value.shape
        if state is None:
            state = value.new_zeros(batch, width, self.window)
        if state.shape != (batch, width, self.window) or state.device != value.device:
            raise ValueError('Kimi convolution cache shape/device differs')
        state = state.to(value.dtype)
        result, _ = causal_depthwise_conv1d(value, self.weight.to(value.dtype),
                                           state=state[..., 1:], activation='silu')
        joined = torch.cat((state, value.transpose(1, 2)), -1)
        if layout.lengths is None:
            tail = joined[..., -self.window:]
        else:
            index = layout.lengths[:, None] + positions(self.window, value.device)[None, :]
            tail = joined.gather(-1, index[:, None, :].expand(batch, width, self.window))
        return result, tail.contiguous()


class KimiDeltaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        linear = config.linear_attn_config
        self.head_dim, self.num_heads = linear['head_dim'], linear['num_heads']
        width, hidden = self.head_dim * self.num_heads, config.hidden_size
        for name in ('q_proj', 'k_proj', 'v_proj'):
            setattr(self, name, nn.Linear(hidden, width, bias=False))
        for name in ('q_conv1d', 'k_conv1d', 'v_conv1d'):
            setattr(self, name, KimiShortConvolution(width, linear['short_conv_kernel_size']))
        self.A_log = nn.Parameter(torch.empty(1, 1, self.num_heads, 1, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(width, dtype=torch.float32))
        self.f_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, width, bias=False)
        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False)
        self.g_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, width, bias=False)
        self.o_norm = KimiGatedRMSNorm(self.head_dim, config.rms_norm_eps)
        self.o_proj = nn.Linear(width, hidden, bias=False)

    def forward(self, hidden, layout, state=None):
        hidden = layout.pack(hidden)
        batch, length = hidden.shape[:2]
        convolutions = (None,) * 3 if state is None else state.convolution
        projected, tails = [], []
        for projection, convolution, previous in zip(
                (self.q_proj, self.k_proj, self.v_proj),
                (self.q_conv1d, self.k_conv1d, self.v_conv1d), convolutions):
            value, tail = convolution(projection(hidden), previous, layout)
            projected.append(value.reshape(batch, length, self.num_heads, self.head_dim))
            tails.append(tail)
        q, k, v = projected
        raw_decay = self.f_b_proj(self.f_a_proj(hidden))
        raw_beta = self.b_proj(hidden)
        with precision_context(hidden):
            q = (q.float() * (q.float().square().sum(-1, keepdim=True) + 1e-6).rsqrt()).to(q.dtype)
            k = (k.float() * (k.float().square().sum(-1, keepdim=True) + 1e-6).rsqrt()).to(k.dtype)
            decay = -F.softplus(raw_decay.float() + self.dt_bias.float())
            decay = decay.reshape(batch, length, self.num_heads, self.head_dim) * self.A_log.float().exp()
            beta = raw_beta.float().sigmoid()
        if layout.packed_valid is not None:
            valid = layout.packed_valid[:, :, None, None]
            q = torch.where(valid, q, torch.zeros_like(q))
            k = torch.where(valid, k, torch.zeros_like(k))
            v = torch.where(valid, v, torch.zeros_like(v))
            decay = torch.where(valid, decay, torch.zeros_like(decay))
            beta = torch.where(valid[..., 0], beta, torch.zeros_like(beta))
        output, recurrent = kda(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                decay.transpose(1, 2), beta.transpose(1, 2),
                                initial_state=None if state is None else state.recurrent,
                                normalize_qk=False, chunk_size=64)
        gate = self.g_b_proj(self.g_a_proj(hidden)).reshape(batch, length, self.num_heads, self.head_dim)
        output = self.o_norm(output.transpose(1, 2), gate).reshape(batch, length, self.num_heads * self.head_dim)
        output = layout.unpack(self.o_proj(output))
        seen = (0 if state is None else state.seen_tokens) + length
        return output, KDAState(recurrent, tuple(tails), seen)


class KimiMLAAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.nope, self.shared = config.qk_nope_head_dim, config.qk_rope_head_dim
        self.rank, self.value_dim = config.kv_lora_rank, config.v_head_dim
        self.query_dim = self.nope + self.shared
        self.dropout = getattr(config, 'attention_dropout', 0.)
        self.q_proj = nn.Linear(config.hidden_size, self.heads * self.query_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(config.hidden_size, self.rank + self.shared, bias=False)
        self.kv_a_layernorm = KimiRMSNorm(self.rank, 1e-6)
        self.kv_b_proj = nn.Linear(self.rank, self.heads * (self.nope + self.value_dim), bias=False)
        self.o_proj = nn.Linear(self.heads * self.value_dim, config.hidden_size, bias=False)

    def forward(self, hidden, query_indices, key_indices, cu_q, cu_k, state=None):
        batch, length = hidden.shape[:2]
        q = self.q_proj(hidden).reshape(batch, length, self.heads, self.query_dim)
        compressed, shared = self.kv_a_proj_with_mqa(hidden).split((self.rank, self.shared), -1)
        expanded = self.kv_b_proj(self.kv_a_layernorm(compressed))
        k, v = expanded.reshape(batch, length, self.heads, self.nope + self.value_dim).split((self.nope, self.value_dim), -1)
        k = torch.cat((k, shared[:, :, None, :].expand(batch, length, self.heads, self.shared)), -1)
        cache = (KVCache(k.transpose(1, 2), v.transpose(1, 2), length) if state is None else
                 state.append(k.transpose(1, 2), v.transpose(1, 2)))
        k = cache.key.transpose(1, 2).reshape(-1, self.heads, self.query_dim).index_select(0, key_indices)
        v = cache.value.transpose(1, 2).reshape(-1, self.heads, self.value_dim).index_select(0, key_indices)
        q = q.reshape(-1, self.heads, self.query_dim).index_select(0, query_indices)
        output = varlen_scaled_dot_product_attention(q, k, v, cu_q, cu_k,
            is_causal=True, causal_alignment='lower_right', scale=self.query_dim ** -.5,
            dropout_p=self.dropout if self.training else 0.)
        output = output.reshape(-1, self.heads * self.value_dim)
        result = output.new_zeros(batch * length, self.heads * self.value_dim)
        result = result.index_add(0, query_indices, output).reshape(batch, length, self.heads * self.value_dim)
        return self.o_proj(result), cache


class KimiMLP(nn.Module):
    def __init__(self, hidden, intermediate, expert=False):
        super().__init__()
        self.expert = expert
        for name, inp, out in (('w1' if expert else 'gate_proj', hidden, intermediate),
                               ('w3' if expert else 'up_proj', hidden, intermediate),
                               ('w2' if expert else 'down_proj', intermediate, hidden)):
            setattr(self, name, nn.Linear(inp, out, bias=False))

    def forward(self, value):
        gate, up, down = ((self.w1, self.w3, self.w2) if self.expert else
                          (self.gate_proj, self.up_proj, self.down_proj))
        return down(F.silu(gate(value)) * up(value))


class KimiMoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.empty(config.num_experts, config.hidden_size))
        self.e_score_correction_bias = nn.Parameter(torch.empty(config.num_experts))

    def forward(self, value):
        config = self.config
        with precision_context(value):
            logits = F.linear(value.reshape(-1, value.shape[-1]).float(), self.weight.float())
            scores = logits.sigmoid() if config.moe_router_activation_func == 'sigmoid' else logits.softmax(-1)
            scores = scores + self.e_score_correction_bias.float()
            grouped = scores.reshape(-1, config.num_expert_group, config.num_experts // config.num_expert_group)
            group_scores = grouped.topk(2, dim=-1).values.sum(-1)
            chosen = group_scores.topk(config.topk_group, dim=-1, sorted=False).indices
            groups = positions(config.num_expert_group, value.device)
            allowed = (chosen[:, :, None] == groups[None, None, :]).any(1)
            allowed = allowed[:, :, None].expand(grouped.shape).reshape(scores.shape)
            ids = scores.masked_fill(~allowed, 0.).topk(config.num_experts_per_token, dim=-1, sorted=False).indices
            weights = scores.gather(-1, ids)
            if config.num_experts_per_token > 1 and config.moe_renormalize:
                weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
            return ids, weights * config.routed_scaling_factor


class KimiSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = KimiMoEGate(config)
        self.experts = nn.ModuleList([KimiMLP(config.hidden_size, config.moe_intermediate_size, True)
                                      for _ in range(config.num_experts)])
        if config.num_shared_experts is not None:
            self.shared_experts = KimiMLP(config.hidden_size, config.moe_intermediate_size * config.num_shared_experts)

    def forward(self, hidden):
        ids, weights = self.gate(hidden)
        from .expert_parallel import ExpertParallelMLP
        if isinstance(self.experts, ExpertParallelMLP):
            shape = hidden.shape
            output = self.experts(hidden.reshape(-1, shape[-1]), ids, weights).reshape(shape)
            if hasattr(self, 'shared_experts'):
                output = output + self.shared_experts(hidden)
            return output
        flat = hidden.reshape(-1, hidden.shape[-1])
        if not flat.shape[0]:
            return hidden
        order = ids.reshape(-1).argsort(stable=True)
        routed = flat.index_select(0, torch.div(order, ids.shape[-1], rounding_mode='floor'))
        from ._indexed_experts import dense_bank
        output = dense_bank(self.experts, routed, ids.reshape(-1).index_select(0, order))
        if output is None:
            counts = torch.stack([(ids == expert).to(torch.int64).sum() for expert in range(len(self.experts))])
            counts = counts.detach().to('cpu').tolist()
            outputs, start = [], 0
            for expert, count in zip(self.experts, counts):
                if count:
                    outputs.append(expert(routed[start:start + count]))
                start += count
            output = torch.cat(outputs, 0)
        output = output.index_select(0, order.argsort())
        with precision_context(output):
            output = (output.reshape(*ids.shape, -1).float() * weights[:, :, None]).sum(1).to(output.dtype)
        output = output.reshape(hidden.shape)
        if hasattr(self, 'shared_experts'):
            output = output + self.shared_experts(hidden)
        return output


class KimiDecoderLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.is_linear_attn = config.is_kda_layer(index)
        self.self_attn = KimiDeltaAttention(config) if self.is_linear_attn else KimiMLAAttention(config)
        if config.is_moe_layer(index):
            self.block_sparse_moe = KimiSparseMoeBlock(config)
        else:
            self.mlp = KimiMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = KimiRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = KimiRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden, layout, attention, state=None, documents=None):
        normalized = self.input_layernorm(hidden)
        sequence_group = getattr(layout, 'sequence_group', None)
        if sequence_group is not None:
            from .parallel_training import gather_from_tensor_parallel
            normalized = gather_from_tensor_parallel(normalized, sequence_group, axis=1)
        if getattr(layout, 'context', None) is not None:
            from .hybrid_context_parallel import kimi_recurrent_forward, kimi_mla_forward
            operation = kimi_recurrent_forward if self.is_linear_attn else kimi_mla_forward
            mixed, updated = operation(self.self_attn, normalized, layout)
        elif self.is_linear_attn:
            if documents is None:
                mixed, updated = self.self_attn(normalized, layout, state)
            else:
                pieces = []
                for start, stop in zip(documents, documents[1:]):
                    if start != stop:
                        document = normalized[:, start:stop]
                        piece, _ = self.self_attn(document, TokenLayout(1, stop - start, hidden.device))
                        pieces.append(piece)
                mixed = torch.cat(pieces, 1) if pieces else normalized * 0
                updated = None
        else:
            mixed, updated = self.self_attn(normalized, *attention, state)
        if sequence_group is not None:
            from .parallel_training import scatter_to_tensor_parallel
            mixed = scatter_to_tensor_parallel(mixed, sequence_group, axis=1)
        hidden = hidden + mixed
        normalized = self.post_attention_layernorm(hidden)
        if sequence_group is not None:
            normalized = gather_from_tensor_parallel(normalized, sequence_group, axis=1)
        output = self.block_sparse_moe(normalized) if hasattr(self, 'block_sparse_moe') else self.mlp(normalized)
        if sequence_group is not None:
            output = scatter_to_tensor_parallel(output, sequence_group, axis=1)
        return hidden + output, updated
