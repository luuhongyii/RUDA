"""Native context-sharded ring attention with packed-document and GQA support."""
from __future__ import annotations

import math
import torch
from torch import nn
from ._expert_exchange import register_group


def ring_scaled_dot_product_attention(query, key, value, group, *, is_causal=False,
        scale=None, dropout_p=0., enable_gqa=False, window_size=None, softcap=None,
        query_positions=None, key_positions=None, query_documents=None, key_documents=None,
        query_mask=None, key_mask=None, rng_state=None, group_key='context-ring',
        tensor_group=None, global_heads=None, head_start=None):
    """Attend to context shards in rank order without assembling global K/V.

    Inputs use [batch,heads,local_tokens,features]. Token counts may differ by
    rank, including zero. Positions default to contiguous global offsets;
    explicit document IDs/positions describe packed streams and cached queries.
    Negative document IDs or false padding masks disable those tokens. The
    local output corresponds exactly to the local query slice. Dropout uses
    one global [B,H,Q,K] Philox range and recomputes the same mask in backward.
    tensor_group describes contiguous query-head shards; CP shards token axes.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError('ring attention requires [batch,heads,tokens,features] tensors')
    batch, heads, queries, dimension = query.shape
    if (batch < 1 or heads < 1 or dimension < 1 or key.shape[0] != batch or value.shape[0] != batch
            or key.shape[1] < 1 or key.shape[1] != value.shape[1] or key.shape[2] != value.shape[2]
            or key.shape[3] != dimension or value.shape[3] < 1
            or heads % key.shape[1] or not enable_gqa and heads != key.shape[1]):
        raise ValueError('ring attention batch/head/key/value geometry differs')
    if (query.device.type != 'ruda' or group.device_type != 'ruda'
            or query.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or any(t.device != query.device or t.dtype != query.dtype for t in (key, value))):
        raise ValueError('native ring attention requires same-device RUDA floating tensors')
    if type(is_causal) is not bool or type(enable_gqa) is not bool or not 0 <= dropout_p <= 1:
        raise ValueError('invalid ring attention causal/GQA/dropout policy')
    scale = dimension ** -.5 if scale is None else float(scale)
    softcap = 0. if softcap is None else float(softcap)
    if not math.isfinite(scale) or not math.isfinite(softcap) or softcap < 0:
        raise ValueError('ring attention scale/softcap must be finite')
    window = (-1, -1) if window_size is None else tuple(window_size)
    if len(window) != 2 or any(type(n) is not int or n < -1 for n in window):
        raise ValueError('ring attention window must contain two distances >= -1')

    def metadata(tensor, length, dtype):
        if tensor is None:
            return torch.empty(0, device=query.device, dtype=dtype)
        if tensor.device != query.device or tensor.dtype != dtype or tensor.shape != (batch, length):
            raise ValueError('ring token metadata must match the rank-local batch/token shape and dtype')
        return tensor.contiguous()

    qposition, kposition = metadata(query_positions, queries, torch.int64), metadata(key_positions, key.shape[2], torch.int64)
    qdocument, kdocument = metadata(query_documents, queries, torch.int64), metadata(key_documents, key.shape[2], torch.int64)
    qvalid, kvalid = metadata(query_mask, queries, torch.bool), metadata(key_mask, key.shape[2], torch.bool)
    if rng_state is None:
        rng_state = torch.empty(0, device='cpu', dtype=torch.int64)
    if rng_state.device.type != 'cpu' or rng_state.dtype != torch.int64 or rng_state.shape not in ((0,), (4,)):
        raise ValueError('explicit ring RNG state requires CPU int64[4]')
    from ._parallel_rng import head_partition
    global_heads,head_start,groups = head_partition(heads,tensor_group,global_heads,head_start)
    tensor_key = groups[0] if groups else ''
    from ._ring_ops import indexed_forward
    result = indexed_forward(query.transpose(1, 2).contiguous(), key.transpose(1, 2).contiguous(), value.transpose(1, 2).contiguous(),
        qposition, kposition, qdocument, kdocument, qvalid, kvalid, register_group(group_key, group),
        is_causal, window[0], window[1], scale, float(dropout_p), softcap, rng_state,
        global_heads, head_start, tensor_key)[0]
    return result.transpose(1, 2).contiguous()


class RingAttention(nn.Module):
    def __init__(self, group, *, group_key='context-ring', is_causal=True, enable_gqa=True,
                 dropout_p=0., scale=None, window_size=None, softcap=None,
                 tensor_group=None, global_heads=None, head_start=None):
        super().__init__()
        self.group, self.group_key = group, register_group(group_key, group)
        self.is_causal, self.enable_gqa = is_causal, enable_gqa
        self.dropout_p, self.scale = dropout_p, scale
        self.window_size, self.softcap = window_size, softcap
        self.tensor_group, self.global_heads, self.head_start = tensor_group, global_heads, head_start

    def forward(self, query, key, value, *, query_positions=None, key_positions=None,
                query_documents=None, key_documents=None, query_mask=None, key_mask=None, rng_state=None):
        return ring_scaled_dot_product_attention(query, key, value, self.group,
            group_key=self.group_key, is_causal=self.is_causal, enable_gqa=self.enable_gqa,
            tensor_group=self.tensor_group, global_heads=self.global_heads, head_start=self.head_start,
            dropout_p=self.dropout_p if self.training else 0., scale=self.scale,
            window_size=self.window_size, softcap=self.softcap,
            query_positions=query_positions, key_positions=key_positions,
            query_documents=query_documents, key_documents=key_documents,
            query_mask=query_mask, key_mask=key_mask, rng_state=rng_state)
