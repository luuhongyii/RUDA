"""Packed-document attention with fused native forward and recomputed backward."""
from __future__ import annotations

import torch
import math
from .block_attention import block_scaled_dot_product_attention


def _boundaries(values,length):
    if not isinstance(values,torch.Tensor) or values.device.type!='cpu' or values.dtype not in (torch.int32,torch.int64) or values.ndim!=1:
        raise ValueError('cu_seqlens must be explicit CPU integer metadata')
    result=values.tolist()
    if len(result)<2 or result[0]!=0 or result[-1]!=length or any(a>b for a,b in zip(result,result[1:])):
        raise ValueError('cu_seqlens must partition exactly the actual token payload')
    return result


def varlen_scaled_dot_product_attention(query,key,value,cu_seqlens_q,cu_seqlens_k=None,*,
        dropout_p=0.,is_causal=False,scale=None,enable_gqa=False,query_block_size=128,key_block_size=256,
        generator=None,window_size=None,causal_alignment='upper_left',softcap=None):
    """Attend independently within each real document, without an N-by-N mask.

    Payloads are [total_tokens,heads,features]. CPU boundaries are input metadata,
    never downloaded GPU activations. RUDA uses fused packed forward/backward;
    explicitly selected CPU/CUDA tensors retain same-device block attention.
    Causal alignment is explicit for unequal query/key lengths (including cache).
    """
    if query.ndim!=3 or key.ndim!=3 or value.ndim!=3 or key.shape[0]!=value.shape[0]:
        raise ValueError('varlen attention requires [tokens,heads,features] operands')
    if causal_alignment not in ('upper_left','lower_right'):raise ValueError('select an explicit causal alignment')
    if query.device.type == 'ruda':
        if query.device != key.device or query.device != value.device or query.dtype != key.dtype or query.dtype != value.dtype:
            raise ValueError('varlen operands must share dtype and device')
        if query.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError('native varlen attention requires FP32/FP16/BF16')
        if query.shape[-1] <= 0 or value.shape[-1] <= 0 or key.shape[-1] != query.shape[-1]:
            raise ValueError('invalid varlen feature dimensions')
        if type(is_causal) is not bool or type(enable_gqa) is not bool or not 0 <= dropout_p <= 1:
            raise ValueError('invalid varlen causal/GQA/dropout policy')
        if any(type(n) is not int or n <= 0 for n in (query_block_size, key_block_size)):
            raise ValueError('attention tile sizes must be positive integers')
        if window_size is not None and (not isinstance(window_size, (tuple, list)) or len(window_size) != 2 or
                any(type(n) is not int or n < -1 or n >= 1 << 63 for n in window_size)):
            raise ValueError('attention window must contain two int64 distances >= -1')
        factor = query.shape[-1] ** -.5 if scale is None else float(scale)
        if not math.isfinite(factor):
            raise ValueError('attention scale must be finite')
        if softcap is not None and (not math.isfinite(float(softcap)) or softcap <= 0):
            raise ValueError('softcap must be finite and positive')
        if enable_gqa:
            if key.shape[1] <= 0 or key.shape[1] != value.shape[1] or query.shape[1] % key.shape[1]:
                raise ValueError('GQA requires common KV heads dividing query heads')
        else:
            heads = torch.broadcast_shapes((query.shape[1],), (key.shape[1],), (value.shape[1],))[0]
            query = query.expand(query.shape[0], heads, query.shape[2])
            key = key.expand(key.shape[0], heads, key.shape[2])
            value = value.expand(value.shape[0], heads, value.shape[2])
        if query.shape[1] <= 0:
            raise ValueError('native varlen head counts must be positive')
        cu_seqlens_k = cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k
        for boundaries in (cu_seqlens_q, cu_seqlens_k):
            if not isinstance(boundaries, torch.Tensor) or boundaries.device.type != 'cpu' or boundaries.dtype not in (torch.int32, torch.int64) or boundaries.ndim != 1 or boundaries.numel() < 2:
                raise ValueError('cu_seqlens must be explicit CPU integer metadata')
        if cu_seqlens_q.numel() != cu_seqlens_k.numel():
            raise ValueError('query/key document counts differ')
        from . import _varlen_ops
        words = [-1, 0, 0, 0, 0]
        if generator is not None and 0 < dropout_p < 1:
            _, blocks = _varlen_ops._schedule(query, key, cu_seqlens_q, cu_seqlens_k)
            words = [0, *_varlen_ops.reserve(blocks, generator)]
        rng = torch.tensor(words, dtype=torch.int64, device='cpu')
        left, right = (-1, -1) if window_size is None else window_size
        return _varlen_ops.forward(query.contiguous(), key.contiguous(), value.contiguous(),
            cu_seqlens_q, cu_seqlens_k, is_causal, causal_alignment == 'lower_right', left, right,
            factor, float(dropout_p), 0.0 if softcap is None else float(softcap), rng)[0]
    queries=_boundaries(cu_seqlens_q,query.shape[0])
    keys=_boundaries(cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k,key.shape[0])
    if len(queries)!=len(keys):raise ValueError('query/key document counts differ')
    outputs=[]
    for index in range(len(queries)-1):
        qa,qb=queries[index:index+2]
        ka,kb=keys[index:index+2]
        q=query[qa:qb].transpose(0,1)
        k=key[ka:kb].transpose(0,1)
        v=value[ka:kb].transpose(0,1)
        offset=0 if causal_alignment=='upper_left' else (kb-ka)-(qb-qa)
        outputs.append(block_scaled_dot_product_attention(q,k,v,dropout_p=dropout_p,is_causal=is_causal,
            scale=scale,enable_gqa=enable_gqa,query_block_size=query_block_size,key_block_size=key_block_size,
            generator=generator,causal_offset=offset,window_size=window_size,softcap=softcap).transpose(0,1))
    return torch.cat(outputs,dim=0)
