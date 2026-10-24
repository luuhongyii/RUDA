"""Device-preserving differentiable attention with explicit GQA and mask semantics."""
from __future__ import annotations

import math
import ctypes
from threading import RLock
import torch
from ._architecture_ops import masked_softmax,positions,precision_context,work_dtype


_native_lock = RLock()
_native_ready = False
_indexed_native_ready = False


def _initialize_dense(indexed=False):
    global _native_ready, _indexed_native_ready
    from . import _C, _native
    with _native_lock:
        if not _native_ready:
            if not hasattr(_C, 'initialize_dense_attention') or not hasattr(_native, 'ruda_torch_dense_attention_api_version'):
                raise RuntimeError('fused dense attention requires rebuilding both Rust and C++ libraries')
            _native.ruda_torch_dense_attention_api_version.restype = ctypes.c_uint32
            if _native.ruda_torch_dense_attention_api_version()!=1 or _C.dense_attention_api_version!=1:
                raise RuntimeError('dense attention API mismatch; rebuild both native libraries')
            _C.initialize_dense_attention(ctypes.cast(_native.ruda_torch_dense_attention,ctypes.c_void_p).value)
            _native_ready = True
        if indexed and not _indexed_native_ready:
            version=getattr(_native,'ruda_torch_indexed_dense_attention_api_version',None)
            if version is None or getattr(_C,'indexed_dense_attention_api_version',0)!=1:
                raise RuntimeError('indexed dense attention requires rebuilding both native libraries')
            version.restype=ctypes.c_uint32
            if version()!=1:
                raise RuntimeError('indexed dense attention API mismatch')
            _C.initialize_indexed_dense_attention(ctypes.cast(_native.ruda_torch_indexed_dense_attention,ctypes.c_void_p).value)
            _indexed_native_ready=True
    return _C


def _fused_dense(query,key,value,mask,p,causal,scale,gqa,tensor_group=None,global_heads=None,head_start=None,rng_state=None,precision='fp32'):
    batch = torch.broadcast_shapes(query.shape[:-3],key.shape[:-3],value.shape[:-3])
    if gqa:
        heads,kv_heads = query.shape[-3],key.shape[-3]
    else:
        heads = torch.broadcast_shapes((query.shape[-3],),(key.shape[-3],),(value.shape[-3],))[0]
        kv_heads = heads
    l,s,d,dv = query.shape[-2],key.shape[-2],query.shape[-1],value.shape[-1]
    if heads==0 or kv_heads==0:
        raise ValueError('attention head counts must be positive')
    kind = 0
    if mask is None:
        mask = torch.zeros(1,device=query.device,dtype=torch.float32)
    else:
        if mask.device!=query.device:
            raise ValueError('attention mask must be on the input device')
        if mask.dtype==torch.bool: kind = 1
        elif mask.is_floating_point(): kind = 2
        else: raise ValueError('attention masks must be boolean or floating point')
        mask = mask.to(torch.float32).contiguous()
        if kind==2:
            # Addition can broadcast singleton query/head/batch axes too, as
            # in the existing composition. K/V retain their GQA head mapping.
            shape = torch.broadcast_shapes((*batch,heads,l,s),mask.shape)
            if shape[-1]!=s:
                raise ValueError('attention mask expands the KV length incompatibly with value')
            batch,heads,l = shape[:-3],shape[-3],shape[-2]
            if not gqa: kv_heads = heads
    b = math.prod(batch)
    q = query.expand(*batch,heads,l,d).contiguous().reshape(b,heads,l,d)
    k = key.expand(*batch,kv_heads,s,d).contiguous().reshape(b,kv_heads,s,d)
    v = value.expand(*batch,kv_heads,s,dv).contiguous().reshape(b,kv_heads,s,dv)
    shape = (*batch,heads,l,s)
    with precision_context(query):
        from ._dense_ops import forward,indexed_forward
        if tensor_group is None and global_heads is None and head_start is None and rng_state is None:
            result = forward(q,k,v,mask,list(shape),kind,causal,scale,p,precision=='flash')[0]
        else:
            from ._parallel_rng import head_partition
            global_heads,head_start,groups=head_partition(heads,tensor_group,global_heads,head_start)
            if rng_state is None:rng_state=torch.empty(0,device='cpu',dtype=torch.int64)
            result=indexed_forward(q,k,v,mask,list(shape),kind,causal,scale,p,global_heads,head_start,groups,rng_state,precision=='flash')[0]
    return result.reshape(*batch,heads,l,dv)


def scaled_dot_product_attention(query,key,value,*,attn_mask=None,dropout_p=0.,is_causal=False,
                                 scale=None,enable_gqa=False,tensor_group=None,global_heads=None,head_start=None,rng_state=None,precision='fp32'):
    """First-order MHA/GQA attention; tensor operations stay on the source device.

    RUDA uses plane-cooperative fused online softmax and recomputed backward,
    without storing scores, probabilities or dropout masks. CPU/CUDA retain
    the differentiable composition. Atomic KV/mask gradients are nondeterministic
    and obey torch.use_deterministic_algorithms. This is not a Tensor Core kernel.
    Boolean masks use True=allowed; floating masks are added before softmax.
    precision='flash' opts into BF16 Flash storage/gradient boundaries for native 512x128, dropout-free unmasked/causal inputs.
    """
    if query.ndim<3 or key.ndim!=query.ndim or value.ndim!=query.ndim:
        raise ValueError('attention requires [...,heads,tokens,features] operands')
    if query.device!=key.device or query.device!=value.device or query.dtype!=key.dtype or query.dtype!=value.dtype:
        raise ValueError('attention operands must share floating dtype and device')
    if query.shape[-1]!=key.shape[-1] or key.shape[-2]!=value.shape[-2]:
        raise ValueError('attention key/query/value dimensions differ')
    if not 0<=dropout_p<=1 or type(is_causal) is not bool or type(enable_gqa) is not bool:
        raise ValueError('invalid attention dropout/causal/GQA policy')
    if is_causal and attn_mask is not None:
        raise ValueError('supply either an attention mask or causal masking, not both')
    factor=query.shape[-1]**-.5 if scale is None else float(scale)
    if precision not in ('fp32','flash'):
        raise ValueError('attention precision must be fp32 or flash')
    if precision=='flash' and (query.device.type!='ruda' or query.dtype!=torch.bfloat16
            or query.shape[-2:]!=(512,128) or key.shape[-2:]!=(512,128)
            or value.shape[-2:]!=(512,128) or attn_mask is not None or dropout_p!=0 or factor<=0):
        raise ValueError('flash precision requires native BF16 512x128 attention, no explicit mask/dropout, and positive scale')
    if not math.isfinite(factor):
        raise ValueError('attention scale must be finite')
    if enable_gqa:
        if key.shape[-3]!=value.shape[-3] or query.shape[-3]%key.shape[-3]:
            raise ValueError('GQA query heads must divide the common KV head count')
    if query.device.type=='ruda' and query.dtype in (torch.float16,torch.bfloat16,torch.float32) and query.shape[-1]>0 and value.shape[-1]>0:
        return _fused_dense(query,key,value,attn_mask,dropout_p,is_causal,factor,enable_gqa,
            tensor_group,global_heads,head_start,rng_state,precision)
    if tensor_group is not None or global_heads is not None or head_start is not None or rng_state is not None:
        raise ValueError('globally indexed attention requires the RUDA native attention path')
    if enable_gqa:
        repeats=query.shape[-3]//key.shape[-3]
        key=key.repeat_interleave(repeats,dim=-3)
        value=value.repeat_interleave(repeats,dim=-3)
    if not key.shape[-2]:
        return query.new_zeros((*query.shape[:-1],value.shape[-1]))+query.sum()*0+value.sum()*0
    with precision_context(query):
        dtype=work_dtype(query)
        scores=query.to(dtype)@key.to(dtype).transpose(-1,-2)*factor
        allowed=torch.ones_like(scores,dtype=torch.bool)
        if is_causal:
            row=positions(query.shape[-2],query.device).unsqueeze(-1)
            col=positions(key.shape[-2],key.device).unsqueeze(0)
            allowed=col<=row
        elif attn_mask is not None:
            if attn_mask.device!=query.device:
                raise ValueError('attention mask must be on the input device')
            if attn_mask.dtype==torch.bool:
                allowed=attn_mask
            elif attn_mask.is_floating_point():
                scores=scores+attn_mask.to(dtype)
                allowed=scores!=float('-inf')
            else:
                raise ValueError('attention masks must be boolean or floating point')
        probability=masked_softmax(scores,allowed)
        if dropout_p:
            if query.device.type=='ruda':
                from .random import native_dropout
                probability=native_dropout(probability,dropout_p)[0]
            else:
                probability=torch.nn.functional.dropout(probability,p=dropout_p,training=True)
        return (probability@value.to(dtype)).to(query.dtype)
