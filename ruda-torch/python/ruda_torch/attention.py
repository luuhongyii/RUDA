"""Device-preserving differentiable attention with explicit GQA and mask semantics."""
from __future__ import annotations

import math
import ctypes
from threading import RLock
import torch
from torch.autograd.function import once_differentiable
from ._architecture_ops import masked_softmax,positions,precision_context,work_dtype


_native_lock = RLock()
_native_ready = False


def _initialize_dense():
    global _native_ready
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
    return _C


class _DenseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,mask,score_shape,mask_kind,causal,scale,p):
        native = _initialize_dense()
        from .random import default_generator
        total = math.prod(score_shape)
        seed, counter = 0, 0
        if 0 < p < 1:
            with default_generator.lock:
                blocks = (total+3)//4
                if default_generator.counter+blocks >= 1<<64:
                    raise OverflowError('RUDA attention RNG counter exhausted')
                seed, counter = default_generator.seed_value, default_generator.counter
                default_generator.counter += blocks
        out = torch.empty((*q.shape[:-1],v.shape[-1]),device=q.device,dtype=q.dtype)
        stats = torch.empty((q.shape[0]*q.shape[1]*q.shape[2],2),device=q.device,dtype=torch.float32)
        native.dense_attention(0,[q,k,v,mask,out,stats],mask_kind,list(score_shape),causal,scale,p,seed,counter,False)
        ctx.save_for_backward(q,k,v,mask,stats)
        ctx.policy = score_shape,mask_kind,causal,scale,p,seed,counter
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        q,k,v,mask,stats = ctx.saved_tensors
        score_shape,mask_kind,causal,scale,p,seed,counter = ctx.policy
        dq,dk,dv = torch.empty_like(q),torch.empty_like(k),torch.empty_like(v)
        need_mask = mask_kind==2 and ctx.needs_input_grad[3]
        dm = torch.empty_like(mask) if need_mask else torch.empty(1,device=q.device,dtype=torch.float32)
        _initialize_dense().dense_attention(1,[q,k,v,mask,grad.contiguous(),stats,dq,dk,dv,dm],
            mask_kind,list(score_shape),causal,scale,p,seed,counter,need_mask)
        return dq,dk,dv,dm if need_mask else None,None,None,None,None,None


def _fused_dense(query,key,value,mask,p,causal,scale,gqa):
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
        mask = torch.empty(1,device=query.device,dtype=torch.float32)
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
        result = _DenseAttention.apply(q,k,v,mask,shape,kind,causal,scale,p)
    return result.reshape(*batch,heads,l,dv)


def scaled_dot_product_attention(query,key,value,*,attn_mask=None,dropout_p=0.,is_causal=False,
                                 scale=None,enable_gqa=False):
    """First-order MHA/GQA attention; tensor operations stay on the source device.

    RUDA uses plane-cooperative fused online softmax and recomputed backward,
    without storing scores, probabilities or dropout masks. CPU/CUDA retain
    the differentiable composition. Atomic KV/mask gradients are nondeterministic
    and obey torch.use_deterministic_algorithms. This is not a Tensor Core kernel.
    Boolean masks use True=allowed; floating masks are added before softmax.
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
    if not math.isfinite(factor):
        raise ValueError('attention scale must be finite')
    if enable_gqa:
        if key.shape[-3]!=value.shape[-3] or query.shape[-3]%key.shape[-3]:
            raise ValueError('GQA query heads must divide the common KV head count')
    if query.device.type=='ruda' and query.dtype in (torch.float16,torch.bfloat16,torch.float32) and query.shape[-1]>0 and value.shape[-1]>0:
        return _fused_dense(query,key,value,attn_mask,dropout_p,is_causal,factor,enable_gqa)
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
