import math
import torch
from torch.autograd.function import once_differentiable


def _columns(flash):
    if flash:
        from . import _native
        version=getattr(_native,'ruda_torch_flash_precision_api_version',None)
        if version is None or version()!=1:
            raise RuntimeError('Flash precision requires rebuilding the RUDA native library')
    return 130 if flash else 2


def _rng_state(scores,p):
    from ._varlen_ops import reserve
    words=reserve((scores+3)//4) if 0<p<1 else [0,0,0,0]
    return torch.tensor(words,dtype=torch.int64,device='cpu')


def rng_words(state):
    if state.device.type!='cpu' or state.dtype!=torch.int64 or state.shape!=(4,):
        raise ValueError('attention RNG state requires four CPU int64 words')
    words=state.tolist()
    if any(word<0 or word>0xffffffff for word in words):raise ValueError('attention RNG word exceeds uint32')
    return words[0]|words[1]<<32,words[2]|words[3]<<32


@torch.library.custom_op('ruda::dense_attention_forward',mutates_args=(),tags=(torch.Tag.nondeterministic_seeded,))
def forward(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,mask:torch.Tensor,score_shape:list[int],
            mask_kind:int,causal:bool,scale:float,p:float,flash:bool=False)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    from .attention import _initialize_dense
    out=q.new_empty((*q.shape[:-1],v.shape[-1]))
    stats=torch.empty((q.shape[0]*q.shape[1]*q.shape[2],_columns(flash)),device=q.device,dtype=torch.float32)
    state=_rng_state(math.prod(score_shape),p);seed,counter=rng_words(state)
    _initialize_dense().dense_attention(0,[q,k,v,mask,out,stats],mask_kind,score_shape,causal,scale,p,seed,counter,False)
    return out,stats,state


@forward.register_fake
def _forward_fake(q,k,v,mask,score_shape,mask_kind,causal,scale,p,flash=False):
    return (q.new_empty((*q.shape[:-1],v.shape[-1])),
            torch.empty((q.shape[0]*q.shape[1]*q.shape[2],130 if flash else 2),device=q.device,dtype=torch.float32),
            torch.empty(4,device='cpu',dtype=torch.int64))


@torch.library.custom_op('ruda::indexed_dense_attention_forward',mutates_args=(),tags=(torch.Tag.nondeterministic_seeded,),
    schema='(Tensor q, Tensor k, Tensor v, Tensor mask, SymInt[] score_shape, SymInt mask_kind, bool causal, float scale, float p, SymInt global_heads, SymInt head_start, str[] rng_groups, Tensor requested_rng, bool flash=False) -> (Tensor, Tensor, Tensor)')
def indexed_forward(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,mask:torch.Tensor,score_shape:list[int],
            mask_kind:int,causal:bool,scale:float,p:float,global_heads:int,head_start:int,
            rng_groups:list[str],requested_rng:torch.Tensor,flash:bool=False)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    from .attention import _initialize_dense
    from ._parallel_rng import attention_state
    native=_initialize_dense(True)
    if global_heads<q.shape[1] or not 0<=head_start<=global_heads-q.shape[1]:
        raise ValueError('dense query heads lie outside the global head interval')
    out=q.new_empty((*q.shape[:-1],v.shape[-1]))
    stats=torch.empty((q.shape[0]*q.shape[1]*q.shape[2],_columns(flash)),device=q.device,dtype=torch.float32)
    state=attention_state((q.shape[0],global_heads,q.shape[2],k.shape[2]),p,rng_groups,requested_rng)
    seed,counter=rng_words(state)
    native.indexed_dense_attention(0,[q,k,v,mask,out,stats],mask_kind,score_shape,causal,scale,p,
        seed,counter,False,global_heads,head_start)
    return out,stats,state


@indexed_forward.register_fake
def _indexed_forward_fake(q,k,v,mask,score_shape,mask_kind,causal,scale,p,global_heads,head_start,rng_groups,requested_rng,flash=False):
    return _forward_fake(q,k,v,mask,score_shape,mask_kind,causal,scale,p,flash)


@torch.library.custom_op('ruda::dense_attention_backward',mutates_args=())
def backward(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,mask:torch.Tensor,grad:torch.Tensor,stats:torch.Tensor,
             rng:torch.Tensor,score_shape:list[int],mask_kind:int,causal:bool,scale:float,p:float,need_mask:bool
             )->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    from .attention import _initialize_dense
    dq,dk,dv=torch.empty_like(q),torch.empty_like(k),torch.empty_like(v)
    dm=torch.empty_like(mask) if need_mask else torch.empty(1,device=q.device,dtype=torch.float32)
    seed,counter=rng_words(rng)
    _initialize_dense().dense_attention(1,[q,k,v,mask,grad.contiguous(),stats,dq,dk,dv,dm],mask_kind,score_shape,causal,scale,p,seed,counter,need_mask)
    return dq,dk,dv,dm


@backward.register_fake
def _backward_fake(q,k,v,mask,grad,stats,rng,score_shape,mask_kind,causal,scale,p,need_mask):
    return (torch.empty_like(q),torch.empty_like(k),torch.empty_like(v),
            torch.empty_like(mask) if need_mask else torch.empty(1,device=q.device,dtype=torch.float32))


@torch.library.custom_op('ruda::indexed_dense_attention_backward',mutates_args=())
def indexed_backward(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,mask:torch.Tensor,grad:torch.Tensor,stats:torch.Tensor,
             rng:torch.Tensor,score_shape:list[int],mask_kind:int,causal:bool,scale:float,p:float,need_mask:bool,
             global_heads:int,head_start:int)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    from .attention import _initialize_dense
    dq,dk,dv=torch.empty_like(q),torch.empty_like(k),torch.empty_like(v)
    dm=torch.empty_like(mask) if need_mask else torch.empty(1,device=q.device,dtype=torch.float32)
    seed,counter=rng_words(rng)
    _initialize_dense(True).indexed_dense_attention(1,[q,k,v,mask,grad.contiguous(),stats,dq,dk,dv,dm],
        mask_kind,score_shape,causal,scale,p,seed,counter,need_mask,global_heads,head_start)
    return dq,dk,dv,dm


@indexed_backward.register_fake
def _indexed_backward_fake(q,k,v,mask,grad,stats,rng,score_shape,mask_kind,causal,scale,p,need_mask,global_heads,head_start):
    return _backward_fake(q,k,v,mask,grad,stats,rng,score_shape,mask_kind,causal,scale,p,need_mask)


def _context(ctx,inputs,output):
    q,k,v,mask,score_shape,mask_kind,causal,scale,p=inputs[:9]
    _,stats,rng=output
    ctx.save_for_backward(q,k,v,mask,stats,rng)
    ctx.policy=score_shape,mask_kind,causal,scale,p
    ctx.partition=inputs[9:11] if len(inputs)>=13 else ()
    ctx.argument_count=len(inputs)
    ctx.mark_non_differentiable(stats,rng);ctx.set_materialize_grads(False)


@once_differentiable
def _gradient(ctx,grad,_stats,_rng):
    if grad is None:return (None,)*ctx.argument_count
    q,k,v,mask,stats,rng=ctx.saved_tensors
    need_mask=ctx.policy[1]==2 and ctx.needs_input_grad[3]
    operation=indexed_backward if ctx.partition else backward
    dq,dk,dv,dm=operation(q,k,v,mask,grad,stats,rng,*ctx.policy,need_mask,*ctx.partition)
    return dq,dk,dv,dm if need_mask else None,*(None for _ in range(ctx.argument_count-4))


forward.register_autograd(_gradient,setup_context=_context)
indexed_forward.register_autograd(_gradient,setup_context=_context)


class _DenseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,mask,score_shape,mask_kind,causal,scale,p,flash):
        inputs=q,k,v,mask,score_shape,mask_kind,causal,scale,p,flash
        output=forward(*inputs)
        _context(ctx,inputs,output)
        return output

    @staticmethod
    def backward(ctx,grad,stats,rng):
        return _gradient(ctx,grad,stats,rng)


def eager_forward(q,k,v,mask,score_shape,mask_kind,causal,scale,p,flash=False):
    operation=(_DenseAttention.apply if not torch.compiler.is_compiling() and torch.is_grad_enabled()
               and any(t.requires_grad for t in (q,k,v,mask)) else forward)
    return operation(q,k,v,mask,score_shape,mask_kind,causal,scale,p,flash)
