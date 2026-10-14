import math
import torch
from torch.autograd.function import once_differentiable


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
            mask_kind:int,causal:bool,scale:float,p:float)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    from .attention import _initialize_dense
    out=q.new_empty((*q.shape[:-1],v.shape[-1]))
    stats=torch.empty((q.shape[0]*q.shape[1]*q.shape[2],2),device=q.device,dtype=torch.float32)
    state=_rng_state(math.prod(score_shape),p);seed,counter=rng_words(state)
    _initialize_dense().dense_attention(0,[q,k,v,mask,out,stats],mask_kind,score_shape,causal,scale,p,seed,counter,False)
    return out,stats,state


@forward.register_fake
def _forward_fake(q,k,v,mask,score_shape,mask_kind,causal,scale,p):
    return (q.new_empty((*q.shape[:-1],v.shape[-1])),
            torch.empty((q.shape[0]*q.shape[1]*q.shape[2],2),device=q.device,dtype=torch.float32),
            torch.empty(4,device='cpu',dtype=torch.int64))


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


def _context(ctx,inputs,output):
    q,k,v,mask,score_shape,mask_kind,causal,scale,p=inputs
    _,stats,rng=output
    ctx.save_for_backward(q,k,v,mask,stats,rng)
    ctx.policy=score_shape,mask_kind,causal,scale,p
    ctx.mark_non_differentiable(stats,rng);ctx.set_materialize_grads(False)


@once_differentiable
def _gradient(ctx,grad,_stats,_rng):
    if grad is None:return (None,)*9
    q,k,v,mask,stats,rng=ctx.saved_tensors
    need_mask=ctx.policy[1]==2 and ctx.needs_input_grad[3]
    dq,dk,dv,dm=backward(q,k,v,mask,grad,stats,rng,*ctx.policy,need_mask)
    return dq,dk,dv,dm if need_mask else None,None,None,None,None,None


forward.register_autograd(_gradient,setup_context=_context)
