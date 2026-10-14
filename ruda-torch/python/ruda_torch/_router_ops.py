"""Opaque dispatcher boundaries for the existing native router and adjoint."""
import math
import torch
from torch.autograd.function import once_differentiable


def metadata(logits,indices,scoring,renormalize,scale,gradient=None):
    if logits.ndim!=2 or indices.ndim!=2 or logits.shape[0]!=indices.shape[0] or logits.shape[1]==0 or not 1<=indices.shape[1]<=min(64,logits.shape[1]):
        raise ValueError('router requires [tokens,experts] and [tokens,top_k], 1 <= top_k <= min(experts,64)')
    if logits.dtype not in (torch.float32,torch.float16,torch.bfloat16) or indices.dtype not in (torch.int32,torch.int64):
        raise ValueError('router requires FP32/FP16/BF16 logits and int32/int64 indices')
    if logits.device!=indices.device or logits.device.type!='ruda' or logits.device.index not in (None,0):
        raise ValueError('router operands require ruda:0')
    if scoring not in (0,1) or type(renormalize) is not bool or not math.isfinite(scale) or scale<=0:
        raise ValueError('invalid router scoring, normalization or scale')
    if gradient is not None and (gradient.dtype!=torch.float32 or gradient.shape!=indices.shape or gradient.device!=logits.device):
        raise ValueError('router gradient must match selected FP32 weights')


@torch.library.custom_op('ruda::router_weights_forward',mutates_args=())
def forward(logits:torch.Tensor,indices:torch.Tensor,scoring:int,renormalize:bool,scale:float)->torch.Tensor:
    from . import _C
    metadata(logits,indices,scoring,renormalize,scale)
    return _C.router_weights_forward(logits,indices,scoring,renormalize,scale)


@forward.register_fake
def _forward_fake(logits,indices,scoring,renormalize,scale):
    metadata(logits,indices,scoring,renormalize,scale)
    return torch.empty(indices.shape,device=logits.device,dtype=torch.float32)


@torch.library.custom_op('ruda::router_weights_backward',mutates_args=())
def backward(logits:torch.Tensor,indices:torch.Tensor,gradient:torch.Tensor,scoring:int,renormalize:bool,scale:float)->torch.Tensor:
    from . import _C
    metadata(logits,indices,scoring,renormalize,scale,gradient)
    return _C.router_weights_backward(logits,indices,gradient.contiguous(),scoring,renormalize,scale)


@backward.register_fake
def _backward_fake(logits,indices,gradient,scoring,renormalize,scale):
    metadata(logits,indices,scoring,renormalize,scale,gradient)
    return torch.empty(logits.shape,device=logits.device,dtype=logits.dtype)


def _context(ctx,inputs,output):
    ctx.save_for_backward(*inputs[:2]);ctx.policy=inputs[2:];ctx.set_materialize_grads(False)


@once_differentiable
def _gradient(ctx,gradient):
    if gradient is None:return (None,)*5
    logits,indices=ctx.saved_tensors
    return backward(logits,indices,gradient,*ctx.policy),None,None,None,None


forward.register_autograd(_gradient,setup_context=_context)
