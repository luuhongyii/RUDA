"""Dispatcher boundaries for native arbitrary-row expert GEMM and its VJP."""
import torch
from torch.autograd.function import once_differentiable


def metadata(x,weights,ids,gradient=None):
    if x.ndim!=2 or weights.ndim!=3 or ids.ndim!=1 or ids.shape[0]!=x.shape[0] or weights.shape[2]!=x.shape[1]:
        raise ValueError('grouped GEMM expects [M,K], [E,N,K] and expert IDs [M]')
    if x.shape[1]==0 or weights.shape[1]==0 or weights.shape[0]==0 and x.shape[0]!=0:
        raise ValueError('grouped GEMM requires positive K/N and experts for nonempty rows')
    if x.dtype not in (torch.float32,torch.float16,torch.bfloat16) or weights.dtype!=x.dtype or ids.dtype not in (torch.int32,torch.int64):
        raise ValueError('grouped GEMM requires matching FP32/FP16/BF16 operands and int32/int64 expert IDs')
    if x.device.type!='ruda' or x.device.index not in (None,0) or weights.device!=x.device or ids.device!=x.device:
        raise ValueError('grouped GEMM operands require the same ruda:0 device')
    if gradient is not None and (gradient.shape!=(x.shape[0],weights.shape[1]) or gradient.dtype!=x.dtype or gradient.device!=x.device):
        raise ValueError('grouped GEMM gradient shape, dtype or device mismatch')


@torch.library.custom_op('ruda::grouped_gemm',mutates_args=())
def forward(x:torch.Tensor,weights:torch.Tensor,ids:torch.Tensor,strategy:int)->torch.Tensor:
    from . import _C
    metadata(x,weights,ids)
    if strategy not in (0,1,2):raise ValueError('invalid grouped GEMM strategy')
    return _C.blas_grouped(x,weights,ids,strategy)


@forward.register_fake
def _forward_fake(x,weights,ids,strategy):
    metadata(x,weights,ids)
    if strategy not in (0,1,2):raise ValueError('invalid grouped GEMM strategy')
    return torch.empty((x.shape[0],weights.shape[1]),dtype=x.dtype,device=x.device)


@torch.library.custom_op('ruda::grouped_gemm_backward',mutates_args=())
def backward(x:torch.Tensor,weights:torch.Tensor,ids:torch.Tensor,gradient:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
    from . import _C
    metadata(x,weights,ids,gradient)
    return _C.blas_grouped_backward(x,weights,ids,gradient)


@backward.register_fake
def _backward_fake(x,weights,ids,gradient):
    metadata(x,weights,ids,gradient)
    return (torch.empty(x.shape,dtype=x.dtype,device=x.device),
            torch.empty(weights.shape,dtype=weights.dtype,device=weights.device))


def _context(ctx,inputs,output):
    ctx.save_for_backward(*inputs[:3]);ctx.set_materialize_grads(False)


@once_differentiable
def _gradient(ctx,gradient):
    if gradient is None:return (None,)*4
    x,weights,ids=ctx.saved_tensors
    dx,dw=backward(x,weights,ids,gradient)
    return dx if ctx.needs_input_grad[0] else None,dw if ctx.needs_input_grad[1] else None,None,None


forward.register_autograd(_gradient,setup_context=_context)
