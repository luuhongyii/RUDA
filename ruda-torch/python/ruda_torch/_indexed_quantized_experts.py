"""GPU-selected frozen expert projections retaining NF4 and packed storage."""
from __future__ import annotations
import torch
from torch.autograd.function import once_differentiable


def metadata(value,buffers,ids,start,layouts,transpose):
    if len(layouts)%6 or not 1<=len(layouts)//6<=16 or len(buffers)!=len(layouts)//6*4:
        raise ValueError('quantized expert bank requires four buffers and six layout words per expert')
    count=len(layouts)//6
    if type(start) is not int or not 0<=start<=0xffffffff-count or type(transpose) is not bool:
        raise ValueError('invalid quantized expert offset or transpose')
    if value.device.type!='ruda' or value.ndim!=2 or value.dtype not in (torch.float32,torch.float16,torch.bfloat16):
        raise ValueError('quantized experts require a floating RUDA matrix')
    if ids.shape!=(value.shape[0],) or ids.dtype not in (torch.int32,torch.int64) or ids.device!=value.device:
        raise ValueError('quantized expert IDs must match the routed rows')
    width=None
    for expert in range(count):
        format,group,inputs,outputs,groups,core=layouts[6*expert:6*expert+6]
        if any(type(n) is not int or not 0<=n<=0xffffffff for n in (format,group,inputs,outputs,groups,core)):
            raise ValueError('quantized expert layout exceeds uint32')
        if min(group,inputs,outputs)<1 or inputs*outputs>0xffffffff or core not in (0,1):
            raise ValueError('invalid quantized expert dimensions or strategy')
        if core and value.dtype not in (torch.float16,torch.bfloat16):
            raise ValueError('quantized Tensor Core experts require FP16/BF16 activations')
        expected=outputs if transpose else inputs
        columns=inputs if transpose else outputs
        if value.shape[1]!=expected or width is not None and columns!=width:
            raise ValueError('expert projections must share input/output dimensions')
        width=columns
        code,scale,offset,index=buffers[4*expert:4*expert+4]
        if any(t.device!=value.device or t.requires_grad for t in (code,scale,offset,index)):
            raise ValueError('quantized expert buffers must be frozen on the input device')
        if (code.dtype,scale.dtype,offset.dtype,index.dtype)!=(torch.uint8,torch.float32,torch.float32,torch.int32):
            raise ValueError('quantized expert buffer dtypes differ')
        if format==0:
            if (group%2 or groups!=0 or index.numel()!=0 or code.numel()!=(inputs*outputs+1)//2
                    or scale.numel()!=(inputs*outputs+group-1)//group or offset.numel()!=16):
                raise ValueError('NF4 expert layout differs')
        else:
            if format not in (2,3,4,5,6,8,14,18,19) or groups<1:
                raise ValueError('unsupported packed expert format')
            bits=4 if format==14 else 8 if format>=18 else format
            if (code.numel()!=outputs*((inputs*bits+7)//8) or scale.shape!=(outputs,groups)
                    or offset.shape!=scale.shape or index.numel() not in (0,inputs)
                    or not index.numel() and groups!=(inputs+group-1)//group):
                raise ValueError('packed expert affine/group layout differs')
    if value.shape[0]*width>0xffffffff:
        raise ValueError('quantized expert output exceeds uint32 indexing')
    return width


@torch.library.custom_op('ruda::indexed_quantized_expert_linear',mutates_args=())
def forward(value:torch.Tensor,buffers:list[torch.Tensor],ids:torch.Tensor,start:int,
            layouts:list[int],transpose:bool)->torch.Tensor:
    from . import _C,_indexed_quantized_expert_available
    metadata(value,buffers,ids,start,layouts,transpose)
    if not _indexed_quantized_expert_available:
        raise RuntimeError('indexed quantized expert API 1 requires matching native libraries')
    return _C.indexed_quantized_expert_linear(value,buffers,ids,start,layouts,transpose)


@forward.register_fake
def _fake(value,buffers,ids,start,layouts,transpose):
    return value.new_empty((value.shape[0],metadata(value,buffers,ids,start,layouts,transpose)))


def _context(ctx,inputs,output):
    value,buffers,ids,ctx.start,ctx.layouts,ctx.transpose=inputs
    ctx.save_for_backward(ids,*buffers)
    ctx.set_materialize_grads(False)


@once_differentiable
def _backward(ctx,gradient):
    ids,*buffers=ctx.saved_tensors
    dx=None if gradient is None or not ctx.needs_input_grad[0] else forward(
        gradient.contiguous(),buffers,ids,ctx.start,ctx.layouts,not ctx.transpose)
    return dx,[None]*len(buffers),None,None,None,None


forward.register_autograd(_backward,setup_context=_context)


def linear(value,layers,ids,*,start=0):
    from .finetuning import NF4Linear
    from .packed_quantization import PackedLinear,FORMATS
    result=None
    for begin in range(0,len(layers),16):
        buffers=[];layouts=[]
        for layer in layers[begin:begin+16]:
            if type(layer) is NF4Linear:
                buffers.extend((layer.packed,layer.scales,layer.codebook,
                                torch.empty(0,device=value.device,dtype=torch.int32)))
                layouts.extend((0,layer.block_size,layer.in_features,layer.out_features,0,
                                int(value.dtype in (torch.float16,torch.bfloat16))))
            elif type(layer) is PackedLinear:
                buffers.extend((layer.codes,layer.scales,layer.offsets,layer.group_index))
                layouts.extend((FORMATS[layer.format],layer.group_size,layer.in_features,layer.out_features,
                                layer.scales.shape[1],int(layer.strategy=='tensor_core')))
            else:
                raise TypeError('indexed quantized projection requires NF4Linear or PackedLinear')
            if layer.bias is not None:
                raise ValueError('indexed expert bias must be handled outside the frozen projection')
        part=forward(value.contiguous(),buffers,ids.contiguous(),start+begin,layouts,False)
        result=part if result is None else result+part
    return result
