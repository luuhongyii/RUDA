"""Compact block metadata for device-indexed FP8 expert products."""
from __future__ import annotations
import torch
from ._indexed_float8_experts import metadata,_format
from .float8_training import _FLOATS,_STRATEGIES


def describe(value,weights,ids,gradient,start,mode,geometry):
    metadata(value,weights,ids,start)
    if mode not in (0,1,2) or len(geometry)!=6 or any(type(v) is not int for v in geometry):
        raise ValueError('invalid block FP8 product mode or geometry')
    ra,rb,bk,axis,rank,world=geometry
    if (min(ra,rb,bk,world)<1 or bk%32 or max(ra*bk,rb*bk)>0xffffffff-32
            or world>0xffffffff or axis not in (-1,0,1) or not 0<=rank<world
            or axis==-1 and (rank,world)!=(0,1)):
        raise ValueError('invalid FP8 block geometry or actual tensor group')
    n,k=weights[0].shape;m=value.shape[0]
    if value.shape[1]!=(n if mode==1 else k) or gradient.device!=value.device:
        raise ValueError('block FP8 expert operand shape/device differs')
    if mode==2:
        if gradient.shape!=(m,n) or gradient.dtype not in _FLOATS:
            raise ValueError('block FP8 dW requires projected row gradients')
    elif gradient.shape!=(0,) or gradient.dtype!=value.dtype:
        raise ValueError('unused block FP8 gradient operand must be empty')
    ar,br,cols=(m,n,k) if mode==0 else (m,k,n) if mode==1 else (n,k,m)
    if (mode,axis) in ((0,1),(1,0)):
        cols*=world
    elif (mode,axis) in ((0,0),(1,1),(2,1)):
        br*=world
    elif (mode,axis)==(2,0):
        ar*=world
    if max(ar,br,cols)>0xffffffff:
        raise ValueError('logical block FP8 matrix exceeds uint32 indexing')
    ceil=lambda n,d:(n+d-1)//d
    count=len(weights)
    if mode==2:
        groups=ceil(m,bk)+count-1
        capacities=(ceil(ar,ra)*groups,ceil(br,rb)*groups)
    else:
        groups=ceil(cols,bk)
        capacities=((ceil(m,ra)+count-1)*groups,count*ceil(br,rb)*groups)
    if any(c>0xffffffff//2 for c in capacities):
        raise ValueError('compact block FP8 statistics exceed uint32 indexing')
    return [((2,c),torch.float32) for c in capacities]


@torch.library.custom_op('ruda::indexed_block_float8_statistics',mutates_args=())
def statistics(value:torch.Tensor,weights:list[torch.Tensor],ids:torch.Tensor,gradient:torch.Tensor,
               start:int,mode:int,geometry:list[int])->tuple[torch.Tensor,torch.Tensor]:
    from . import _C,_indexed_block_float8_expert_available
    specs=describe(value,weights,ids,gradient,start,mode,geometry)
    if not _indexed_block_float8_expert_available:
        raise RuntimeError('indexed block FP8 API 1 requires matching rebuilt native components')
    return tuple(_C.indexed_block_float8_statistics(value,weights,ids,gradient,start,mode,geometry,
                                                   [shape[1] for shape,_ in specs]))


@statistics.register_fake
def _statistics_fake(value,weights,ids,gradient,start,mode,geometry):
    return tuple(torch.empty(shape,device=value.device,dtype=dtype)
                 for shape,dtype in describe(value,weights,ids,gradient,start,mode,geometry))


def product_metadata(value,weights,ids,gradient,stats,start,mode,geometry,formats,margin,power,strategy,needed):
    specs=describe(value,weights,ids,gradient,start,mode,geometry)
    if len(stats)!=2 or any(t.shape!=shape or t.dtype!=dtype or t.device!=value.device
                           for t,(shape,dtype) in zip(stats,specs,strict=True)):
        raise ValueError('block FP8 statistics must match the compact expert layout')
    if (len(formats)!=2 or any(f not in (18,19) for f in formats) or type(margin) is not int or not 0<=margin<=16
            or type(power) is not bool or strategy not in (0,1,2) or len(needed)!=len(weights)
            or any(type(flag) is not bool for flag in needed)):
        raise ValueError('invalid indexed block FP8 recipe or derivative selection')
    if mode==2:
        return [(tuple(w.shape) if need else (0,),torch.float32) for w,need in zip(weights,needed,strict=True)]
    return [((value.shape[0],weights[0].shape[0 if mode==0 else 1]),torch.float32)]


@torch.library.custom_op('ruda::indexed_block_float8_product',mutates_args=())
def product(value:torch.Tensor,weights:list[torch.Tensor],ids:torch.Tensor,gradient:torch.Tensor,
            stats:list[torch.Tensor],start:int,mode:int,geometry:list[int],formats:list[int],margin:int,
            power:bool,strategy:int,needed:list[bool])->list[torch.Tensor]:
    from . import _C,_indexed_block_float8_expert_available
    product_metadata(value,weights,ids,gradient,stats,start,mode,geometry,formats,margin,power,strategy,needed)
    if not _indexed_block_float8_expert_available:
        raise RuntimeError('indexed block FP8 API 1 requires matching rebuilt native components')
    return _C.indexed_block_float8_product(value,weights,ids,gradient,stats,start,mode,geometry,formats,margin,power,strategy,needed)


@product.register_fake
def _product_fake(value,weights,ids,gradient,stats,start,mode,geometry,formats,margin,power,strategy,needed):
    return [torch.empty(shape,device=value.device,dtype=dtype) for shape,dtype in
            product_metadata(value,weights,ids,gradient,stats,start,mode,geometry,formats,margin,power,strategy,needed)]


def multiply(value,weights,ids,gradient,start,mode,recipe,group,axis,needed):
    roles=(0,1) if mode==0 else (2,1) if mode==1 else (2,0)
    blocks=(recipe.input_block_rows,recipe.weight_block_rows,recipe.grad_block_rows)
    formats=(recipe.input_dtype,recipe.weight_dtype,recipe.grad_dtype)
    rank,world=(0,1) if group is None else (group.rank,group.world_size)
    geometry=[blocks[roles[0]],blocks[roles[1]],recipe.block_size,axis,rank,world]
    stats=statistics(value,weights,ids,gradient,start,mode,geometry)
    if group is not None and group.world_size>1:
        from ._collective_ops import collective_reduce,group_key
        key=group_key(group)
        stats=tuple(collective_reduce(t,1,key) for t in stats)
    return product(value,weights,ids,gradient,list(stats),start,mode,geometry,
                   [_format(formats[roles[0]]),_format(formats[roles[1]])],recipe.margin,
                   recipe.power_of_two_scale,_STRATEGIES[recipe.strategy],needed)
