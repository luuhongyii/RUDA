import math

import torch

from . import _C


_floating = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _tensor(a, device, dtype):
    if a.layout != torch.strided or a.device.type != 'ruda' or a.device != device or a.dtype != dtype:
        raise RuntimeError('weight norm tensors must have matching RUDA device and required dtype')


def _dimensions(v, g, dim):
    if v.dtype not in _floating:
        raise RuntimeError('fused weight norm requires real floating tensors')
    _tensor(v, v.device, v.dtype)
    _tensor(g, v.device, v.dtype)
    if v.ndim == 0 or dim not in (0, v.ndim - 1):
        raise RuntimeError('fused weight norm requires the first or last dimension')
    first = dim == 0
    rows = v.size(0) if first else math.prod(v.shape[:-1])
    columns = math.prod(v.shape[1:]) if first else v.size(-1)
    groups = rows if first else columns
    if g.numel() < groups:
        raise RuntimeError('weight norm magnitude storage is smaller than the normalized dimension')
    return [rows, columns, int(first)]


def _physical(a):
    return a.resolve_neg().as_strided((a.numel(),), (1,))


def weight_norm(v, g, dim=0):
    dimensions = _dimensions(v, g, dim)
    output = torch.empty_like(v, memory_format=torch.contiguous_format)
    norms = torch.empty_strided(g.shape, g.stride(), dtype=torch.float64 if g.dtype == torch.float64
                               else torch.float32, device=g.device)
    _C.scientific_call(447, [_physical(v), _physical(g), _physical(output), _physical(norms)], dimensions, [])
    return output, norms


def weight_norm_backward(grad_w, saved_v, saved_g, saved_norms, dim):
    dimensions = _dimensions(saved_v, saved_g, dim)
    _tensor(grad_w, saved_v.device, saved_v.dtype)
    _tensor(saved_norms, saved_v.device, torch.float64 if saved_v.dtype == torch.float64 else torch.float32)
    if not all(value.is_contiguous() for value in (saved_v, saved_g, saved_norms)):
        raise RuntimeError('saved weight norm tensors must be contiguous')
    if grad_w.shape != saved_v.shape or saved_norms.shape != saved_g.shape:
        raise RuntimeError('weight norm gradient or saved norm shape does not match')
    grad_v = torch.empty_like(saved_v, memory_format=torch.contiguous_format)
    grad_g = torch.empty_like(saved_g, memory_format=torch.contiguous_format)
    tensors = [grad_w, saved_v, saved_g, saved_norms, grad_v, grad_g]
    _C.scientific_call(448, [_physical(value) for value in tensors], dimensions, [])
    return grad_v, grad_g


def register(library):
    library.impl('_weight_norm_interface', weight_norm)
    library.impl('_weight_norm_interface_backward', weight_norm_backward)
