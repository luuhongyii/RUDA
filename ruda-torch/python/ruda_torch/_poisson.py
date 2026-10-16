import torch

from . import _C
from ._random_extra import _reservation


def poisson(a, generator=None):
    if a.device.type != 'ruda' or a.layout != torch.strided:
        raise RuntimeError('poisson requires a strided RUDA tensor')
    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise RuntimeError('poisson requires a supported floating-point dtype')
    if a.numel() > 2**32 - 1:
        raise RuntimeError('poisson exceeds native U32 indexing')
    result = torch.empty(a.shape, dtype=a.dtype, device=a.device)
    if not a.numel():
        return result
    with torch.no_grad(), _reservation(generator, a.numel()) as state:
        _C.scientific_call(424, [a.resolve_neg(), result], state, [])
    return result
