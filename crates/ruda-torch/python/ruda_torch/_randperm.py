import operator

import torch

from . import _C
from ._random_extra import _reservation


def randperm_out(n, *, generator=None, out):
    n = operator.index(n)
    bits = _C.randperm_prepare(n, out)
    if n == 0:
        return out
    with torch.no_grad():
        keys = torch.empty((n,), dtype=torch.int32 if bits <= 32 else torch.int64, device=out.device)
        with _reservation(generator, (2 * n + 3) // 4) as state:
            _C.scientific_call(422, [keys], [*state, bits], [])
        ordered, indices = torch.sort(keys, stable=True)
        del keys
        target = torch.empty((n,), dtype=out.dtype, device=out.device) if out.is_neg() else out
        target.copy_(indices)
        del indices
        with _reservation(generator, (n + 3) // 4) as state:
            _C.scientific_call(423, [ordered, target], state, [])
        if target is not out:
            out.copy_(target)
    return out
