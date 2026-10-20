import operator

import torch

from . import _C
from ._ordering import _output
from ._random_extra import _reservation


def multinomial(a, num_samples, replacement=False, *, generator=None, out=None):
    num_samples = operator.index(num_samples)
    if a.device.type != 'ruda' or a.layout != torch.strided or a.ndim not in (1, 2):
        raise RuntimeError('multinomial requires a one- or two-dimensional strided RUDA tensor')
    if not a.is_floating_point():
        raise RuntimeError('multinomial requires floating-point probabilities')
    if num_samples <= 0:
        raise RuntimeError('multinomial requires a positive sample count')
    categories = a.shape[-1]
    if not replacement and num_samples > categories:
        raise RuntimeError('cannot sample more categories than exist without replacement')
    if categories > 1 << 24:
        raise RuntimeError('multinomial category count cannot exceed 2^24')
    rows = 1 if a.ndim == 1 else a.shape[0]
    shape = (num_samples,) if a.ndim == 1 else (rows, num_samples)
    if a.numel() > 2**32 - 1 or rows * num_samples > 2**32 - 1:
        raise RuntimeError('multinomial exceeds native U32 indexing')
    if out is not None:
        if out.device != a.device or out.dtype != torch.int64 or out.layout != torch.strided:
            raise RuntimeError('multinomial output must be strided int64 on the input device')
        _C.check_inplace(out, a)
        if tuple(out.shape) != shape:
            _C.prepare_index_output(out, list(shape))
    if rows == 0:
        result = torch.empty(shape, dtype=torch.int64, device=a.device)
        return _output(result, out, (a,))
    if categories == 0:
        raise RuntimeError('cannot sample an empty multinomial distribution')
    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise RuntimeError('unsupported multinomial probability dtype')
    with torch.no_grad():
        source = a.resolve_neg().reshape(rows, categories)
        if not replacement or num_samples == 1:
            scores = torch.empty((rows, categories), dtype=a.dtype, device=a.device)
            words = source.numel() * (2 if a.dtype == torch.float64 else 1)
            with _reservation(generator, (words + 3) // 4) as state:
                _C.scientific_call(419, [source, scores], state, [])
            if num_samples == 1:
                result = torch.argmax(scores, dim=1, keepdim=True)
            else:
                result = torch.topk(scores, num_samples, dim=1, largest=True, sorted=True).indices
        else:
            normalized = torch.empty((rows, categories), dtype=a.dtype, device=a.device)
            _C.scientific_call(420, [source, normalized], [], [])
            cumulative = torch.cumsum(normalized, dim=1)
            result = torch.empty((rows, num_samples), dtype=torch.int64, device=a.device)
            with _reservation(generator, (result.numel() + 3) // 4) as state:
                _C.scientific_call(421, [normalized, cumulative, result], state, [])
        return _output(result.reshape(shape), out, (a,))
