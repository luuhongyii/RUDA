import operator
from functools import partial

import torch

from . import _C


def triangular_indices(row, col, offset=0, *, upper=False, dtype=torch.int64,
                       layout=None, device=None, pin_memory=None):
    row, col, offset = operator.index(row), operator.index(col), operator.index(offset)
    if row < 0 or col < 0:
        raise RuntimeError('triangular index dimensions must be nonnegative')
    if layout not in (None, torch.strided) or pin_memory:
        raise RuntimeError('RUDA triangular indices require strided device storage')
    if upper:
        row_start = min(row, max(0, -offset))
        first = min(col, max(0, col - offset))
        ramp_rows = min(row - row_start, first)
        boundary = row_start * col
        count = boundary + ramp_rows * (2 * first - ramp_rows + 1) // 2
    else:
        row_start = min(row, max(0, -offset))
        first = min(col, max(0, row_start + offset + 1))
        ramp_rows = min(row - row_start, col - first + 1) if first else 0
        boundary = ramp_rows * (2 * first + ramp_rows - 1) // 2
        count = boundary + (row - row_start - ramp_rows) * col
    if count * 2 > 2**32 - 1:
        raise RuntimeError('triangular indices exceed native U32 indexing')
    dtype = torch.get_default_dtype() if dtype is None else dtype
    if count and dtype not in (torch.int32, torch.int64):
        raise RuntimeError('nonempty triangular indices require int32 or int64 storage')
    result = torch.empty((2, count), dtype=dtype, layout=layout or torch.strided, device=device or 'ruda')
    if count:
        _C.scientific_call(412 if upper else 411, [result],
                           [col, row_start, first, ramp_rows, boundary], [])
    return result


def register(library):
    library.impl('tril_indices', triangular_indices)
    library.impl('triu_indices', partial(triangular_indices, upper=True))
