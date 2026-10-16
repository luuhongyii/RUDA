import torch

from . import _C
from ._indexing import _logical
from .math_backend import _nvidia


def spsolve(A, B, *, left=True):
    if A.layout != torch.sparse_csr or B.layout != torch.strided:
        raise RuntimeError('sparse solve requires CSR coefficients and a strided right-hand side')
    if A.device.type != torch._C._get_privateuse1_backend_name() or A.device != B.device:
        raise RuntimeError('sparse solve requires coefficients and right-hand side on the same RUDA device')
    if A.ndim != 2 or A.dense_dim() or A.shape[0] != A.shape[1]:
        raise RuntimeError('sparse solve requires a non-batched square CSR matrix without dense payload dimensions')
    if B.ndim != 1 or B.shape[0] != A.shape[0]:
        raise RuntimeError('sparse solve requires a one-dimensional right-hand side matching the matrix size')
    if not left:
        raise RuntimeError('only left=True is supported by sparse solve')
    if A.dtype not in (torch.float32, torch.float64) or B.dtype != A.dtype:
        raise RuntimeError('sparse solve requires matching float32 or float64 dtypes')
    if _nvidia():
        raise NotImplementedError('explicit NVIDIA sparse solve requires a cuDSS route; native RUDA is not an implicit fallback')
    n = A.shape[0]
    if n > 2**31 - 1:
        raise RuntimeError('sparse solve exceeds the native pivot index limit')
    pointers, columns = A.crow_indices(), A.col_indices()
    if pointers.dtype not in (torch.int32, torch.int64) or columns.dtype != pointers.dtype:
        raise RuntimeError('sparse solve requires matching int32/int64 CSR indices')
    rhs = _logical(B).contiguous()
    if not n:
        return rhs.clone()
    pointers = _logical(pointers).to(torch.int64).contiguous()
    columns = _logical(columns).to(torch.int64).contiguous()
    values = _logical(A.values()).contiguous()
    rank = torch.full((n,), -1, dtype=torch.int64, device=A.device)
    pivots = torch.empty((n,), dtype=torch.int64, device=A.device)
    info = torch.zeros((1,), dtype=torch.int32, device=A.device)
    counts = torch.empty((n + 1,), dtype=torch.int64, device=A.device)
    for column in range(n):
        _C.scientific_call(542, [pointers, columns, values, rank, pivots, info, counts], [column], [])
        failure = int(info.item())
        if failure:
            raise torch.linalg.LinAlgError(f'sparse solve: singular matrix at pivot {failure}')
        new_pointers = counts.cumsum(0)
        nnz = int(new_pointers[-1].item())
        if not 0 <= nnz <= 2**32 - 1:
            raise RuntimeError('sparse solve fill-in exceeds native U32 storage indexing')
        new_columns = torch.empty((nnz,), dtype=torch.int64, device=A.device)
        new_values = torch.empty((nnz,), dtype=A.dtype, device=A.device)
        new_rhs = torch.empty_like(rhs)
        _C.scientific_call(543, [pointers, columns, values, rank, pivots, rhs,
                               new_pointers, new_columns, new_values, new_rhs], [column], [])
        pointers, columns, values, rhs = new_pointers, new_columns, new_values, new_rhs
        del new_pointers, new_columns, new_values, new_rhs
    result = torch.empty_like(rhs)
    _C.scientific_call(544, [pointers, columns, values, pivots, rhs, result], [], [])
    return result


def register(library):
    library.impl('_spsolve', spsolve)
