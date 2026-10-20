import torch

from . import _C
from ._indexing import _logical
from ._ordering import _outputs
from .math_backend import _nvidia


def triangular_solve(self, A, upper=True, transpose=False, unitriangular=False, *, X=None, M=None):
    if A.layout not in (torch.sparse_csr, torch.sparse_bsr) or self.layout != torch.strided:
        raise RuntimeError('sparse triangular_solve requires CSR/BSR coefficients and a strided right-hand side')
    if (A.device.type != torch._C._get_privateuse1_backend_name() or A.device != self.device
            or A.dtype != self.dtype):
        raise RuntimeError('sparse triangular_solve requires matching dtypes and RUDA devices')
    if A.ndim != 2 or A.dense_dim() != 0 or self.ndim != 2:
        raise RuntimeError('sparse triangular_solve requires non-batched matrices without sparse dense payload dimensions')
    if A.shape[0] != A.shape[1] or A.shape[1] != self.shape[0]:
        raise RuntimeError('sparse triangular_solve requires square coefficients and a matching right-hand side')
    if (X is None) != (M is None):
        raise RuntimeError('triangular_solve requires both output tensors')
    if X is not None:
        for out in (X, M):
            if out.layout != torch.strided or out.dtype != self.dtype or out.device != self.device:
                raise RuntimeError('triangular_solve outputs must be strided with matching dtype and device')
    if _nvidia():
        raise NotImplementedError('explicit NVIDIA sparse triangular solve requires a cuSPARSE route; native RUDA is not an implicit fallback')
    pointers, indices, values = A.crow_indices(), A.col_indices(), A.values()
    if pointers.dtype not in (torch.int32, torch.int64) or pointers.dtype != indices.dtype:
        raise RuntimeError('sparse triangular_solve requires matching int32/int64 indices')
    result = torch.empty(self.shape, dtype=self.dtype, device=self.device)
    cloned = torch.empty((0,), dtype=self.dtype, device=self.device)
    if result.numel() and A._nnz():
        if A.dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
            raise RuntimeError('nonempty sparse triangular_solve requires float32/float64/complex64/complex128')
        values = _logical(values)
        if A.layout == torch.sparse_csr:
            values = values.unsqueeze(-1).unsqueeze(-1)
        elif values.shape[-2] != values.shape[-1] or values.shape[-1] <= 1:
            raise RuntimeError('BSR triangular_solve requires square blocks larger than one')
        block_size = values.shape[-1]
        pointers, indices = _logical(pointers), _logical(indices)
        order = torch.empty((0,), dtype=torch.int64, device=self.device)
        if transpose:
            from ._sparse_compressed import coo_to_csr
            order = indices.argsort(stable=True)
            rows = torch.empty_like(indices)
            _C.scientific_call(537, [pointers, order, rows], [], [])
            pointers = coo_to_csr(indices.index_select(0, order), A.shape[0] // block_size,
                                  out_int32=indices.dtype == torch.int32)
            indices = rows
        _C.scientific_call(536, [pointers, indices, values, _logical(self), result, order],
                           [int(bool(upper) != bool(transpose)), int(unitriangular), int(transpose)], [])
    elif result.numel():
        result.fill_(float('nan'))
    inputs = (self, A.crow_indices(), A.col_indices(), A.values())
    return _outputs((result, cloned), None if X is None else (X, M), inputs)


def register(library):
    library.impl('triangular_solve', triangular_solve)
    library.impl('triangular_solve.X', triangular_solve)
