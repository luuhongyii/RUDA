import torch

from . import _C
from ._indexing import _logical
from ._sparse_compressed import _compressed_output
from .math_backend import _nvidia


def sampled_addmm(self, mat1, mat2, *, beta=1, alpha=1, out=None):
    if self.layout != torch.sparse_csr or self.dense_dim() != 0:
        raise RuntimeError('sampled_addmm requires a non-hybrid CSR pattern')
    if self.device.type != torch._C._get_privateuse1_backend_name():
        raise RuntimeError('sampled_addmm requires RUDA storage')
    for value in (mat1, mat2):
        if value.layout != torch.strided or value.ndim < 2:
            raise RuntimeError('sampled_addmm multiplicands must be strided matrices with optional batch dimensions')
        if value.dtype != self.dtype or value.device != self.device:
            raise RuntimeError('sampled_addmm operands require matching dtypes and devices')
    batch_shape = tuple(mat1.shape[:-2])
    if tuple(mat2.shape[:-2]) != batch_shape or self.ndim > 2 and tuple(self.shape[:-2]) != batch_shape:
        raise RuntimeError('sampled_addmm requires equal matrix batch shapes and an equal-batch or unbatched pattern')
    if (mat1.shape[-1] != mat2.shape[-2] or self.shape[-2] != mat1.shape[-2]
            or self.shape[-1] != mat2.shape[-1]):
        raise RuntimeError('sampled_addmm matrix dimensions do not match the pattern')
    if out is not None:
        if out.layout != torch.sparse_csr or out.dense_dim() != 0 or out.dtype != self.dtype or out.device != self.device:
            raise RuntimeError('sampled_addmm output must be a non-hybrid CSR tensor with matching dtype and device')
    if _nvidia():
        raise NotImplementedError('explicit NVIDIA sampled matmul requires a cuSPARSE route; native RUDA is not an implicit fallback')
    empty_product = mat1.numel() == 0 or mat2.numel() == 0
    if not empty_product and self.dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
        raise RuntimeError('nonempty sampled_addmm requires float32/float64/complex64/complex128')
    index_dtype = torch.int64 if out is None else out.crow_indices().dtype
    if index_dtype not in (torch.int32, torch.int64):
        raise RuntimeError('sampled_addmm output indices must use int32 or int64')
    nnz = self._nnz()
    pointers = _logical(self.crow_indices()).to(index_dtype).expand((*batch_shape, self.shape[-2] + 1)).clone()
    indices = _logical(self.col_indices()).to(index_dtype).expand((*batch_shape, nnz)).clone()
    bias = _logical(self.values()).expand((*batch_shape, nnz))
    if empty_product:
        values = bias.clone()
        values.mul_(beta)
    else:
        values = torch.empty(bias.shape, dtype=self.dtype, device=self.device)
        if values.numel():
            coefficients, _ = _C.blas_accumulate_scalars(alpha, beta, self.dtype, False)
            _C.scientific_call(522, [pointers, indices, _logical(mat1), _logical(mat2), bias, values], [], coefficients)
    result = torch.ops.aten._sparse_compressed_tensor_unsafe.default(
        pointers, indices, values, (*batch_shape, *self.shape[-2:]),
        dtype=self.dtype, layout=torch.sparse_csr, device=self.device)
    return result if out is None else _compressed_output(result, out)


def register(library):
    library.impl('sparse_sampled_addmm', sampled_addmm)
    library.impl('sparse_sampled_addmm.out', sampled_addmm)
