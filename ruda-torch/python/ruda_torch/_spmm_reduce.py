import torch

from . import _C
from ._indexing import _logical
from ._sparse_compressed import coo_to_csr
from .math_backend import _nvidia

_modes = {'sum': 0, 'mean': 1, 'amax': 2, 'max': 2, 'amin': 3, 'min': 3}
_dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _inputs(self, other, reduce):
    if self.layout != torch.sparse_csr or self.ndim != 2 or self.dense_dim() != 0:
        raise RuntimeError('sparse_mm reduce requires a two-dimensional non-hybrid CSR matrix')
    if self.device.type != torch._C._get_privateuse1_backend_name():
        raise RuntimeError('sparse_mm reduce requires RUDA storage')
    if other.layout != torch.strided or other.ndim != 2 or self.shape[1] != other.shape[0]:
        raise RuntimeError('sparse_mm reduce requires a strided matrix with matching inner dimension')
    if self.device != other.device or self.dtype != other.dtype:
        raise RuntimeError('sparse_mm reduce operands must share dtype and device')
    if reduce == 'prod':
        raise RuntimeError('sparse_mm reduce type prod has not been enabled')
    if reduce not in _modes:
        raise RuntimeError('sparse_mm reduce must be sum, mean, amax or amin')
    pointers, columns = _logical(self.crow_indices()), _logical(self.col_indices())
    if pointers.dtype not in (torch.int32, torch.int64) or columns.dtype != pointers.dtype:
        raise RuntimeError('sparse_mm reduce requires matching int32/int64 indices')
    if self._nnz() and self.dtype not in _dtypes:
        raise RuntimeError('nonempty sparse_mm reduce requires float16/bfloat16/float32/float64 values')
    if _nvidia():
        raise NotImplementedError('explicit NVIDIA sparse_mm reduce has no vendor route; native RUDA is not an implicit fallback')
    return pointers, columns, _modes[reduce]


def forward(self, other, reduce):
    pointers, columns, mode = _inputs(self, other, reduce)
    shape = (self.shape[0], other.shape[1]); nnz = self._nnz()
    output = torch.zeros(shape, dtype=self.dtype, device=self.device)
    need_arg = nnz > 0 and mode >= 2 and torch.is_grad_enabled() and (self.requires_grad or other.requires_grad)
    arguments = torch.empty(shape if need_arg else (0,), dtype=columns.dtype, device=self.device)
    if nnz and output.numel():
        _C.scientific_call(533, [pointers, columns, _logical(self.values()), _logical(other), output, arguments], [mode], [])
    return output, arguments


def backward(self, grad_out, weight, reduce, arg_out, output_mask):
    pointers, columns, mode = _inputs(self, weight, reduce)
    if (grad_out.layout != torch.strided or grad_out.shape != (self.shape[0], weight.shape[1])
            or grad_out.dtype != self.dtype or grad_out.device != self.device):
        raise RuntimeError('sparse_mm reduce gradient must match the dense output shape, dtype and device')
    if len(output_mask) != 2:
        raise RuntimeError('sparse_mm reduce backward requires two output-mask flags')
    nnz = self._nnz()
    if mode >= 2 and nnz and any(output_mask):
        if (arg_out.layout != torch.strided or arg_out.shape != grad_out.shape
                or arg_out.dtype != columns.dtype or arg_out.device != self.device):
            raise RuntimeError('sparse_mm extrema backward requires saved argument indices matching the forward output')
        arguments = _logical(arg_out)
    else:
        arguments = torch.empty((0,), dtype=columns.dtype, device=self.device)
    grad_self, grad_other = None, None
    gradient = _logical(grad_out)
    if output_mask[0]:
        result_values = torch.zeros((nnz,), dtype=self.dtype, device=self.device)
        if nnz:
            _C.scientific_call(534, [pointers, columns, _logical(weight), gradient, arguments, result_values], [mode], [])
        grad_self = torch.ops.aten._sparse_csr_tensor_unsafe.default(
            pointers.clone(), columns.clone(), result_values, list(self.shape),
            dtype=self.dtype, layout=torch.sparse_csr, device=self.device)
    if output_mask[1]:
        grad_other = torch.zeros(weight.shape, dtype=self.dtype, device=self.device)
        if nnz and grad_other.numel():
            sorted_columns, order = torch.sort(columns, stable=True)
            column_pointers = coo_to_csr(sorted_columns, self.shape[1], out_int32=columns.dtype == torch.int32)
            _C.scientific_call(535, [pointers, _logical(self.values()), gradient, arguments,
                                      column_pointers, order, grad_other], [mode], [])
    return grad_self, grad_other


def register(library):
    library.impl('_sparse_mm_reduce_impl', forward)
    library.impl('_sparse_mm_reduce_impl_backward', backward)
