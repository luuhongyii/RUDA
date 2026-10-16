import operator
from functools import partial

import torch

from . import _C
from ._indexing import _logical
from ._ops import _dimension, _numeric_dtypes, _integral_dtypes
from ._sparse_compressed import _compressed_output


def reduce(self, dim, keepdim=False, *, dtype=None, out=None, product=False):
    if self.layout != torch.sparse_csr or self.device.type != torch._C._get_privateuse1_backend_name():
        raise RuntimeError('compressed dimensional reduction requires a RUDA CSR tensor')
    if self.ndim != 2:
        raise RuntimeError('compressed dimensional reduction requires a two-dimensional, non-hybrid CSR tensor')
    if not keepdim:
        raise RuntimeError('reduction operations on CSR tensors with keepdim=False are unsupported')
    dimensions = tuple(_dimension(operator.index(axis), 2) for axis in dim)
    if len(set(dimensions)) != len(dimensions):
        raise RuntimeError('duplicate CSR reduction dimension')
    if not dimensions:
        dimensions = (0, 1)
    dtype = self.dtype if dtype is None else dtype
    if dtype not in _numeric_dtypes + _integral_dtypes or dtype == torch.bool:
        raise RuntimeError('CSR dimensional reduction requires a floating, complex64/128 or non-bool integral dtype')
    values = _logical(self.values()).to(dtype)
    result_dtype = torch.int64 if dtype in _integral_dtypes else dtype
    pointers = _logical(self.crow_indices()); columns = _logical(self.col_indices())
    if pointers.dtype not in (torch.int32, torch.int64) or pointers.dtype != columns.dtype:
        raise RuntimeError('CSR reduction requires matching int32/int64 index dtypes')
    if len(dimensions) == 2:
        nnz = min(1, values.numel())
        result_values = torch.empty((nnz,), dtype=result_dtype, device=self.device)
        if nnz:
            _C.scientific_call(529, [values, result_values], [int(product)], [])
        result_columns = torch.zeros((nnz,), dtype=columns.dtype, device=self.device)
        result_pointers = torch.tensor([0, nnz], dtype=pointers.dtype, device=self.device)
        shape = (1, min(1, self.shape[1]))
    else:
        if dimensions[0] == 0:
            sorted_columns, order = torch.sort(columns, stable=True)
            result_columns, counts = torch.unique_consecutive(sorted_columns, return_counts=True)
            ends = counts.cumsum(0, dtype=torch.int64)
            starts = ends - counts
            nnz = result_columns.numel()
            result_pointers = torch.tensor([0, nnz], dtype=pointers.dtype, device=self.device)
            shape = (1, self.shape[1])
        else:
            occupied = pointers[1:] != pointers[:-1]
            result_pointers = torch.cat((torch.zeros((1,), dtype=pointers.dtype, device=self.device),
                                         occupied.cumsum(0, dtype=pointers.dtype)))
            rows = occupied.nonzero().reshape(-1)
            starts = pointers.index_select(0, rows).to(torch.int64)
            ends = pointers.index_select(0, rows + 1).to(torch.int64)
            nnz = rows.numel()
            order = torch.empty((0,), dtype=torch.int64, device=self.device)
            result_columns = torch.zeros((nnz,), dtype=columns.dtype, device=self.device)
            shape = (self.shape[0], 1)
        result_values = torch.empty((nnz,), dtype=result_dtype, device=self.device)
        if nnz:
            _C.scientific_call(528, [starts, ends, order, values, result_values], [int(product)], [])
    result = torch.ops.aten._sparse_csr_tensor_unsafe.default(
        result_pointers, result_columns, result_values, list(shape), dtype=result_dtype,
        layout=torch.sparse_csr, device=self.device)
    if out is None:
        return result
    if out.dtype != result_dtype:
        raise RuntimeError('CSR reduction output dtype must equal the functional result dtype')
    return _compressed_output(result, out)


def register(library):
    for name, product in (('_sparse_csr_sum', False), ('_sparse_csr_prod', True)):
        implementation = partial(reduce, product=product)
        library.impl(name + '.dim_dtype', implementation)
        library.impl(name + '.dim_dtype_out', implementation)
