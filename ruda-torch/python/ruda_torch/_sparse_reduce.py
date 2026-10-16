import torch

from . import _C
from ._indexing import _logical
from ._ops import _dimension
from ._sparse_coo import _input, _make, _sparse_output


def sum_backward(grad, self, dim, *, out=None):
    _input(self, sparse=True)
    _input(grad, sparse=grad.layout == torch.sparse_coo)
    if self.device != grad.device:
        raise RuntimeError('sparse sum gradient and input must use the same device')
    if not grad.numel() or grad.layout == torch.sparse_coo and not grad._nnz():
        indices = torch.empty((self.sparse_dim(), 0), dtype=torch.int64, device=self.device)
        values = torch.empty((0, *self.shape[self.sparse_dim():]), dtype=self.dtype, device=self.device)
        result = _make(indices, values, self.shape, True)
        return result if out is None else _sparse_output(result, out)
    value = self.coalesce()
    if value.ndim > 64:
        raise RuntimeError('sparse sum supports at most64 dimensions')
    dimensions = tuple(_dimension(axis, value.ndim) for axis in dim)
    if len(set(dimensions)) != len(dimensions):
        raise RuntimeError('duplicate sparse sum reduction dimension')
    sd = value.sparse_dim()
    kept = tuple(axis for axis in range(sd) if axis not in dimensions)
    dense_reduced = tuple(axis + 1 - sd for axis in range(sd, value.ndim) if axis in dimensions)
    shape = value._values().shape
    indices = _logical(value._indices()).clone(memory_format=torch.contiguous_format)
    if not kept:
        if grad.layout != torch.strided:
            raise RuntimeError('sparse sum backward requires a dense gradient when all sparse dimensions were reduced')
        values = _logical(grad)
        for axis in dense_reduced:
            values = values.unsqueeze(axis - 1)
        values = values.expand(shape).clone(memory_format=torch.contiguous_format)
    else:
        if grad.layout != torch.sparse_coo:
            raise RuntimeError('sparse sum backward requires a COO gradient when sparse dimensions remain')
        gradient = grad.coalesce()
        values = _logical(gradient._values())
        for axis in dense_reduced:
            values = values.unsqueeze(axis)
        if len(kept) == sd:
            if dense_reduced:
                values = values.expand(shape).clone(memory_format=torch.contiguous_format)
        else:
            if gradient.sparse_dim() != len(kept):
                raise RuntimeError('sparse sum gradient has the wrong number of sparse dimensions')
            values = values.expand((gradient._nnz(), *shape[1:]))
            result_values = torch.empty(shape, dtype=grad.dtype, device=self.device)
            if result_values.numel():
                _C.scientific_call(515, [_logical(value._indices()), _logical(gradient._indices()), values, result_values], list(kept), [])
            values = result_values
    result = _make(indices, values, self.shape, value._nnz() < 2)
    return result if out is None else _sparse_output(result, out)


def register(library):
    library.impl('_sparse_sum_backward', sum_backward)
    library.impl('_sparse_sum_backward.out', sum_backward)
