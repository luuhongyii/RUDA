import torch

from . import _C
from ._indexing import _logical
from ._sparse_algebra import _dense_output
from ._sparse_coo import _input, _make, _sparse_output
from .math_backend import _nvidia


_dtypes = (torch.float32, torch.float64, torch.complex64, torch.complex128)


def _inputs(self, mat2, rank, out, layout):
    _input(self, sparse=True)
    _input(mat2, sparse=False)
    if self.sparse_dim() != rank or self.dense_dim() or mat2.ndim != rank:
        raise RuntimeError(f'COO multiplication requires {rank}D scalar-valued sparse and dense operands')
    if self.device != mat2.device or self.dtype != mat2.dtype:
        raise RuntimeError('COO multiplication requires matching dtypes and devices')
    if self.shape[-1] != mat2.shape[-2] or rank == 3 and self.shape[0] != mat2.shape[0]:
        raise RuntimeError('COO multiplication inner dimensions and batch sizes must match')
    if out is not None and (out.layout != layout or out.dtype != self.dtype or out.device != self.device):
        raise RuntimeError('COO multiplication output layout, dtype or device mismatch')
    if _nvidia():
        raise NotImplementedError('explicit NVIDIA sparse multiplication requires a cuSPARSE route; native RUDA is not an implicit fallback')


def bmm(self, mat2, *, out=None):
    _inputs(self, mat2, 3, out, torch.strided)
    shape = (self.shape[0], self.shape[1], mat2.shape[2])
    if not self._nnz() or not self.shape[2] or not mat2.shape[2]:
        result = torch.zeros(shape, dtype=self.dtype, device=self.device)
    else:
        if self.dtype not in _dtypes:
            raise RuntimeError('nonempty COO bmm requires float32/float64 or complex64/complex128')
        source = self.coalesce()
        result = torch.empty(shape, dtype=self.dtype, device=self.device)
        if result.numel():
            _C.scientific_call(540, [_logical(source._indices()), _logical(source._values()),
                                   _logical(mat2), result], [], [])
    return _dense_output(result, out, (self, mat2))


def hspmm(mat1, mat2, *, out=None):
    _inputs(mat1, mat2, 2, out, torch.sparse_coo)
    source = mat1.coalesce()
    nnz = source._nnz()
    if nnz and mat1.dtype not in _dtypes:
        raise RuntimeError('nonempty COO hspmm requires float32/float64 or complex64/complex128')
    indices = source._indices()[0:1].clone(memory_format=torch.contiguous_format)
    values = torch.empty((mat2.shape[1], nnz), dtype=mat1.dtype, device=mat1.device).transpose(0, 1)
    if values.numel():
        _C.scientific_call(541, [_logical(source._indices()), _logical(source._values()),
                               _logical(mat2), values], [], [])
    result = _make(indices, values, (mat1.shape[0], mat2.shape[1]), nnz < 2)
    return result if out is None else _sparse_output(result, out)


def register(library):
    library.impl('bmm', bmm)
    library.impl('bmm.out', bmm)
    library.impl('hspmm', hspmm)
    library.impl('hspmm.out', hspmm)
