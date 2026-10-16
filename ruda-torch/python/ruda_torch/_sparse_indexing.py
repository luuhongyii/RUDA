import torch

from . import _C
from ._indexing import _logical
from ._ops import _dimension
from ._sparse_coo import _make


def index_select(self, dim, index):
    if self.layout != torch.sparse_coo or self.device.type != torch._C._get_privateuse1_backend_name():
        raise RuntimeError('sparse index_select requires a COO tensor on RUDA')
    if not self.ndim:
        raise IndexError('index_select cannot be applied to a 0-dim tensor')
    if index.layout != torch.strided or index.ndim != 1 or index.dtype != torch.int64:
        raise IndexError('sparse index_select index must be a 1-D strided int64 tensor')
    if index.device != self.device:
        raise RuntimeError('sparse index_select requires indices on the same RUDA device')
    axis = _dimension(dim, self.ndim)
    sd = self.sparse_dim()
    indices, values, index = self._indices(), _logical(self._values()), _logical(index)
    shape = list(self.shape)
    shape[axis] = index.numel()
    if axis >= sd:
        result_values = values.index_select(axis - sd + 1, index)
        return _make(indices, result_values, shape, self._nnz() < 2)
    if not index.numel():
        result_indices = indices.index_select(1, index)
        result_values = values.index_select(0, index)
        return _make(result_indices, result_values, shape, True)
    identity = axis == 0 and self.is_coalesced()
    coordinates = _logical(indices[axis]).contiguous()
    if identity:
        order = torch.empty((0,), dtype=torch.int64, device=self.device)
    else:
        coordinates, order = coordinates.sort()
    first = torch.empty_like(index, memory_format=torch.contiguous_format)
    counts = torch.empty((index.numel() + 1,), dtype=torch.int64, device=self.device)
    _C.scientific_call(538, [coordinates, index, first, counts], [self.shape[axis]], [])
    offsets = counts.cumsum(0, dtype=torch.int64)
    del counts
    nnz = int(offsets[-1].item())
    if not 0 <= nnz <= (2**32 - 1) // sd:
        raise RuntimeError('sparse index_select coordinates exceed native U32 indexing')
    result_indices = torch.empty((sd, nnz), dtype=torch.int64, device=self.device)
    selected = torch.empty((nnz,), dtype=torch.int64, device=self.device)
    if nnz:
        _C.scientific_call(539, [_logical(indices), first, offsets, order, result_indices, selected],
                           [axis, int(identity)], [])
    result_values = values.index_select(0, selected)
    return _make(result_indices, result_values, shape, nnz < 2)


def register(library):
    library.impl('index_select', index_select)
