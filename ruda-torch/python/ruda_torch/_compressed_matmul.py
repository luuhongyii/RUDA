from functools import partial

import torch

from . import _C
from ._indexing import _logical
from ._ordering import _output
from ._sparse_compressed import _layouts
from .math_backend import _nvidia

_dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.complex64, torch.complex128)


def _input(value, *, dense=False):
    if value.device.type != torch._C._get_privateuse1_backend_name() or value.dtype not in _dtypes:
        raise RuntimeError('compressed matrix operations require supported floating or complex RUDA tensors')
    if value.layout != torch.strided and (dense or value.layout not in _layouts):
        raise RuntimeError('compressed matrix operations require strided or compressed sparse storage')


def _matrix(value, rank):
    _input(value)
    if value.ndim != rank or value.layout in _layouts and value.dense_dim() != 0:
        raise RuntimeError(f'compressed matrix multiplication requires rank-{rank} operands without dense sparse payload dimensions')


def _output_metadata(self, out, shape, inplace):
    if inplace:
        out = self
    if out is not None:
        if out.layout != torch.strided or out.dtype != self.dtype or out.device != self.device:
            raise RuntimeError('compressed matrix output must be strided with matching dtype and device')
        if out._cdata == self._cdata and out.shape != shape:
            raise RuntimeError('in-place compressed matrix accumulation cannot broadcast or resize its bias')
    return out


def _finish(result, out, inputs):
    if out is not None:
        for value in inputs:
            if value.layout in _layouts:
                row = value.layout in (torch.sparse_csr, torch.sparse_bsr)
                parts = (value.crow_indices(), value.col_indices(), value.values()) if row else (
                    value.ccol_indices(), value.row_indices(), value.values())
                for part in parts:
                    _C.check_inplace(out, part)
            else:
                _C.check_inplace(out, value)
    return _output(result, out, (), cast=False)


def _scaled_bias(bias, beta):
    if beta == 0:
        return torch.zeros(bias.shape, dtype=bias.dtype, device=bias.device)
    result = bias.clone(memory_format=torch.contiguous_format)
    if beta != 1:
        result.mul_(beta)
    return result


def _native(sparse, dense, bias, out, coefficients, batched):
    if sparse.layout == torch.sparse_csc:
        sparse = sparse.to_sparse_csr()
    elif sparse.layout == torch.sparse_bsc:
        axis = sparse.ndim - 1
        sparse = sparse.to_sparse_bsr(tuple(sparse.values().shape[axis:axis + 2]))
    pointers, indices, values = sparse.crow_indices(), sparse.col_indices(), _logical(sparse.values())
    if sparse.layout == torch.sparse_csr:
        values = values.unsqueeze(-1).unsqueeze(-1)
    dense, bias = _logical(dense), _logical(bias)
    if not batched:
        pointers, indices, values, dense, bias, out = (
            value.unsqueeze(0) for value in (pointers, indices, values, dense, bias, out))
    _C.scientific_call(521, [_logical(pointers), _logical(indices), values, dense, bias, out], [], coefficients)


def _multiply(self, mat1, mat2, *, beta, alpha, batched, out=None, inplace=False):
    _input(self, dense=True)
    rank = 3 if batched else 2
    _matrix(mat1, rank)
    _matrix(mat2, rank)
    if mat1.device != self.device or mat2.device != self.device or mat1.dtype != self.dtype or mat2.dtype != self.dtype:
        raise RuntimeError('compressed matrix operands and bias must have matching dtypes and devices')
    left_sparse, right_sparse = mat1.layout in _layouts, mat2.layout in _layouts
    if left_sparse == right_sparse:
        raise NotImplementedError('this compressed matrix route requires one sparse and one strided multiplicand')
    if mat1.shape[-1] != mat2.shape[-2] or batched and mat1.shape[0] != mat2.shape[0]:
        raise RuntimeError('compressed matrix inner or batch dimensions do not match')
    shape = ((mat1.shape[0],) if batched else ()) + (mat1.shape[-2], mat2.shape[-1])
    if torch.broadcast_shapes(self.shape, shape) != shape:
        raise RuntimeError('compressed matrix bias cannot broadcast to the result')
    out = _output_metadata(self, out, shape, inplace)
    if _nvidia():
        raise NotImplementedError('explicit NVIDIA sparse matmul requires a cuSPARSE route; native RUDA is not an implicit fallback')
    bias = _logical(self).expand(shape)
    sparse = mat1 if left_sparse else mat2
    if any(size == 0 for size in shape):
        result = torch.empty(shape, dtype=self.dtype, device=self.device)
    elif sparse._nnz() == 0:
        result = _scaled_bias(bias, beta)
    else:
        coefficients, _ = _C.blas_accumulate_scalars(alpha, beta, self.dtype, False)
        result = torch.empty(shape, dtype=self.dtype, device=self.device)
        if left_sparse:
            _native(mat1, mat2, bias, result, coefficients, batched)
        else:
            _native(mat2.transpose(-2, -1), mat1.transpose(-2, -1), bias.transpose(-2, -1),
                    result.transpose(-2, -1), coefficients, batched)
    return _finish(result, out, (self, mat1, mat2))


def addmm(self, mat1, mat2, *, beta=1, alpha=1, out=None, inplace=False):
    if mat1.layout in _layouts and mat2.layout in _layouts:
        from ._compressed_spgemm import multiply
        return multiply(mat1, mat2, bias=self, beta=beta, alpha=alpha, out=out, inplace=inplace)
    return _multiply(self, mat1, mat2, beta=beta, alpha=alpha, batched=False, out=out, inplace=inplace)


def baddbmm(self, batch1, batch2, *, beta=1, alpha=1, out=None, inplace=False):
    return _multiply(self, batch1, batch2, beta=beta, alpha=alpha, batched=True, out=out, inplace=inplace)


def mm(self, mat2, *, out=None):
    if self.layout in _layouts and mat2.layout in _layouts:
        from ._compressed_spgemm import multiply
        return multiply(self, mat2, out=out)
    bias = torch.zeros((), dtype=self.dtype, device=self.device)
    return addmm(bias, self, mat2, beta=0, out=out)


def bmm(self, mat2, *, out=None):
    bias = torch.zeros((), dtype=self.dtype, device=self.device)
    return baddbmm(bias, self, mat2, beta=0, out=out)


def addmv(self, mat, vec, *, beta=1, alpha=1, out=None, inplace=False):
    _input(self, dense=True)
    _matrix(mat, 2)
    _input(vec, dense=True)
    if vec.ndim != 1 or mat.layout not in _layouts:
        raise RuntimeError('compressed addmv requires a compressed matrix and a strided vector')
    shape = (mat.shape[0],)
    if torch.broadcast_shapes(self.shape, shape) != shape:
        raise RuntimeError('compressed addmv bias cannot broadcast to the result')
    out = _output_metadata(self, out, shape, inplace)
    result = addmm(self.expand(shape).unsqueeze(-1), mat, vec.unsqueeze(-1), beta=beta, alpha=alpha).squeeze(-1)
    return _finish(result, out, (self, mat, vec))


def mv(self, vec, *, out=None):
    bias = torch.zeros((), dtype=self.dtype, device=self.device)
    return addmv(bias, self, vec, beta=0, out=out)


def register(library):
    for name, function in (('mm', mm), ('bmm', bmm), ('mv', mv),
                           ('addmm', addmm), ('baddbmm', baddbmm), ('addmv', addmv)):
        library.impl(name, function)
        library.impl(name + '.out', function)
    for name, function in (('addmm_', addmm), ('baddbmm_', baddbmm), ('addmv_', addmv)):
        library.impl(name, partial(function, inplace=True))
