from functools import partial

import torch

from . import _C
from ._indexing import _logical
from ._ordering import _output
from ._sparse_coo import _input, _make, _sorted, _sparse_output
from .math_backend import _nvidia

_matmul_dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.complex64, torch.complex128)


def _same_device(a, b):
    if a.device != b.device:
        raise RuntimeError('sparse operands must share a device')


def _dense_output(result, out, inputs):
    if out is not None:
        if out.layout != torch.strided:
            raise RuntimeError('sparse-dense operation requires a strided output')
        for value in inputs:
            if value.layout == torch.sparse_coo:
                _C.check_inplace(out, value._indices())
                _C.check_inplace(out, value._values())
            else:
                _C.check_inplace(out, value)
    return _output(result, out, (), cast=True)


def add_out(self, other, *, alpha=1, out):
    _input(other, sparse=True)
    _input(self, sparse=self.layout == torch.sparse_coo)
    _same_device(self, other)
    if self.shape != other.shape:
        raise RuntimeError('sparse addition requires matching shapes; broadcasting is unsupported')
    dtype = torch.result_type(self, other)
    if out.device != self.device or out.layout != self.layout or not torch.can_cast(dtype, out.dtype):
        raise RuntimeError('sparse addition output has incompatible dtype, device or layout')
    if self.layout == torch.strided:
        if not other._nnz():
            return _dense_output(self.clone(), out, (self, other))
        result = _logical(self).to(dtype).clone()
        values = _logical(other._values()).to(dtype)
        if values.numel():
            indices = _logical(other._indices())
            order, _, _ = _sorted(indices)
            coefficient = torch.scalar_tensor(alpha, dtype=dtype, device=self.device)
            _C.scientific_call(508, [indices, values, order, result, coefficient], [], [])
        return _dense_output(result, out, (self, other))
    if not other._nnz():
        return _sparse_output(self, out)
    if not self._nnz():
        return scalar_mul(other, alpha, out=out)
    if self.sparse_dim() != other.sparse_dim():
        raise RuntimeError('sparse addition requires matching sparse and dense dimensions')
    if dtype in (torch.bool, torch.complex32):
        raise RuntimeError('nonempty COO+COO addition does not support bool or ComplexHalf')
    left = _logical(self._values()).to(dtype)
    right = _logical(other._values()).to(dtype)
    if not _C.sparse_alpha_is_one(alpha, dtype):
        right = right * alpha
    indices = torch.cat((self._indices(), other._indices()), dim=1)
    values = torch.cat((left, right), dim=0)
    result = _make(indices, values, self.shape, False)
    if out.dtype != dtype:
        result = result.coalesce()
        result = _make(result._indices(), result._values().to(out.dtype), self.shape, result._nnz() < 2)
    if result._nnz() > result.numel():
        result = result.coalesce()
        result = _make(result._indices(), result._values(), self.shape, result._nnz() < 2)
    return _sparse_output(result, out)


def scalar_mul(self, other, *, out=None, inplace=False):
    if isinstance(other, torch.Tensor) and self.layout != torch.sparse_coo and other.layout == torch.sparse_coo:
        if inplace:
            raise NotImplementedError('in-place dense multiplied by sparse is not implemented')
        self, other = other, self
    _input(self, sparse=True)
    if isinstance(other, torch.Tensor):
        if other.layout != torch.strided or other.ndim != 0:
            raise RuntimeError('scalar multiplication requires a dense zero-dimensional multiplier')
        if other.device != self.device and other.device.type != 'cpu':
            raise RuntimeError('sparse scalar multiplication requires the same device or a CPU scalar')
    dtype = torch.result_type(self, other)
    if inplace or out is not None and out._cdata == self._cdata:
        if not torch.can_cast(dtype, self.dtype):
            raise RuntimeError('in-place sparse multiplication cannot cast the result dtype')
        self._values().mul_(other)
        return self
    values = torch.mul(_logical(self._values()).to(dtype), other)
    result = _make(self._indices().clone(), values, self.shape, self.is_coalesced())
    return result if out is None else _sparse_output(result, out)


def _matrix(a, *, sparse):
    _input(a, sparse=sparse)
    if a.ndim != 2 or sparse and (a.sparse_dim() != 2 or a.dense_dim() != 0):
        raise RuntimeError('sparse matrix multiplication requires matrices with scalar COO values')
    if a.dtype not in _matmul_dtypes:
        raise RuntimeError('sparse matrix multiplication requires floating or complex inputs')


def addmm(self, mat1, mat2, *, beta=1, alpha=1, out=None, inplace=False):
    _input(self, sparse=False)
    _matrix(mat1, sparse=True)
    _matrix(mat2, sparse=False)
    _same_device(self, mat1)
    _same_device(mat1, mat2)
    if self.dtype != mat1.dtype or mat1.dtype != mat2.dtype:
        raise RuntimeError('sparse addmm requires matching dtypes')
    if mat1.shape[1] != mat2.shape[0]:
        raise RuntimeError('sparse addmm inner dimensions do not match')
    shape = (mat1.shape[0], mat2.shape[1])
    if torch.broadcast_shapes(self.shape, shape) != shape:
        raise RuntimeError('sparse addmm bias cannot broadcast to the result')
    if inplace:
        out = self
        if self.shape != shape:
            raise RuntimeError('in-place sparse addmm does not broadcast the bias')
    if out is not None:
        if out.dtype != self.dtype or out.device != self.device or out.layout != torch.strided:
            raise RuntimeError('sparse addmm output must match the dtype/device and be strided')
        if out._cdata == self._cdata and out.shape != shape:
            raise RuntimeError('sparse addmm cannot resize an aliased bias')
    if _nvidia():
        raise NotImplementedError('explicit NVIDIA sparse matmul requires a cuSPARSE route; native RUDA is not an implicit fallback')
    sparse = mat1.coalesce()
    bias = _logical(self).expand(shape)
    if sparse._nnz() == 0:
        result = bias * torch.scalar_tensor(beta, dtype=self.dtype, device=self.device)
    else:
        coefficients, _ = _C.blas_accumulate_scalars(alpha, beta, self.dtype, False)
        result = torch.empty(shape, dtype=self.dtype, device=self.device)
        _C.scientific_call(509, [_logical(sparse._indices()), _logical(sparse._values()),
                               _logical(mat2), bias, result], [], coefficients)
    return _dense_output(result, out, (self, mat1, mat2))


def sparse_sparse_mm(self, other, *, out=None):
    _matrix(self, sparse=True)
    _matrix(other, sparse=True)
    _same_device(self, other)
    if self.dtype != other.dtype or self.shape[1] != other.shape[0]:
        raise RuntimeError('sparse matrix multiplication requires matching dtypes and inner dimensions')
    if _nvidia():
        raise NotImplementedError('explicit NVIDIA sparse matmul requires a cuSPARSE route; native RUDA is not an implicit fallback')
    left, right = self.coalesce(), other.coalesce()
    shape = (self.shape[0], other.shape[1])
    count = 0
    if left._nnz() and right._nnz():
        starts = torch.empty((left._nnz(),), dtype=torch.int32, device=self.device)
        prefix = torch.empty((left._nnz(),), dtype=torch.int64, device=self.device)
        total = torch.empty((1,), dtype=torch.int64, device=self.device)
        _C.scientific_call(510, [_logical(left._indices()), _logical(right._indices()), starts, prefix, total], [], [])
        count = int(total.item())
    if not 0 <= count <= (2**32 - 1) // 2:
        raise RuntimeError('sparse matmul product coordinates exceed native U32 indexing')
    indices = torch.empty((2, count), dtype=torch.int64, device=self.device)
    math_dtype = torch.float32 if self.dtype in (torch.float16, torch.bfloat16) else self.dtype
    values = torch.empty((count,), dtype=math_dtype, device=self.device)
    if count:
        _C.scientific_call(511, [_logical(left._indices()), _logical(left._values()), _logical(right._indices()),
                               _logical(right._values()), starts, prefix, indices, values], [], [])
    result = _make(indices, values, shape, count < 2).coalesce()
    if result.dtype != self.dtype:
        result = _make(result._indices(), result._values().to(self.dtype), shape, True)
    return result if out is None else _sparse_output(result, out)


def mm(self, mat2, *, out=None):
    if out is not None and out.dtype != self.dtype:
        raise RuntimeError('mm output must match the operand dtype')
    if self.layout == torch.sparse_coo and mat2.layout == torch.sparse_coo:
        return sparse_sparse_mm(self, mat2, out=out)
    if self.layout == torch.sparse_coo:
        _matrix(self, sparse=True)
        _matrix(mat2, sparse=False)
        bias = torch.zeros((), dtype=self.dtype, device=self.device)
        return addmm(bias, self, mat2, beta=0, out=out)
    _matrix(self, sparse=False)
    _matrix(mat2, sparse=True)
    result = mm(mat2.transpose(0, 1), self.transpose(0, 1)).transpose(0, 1)
    return _dense_output(result, out, (self, mat2))


def register(library):
    from ._sparse_product import mul_out
    library.impl('add.out', add_out)
    library.impl('mul.Scalar', scalar_mul)
    library.impl('mul.out', mul_out)
    library.impl('mul.Scalar_out', scalar_mul)
    library.impl('mul_.Scalar', partial(scalar_mul, inplace=True))
    library.impl('mm', mm)
    library.impl('mm.out', mm)
    library.impl('_sparse_sparse_matmul', sparse_sparse_mm)
    library.impl('_sparse_sparse_matmul.out', sparse_sparse_mm)
    library.impl('addmm', addmm)
    library.impl('addmm.out', addmm)
    library.impl('addmm_', partial(addmm, inplace=True))
    from ._sparse_batched_hybrid import register as register_batched_hybrid
    register_batched_hybrid(library)
