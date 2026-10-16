from functools import partial

import torch

from . import _C
from ._indexing import _logical
from ._ordering import _output
from .math_backend import _nvidia

_dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.complex64, torch.complex128)


def _inputs(bias, a, b, dtype):
    if a.dtype not in _dtypes or a.dtype != b.dtype:
        raise RuntimeError('BLAS accumulation requires matching floating or complex input dtypes')
    if a.device.type != torch._C._get_privateuse1_backend_name():
        raise RuntimeError('BLAS accumulation requires RUDA tensors')
    for value in (bias, a, b):
        if value.device != a.device or value.layout != torch.strided:
            raise RuntimeError('BLAS accumulation requires strided tensors on the same device')
    if bias.dtype not in (a.dtype, dtype):
        raise RuntimeError('BLAS bias dtype must match the input or output dtype')


def _run(bias, a, b, shape, mode, beta, alpha, dtype, out, inplace, *, fixed_out=False):
    if inplace:
        out = bias
        if tuple(out.shape) != tuple(shape):
            raise RuntimeError('in-place BLAS accumulation cannot broadcast or resize the output')
    if out is not None:
        if out.device != a.device or out.dtype != dtype or out.layout != torch.strided:
            raise RuntimeError('BLAS output must have the expected dtype, device and strided layout')
        if fixed_out and tuple(out.shape) != tuple(shape):
            raise RuntimeError('baddbmm.dtype_out requires an already correctly sized output')
        for value in (bias, a, b):
            _C.check_inplace(out, value)
    coefficients, zero_beta = _C.blas_accumulate_scalars(alpha, beta, a.dtype, mode == 0)
    source = _logical(bias).to(dtype).expand(shape)
    left, right = _logical(a), _logical(b)
    if _nvidia():
        from ._vendor_blas import gemm
        ar, ai, br, bi = coefficients
        alpha_value = complex(ar, ai) if a.is_complex() else ar
        beta_value = complex(br, bi) if a.is_complex() else br
        if a.shape[-1] == 0 or mode == 2 and a.shape[0] == 0:
            result = torch.zeros(shape, dtype=dtype, device=a.device) if zero_beta else source * beta_value
        elif mode == 0:
            result = gemm(left, right.unsqueeze(-1), source.unsqueeze(-1), alpha=alpha_value, beta=beta_value).squeeze(-1)
        elif mode == 1:
            result = gemm(left, right, source, alpha=alpha_value, beta=beta_value, out_dtype=dtype)
        else:
            result = source
            for batch in range(a.shape[0]):
                result = gemm(left[batch], right[batch], result, alpha=alpha_value, beta=beta_value if batch == 0 else 1)
    else:
        result = torch.empty(shape, dtype=dtype, device=a.device)
        _C.scientific_call(496, [source, left, right, result], [mode, int(zero_beta)], coefficients)
    return _output(result, out, (bias, a, b))


def addmv(self, mat, vec, *, beta=1, alpha=1, out=None, inplace=False):
    _inputs(self, mat, vec, mat.dtype)
    if mat.ndim != 2 or vec.ndim != 1 or self.ndim > 1:
        raise RuntimeError('addmv requires a scalar/vector bias, a matrix and a vector')
    if mat.shape[1] != vec.shape[0] or self.numel() not in (1, mat.shape[0]):
        raise RuntimeError('addmv input sizes are incompatible')
    return _run(self, mat, vec, (mat.shape[0],), 0, beta, alpha, mat.dtype, out, inplace)


def mv(self, vec, *, out=None):
    if self.ndim != 2 or vec.ndim != 1 or self.shape[1] != vec.shape[0]:
        raise RuntimeError('mv requires a matrix and a compatible vector')
    bias = torch.empty((), dtype=self.dtype, device=self.device)
    return addmv(bias, self, vec, beta=0, alpha=1, out=out)


def batched(self, batch1, batch2, *, beta=1, alpha=1, reduce_batch=False, out_dtype=None, out=None, inplace=False):
    explicit_dtype = out_dtype is not None
    dtype = batch1.dtype if out_dtype is None else out_dtype
    if dtype != batch1.dtype and not (batch1.dtype in (torch.float16, torch.bfloat16) and dtype == torch.float32):
        raise RuntimeError('out_dtype must match the input dtype or be float32 for float16/bfloat16 inputs')
    _inputs(self, batch1, batch2, dtype)
    if batch1.ndim != 3 or batch2.ndim != 3 or batch1.shape[0] != batch2.shape[0] or batch1.shape[2] != batch2.shape[1]:
        raise RuntimeError('batched BLAS inputs must have compatible batch and matrix dimensions')
    shape = (batch1.shape[1], batch2.shape[2])
    if not reduce_batch:
        shape = (batch1.shape[0],) + shape
    if torch.broadcast_shapes(self.shape, shape) != shape:
        raise RuntimeError('BLAS bias is not broadcastable to the result')
    return _run(self, batch1, batch2, shape, 2 if reduce_batch else 1, beta, alpha, dtype, out, inplace,
                fixed_out=explicit_dtype)


def baddbmm_dtype(self, batch1, batch2, out_dtype, *, beta=1, alpha=1, out=None):
    return batched(self, batch1, batch2, out_dtype=out_dtype, beta=beta, alpha=alpha, out=out)


def _matmul_output(out, a, b, shape, dtype, *, fixed=False):
    if out is not None:
        if out.device != a.device or out.dtype != dtype or out.layout != torch.strided:
            raise RuntimeError('matrix multiplication output has incompatible dtype, device or layout')
        if fixed and tuple(out.shape) != tuple(shape):
            raise RuntimeError('matrix multiplication dtype_out requires an already correctly sized output')
        _C.check_inplace(out, a)
        _C.check_inplace(out, b)


def matmul_dtype(self, mat2, out_dtype, *, batch=False, out=None):
    rank = 3 if batch else 2
    if self.ndim != rank or mat2.ndim != rank or self.shape[-1] != mat2.shape[-2]:
        raise RuntimeError('matrix multiplication requires compatible matrix dimensions')
    if batch and self.shape[0] != mat2.shape[0]:
        raise RuntimeError('bmm inputs must have the same batch size')
    shape = self.shape[:-1] + (mat2.shape[-1],)
    _matmul_output(out, self, mat2, shape, out_dtype, fixed=True)
    bias = torch.empty((), dtype=out_dtype, device=self.device)
    left, right = (self, mat2) if batch else (self.unsqueeze(0), mat2.unsqueeze(0))
    result = batched(bias, left, right, beta=0, alpha=1, out_dtype=out_dtype)
    if not batch:
        result = result.squeeze(0)
    return _output(result, out, (self, mat2))


def addmm_dtype(self, mat1, mat2, out_dtype, *, beta=1, alpha=1, out=None):
    if mat1.ndim != 2 or mat2.ndim != 2 or mat1.shape[1] != mat2.shape[0]:
        raise RuntimeError('addmm requires compatible two-dimensional matrix operands')
    shape = (mat1.shape[0], mat2.shape[1])
    if torch.broadcast_shapes(self.shape, shape) != shape:
        raise RuntimeError('addmm bias is not broadcastable to the result')
    _matmul_output(out, mat1, mat2, shape, out_dtype)
    if out is not None:
        _C.check_inplace(out, self)
        if out._cdata == self._cdata and tuple(out.shape) != shape:
            raise RuntimeError('addmm output cannot resize an aliased bias')
    result = batched(self.expand(shape).unsqueeze(0), mat1.unsqueeze(0), mat2.unsqueeze(0),
                     out_dtype=out_dtype, beta=beta, alpha=alpha).squeeze(0)
    return _output(result, out, (self, mat1, mat2))


def matmul_out(self, mat2, *, batch=False, out):
    from ._ops import mm, bmm
    _matmul_output(out, self, mat2, (), self.dtype)
    return _output((bmm if batch else mm)(self, mat2), out, (self, mat2))


def addmm_out(self, mat1, mat2, *, beta=1, alpha=1, out=None, inplace=False):
    from ._ops import addmm
    if inplace:
        out = self
    if mat1.ndim != 2 or mat2.ndim != 2:
        raise RuntimeError('addmm requires matrix operands')
    shape = (mat1.shape[0], mat2.shape[1])
    _matmul_output(out, mat1, mat2, shape, mat1.dtype)
    if out is not None:
        _C.check_inplace(out, self)
        if out._cdata == self._cdata and tuple(out.shape) != shape:
            raise RuntimeError('in-place addmm cannot broadcast or resize the bias')
    return _output(addmm(self, mat1, mat2, beta=beta, alpha=alpha), out, (self, mat1, mat2))


def register(library):
    library.impl('mv.out', mv)
    library.impl('addmv.out', addmv)
    library.impl('addmv_', partial(addmv, inplace=True))
    for name, reduce_batch in (('addbmm', True), ('baddbmm', False)):
        library.impl(name + '.out', partial(batched, reduce_batch=reduce_batch))
        library.impl(name + '_', partial(batched, reduce_batch=reduce_batch, inplace=True))
    library.impl('baddbmm.dtype', baddbmm_dtype)
    library.impl('baddbmm.dtype_out', baddbmm_dtype)
    for name, batch in (('mm', False), ('bmm', True)):
        library.impl(name + '.out', partial(matmul_out, batch=batch))
        library.impl(name + '.dtype', partial(matmul_dtype, batch=batch))
        library.impl(name + '.dtype_out', partial(matmul_dtype, batch=batch))
    library.impl('addmm.out', addmm_out)
    library.impl('addmm_', partial(addmm_out, inplace=True))
    library.impl('addmm.dtype', addmm_dtype)
    library.impl('addmm.dtype_out', addmm_dtype)
