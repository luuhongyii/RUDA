import torch

from . import _C
from ._blas_accumulate import _inputs, _matmul_output
from ._indexing import _logical
from ._ordering import _output
from .math_backend import _nvidia


def int_mm(self, mat2, *, out=None):
    if self.ndim != 2 or mat2.ndim != 2 or self.shape[1] != mat2.shape[0]:
        raise RuntimeError('_int_mm requires compatible two-dimensional matrices')
    if (self.device.type != torch._C._get_privateuse1_backend_name() or self.device != mat2.device
            or self.layout != torch.strided or mat2.layout != torch.strided):
        raise RuntimeError('_int_mm requires strided tensors on the same RUDA device')
    if self.dtype != torch.int8 or mat2.dtype != torch.int8:
        raise RuntimeError('_int_mm requires int8 inputs')
    m, k, n = self.shape[0], self.shape[1], mat2.shape[1]
    if m <= 16 or k <= 0 or k % 8 or n <= 0 or n % 8:
        raise RuntimeError('_int_mm requires m > 16 and positive k/n divisible by 8')
    _matmul_output(out, self, mat2, (m, n), torch.int32, fixed=True)
    if out is not None and not out.is_contiguous():
        raise RuntimeError('_int_mm requires a contiguous output')
    a, b = _logical(self), _logical(mat2)
    if _nvidia():
        from ._vendor_blas import int_mm as vendor_int_mm
        result = vendor_int_mm(a, b)
    else:
        result = torch.empty((m, n), dtype=torch.int32, device=self.device)
        _C.scientific_call(498, [a, b, result], [], [])
    return _output(result, out, (self, mat2))


def addmm_activation(self, mat1, mat2, *, beta=1, alpha=1, use_gelu=False, out=None):
    _inputs(self, mat1, mat2, mat1.dtype)
    if mat1.ndim != 2 or mat2.ndim != 2 or mat1.shape[1] != mat2.shape[0]:
        raise RuntimeError('_addmm_activation requires compatible two-dimensional matrices')
    shape = (mat1.shape[0], mat2.shape[1])
    if torch.broadcast_shapes(self.shape, shape) != shape:
        raise RuntimeError('_addmm_activation bias is not broadcastable to the result')
    _matmul_output(out, mat1, mat2, shape, mat1.dtype)
    if out is not None:
        _C.check_inplace(out, self)
        if out._cdata == self._cdata and tuple(out.shape) != shape:
            raise RuntimeError('_addmm_activation cannot resize an aliased bias')
    if shape[0] == 0 or shape[1] == 0:
        result = torch.empty(shape, dtype=mat1.dtype, device=mat1.device)
        return _output(result, out, (self, mat1, mat2))
    empty_inner = mat1.shape[1] == 0
    if mat1.is_complex() and not empty_inner:
        raise RuntimeError('_addmm_activation requires a real floating dtype for nonempty products')
    coefficients, zero_beta = _C.blas_accumulate_scalars(1 if empty_inner else alpha, beta, mat1.dtype, empty_inner)
    source = _logical(self).expand(shape)
    a, b = _logical(mat1), _logical(mat2)
    if _nvidia():
        from ._vendor_blas import gemm
        ar, ai, br, bi = coefficients
        alpha_value = complex(ar, ai) if a.is_complex() else ar
        beta_value = complex(br, bi) if a.is_complex() else br
        if empty_inner:
            result = torch.zeros(shape, dtype=a.dtype, device=a.device) if zero_beta else source * beta_value
        else:
            result = gemm(a, b, source, alpha=alpha_value, beta=beta_value)
            result = torch.nn.functional.gelu(result, approximate='tanh') if use_gelu else result.relu()
    else:
        result = torch.empty(shape, dtype=mat1.dtype, device=mat1.device)
        activation = 0 if empty_inner else 2 if use_gelu else 1
        _C.scientific_call(496, [source.unsqueeze(0), a.unsqueeze(0), b.unsqueeze(0), result.unsqueeze(0)],
                           [1, int(zero_beta), activation], coefficients)
    return _output(result, out, (self, mat1, mat2))


def register(library):
    library.impl('_int_mm', int_mm)
    library.impl('_int_mm.out', int_mm)
    library.impl('_addmm_activation', addmm_activation)
    library.impl('_addmm_activation.out', addmm_activation)
