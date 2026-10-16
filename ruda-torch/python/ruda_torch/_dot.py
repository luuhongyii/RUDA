import torch

from . import _C
from ._indexing import _logical
from .math_backend import _nvidia


def dot(self, other, *, conjugate=False):
    if self.ndim != 1 or other.ndim != 1 or self.shape != other.shape:
        raise RuntimeError('dot/vdot require vectors with the same length')
    if self.device.type != torch._C._get_privateuse1_backend_name() or self.device != other.device:
        raise RuntimeError('dot/vdot require inputs on the same RUDA device')
    if self.dtype != other.dtype or self.layout != torch.strided or other.layout != torch.strided:
        raise RuntimeError('dot/vdot require matching dtypes and strided inputs')
    if max(self.numel(), self.stride(0), other.stride(0)) > 0x7fffffff:
        raise RuntimeError('dot/vdot length and strides must fit signed int32')

    left, right = self, other
    conjugate_result = False
    if self.is_complex():
        if conjugate:
            if self.is_conj():
                if other.is_conj():
                    left, right = other.conj(), self.conj()
                else:
                    left, conjugate = self.conj(), False
            elif other.is_conj():
                right, conjugate, conjugate_result = other.conj(), False, True
        elif self.is_conj():
            if other.is_conj():
                left, right, conjugate_result = self.conj(), other.conj(), True
            else:
                left, conjugate = self.conj(), True
        elif other.is_conj():
            left, right, conjugate = other.conj(), self, True
    else:
        conjugate = False

    if torch._is_zerotensor(self) or torch._is_zerotensor(other):
        result = torch.ops.aten._efficientzerotensor.default((), dtype=self.dtype, device=self.device)
    else:
        if self.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.complex64, torch.complex128):
            raise RuntimeError('dot/vdot require float16/bfloat16/float32/float64/complex64/complex128')
        left, right = _logical(left), _logical(right)
        if _nvidia():
            from ._vendor_blas import dot as vendor_dot
            result = vendor_dot(left, right, conjugate=conjugate)
        else:
            result = torch.empty((), dtype=self.dtype, device=self.device)
            _C.scientific_call(497, [left, right, result], [int(conjugate)], [])
    return result.conj() if conjugate_result else result
