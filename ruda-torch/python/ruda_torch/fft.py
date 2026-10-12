"""Exact-length GPU FFTs with explicit plans and differentiable adjoints."""
from __future__ import annotations

import ctypes
import operator
from threading import RLock
import torch
from .math_backend import _nvidia, get_math_backend, math_backend

_lock = RLock()
_ready = False
_libraries = []


def _native():
    global _ready
    from . import _C, _native as library
    with _lock:
        if not _ready:
            if not hasattr(library, 'ruda_torch_fft_api_version') or not hasattr(_C, 'initialize_fft'):
                raise RuntimeError('RUDA FFT requires rebuilding both Rust and C++ libraries')
            library.ruda_torch_fft_api_version.restype = ctypes.c_uint32
            if library.ruda_torch_fft_api_version() != 1 or _C.fft_api_version != 1:
                raise RuntimeError('RUDA FFT API mismatch')
            _C.initialize_fft(ctypes.cast(library.ruda_torch_fft, ctypes.c_void_p).value,
                              ctypes.cast(library.ruda_torch_fft_plan, ctypes.c_void_p).value)
            _ready = True
    return _C


def _input(input, kind):
    if input.device.type != 'ruda':
        raise ValueError('RUDA FFT input must be on a ruda device')
    if input.layout != torch.strided:
        raise ValueError('FFT requires strided tensors')
    if input.dtype in (torch.float16, torch.bfloat16, torch.complex32):
        raise NotImplementedError('RUDA FFT supports float32/float64 and complex64/complex128; half/BF16 are not silently promoted')
    if input.dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
        if input.is_floating_point() or input.is_complex():
            raise NotImplementedError(f'unsupported FFT dtype: {input.dtype}')
        dtype = torch.get_default_dtype()
        if dtype not in (torch.float32, torch.float64):
            raise NotImplementedError('integer FFT requires a float32/float64 default dtype')
        input = input.to(dtype)
    if kind in (2, 5) and input.is_complex():
        raise RuntimeError('real FFT and inverse Hermitian FFT require real input')
    return input.resolve_conj().resolve_neg()


def _norm(norm):
    if norm is None or norm == 'backward':
        return 0
    if norm == 'forward':
        return 1
    if norm == 'ortho':
        return 2
    raise ValueError(f'invalid FFT normalization: {norm!r}')


def _axis(dim, rank):
    dim = operator.index(dim)
    if rank == 0 or not -rank <= dim < rank:
        raise IndexError(f'FFT dimension {dim} out of range for rank {rank}')
    return dim % rank


def _sequence(value):
    try:
        return (operator.index(value),)
    except TypeError:
        return tuple(operator.index(x) for x in value)


def _canonical(input, s, dim, kind, two_dim=False):
    rank = input.ndim
    sizes = None if s is None else _sequence(s)
    if dim is None:
        if sizes is not None:
            if len(sizes) > rank:
                raise ValueError('FFT shape has more dimensions than the input')
            axes = tuple(range(rank-len(sizes), rank))
        else:
            axes = tuple(range(rank))
    else:
        axes = tuple(_axis(d, rank) for d in _sequence(dim))
    if len(set(axes)) != len(axes):
        raise ValueError('FFT dimensions must be unique')
    if sizes is not None and len(sizes) != len(axes):
        raise ValueError('FFT shape and dimensions must have the same length')
    lengths = [input.shape[a] if sizes is None or sizes[i] == -1 else sizes[i]
               for i, a in enumerate(axes)]
    if kind in (3, 4) and axes and (sizes is None or sizes[-1] == -1):
        lengths[-1] = 2*(input.shape[axes[-1]]-1)
    if kind in (2, 3, 4, 5) and not axes:
        raise ValueError('real/Hermitian FFT requires at least one transform dimension')
    if any(n <= 0 for n in lengths):
        raise ValueError('FFT lengths must be positive')
    return axes, tuple(lengths)


def _shape(input_shape, kind, axes, lengths):
    shape = list(input_shape)
    for a, n in zip(axes, lengths):
        shape[a] = n
    if kind in (2, 5):
        shape[axes[-1]] = lengths[-1]//2+1
    return tuple(shape)


def _dtype(input_dtype, kind):
    double = input_dtype in (torch.float64, torch.complex128)
    if kind in (3, 4):
        return torch.float64 if double else torch.float32
    return torch.complex128 if double else torch.complex64


def _header(norm, axes, lengths):
    return [norm, len(axes), *axes, *lengths]


def _run(input, kind, axes, lengths, norm, out=None, plan=None):
    shape = _shape(input.shape, kind, axes, lengths)
    destination = out
    if destination is not None:
        if out.device != input.device or out.layout != torch.strided:
            raise ValueError('FFT out must be a strided tensor on the input device')
        allowed = (torch.float32, torch.float64) if kind in (3, 4) else (torch.complex64, torch.complex128)
        if out.dtype not in allowed:
            raise TypeError('FFT out dtype must match the real/complex result category')
    # Compute before resizing out: input and out may share storage. Native
    # copy kernels also retain the transform precision before an out cast.
    out = torch.empty(shape, dtype=_dtype(input.dtype, kind), device=input.device)
    if not axes:
        _native().fft_execute(15, [input, out], [], [])
    elif plan is not None:
        plan.execute(input, out)
    elif _nvidia():
        from ._vendor_fft import transform
        transform(input,kind,axes,lengths,norm,out)
    else:
        _native().fft_execute(kind, [input, out], _header(norm, axes, lengths), [])
    if destination is not None:
        if tuple(destination.shape) != shape:
            destination.resize_(shape)
        destination.copy_(out)
        return destination
    return out


class _FFT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, kind, axes, lengths, norm, plan):
        ctx.config = kind, axes, lengths, norm, tuple(input.shape), input.dtype
        ctx.math_backend = ('nvidia' if getattr(plan,'vendor',False) else 'ruda') if plan is not None else get_math_backend()
        return _run(input, kind, axes, lengths, norm, plan=plan)

    @staticmethod
    def backward(ctx, grad):
        kind, axes, lengths, norm, shape, dtype = ctx.config
        with math_backend(ctx.math_backend):
            result = _Adjoint.apply(grad,kind,axes,lengths,norm,shape,dtype)
        return result, None, None, None, None, None


class _Adjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad, kind, axes, lengths, norm, shape, dtype):
        ctx.config = kind, axes, lengths, norm
        ctx.math_backend = get_math_backend()
        output = torch.empty(shape, device=grad.device, dtype=dtype)
        if axes and _nvidia():
            from ._vendor_fft import adjoint
            adjoint(grad,kind,axes,lengths,norm,shape,dtype,output)
        elif axes:
            _native().fft_execute(20+kind, [grad.resolve_conj().resolve_neg(), output],
                                  _header(norm, axes, lengths), [])
        else:
            _native().fft_execute(15, [grad.resolve_conj().resolve_neg(), output], [], [])
        return output

    @staticmethod
    def backward(ctx, grad):
        kind, axes, lengths, norm = ctx.config
        with math_backend(ctx.math_backend):
            result = _FFT.apply(grad,kind,axes,lengths,norm,None)
        return result,None,None,None,None,None,None


def _transform(input, kind, s=None, dim=None, norm=None, out=None, two_dim=False):
    input = _input(input, kind)
    axes, lengths = _canonical(input, s, dim, kind, two_dim)
    normalization = _norm(norm)
    if out is not None:
        if torch.is_grad_enabled() and (input.requires_grad or out.requires_grad):
            raise RuntimeError('FFT out= operations do not support automatic differentiation')
        return _run(input, kind, axes, lengths, normalization, out)
    return _FFT.apply(input, kind, axes, lengths, normalization, None)


def _one(input, kind, n, dim, norm, out):
    # Unlike the n-D shape sentinel, a one-dimensional n=-1 is invalid.
    if n is not None and operator.index(n) <= 0:
        raise ValueError('FFT length must be positive')
    return _transform(input, kind, None if n is None else (operator.index(n),), (dim,), norm, out)


def fft(input, n=None, dim=-1, norm=None, *, out=None):
    return _one(input, 0, n, dim, norm, out)


def ifft(input, n=None, dim=-1, norm=None, *, out=None):
    return _one(input, 1, n, dim, norm, out)


def rfft(input, n=None, dim=-1, norm=None, *, out=None):
    return _one(input, 2, n, dim, norm, out)


def irfft(input, n=None, dim=-1, norm=None, *, out=None):
    return _one(input, 3, n, dim, norm, out)


def hfft(input, n=None, dim=-1, norm=None, *, out=None):
    return _one(input, 4, n, dim, norm, out)


def ihfft(input, n=None, dim=-1, norm=None, *, out=None):
    return _one(input, 5, n, dim, norm, out)


def fftn(input, s=None, dim=None, norm=None, *, out=None):
    return _transform(input, 0, s, dim, norm, out)


def ifftn(input, s=None, dim=None, norm=None, *, out=None):
    return _transform(input, 1, s, dim, norm, out)


def rfftn(input, s=None, dim=None, norm=None, *, out=None):
    return _transform(input, 2, s, dim, norm, out)


def irfftn(input, s=None, dim=None, norm=None, *, out=None):
    return _transform(input, 3, s, dim, norm, out)


def hfftn(input, s=None, dim=None, norm=None, *, out=None):
    return _transform(input, 4, s, dim, norm, out)


def ihfftn(input, s=None, dim=None, norm=None, *, out=None):
    return _transform(input, 5, s, dim, norm, out)


def fft2(input, s=None, dim=(-2, -1), norm=None, *, out=None):
    return _transform(input, 0, s, dim, norm, out, True)


def ifft2(input, s=None, dim=(-2, -1), norm=None, *, out=None):
    return _transform(input, 1, s, dim, norm, out, True)


def rfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None):
    return _transform(input, 2, s, dim, norm, out, True)


def irfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None):
    return _transform(input, 3, s, dim, norm, out, True)


def hfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None):
    return _transform(input, 4, s, dim, norm, out, True)


def ihfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None):
    return _transform(input, 5, s, dim, norm, out, True)


class _Shift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, axes, inverse):
        ctx.config = axes, inverse
        output = torch.empty_like(input, memory_format=torch.contiguous_format)
        _native().fft_execute(11 if inverse else 10, [input.resolve_conj().resolve_neg(), output], list(axes), [])
        return output

    @staticmethod
    def backward(ctx, grad):
        axes, inverse = ctx.config
        return _Shift.apply(grad, axes, not inverse), None, None


def _shift(input, dim, inverse):
    if input.device.type != 'ruda' or input.layout != torch.strided:
        raise ValueError('RUDA spectral shifts require a strided ruda tensor')
    axes = tuple(range(input.ndim)) if dim is None else tuple(_axis(d, input.ndim) for d in _sequence(dim))
    return _Shift.apply(input, axes, inverse)


def fftshift(input, dim=None):
    return _shift(input, dim, False)


def ifftshift(input, dim=None):
    return _shift(input, dim, True)


def _frequencies(n, d, onesided, *, dtype=None, layout=torch.strided, device=None,
                 requires_grad=False, pin_memory=False, out=None):
    n = operator.index(n)
    if n < 0:
        raise ValueError('frequency length must be nonnegative')
    shape = (n//2+1 if onesided else n,)
    if out is not None:
        if dtype is not None or device is not None:
            raise TypeError('frequency out= cannot be combined with dtype/device')
        if out.requires_grad and torch.is_grad_enabled():
            raise RuntimeError('frequency out= does not support automatic differentiation')
        device, dtype = out.device, out.dtype
    else:
        device = torch.device('ruda' if device is None else device)
        dtype = torch.get_default_dtype() if dtype is None else dtype
    if torch.device(device).type != 'ruda' or layout not in (None, torch.strided) or pin_memory:
        raise ValueError('RUDA frequencies require unpinned strided ruda storage')
    if dtype not in (torch.float32, torch.float64):
        raise TypeError('RUDA frequency helpers support float32/float64')
    if out is None:
        out = torch.empty(shape, dtype=dtype, device=device)
    elif tuple(out.shape) != shape:
        out.resize_(shape)
    if n or onesided:
        _native().fft_execute(13 if onesided else 12, [out], [n], [float(d)])
    if requires_grad:
        out.requires_grad_(True)
    return out


def fftfreq(n, d=1.0, *, dtype=None, layout=torch.strided, device=None,
            requires_grad=False, pin_memory=False, out=None):
    return _frequencies(n, d, False, dtype=dtype, layout=layout, device=device,
                        requires_grad=requires_grad, pin_memory=pin_memory, out=out)


def rfftfreq(n, d=1.0, *, dtype=None, layout=torch.strided, device=None,
             requires_grad=False, pin_memory=False, out=None):
    return _frequencies(n, d, True, dtype=dtype, layout=layout, device=device,
                        requires_grad=requires_grad, pin_memory=pin_memory, out=out)


class FFTPlan:
    """Explicitly owned, queue-bound exact FFT plan; no global plan cache.

    Input shape/dtype and transform axes/lengths are fixed. Signal values may
    change between calls. ``close`` releases retained workspace; context-manager
    use provides deterministic ownership. Execution is serialized natively.
    """
    def __init__(self, input, *, kind='fft', s=None, dim=None, norm=None):
        kinds = {'fft': 0, 'ifft': 1, 'rfft': 2, 'irfft': 3, 'hfft': 4, 'ihfft': 5}
        if kind not in kinds:
            raise ValueError('invalid FFT plan kind')
        self.kind = kinds[kind]
        input = _input(input, self.kind)
        self.axes, self.lengths = _canonical(input, s, dim, self.kind)
        if not self.axes:
            raise ValueError('explicit FFT plan requires at least one transform axis')
        self.norm = _norm(norm)
        self.shape, self.dtype, self.device = tuple(input.shape), input.dtype, input.device
        if _nvidia():
            from ._vendor_fft import VendorFFTPlan
            self._plan = VendorFFTPlan(input,self.kind,self.axes,self.lengths,self.norm)
        else:
            self._plan = _native().NativeFftPlan(input,[self.kind,*_header(self.norm,self.axes,self.lengths)])

    def execute(self, input, *, out=None):
        if self._plan is None:
            raise RuntimeError('FFT plan is closed')
        input = _input(input, self.kind)
        if tuple(input.shape) != self.shape or input.dtype != self.dtype or input.device != self.device:
            raise ValueError('FFT plan input shape/dtype/device changed')
        if out is not None:
            if torch.is_grad_enabled() and (input.requires_grad or out.requires_grad):
                raise RuntimeError('FFT out= operations do not support automatic differentiation')
            return _run(input, self.kind, self.axes, self.lengths, self.norm, out, self._plan)
        return _FFT.apply(input, self.kind, self.axes, self.lengths, self.norm, self._plan)

    @property
    def retained_bytes(self):
        if self._plan is None:
            return 0
        return self._plan.retained_bytes()

    def clear_workspace(self):
        if self._plan is None:
            raise RuntimeError('FFT plan is closed')
        self._plan.clear_workspace()

    def close(self):
        if self._plan is not None:
            self._plan.close()
            self._plan = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _aten_norm(normalization, forward):
    normalization = operator.index(normalization)
    if normalization == 0:
        return 'backward' if forward else 'forward'
    if normalization == 1:
        return 'ortho'
    if normalization == 2:
        return 'forward' if forward else 'backward'
    raise ValueError('invalid ATen FFT normalization')


def _c2c(self, dim, normalization, forward, *, out=None):
    if not self.is_complex():
        raise TypeError('_fft_c2c requires complex input')
    return _transform(self, 0 if forward else 1, None, dim, _aten_norm(normalization, forward), out)


def _r2c(self, dim, normalization, onesided, *, out=None):
    if self.is_complex():
        raise TypeError('_fft_r2c requires real input')
    return _transform(self, 2 if onesided else 0, None, dim, _aten_norm(normalization, True), out)


def _c2r(self, dim, normalization, last_dim_size, *, out=None):
    if not self.is_complex():
        raise TypeError('_fft_c2r requires complex input')
    axes = tuple(_axis(a, self.ndim) for a in _sequence(dim))
    if not axes:
        raise ValueError('_fft_c2r requires a transform axis')
    sizes = [self.shape[a] for a in axes]
    sizes[-1] = operator.index(last_dim_size)
    return _transform(self, 3, sizes, axes, _aten_norm(normalization, False), out)


def register():
    """Register standard ATen FFT schemas without replacing torch.fft functions."""
    if _libraries:
        return
    device = torch.library.Library('aten', 'IMPL', 'PrivateUse1')
    autograd = torch.library.Library('aten', 'IMPL', 'AutogradPrivateUse1')
    functions = {
        'fft_fft': fft, 'fft_ifft': ifft, 'fft_rfft': rfft, 'fft_irfft': irfft,
        'fft_hfft': hfft, 'fft_ihfft': ihfft,
        'fft_fftn': fftn, 'fft_ifftn': ifftn, 'fft_rfftn': rfftn, 'fft_irfftn': irfftn,
        'fft_hfftn': hfftn, 'fft_ihfftn': ihfftn,
        'fft_fft2': fft2, 'fft_ifft2': ifft2, 'fft_rfft2': rfft2, 'fft_irfft2': irfft2,
        'fft_hfft2': hfft2, 'fft_ihfft2': ihfft2,
        '_fft_c2c': _c2c, '_fft_r2c': _r2c, '_fft_c2r': _c2r,
    }
    for name, function in functions.items():
        device.impl(name, function)
        device.impl(name+'.out', function)
        autograd.impl(name, function)
        autograd.impl(name+'.out', function)
    for name, function in {'fft_fftshift': fftshift, 'fft_ifftshift': ifftshift}.items():
        device.impl(name, function)
        autograd.impl(name, function)
    for name, function in {'fft_fftfreq': fftfreq, 'fft_rfftfreq': rfftfreq}.items():
        device.impl(name, function)
        device.impl(name+'.out', function)
    _libraries.extend((device, autograd))
