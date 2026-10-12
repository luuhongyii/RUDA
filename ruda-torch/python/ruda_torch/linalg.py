from __future__ import annotations

import ctypes
import torch
from . import _C

_available = False
_installed = False
_libraries = []


def initialize(library):
    global _available
    if not hasattr(library, 'ruda_torch_linalg_api_version') or not hasattr(_C, 'initialize_linalg'):
        return False
    library.ruda_torch_linalg_api_version.restype = ctypes.c_uint32
    if library.ruda_torch_linalg_api_version() != 1 or getattr(_C, 'linalg_api_version', None) != 1:
        raise RuntimeError('RUDA linalg API mismatch; rebuild Rust and C++ extensions')
    _C.initialize_linalg(ctypes.cast(library.ruda_torch_linalg, ctypes.c_void_p).value)
    _available = True
    return True


def available():
    return _available


def _matrix(a, *, square=False):
    if a.device.type != torch._C._get_privateuse1_backend_name() or a.dtype not in (torch.float32, torch.float64):
        raise RuntimeError('RUDA linalg requires real float32/float64 tensors on the RUDA device')
    if a.ndim < 2 or square and a.shape[-2] != a.shape[-1]:
        raise RuntimeError('linalg requires a matrix' + (' with square final dimensions' if square else ''))
    return a


def _matching(a, b):
    if a.device != b.device or a.dtype != b.dtype:
        raise RuntimeError('linalg inputs must have the same dtype and device')


def _empty(a, shape, dtype=None):
    return torch.empty(tuple(shape), device=a.device, dtype=a.dtype if dtype is None else dtype)


def _info(a, batch=None):
    return _empty(a, a.shape[:-2] if batch is None else batch, torch.int32)


def _call(op, tensors, ints=()):
    if not _available:
        raise RuntimeError('RUDA linalg native API is unavailable')
    _C.linalg(op, [t.resolve_conj().resolve_neg() for t in tensors], list(ints), [])


def _check_errors(info, api_name='linalg', *, is_matrix=True):
    if info.device.type != torch._C._get_privateuse1_backend_name() or info.dtype != torch.int32:
        raise RuntimeError('linalg info must be a RUDA int32 tensor')
    try:
        _call(255, [info])
    except torch.linalg.LinAlgError as error:
        raise torch.linalg.LinAlgError(f'{api_name}: {error}') from error


def _rhs(a, b, *, left=True, vector=False):
    _matching(a, b)
    is_vector = vector and (b.ndim == 1 or b.ndim == a.ndim - 1 and tuple(b.shape) == tuple(a.shape[:-1]))
    if is_vector:
        if not left:
            raise RuntimeError('linalg.solve: vector RHS is not supported with left=False')
        b = b.unsqueeze(-1)
    if b.ndim < 2 or (b.shape[-2] if left else b.shape[-1]) != a.shape[-2]:
        raise RuntimeError('linalg RHS matrix dimensions are incompatible')
    batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    a = a.expand(batch + a.shape[-2:])
    b = b.expand(batch + b.shape[-2:])
    return a, b, batch, is_vector


def lu_factor_ex(A, *, pivot=True, check_errors=False):
    _matrix(A)
    LU = _empty(A, A.shape)
    pivots = _empty(A, A.shape[:-2] + (min(A.shape[-2:]),), torch.int32)
    info = _info(A)
    _call(0, [A, LU, pivots, info], [int(pivot)])
    if check_errors:
        _check_errors(info, 'linalg.lu_factor_ex')
    return LU, pivots, info


def _pivots(LU, pivots, batch):
    if pivots.device != LU.device or pivots.dtype != torch.int32 or pivots.ndim < 1 or pivots.shape[-1] != min(LU.shape[-2:]):
        raise RuntimeError('LU pivots require int32, the same device, and min(m,n) entries')
    if torch.broadcast_shapes(pivots.shape[:-1], batch) != batch:
        raise RuntimeError('LU pivot batch is not broadcastable to the system batch')
    return pivots.expand(batch + (pivots.shape[-1],))


def lu_solve(LU, pivots, B, *, left=True, adjoint=False):
    _matrix(LU, square=True)
    a, b, batch, _ = _rhs(LU, B, left=left)
    p = _pivots(LU, pivots, batch)
    rhs = b if left else b.transpose(-2, -1)
    x = _empty(B, rhs.shape)
    info = _info(LU, batch)
    _call(1, [a, p, rhs, x, info], [int(bool(adjoint) != (not left))])
    return x if left else x.transpose(-2, -1)


def solve_ex(A, B, *, left=True, check_errors=False):
    _matrix(A, square=True)
    a, b, batch, vector = _rhs(A, B, left=left, vector=True)
    LU, pivots, info = lu_factor_ex(A)
    if check_errors:
        _check_errors(info, 'linalg.solve')
    x = lu_solve(LU, pivots, b, left=left)
    if vector:
        x = x.squeeze(-1)
    return x, LU, pivots, info


def cholesky_ex(self, *, upper=False, check_errors=False):
    _matrix(self, square=True)
    factor, info = _empty(self, self.shape), _info(self)
    _call(2, [self, factor, info], [int(upper)])
    if check_errors:
        _check_errors(info, 'linalg.cholesky_ex')
    return factor, info


def cholesky_solve(self, input2, upper=False):
    _matrix(input2, square=True)
    a, b, batch, _ = _rhs(input2, self)
    x, info = _empty(self, b.shape), _info(input2, batch)
    _call(4, [a, b, x, info], [int(upper)])
    return x


def cholesky_inverse(self, upper=False):
    _matrix(self, square=True)
    inverse, info = _empty(self, self.shape), _info(self)
    _call(13, [self, inverse, info], [int(upper)])
    _check_errors(info, 'cholesky_inverse')
    return inverse


def solve_triangular(self, B, *, upper, left=True, unitriangular=False):
    _matrix(self, square=True)
    a, b, batch, _ = _rhs(self, B, left=left)
    rhs = b if left else b.transpose(-2, -1)
    x, info = _empty(B, rhs.shape), _info(self, batch)
    _call(3, [a, rhs, x, info], [int(upper), int(not left), int(unitriangular)])
    return x if left else x.transpose(-2, -1)


def triangular_solve(self, A, upper=True, transpose=False, unitriangular=False):
    _matrix(A, square=True)
    a, b, batch, _ = _rhs(A, self)
    x, info = _empty(self, b.shape), _info(A, batch)
    _call(3, [a, b, x, info], [int(upper), int(transpose), int(unitriangular)])
    return x, a.clone()


def qr(A, mode='reduced'):
    _matrix(A)
    if mode not in ('reduced', 'complete', 'r'):
        raise RuntimeError("qr mode must be 'reduced', 'complete', or 'r'")
    batch, m, n = A.shape[:-2], A.shape[-2], A.shape[-1]
    k = min(m, n)
    Q = _empty(A, (0,) if mode == 'r' else batch + (m, m if mode == 'complete' else k))
    R = _empty(A, batch + (m if mode == 'complete' else k, n))
    packed, tau, info = _empty(A, A.shape), _empty(A, batch + (k,)), _info(A)
    _call(5, [A, Q, R, packed, tau, info], [('reduced', 'complete', 'r').index(mode)])
    return Q, R


def geqrf(self):
    _matrix(self)
    packed = _empty(self, self.shape)
    tau = _empty(self, self.shape[:-2] + (min(self.shape[-2:]),))
    info = _info(self)
    _call(12, [self, packed, tau, info])
    return packed, tau


def ormqr(self, input2, input3, left=True, transpose=False):
    _matrix(self)
    _matching(self, input2)
    _matching(self, input3)
    if input2.ndim < 1 or input2.shape[-1] > min(self.shape[-2:]):
        raise RuntimeError('ormqr tau must contain at most min(m,n) reflectors')
    if input3.ndim < 2 or input3.shape[-2 if left else -1] != self.shape[-2]:
        raise RuntimeError('ormqr matrix dimensions do not match the order of Q')
    batch = torch.broadcast_shapes(self.shape[:-2], input2.shape[:-1], input3.shape[:-2])
    a = self.expand(batch + self.shape[-2:])
    tau = input2.expand(batch + (input2.shape[-1],))
    b = input3.expand(batch + input3.shape[-2:])
    rhs = b if left else b.transpose(-2, -1)
    x = _empty(self, rhs.shape)
    _call(11, [a, tau, rhs, x], [int(bool(transpose) != (not left))])
    return x if left else x.transpose(-2, -1)


def eigh(A, UPLO='L', compute_v=True):
    _matrix(A, square=True)
    if UPLO not in ('L', 'U', 'l', 'u'):
        raise RuntimeError("eigh UPLO must be 'L' or 'U'")
    values = _empty(A, A.shape[:-1])
    vectors = _empty(A, A.shape if compute_v else (0,))
    info = _info(A)
    _call(7, [A, values, vectors, info], [int(UPLO.upper() == 'U'), int(compute_v)])
    _check_errors(info, 'linalg.eigh')
    return values, vectors


def svd(A, full_matrices=False, compute_uv=True, *, driver=None):
    _matrix(A)
    if driver is not None:
        raise RuntimeError('RUDA SVD uses native one-sided Jacobi; named cuSOLVER drivers are not available')
    batch, m, n = A.shape[:-2], A.shape[-2], A.shape[-1]
    k = min(m, n)
    U = _empty(A, batch + (m, m if full_matrices else k) if compute_uv else (0,))
    S = _empty(A, batch + (k,))
    Vh = _empty(A, batch + (n if full_matrices else k, n) if compute_uv else (0,))
    info = _info(A)
    _call(8, [A, U, S, Vh, info], [int(full_matrices), int(compute_uv)])
    _check_errors(info, 'linalg.svd')
    return U, S, Vh


def inv_ex(A, *, check_errors=False):
    _matrix(A, square=True)
    inverse, LU, pivots, info = _empty(A, A.shape), _empty(A, A.shape), _empty(A, A.shape[:-1], torch.int32), _info(A)
    _call(9, [A, inverse, LU, pivots, info])
    if check_errors:
        _check_errors(info, 'linalg.inv_ex')
    return inverse, info


def _determinants(A):
    _matrix(A, square=True)
    LU, pivots = _empty(A, A.shape), _empty(A, A.shape[:-1], torch.int32)
    det, sign, logabs = (_empty(A, A.shape[:-2]) for _ in range(3))
    _call(10, [A, LU, pivots, det, sign, logabs])
    return det, sign, logabs, LU, pivots


def det(A):
    determinant, _, _, LU, pivots = _determinants(A)
    return determinant, LU, pivots


def slogdet(A):
    _, sign, logabs, LU, pivots = _determinants(A)
    return sign, logabs, LU, pivots


def lstsq(self, b, rcond=None, *, driver=None):
    _matrix(self)
    if not 0 <= self.ndim - b.ndim <= 1:
        raise RuntimeError('linalg.lstsq requires input.ndim >= b.ndim and input.ndim - b.ndim <= 1')
    if isinstance(driver, str):
        driver = driver.lower()
    if driver not in (None, 'gels'):
        raise RuntimeError("RUDA GPU lstsq supports only the full-rank 'gels' driver")
    a, rhs, batch, vector = _rhs(self, b, vector=True)
    m, n, nrhs = self.shape[-2], self.shape[-1], rhs.shape[-1]
    solution = _empty(self, batch + (n, nrhs))
    residuals = _empty(self, batch + (nrhs,) if m > n else (0,))
    info = _info(self, batch)
    _call(6, [a, rhs, solution, residuals, info])
    _check_errors(info, 'linalg.lstsq')
    return (solution.squeeze(-1) if vector else solution), residuals, _empty(self, (0,), torch.int64), _empty(self, (0,))


def lu_unpack(LU_data, LU_pivots, unpack_data=True, unpack_pivots=True):
    _matrix(LU_data)
    batch, m, n = LU_data.shape[:-2], LU_data.shape[-2], LU_data.shape[-1]
    k = min(m, n)
    pivots = _pivots(LU_data, LU_pivots, batch)
    P = _empty(LU_data, batch + (m, m) if unpack_pivots else (0,))
    L = _empty(LU_data, batch + (m, k) if unpack_data else (0,))
    U = _empty(LU_data, batch + (k, n) if unpack_data else (0,))
    _call(15, [LU_data, pivots, P, L, U], [int(unpack_data), int(unpack_pivots)])
    return P, L, U


def _copy_out(value, out):
    if out.device != value.device or not torch.can_cast(value.dtype, out.dtype):
        raise RuntimeError('linalg output has incompatible dtype or device')
    if tuple(out.shape) != tuple(value.shape):
        out.resize_(value.shape)
    out.copy_(value)
    return out


def _out(function, names):
    def implementation(*args, **kwargs):
        outputs = [kwargs.pop(name) for name in names]
        values = function(*args, **kwargs)
        if len(names) == 1:
            return _copy_out(values, outputs[0])
        for value, target in zip(values, outputs):
            _copy_out(value, target)
        return tuple(outputs)
    return implementation


def install(registry=None):
    global _installed
    if not _available or _installed:
        return False
    if registry is None:
        registry = torch.library.Library('aten', 'IMPL', 'PrivateUse1')
        _libraries.append(registry)
    operators = {
        '_linalg_solve_ex': (solve_ex, 'result', ('result', 'LU', 'pivots', 'info')),
        'linalg_lu_factor_ex': (lu_factor_ex, 'out', ('LU', 'pivots', 'info')),
        'linalg_lu_solve': (lu_solve, 'out', ('out',)),
        'lu_solve': (lambda self, LU_data, LU_pivots: lu_solve(LU_data, LU_pivots, self), 'out', ('out',)),
        'linalg_cholesky_ex': (cholesky_ex, 'L', ('L', 'info')),
        'cholesky_solve': (cholesky_solve, 'out', ('out',)),
        'cholesky_inverse': (cholesky_inverse, 'out', ('out',)),
        'linalg_solve_triangular': (solve_triangular, 'out', ('out',)),
        'triangular_solve': (triangular_solve, 'X', ('X', 'M')),
        'linalg_qr': (qr, 'out', ('Q', 'R')),
        'geqrf': (geqrf, 'a', ('a', 'tau')),
        'ormqr': (ormqr, 'out', ('out',)),
        '_linalg_eigh': (eigh, 'eigenvalues', ('eigenvalues', 'eigenvectors')),
        '_linalg_svd': (svd, 'U', ('U', 'S', 'Vh')),
        'linalg_inv_ex': (inv_ex, 'inverse', ('inverse', 'info')),
        '_linalg_det': (det, 'result', ('result', 'LU', 'pivots')),
        '_linalg_slogdet': (slogdet, 'sign', ('sign', 'logabsdet', 'LU', 'pivots')),
        'linalg_lstsq': (lstsq, 'out', ('solution', 'residuals', 'rank', 'singular_values')),
        'lu_unpack': (lu_unpack, 'out', ('P', 'L', 'U')),
    }
    for name, (function, overload, names) in operators.items():
        registry.impl(name, function)
        registry.impl(name + '.' + overload, _out(function, names))
    registry.impl('_linalg_check_errors', _check_errors)
    _installed = True
    return True
