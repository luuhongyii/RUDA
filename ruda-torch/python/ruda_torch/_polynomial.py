from functools import partial

import torch

from . import _C
from ._indexing import _logical
from ._ops import _dtypes, _integral_dtypes
from ._ordering import _output


@torch.no_grad()
def polynomial(x, n, *, op, out=None):
    original = (x, n)
    if any(value.layout != torch.strided or value.dtype not in _dtypes + _integral_dtypes for value in original):
        raise RuntimeError('special polynomials require strided real operands')
    dtype = torch.result_type(x, n)
    if dtype in _integral_dtypes:
        dtype = torch.get_default_dtype()
    if dtype not in (torch.float32, torch.float64):
        raise RuntimeError('special polynomials require a float32 or float64 common dtype')
    device = next((value.device for value in original if value.device.type != 'cpu'), None)
    if device is None:
        raise RuntimeError('RUDA special polynomials require a non-CPU tensor operand')
    for value in original:
        if value.device != device and not (value.device.type == 'cpu' and value.ndim == 0):
            raise RuntimeError('special polynomial operands must share a device except CPU scalars')
    left, right = torch.broadcast_tensors(*(_logical(value).to(device=device, dtype=dtype) for value in original))
    if left.numel() > 2**32 - 1:
        raise RuntimeError('special polynomial output exceeds native U32 indexing')
    if out is not None:
        for value in original:
            _C.check_inplace(out, value)
    result = torch.empty_like(left)
    if result.numel():
        _C.scientific_call(op, [left, right, result], [], [])
    return _output(result, out, original, cast=True)


def register(library):
    names = (
        'chebyshev_polynomial_t', 'chebyshev_polynomial_u',
        'chebyshev_polynomial_v', 'chebyshev_polynomial_w',
        'shifted_chebyshev_polynomial_t', 'shifted_chebyshev_polynomial_u',
        'shifted_chebyshev_polynomial_v', 'shifted_chebyshev_polynomial_w',
        'hermite_polynomial_h', 'hermite_polynomial_he',
        'laguerre_polynomial_l', 'legendre_polynomial_p',
    )
    for op, name in enumerate(names, 368):
        function = partial(polynomial, op=op)
        library.impl('special_' + name, function)
        library.impl('special_' + name + '.out', function)
