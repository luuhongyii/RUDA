import numbers

import torch

from . import _C
from ._indexing import _logical, index_select
from ._ops import _dtypes, _integral_dtypes, _typed_value
from ._ordering import _output
from ._unique import unique


def _input(value):
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.dtype not in _dtypes + _integral_dtypes + (torch.complex32,) or value.dtype == torch.bool:
            raise RuntimeError('unsupported isin input dtype or layout')
        if value.numel() > 2**32 - 1:
            raise RuntimeError('isin input exceeds native U32 indexing')
    elif isinstance(value, (bool, complex)) or not isinstance(value, numbers.Real):
        raise RuntimeError('isin scalars must be real and non-boolean')


def _scalar(value, device):
    return _typed_value(value, torch.int64 if isinstance(value, numbers.Integral) else torch.float64, device)


def _parts(value):
    if value.is_complex():
        pair = torch.view_as_real(value)
        return [pair[..., 0], pair[..., 1]]
    return [value, value]


def isin(elements, test_elements, *, assume_unique=False, invert=False, out=None):
    _input(elements)
    _input(test_elements)
    original = tuple(value for value in (elements, test_elements) if isinstance(value, torch.Tensor))
    if not original:
        raise RuntimeError('isin requires at least one tensor operand')
    device = original[0].device
    if device.type == 'cpu' or any(value.device != device for value in original):
        raise RuntimeError('RUDA isin operands must be on the same GPU device')
    shape = elements.shape if isinstance(elements, torch.Tensor) else ()
    if out is not None:
        if out.dtype != torch.bool or out.device != device or out.layout != torch.strided:
            raise RuntimeError('isin output must be a bool tensor on the input device')
        for value in original:
            _C.check_inplace(out, value)
    scalar_test = not isinstance(test_elements, torch.Tensor)
    if scalar_test:
        dtype = torch.result_type(elements, test_elements)
        left = _logical(elements).to(dtype)
        right = _scalar(test_elements, device).to(dtype)
    else:
        left = _logical(elements) if isinstance(elements, torch.Tensor) else _scalar(elements, device)
        right = _logical(test_elements)
    result = torch.empty(shape, dtype=torch.bool, device=device)
    if not result.numel():
        return _output(result, out, original)
    if scalar_test or right.numel() < int(10.0 * left.numel() ** 0.145):
        if not scalar_test:
            right = right.view((1,) * left.ndim + (-1,))
            dtype = torch.promote_types(left.dtype, right.dtype)
            left, right = left.to(dtype), right.to(dtype)
        _C.scientific_call(402, _parts(left) + _parts(right) + [result], [int(invert), int(left.is_complex())], [])
    else:
        if assume_unique:
            left_flat, right_flat = left.reshape(-1), right.reshape(-1)
            inverse = None
        else:
            left_flat, inverse = unique(left, sorted=False, return_inverse=True)[:2]
            right_flat = unique(right, sorted=False)[0]
        if left_flat.numel() + right_flat.numel() > 2**32 - 1:
            raise RuntimeError('isin sorting workspace exceeds native U32 indexing')
        values, order = torch.cat((left_flat, right_flat)).sort(stable=True)
        mask = torch.empty(values.shape, dtype=torch.bool, device=device)
        _C.scientific_call(403, [values, order, mask], [int(invert)], [])
        selected = mask[:left.numel()] if assume_unique else index_select(mask, 0, inverse.reshape(-1))
        result.copy_(selected.reshape(shape))
    return _output(result, out, original)


def register(library):
    for overload in ('Tensor_Tensor', 'Tensor_Scalar', 'Scalar_Tensor',
                     'Tensor_Tensor_out', 'Tensor_Scalar_out', 'Scalar_Tensor_out'):
        library.impl('isin.' + overload, isin)
