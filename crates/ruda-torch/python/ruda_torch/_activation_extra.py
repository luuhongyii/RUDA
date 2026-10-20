import torch

from . import _C
from ._indexing import _logical
from ._ops import _dimension, _dtypes, _glu_halves, _storage_dtypes
from ._ordering import _output


def _matching(tensors, *, floating=True):
    first = tensors[0]
    allowed = _dtypes if floating else _storage_dtypes
    if first.dtype not in allowed:
        raise RuntimeError('unsupported GLU dtype')
    for value in tensors:
        if value.layout != torch.strided or value.device != first.device or value.dtype != first.dtype:
            raise RuntimeError('GLU inputs must be strided tensors with matching dtype and device')


def _target(out, reference, inputs, *, cast=False):
    if out is None:
        return
    valid_dtype = torch.can_cast(reference.dtype, out.dtype) if cast else out.dtype == reference.dtype
    if out.layout != torch.strided or out.device != reference.device or not valid_dtype:
        raise RuntimeError('activation output dtype/device/layout mismatch')
    for value in inputs:
        _C.check_inplace(out, value)


def _native(op, tensors):
    if tensors[0].numel() > 2**32 - 1:
        raise RuntimeError('activation exceeds native U32 indexing')
    if tensors[0].numel():
        _C.scientific_call(op, tensors, [], [])


def glu(a, dim=-1, *, out=None):
    left, gate, _ = _glu_halves(a, dim)
    _matching((a,))
    _target(out, a, (left, gate), cast=True)
    result = torch.empty_like(left)
    _native(407, [_logical(left), _logical(gate), result])
    return _output(result, out, (left, gate), cast=True)


def glu_backward(grad, a, dim, *, grad_input=None):
    left, gate, axis = _glu_halves(a, dim)
    _matching((a, grad), floating=bool(a.numel()))
    if grad.shape != left.shape:
        raise RuntimeError('GLU gradient shape must match the forward output')
    _target(grad_input, a, (a, grad))
    result = torch.empty_like(a, memory_format=torch.contiguous_format)
    da, db, _ = _glu_halves(result, axis)
    _native(408, [_logical(left), _logical(gate), _logical(grad), da, db])
    return _output(result, grad_input, (a, grad))


def glu_jvp(result, x, dx, dim):
    axis = _dimension(dim, x.ndim)
    width = result.size(axis)
    gate = x.narrow(axis, width, width)
    da = dx.narrow(axis, 0, width)
    db = dx.narrow(axis, width, width)
    _matching((result, gate, da, db))
    operands = torch.broadcast_tensors(*(_logical(value) for value in (result, gate, da, db)))
    output = torch.empty_like(operands[0])
    _native(409, [*operands, output])
    return output


def glu_backward_jvp(grad_x, grad_glu, x, dgrad_glu, dx, dim):
    axis = _dimension(dim, x.ndim)
    width = grad_glu.size(axis)
    a = _logical(x.narrow(axis, 0, width))
    b = _logical(x.narrow(axis, width, width))
    da = _logical(dx.narrow(axis, 0, width))
    db = _logical(dx.narrow(axis, width, width))
    grad_a = _logical(grad_x.narrow(axis, 0, width))
    grad_x.narrow(axis, width, width)
    sigmoid = b.sigmoid()
    result = a * sigmoid
    db_scaled = db - db * sigmoid
    dgrad_a = _logical(dgrad_glu) * sigmoid + grad_a * db_scaled
    dgrad_b = dgrad_a * (a - result) + grad_a * (da - da * sigmoid - result * db_scaled)
    return torch.cat((dgrad_a, dgrad_b), dim=axis)


def register(library):
    library.impl('glu.out', glu)
    library.impl('glu_backward.grad_input', glu_backward)
    library.impl('glu_jvp', glu_jvp)
    library.impl('glu_backward_jvp', glu_backward_jvp)
