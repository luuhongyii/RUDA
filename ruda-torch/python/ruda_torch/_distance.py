import math

import torch
from . import _C
from ._ordering import _output


def _floating(a):
    if not a.is_floating_point() or a.layout != torch.strided:
        raise RuntimeError('distance operators require strided floating-point input')


def _direct(a):
    if a.dtype not in (torch.float32, torch.float64):
        raise RuntimeError('direct distance kernels require float32 or float64 input')


def _same(a, *values):
    for value in values:
        if value.dtype != a.dtype or value.device != a.device or value.layout != torch.strided:
            raise RuntimeError('distance operands and saved tensors must have matching dtype and device')


def _cdist_shape(x1, x2):
    _floating(x1)
    _floating(x2)
    if x1.ndim < 2 or x2.ndim < 2 or x1.shape[-1] != x2.shape[-1]:
        raise RuntimeError('cdist requires rank >= 2 inputs with the same feature dimension')
    if x1.device != x2.device:
        raise RuntimeError('cdist inputs must be on the same device')
    batch = torch.broadcast_shapes(x1.shape[:-2], x2.shape[:-2])
    return (*batch, x1.shape[-2], x2.shape[-2]), batch


def _workspace(shape, a):
    if math.prod(shape) > 2**32 - 1:
        raise RuntimeError('distance gradient workspace exceeds native U32 indexing')
    return a.new_empty(shape)


def cdist_forward(x1, x2, p, compute_mode=None, *, out=None):
    shape, batch = _cdist_shape(x1, x2)
    mode = 0 if compute_mode is None else compute_mode
    if not p >= 0 or mode not in (0, 1, 2):
        raise RuntimeError('cdist requires nonnegative p and compute_mode 0, 1, or 2')
    rows, columns, features = x1.shape[-2], x2.shape[-2], x1.shape[-1]
    if math.prod(shape) == 0:
        result = x1.new_empty(shape)
    elif features == 0:
        result = x1.new_zeros(shape)
    else:
        _same(x1, x2)
        left = x1.resolve_neg().expand(*batch, rows, features)
        right = x2.resolve_neg().expand(*batch, columns, features)
        if p == 2 and (mode == 1 or mode == 0 and (rows > 25 or columns > 25)):
            count = math.prod(batch)
            if count == 1:
                result = torch.ops.aten._euclidean_dist.default(x1, x2).view(shape)
            else:
                result = torch.ops.aten._euclidean_dist.default(
                    left.contiguous().view(count, rows, features),
                    right.contiguous().view(count, columns, features)).view(shape)
        else:
            _direct(x1)
            result = x1.new_empty(shape)
            _C.scientific_call(339, [left, right, result], [0], [p])
    return _output(result, out, (x1, x2))


def cdist_backward(grad, x1, x2, p, cdist, *, out=None):
    shape, batch = _cdist_shape(x1, x2)
    rows, columns, features = x1.shape[-2], x2.shape[-2], x1.shape[-1]
    if rows == 0 or columns == 0 or features == 0 or math.prod(batch) == 0:
        result = torch.zeros_like(x1)
    else:
        _same(x1, x2, grad, cdist)
        if tuple(grad.shape) != shape or tuple(cdist.shape) != shape:
            raise RuntimeError('cdist gradients and saved distances must match the broadcast output shape')
        expanded = (*batch, rows, features)
        if p == 0:
            result = x1.new_zeros(expanded)
        else:
            _direct(x1)
            left = x1.resolve_neg().expand(expanded)
            right = x2.resolve_neg().expand(*batch, columns, features)
            terms = _workspace((math.prod(batch), columns, rows, features), x1)
            _C.scientific_call(340, [left, right, grad.resolve_neg(), cdist.resolve_neg(), terms], [0], [p])
            result = terms.sum(1).view(expanded)
    return _output(result, out, (grad, x1, x2, cdist))


def _pdist_shape(a):
    _floating(a)
    if a.ndim != 2 or not a.is_contiguous():
        raise RuntimeError('_pdist requires contiguous two-dimensional input')
    return a.shape[0] * (a.shape[0] - 1) // 2


def pdist_forward(a, p=2, *, out=None):
    count = _pdist_shape(a)
    if count == 0:
        result = a.new_empty((0,))
    elif a.shape[1] == 0:
        result = a.new_zeros((count,))
    else:
        _direct(a)
        value = a.resolve_neg()
        result = a.new_empty((count,))
        _C.scientific_call(339, [value, value, result], [1], [p])
    return _output(result, out, (a,))


def pdist_backward(grad, a, p, pdist, *, out=None):
    count = _pdist_shape(a)
    if not pdist.is_contiguous():
        raise RuntimeError('_pdist_backward requires contiguous saved distances')
    if p == 0 or not grad.numel() or not a.numel():
        result = torch.zeros_like(a)
    else:
        _direct(a)
        _same(a, grad, pdist)
        if tuple(grad.shape) != (count,) or tuple(pdist.shape) != (count,):
            raise RuntimeError('pdist gradients and saved distances must match the condensed pair count')
        value = a.resolve_neg()
        terms = _workspace((a.shape[0] - 1, *a.shape), a)
        _C.scientific_call(340, [value, value, grad.resolve_neg(), pdist.resolve_neg(), terms], [1], [p])
        result = terms.sum(0)
    return _output(result, out, (grad, a, pdist))


def register(library):
    for name, function in (('_cdist_forward', cdist_forward), ('_cdist_backward', cdist_backward),
                           ('_pdist_forward', pdist_forward), ('_pdist_backward', pdist_backward)):
        library.impl(name, function)
        library.impl(name + '.out', function)
