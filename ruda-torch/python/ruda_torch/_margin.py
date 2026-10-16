import math

import torch

from . import _C
from ._indexing import _logical
from ._loss import _reduction
from ._ops import _dtypes
from ._ordering import _output, _outputs


def _shape(a, target, *, multilabel):
    if a.layout != torch.strided or target.layout != torch.strided or target.device != a.device:
        raise RuntimeError('margin loss requires strided input and target on the same device')
    if a.ndim > 2 or (a.ndim and a.shape[-1] == 0):
        raise RuntimeError('margin loss requires a scalar, nonempty vector, or matrix with nonempty class dimension')
    frames, classes = (a.shape[0], a.shape[1]) if a.ndim == 2 else (1, a.numel())
    if multilabel:
        valid = (target.ndim == 2 and target.shape == a.shape) if a.ndim == 2 else (target.ndim <= 1 and target.numel() == classes)
    else:
        valid = target.ndim <= 1 and target.numel() == frames
    if not valid:
        raise RuntimeError('margin loss target shape is inconsistent with input')
    if a.numel() > 2**32 - 1 or max(frames, classes) > 2**32 - 1:
        raise RuntimeError('margin loss exceeds native U32 indexing')
    return frames, classes


def _types(a, target):
    if a.dtype not in _dtypes or target.dtype != torch.int64:
        raise RuntimeError('margin loss requires real floating input and int64 target')


def _scalar(value):
    if isinstance(value, complex):
        if value.imag != 0:
            raise RuntimeError('margin loss scalar must have zero imaginary part')
        value = value.real
    return value


def _power(p):
    try:
        value = int(_scalar(p))
    except (ValueError, OverflowError) as exc:
        raise RuntimeError('multi_margin_loss p must convert to 1 or 2') from exc
    if value not in (1, 2):
        raise RuntimeError('multi_margin_loss p must convert to 1 or 2')
    return value


def _margin(margin, dtype):
    value = float(_scalar(margin))
    if math.isfinite(value) and abs(value) > torch.finfo(dtype).max:
        raise RuntimeError('multi_margin_loss margin overflows the input dtype')
    return value


def _weight(a, weight, classes):
    if weight is not None:
        if (weight.layout != torch.strided or weight.device != a.device or weight.dtype != a.dtype
                or weight.ndim > 1 or weight.numel() != classes):
            raise RuntimeError('multi_margin_loss weight must match the input dtype/device and class count')
        return _logical(weight).reshape(-1)
    return a.new_empty((0,))


def _empty_output(a, shape, out, inputs):
    if out is None:
        return torch.empty(shape, dtype=a.dtype, device=a.device)
    if out.dtype != a.dtype or out.device != a.device:
        raise RuntimeError('margin loss output dtype or device mismatch')
    for value in inputs:
        if isinstance(value, torch.Tensor):
            _C.check_inplace(out, value)
    if tuple(out.shape) != tuple(shape):
        _C.prepare_index_output(out, list(shape))
    return out


def _grad(grad, a, frames, reduction):
    if (grad.layout != torch.strided or grad.device != a.device or grad.dtype != a.dtype
            or grad.numel() != (frames if reduction == 0 else 1)):
        raise RuntimeError('margin loss grad_output shape, dtype or device mismatch')
    return _logical(grad).reshape(-1)


def multi_margin(a, target, p=1, margin=1, weight=None, reduction=1, *, out=None):
    _reduction(reduction)
    power = _power(p)
    frames, classes = _shape(a, target, multilabel=False)
    w = _weight(a, weight, classes)
    shape = (frames,) if reduction == 0 and target.ndim > 0 else ()
    if not a.numel():
        return _empty_output(a, shape, out, (a, target, weight))
    _types(a, target)
    boundary = _margin(margin, a.dtype)
    x, t = _logical(a).reshape(frames, classes), _logical(target).reshape(frames)
    rows = torch.empty((frames,), dtype=a.dtype, device=a.device)
    _C.scientific_call(361, [x, t, w, rows], [power, reduction, int(weight is not None)], [boundary])
    result = rows.sum() if a.ndim == 2 and reduction != 0 else rows.reshape(shape)
    return _output(result, out, (a, target, weight))


def multi_margin_backward(grad_output, a, target, p, margin, weight=None, reduction=1, *, grad_input=None):
    _reduction(reduction)
    power = _power(p)
    frames, classes = _shape(a, target, multilabel=False)
    w = _weight(a, weight, classes)
    if not a.numel():
        return _empty_output(a, a.shape, grad_input, (grad_output, a, target, weight))
    _types(a, target)
    g, boundary = _grad(grad_output, a, frames, reduction), _margin(margin, a.dtype)
    x, t = _logical(a).reshape(frames, classes), _logical(target).reshape(frames)
    value = torch.empty((frames, classes), dtype=a.dtype, device=a.device)
    _C.scientific_call(362, [x, t, w, g, value], [power, reduction, int(weight is not None)], [boundary])
    return _output(value.reshape(a.shape), grad_input, (grad_output, a, target, weight))


def multilabel_forward(a, target, reduction, *, output=None, is_target=None):
    _reduction(reduction)
    frames, classes = _shape(a, target, multilabel=True)
    if (output is None) != (is_target is None):
        raise RuntimeError('multilabel margin loss requires output and is_target together')
    if not a.numel():
        if output is not None:
            return output, is_target
        return a.new_empty((0,)), a.new_empty((0,))
    _types(a, target)
    x, t = _logical(a).reshape(frames, classes), _logical(target).reshape(frames, classes)
    rows = torch.empty((frames,), dtype=a.dtype, device=a.device)
    mask = torch.empty((frames, classes), dtype=a.dtype, device=a.device)
    _C.scientific_call(363, [x, t, mask, rows], [reduction], [])
    result = rows if a.ndim == 2 and reduction == 0 else rows.sum() if a.ndim == 2 else rows.reshape(())
    return _outputs((result, mask.reshape(target.shape)), None if output is None else (output, is_target), (a, target))


def multilabel_backward(grad_output, a, target, reduction, is_target, *, grad_input=None):
    _reduction(reduction)
    frames, classes = _shape(a, target, multilabel=True)
    if not a.numel():
        return torch.zeros_like(a, memory_format=torch.contiguous_format) if grad_input is None else grad_input
    _types(a, target)
    if (is_target.layout != torch.strided or is_target.device != a.device or is_target.dtype != a.dtype
            or is_target.shape != target.shape):
        raise RuntimeError('multilabel margin loss is_target must match target shape and input dtype/device')
    x, t = _logical(a).reshape(frames, classes), _logical(target).reshape(frames, classes)
    mask = _logical(is_target).reshape(frames, classes)
    g = _grad(grad_output, a, frames, reduction)
    value = torch.empty((frames, classes), dtype=a.dtype, device=a.device)
    _C.scientific_call(364, [x, t, mask, g, value], [reduction], [])
    return _output(value.reshape(a.shape), grad_input, (grad_output, a, target, is_target))


def register(library):
    for name, function in (('multi_margin_loss', multi_margin), ('multi_margin_loss.out', multi_margin),
                           ('multi_margin_loss_backward', multi_margin_backward),
                           ('multi_margin_loss_backward.grad_input', multi_margin_backward),
                           ('multilabel_margin_loss_forward', multilabel_forward),
                           ('multilabel_margin_loss_forward.output', multilabel_forward),
                           ('multilabel_margin_loss_backward', multilabel_backward),
                           ('multilabel_margin_loss_backward.grad_input', multilabel_backward)):
        library.impl(name, function)
