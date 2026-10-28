from functools import lru_cache, partial

import torch
from torch._prims_common import elementwise_dtypes, ELEMENTWISE_TYPE_PROMOTION_KIND
from . import _C
from ._ops import _dtypes
from ._ordering import _output, _outputs


def _reduction(reduction):
    if reduction not in (0, 1, 2):
        raise RuntimeError('loss reduction must be none, mean or sum')


def _inputs(a, target, grad=None, *, promote=False, compute_dtype=None):
    tensors = (a, target) if grad is None else (a, target, grad)
    if a.dtype not in _dtypes or any(t.device != a.device for t in tensors):
        raise RuntimeError('loss inputs must use a real floating input on the same device')
    dtype = elementwise_dtypes(*tensors, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)[1] if promote else a.dtype
    if not promote and any(t.dtype != dtype for t in tensors):
        raise RuntimeError('loss inputs must have the same dtype')
    if dtype not in _dtypes:
        raise RuntimeError('loss requires real floating computation')
    if compute_dtype is not None:
        if compute_dtype not in _dtypes or not torch.can_cast(dtype, compute_dtype):
            raise RuntimeError('loss backward output cannot safely receive the result')
        dtype = compute_dtype
    return torch.broadcast_tensors(*(t.to(dtype).resolve_neg().resolve_conj() for t in tensors))


def _pointwise(a, target, grad=None, *, op, boundary=0., norm=1.):
    result = torch.empty_like(a)
    _C.scientific_call(op, [a, target, a if grad is None else grad, result], [], [float(boundary), float(norm)])
    return result


def _reduce(value, reduction, *, dtype=None):
    return value.mean(dtype=dtype) if reduction == 1 else value.sum(dtype=dtype) if reduction == 2 else value


def regression(a, target, reduction=1, boundary=1., *, op, out=None):
    _reduction(reduction)
    if op == 295 and not boundary >= 0:
        raise RuntimeError('smooth_l1_loss beta must be nonnegative')
    if op == 297 and not boundary > 0:
        raise RuntimeError('huber_loss delta must be positive')
    x, y = _inputs(a, target, promote=True)
    value = _pointwise(x, y, op=op, boundary=boundary)
    if op == 297:
        value = value.to(a.dtype if out is None else out.dtype)
    return _output(_reduce(value, reduction, dtype=out.dtype if out is not None else None), out, (a, target), cast=True)


@lru_cache(maxsize=1)
def _mse_backward_promotes():
    # MSE's shared TensorIterator contract changed between supported Torch builds.
    x = torch.empty(0, dtype=torch.float32, device='cpu')
    y = torch.empty(0, dtype=torch.float64, device='cpu')
    try:
        torch.ops.aten.mse_loss_backward.default(x, x, y, 2)
    except RuntimeError:
        return False
    return True


def regression_backward(grad, a, target, reduction, boundary=1., *, op, grad_input=None):
    _reduction(reduction)
    promote = op != 298 and (op != 294 or _mse_backward_promotes())
    if not promote and grad_input is not None and grad_input.dtype != a.dtype:
        raise RuntimeError('loss backward output must match the input dtype')
    compute_dtype = (a.dtype if grad_input is None else grad_input.dtype) if op == 294 and promote else None
    x, y, g = _inputs(a, target, grad, promote=promote, compute_dtype=compute_dtype)
    norm = 2. if op == 294 else 1.
    if reduction == 1:
        norm = norm / a.numel() if a.numel() else float('inf')
    value = _pointwise(x, y, g, op=op, boundary=boundary, norm=norm)
    if grad_input is None:
        value = value.to(a.dtype)
    return _output(value, grad_input, (a, target, grad), cast=True)


def _bce_inputs(a, target, grad=None):
    # CUDA BCE's TensorIterator operates on squeezed operands.
    tensors = _inputs(a.squeeze(), target.squeeze(), promote=False)
    if tensors[0].numel() != a.numel():
        raise RuntimeError('BCE target cannot expand the input shape')
    if grad is None:
        return tensors
    g = grad.expand_as(a).squeeze()
    return _inputs(*tensors, g)


def binary_cross_entropy(a, target, weight=None, reduction=1, *, out=None):
    _reduction(reduction)
    x, y = _bce_inputs(a, target)
    value = _pointwise(x, y, op=299).reshape(a.shape)
    if weight is not None:
        value.mul_(weight)
    return _output(_reduce(value, reduction), out, (a, target, weight))


def binary_cross_entropy_backward(grad, a, target, weight=None, reduction=1, *, grad_input=None):
    _reduction(reduction)
    x, y, g = _bce_inputs(a, target, grad)
    value = _pointwise(x, y, g, op=300).reshape(a.shape)
    if weight is not None:
        value.mul_(weight)
    if reduction == 1:
        value.div_(a.numel())
    return _output(value, grad_input, (a, target, grad, weight))


def soft_margin_loss(a, target, reduction=1, *, out=None):
    _reduction(reduction)
    if out is not None and out.dtype != a.dtype:
        raise RuntimeError('soft_margin_loss output must match the input dtype')
    x, y = _inputs(a, target, promote=True)
    if a.dtype == target.dtype:
        value = _pointwise(x, y, op=301)
    else:
        value = (-a).mul_(target).exp_().log1p_()
    return _output(_reduce(value, reduction), out, (a, target))


def soft_margin_loss_backward(grad, a, target, reduction, *, grad_input=None):
    _reduction(reduction)
    x, y, g = _inputs(a, target, grad, promote=True)
    norm = (1. / a.numel() if a.numel() else float('inf')) if reduction == 1 else 1.
    if a.dtype == target.dtype == grad.dtype:
        value = _pointwise(x, y, g, op=302, norm=norm)
    else:
        z = (-target * a).exp()
        value = (target * z).to(a.dtype).mul_(-norm)
        value.div_(z.add_(1)).mul_(grad)
    return _output(value, grad_input, (a, target, grad), cast=True)


def _nll_inputs(a, target, weight, reduction, spatial):
    _reduction(reduction)
    if a.dtype not in _dtypes or target.device != a.device:
        raise RuntimeError('NLL input must be real floating and target on the same device')
    if spatial:
        if a.ndim != 4 or target.ndim != 3 or target.shape != (a.shape[0], *a.shape[2:]):
            raise RuntimeError('NLL2d requires matching (N,C,H,W) input and (N,H,W) target')
        if target.dtype != torch.int64:
            raise RuntimeError('NLL2d target must be int64')
        mode, classes = 2, a.shape[1]
        shape = (a.shape[0], classes, a.shape[2] * a.shape[3])
    else:
        if a.ndim not in (1, 2) or target.ndim > 1 or target.numel() != (1 if a.ndim == 1 else a.shape[0]):
            raise RuntimeError('NLL requires 1D/2D input and matching scalar/1D target')
        if a.ndim == 2 and target.ndim != 1:
            raise RuntimeError('batched NLL requires a 1D target')
        if target.dtype not in (torch.int64, torch.uint8):
            raise RuntimeError('NLL target must be int64 or uint8')
        mode, classes = a.ndim - 1, a.shape[-1]
        shape = (1 if a.ndim == 1 else a.shape[0], classes, 1)
    if weight is not None:
        if weight.device != a.device or weight.dtype != a.dtype or weight.numel() != classes or (not spatial and weight.ndim != 1):
            raise RuntimeError('NLL weight must match all classes and input dtype/device')
        w = weight.resolve_neg().reshape(-1)
    else:
        w = a.new_empty((0,))
    return a.resolve_neg().reshape(shape), target.to(torch.int64).reshape(-1), w, mode


def nll_forward(a, target, weight, reduction, ignore_index, *, spatial=False, output=None, total_weight=None):
    x, t, w, mode = _nll_inputs(a, target, weight, reduction, spatial)
    shape = target.shape if reduction == 0 and mode != 0 else ()
    # Every NLL forward kernel writes total_weight, so no zero-fill launch is needed.
    result, total = a.new_empty(shape), a.new_empty(())
    _C.scientific_call(303, [x, t, w, result, total], [reduction, ignore_index, mode, int(weight is not None)], [])
    if (output is None) != (total_weight is None):
        raise RuntimeError('NLL output and total_weight must be provided together')
    return _outputs((result, total), None if output is None else (output, total_weight), (a, target, weight))


def nll_backward(grad, a, target, weight, reduction, ignore_index, total_weight, *, spatial=False, grad_input=None):
    x, t, w, mode = _nll_inputs(a, target, weight, reduction, spatial)
    if total_weight.device != a.device or total_weight.dtype != a.dtype or total_weight.numel() != 1:
        raise RuntimeError('NLL total_weight must be a single element of the input dtype/device')
    expected = target.shape if reduction == 0 and mode != 0 else ()
    if grad.device != a.device or grad.dtype != a.dtype or (grad.shape != expected and not (expected == () and grad.shape == (1,))):
        raise RuntimeError('NLL grad_output shape, device or dtype mismatch')
    value = torch.zeros_like(x, memory_format=torch.contiguous_format)
    _C.scientific_call(304, [x, t, w, grad.resolve_neg(), total_weight.resolve_neg(), value],
                       [reduction, ignore_index, mode, int(weight is not None)], [])
    return _output(value.reshape(a.shape), grad_input, (grad, a, target, weight, total_weight))


def register(library):
    for name, op in {'mse_loss': 293, 'smooth_l1_loss': 295, 'huber_loss': 297}.items():
        fn = partial(regression, op=op)
        library.impl(name, fn)
        library.impl(name + '.out', fn)
        back = partial(regression_backward, op=op + 1)
        library.impl(name + '_backward', back)
        library.impl(name + '_backward' + ('.out' if name == 'huber_loss' else '.grad_input'), back)
    for name, fn, back in [('binary_cross_entropy', binary_cross_entropy, binary_cross_entropy_backward),
                           ('soft_margin_loss', soft_margin_loss, soft_margin_loss_backward)]:
        library.impl(name, fn)
        library.impl(name + '.out', fn)
        library.impl(name + '_backward', back)
        library.impl(name + '_backward.grad_input', back)
    for name, spatial in [('nll_loss', False), ('nll_loss2d', True)]:
        library.impl(name + '_forward', partial(nll_forward, spatial=spatial))
        library.impl(name + '_forward.output', partial(nll_forward, spatial=spatial))
        library.impl(name + '_backward', partial(nll_backward, spatial=spatial))
        library.impl(name + '_backward.grad_input', partial(nll_backward, spatial=spatial))
