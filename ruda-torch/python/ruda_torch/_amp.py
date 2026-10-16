import operator

import torch

from . import _C


def _scalar(tensor, dtype, name):
    if (tensor.device.type != 'ruda' or tensor.layout != torch.strided
            or tensor.dtype != dtype or tensor.numel() != 1):
        raise RuntimeError(f'{name} must be a one-element strided RUDA {dtype} tensor')


def _logical(tensor):
    return tensor.resolve_conj().resolve_neg()


def unscale_(gradients, found_inf, inv_scale):
    if not gradients:
        return
    _scalar(found_inf, torch.float32, 'found_inf')
    _scalar(inv_scale, torch.float32, 'inv_scale')
    _C.check_inplace(found_inf, found_inf)
    for grad in gradients:
        if (grad.device != found_inf.device or grad.layout != torch.strided
                or grad.dtype not in (torch.float16, torch.float32, torch.float64)):
            raise RuntimeError('AMP unscale requires strided RUDA float16, float32 or float64 gradients')
        _C.check_inplace(grad, grad)
    if inv_scale.device != found_inf.device:
        raise RuntimeError('AMP scale and flag must be on the gradient device')
    with torch.no_grad():
        targets = [_logical(grad) for grad in gradients]
        flag = _logical(found_inf)
        inverse = _logical(inv_scale)
        _C.scientific_call(434, [*targets, flag, inverse], [], [])
        for grad, target in zip(gradients, targets):
            if target is not grad:
                grad.copy_(target)
        if flag is not found_inf:
            found_inf.copy_(flag)


def update_scale_(scale, growth_tracker, found_inf,
                  scale_growth_factor, scale_backoff_factor, growth_interval):
    _scalar(scale, torch.float32, 'scale')
    _scalar(growth_tracker, torch.int32, 'growth_tracker')
    _scalar(found_inf, torch.float32, 'found_inf')
    if scale.device != growth_tracker.device or scale.device != found_inf.device:
        raise RuntimeError('AMP scale, tracker and flag must be on the same device')
    _C.check_inplace(scale, scale)
    _C.check_inplace(growth_tracker, growth_tracker)
    with torch.no_grad():
        target = _logical(scale)
        tracker = _logical(growth_tracker)
        flag = _logical(found_inf)
        _C.scientific_call(435, [target, tracker, flag], [operator.index(growth_interval)],
                           [float(scale_growth_factor), float(scale_backoff_factor)])
        if target is not scale:
            scale.copy_(target)
        if tracker is not growth_tracker:
            growth_tracker.copy_(tracker)
    return scale


def register(library):
    library.impl('_amp_foreach_non_finite_check_and_unscale_', unscale_)
    library.impl('_amp_update_scale_', update_scale_)
