from functools import partial

import torch

from . import _C


_floating = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _groups(groups, *, allow_mixed=False, warn_no_momentum=False):
    mixed = _C.fused_optimizer_check(groups, allow_mixed, warn_no_momentum)
    if groups[0][0].dtype not in _floating:
        raise RuntimeError('fused optimizers require real floating parameters')
    if mixed and (groups[0][0].dtype != torch.float32 or groups[1][0].dtype != torch.float32
                  or any(group[0].dtype != torch.bfloat16 for group in groups[2:])):
        raise RuntimeError('mixed fused Adam requires float32 parameters/gradients and bfloat16 states')


def _device_scalar(tensor, device, name):
    if (tensor.device != device or tensor.layout != torch.strided
            or tensor.dtype != torch.float32 or not tensor.numel()):
        raise RuntimeError(f'{name} requires nonempty float32 storage on the parameter device')


def _controls(lr, grad_scale, found_inf, device):
    controls, flags = [], 0
    if isinstance(lr, torch.Tensor) and lr.device.type == 'cpu':
        lr = _C.fused_optimizer_cpu_lr(lr)
    if isinstance(lr, torch.Tensor):
        _device_scalar(lr, device, 'lr')
        controls.append(lr)
        flags |= 1
        lr = 0.
    for bit, name, value in ((2, 'grad_scale', grad_scale), (4, 'found_inf', found_inf)):
        if value is not None:
            _device_scalar(value, device, name)
            controls.append(value)
            flags |= bit
    return float(lr), controls, flags


def _steps(steps, params):
    if len(steps) != len(params):
        raise RuntimeError('fused optimizer requires one state step per parameter')
    for step in steps:
        _device_scalar(step, params[0].device, 'state_steps')


def _run(op, groups, steps, controls, flags, options, values):
    with torch.no_grad():
        targets = [[value.resolve_conj().resolve_neg() for value in group] for group in groups]
        step_inputs = [value.resolve_neg() for value in steps]
        control_inputs = [value.resolve_neg() for value in controls]
        tensors = [value for group in targets for value in group] + step_inputs + control_inputs
        _C.scientific_call(op, tensors, [len(groups[0]), flags, *options], values)
        for field, (group, target_group) in enumerate(zip(groups, targets)):
            if field == 1 and not flags & 2:
                continue
            for value, target in zip(group, target_group):
                if target is not value:
                    value.copy_(target)


def adam_(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, *,
          lr, beta1, beta2, weight_decay, eps, amsgrad, maximize,
          grad_scale=None, found_inf=None, decoupled=False):
    groups = [params, grads, exp_avgs, exp_avg_sqs]
    if amsgrad:
        groups.append(max_exp_avg_sqs)
    _groups(groups, allow_mixed=True)
    _steps(state_steps, params)
    lr, controls, flags = _controls(lr, grad_scale, found_inf, params[0].device)
    _run(437 if decoupled else 436, groups, state_steps, controls, flags,
         [int(maximize), int(amsgrad), 0, 0],
         [lr, float(beta1), float(beta2), float(weight_decay), float(eps)])


def sgd_(params, grads, momentum_buffer_list, *, weight_decay, momentum, lr, dampening,
         nesterov, maximize, is_first_step, grad_scale=None, found_inf=None):
    momentum = float(momentum)
    with_momentum = bool(momentum_buffer_list)
    if (with_momentum and not momentum > 0) or (not with_momentum and momentum != 0):
        raise RuntimeError('fused SGD requires positive momentum with buffers, or zero momentum without buffers')
    groups = [params, grads]
    if with_momentum:
        groups.append(momentum_buffer_list)
    _groups(groups, warn_no_momentum=is_first_step and not with_momentum)
    lr, controls, flags = _controls(lr, grad_scale, found_inf, params[0].device)
    _run(438, groups, [], controls, flags,
         [int(maximize), int(with_momentum), int(nesterov), int(is_first_step)],
         [lr, float(weight_decay), momentum, float(dampening)])


def adagrad_(params, grads, state_sums, state_steps, *, lr, lr_decay, weight_decay, eps,
             maximize, grad_scale=None, found_inf=None):
    groups = [params, grads, state_sums]
    _groups(groups)
    _steps(state_steps, params)
    lr, controls, flags = _controls(lr, grad_scale, found_inf, params[0].device)
    _run(439, groups, state_steps, controls, flags, [int(maximize), 0, 0, 0],
         [lr, float(lr_decay), float(weight_decay), float(eps)])


def register(library):
    for name, function in (
        ('_fused_adam_', adam_),
        ('_fused_adamw_', partial(adam_, decoupled=True)),
        ('_fused_sgd_', sgd_),
        ('_fused_adagrad_', adagrad_),
    ):
        library.impl(name, function)
        library.impl(name + '.tensor_lr', function)
