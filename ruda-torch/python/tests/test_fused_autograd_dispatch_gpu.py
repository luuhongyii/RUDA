import importlib
import sys
import pytest
import torch
import ruda_torch as r
from ruda_torch import _fused_training_ops as training, fused_cross_entropy as ce, rotary

lora = importlib.import_module('ruda_torch.lora_epilogue')
NAMES = ['training_rms_forward', 'training_rms_backward', 'training_layer_forward',
         'training_layer_backward', 'training_silu_forward', 'training_silu_backward',
         'rotary_qk_forward', 'rotary_qk_backward', 'cross_entropy_forward',
         'cross_entropy_backward', 'lora_epilogue_forward', 'lora_epilogue_backward', 'lora_rank16_residual']


def case(name, requires_grad):
    def floating(*shape, dtype=torch.float32):
        return torch.randn(*shape, dtype=dtype).to('ruda').requires_grad_(requires_grad)
    x, w, dy = floating(16, 128), floating(128), floating(16, 128)
    if name.startswith('training_rms'):
        args = (x, w, 1e-5, False)
        if name.endswith('forward'): return training.rms_forward, args
        with torch.no_grad(): stats = training.rms_forward(*args)[1]
        return training.rms_backward, (x, w, dy, stats, True, True, False)
    if name.startswith('training_layer'):
        args = (x, w, None, 1e-5)
        if name.endswith('forward'): return training.layer_forward, args
        with torch.no_grad(): _, mean, stats = training.layer_forward(*args)
        return training.layer_backward, (x, w, None, dy, mean, stats, True, True, False)
    if name.startswith('training_silu'):
        if name.endswith('forward'): return training.silu_forward, (x, dy)
        return training.silu_backward, (x, dy, floating(16, 128), True, True)
    if name.startswith('rotary'):
        q, k = floating(1, 2, 16, 128), floating(1, 1, 16, 128)
        cos, sin = floating(1, 1, 16, 128), floating(1, 1, 16, 128)
        if name.endswith('forward'): return rotary._forward, (q, k, cos, sin)
        return rotary._backward, (q, None, cos, sin, list(q.shape), list(k.shape))
    if name.startswith('cross_entropy'):
        target = torch.arange(16, dtype=torch.int64, device='ruda')
        if name.endswith('forward'): return ce._forward, (x, target, -100)
        with torch.no_grad(): stats = ce._forward(x, target, -100)[1]
        return ce._backward, (x, target, stats, floating(16), -100)
    base = floating(16, 128, dtype=torch.bfloat16)
    if name == 'lora_epilogue_forward': return lora._forward, (base, x, .5)
    if name == 'lora_epilogue_backward': return lora._backward, (base, .5)
    return lora._rank16_residual, (base, floating(16, 16), floating(128, 16), .5)


def same(actual, expected):
    if isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected): same(a, b)
    elif expected is None: assert actual is None
    else:
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.stride() == expected.stride()
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('grad_enabled,requires_grad', [(False, True), (True, False), (True, True)])
def test_fused_autograd_dispatch_skips_only_unneeded_python_autograd(name, grad_enabled, requires_grad):
    table = torch._C._dispatch_dump_table('ruda::' + name)
    entry = next(line for line in table.splitlines() if line.startswith('AutogradPrivateUse1:'))
    assert 'fused_autograd_dispatch.inc' in entry
    fn, args = case(name, requires_grad)
    calls = []
    def profile(frame, event, arg):
        if (event == 'call' and frame.f_code.co_name == 'autograd_impl'
                and frame.f_code.co_filename.replace('\\', '/').endswith('torch/_library/autograd.py')):
            calls.append(frame.f_code)
    previous = sys.getprofile()
    with torch.set_grad_enabled(grad_enabled):
        sys.setprofile(profile)
        try: actual = fn(*args)
        finally: sys.setprofile(previous)
    with torch.no_grad(): expected = fn._init_fn(*args)
    same(actual, expected)
    assert bool(calls) == (grad_enabled and requires_grad)


@pytest.mark.parametrize('kind', ['rms', 'layer', 'silu'])
def test_fused_autograd_dispatch_preserves_missing_formula_errors(kind):
    fn, args = case('training_' + kind + '_forward', True)
    output = fn(*args)
    if isinstance(output, tuple): output = output[0]
    with pytest.raises(RuntimeError, match='no autograd formula was registered'):
        output.sum().backward()


def test_fused_autograd_dispatch_preserves_registered_formula_updates():
    original_backward = lora._forward._backward_fn
    original_setup = lora._forward._setup_context_fn
    calls = []
    def observed(ctx, grad):
        calls.append(1)
        return original_backward(ctx, grad)
    lora._forward.register_autograd(observed, setup_context=original_setup)
    try:
        base = torch.ones(3, 8, dtype=torch.bfloat16, device='ruda').requires_grad_()
        update = torch.ones(3, 8, device='ruda').requires_grad_()
        output = lora._forward(base, update, .5)
        a, b = torch.autograd.grad(output.float().sum(), (base, update), create_graph=True)
        torch.testing.assert_close(a.cpu(), torch.ones_like(a.cpu()), rtol=0, atol=0)
        torch.testing.assert_close(b.cpu(), torch.full_like(b.cpu(), .5), rtol=0, atol=0)
        assert calls == [1]
    finally:
        lora._forward.register_autograd(original_backward, setup_context=original_setup)


def test_fused_autograd_dispatch_keeps_torch_dispatch_modes_and_inference_mode():
    from torch.utils._python_dispatch import TorchDispatchMode
    calls = []
    class Observe(TorchDispatchMode):
        def __torch_dispatch__(self, fn, types, args=(), kwargs=None):
            if str(fn) == 'ruda.training_silu_forward.default': calls.append(str(fn))
            return fn(*args, **(kwargs or {}))
    fn, args = case('training_silu_forward', True)
    with torch.no_grad(), Observe(): actual = fn(*args)
    with torch.no_grad(): same(actual, fn._init_fn(*args))
    assert calls == ['ruda.training_silu_forward.default']
    with torch.inference_mode():
        inferred = fn(*args)
        assert inferred.is_inference() and not inferred.requires_grad
    same(inferred, actual)
