import os

import pytest
import torch

if os.environ.get('RUDA_REQUIRE_GPU') != '1':
    pytest.skip('requires the native RUDA GPU backend', allow_module_level=True)

import ruda_torch


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('logarithmic', [False, True])
@pytest.mark.parametrize('shape,axis,policy', [
    ((7, 31), -1, 'auto'),
    ((7, 512), -1, 'auto'),
    ((1025, 512), -1, 'auto'),
    ((7, 513), -1, 'auto'),
    ((7, 5120), -1, 'auto'),
    ((7, 10240), -1, 'auto'),
    ((7, 32769), -1, 'auto'),
    ((2, 7, 64), 1, 'auto'),
    ((7, 257), -1, 'strided'),
    ((7, 777), -1, 'warp'),
    ((7, 777), -1, 'scalar'),
])
def test_softmax_fast_exp_matches_cuda(dtype, logarithmic, shape, axis, policy,
                                      monkeypatch, record_property):
    monkeypatch.setenv('RUDA_TORCH_SOFTMAX', 'auto' if policy == 'strided' else policy)
    torch.manual_seed(3261029)
    source = torch.randn(shape, dtype=dtype) * 3
    if len(shape) == 2:
        source[1, 0] = float('nan')
        source[2, 0] = float('inf')
        source[3] = -float('inf')
        source[4, ::2] = -float('inf')
        source[5] = torch.linspace(-100, 0, shape[-1]).to(dtype)
    else:
        source[0, 0, 0] = float('nan')
        source[0, 0, 1] = float('inf')
        source[0, :, 2] = -float('inf')
        source[0, ::2, 3] = -float('inf')
    if policy == 'strided':
        source = source.t().contiguous().t()
    native = source.to('ruda').requires_grad_()
    reference = source.cuda()
    operation = torch.log_softmax if logarithmic else torch.softmax
    actual, expected = operation(native, axis), operation(reference, axis)
    gradient = torch.randn(shape, dtype=dtype)
    actual.backward(gradient.to('ruda'))
    backward = (torch.ops.aten._log_softmax_backward_data if logarithmic
                else torch.ops.aten._softmax_backward_data)
    expected_gradient = backward(gradient.cuda(), expected, axis, dtype)
    tolerance = {torch.float32: 5e-6, torch.float16: .003, torch.bfloat16: .02}[dtype]
    for name, value, target in [('output', actual, expected), ('gradient', native.grad, expected_gradient)]:
        value, target = value.detach().cpu(), target.detach().cpu()
        assert torch.equal(torch.isnan(value), torch.isnan(target))
        assert torch.equal(torch.isposinf(value), torch.isposinf(target))
        assert torch.equal(torch.isneginf(value), torch.isneginf(target))
        finite = torch.isfinite(target)
        difference = (value[finite].float() - target[finite].float()).abs()
        record_property(name + '_max_abs_error', difference.max().item())
        torch.testing.assert_close(value, target, rtol=tolerance, atol=tolerance, equal_nan=True)
