import os

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(scope='module', autouse=True)
def backend():
    if os.environ.get('RUDA_REQUIRE_GPU') != '1':
        pytest.skip('requires native RUDA GPU backend')
    import ruda_torch
    return ruda_torch


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('policy', ['plain', 'causal', 'bool_mask', 'float_mask', 'float32_mask', 'gqa', 'strided'])
def test_eager_sdpa_forward_backward(dtype, policy):
    torch.manual_seed(3526)
    heads = 2 if policy == 'gqa' else 4
    host = [torch.randn(2, h, n, 32, dtype=dtype) for h, n in [(4, 17), (heads, 23), (heads, 23)]]
    if policy == 'strided':
        host = [x.transpose(-1, -2).contiguous().transpose(-1, -2) for x in host]
    mask = None
    if policy == 'bool_mask':
        mask = torch.rand(17, 23) > .2
        mask[0] = False
    if policy in ('float_mask', 'float32_mask'):
        mask = torch.randn(17, 23, dtype=torch.float32 if policy == 'float32_mask' else dtype)
        mask[0] = -float('inf')
    inputs = [x.to('ruda').requires_grad_() for x in host]
    refs = [x.float().requires_grad_() for x in host]
    options = dict(is_causal=policy == 'causal', enable_gqa=policy == 'gqa', scale=.17)
    actual = F.scaled_dot_product_attention(*inputs, attn_mask=None if mask is None else mask.to('ruda'), **options)
    expected = F.scaled_dot_product_attention(*refs, attn_mask=None if mask is None else mask if mask.dtype == torch.bool else mask.float(), **options)
    grad = torch.randn_like(host[0])
    actual.backward(grad.to('ruda'))
    expected.backward(grad.float())
    tolerance = .03 if dtype == torch.bfloat16 else .005 if dtype == torch.float16 else 2e-5
    for a, b in [(actual, expected), *[(a.grad, b.grad) for a, b in zip(inputs, refs)]]:
        torch.testing.assert_close(a.cpu().float(), b, atol=tolerance, rtol=tolerance, equal_nan=True)


@pytest.mark.parametrize('grad_enabled', [True, False])
def test_eager_sdpa_routes_native_with_dropout(backend, monkeypatch, grad_enabled):
    from ruda_torch import attention
    original = attention._fused_dense
    calls = []
    def record(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(attention, '_fused_dense', record)
    values = [torch.randn(1, 2, 17, 32).to('ruda') for _ in range(3)]
    with torch.set_grad_enabled(grad_enabled):
        backend.manual_seed(3527)
        actual = F.scaled_dot_product_attention(*values, dropout_p=.25)
        assert len(calls) == 1
        backend.manual_seed(3527)
        expected = backend.scaled_dot_product_attention(*values, dropout_p=.25)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)


@pytest.mark.parametrize('shape', [(5, 16), (1, 2, 0, 16)])
def test_eager_sdpa_math_compatibility(shape):
    values = [torch.randn(shape) for _ in range(3)]
    actual = F.scaled_dot_product_attention(*(x.to('ruda') for x in values))
    expected = F.scaled_dot_product_attention(*values)
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-5, rtol=2e-5)


@torch.inference_mode()
def test_tiled_sdpa_logical_scores_exceed_u32():
    torch.manual_seed(3530)
    shape = (4, 32, 6144, 128)
    assert shape[0] * shape[1] * shape[2] ** 2 > 2**32
    values = [torch.randn(shape, dtype=torch.float16, device='cuda') for _ in range(3)]
    expected = F.scaled_dot_product_attention(*values, is_causal=True).cpu()
    actual = F.scaled_dot_product_attention(*(x.cpu().to('ruda') for x in values), is_causal=True)
    torch.testing.assert_close(actual.cpu(), expected, atol=.005, rtol=.005)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('width', [512, 777, 4096, 4097, 10240, 16384, 16385])
@pytest.mark.parametrize('affine', [False, True])
def test_cached_row_norms(dtype, width, affine):
    torch.manual_seed(3528)
    x = torch.randn(5, width, dtype=dtype)
    weight = torch.randn(width, dtype=dtype) if affine else None
    bias = torch.randn(width, dtype=dtype) if affine else None
    rx = x.to('ruda')
    rw, rb = [None if t is None else t.to('ruda') for t in (weight, bias)]
    tolerance = .02 if dtype == torch.bfloat16 else .003 if dtype == torch.float16 else 5e-6
    for actual, expected in [
        (F.layer_norm(rx, (width,), rw, rb), F.layer_norm(x, (width,), weight, bias)),
        (F.rms_norm(rx, (width,), rw, 1e-6), F.rms_norm(x, (width,), weight, 1e-6)),
    ]:
        torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('width', [512, 777, 4096, 4097, 10240, 16384, 16385])
@pytest.mark.parametrize('logarithmic', [False, True])
def test_cached_row_softmax(dtype, width, logarithmic):
    torch.manual_seed(3529)
    x = torch.randn(7, width, dtype=dtype)
    x[1, 0] = float('nan')
    x[2, 0] = float('inf')
    x[3] = -float('inf')
    x[4, 0] = -float('inf')
    operation = F.log_softmax if logarithmic else F.softmax
    inputs = x.to('ruda').requires_grad_()
    reference = x.requires_grad_()
    actual, expected = operation(inputs, -1), operation(reference, -1)
    tolerance = .02 if dtype == torch.bfloat16 else .003 if dtype == torch.float16 else 5e-6
    torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance, equal_nan=True)
    gradient = torch.randn_like(x)
    actual.backward(gradient.to('ruda'))
    expected.backward(gradient)
    torch.testing.assert_close(inputs.grad.cpu(), reference.grad, atol=tolerance, rtol=tolerance, equal_nan=True)
