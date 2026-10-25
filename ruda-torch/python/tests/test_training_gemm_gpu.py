"""Actual device coverage for fixed-layout BF16 and strict-FP32 LoRA GEMMs."""
import pytest
import torch
import ruda_torch as r


@pytest.mark.parametrize('nt', [False, True])
@pytest.mark.parametrize('padded', [False, True])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('k', [32, 64, 96, 160])
@pytest.mark.parametrize('m', [32, 512])
def test_bf16_training_tile_layout(nt, padded, dtype, k, m, monkeypatch):
    monkeypatch.setenv('RUDA_TORCH_BF16_PRECISION', 'cuda')
    torch.manual_seed(261025)
    n = 256
    pad = 8 if padded else 0
    a = torch.randn(m, k + pad, dtype=torch.bfloat16)
    b = torch.randn(n, k + pad, dtype=torch.bfloat16) if nt else torch.randn(k, n + pad, dtype=torch.bfloat16)
    x = a.to('ruda')[:, pad:]
    weight = b.to('ruda')[:, pad:]
    if nt:
        weight = weight.t()
    actual = torch.empty(m, n, dtype=dtype, device='ruda')
    r._C.execute(7, x, weight, actual, 0.0)
    left = a[:, pad:].double()
    right = b[:, pad:].double()
    reference = (left @ (right.t() if nt else right)).to(dtype)
    host = actual.cpu()
    tolerance = 2e-5 if dtype == torch.float32 else 5e-3
    assert float((host.double() - reference.double()).norm() / reference.double().norm()) < tolerance
    if dtype == torch.bfloat16:
        accumulator = torch.empty(m, n, dtype=torch.float32, device='ruda')
        r._C.execute(7, x, weight, accumulator, 0.0)
        accumulator = accumulator.cpu()
        torch.testing.assert_close(host, accumulator.bfloat16(), rtol=0, atol=0)
        host = accumulator
        reference = (left @ (right.t() if nt else right)).float()
    torch.testing.assert_close(host, reference, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('m', [32, 512])
@pytest.mark.parametrize('nt', [False, True])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_batched_training_tile_preserves_gapped_strides(batch, m, nt, dtype, monkeypatch):
    monkeypatch.setenv('RUDA_TORCH_BF16_PRECISION', 'cuda')
    torch.manual_seed(261031)
    k, n, pad = 96, 256, 8
    a = torch.randn(batch+1, m, k+pad, dtype=torch.bfloat16)
    b = torch.randn(batch+1, n, k+pad, dtype=torch.bfloat16) if nt else torch.randn(batch+1, k, n+pad, dtype=torch.bfloat16)
    x, weight = a.to('ruda')[1:, :, pad:], b.to('ruda')[1:, :, pad:]
    left, right = a[1:, :, pad:].double(), b[1:, :, pad:].double()
    if nt:
        weight = weight.transpose(1, 2); right = right.transpose(1, 2)
    out = torch.empty(batch+1, m, n, dtype=dtype, device='ruda')[1:]
    r._C.execute(30, x, weight, out, 0.)
    reference = left @ right
    actual = out.cpu()
    if dtype == torch.bfloat16:
        accum = torch.empty(batch, m, n, dtype=torch.float32, device='ruda')
        r._C.execute(30, x, weight, accum, 0.)
        accum = accum.cpu()
        torch.testing.assert_close(actual, accum.bfloat16(), rtol=0, atol=0)
        actual = accum
    torch.testing.assert_close(actual, reference.float(), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize('shape', [(32, 32, 64), (64, 96, 192), (96, 64, 384),
                                   (128, 160, 64), (160, 96, 192), (256, 256, 512),
                                   (768, 64, 128), (1024, 96, 256)])
@pytest.mark.parametrize('nt', [False, True])
@pytest.mark.parametrize('precision', ['fp32', 'cuda'])
def test_ordered_bf16_tiles_general_shapes(shape, nt, precision, monkeypatch):
    monkeypatch.setenv('RUDA_TORCH_BF16_PRECISION', precision)
    torch.manual_seed(261034)
    m, k, n = shape
    a = torch.randn(2, m, k + 8, dtype=torch.bfloat16)
    b = torch.randn(2, n, k + 8, dtype=torch.bfloat16) if nt else torch.randn(2, k, n + 8, dtype=torch.bfloat16)
    x, y = a.to('ruda')[:, :, 8:], b.to('ruda')[:, :, 8:]
    left, right = a[:, :, 8:].double(), b[:, :, 8:].double()
    if nt:
        y = y.transpose(1, 2); right = right.transpose(1, 2)
    expected = (left @ right).float()
    actual = torch.empty(2, m, n, dtype=torch.float32, device='ruda')
    r._C.execute(30, x, y, actual, 0.)
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-5)
    for index in range(2):
        single = torch.empty(m, n, dtype=torch.float32, device='ruda')
        r._C.execute(7, x[index], y[index], single, 0.)
        torch.testing.assert_close(single.cpu(), actual[index].cpu(), rtol=0, atol=0)
    torch.testing.assert_close((x @ y).cpu(), actual.cpu().bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize('m,n', [(16, 128), (32, 3584), (512, 18944)])
@pytest.mark.parametrize('offset', [0, 1])
def test_fp32_rank16_projection(m, n, offset):
    torch.manual_seed(261026)
    a = torch.randn(m, 16)
    backing = torch.randn(n * 16 + offset)
    b = backing[offset:].reshape(n, 16)
    x = a.to('ruda')
    y = backing.to('ruda')[offset:].reshape(n, 16).t()
    reference = (a.double() @ b.double().t()).float()
    actual = (x @ y).cpu()
    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)
    assert float((actual.double() - reference.double()).norm() / reference.double().norm()) < 1e-6


def test_fp32_rank16_nonfinite_values():
    a = torch.zeros(16, 16)
    b = torch.ones(128, 16)
    a[0, 0] = float('nan')
    b[1, 1] = float('inf')
    b[2, 2] = float('-inf')
    torch.testing.assert_close((a.to('ruda') @ b.to('ruda').t()).cpu(), a @ b.t(), equal_nan=True)


@pytest.mark.parametrize('m,k', [(16, 512), (64, 1024), (3584, 1536)])
@pytest.mark.parametrize('padded', [False, True])
@pytest.mark.parametrize('transposed', [False, True])
def test_fp32_transposed_lora_gradient(m, k, padded, transposed):
    torch.manual_seed(261027)
    pad = 4 if padded else 0
    a = torch.randn(k, m + pad) if transposed else torch.randn(m, k + pad)
    b = torch.randn(k, 16 + pad)
    x, y = a.to('ruda')[:, pad:], b.to('ruda')[:, pad:]
    left = a[:, pad:].double()
    if transposed:
        x = x.t(); left = left.t()
    reference = (left @ b[:, pad:].double()).float()
    actual = (x @ y).cpu()
    torch.testing.assert_close(actual, reference, rtol=2e-4, atol=5e-5)
    assert float((actual.double() - reference.double()).norm() / reference.double().norm()) < 1e-6


@pytest.mark.parametrize('m,k', [(4, 513), (32, 1536), (512, 3584)])
@pytest.mark.parametrize('padded', [False, True])
def test_fp32_lora_projection_reuses_input(m, k, padded):
    torch.manual_seed(261029)
    pad = 3 if padded else 0
    a = torch.randn(m, k + pad)
    b = torch.randn(16, k + pad)
    x, y = a.to('ruda')[:, pad:], b.to('ruda')[:, pad:].t()
    reference = (a[:, pad:].double() @ b[:, pad:].double().t()).float()
    actual = (x @ y).cpu()
    torch.testing.assert_close(actual, reference, rtol=2e-4, atol=5e-5)
    assert float((actual.double() - reference.double()).norm() / reference.double().norm()) < 1e-6
