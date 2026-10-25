"""Actual device coverage for fixed-layout BF16 and strict-FP32 LoRA GEMMs."""
import pytest
import torch
import ruda_torch as r


@pytest.mark.parametrize('nt', [False, True])
@pytest.mark.parametrize('padded', [False, True])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_bf16_training_tile_layout(nt, padded, dtype, monkeypatch):
    monkeypatch.setenv('RUDA_TORCH_BF16_PRECISION', 'cuda')
    torch.manual_seed(261025)
    m, k, n = 512, 96, 256
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
    torch.testing.assert_close(host, reference, rtol=2e-3 if dtype == torch.bfloat16 else 2e-5,
                               atol=.125 if dtype == torch.bfloat16 else 2e-5)


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
