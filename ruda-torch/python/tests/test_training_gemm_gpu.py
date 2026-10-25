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


@pytest.mark.parametrize('shape', [(32, 32, 32), (64, 96, 96), (96, 64, 160),
                                   (128, 160, 32), (160, 96, 96), (256, 256, 512),
                                   (64, 4096, 32), (128, 8192, 32)])
@pytest.mark.parametrize('ta,nt', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('precision', ['fp32', 'cuda'])
def test_ordered_bf16_tiles_transposed_left_and_narrow_output(shape, ta, nt, precision, monkeypatch):
    monkeypatch.setenv('RUDA_TORCH_BF16_PRECISION', precision)
    torch.manual_seed(261037)
    m, k, n = shape
    a = torch.randn(3, k, m + 8, dtype=torch.bfloat16) if ta else torch.randn(3, m, k + 8, dtype=torch.bfloat16)
    b = torch.randn(3, n, k + 8, dtype=torch.bfloat16) if nt else torch.randn(3, k, n + 8, dtype=torch.bfloat16)
    x, y = a.to('ruda')[1:, :, 8:], b.to('ruda')[1:, :, 8:]
    left, right = a[1:, :, 8:].double(), b[1:, :, 8:].double()
    if ta:
        x = x.transpose(1, 2); left = left.transpose(1, 2)
    if nt:
        y = y.transpose(1, 2); right = right.transpose(1, 2)
    expected = (left @ right).float()
    backing = torch.full((3, m, n), float('nan'), dtype=torch.float32, device='ruda')
    actual = backing[1:]
    r._C.execute(30, x, y, actual, 0.)
    values = actual.cpu()
    if k <= 512:
        torch.testing.assert_close(values, expected, rtol=1e-4, atol=2e-4)
        assert float((values.double() - expected.double()).norm() / expected.double().norm()) < 2e-6
    else:
        exact = left @ right
        unit = torch.finfo(torch.float32).eps
        bound = (k * unit / (1 - k * unit)) * (left.abs() @ right.abs())
        assert ((values.double() - exact).abs() <= bound).all()
        padded = torch.zeros(2, k, 64, dtype=torch.bfloat16)
        padded[:, :, :n] = right.bfloat16()
        reference = torch.empty(2, m, 64, dtype=torch.float32, device='ruda')
        r._C.execute(30, left.bfloat16().contiguous().to('ruda'), padded.to('ruda'), reference, 0.)
        torch.testing.assert_close(values, reference.cpu()[:, :, :n], rtol=0, atol=0)
    assert torch.isnan(backing[0].cpu()).all()
    for index in range(2):
        single = torch.empty(m, n, dtype=torch.float32, device='ruda')
        r._C.execute(7, x[index], y[index], single, 0.)
        torch.testing.assert_close(single.cpu(), values[index], rtol=0, atol=0)
    torch.testing.assert_close((x @ y).cpu(), values.bfloat16(), rtol=0, atol=0)


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


@pytest.mark.parametrize('m,k', [(16, 512), (48, 1536), (80, 1024)])
@pytest.mark.parametrize('transposed', [False, True])
def test_fp32_skinny_shared_tiles_with_unaligned_views(m, k, transposed):
    torch.manual_seed(261041)
    a = torch.randn(k, m + 1) if transposed else torch.randn(m, k + 1)
    b = torch.randn(k, 17)
    x, y = a.to('ruda')[:, 1:], b.to('ruda')[:, 1:]
    left = a[:, 1:].double()
    if transposed:
        x = x.t(); left = left.t()
    expected = (left @ b[:, 1:].double()).float()
    actual = (x @ y).cpu()
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=5e-5)
    assert float((actual.double() - expected.double()).norm() / expected.double().norm()) < 1e-6


def test_fp32_skinny_shared_tiles_nonfinite_values():
    a, b = torch.ones(48, 512), torch.ones(512, 16)
    a[0, 0], a[1, 0], a[2, 0] = float('inf'), float('-inf'), float('nan')
    b[0, 1] = 0
    torch.testing.assert_close((a.to('ruda') @ b.to('ruda')).cpu(), a @ b, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize('m,k,n', [(32, 64, 64), (64, 96, 32), (96, 128, 96)])
@pytest.mark.parametrize('layout', ['interleaved_left', 'interleaved_right', 'broadcast_left', 'broadcast_right'])
@pytest.mark.parametrize('precision', ['fp32', 'cuda'])
def test_bf16_tiles_readonly_overlapping_batches(m, k, n, layout, precision, monkeypatch):
    monkeypatch.setenv('RUDA_TORCH_BF16_PRECISION', precision)
    torch.manual_seed(261044)
    batch = 3
    a = torch.randn(1 if layout == 'broadcast_left' else batch, m, k + 8, dtype=torch.bfloat16)
    b = torch.randn(1 if layout == 'broadcast_right' else batch, k, n + 8, dtype=torch.bfloat16)
    left, right = a[:, :, 8:].expand(batch, m, k), b[:, :, 8:].expand(batch, k, n)
    x = a.to('ruda')[:, :, 8:].expand(batch, m, k)
    y = b.to('ruda')[:, :, 8:].expand(batch, k, n)
    if layout == 'interleaved_left':
        a = torch.randn(m, batch * k + 8, dtype=torch.bfloat16)
        shape, strides = (batch, m, k), (k, batch * k + 8, 1)
        left = a.as_strided(shape, strides, 8)
        x = a.to('ruda').as_strided(shape, strides, 8)
    if layout == 'interleaved_right':
        b = torch.randn(n, batch * k + 8, dtype=torch.bfloat16)
        shape, strides = (batch, k, n), (k, 1, batch * k + 8)
        right = b.as_strided(shape, strides, 8)
        y = b.to('ruda').as_strided(shape, strides, 8)
    expected = (left.double() @ right.double()).float()
    backing = torch.full((batch + 1, m, n), float('nan'), dtype=torch.float32, device='ruda')
    actual = backing[1:]
    r._C.execute(30, x, y, actual, 0.)
    values = actual.cpu()
    torch.testing.assert_close(values, expected, rtol=1e-4, atol=2e-4)
    assert float((values.double() - expected.double()).norm() / expected.double().norm()) < 2e-6
    assert torch.isnan(backing[0].cpu()).all()
    torch.testing.assert_close(x.cpu(), left, rtol=0, atol=0)
    torch.testing.assert_close(y.cpu(), right, rtol=0, atol=0)
    for index in range(batch):
        single = torch.empty(m, n, dtype=torch.float32, device='ruda')
        r._C.execute(7, x[index], y[index], single, 0.)
        torch.testing.assert_close(single.cpu(), values[index], rtol=0, atol=0)
    torch.testing.assert_close((x @ y).cpu(), values.bfloat16(), rtol=0, atol=0)
