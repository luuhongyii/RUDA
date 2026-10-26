"""Long-K native FP16 accumulation, including views and cancellation."""
import pytest
import torch
import ruda_torch


@pytest.mark.parametrize('transpose_a,transpose_b', [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize('shape', [(64, 8193, 128), (32, 16384, 256)])
def test_long_k_fp16_views_against_fp64(shape, transpose_a, transpose_b):
    torch.manual_seed(227)
    m, k, n = shape
    a_storage = torch.randn(k + 2, m, dtype=torch.float16) if transpose_a else torch.randn(m, k + 2, dtype=torch.float16)
    b_storage = torch.randn(n, k + 2, dtype=torch.float16) if transpose_b else torch.randn(k + 2, n, dtype=torch.float16)
    a_native, b_native = a_storage.to('ruda'), b_storage.to('ruda')
    a = a_storage[1:k + 1].t() if transpose_a else a_storage[:, 1:k + 1]
    b = b_storage[:, 1:k + 1].t() if transpose_b else b_storage[1:k + 1]
    x = a_native[1:k + 1].t() if transpose_a else a_native[:, 1:k + 1]
    y = b_native[:, 1:k + 1].t() if transpose_b else b_native[1:k + 1]
    reference = (a.double().cuda() @ b.double().cuda()).cpu()
    actual = (x @ y).cpu().double()
    torch.testing.assert_close(actual, reference.to(torch.float16).double(), rtol=1e-3, atol=5e-3)
    assert (actual - reference).norm() / reference.norm() < 3e-4


def test_long_k_fp16_cancellation():
    torch.manual_seed(229)
    a = torch.randn(32, 8192, dtype=torch.float16)
    b = torch.randn(8192, 128, dtype=torch.float16)
    a[:, 4096:] = a[:, :4096]
    b[4096:] = -b[:4096]
    expected = torch.zeros(32, 128, dtype=torch.float16)
    actual = (a.to('ruda') @ b.to('ruda')).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-4)
