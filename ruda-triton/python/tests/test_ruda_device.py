"""Triton kernels on the RUDA PrivateUse1 device (ruda_torch), interleaved with RUDA's own kernels.

Run in its own process: RUDA_TRITON_DEVICE=ruda selects the RUDA device for every Triton kernel.
"""
import os

os.environ["TRITON_DEFAULT_BACKEND"] = "ruda"
os.environ["RUDA_TRITON_DEVICE"] = "ruda"

import pytest
import torch

ruda_torch = pytest.importorskip("ruda_torch")
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

DEV = "ruda"


@triton.jit
def axpy_kernel(x_ptr, y_ptr, out_ptr, a, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, a * tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)


@triton.jit
def row_softmax_kernel(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * n_cols + cols, mask=cols < n_cols, other=-float("inf"))
    x = tl.exp(x - tl.max(x, axis=0))
    tl.store(out_ptr + row * n_cols + cols, x / tl.sum(x, axis=0), mask=cols < n_cols)


def test_device_is_ruda():
    driver = triton.runtime.driver.active
    assert driver.get_active_torch_device().type == "ruda"
    assert isinstance(driver.get_current_stream(), int)


def test_axpy_interleaved_with_ruda_ops():
    n = 100_000
    x_cpu, y_cpu = torch.randn(n), torch.randn(n)
    x, y = x_cpu.to(DEV), y_cpu.to(DEV)
    out = torch.empty_like(x)
    for _ in range(3):
        y = y * 2  # RUDA kernel producing the Triton input
        axpy_kernel[(triton.cdiv(n, 1024), )](x, y, out, 0.5, n, BLOCK=1024)
        out = out + 1  # RUDA kernel consuming the Triton output
        y_cpu = y_cpu * 2
    torch.testing.assert_close(out.cpu(), 0.5 * x_cpu + y_cpu + 1)


def test_softmax_on_ruda_stream():
    x_cpu = torch.randn(64, 300)
    x = x_cpu.to(DEV)
    out = torch.empty_like(x)
    s = ruda_torch.Stream()
    with ruda_torch.stream(s):
        row_softmax_kernel[(64, )](x, out, 300, BLOCK=512)
    s.synchronize()
    torch.testing.assert_close(out.cpu(), torch.softmax(x_cpu, 1), atol=1e-5, rtol=1e-5)
