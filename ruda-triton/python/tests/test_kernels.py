import os

os.environ["TRITON_DEFAULT_BACKEND"] = "ruda"

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)


@triton.jit
def softmax_kernel(x_ptr, out_ptr, stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    tl.store(out_ptr + row * stride + cols, (num / tl.sum(num, axis=0)).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def rowsum_2d_kernel(x_ptr, out_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for start in range(0, N, BN):
        cols = start + tl.arange(0, BN)[None, :]
        mask = (rows < M) & (cols < N)
        acc += tl.load(x_ptr + rows * N + cols, mask=mask, other=0.0)
    total = tl.sum(acc, axis=1)
    rmax = tl.max(acc, axis=1)
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    tl.store(out_ptr + r * 2, total, mask=r < M)
    tl.store(out_ptr + r * 2 + 1, rmax, mask=r < M)


@triton.jit
def outer_kernel(a_ptr, b_ptr, out_ptr, BM: tl.constexpr, BN: tl.constexpr):
    a = tl.load(a_ptr + tl.arange(0, BM))
    b = tl.load(b_ptr + tl.arange(0, BN))
    out = a[:, None] * b[None, :]
    tl.store(out_ptr + tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :], out)


@triton.jit
def interleave_kernel(a_ptr, b_ptr, out_ptr, B: tl.constexpr):
    a = tl.load(a_ptr + tl.arange(0, B))
    b = tl.load(b_ptr + tl.arange(0, B))
    out = tl.reshape(tl.join(a, b), (2 * B, ))
    tl.store(out_ptr + tl.arange(0, 2 * B), out)


@triton.jit
def argmax_kernel(x_ptr, out_ptr, B: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, B))
    tl.store(out_ptr, tl.argmax(x, axis=0))


DEV = "cuda"


def test_add():
    n = 10_000
    x, y = torch.randn(n, device=DEV), torch.randn(n, device=DEV)
    out = torch.empty_like(x)
    add_kernel[(triton.cdiv(n, 1024), )](x, y, out, n, BLOCK=1024)
    torch.testing.assert_close(out, x + y)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("n_cols,num_warps", [(781, 4), (64, 4), (2048, 8)])
def test_softmax(dtype, n_cols, num_warps):
    x = torch.randn(37, n_cols, device=DEV, dtype=dtype)
    out = torch.empty_like(x)
    softmax_kernel[(37, )](x, out, x.stride(0), n_cols, BLOCK=triton.next_power_of_2(n_cols), num_warps=num_warps)
    torch.testing.assert_close(out.float(), torch.softmax(x.float(), dim=1), atol=2e-2 if dtype == torch.bfloat16 else 1e-5,
                               rtol=1e-2 if dtype == torch.bfloat16 else 1e-5)


def test_rowsum_2d_loop():
    M, N = 50, 300
    x = torch.randn(M, N, device=DEV)
    out = torch.empty(M, 2, device=DEV)
    rowsum_2d_kernel[(triton.cdiv(M, 16), )](x, out, M, N, BM=16, BN=64)
    ref_sum = torch.stack([x[:, i::64].sum(1) for i in range(64)], 1)
    torch.testing.assert_close(out[:, 0], x.sum(1), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out[:, 1], ref_sum.max(1).values, atol=1e-4, rtol=1e-4)


def test_outer_broadcast():
    a, b = torch.randn(32, device=DEV), torch.randn(64, device=DEV)
    out = torch.empty(32, 64, device=DEV)
    outer_kernel[(1, )](a, b, out, BM=32, BN=64)
    torch.testing.assert_close(out, a[:, None] * b[None, :])


def test_join_interleave():
    a, b = torch.randn(256, device=DEV), torch.randn(256, device=DEV)
    out = torch.empty(512, device=DEV)
    interleave_kernel[(1, )](a, b, out, B=256)
    torch.testing.assert_close(out, torch.stack([a, b], 1).reshape(-1))


def test_argmax_multi_operand_reduce():
    x = torch.randn(1024, device=DEV)
    out = torch.empty(1, device=DEV, dtype=torch.int32)
    argmax_kernel[(1, )](x, out, B=1024)
    assert out.item() == x.argmax().item()
