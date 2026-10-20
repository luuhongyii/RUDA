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


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  PRECISION: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptr + rm[:, None] * K + (k + rk)[None, :], mask=(rm[:, None] < M) & ((k + rk)[None, :] < K), other=0.0)
        b = tl.load(b_ptr + (k + rk)[:, None] * N + rn[None, :], mask=((k + rk)[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, input_precision=PRECISION)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc.to(c_ptr.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def histogram_kernel(x_ptr, hist_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.atomic_add(hist_ptr + tl.load(x_ptr + offs, mask=mask, other=0), 1, mask=mask)


@triton.jit
def atomic_sum_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.atomic_add(out_ptr, tl.sum(tl.load(x_ptr + offs, mask=offs < n, other=0.0), axis=0))


@triton.jit
def atomic_max_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.atomic_max(out_ptr + offs % 7, tl.load(x_ptr + offs, mask=mask, other=0), mask=mask)


@triton.jit
def lock_counter_kernel(lock_ptr, count_ptr):
    while tl.atomic_cas(lock_ptr, 0, 1) == 1:
        pass
    tl.store(count_ptr, tl.load(count_ptr) + 1)
    tl.atomic_xchg(lock_ptr, 0)


@triton.jit
def cumsum_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, AXIS: tl.constexpr, REVERSE: tl.constexpr):
    rows = tl.arange(0, M)[:, None]
    cols = tl.arange(0, N)[None, :]
    x = tl.load(x_ptr + rows * N + cols)
    tl.store(out_ptr + rows * N + cols, tl.cumsum(x, axis=AXIS, reverse=REVERSE))


@triton.jit
def while_kernel(x_ptr, out_ptr, limit):
    x = tl.load(x_ptr)
    steps = 0
    while x < limit:
        x = x * 2 + 1
        steps += 1
    tl.store(out_ptr, x)
    tl.store(out_ptr + 1, steps)


GX1 = os.environ.get("RUDA_TRITON_DEVICE") == "gx1"
DEV = "cpu" if GX1 else "cuda"

@pytest.mark.parametrize("dtype,precision", [(torch.float32, "ieee"), (torch.float32, "tf32"), (torch.bfloat16, "ieee"),
                                             (torch.float16, "ieee")])
def test_matmul(dtype, precision):
    M, N, K = 70, 90, 100
    a = torch.randn(M, K, device=DEV, dtype=dtype)
    b = torch.randn(K, N, device=DEV, dtype=dtype)
    c = torch.empty(M, N, device=DEV, dtype=torch.float32)
    matmul_kernel[(triton.cdiv(M, 32), triton.cdiv(N, 32))](a, b, c, M, N, K, BM=32, BN=32, BK=32, PRECISION=precision)
    ref = a.double() @ b.double()
    tol = 2e-2 if precision == "tf32" or dtype != torch.float32 else 1e-4
    torch.testing.assert_close(c.double(), ref, atol=tol * K ** 0.5, rtol=tol)


def test_atomic_histogram():
    x = torch.randint(0, 50, (10_000, ), device=DEV, dtype=torch.int32)
    hist = torch.zeros(50, device=DEV, dtype=torch.int32)
    histogram_kernel[(triton.cdiv(x.numel(), 512), )](x, hist, x.numel(), BLOCK=512)
    assert torch.equal(hist.long(), torch.bincount(x.long(), minlength=50))


def test_atomic_scalar_add():
    x = torch.randn(5000, device=DEV)
    out = torch.zeros(1, device=DEV)
    atomic_sum_kernel[(triton.cdiv(x.numel(), 256), )](x, out, x.numel(), BLOCK=256)
    torch.testing.assert_close(out[0], x.sum(), atol=1e-3, rtol=1e-4)


def test_atomic_max():
    x = torch.randint(-1000, 1000, (4096, ), device=DEV, dtype=torch.int32)
    out = torch.full((7, ), -2**31, device=DEV, dtype=torch.int32)
    atomic_max_kernel[(4, )](x, out, x.numel(), BLOCK=1024)
    ref = torch.stack([x[i::7].max() for i in range(7)])
    assert torch.equal(out, ref)


def test_atomic_cas_lock():
    lock = torch.zeros(1, device=DEV, dtype=torch.int32)
    count = torch.zeros(1, device=DEV, dtype=torch.int32)
    lock_counter_kernel[(64, )](lock, count, num_warps=1)
    assert count.item() == 64 and lock.item() == 0


@pytest.mark.parametrize("axis,reverse", [(0, False), (1, False), (1, True), (0, True)])
def test_cumsum(axis, reverse):
    x = torch.randint(-5, 5, (8, 256), device=DEV, dtype=torch.int32)
    out = torch.empty_like(x)
    cumsum_kernel[(1, )](x, out, M=8, N=256, AXIS=axis, REVERSE=reverse)
    ref = (x.flip(axis).cumsum(axis).flip(axis) if reverse else x.cumsum(axis)).to(torch.int32)
    assert torch.equal(out, ref)


def test_while_loop():
    x = torch.tensor([3], device=DEV, dtype=torch.int32)
    out = torch.empty(2, device=DEV, dtype=torch.int32)
    while_kernel[(1, )](x, out, 1000)
    v, steps = 3, 0
    while v < 1000:
        v, steps = v * 2 + 1, steps + 1
    assert out.tolist() == [v, steps]


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
