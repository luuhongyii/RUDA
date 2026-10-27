import os

import pytest
import torch


@pytest.fixture(scope='module', autouse=True)
def backend():
    if os.environ.get('RUDA_REQUIRE_GPU') != '1':
        pytest.skip('requires native RUDA GPU backend')
    import ruda_torch
    return ruda_torch


@pytest.mark.parametrize('op', [1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 17, 19, 20, 21, 27])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('layout', ['packed', 'rows', 'broadcast'])
def test_vector_pointwise_matches_scalar_storage(op, dtype, layout, backend):
    torch.manual_seed(3401)
    a = torch.randn(4, 8, 128, dtype=dtype).to('ruda')
    b = (torch.randn(4, 8, 128, dtype=dtype) + .25).to('ruda')
    if layout == 'packed':
        a, b = a[..., :64].contiguous(), b[..., :64].contiguous()
    elif layout == 'rows':
        a, b = a[..., 64:], b[..., :64]
    else:
        a = a[..., :64]
        b = b[:1, :1, :64].expand_as(a)
    fast = torch.empty(a.shape, dtype=dtype, device='ruda')
    storage = torch.full((a.numel() + 2,), 71., dtype=dtype, device='ruda')
    scalar = storage[1:-1].view(a.shape)
    backend._C.execute(op, a, b, fast, .375)
    backend._C.execute(op, a, b, scalar, .375)
    torch.testing.assert_close(fast.cpu(), scalar.cpu(), rtol=0, atol=0, equal_nan=True)
    assert storage[0].cpu().item() == 71. and storage[-1].cpu().item() == 71.


@pytest.mark.parametrize('a_dtype,b_dtype', [(torch.float16, torch.float32), (torch.bfloat16, torch.float16)])
@pytest.mark.parametrize('op', [1, 2, 8])
@pytest.mark.parametrize('out_dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_vector_pointwise_mixed_storage(op, a_dtype, b_dtype, out_dtype, backend):
    torch.manual_seed(3402)
    a = torch.randn(16, 64, dtype=a_dtype).to('ruda')
    b = (torch.randn(16, 64, dtype=b_dtype) + .25).to('ruda')
    fast = torch.empty(a.shape, dtype=out_dtype, device='ruda')
    scalar = torch.empty(a.numel() + 1, dtype=out_dtype, device='ruda')[1:].view(a.shape)
    backend._C.execute(op, a, b, fast, -.375)
    backend._C.execute(op, a, b, scalar, -.375)
    torch.testing.assert_close(fast.cpu(), scalar.cpu(), rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_vector_pointwise_nonfinite_and_inplace(dtype, backend):
    a = torch.tensor([float('nan'), float('inf'), -float('inf'), -0., 0., -2., 2., 1.], dtype=dtype).repeat(128).to('ruda')
    b = torch.ones_like(a)
    for op in [1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 17, 19, 20, 21, 27]:
        fast = torch.empty_like(a)
        scalar = torch.empty(a.numel() + 1, dtype=dtype, device='ruda')[1:]
        backend._C.execute(op, a, b, fast, .5)
        backend._C.execute(op, a, b, scalar, .5)
        torch.testing.assert_close(fast.cpu(), scalar.cpu(), rtol=0, atol=0, equal_nan=True)
    fast = a.clone()
    scalar = torch.empty(a.numel() + 1, dtype=dtype, device='ruda')[1:]
    scalar.copy_(a)
    backend._C.execute(2, fast, b, fast, 0.)
    backend._C.execute(2, scalar, b, scalar, 0.)
    torch.testing.assert_close(fast.cpu(), scalar.cpu(), rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize('ta,nt', [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('output_dtype', [torch.float16, torch.float32])
def test_f16_ordered_tiled_layouts(ta, nt, batched, output_dtype, backend):
    m, k, n = 96, 160, 96
    batch = (2,) if batched else ()
    a_shape = (*batch, k, m + 8) if ta else (*batch, m, k + 8)
    b_shape = (*batch, n, k + 8) if nt else (*batch, k, n + 8)
    a = ((torch.arange(torch.tensor(a_shape).prod()).reshape(a_shape) % 7) - 3).half()
    b = ((torch.arange(torch.tensor(b_shape).prod()).reshape(b_shape) % 5) - 2).half()
    x, y = a.to('ruda')[..., 8:], b.to('ruda')[..., 8:]
    left, right = a[..., 8:].double(), b[..., 8:].double()
    if ta:
        x, left = x.transpose(-1, -2), left.transpose(-1, -2)
    if nt:
        y, right = y.transpose(-1, -2), right.transpose(-1, -2)
    expected = (left @ right).to(output_dtype)
    for offset in [0, 1]:
        storage = torch.full((expected.numel() + offset + 8,), float('nan'), dtype=output_dtype, device='ruda')
        out = storage[offset:offset + expected.numel()].view(expected.shape)
        backend._C.execute(30 if batched else 7, x, y, out, 0.)
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
        assert torch.isnan(storage[:offset].cpu()).all()
        assert torch.isnan(storage[offset + expected.numel():].cpu()).all()


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_ordered_tuning_separates_input_dtype_with_f32_output(dtype, backend):
    torch.manual_seed(3403)
    a = torch.randn(128, 256, dtype=dtype)
    b = torch.randn(256, 128, dtype=dtype)
    out = torch.empty(128, 128, dtype=torch.float32, device='ruda')
    backend._C.execute(7, a.to('ruda'), b.to('ruda'), out, 0.)
    torch.testing.assert_close(out.cpu(), (a.double() @ b.double()).float(), atol=2e-4, rtol=1e-4)
