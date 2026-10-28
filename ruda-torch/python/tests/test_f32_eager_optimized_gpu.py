import os

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(scope='module', autouse=True)
def backend():
    if os.environ.get('RUDA_REQUIRE_GPU') != '1':
        pytest.skip('requires native RUDA GPU backend')
    import ruda_torch
    assert ruda_torch._matmul_precision_available
    return ruda_torch


@pytest.fixture(autouse=True)
def precision():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(3501)
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize('allow_tf32', [False, True])
@pytest.mark.parametrize('ta,nt', [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize('batched', [False, True])
def test_f32_precision_layouts(allow_tf32, ta, nt, batched, backend):
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    batch = (2,) if batched else ()
    a = torch.randn((*batch, 160, 100) if ta else (*batch, 96, 164))
    b = torch.randn((*batch, 128, 164) if nt else (*batch, 160, 132))
    left, right = a[..., 4:], b[..., 4:]
    if ta:
        left = left.transpose(-1, -2)
    if nt:
        right = right.transpose(-1, -2)
    x, y = a.to('ruda')[..., 4:], b.to('ruda')[..., 4:]
    if ta:
        x = x.transpose(-1, -2)
    if nt:
        y = y.transpose(-1, -2)
    expected = (left.double() @ right.double()).float()
    for offset in (0, 1, 2):
        storage = torch.full((expected.numel() + offset + 8,), float('nan'), device='ruda')
        out = storage[offset:offset + expected.numel()].view(expected.shape)
        backend._C.execute(30 if batched else 7, x, y, out, 0.)
        torch.testing.assert_close(out.cpu(), expected, atol=.04 if allow_tf32 else 3e-5,
                                   rtol=.004 if allow_tf32 else 1e-5)
        assert torch.isnan(storage[:offset].cpu()).all()
        assert torch.isnan(storage[offset + expected.numel():].cpu()).all()


@pytest.mark.parametrize('allow_tf32', [False, True])
@pytest.mark.parametrize('shape', [(32, 8, 32), (96, 160, 224), (352, 352, 352), (128, 8192, 128)])
def test_f32_tiled_precision_and_edges(allow_tf32, shape):
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    m, k, n = shape
    a, b = torch.randn(2, m, k), torch.randn(2, k, n)
    actual = torch.bmm(a.to('ruda'), b.to('ruda')).cpu()
    expected = (a.double() @ b.double()).float()
    tolerance = .004 if allow_tf32 else 2e-5
    torch.testing.assert_close(actual / k ** .5, expected / k ** .5, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('shape', [(32, 8, 32), (64, 72, 64), (128, 256, 128), (96, 160, 224),
                                  (224, 72, 96), (37, 19, 45), (32, 0, 64)])
def test_f32_eager_matmul_addmm_and_grad(shape):
    m, k, n = shape
    a, b, bias = torch.randn(m, k), torch.randn(k, n), torch.randn(n)
    refs = [x.requires_grad_() for x in (a, b, bias)]
    inputs = [x.detach().to('ruda').requires_grad_() for x in refs]
    expected = torch.addmm(refs[2], refs[0], refs[1], alpha=.7, beta=.3)
    actual = torch.addmm(inputs[2], inputs[0], inputs[1], alpha=.7, beta=.3)
    grad = torch.randn_like(expected)
    expected.backward(grad)
    actual.backward(grad.to('ruda'))
    for x, y in [(actual, expected), *[(x.grad, y.grad) for x, y in zip(inputs, refs)]]:
        torch.testing.assert_close(x.cpu(), y, atol=5e-5, rtol=2e-5)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('width', [16, 33, 128, 320])
def test_middle_axis_reduction(dtype, width):
    source = torch.randn(2, 4, width, 128, dtype=dtype)
    for promoted in (False, True):
        output_dtype = torch.float32 if promoted else dtype
        for mean in (False, True):
            op = torch.mean if mean else torch.sum
            actual = op(source.to('ruda'), dim=2, keepdim=mean, dtype=output_dtype)
            expected = op(source, dim=2, keepdim=mean, dtype=output_dtype)
            tolerance = .03 if output_dtype == torch.bfloat16 else .003 if output_dtype == torch.float16 else 1e-5
            torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('broadcast', [False, True])
def test_eager_strided_arithmetic(dtype, broadcast):
    a = torch.randn(4, 8, 128, dtype=dtype)
    b = torch.randn(4, 8, 1 if broadcast else 128, dtype=dtype)
    x, y = a.to('ruda')[..., 64:], b.to('ruda')
    a = a[..., 64:]
    if not broadcast:
        b, y = b[..., :64], y[..., :64]
    for op in (torch.add, torch.sub, torch.mul):
        torch.testing.assert_close(op(x, y).cpu(), op(a, b), rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_silu_lookup_vector_width_and_nonfinite(dtype):
    values = torch.tensor([float('nan'), float('inf'), -float('inf'), 0., -0., -2., 2., 1.], dtype=dtype).repeat(8192)
    aligned = values.to('ruda')
    unaligned = torch.empty(values.numel() + 1, device='ruda', dtype=dtype)[1:]
    unaligned.copy_(aligned)
    torch.testing.assert_close(F.silu(aligned).cpu(), F.silu(unaligned).cpu(), atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('queries,keys,width,policy', [(65, 127, 32, 'plain'), (128, 257, 64, 'causal'),
                                                     (127, 65, 128, 'mask'), (257, 128, 64, 'gqa'),
                                                     (193, 193, 128, 'causal'), (65, 513, 80, 'plain'),
                                                     (257, 129, 16, 'plain'), (257, 193, 48, 'gqa'),
                                                     (257, 257, 80, 'causal'), (385, 321, 112, 'causal'),
                                                     (1024, 1089, 64, 'causal'), (1089, 1025, 128, 'mask'),
                                                     (1025, 1089, 128, 'gqa')])
def test_async_attention_tiles(dtype, queries, keys, width, policy):
    host = [torch.randn(1, h, n, width, dtype=dtype) for h, n in
            [(4, queries), (2 if policy == 'gqa' else 4, keys), (2 if policy == 'gqa' else 4, keys)]]
    mask = None
    if policy == 'mask':
        mask = torch.rand(queries, keys) > .2
        mask[0] = False
    options = dict(is_causal=policy == 'causal', enable_gqa=policy == 'gqa')
    expected = F.scaled_dot_product_attention(*(x.float() for x in host), attn_mask=mask, **options)
    actual = F.scaled_dot_product_attention(*(x.to('ruda') for x in host),
        attn_mask=None if mask is None else mask.to('ruda'), **options)
    tolerance = .02 if dtype == torch.bfloat16 else .003
    torch.testing.assert_close(actual.cpu().float(), expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_long_attention_tile_gradients(dtype):
    host=[torch.randn(1,2,1025,64,dtype=dtype) for _ in range(3)]
    inputs=[x.to('ruda').requires_grad_() for x in host]
    refs=[x.float().requires_grad_() for x in host]
    actual=F.scaled_dot_product_attention(*inputs,is_causal=True)
    expected=F.scaled_dot_product_attention(*refs,is_causal=True)
    gradient=torch.randn_like(host[0]);actual.backward(gradient.to('ruda'));expected.backward(gradient.float())
    tolerance=.04 if dtype==torch.bfloat16 else .006
    for value,target in [(actual,expected),*[(x.grad,y.grad) for x,y in zip(inputs,refs)]]:
        torch.testing.assert_close(value.cpu().float(),target.float(),rtol=tolerance,atol=tolerance)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('width,inner',[(3,64),(7,132),(16,16),(3,65)])
@pytest.mark.parametrize('logarithmic',[False,True])
def test_spatial_softmax_forward_and_grad(dtype,width,inner,logarithmic):
    host=torch.randn(2,width,inner,dtype=dtype)
    x=host.to('ruda').requires_grad_();ref=host.float().requires_grad_()
    op=F.log_softmax if logarithmic else F.softmax
    actual=op(x,dim=1);expected=op(ref,dim=1).to(dtype)
    gradient=torch.randn_like(host);actual.backward(gradient.to('ruda'));expected.backward(gradient)
    tolerance=.04 if dtype==torch.bfloat16 else .005 if dtype==torch.float16 else 2e-5
    for value,target in [(actual,expected),(x.grad,ref.grad)]:
        torch.testing.assert_close(value.cpu().float(),target.float(),rtol=tolerance,atol=tolerance)


@pytest.mark.parametrize('shape', [(32, 24, 32), (64, 72, 64), (128, 128, 128), (224, 96, 160)])
@pytest.mark.parametrize('bias_kind', ['scalar', 'row', 'column', 'matrix'])
def test_f32_fused_addmm_epilogue(shape, bias_kind, backend):
    m, k, n = shape
    a = ((torch.arange(m * k).reshape(m, k) % 7) - 3).float()
    b = ((torch.arange(k * n).reshape(k, n) % 5) - 2).float()
    bias_shape = {'scalar': (1, 1), 'row': (1, n), 'column': (m, 1), 'matrix': (m, n)}[bias_kind]
    bias = torch.arange(bias_shape[0] * bias_shape[1]).reshape(bias_shape).float() % 11
    x, y = a.to('ruda'), b.to('ruda')
    for beta in (0., -.5):
        c = (torch.full_like(bias, float('nan')) if beta == 0 else bias).to('ruda').expand(m, n)
        expected = (a.double() @ b.double()) * .5
        if beta != 0:
            expected += beta * bias.double()
        for offset in (0, 1, 2):
            storage = torch.full((m * n + offset + 4,), float('nan'), device='ruda')
            out = storage[offset:offset + m * n].view(m, n)
            before = backend.execution_stats()
            backend._C.addmm(c, x, y, out, .5, beta)
            after = backend.execution_stats()
            assert after['kernel_launches'] - before['kernel_launches'] == 1
            torch.testing.assert_close(out.cpu(), expected.float(), rtol=0, atol=0)
            assert torch.isnan(storage[:offset].cpu()).all()
            assert torch.isnan(storage[offset + m * n:].cpu()).all()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('width', [512, 777, 4096, 10240])
@pytest.mark.parametrize('kind', ['rms', 'hf_rms', 'layer'])
def test_cached_norm_gradients(dtype, width, kind, backend):
    source = torch.randn(3, width, dtype=dtype)
    weight = torch.randn(width, dtype=dtype)
    bias = torch.randn(width, dtype=dtype)
    x = source.to('ruda').requires_grad_()
    w = weight.to('ruda').requires_grad_()
    b = bias.to('ruda').requires_grad_()
    ref_x = source.float().requires_grad_()
    ref_w = weight.clone().requires_grad_()
    ref_b = bias.clone().requires_grad_()
    if kind == 'layer':
        actual = backend.layer_norm(x, w, b, eps=1e-5)
        expected = F.layer_norm(ref_x, (width,), ref_w.float(), ref_b.float(), eps=1e-5).to(dtype)
    else:
        actual = backend.rms_norm(x, w, eps=1e-5, round_normalized=kind == 'hf_rms')
        normalized = ref_x * (ref_x.square().mean(-1, keepdim=True) + 1e-5).rsqrt()
        expected = (normalized.to(dtype) * ref_w) if kind == 'hf_rms' else (normalized * ref_w.float()).to(dtype)
    gradient = torch.randn_like(source)
    actual.backward(gradient.to('ruda'))
    expected.backward(gradient)
    tolerance = .05 if dtype == torch.bfloat16 else .006 if dtype == torch.float16 else 5e-4
    pairs = [(actual, expected), (x.grad, ref_x.grad), (w.grad, ref_w.grad)]
    if kind == 'layer':
        pairs.append((b.grad, ref_b.grad))
    for a, ref in pairs:
        torch.testing.assert_close(a.cpu().float(), ref.float(), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('needs', [(True, True), (False, True), (True, False)])
@pytest.mark.parametrize('precision_mode', ['fp32', 'cuda'])
def test_fused_swiglu_vector_matches_scalar(dtype, needs, precision_mode, backend, monkeypatch):
    monkeypatch.setenv('RUDA_TORCH_BF16_PRECISION', precision_mode)
    gate, up, grad = [torch.randn(4, 256, dtype=dtype) for _ in range(3)]
    results = []
    for offset in (0, 1):
        inputs = []
        for source, need in zip((gate, up), needs):
            value = torch.empty(source.numel() + offset, dtype=dtype, device='ruda')[offset:].view_as(source)
            value.copy_(source.to('ruda'))
            inputs.append(value.requires_grad_(need))
        result = backend.silu_mul(*inputs)
        result.backward(grad.to('ruda'))
        results.append([result.cpu(), *[x.grad.cpu() for x in inputs if x.requires_grad]])
    for aligned, scalar in zip(*results):
        torch.testing.assert_close(aligned, scalar, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('width', [32, 64, 128])
@pytest.mark.parametrize('needs', [(True, True), (False, True), (True, False)])
def test_rotary_paired_vector_matches_scalar(dtype, width, needs):
    from ruda_torch import rotary
    q = torch.randn(2, 7, 4, width, dtype=dtype).to('ruda').transpose(1, 2)
    k = torch.randn(2, 7, 2, width, dtype=dtype).to('ruda').transpose(1, 2)
    cos = torch.randn(1, 1, 7, width, dtype=dtype).to('ruda')
    sin = torch.randn(2, 1, 1, width, dtype=dtype).to('ruda')
    # An unaligned coefficient forces the existing scalar path with identical values.
    unaligned = torch.empty(cos.numel() + 1, dtype=dtype, device='ruda')[1:].view_as(cos)
    unaligned.copy_(cos)
    aligned = rotary._forward(q, k, cos, sin)
    scalar = rotary._forward(q, k, unaligned, sin)
    for a, b in zip(aligned, scalar):
        torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)
    grads = (q if needs[0] else None, k if needs[1] else None)
    aligned = rotary._backward(*grads, cos, sin, list(q.shape), list(k.shape))
    scalar = rotary._backward(*grads, unaligned, sin, list(q.shape), list(k.shape))
    for a, b in zip(aligned, scalar):
        if a is not None:
            torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)
