import os

import pytest
import torch

if os.environ.get('RUDA_REQUIRE_GPU') != '1':
    pytest.skip('requires a real GPU', allow_module_level=True)

import ruda_torch


@pytest.fixture
def fp32_cuda_accumulation():
    previous=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=previous


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('k', [32, 64, 96, 128, 160, 192, 224, 576, 3584])
def test_streamed_mma_k_tiles(dtype, k):
    torch.manual_seed(988)
    a = torch.randn(2, 128, k).to(dtype)
    b = torch.randn(2, 256, k).to(dtype)
    expected = (a.cuda() @ b.cuda().transpose(-1, -2)).cpu()
    actual = a.to('ruda') @ b.to('ruda').transpose(-1, -2)
    tolerance = .04 if dtype == torch.bfloat16 else .007
    torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('batch,m,n,k', [(1,128,256,8192),(1,129,193,8256),(2,128,256,8256)])
def test_split_k_fp32_partials(dtype, batch, m, n, k):
    torch.manual_seed(1028)
    shape=() if batch==1 else (batch,)
    a=(torch.randn((*shape,m,k))/16).to(dtype)
    w=(torch.randn((*shape,n,k))/16).to(dtype)
    x,weight=a.to('ruda'),w.to('ruda')
    before=ruda_torch.execution_stats()
    actual=x@weight.transpose(-1,-2)
    after=ruda_torch.execution_stats()
    assert after['kernel_launches']-before['kernel_launches']==2
    for key in ['host_to_device_bytes','device_to_host_bytes']:
        assert after[key]==before[key]
    reference=(a.cuda()@w.cuda().transpose(-1,-2)).cpu()
    tolerance=.004 if dtype==torch.bfloat16 else .0006
    torch.testing.assert_close(actual.cpu(),reference,atol=tolerance,rtol=tolerance)
    precise=a.double()@w.double().transpose(-1,-2)
    torch.testing.assert_close(actual.cpu(),precise.to(dtype),atol=tolerance,rtol=tolerance)
    if batch==1:
        for bias_shape in [(n,),(m,1),(m,n)]:
            bias=torch.randn(bias_shape).to(dtype)/16
            device_bias=bias.to('ruda')
            for alpha,beta in [(.75,-.25),(1.,1.),(.5,0.)]:
                if beta==0:device_bias.fill_(float('nan'))
                actual=torch.addmm(device_bias,x,weight.t(),alpha=alpha,beta=beta)
                expected=precise*alpha
                if beta!=0:expected=expected+bias.double()*beta
                torch.testing.assert_close(actual.cpu(),expected.to(dtype),atol=tolerance,rtol=tolerance)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('layout', ['contiguous', 'strided', 'expanded', 'empty'])
@pytest.mark.parametrize('dim', [0, 1, -1])
def test_single_dispatch_cat(dtype, layout, dim):
    a = torch.arange(120, dtype=torch.float32).reshape(3, 4, 10).to(dtype)
    b = a + 1
    x, y = a.to('ruda'), b.to('ruda')
    if layout == 'strided':
        a, b, x, y = [t.transpose(0, 1)[..., 1::2] for t in (a, b, x, y)]
    elif layout == 'expanded':
        a, b, x, y = [t[:1].expand(3, 4, 10) for t in (a, b, x, y)]
    elif layout == 'empty':
        axis = dim % 3
        a, x = a.narrow(axis, 0, 0), x.narrow(axis, 0, 0)
    torch.testing.assert_close(torch.cat((x, y), dim).cpu(), torch.cat((a, b), dim), atol=0, rtol=0)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('offset', [0, 1, 3])
def test_vector_memory_offset_roundtrip(dtype, offset):
    torch.manual_seed(989)
    source = torch.randn(8192 + offset).to(dtype)
    source[offset:offset+4] = torch.tensor([float('inf'), -float('inf'), float('nan'), -0.], dtype=dtype)
    expected = source[offset:offset+8192]
    device = source.to('ruda')[offset:offset+8192]
    torch.testing.assert_close(device.clone().cpu(), expected, atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(device.float().cpu(), expected.float(), atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close((device * 2).cpu(), expected * 2, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('shape', [(64, 64, 64), (64, 128, 64), (128, 64, 128), (192, 128, 192), (128, 192, 128), (192, 192, 192), (1024, 576, 576), (1024, 576, 192), (1024, 576, 1536), (128, 256, 128), (256, 1024, 512), (2, 128, 128, 256), (65, 32, 67), (129, 96, 193), (2, 127, 64, 191), (64, 160, 65), (257, 128, 577), (129, 768, 193), (2, 257, 192, 641), (512,2048,4096), (513,2112,4353)])
def test_swizzled_transposed_weight_matmul(dtype, shape, fp32_cuda_accumulation):
    torch.manual_seed(990)
    batch, (m, k, n) = shape[:-3], shape[-3:]
    a = torch.randn((*batch, m, k)).to(dtype)
    w = torch.randn((*batch, n, k)).to(dtype)
    expected = (a.cuda() @ w.cuda().transpose(-1, -2)).cpu()
    x, weight = a.to('ruda'), w.to('ruda')
    tolerance = .04 if dtype == torch.bfloat16 else .007
    torch.testing.assert_close((x @ weight.transpose(-1, -2)).cpu(), expected, atol=tolerance, rtol=tolerance)
    if not batch:
        bias = torch.randn(n).to(dtype)
        for alpha, beta in [(1., 1.), (.75, -.25), (1., 0.)]:
            actual = torch.addmm(bias.to('ruda'), x, weight.t(), alpha=alpha, beta=beta)
            reference = torch.addmm(bias.cuda(), a.cuda(), w.cuda().t(), alpha=alpha, beta=beta)
            torch.testing.assert_close(actual.cpu(), reference.cpu(), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('k', [1024, 1056])
def test_grouped_mma_tails_against_f64(dtype, k):
    torch.manual_seed(990)
    a = torch.randn(513, k).to(dtype)
    w = torch.randn(577, k).to(dtype)
    product = a.double() @ w.double().t()
    x, weight = a.to('ruda'), w.to('ruda')
    tolerance = .04 if dtype == torch.bfloat16 else .007
    torch.testing.assert_close((x @ weight.t()).cpu(), product.to(dtype), atol=tolerance, rtol=tolerance)
    bias = torch.randn(577).to(dtype)
    for alpha, beta in [(1., 1.), (.75, -.25), (1., 0.)]:
        actual = torch.addmm(bias.to('ruda'), x, weight.t(), alpha=alpha, beta=beta)
        expected = (product * alpha + bias.double() * beta).to(dtype)
        torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('shape', [(64, 64, 64), (129, 96, 193), (513, 1024, 577)])
@pytest.mark.parametrize('bias_layout', ['scalar', 'row', 'column', 'matrix', 'strided', 'ignored_nan'])
def test_pipelined_addmm_single_launch(dtype, shape, bias_layout):
    m, k, n = shape
    generator = torch.Generator().manual_seed(1006)
    def values(size):
        return (torch.randint(-8, 9, size, generator=generator).float() / 64).to(dtype)
    a, w = values((m, k)), values((n, k))
    product = a.double() @ w.double().t()
    x, weight = a.to('ruda'), w.to('ruda')
    if bias_layout == 'strided':
        storage = values((m, n * 2 + 1))
        bias, device_bias = storage[:, 1::2], storage.to('ruda')[:, 1::2]
    else:
        size = {'scalar': (), 'row': (n,), 'column': (m, 1), 'matrix': (m, n), 'ignored_nan': (n,)}[bias_layout]
        bias = values(size)
        if bias_layout == 'ignored_nan':
            bias.fill_(float('nan'))
        device_bias = bias.to('ruda')
    for alpha, beta in [(1., 1.), (.75, -.25), (.5, 0.)]:
        if bias_layout == 'ignored_nan' and beta != 0:
            continue
        before = ruda_torch.execution_stats()
        actual = torch.addmm(device_bias, x, weight.t(), alpha=alpha, beta=beta)
        after = ruda_torch.execution_stats()
        assert after['kernel_launches'] - before['kernel_launches'] == 1
        assert after['rublas_calls'] - before['rublas_calls'] == 1
        for key in ['addmm_workspace_bytes_total', 'addmm_epilogues', 'host_to_device_bytes', 'device_to_host_bytes']:
            assert after[key] == before[key]
        expected = product * alpha
        if beta != 0:
            expected = expected + bias.double() * beta
        torch.testing.assert_close(actual.cpu(), expected.to(dtype), atol=0, rtol=0)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('width', [16, 64, 96, 128])
@pytest.mark.parametrize('causal', [False, True])
def test_register_attention_tails(dtype, width, causal):
    torch.manual_seed(991)
    q = torch.randn(1, 4, 129, width).to(dtype)
    k, v = [torch.randn(1, 2, 137, width).to(dtype) for _ in range(2)]
    with torch.inference_mode():
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.cuda(), k.cuda(), v.cuda(), is_causal=causal, enable_gqa=True)
        actual = ruda_torch.scaled_dot_product_attention(
            q.to('ruda'), k.to('ruda'), v.to('ruda'), is_causal=causal, enable_gqa=True)
    tolerance = .025 if dtype == torch.bfloat16 else .005
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_register_attention_unaligned_contiguous_views(dtype):
    torch.manual_seed(992)
    cpu, device = [], []
    for shape in [(1, 4, 65, 32), (1, 2, 71, 32), (1, 2, 71, 32)]:
        count = 1
        for size in shape:
            count *= size
        storage = torch.randn(count + 1).to(dtype)
        cpu.append(storage[1:].reshape(shape))
        device.append(storage.to('ruda')[1:].reshape(shape))
        assert device[-1].is_contiguous() and device[-1].data_ptr() % 16 != 0
    with torch.inference_mode():
        expected = torch.nn.functional.scaled_dot_product_attention(
            *(value.cuda() for value in cpu), is_causal=True, enable_gqa=True)
        actual = ruda_torch.scaled_dot_product_attention(*device, is_causal=True, enable_gqa=True)
    tolerance = .025 if dtype == torch.bfloat16 else .005
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.complex64])
def test_native_eager_compatibility(dtype):
    a = (torch.arange(24).reshape(2, 3, 4) - 8).to(dtype)
    b = a[:, :1, :]
    x, y = a.to('ruda'), b.to('ruda')
    torch.testing.assert_close(torch.add(x, y, alpha=2).cpu(), torch.add(a, b, alpha=2))
    torch.testing.assert_close(torch.sub(x, y, alpha=-2).cpu(), torch.sub(a, b, alpha=-2))
    torch.testing.assert_close((-x).cpu(), -a)
    for dim in [-1, 0, 1]:
        values = [a, a] if dim != 1 else [a, b]
        result = torch.cat([t.to('ruda') for t in values], dim)
        torch.testing.assert_close(result.cpu(), torch.cat(values, dim))
    torch.testing.assert_close(torch.cat([x[:0], x], 0).cpu(), a)
    if dtype in (torch.float32, torch.float16, torch.bfloat16):
        for operation in [torch.add, torch.mul]:
            torch.testing.assert_close(operation(x, 1e100).cpu(), operation(a, 1e100), equal_nan=True)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16, torch.float64, torch.complex64])
def test_native_eager_normalization(dtype):
    values=(torch.arange(24).reshape(2,3,4)/8+1).to(dtype).transpose(0,1)
    x=values.to('ruda')
    tolerance=.025 if dtype==torch.bfloat16 else .003
    for exponent in [0.,1.,2.,3.,-2.,.5,-.5,-1.,1.7]:
        torch.testing.assert_close(x.pow(exponent).cpu(),values.pow(exponent),rtol=tolerance,atol=tolerance)
    torch.testing.assert_close(x.rsqrt().cpu(),values.rsqrt(),rtol=tolerance,atol=tolerance)
    for dimensions in [None,[],[-1],[0,2]]:
        for keep in [True,False]:
            torch.testing.assert_close(x.mean(dim=dimensions,keepdim=keep).cpu(),values.mean(dim=dimensions,keepdim=keep),rtol=tolerance,atol=tolerance)
    if dtype in (torch.float16,torch.bfloat16):
        torch.testing.assert_close(x.mean(dim=-1,dtype=torch.float32).cpu(),values.mean(dim=-1,dtype=torch.float32))
    for value in [values[0,0,0],values[:0]]:
        torch.testing.assert_close(value.to('ruda').mean(dim=-1).cpu(),value.mean(dim=-1),equal_nan=True,rtol=tolerance,atol=tolerance)


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('keys,width,value_width',[(257,33,47),(1024,128,128),(2048,256,96),(2049,64,64)])
def test_parallel_decode_attention(dtype,keys,width,value_width):
    torch.manual_seed(992)
    q=torch.randn(2,4,1,width).to(dtype)
    k=torch.randn(2,2,keys,width).to(dtype)
    v=torch.randn(2,2,keys,value_width).to(dtype)
    with torch.inference_mode():
        expected=torch.nn.functional.scaled_dot_product_attention(q.cuda(),k.cuda(),v.cuda(),enable_gqa=True)
        actual=ruda_torch.scaled_dot_product_attention(q.to('ruda'),k.to('ruda'),v.to('ruda'),enable_gqa=True)
    tolerance=.025 if dtype==torch.bfloat16 else .004
    torch.testing.assert_close(actual.cpu(),expected.cpu(),atol=tolerance,rtol=tolerance)


@pytest.mark.parametrize('mask_kind',[None,'bool','additive'])
def test_inference_attention_without_python_custom_op(monkeypatch,mask_kind):
    from ruda_torch import _dense_ops
    def forbidden(*args,**kwargs):
        raise AssertionError('inference unexpectedly entered the Python custom op')
    monkeypatch.setattr(_dense_ops,'forward',forbidden)
    torch.manual_seed(993)
    q=torch.randn(2,4,19,32)
    k=torch.randn(1,2,23,32)
    v=torch.randn(1,2,23,48)
    mask=None if mask_kind is None else torch.rand(19,23)>.2
    if mask_kind=='additive':mask=torch.where(mask,0.,float('-inf'))
    with torch.inference_mode():
        expected=torch.nn.functional.scaled_dot_product_attention(q.cuda(),k.cuda(),v.cuda(),attn_mask=None if mask is None else mask.cuda(),enable_gqa=True)
        actual=ruda_torch.scaled_dot_product_attention(q.to('ruda'),k.to('ruda'),v.to('ruda'),attn_mask=None if mask is None else mask.to('ruda'),enable_gqa=True)
    torch.testing.assert_close(actual.cpu(),expected.cpu(),atol=2e-4,rtol=2e-4)
