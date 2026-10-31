import os
import pytest
import torch


@pytest.fixture(scope='module')
def backend():
    if os.environ.get('RUDA_REQUIRE_GPU') != '1':
        pytest.skip('explicit native GPU execution required')
    import ruda_torch as r
    assert r.get_math_backend() == 'ruda'
    torch.backends.cuda.matmul.allow_tf32 = False
    return r


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('reuse', [False, True])
def test_mixed_pointwise_public_intermediates(backend, dtype, reuse):
    from ruda_torch import GraphOp, StaticGraph
    x = torch.randn(7, 513, dtype=dtype).to('ruda')
    w = torch.randn(513).to('ruda')
    nodes = [GraphOp('cast', 'wide', 'x', dtype='float32'),
             GraphOp('mul_broadcast', 'weighted', 'wide', 'w'),
             GraphOp('add_scalar', 'shifted', 'weighted', scalar=.125),
             GraphOp('cast', 'rounded', 'shifted', dtype=str(dtype).removeprefix('torch.')),
             GraphOp('silu', 'activated', 'rounded'), GraphOp.mul('product', 'activated', 'rounded')]
    with StaticGraph({'x': x, 'w': w}, nodes, outputs=['rounded', 'product'],
                     optimize=True, reuse_workspace=reuse) as graph:
        for offset in (0., .03125, -.0625):
            x.add_(offset)
            xc, wc = x.cpu().cuda(), w.cpu().cuda()
            rounded = (xc.float() * wc + .125).to(dtype)
            expected = torch.nn.functional.silu(rounded) * rounded
            before = backend.execution_stats()['kernel_launches']
            actual = graph.replay()
            launches = backend.execution_stats()['kernel_launches'] - before
            assert launches < len(graph._plan.nodes)
            torch.testing.assert_close(actual['rounded'].cpu(), rounded.cpu())
            torch.testing.assert_close(actual['product'].cpu(), expected.cpu(), rtol=.02, atol=.002)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('shape', [(3, 513), (2, 3, 1024), (4, 256)])
def test_compiled_reductions_softmax(backend, dtype, shape):
    def fn(x):
        return x.sum(-1), x.mean(0), x.softmax(-1), x.log_softmax(-1)
    source = torch.randn(shape, dtype=dtype)
    compiled = backend.compile(fn, native='required', min_native_ops=1, fullgraph=True)
    try:
        for offset in (0., .125):
            values = source + offset
            actual = compiled(values.to('ruda'))
            expected = fn(values.cuda())
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a.cpu(), b.cpu(), rtol=.025 if dtype == torch.bfloat16 else .003,
                                           atol=.025 if dtype == torch.bfloat16 else .003)
    finally:
        compiled.close()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('m,n,k', [(3, 129, 256), (64, 128, 256), (17, 33, 19), (4, 7, 0)])
def test_transposed_matrix_graph(backend, dtype, m, n, k):
    from ruda_torch import GraphOp, StaticGraph
    a = torch.randn(m, k, dtype=dtype).to('ruda')
    w = torch.randn(n, k, dtype=dtype).to('ruda')
    nodes = [GraphOp('permute_copy', 'wt', 'w', scalar=1, shape=(k, n)),
             GraphOp('mm', 'y', 'a', 'wt')]
    with StaticGraph({'a': a, 'w': w}, nodes, outputs=['y'], optimize=True) as graph:
        assert graph._plan.nodes[0].kind == 'mm_rhs_transposed'
        for offset in (0., .125):
            a.add_(offset)
            expected = a.cpu().cuda() @ w.cpu().cuda().t()
            actual = graph.replay()['y']
            tolerance = .04 if dtype == torch.bfloat16 else .005 if dtype == torch.float16 else 2e-5
            torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_fp4_long_k_split(backend, dtype):
    m, n, k = 2, 128, 4096
    a = torch.randint(0, 256, (m, k // 2), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    sa = torch.full((128 * (k // 16),), .5).to(torch.float8_e4m3fn)
    sb = torch.full((128 * (k // 16),), .25).to(torch.float8_e4m3fn)
    expected = torch._scaled_mm(a.cuda(), w.cuda().t(), sa.cuda(), sb.cuda(), out_dtype=dtype)
    before = backend.execution_stats()['kernel_launches']
    actual = torch._scaled_mm(a.to('ruda'), w.to('ruda').t(), sa.to('ruda'), sb.to('ruda'), out_dtype=dtype)
    assert backend.execution_stats()['kernel_launches'] - before >= 2
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=.02 if dtype == torch.bfloat16 else 2e-5,
                               atol=.02 if dtype == torch.bfloat16 else 2e-5)
