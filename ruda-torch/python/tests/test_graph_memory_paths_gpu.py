import os
import pytest
import torch


@pytest.fixture(scope='module')
def backend():
    if os.environ.get('RUDA_REQUIRE_GPU') != '1':
        pytest.skip('explicit native GPU execution required')
    import ruda_torch as r
    assert r.get_math_backend() == 'ruda'
    return r


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('weight_shape,offset', [((128,), 0), ((1, 3, 128), 0), ((), 0), ((2, 1, 128), 0), ((128,), 1)])
def test_vector_chain_broadcast_and_storage_casts(backend, dtype, weight_shape, offset):
    from ruda_torch import GraphOp, StaticGraph
    shape = (2, 3, 128)
    cpu = torch.randn(shape, dtype=dtype)
    storage = torch.empty(cpu.numel() + offset, dtype=dtype, device='ruda')
    x = storage[offset:].view(shape)
    x.copy_(cpu.to('ruda'))
    weight = torch.randn(weight_shape)
    w = weight.to('ruda')
    nodes = [GraphOp('cast', 'wide', 'x', dtype='float32'),
             GraphOp('mul_broadcast', 'scaled', 'wide', 'w'),
             GraphOp('add_scalar', 'shifted', 'scaled', scalar=.125),
             GraphOp('cast', 'rounded', 'shifted', dtype=str(dtype).removeprefix('torch.')),
             GraphOp.mul('product', 'rounded', 'rounded')]
    with StaticGraph({'x': x, 'w': w}, nodes, outputs=['rounded', 'product'], optimize=False) as graph:
        for shift in (0., .125):
            values = cpu + shift
            x.copy_(values.to('ruda'))
            rounded = (values.cuda().float() * weight.cuda() + .125).to(dtype)
            result = graph.replay()
            torch.testing.assert_close(result['rounded'].cpu(), rounded.cpu(), rtol=0, atol=0)
            torch.testing.assert_close(result['product'].cpu(), (rounded * rounded).cpu(), rtol=0, atol=0)


@pytest.mark.parametrize('reuse', [False, True])
def test_fusion_workspace_lifetimes(backend, reuse):
    from ruda_torch import GraphOp, StaticGraph
    cpu = torch.randn(8, 128)
    x = cpu.to('ruda')
    nodes = [GraphOp('mul_scalar', 'p' + str(i), 'x' if i == 0 else 'p' + str(i - 1), scalar=.5) for i in range(10)]
    with StaticGraph({'x': x}, nodes, outputs=['p9'], reuse_workspace=reuse) as graph:
        for shift in (0., 1.):
            x.copy_((cpu + shift).to('ruda'))
            torch.testing.assert_close(graph.replay()['p9'].cpu(), (cpu + shift) / 1024, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('keepdim', [False, True])
@pytest.mark.parametrize('shape,axes', [((129, 4), (1,)), ((7, 1), (1,)), ((131, 512), (1,)),
    ((3, 4096), (1,)), ((3, 513), (1,)), ((65, 129), (0,)), ((2, 64, 65), (1,)),
    ((2, 3, 64, 33), (1, 2)), ((2, 3, 65), (0, 2))])
def test_reduction_layouts(backend, dtype, keepdim, shape, axes):
    from ruda_torch import GraphOp, StaticGraph
    values = torch.randn(shape, dtype=dtype) * .125
    x = values.to('ruda')
    mask = sum(1 << d for d in axes)
    nodes = [GraphOp(key + ('_keepdim' if keepdim else ''), key, 'x', scalar=mask) for key in ('sum', 'mean')]
    with StaticGraph({'x': x}, nodes, outputs=['sum', 'mean']) as graph:
        actual = graph.replay()
        for key in ('sum', 'mean'):
            expected = getattr(values.cuda(), key)(axes, keepdim=keepdim)
            tolerance = .02 if dtype == torch.bfloat16 else .003 if dtype == torch.float16 else 2e-4
            torch.testing.assert_close(actual[key].cpu(), expected.cpu(), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64, torch.uint8])
@pytest.mark.parametrize('shape', [(33, 65), (2, 17, 67), (2, 3, 129, 257)])
def test_tiled_transpose_and_copy(backend, dtype, shape):
    from ruda_torch import GraphOp, StaticGraph
    values = torch.randint(0, 100, shape).to(dtype)
    x = values.to('ruda')
    expected = values.transpose(-1, -2).contiguous()
    copied = x.transpose(-1, -2).contiguous()
    torch.testing.assert_close(copied.cpu(), expected, rtol=0, atol=0)
    permutation = list(range(len(shape)))
    permutation[-1], permutation[-2] = permutation[-2], permutation[-1]
    code = sum(axis << (3 * d) for d, axis in enumerate(permutation))
    nodes = [GraphOp('permute_copy', 'y', 'x', scalar=code, shape=tuple(expected.shape))]
    with StaticGraph({'x': x}, nodes, outputs=['y']) as graph:
        torch.testing.assert_close(graph.replay()['y'].cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_transpose_cast_and_offset(backend, dtype):
    storage = torch.randn(1 + 33 * 65, dtype=dtype).to('ruda')
    x = storage[1:].view(33, 65).t()
    expected = x.cpu().float().contiguous()
    output = torch.empty(x.shape, dtype=torch.float32, device='ruda')
    output.copy_(x)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
