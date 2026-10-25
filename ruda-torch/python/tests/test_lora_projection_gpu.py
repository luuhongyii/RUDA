import itertools
import pytest
import torch
from torch.nn import functional as F
import ruda_torch as r
from ruda_torch.lora_epilogue import _rank16_residual, rank16_residual_eligible


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('shape', [(16, 128), (1, 32, 256), (2, 256, 512)])
@pytest.mark.parametrize('scale', [1., 1. / 3., 1.75])
def test_rank16_fusion_preserves_native_forward_and_gradients(dtype, shape, scale):
    assert r._lora_projection_available
    torch.manual_seed(261030)
    hosts = [torch.randn(shape, dtype=dtype), torch.randn((*shape[:-1], 16)),
             torch.randn(shape[-1], 16)]
    fused = [x.to('ruda').requires_grad_() for x in hosts]
    reference = [x.to('ruda').requires_grad_() for x in hosts]
    actual = _rank16_residual(*fused, scale)
    expected = r.lora_epilogue(reference[0], F.linear(reference[1], reference[2]), scale)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    gradient = torch.randn(shape[-1], dtype=dtype).to('ruda').expand(shape)
    actual.backward(gradient); expected.backward(gradient)
    for a, b in zip(fused, reference):
        torch.testing.assert_close(a.grad.cpu(), b.grad.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize('needs', list(itertools.product([False, True], repeat=3)))
def test_rank16_fusion_selected_gradients(needs):
    hosts = [torch.ones(16, 128, dtype=torch.bfloat16), torch.ones(16, 16), torch.ones(128, 16)]
    inputs = [x.to('ruda').requires_grad_(need) for x, need in zip(hosts, needs)]
    result = _rank16_residual(*inputs, .5)
    assert result.requires_grad == any(needs)
    if any(needs):
        result.sum().backward()
        for index, (x, need) in enumerate(zip(inputs, needs)):
            if need:
                expected = torch.full(x.shape, [1., 64., 8.][index], dtype=x.dtype)
                torch.testing.assert_close(x.grad.cpu(), expected, rtol=0, atol=0)
            else:
                assert x.grad is None


def test_rank16_fusion_rounding_and_nonfinite_values():
    base = torch.full((16, 128), -1., dtype=torch.bfloat16)
    rank = torch.zeros(16, 16); rank[:, 0] = 1.
    weight = torch.zeros(128, 16); weight[:, 0] = 1.003
    args = [x.to('ruda') for x in (base, rank, weight)]
    torch.testing.assert_close(_rank16_residual(*args, 1.).cpu(), torch.zeros_like(base), rtol=0, atol=0)
    rank[0, 1] = float('nan'); weight[1, 2] = float('inf'); base[2, 2] = float('-inf')
    args = [x.to('ruda') for x in (base, rank, weight)]
    expected = r.lora_epilogue(args[0], F.linear(args[1], args[2]), 1.)
    torch.testing.assert_close(_rank16_residual(*args, 1.).cpu(), expected.cpu(), rtol=0, atol=0, equal_nan=True)


def test_rank16_fusion_rejects_unaligned_weight_and_row_tail():
    base = torch.ones(32, 128, dtype=torch.bfloat16).to('ruda')
    rank = torch.ones(32, 16).to('ruda')
    weight = torch.ones(128 * 16 + 1).to('ruda')[1:].reshape(128, 16)
    assert not rank16_residual_eligible(base, rank, weight)
    assert not rank16_residual_eligible(base[:31], rank[:31], weight.clone())


def test_rank16_fusion_preserves_second_derivatives():
    inputs = [torch.ones(16, 128, dtype=torch.bfloat16), torch.ones(16, 16), torch.ones(128, 16)]
    derivatives = []
    for fused in (False, True):
        base, rank, weight = [x.to('ruda').requires_grad_() for x in inputs]
        out = _rank16_residual(base, rank, weight, .5) if fused else r.lora_epilogue(base, F.linear(rank, weight), .5)
        first = torch.autograd.grad(out.float().square().sum(), rank, create_graph=True)[0]
        derivatives.append(torch.autograd.grad(first.sum(), weight)[0].cpu())
    torch.testing.assert_close(*derivatives, rtol=0, atol=0)


def test_rank16_fusion_aot_forward_and_backward():
    from torch._dynamo.backends.common import aot_autograd
    graphs = []
    def capture(graph, inputs):
        graphs.append(graph)
        return graph.forward
    compiled = torch.compile(lambda base, rank, weight: _rank16_residual(base, rank, weight, .5).sum(),
                             backend=aot_autograd(fw_compiler=capture, bw_compiler=capture), fullgraph=True)
    inputs = [torch.ones(16, 128, dtype=torch.bfloat16).to('ruda').requires_grad_(),
              torch.ones(16, 16).to('ruda').requires_grad_(),
              torch.ones(128, 16).to('ruda').requires_grad_()]
    compiled(*inputs).backward()
    assert all(x.grad is not None for x in inputs)
    assert any('lora_rank16_residual' in str(node.target) for graph in graphs for node in graph.graph.nodes)
