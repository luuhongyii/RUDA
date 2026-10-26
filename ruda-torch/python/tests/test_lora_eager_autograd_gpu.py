import itertools

import pytest
import torch
from ruda_torch.lora_epilogue import _rank16_residual, rank16_residual


def inputs(dtype=torch.bfloat16, needs=(True, True, True)):
    torch.manual_seed(261028)
    values = [torch.randn(32, 256, dtype=dtype), torch.randn(32, 16), torch.randn(256, 16)]
    return [x.to('ruda').requires_grad_(need) for x, need in zip(values, needs)]


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('needs', list(itertools.product([False, True], repeat=3)))
def test_eager_lora_preserves_outputs_and_selected_gradients(dtype, needs):
    values = inputs(dtype, needs)
    expected = _rank16_residual(*values, .375)
    actual = rank16_residual(*values, .375)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    assert actual.requires_grad == expected.requires_grad == any(needs)
    if any(needs):
        wanted = [x for x in values if x.requires_grad]
        grad = torch.linspace(-1., 1., 256, dtype=dtype).to('ruda').expand_as(actual)
        a = torch.autograd.grad(actual, wanted, grad)
        b = torch.autograd.grad(expected, wanted, grad)
        for x, y in zip(a, b):
            torch.testing.assert_close(x.cpu(), y.cpu(), rtol=0, atol=0)


def test_eager_lora_preserves_second_derivatives():
    results = []
    for function in (_rank16_residual, rank16_residual):
        base, rank, weight = inputs()
        loss = function(base, rank, weight, .375).float().square().sum()
        first = torch.autograd.grad(loss, rank, create_graph=True)[0]
        results.append(torch.autograd.grad(first.sum(), weight)[0].cpu())
    torch.testing.assert_close(*results, rtol=0, atol=0)


def test_eager_lora_preserves_checkpoint_recomputation():
    from torch.utils.checkpoint import checkpoint
    values = inputs()
    plain = _rank16_residual(*values, .375)
    checked = checkpoint(lambda *xs: rank16_residual(*xs, .375), *values, use_reentrant=False)
    torch.testing.assert_close(checked.cpu(), plain.cpu(), rtol=0, atol=0)
    a = torch.autograd.grad(checked.float().square().sum(), values)
    b = torch.autograd.grad(plain.float().square().sum(), values)
    for x, y in zip(a, b):
        torch.testing.assert_close(x.cpu(), y.cpu(), rtol=0, atol=0)


def test_eager_lora_preserves_saved_tensor_version_checks():
    values = inputs()
    loss = rank16_residual(*values, .375).sum()
    with torch.no_grad():
        values[2].add_(1)
    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        loss.backward()


def test_eager_lora_keeps_native_aot_boundary():
    from torch._dynamo.backends.common import aot_autograd
    graphs = []
    def capture(graph, args):
        graphs.append(graph)
        return graph.forward
    compiled = torch.compile(lambda *xs: rank16_residual(*xs, .375).sum(),
        backend=aot_autograd(fw_compiler=capture, bw_compiler=capture), fullgraph=True)
    values = inputs()
    a = torch.autograd.grad(compiled(*values), values)
    b = torch.autograd.grad(_rank16_residual(*values, .375).sum(), values)
    for x, y in zip(a, b):
        torch.testing.assert_close(x.cpu(), y.cpu(), rtol=0, atol=0)
    assert any('lora_rank16_residual' in str(node.target) for graph in graphs for node in graph.graph.nodes)
