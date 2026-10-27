import itertools

import pytest
import torch
import ruda_torch as r
from ruda_torch import _dense_ops as dense, fused_cross_entropy as ce, rotary


def same(actual, expected):
    if actual is None or expected is None:
        assert actual is expected
    else:
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('reduction', ['none', 'sum', 'mean'])
def test_eager_ce_preserves_reductions_and_gradients(dtype, reduction):
    torch.manual_seed(261029)
    x = torch.randn(7, 129, dtype=dtype).to('ruda').requires_grad_()
    target = torch.tensor([0, 17, -100, 2, 128, -100, 4]).to('ruda')
    expected = ce._forward(x, target, -100)[0]
    if reduction != 'none': expected = expected.sum()
    if reduction == 'mean': expected = expected / (target != -100).sum().clamp_min(1).to(torch.float32)
    actual = r.cross_entropy(x, target, reduction=reduction)
    same(actual, expected)
    grad = torch.tensor(.375, device='ruda').expand_as(actual)
    same(torch.autograd.grad(actual, x, grad)[0], torch.autograd.grad(expected, x, grad)[0])


@pytest.mark.parametrize('needs', list(itertools.product([False, True], repeat=2)))
@pytest.mark.parametrize('used', [0, 1, 2])
def test_eager_rotary_preserves_selected_and_unused_gradients(needs, used):
    torch.manual_seed(261030)
    q = torch.randn(1, 17, 4, 128, dtype=torch.bfloat16).to('ruda').transpose(1, 2).requires_grad_(needs[0])
    k = torch.randn(1, 17, 2, 128, dtype=torch.bfloat16).to('ruda').transpose(1, 2).requires_grad_(needs[1])
    c, s = [torch.randn(1, 17, 128, dtype=torch.bfloat16).to('ruda') for _ in range(2)]
    actual = r.apply_rotary_pos_emb(q, k, c, s)
    expected = rotary._forward(q, k, c.unsqueeze(1), s.unsqueeze(1))
    for a, b in zip(actual, expected):
        same(a, b); assert a.requires_grad == b.requires_grad
    selected = [i for i in range(2) if actual[i].requires_grad and (used == 2 or used == i)]
    if selected:
        arguments = [v for v in (q, k) if v.requires_grad]
        a = torch.autograd.grad(tuple(actual[i] for i in selected), arguments,
            tuple(torch.ones_like(actual[i]) for i in selected), allow_unused=True)
        b = torch.autograd.grad(tuple(expected[i] for i in selected), arguments,
            tuple(torch.ones_like(expected[i]) for i in selected), allow_unused=True)
        for x, y in zip(a, b): same(x, y)


@pytest.mark.parametrize('needs', list(itertools.product([False, True], repeat=3)))
def test_eager_flash_preserves_all_gradient_selections(needs):
    torch.manual_seed(261031)
    q = torch.randn(1, 4, 512, 128, dtype=torch.bfloat16).to('ruda').requires_grad_(needs[0])
    k, v = [torch.randn(1, 2, 512, 128, dtype=torch.bfloat16).to('ruda').requires_grad_(need) for need in needs[1:]]
    args = (q, k, v, torch.zeros(1, device='ruda'), [1, 4, 512, 512], 0, True, 128**-.5, 0., True)
    expected = dense.forward(*args)
    actual = dense.eager_forward(*args)
    for a, b in zip(actual, expected): same(a, b)
    assert actual[0].requires_grad == expected[0].requires_grad == any(needs)
    assert not any(x.requires_grad for x in actual[1:])
    if any(needs):
        wanted = [x for x in (q, k, v) if x.requires_grad]
        dy = torch.ones_like(q)
        a = torch.autograd.grad(actual[0], wanted, dy)
        b = torch.autograd.grad(expected[0], wanted, dy)
        for x, y in zip(a, b): same(x, y)


@pytest.mark.parametrize('p', [0., .25, 1.])
def test_eager_attention_preserves_rng_and_mask_gradient(p):
    from test_dense_dispatch_gpu import same_atomic_gradient
    torch.manual_seed(261032)
    q, k, v = [torch.randn(1, 2, 17, 32).to('ruda').requires_grad_() for _ in range(3)]
    mask = torch.randn(1, 1, 1, 17).to('ruda').requires_grad_()
    args = (q, k, v, mask, [1, 2, 17, 17], 2, False, .125, p, False)
    r.manual_seed(293); expected = dense.forward(*args)
    r.manual_seed(293); actual = dense.eager_forward(*args)
    for a, b in zip(actual, expected): same(a, b)
    a = torch.autograd.grad(actual[0].sum(), (q, k, v, mask))
    b = torch.autograd.grad(expected[0].sum(), (q, k, v, mask))
    same(a[0], b[0])
    for x, y in zip(a[1:], b[1:]): same_atomic_gradient(x, y)


def case(kind):
    torch.manual_seed(261033)
    if kind == 'ce':
        x = torch.randn(3, 129).to('ruda').requires_grad_()
        target = torch.arange(3).to('ruda')
        return lambda x: r.cross_entropy(x, target, reduction='sum'), (x,), x
    if kind == 'rope':
        q = torch.randn(1, 2, 17, 32).to('ruda').requires_grad_()
        c, s = [torch.randn(1, 17, 32).to('ruda') for _ in range(2)]
        return lambda q: r.apply_rotary_pos_emb(q, q, c, s)[0].sum(), (q,), c
    q = torch.randn(1, 4, 512, 128, dtype=torch.bfloat16).to('ruda').requires_grad_()
    mask = torch.zeros(1, device='ruda')
    return lambda q: dense.eager_forward(q, q, q, mask, [1, 4, 512, 512],
        0, True, 128**-.5, 0., True)[0].sum(), (q,), q


@pytest.mark.parametrize('kind', ['ce', 'rope', 'attention'])
def test_eager_fusion_preserves_checkpoint(kind):
    from torch.utils.checkpoint import checkpoint
    fn, args, _ = case(kind)
    plain = fn(*args)
    checked = checkpoint(fn, *args, use_reentrant=False)
    same(checked, plain)
    for x, y in zip(torch.autograd.grad(checked, args), torch.autograd.grad(plain, args)): same(x, y)


@pytest.mark.parametrize('kind', ['ce', 'rope', 'attention'])
def test_eager_fusion_preserves_saved_tensor_versions(kind):
    fn, args, saved = case(kind)
    result = fn(*args)
    with torch.no_grad(): saved.add_(1)
    with pytest.raises(RuntimeError, match='modified by an inplace operation'): result.backward()


@pytest.mark.parametrize('kind', ['ce', 'attention'])
def test_eager_fusion_preserves_aot_boundary(kind):
    from torch._dynamo.backends.common import aot_autograd
    fn, args, _ = case(kind)
    graphs = []
    def capture(graph, args):
        graphs.append(graph)
        return graph.forward
    compiled = torch.compile(fn, backend=aot_autograd(fw_compiler=capture, bw_compiler=capture), fullgraph=True)
    actual = torch.autograd.grad(compiled(*args), args)
    expected = torch.autograd.grad(fn(*args), args)
    for x, y in zip(actual, expected): same(x, y)
    op = {'ce': 'cross_entropy_forward', 'rope': 'rotary_qk_forward', 'attention': 'dense_attention_forward'}[kind]
    assert any(op in str(node.target) for graph in graphs for node in graph.graph.nodes)
