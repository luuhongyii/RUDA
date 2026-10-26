"""Native dispatcher entrypoints must preserve the original fused operator bodies."""
import itertools
import pytest
import torch
import ruda_torch as r
from ruda_torch import _fused_training_ops as training, fused_cross_entropy as ce, rotary


def same(actual, expected):
    if isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected): same(a, b)
    elif expected is None:
        assert actual is None
    else:
        assert actual.shape == expected.shape and actual.stride() == expected.stride()
        assert actual.dtype == expected.dtype and actual.device == expected.device
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize('name', ['training_rms', 'training_layer', 'training_silu', 'rotary_qk', 'cross_entropy'])
def test_fused_dispatch_has_native_entrypoints(name):
    for direction in ('forward', 'backward'):
        assert torch._C._dispatch_has_kernel_for_dispatch_key('ruda::' + name + '_' + direction, 'PrivateUse1')


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('shape', [(0, 129), (7, 129), (2, 3, 512)])
@pytest.mark.parametrize('hf', [False, True])
@torch.no_grad()
def test_native_rms_dispatch_matches_body_with_optional_gradients(dtype, shape, hf):
    torch.manual_seed(261049)
    x = torch.randn(shape, dtype=dtype).to('ruda')
    dy = torch.randn_like(x)
    weights = [None, torch.randn(shape[-1], dtype=dtype).to('ruda')]
    if not hf and dtype != torch.float32: weights.append(weights[-1].float())
    for weight in weights:
        actual = training.rms_forward(x, weight, 1e-6, hf)
        expected = training.rms_forward._init_fn(x, weight, 1e-6, hf)
        same(actual, expected)
        for need_x, need_w in itertools.product([False, True], repeat=2):
            if weight is None and need_w: continue
            args = (x, weight, dy, actual[1], need_x, need_w, hf)
            same(training.rms_backward(*args), training.rms_backward._init_fn(*args))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('affine', list(itertools.product([False, True], repeat=2)))
@torch.no_grad()
def test_native_layer_dispatch_matches_body_with_optional_gradients(dtype, affine):
    torch.manual_seed(261050)
    x = torch.randn(2, 3, 129, dtype=dtype).to('ruda')
    dy = torch.randn_like(x)
    weight, bias = [torch.randn(129).to('ruda') if present else None for present in affine]
    output = training.layer_forward(x, weight, bias, 1e-5)
    same(output, training.layer_forward._init_fn(x, weight, bias, 1e-5))
    for needs in itertools.product([False, True], repeat=3):
        if any(need and value is None for need, value in zip(needs[1:], (weight, bias))): continue
        args = (x, weight, bias, dy, output[1], output[2], *needs)
        same(training.layer_backward(*args), training.layer_backward._init_fn(*args))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('shape', [(0, 129), (3, 129), (2, 3, 512)])
@torch.no_grad()
def test_native_silu_dispatch_matches_body_with_optional_gradients(dtype, shape):
    torch.manual_seed(261051)
    gate, up, grad = [torch.randn(shape, dtype=dtype).to('ruda') for _ in range(3)]
    same(training.silu_forward(gate, up), training.silu_forward._init_fn(gate, up))
    for needs in itertools.product([False, True], repeat=2):
        args = (gate, up, grad, *needs)
        same(training.silu_backward(*args), training.silu_backward._init_fn(*args))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@torch.no_grad()
def test_native_rotary_dispatch_matches_strided_body_and_missing_gradients(dtype):
    torch.manual_seed(261052)
    q = torch.randn(2, 17, 4, 130, dtype=dtype).to('ruda').transpose(1, 2)
    k = torch.randn(2, 17, 2, 130, dtype=dtype).to('ruda').transpose(1, 2)
    cos, sin = [torch.randn(2, 1, 17, 130, dtype=dtype).to('ruda') for _ in range(2)]
    versions = [t._version for t in (q, k, cos, sin)]
    actual = rotary._forward(q, k, cos, sin)
    same(actual, rotary._forward._init_fn(q, k, cos, sin))
    assert [t._version for t in (q, k, cos, sin)] == versions
    assert all(a.is_contiguous() for a in actual)
    for need_q, need_k in itertools.product([False, True], repeat=2):
        args = (q if need_q else None, k if need_k else None, cos, sin, list(q.shape), list(k.shape))
        same(rotary._backward(*args), rotary._backward._init_fn(*args))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('rows', [0, 7])
@torch.no_grad()
def test_native_cross_entropy_dispatch_matches_body_and_broadcast_gradient(dtype, rows):
    torch.manual_seed(261053)
    x = torch.randn(rows, 2053, dtype=dtype).to('ruda')
    target = torch.arange(rows, dtype=torch.int64).to('ruda')
    ignore = 2
    actual = ce._forward(x, target, ignore)
    same(actual, ce._forward._init_fn(x, target, ignore))
    gradient = torch.tensor(.375, device='ruda').expand(rows)
    args = (x, target, actual[1], gradient, ignore)
    same(ce._backward(*args), ce._backward._init_fn(*args))


@pytest.mark.parametrize('kind', ['rms', 'layer', 'silu', 'rotary', 'cross_entropy'])
def test_native_fused_dispatch_schema_fake_and_aot(kind):
    x = torch.randn(3, 129, dtype=torch.bfloat16).to('ruda')
    w = torch.ones(129, dtype=torch.bfloat16).to('ruda')
    if kind == 'rms': fn, args = training.rms_forward, (x, w, 1e-6, True)
    elif kind == 'layer': fn, args = training.layer_forward, (x, w, None, 1e-5)
    elif kind == 'silu': fn, args = training.silu_forward, (x, x)
    elif kind == 'rotary':
        q = torch.randn(1, 2, 17, 130, dtype=torch.bfloat16).to('ruda').requires_grad_()
        k = q[:, :1].detach().requires_grad_()
        c, s = [torch.randn(1, 1, 17, 130, dtype=torch.bfloat16).to('ruda') for _ in range(2)]
        fn, args = rotary._forward, (q, k, c, s)
    else: fn, args = ce._forward, (x.requires_grad_(), torch.arange(3).to('ruda'), -100)
    torch.library.opcheck(fn, args, test_utils=('test_schema', 'test_faketensor', 'test_aot_dispatch_dynamic'))


def test_native_layer_dispatch_aot_forward_backward_dynamic_rows():
    from torch._dynamo.backends.common import aot_autograd
    graphs = []
    def capture(graph, inputs):
        graphs.append(graph)
        return graph.forward
    def function(x, w, b): return r.layer_norm(x, w, b, eps=1e-5).float().square().sum()
    compiled = torch.compile(function, backend=aot_autograd(fw_compiler=capture, bw_compiler=capture),
                             fullgraph=True, dynamic=True)
    for rows in (3, 7):
        args = [torch.randn(rows, 129).to('ruda').requires_grad_(),
                torch.randn(129).to('ruda').requires_grad_(), torch.randn(129).to('ruda').requires_grad_()]
        expected = torch.autograd.grad(function(*args), args)
        actual = torch.autograd.grad(compiled(*args), args)
        same(actual, expected)
    targets = {str(node.target) for graph in graphs for node in graph.graph.nodes}
    assert any('training_layer_forward' in name for name in targets)
    assert any('training_layer_backward' in name for name in targets)


@pytest.mark.parametrize('kind', ['rms', 'layer', 'silu', 'rotary', 'cross_entropy'])
def test_native_fused_dispatch_on_nondefault_stream(kind):
    stream = r.Stream()
    with r.stream(stream):
        x = torch.ones(3, 128, dtype=torch.bfloat16, device='ruda').requires_grad_()
        if kind == 'rms': output = r.rms_norm(x, eps=1e-6)
        elif kind == 'layer': output = r.layer_norm(x, eps=1e-5)
        elif kind == 'silu': output = r.silu_mul(x, x)
        elif kind == 'cross_entropy': output = r.cross_entropy(x, torch.arange(3, device='ruda'))
        else:
            q = x.reshape(1, 1, 3, 128)
            c = torch.ones(1, 3, 128, dtype=x.dtype, device='ruda')
            output = r.apply_rotary_pos_emb(q, q, c, c)[0]
        output.float().sum().backward()
    stream.synchronize()
    assert x.grad is not None and torch.isfinite(x.grad.cpu()).all()
