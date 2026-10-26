import pytest
import torch
import ruda_torch as r
from ruda_torch import _arithmetic, _ops


NAMES = ['add.Tensor', 'add_.Tensor', 'sub.Tensor', 'sub_.Tensor',
         'mul.Tensor', 'mul_.Tensor', 'neg', 'neg_']


def operator(name):
    base, _, overload = name.partition('.')
    return getattr(getattr(torch.ops.aten, base), overload or 'default')


def original(name, a, b, alpha=.5):
    op = {'add': 0, 'sub': 1, 'mul': 2, 'neg': 3}[name.split('.')[0].rstrip('_')]
    return _arithmetic.arithmetic(a, b, op=op, alpha=alpha, inplace=name.split('.')[0].endswith('_'))


def invoke(name, a, b, alpha=.5):
    args = (a,) if name.startswith('neg') else (a, b)
    kwargs = {'alpha': alpha} if name.startswith(('add', 'sub')) else {}
    return operator(name)(*args, **kwargs)


def same(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.stride() == expected.stride() and actual.device == expected.device
    assert torch.equal(actual.cpu().contiguous().reshape(-1).view(torch.uint8),
                       expected.cpu().contiguous().reshape(-1).view(torch.uint8))


@pytest.mark.parametrize('name', NAMES)
def test_arithmetic_dispatch_has_cpp_entrypoint(name):
    assert _ops._cpp_arithmetic_dispatch
    table = torch._C._dispatch_dump_table('aten::' + name)
    entry = next(line for line in table.splitlines() if line.startswith('PrivateUse1:'))
    assert 'backend.cpp' in entry


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize('layout', ['contiguous', 'transpose', 'offset'])
@pytest.mark.parametrize('name', NAMES)
def test_arithmetic_dispatch_matches_original_bits_and_cpu(name, layout, dtype):
    source = torch.arange(41, dtype=torch.float64).sub(20).div(8).to(dtype)
    def tensor():
        value = source.to('ruda')
        value = value[1:] if layout == 'offset' else value[:40]
        value = value.reshape(5, 8)
        return value.t() if layout == 'transpose' else value
    a, b, old_a, old_b = (tensor() for _ in range(4))
    cpu_a, cpu_b = a.cpu(), b.cpu()
    version = a._version
    actual, expected = invoke(name, a, b), original(name, old_a, old_b)
    same(actual, expected)
    torch.testing.assert_close(actual.cpu(), invoke(name, cpu_a, cpu_b), rtol=0, atol=0)
    if name.split('.')[0].endswith('_'):
        assert actual.data_ptr() == a.data_ptr() and a._version == version + 1
    else:
        assert actual.data_ptr() != a.data_ptr() and a._version == version


@pytest.mark.parametrize('name', NAMES)
def test_arithmetic_dispatch_common_path_never_enters_python(name, monkeypatch):
    a = torch.ones(5, 8, device='ruda')
    b = torch.full((5, 8), 2., device='ruda')
    expected = invoke(name, torch.ones(5, 8), torch.full((5, 8), 2.))
    def unexpected(*args, **kwargs): raise AssertionError('entered Python arithmetic')
    monkeypatch.setattr(_arithmetic, 'arithmetic', unexpected)
    torch.testing.assert_close(invoke(name, a, b).cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize('name', NAMES)
def test_arithmetic_dispatch_empty_scalar_alias_and_special_values(name):
    for cpu in (torch.empty(2, 0, 3), torch.tensor(2.),
                torch.tensor([0., -0., float('inf'), -float('inf'), float('nan'), 1.])):
        a, old_a = cpu.to('ruda'), cpu.to('ruda')
        same(invoke(name, a, a), original(name, old_a, old_a))


@pytest.mark.parametrize('name', ['add.Tensor', 'sub.Tensor', 'mul.Tensor'])
@pytest.mark.parametrize('case', ['broadcast', 'mixed', 'cpu_scalar', 'integral', 'complex', 'negative_view'])
def test_arithmetic_dispatch_preserves_fallback(name, case, monkeypatch):
    a = torch.arange(8, dtype=torch.float32).reshape(2, 4).to('ruda')
    b = a.clone()
    if case == 'broadcast': b = b[:1]
    elif case == 'mixed': b = b.to(torch.float16)
    elif case == 'cpu_scalar': b = torch.tensor(2.)
    elif case == 'integral': a, b = a.to(torch.int64), b.to(torch.int64)
    elif case == 'complex': a, b = a.to(torch.complex64), b.to(torch.complex64)
    else: a = torch._neg_view(a)
    expected = original(name, a, b, alpha=1)
    body = _arithmetic.arithmetic
    calls = []
    def observed(*args, **kwargs):
        calls.append(kwargs)
        return body(*args, **kwargs)
    monkeypatch.setattr(_arithmetic, 'arithmetic', observed)
    same(invoke(name, a, b, alpha=1), expected)
    if case != 'negative_view': assert calls


@pytest.mark.parametrize('name', ['add_.Tensor', 'sub_.Tensor', 'mul_.Tensor'])
def test_arithmetic_dispatch_partial_overlap_errors_match_original(name):
    for run in (original, invoke):
        a = torch.ones(9, device='ruda')
        with pytest.raises(RuntimeError) as error: run(name, a[:8], a[1:])
        if run is original: expected = str(error.value)
        else: assert str(error.value) == expected


@pytest.mark.parametrize('name', ['add.Tensor', 'sub.Tensor'])
def test_arithmetic_dispatch_alpha_matches_original(name):
    a = torch.tensor([1., -0., .75, 32.], device='ruda')
    b = torch.tensor([-.25, 2., 3., 4.], device='ruda')
    for alpha in (0, -1, -.125, 2**40, float('inf'), float('nan')):
        same(invoke(name, a, b, alpha), original(name, a, b, alpha))
    for alpha in (True, 1j):
        with pytest.raises(Exception) as expected: original(name, a, b, alpha)
        with pytest.raises(type(expected.value)) as actual: invoke(name, a, b, alpha)
        assert str(actual.value) == str(expected.value)


def test_arithmetic_dispatch_first_and_second_gradients_and_stream():
    def gradients(device):
        a = torch.arange(1, 9, dtype=torch.float32).to(device).requires_grad_()
        b = torch.arange(9, 17, dtype=torch.float32).to(device).requires_grad_()
        loss = (a * b + a - b).neg().square().sum()
        first = torch.autograd.grad(loss, (a, b), create_graph=True)
        second = torch.autograd.grad(first[0].sum() + first[1].sum(), (a, b))
        return first + second
    expected = gradients('cpu')
    stream = r.Stream()
    with r.stream(stream): actual = gradients('ruda')
    stream.synchronize()
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), b, rtol=0, atol=0)


def test_arithmetic_dispatch_aot_dynamic_autograd():
    from torch._dynamo.backends.common import aot_autograd
    graphs = []
    def capture(graph, inputs):
        graphs.append(graph)
        return graph.forward
    def function(a, b): return ((a + b) * (a - b)).neg().sum()
    compiled = torch.compile(function, backend=aot_autograd(fw_compiler=capture, bw_compiler=capture),
                             fullgraph=True, dynamic=True)
    for rows in (3, 7):
        args = [torch.randn(rows, 5).to('ruda').requires_grad_() for _ in range(2)]
        expected = torch.autograd.grad(function(*args), args)
        actual = torch.autograd.grad(compiled(*args), args)
        for a, b in zip(actual, expected): same(a, b)
    assert len(graphs) >= 2
