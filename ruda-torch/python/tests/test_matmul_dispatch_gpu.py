import asyncio
import contextvars
import itertools
import pytest
import torch
import ruda_torch as r
from ruda_torch import _ops


def same(actual, expected):
    assert actual.shape == expected.shape and actual.stride() == expected.stride()
    assert actual.dtype == expected.dtype and actual.device == expected.device
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize('name', ['mm', 'bmm', 'addmm'])
def test_matmul_dispatch_has_cpp_entrypoint(name):
    assert _ops._cpp_matmul_dispatch
    table = torch._C._dispatch_dump_table('aten::' + name)
    entry = next(line for line in table.splitlines() if line.startswith('PrivateUse1:'))
    assert 'backend.cpp' in entry


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('name', ['mm', 'bmm', 'addmm'])
@pytest.mark.parametrize('layout', ['contiguous', 'transpose', 'offset'])
@torch.no_grad()
def test_matmul_dispatch_matches_original_body(dtype, name, layout):
    torch.manual_seed(261060)
    batch = (2,) if name == 'bmm' else ()
    a = torch.randn(*batch, 9, 13, dtype=dtype).to('ruda')
    b = torch.randn(*batch, 13, 7, dtype=dtype).to('ruda')
    if layout == 'transpose':
        a = a.transpose(-1, -2).contiguous().transpose(-1, -2)
        b = b.transpose(-1, -2).contiguous().transpose(-1, -2)
    elif layout == 'offset':
        a = torch.randn(*batch, 9, 16, dtype=dtype).to('ruda')[..., 3:]
        b = torch.randn(*batch, 13, 10, dtype=dtype).to('ruda')[..., 3:]
    versions = (a._version, b._version)
    with r.math_backend('ruda'):
        if name == 'addmm':
            for shape, beta, alpha in itertools.product([(), (7,), (1, 7), (9, 7)], [0., -.25, 1.], [0., .375, 1.]):
                bias = torch.randn(shape, dtype=dtype).to('ruda')
                if beta == 0: bias.fill_(float('nan'))
                same(torch.addmm(bias, a, b, beta=beta, alpha=alpha), _ops.addmm(bias, a, b, beta=beta, alpha=alpha))
        else:
            same(getattr(torch, name)(a, b), getattr(_ops, name)(a, b))
    assert (a._version, b._version) == versions


@pytest.mark.parametrize('name', ['mm', 'bmm', 'addmm'])
@pytest.mark.parametrize('m,k,n', [(0, 13, 7), (9, 0, 7), (9, 13, 0)])
@torch.no_grad()
def test_matmul_dispatch_empty_dimensions(name, m, k, n):
    batch = (2,) if name == 'bmm' else ()
    a, b = [torch.empty(shape, device='ruda') for shape in [(*batch, m, k), (*batch, k, n)]]
    args = (torch.zeros(n, device='ruda'), a, b) if name == 'addmm' else (a, b)
    with r.math_backend('ruda'):
        same(getattr(torch, name)(*args), getattr(_ops, name)(*args))


@pytest.mark.parametrize('name', ['mm', 'bmm', 'addmm'])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex64, torch.complex128])
@torch.no_grad()
def test_matmul_dispatch_preserves_wide_dtype_body(name, dtype):
    torch.manual_seed(261061)
    batch = (2,) if name == 'bmm' else ()
    a = torch.randn(*batch, 3, 5, dtype=dtype).to('ruda')
    b = torch.randn(*batch, 5, 4, dtype=dtype).to('ruda')
    args = (torch.randn(4, dtype=dtype).to('ruda'), a, b) if name == 'addmm' else (a, b)
    options = dict(alpha=.25 + .5j, beta=-.5j) if name == 'addmm' and dtype.is_complex else {}
    with r.math_backend('ruda'):
        same(getattr(torch, name)(*args, **options), getattr(_ops, name)(*args, **options))


@pytest.mark.parametrize('name', ['mm', 'bmm', 'addmm'])
def test_matmul_dispatch_preserves_autograd_and_second_derivative(name):
    torch.manual_seed(261062)
    batch = (2,) if name == 'bmm' else ()
    hosts = [torch.randn(*batch, 3, 5), torch.randn(*batch, 5, 4)]
    if name == 'addmm': hosts.insert(0, torch.randn(4))
    results = []
    with r.math_backend('ruda'):
        for device in ['cpu', 'ruda']:
            values = [x.to(device).detach().requires_grad_() for x in hosts]
            output = getattr(torch, name)(*values)
            grads = torch.autograd.grad(output.square().sum(), values, create_graph=True)
            second = torch.autograd.grad(sum(g.sum() for g in grads), values)
            results.append([output.detach().cpu(), *[g.detach().cpu() for g in grads], *[g.cpu() for g in second]])
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('name', ['mm', 'bmm', 'addmm'])
@torch.no_grad()
def test_matmul_dispatch_native_path_does_not_call_python_body(name, monkeypatch):
    batch = (2,) if name == 'bmm' else ()
    a = torch.ones(*batch, 3, 5, device='ruda')
    b = torch.ones(*batch, 5, 4, device='ruda')
    args = (torch.ones(4, device='ruda'), a, b) if name == 'addmm' else (a, b)
    def unexpected(): raise AssertionError('entered the Python matmul body')
    monkeypatch.setattr(_ops, '_nvidia', unexpected)
    with r.math_backend('ruda'):
        actual = getattr(torch, name)(*args)
    torch.testing.assert_close(actual.cpu(), torch.full((*batch, 3, 4), 6. if name == 'addmm' else 5.), rtol=0, atol=0)


@pytest.mark.parametrize('name', ['mm', 'bmm', 'addmm'])
def test_matmul_dispatch_keeps_contextvar_routing(name, monkeypatch):
    from ruda_torch import _vendor_blas
    batch = (2,) if name == 'bmm' else ()
    a = torch.ones(*batch, 3, 5, device='ruda')
    b = torch.ones(*batch, 5, 4, device='ruda')
    args = (torch.zeros(4, device='ruda'), a, b) if name == 'addmm' else (a, b)
    class VendorRoute(Exception): pass
    def vendor_route(*args, **kwargs): raise VendorRoute
    monkeypatch.setattr(_vendor_blas, 'gemm', vendor_route)
    def invoke(mode):
        if mode == 'nvidia':
            with pytest.raises(VendorRoute): getattr(torch, name)(*args)
        else:
            torch.testing.assert_close(getattr(torch, name)(*args).cpu(), torch.full((*batch, 3, 4), 5.), rtol=0, atol=0)
    with r.math_backend('ruda'):
        native_context = contextvars.copy_context()
        with r.math_backend('nvidia'):
            invoke('nvidia')
            native_context.run(invoke, 'ruda')
            invoke('nvidia')
        invoke('ruda')
    async def task(mode):
        with r.math_backend(mode):
            await asyncio.sleep(0)
            invoke(mode)
    async def both(): await asyncio.gather(task('ruda'), task('nvidia'))
    asyncio.run(both())


@pytest.mark.parametrize('name', ['mm', 'bmm', 'addmm'])
def test_matmul_dispatch_invalid_inputs_keep_original_errors(name):
    batch = (2,) if name == 'bmm' else ()
    a = torch.empty(*batch, 3, 5, device='ruda')
    for b in [torch.empty(*batch, 6, 4, device='ruda'), torch.empty(*batch, 5, 4, dtype=torch.float16, device='ruda')]:
        args = (torch.zeros(4, device='ruda'), a, b) if name == 'addmm' else (a, b)
        with r.math_backend('ruda'):
            with pytest.raises(Exception) as expected: getattr(_ops, name)(*args)
            with pytest.raises(type(expected.value)) as actual: getattr(torch, name)(*args)
        assert str(actual.value) == str(expected.value)


def test_matmul_dispatch_aot_and_nondefault_stream():
    from torch._dynamo.backends.common import aot_autograd
    graphs = []
    def capture(graph, inputs):
        graphs.append(graph)
        return graph.forward
    def function(a, b, bias): return torch.addmm(bias, a, b).square().sum()
    compiled = torch.compile(function, backend=aot_autograd(fw_compiler=capture, bw_compiler=capture),
                             fullgraph=True, dynamic=True)
    stream = r.Stream()
    with r.math_backend('ruda'), r.stream(stream):
        for rows in (3, 7):
            args = [torch.randn(rows, 5).to('ruda').requires_grad_(),
                    torch.randn(5, 4).to('ruda').requires_grad_(), torch.randn(4).to('ruda').requires_grad_()]
            expected = torch.autograd.grad(function(*args), args)
            actual = torch.autograd.grad(compiled(*args), args)
            for a, b in zip(actual, expected): same(a, b)
    stream.synchronize()
    assert len(graphs) >= 2
