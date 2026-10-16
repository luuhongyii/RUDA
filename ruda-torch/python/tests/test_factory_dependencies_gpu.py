import math
import os
import io
import subprocess
import sys

import pytest
import torch


DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64,
          torch.complex64, torch.complex128, torch.int8, torch.uint8,
          torch.int16, torch.int32, torch.int64, torch.bool]


@pytest.fixture(scope='module', autouse=True)
def backend():
    if os.environ.get('RUDA_REQUIRE_GPU') != '1':
        pytest.skip('requires native RUDA GPU backend')
    import ruda_torch
    assert ruda_torch._scientific_available


def close(actual, expected):
    torch.testing.assert_close(actual.cpu(), expected.cpu(), equal_nan=True)


def zero_gradients(device, dtype):
    x = torch.tensor([.5, 2., 3.], device=device, dtype=dtype, requires_grad=True)
    z = torch._efficientzerotensor(x.shape, device=device, dtype=dtype)
    output = ((x + z) * (x + z)).real.sum()
    first = torch.autograd.grad(output, x, create_graph=True)[0]
    second = torch.autograd.grad(first.real.sum(), x)[0]
    return first, second


@pytest.fixture(scope='module')
def cuda_gradients():
    code = '''import io, runpy, sys, torch
module = runpy.run_path(sys.argv[1])
values = [tuple(t.detach().cpu() for t in module['zero_gradients']('cuda', dtype))
          for dtype in module['DTYPES'][:6]]
x = torch.linspace(0., 1., 7, device='cuda', requires_grad=True)
assert x.requires_grad and x.is_leaf
x.sum().backward()
data = io.BytesIO()
torch.save({'zero': values, 'linspace': x.grad.cpu()}, data)
sys.stdout.buffer.write(data.getvalue())
'''
    data = subprocess.check_output([sys.executable, '-B', '-c', code, __file__])
    return torch.load(io.BytesIO(data), weights_only=True)


@pytest.mark.parametrize('dtype', DTYPES[:-1])
@pytest.mark.parametrize('steps', [0, 1, 2, 7, 2051])
def test_linspace_dtypes(dtype, steps):
    pairs = [(2., 11.), (11., 2.), (.1, 7.9), (3., 3.)]
    if dtype.is_complex:
        pairs += [(2+3j, -4+7j), (1e-6-2j, 1+3e-6j)]
    for start, end in pairs:
        expected = torch.linspace(start, end, steps, dtype=dtype, device='cuda')
        close(torch.linspace(start, end, steps, dtype=dtype, device='ruda'), expected)
        out = torch.empty((steps, 2), dtype=dtype, device='ruda')[:, 1]
        assert torch.linspace(start, end, steps, out=out) is out
        close(out, expected)
        resized = torch.empty(0, dtype=dtype, device='ruda')
        close(torch.linspace(start, end, steps, out=resized), expected)


@pytest.mark.parametrize('dtype', DTYPES[:6])
def test_linspace_nonfinite_and_views(dtype):
    pairs = [(math.nan, 1.), (-math.inf, math.inf), (math.inf, 1.),
             (-0., 0.), (-1e-30, 1e-30)]
    if dtype.is_complex:
        pairs += [(complex(math.inf, 1.), 2+3j), (1+2j, complex(1., math.nan))]
    for start, end in pairs:
        for steps in (0, 1, 2, 9):
            expected = torch.linspace(start, end, steps, dtype=dtype, device='cuda')
            close(torch.linspace(start, end, steps, dtype=dtype, device='ruda'), expected)
    for device in ('cuda', 'ruda'):
        base = torch.empty((4, 3), dtype=dtype, device=device)
        out = torch._neg_view(base.t())
        torch.linspace(2., 5., 12, out=out)
        assert out.shape == (3, 4)
        close(out, torch.linspace(2., 5., 12, dtype=dtype, device='cuda').reshape(3, 4))
        if dtype.is_complex:
            out = base.t().conj()
            torch.linspace(1+2j, -3+4j, 12, out=out)
            close(out, torch.linspace(1+2j, -3+4j, 12, dtype=dtype, device='cuda').reshape(3, 4))


@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
def test_linspace_large_integer_endpoints(dtype):
    bounds = [(2**24-3, 2**24+37), (-(2**24+37), -(2**24-3))]
    if dtype == torch.int64:
        bounds += [(2**53+1, 2**53+4097), (-2**53-4097, -2**53-1)]
    for start, end in bounds:
        for steps in (1, 3, 17):
            close(torch.linspace(start, end, steps, device='ruda', dtype=dtype),
                  torch.linspace(start, end, steps, device='cuda', dtype=dtype))


def test_linspace_factory_contracts(cuda_gradients):
    for device in ('cuda', 'ruda'):
        for steps in (0, 1):
            close(torch.linspace(False, True, steps, device=device, dtype=torch.bool),
                  torch.linspace(False, True, steps, device='cuda', dtype=torch.bool))
        with pytest.raises(RuntimeError):
            torch.linspace(0, 1, 2, device=device, dtype=torch.bool)
        with pytest.raises(RuntimeError):
            torch.linspace(0, 1, -1, device=device)
        with pytest.raises(RuntimeError):
            torch.linspace(0, 1, 4, out=torch.empty(1, device=device).expand(4))
        for left, right in ((torch.tensor(.5), 3.), (.5, torch.tensor(3.)),
                            (torch.tensor(.5, device=device), torch.tensor(3., device=device))):
            close(torch.linspace(left, right, 9, device=device), torch.linspace(.5, 3., 9, device='cuda'))
        out = torch.linspace(0., 1., 7, device=device, requires_grad=True)
        assert out.requires_grad and out.is_leaf
        if device == 'ruda':
            out.sum().backward()
            close(out.grad, cuda_gradients['linspace'])


@pytest.mark.parametrize('dtype', DTYPES)
def test_efficient_zero(dtype):
    for shape in ((2, 3), (), (0, 3)):
        value = torch._efficientzerotensor(shape, device='ruda', dtype=dtype)
        reference = torch._efficientzerotensor(shape, device='cuda', dtype=dtype)
        assert value.device == torch.device('ruda:0') and value.dtype == dtype
        assert torch._is_zerotensor(value) and value.data_ptr() == 0
        assert torch._is_zerotensor(value.reshape(-1))
        materialized = value.clone()
        assert not torch._is_zerotensor(materialized)
        close(materialized, reference.clone())
        close(value + torch.ones(shape, dtype=dtype, device='ruda'),
              reference + torch.ones(shape, dtype=dtype, device='cuda'))
        close(value * torch.ones(shape, dtype=dtype, device='ruda'),
              reference * torch.ones(shape, dtype=dtype, device='cuda'))
        close(value + value, reference + reference)
        close(value.to('cpu'), reference.to('cpu'))
        destination = torch.ones(shape, dtype=dtype, device='ruda')
        close(destination.copy_(value), reference.clone())
        if dtype.is_complex:
            close(value.conj().clone(), reference.conj().clone())
        if dtype != torch.bool:
            close(torch._neg_view(value).clone(), torch._neg_view(reference).clone())
    for device in ('cuda', 'ruda'):
        value = torch._efficientzerotensor((2, 3), device=device, dtype=dtype)
        with pytest.raises(RuntimeError, match='ZeroTensor'):
            value.add_(1)
        with pytest.raises(RuntimeError, match='ZeroTensor'):
            value.copy_(torch.ones_like(value))


@pytest.mark.parametrize('dtype', DTYPES[:6])
def test_efficient_zero_autograd(dtype, cuda_gradients):
    for actual, expected in zip(zero_gradients('ruda', dtype), cuda_gradients['zero'][DTYPES.index(dtype)]):
        close(actual, expected)
