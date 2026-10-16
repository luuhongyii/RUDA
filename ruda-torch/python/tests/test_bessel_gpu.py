import math
import os

import pytest
import torch


DIFFERENTIABLE = ['i0', 'i0e', 'i1', 'i1e']
OTHER = ['bessel_j0', 'bessel_j1', 'bessel_y0', 'bessel_y1',
         'modified_bessel_i0', 'modified_bessel_i1', 'modified_bessel_k0', 'modified_bessel_k1',
         'scaled_modified_bessel_k0', 'scaled_modified_bessel_k1', 'spherical_bessel_j0']
NAMES = DIFFERENTIABLE + OTHER
DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
CASES = [(name, dtype) for name in NAMES for dtype in (DTYPES if name in DIFFERENTIABLE else DTYPES[2:])]


@pytest.fixture(scope='module', autouse=True)
def backend():
    if os.environ.get('RUDA_REQUIRE_GPU') != '1':
        pytest.skip('requires native RUDA GPU backend')
    import ruda_torch
    assert ruda_torch._scientific_available


def close(actual, expected, precision=None):
    dtype = expected.dtype if precision is None else precision
    tolerance = {torch.float16: .005, torch.bfloat16: .03, torch.float32: 1e-5, torch.float64: 3e-12}[dtype]
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=tolerance, atol=tolerance, equal_nan=True)


@pytest.mark.parametrize('name,dtype', CASES)
def test_bessel_variants(name, dtype):
    function = getattr(torch.special, name)
    a = torch.linspace(.125, 24., 60, dtype=dtype).reshape(10, 6).t()[::2]
    value = a.to('ruda')
    expected = function(a.cuda())
    close(function(value), expected)
    for out in (torch.empty((10, 3), device='ruda', dtype=dtype).t(), torch.empty(0, device='ruda', dtype=dtype)):
        assert function(value, out=out) is out
        close(out, expected)
    alias = value.clone()
    close(function(alias, out=alias), expected)
    close(function(torch._neg_view(value)), function((-a).cuda()))
    for shape in ((), (0, 3)):
        a = torch.full(shape, .5, dtype=dtype)
        close(function(a.to('ruda')), function(a.cuda()))
    if name == 'i0':
        close(torch.i0(value), expected)
        close(value.clone().i0_(), expected)


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('dtype', [torch.bool, torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8])
def test_bessel_integer_promotion(name, dtype):
    function = getattr(torch.special, name)
    a = torch.tensor([0, 1, 2, 3], dtype=dtype)
    close(function(a.to('ruda')), function(a.cuda()))
    for target in (torch.float32, torch.float64):
        expected = torch.empty(4, dtype=target, device='cuda')
        function(a.cuda(), out=expected)
        actual = torch.empty(4, dtype=target, device='ruda')
        close(function(a.to('ruda'), out=actual), expected, torch.get_default_dtype())
    if name == 'i0':
        for device in ('cuda', 'ruda'):
            with pytest.raises(RuntimeError):
                a.to(device).i0_()


@pytest.mark.parametrize('name,dtype', CASES)
def test_bessel_boundaries(name, dtype):
    function = getattr(torch.special, name)
    values = [0., -0., math.nan, math.inf, -math.inf, 1e-300, -1e-300,
              1e-6, -1e-6, .125, -.125, .4999, .5, .5001, 1.9999, 2., 2.0001,
              4.9999, 5., 5.0001, 7.9999, 8., 8.0001, -8., -8.001,
              50., 80., 88., 89., 700., 709., 710., 1000., 1e30, 1e300, -1e300]
    a = torch.tensor(values, dtype=torch.float64).to(dtype)
    expected = function(a.cuda()).cpu()
    actual = function(a.to('ruda')).cpu()
    close(actual, expected)
    zeros = (actual == 0) & (expected == 0)
    assert torch.equal(torch.signbit(actual[zeros]), torch.signbit(expected[zeros]))


@pytest.mark.parametrize('name', DIFFERENTIABLE)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_bessel_gradients(name, dtype):
    function = getattr(torch.special, name)
    source = torch.tensor([[-8.1, -.75, -.01], [.01, .75, 8.1]], dtype=dtype).t()
    def gradients(device, mutate):
        x = source.to(device).detach().requires_grad_()
        value = x.clone().i0_() if mutate else function(x)
        first = torch.autograd.grad(value.sum(), x, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), x)[0]
        return value, first, second
    for mutate in ([False, True] if name == 'i0' else [False]):
        for actual, expected in zip(gradients('ruda', mutate), gradients('cpu', mutate)):
            close(actual, expected)
    for device in ('cpu', 'ruda'):
        zero = torch.tensor([0., -0.], device=device, dtype=dtype, requires_grad=True)
        gradient = torch.autograd.grad(function(zero).sum(), zero)[0]
        close(gradient, torch.full((2,), .5 if name in ('i1', 'i1e') else 0., dtype=dtype))


@pytest.mark.parametrize('name', NAMES)
def test_bessel_contracts(name):
    function = getattr(torch.special, name)
    for device in ('cuda', 'ruda'):
        a = torch.arange(1., 6., device=device)
        with pytest.raises(RuntimeError):
            function(a.to(torch.complex64))
        with pytest.raises(RuntimeError):
            function(a[:-1], out=a[1:])
        with pytest.raises(RuntimeError):
            function(a, out=a[:1].expand(5))
        if name in DIFFERENTIABLE:
            with pytest.raises(RuntimeError):
                function(a.detach().requires_grad_(), out=torch.empty_like(a))
        if name in OTHER:
            expected = function(a).detach()
            for input_grad, output_grad in ((True, False), (False, True), (True, True)):
                output = torch.empty_like(a, requires_grad=output_grad)
                assert function(a.detach().requires_grad_(input_grad), out=output) is output
                assert output.requires_grad == output_grad and output.is_leaf
                close(output, expected)
            for dtype in (torch.float16, torch.bfloat16):
                with pytest.raises(RuntimeError):
                    function(a.to(dtype))


@pytest.mark.parametrize('name', OTHER)
def test_bessel_autograd_contract(name):
    function = getattr(torch.special, name)
    expected = function(torch.tensor([.5, 2.], requires_grad=True))
    actual = function(torch.tensor([.5, 2.], device='ruda', requires_grad=True))
    assert actual.requires_grad == expected.requires_grad
    if expected.requires_grad:
        for value in (expected, actual):
            with pytest.raises(RuntimeError):
                value.sum().backward()


def test_bessel_default_dtype():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        a = torch.tensor([1, 2, 5])
        for name in NAMES:
            function = getattr(torch.special, name)
            close(function(a.to('ruda')), function(a.cuda()))
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('scaled', [False, True])
def test_bessel_circular_density_gradients(dtype, scaled):
    def evaluate(device):
        k = torch.tensor([[.3], [2.], [8.]], device=device, dtype=dtype, requires_grad=True)
        theta = torch.linspace(-math.pi, math.pi, 17, device=device, dtype=dtype)
        normalizer = torch.special.i0e(k).log() + k if scaled else torch.i0(k).log()
        density = (k * theta.cos() - math.log(2 * math.pi) - normalizer).exp()
        first = torch.autograd.grad(density.sum(), k, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), k)[0]
        return density, first, second
    for actual, expected in zip(evaluate('ruda'), evaluate('cpu')):
        close(actual, expected)
