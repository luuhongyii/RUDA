import math
import os

import pytest
import torch


NAMES = ['erf', 'erfc', 'erfinv', 'erfcx', 'ndtr', 'log_ndtr', 'ndtri']
DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]


@pytest.fixture(scope='module', autouse=True)
def backend():
    if os.environ.get('RUDA_REQUIRE_GPU') != '1':
        pytest.skip('requires native RUDA GPU backend')
    import ruda_torch
    assert ruda_torch._scientific_available


def function(name):
    return getattr(torch.special, name)


def close(actual, expected, *, precision=None, tails=False):
    precision = expected.dtype if precision is None else precision
    tolerance = {torch.float16: .003, torch.bfloat16: .02, torch.float32: 4e-6, torch.float64: 2e-12}[precision]
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=0 if tails else tolerance, equal_nan=True)


def inputs(dtype, name):
    low, high = (.125, .875) if name == 'ndtri' else (-.6, .6) if name == 'erfinv' else (-2., 2.)
    return torch.linspace(low, high, 30, dtype=dtype).reshape(5, 6).t()


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('dtype', DTYPES)
def test_error_variants(name, dtype):
    a = inputs(dtype, name)
    expected = function(name)(a.double()).to(dtype)
    value = a.to('ruda')
    close(function(name)(value), expected)
    target = torch.empty((5, 6), device='ruda', dtype=dtype).t()
    assert function(name)(value, out=target) is target
    close(target, expected)
    empty = torch.empty(0, device='ruda', dtype=dtype)
    close(function(name)(value, out=empty), expected)
    alias = value.clone()
    close(function(name)(alias, out=alias), expected)
    if name in ('erf', 'erfc', 'erfinv'):
        close(getattr(value.clone(), name+'_')(), expected)
        close(getattr(torch, name)(value), expected)
    close(function(name)(torch._neg_view(value)), function(name)(-a.double()).to(dtype))
    for shape in ((), (0, 3)):
        x = torch.full(shape, .5, dtype=dtype)
        close(function(name)(x.to('ruda')), function(name)(x.double()).to(dtype))


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('dtype', [torch.bool, torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8])
def test_error_integer_promotion(name, dtype):
    a = torch.tensor([0, 1, 2], dtype=dtype)
    close(function(name)(a.to('ruda')), function(name)(a))
    for output_dtype in (torch.float32, torch.float64):
        expected = torch.empty(3, dtype=output_dtype)
        function(name)(a, out=expected)
        out = torch.empty(3, dtype=output_dtype, device='ruda')
        close(function(name)(a.to('ruda'), out=out), expected, precision=torch.get_default_dtype())
    if name in ('erf', 'erfc', 'erfinv'):
        for device in ('cpu', 'ruda'):
            with pytest.raises(RuntimeError):
                getattr(a.to(device), name+'_')()


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_error_gradients_and_contracts(name, dtype):
    a = inputs(dtype, name)
    def gradients(x, inplace):
        x = x.detach().requires_grad_()
        y = getattr(x.clone(), name+'_')() if inplace else function(name)(x)
        first = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), x)[0]
        return first, second
    for inplace in ([False, True] if name in ('erf', 'erfc', 'erfinv') else [False]):
        for actual, expected in zip(gradients(a.to('ruda'), inplace), gradients(a, inplace)):
            close(actual, expected)
    for device in ('cpu', 'ruda'):
        with pytest.raises(RuntimeError):
            function(name)(a.to(device, torch.complex128))
        value = a.to(device).detach().requires_grad_()
        if name == 'ndtr':
            result = function(name)(value, out=torch.empty_like(value))
            close(result, function(name)(a))
            assert result.requires_grad
        else:
            with pytest.raises(RuntimeError, match='automatic differentiation|autograd'):
                function(name)(value, out=torch.empty_like(value))
        overlap = torch.arange(10., dtype=dtype, device=device)
        if name == 'ndtr':
            expected = function(name)(overlap[:-1].cpu().clone())
            function(name)(overlap[:-1], out=overlap[1:])
            close(overlap[1:], expected)
        else:
            with pytest.raises(RuntimeError):
                function(name)(overlap[:-1], out=overlap[1:])
        if name in ('erf', 'erfc', 'erfinv'):
            with pytest.raises(RuntimeError):
                getattr(overlap[:1].expand(4), name+'_')()


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('dtype', DTYPES)
def test_error_boundaries_and_tails(name, dtype):
    values = [0., -0., math.inf, -math.inf, math.nan, .5, -.5, 1., -1., 1.25, -1.25,
              .84375, -.84375, .7, -.7, 2.857142857142857, 6., -6., 10., -10.,
              26., -26., 27., -27., 28., -28., 40., -40., 1e-300, -1e-300, 1e300, -1e300]
    a = torch.tensor(values, dtype=torch.float64).to(dtype)
    expected = function(name)(a.double()).to(dtype)
    actual = function(name)(a.to('ruda')).cpu()
    if name == 'ndtr':
        close(actual, expected)
    else:
        close(actual, expected, tails=True)
    zero = (actual == 0) & (expected == 0)
    assert torch.equal(torch.signbit(actual[zero]), torch.signbit(expected[zero]))


def test_error_default_dtype():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        a = torch.tensor([0, 1, 2])
        for name in NAMES:
            close(function(name)(a.to('ruda')), function(name)(a))
    finally:
        torch.set_default_dtype(previous)
