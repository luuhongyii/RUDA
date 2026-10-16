import math
import os

import pytest
import torch


CASES = [('lgamma', 0), ('digamma', 0)] + [('polygamma', n) for n in (0, 1, 2, 3, 5, 10, 20)] + [('mvlgamma', p) for p in (1, 2, 4)]
DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]


@pytest.fixture(scope='module', autouse=True)
def backend():
    if os.environ.get('RUDA_REQUIRE_GPU') != '1':
        pytest.skip('requires native RUDA GPU backend')
    import ruda_torch
    assert ruda_torch._scientific_available


def call(case, value, *, out=None, alias=False):
    name, parameter = case
    function = getattr(torch.special, {'lgamma': 'gammaln', 'digamma': 'digamma', 'polygamma': 'polygamma', 'mvlgamma': 'multigammaln'}[name]) if alias else getattr(torch, name)
    args = (parameter, value) if name == 'polygamma' else (value, parameter) if name == 'mvlgamma' else (value,)
    return function(*args, **({} if out is None else {'out': out}))


def inplace(case, value):
    name, parameter = case
    return getattr(value, name+'_')(*((parameter,) if name in ('polygamma', 'mvlgamma') else ()))


def close(actual, expected, precision=None):
    dtype = expected.dtype if precision is None else precision
    tolerance = {torch.float16: .005, torch.bfloat16: .03, torch.float32: 8e-6, torch.float64: 3e-12}[dtype]
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=tolerance, atol=tolerance, equal_nan=True)


def values(dtype):
    return torch.linspace(2.25, 5.75, 30, dtype=dtype).reshape(5, 6).t()


def reference(case, value):
    return call(case, value.cuda()).cpu() if value.dtype in (torch.float16, torch.bfloat16) else call(case, value)


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('dtype', DTYPES)
def test_gamma_variants(case, dtype):
    a = values(dtype)
    expected = reference(case, a)
    value = a.to('ruda')
    close(call(case, value), expected)
    close(call(case, value, alias=True), expected)
    close(inplace(case, value.clone()), expected)
    for out in (torch.empty((5, 6), dtype=dtype, device='ruda').t(), torch.empty(0, dtype=dtype, device='ruda')):
        assert call(case, value, out=out) is out
        close(out, expected)
    alias = value.clone()
    close(call(case, alias, out=alias), expected)
    negative = -a
    close(call(case, torch._neg_view(value)), reference(case, negative))
    for shape in ((), (0, 3)):
        x = torch.full(shape, 3., dtype=dtype)
        close(call(case, x.to('ruda')), reference(case, x))


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('dtype', [torch.bool, torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8])
def test_gamma_integer_promotion(case, dtype):
    a = torch.tensor([2, 3, 5], dtype=dtype)
    if dtype == torch.bool and case[0] == 'mvlgamma':
        for device in ('cpu', 'ruda'):
            with pytest.raises(RuntimeError):
                call(case, a.to(device))
        return
    close(call(case, a.to('ruda')), call(case, a))
    for target in (torch.float32, torch.float64):
        expected = torch.empty(3, dtype=target)
        call(case, a, out=expected)
        out = torch.empty(3, dtype=target, device='ruda')
        close(call(case, a.to('ruda'), out=out), expected, torch.get_default_dtype())
    for device in ('cpu', 'ruda'):
        with pytest.raises(RuntimeError):
            inplace(case, a.to(device))


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_gamma_gradients_and_contracts(case, dtype):
    a = values(dtype)
    def gradients(x, mutate):
        x = x.detach().requires_grad_()
        y = inplace(case, x.clone()) if mutate else call(case, x)
        first = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), x)[0]
        return first, second
    for mutate in (False, True):
        for actual, expected in zip(gradients(a.to('ruda'), mutate), gradients(a, mutate)):
            close(actual, expected)
    for device in ('cpu', 'ruda'):
        with pytest.raises(RuntimeError):
            call(case, a.to(device, torch.complex128))
        value = a.to(device).detach().requires_grad_()
        with pytest.raises(RuntimeError):
            call(case, value, out=torch.empty_like(value))
        expanded = a.flatten()[:1].to(device).expand(4)
        with pytest.raises(RuntimeError):
            inplace(case, expanded)
        storage = torch.arange(2., 12., device=device, dtype=dtype)
        if case[0] != 'mvlgamma':
            with pytest.raises(RuntimeError):
                call(case, storage[:-1], out=storage[1:])
        else:
            expected = call(case, storage[:-1].cpu().clone())
            call(case, storage[:-1], out=storage[1:])
            close(storage[1:], expected)


@pytest.mark.parametrize('case', CASES[:9])
@pytest.mark.parametrize('dtype', DTYPES)
def test_gamma_boundaries(case, dtype):
    a = torch.tensor([0., -0., math.inf, -math.inf, math.nan, 1., 2., .125, .5, .9,
                      1.2316322326660156, 1.4616321449683622, 1.7316312789916992, 8., 10.,
                      -1., -2., -3.5, -.5, -10.25, 1e-300, -1e-300, 1e300, -1e300], dtype=torch.float64).to(dtype)
    expected = call(case, a if dtype == torch.float64 else a.cuda()).cpu()
    actual = call(case, a.to('ruda')).cpu()
    close(actual, expected)
    zero = (actual == 0) & (expected == 0)
    assert torch.equal(torch.signbit(actual[zero]), torch.signbit(expected[zero]))


def test_gamma_invalid_order_and_default_dtype():
    for device in ('cpu', 'ruda'):
        a = torch.ones(3, device=device)
        for case in [('polygamma', -1), ('mvlgamma', 0)]:
            with pytest.raises(RuntimeError):
                call(case, a)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        for case in CASES:
            a = torch.tensor([2, 3, 5])
            close(call(case, a.to('ruda')), call(case, a))
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('distribution', ['beta', 'dirichlet'])
def test_gamma_distribution_gradients(dtype, distribution):
    def evaluate(device):
        concentration = torch.tensor([[1.5, 2.3, 3.7], [2.1, 4.2, 1.1]], device=device, dtype=dtype, requires_grad=True)
        if distribution == 'beta':
            law = torch.distributions.Beta(concentration[:, 0], concentration[:, 1])
            sample = torch.tensor([.3, .7], device=device, dtype=dtype)
        else:
            law = torch.distributions.Dirichlet(concentration)
            sample = torch.tensor([[.2, .3, .5], [.1, .6, .3]], device=device, dtype=dtype)
        result = law.log_prob(sample) + law.entropy()
        first = torch.autograd.grad(result.sum(), concentration, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), concentration)[0]
        return result, first, second
    for actual, expected in zip(evaluate('ruda'), evaluate('cpu')):
        close(actual, expected)


@pytest.mark.parametrize('name', ['xlogy', 'xlog1py'])
@pytest.mark.parametrize('dtype', DTYPES)
def test_logarithmic_products(name, dtype):
    function = getattr(torch.special, name)
    a = torch.linspace(-2, 2, 12, dtype=dtype).reshape(3, 4).t()
    b = torch.tensor([.25, .5, 2.], dtype=dtype)
    left, right = a.to('ruda'), b.to('ruda')
    close(function(left, right), function(a, b))
    target = torch.empty((3, 4), dtype=dtype, device='ruda').t()
    close(function(left, right, out=target), function(a, b))
    close(function(left, right, out=torch.empty(0, device='ruda', dtype=dtype)), function(a, b))
    close(function(left, .75), function(a, .75))
    close(function(.75, right), function(.75, b))
    close(function(left, .75, out=target), function(a, .75))
    close(function(.75, right, out=torch.empty_like(right)), function(.75, b))
    close(function(left, torch.tensor(.75)), function(a, torch.tensor(.75)))
    if name == 'xlogy':
        close(left.clone().xlogy_(right), function(a, b))
        close(left.clone().xlogy_(.75), function(a, .75))
    special = torch.tensor([0., -0., -1., 1., math.nan, math.inf, -math.inf], dtype=dtype)
    expected = function(special[:, None], special[None, :])
    actual = function(special[:, None].to('ruda'), special[None, :].to('ruda'))
    close(actual, expected)
    for shape in ((), (0, 3)):
        x = torch.full(shape, .25, dtype=dtype)
        close(function(x.to('ruda'), .75), function(x, .75))
    for first, second in ((torch.int64, torch.float32), (torch.bool, torch.int32), (torch.float32, torch.float64)):
        x, y = torch.tensor([0, 1], dtype=first), torch.tensor([1, 2], dtype=second)
        close(function(x.to('ruda'), y.to('ruda')), function(x, y))


@pytest.mark.parametrize('name', ['xlogy', 'xlog1py'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_logarithmic_product_gradients(name, dtype):
    function = getattr(torch.special, name)
    def gradients(device, mutate):
        x = torch.tensor([[.2], [.8]], device=device, dtype=dtype, requires_grad=True)
        y = torch.tensor([[.3, 1.2, 2.1]], device=device, dtype=dtype, requires_grad=True)
        result = x.expand(2, 3).clone().xlogy_(y) if mutate else function(x, y)
        first = torch.autograd.grad(result.sum(), (x, y), create_graph=True)
        second = torch.autograd.grad(sum(g.sum() for g in first), (x, y))
        return (result,) + first + second
    for mutate in ([False, True] if name == 'xlogy' else [False]):
        for actual, expected in zip(gradients('ruda', mutate), gradients('cpu', mutate)):
            close(actual, expected)


@pytest.mark.parametrize('name', ['xlogy', 'xlog1py'])
def test_logarithmic_product_contracts(name):
    function = getattr(torch.special, name)
    for device in ('cpu', 'ruda'):
        x = torch.arange(1., 6., device=device)
        with pytest.raises(RuntimeError):
            function(x[:-1], 2., out=x[1:])
        with pytest.raises(RuntimeError):
            function(x.to(torch.complex64), 2.)
        with pytest.raises(RuntimeError):
            function(x.detach().requires_grad_(), 2., out=torch.empty_like(x))
        if name == 'xlogy':
            with pytest.raises(RuntimeError):
                x.to(torch.int64).xlogy_(2.)
            with pytest.raises(RuntimeError):
                x[:1].expand(3).xlogy_(2.)
    for device in ('cuda', 'ruda'):
        for a, b in ((torch.tensor(.5), torch.tensor(2.)), (torch.tensor(.5), 2.),
                     (.5, torch.tensor(2.)), (torch.tensor([.5]), torch.tensor([2.]))):
            with pytest.raises(RuntimeError):
                function(a, b, out=torch.empty((), device=device))
