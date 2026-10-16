import math
import os

import pytest
import torch


UNARY = ['sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh']
DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.complex64, torch.complex128]


@pytest.fixture(scope='module', autouse=True)
def backend():
    if os.environ.get('RUDA_REQUIRE_GPU') != '1':
        pytest.skip('requires native RUDA GPU backend')
    import ruda_torch
    assert ruda_torch._scientific_available


def close(actual, expected, *, zeros=False, precision=None):
    actual = actual.cpu()
    precision = expected.dtype if precision is None else precision
    tolerance = 2e-2 if precision == torch.bfloat16 else 3e-3 if precision == torch.float16 else 4e-6
    if precision in (torch.float64, torch.complex128):
        tolerance = 2e-12
    if expected.is_complex():
        actual, expected = torch.view_as_real(actual), torch.view_as_real(expected)
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True)
    if zeros:
        mask = (actual == 0) & (expected == 0)
        assert torch.equal(torch.signbit(actual[mask]), torch.signbit(expected[mask]))


def source(dtype, name):
    a = torch.linspace(.125, .875, 30, dtype=torch.float64).reshape(5, 6).t()
    if name == 'acosh':
        a += 1
    if dtype.is_complex:
        a = torch.complex(a, a.flip(0) / 3)
    return a.to(dtype)


@pytest.mark.parametrize('name', UNARY)
@pytest.mark.parametrize('dtype', DTYPES)
def test_unary_variants(name, dtype):
    a = source(dtype, name)
    function = getattr(torch, name)
    expected = function(a)
    device = a.to('ruda')
    close(function(device), expected)
    out = torch.empty((5, 6), dtype=dtype, device='ruda').t()
    assert function(device, out=out) is out
    close(out, expected)
    alias = device.clone()
    assert function(alias, out=alias) is alias
    close(alias, expected)
    value = device.clone()
    assert getattr(value, name + '_')() is value
    close(value, expected)
    resized = torch.empty(0, dtype=dtype, device='ruda')
    close(function(device, out=resized), expected)
    for shape in ((), (0, 3)):
        value = torch.ones(shape, dtype=dtype)
        close(function(value.to('ruda')), function(value))
    if dtype.is_complex:
        close(function(device.conj()), function(a.conj()))
        target = torch.empty_like(device).conj()
        close(function(device.conj(), out=target), function(a.conj()))
    close(function(torch._neg_view(device)), function(-a))


@pytest.mark.parametrize('name', UNARY)
@pytest.mark.parametrize('dtype', [torch.bool, torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8])
def test_unary_integer_promotion(name, dtype):
    a = (torch.arange(8) % 3).to(dtype)
    function = getattr(torch, name)
    close(function(a.to('ruda')), function(a))
    for output_dtype in (torch.float32, torch.float64, torch.complex128):
        expected = torch.empty(a.shape, dtype=output_dtype)
        function(a, out=expected)
        out = torch.empty(a.shape, dtype=output_dtype, device='ruda')
        close(function(a.to('ruda'), out=out), expected, precision=torch.get_default_dtype())
    for value in (a.clone(), a.to('ruda')):
        with pytest.raises(RuntimeError):
            getattr(value, name + '_')()


@pytest.mark.parametrize('name', UNARY)
@pytest.mark.parametrize('dtype', DTYPES[2:])
def test_unary_autograd(name, dtype):
    def derivatives(value, inplace):
        value = value.detach().requires_grad_()
        y = getattr(value.clone(), name + '_')() if inplace else getattr(torch, name)(value)
        loss = y.real.sum() + (y.imag.sum() * .3 if y.is_complex() else 0)
        first = torch.autograd.grad(loss, value, create_graph=True)[0]
        second = torch.autograd.grad((first.real * value.real.detach()).sum(), value)[0]
        return first, second
    a = source(dtype, name)
    for inplace in (False, True):
        if inplace and name in ('asinh', 'acosh', 'atanh'):
            for device in ('cpu', 'ruda'):
                with pytest.raises(NotImplementedError, match='derivative.*inplace version'):
                    derivatives(a.to(device), True)
            continue
        for actual, expected in zip(derivatives(a.to('ruda'), inplace), derivatives(a, inplace)):
            close(actual, expected)
    for device in ('cpu', 'ruda'):
        value = a.to(device).detach().requires_grad_()
        with pytest.raises(RuntimeError, match='automatic differentiation|autograd'):
            getattr(torch, name)(value, out=torch.empty_like(value))


@pytest.mark.parametrize('name', UNARY)
def test_unary_real_boundaries_and_aliases(name):
    a = torch.tensor([0., -0., 1., -1., math.inf, -math.inf, math.nan, 1e-300, -1e-300,
                      .5, -.5, 1e300, -1e300, math.pi / 2, -math.pi / 2], dtype=torch.float64)
    close(getattr(torch, name)(a.to('ruda')), getattr(torch, name)(a), zeros=True)
    aliases = {'asin': 'arcsin', 'acos': 'arccos', 'atan': 'arctan', 'asinh': 'arcsinh',
               'acosh': 'arccosh', 'atanh': 'arctanh'}
    if name in aliases:
        close(getattr(torch, aliases[name])(a.to('ruda')), getattr(torch, name)(a))
    for device in ('cpu', 'ruda'):
        value = torch.arange(9., device=device)
        with pytest.raises(RuntimeError):
            getattr(torch, name)(value[:-1], out=value[1:])
        with pytest.raises(RuntimeError):
            getattr(value[:1].expand(4), name + '_')()


@pytest.mark.parametrize('name', ['tan', 'asin', 'acos', 'atan', 'tanh', 'asinh', 'acosh', 'atanh'])
@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_complex_branches_and_extremes(name, dtype):
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    large = 1e300 if real_dtype == torch.float64 else 1e30
    tiny = 1e-300 if real_dtype == torch.float64 else 1e-30
    points = [0., -0., tiny, -tiny, .5, -.5, 1., -1., 2., -2., 20., -20., large, -large,
              math.inf, -math.inf, math.nan]
    a = torch.tensor([complex(x, y) for x in points for y in points], dtype=dtype)
    close(getattr(torch, name)(a.to('ruda')), getattr(torch, name)(a.to('cuda')).cpu(), zeros=True)
    parts = torch.view_as_real(a)
    finite = a[torch.isfinite(parts).all(-1)]
    result = getattr(torch, name)(finite.to('ruda')).cpu()
    expected = getattr(torch, name)(finite)
    actual_parts, expected_parts = torch.view_as_real(result), torch.view_as_real(expected)
    nonzero_small = (expected_parts != 0) & (expected_parts.abs() < 1e-10)
    torch.testing.assert_close(actual_parts[nonzero_small], expected_parts[nonzero_small],
                               atol=0, rtol=3e-5 if dtype == torch.complex64 else 3e-12)


@pytest.mark.parametrize('name', ['atan2', 'hypot'])
@pytest.mark.parametrize('dtype', DTYPES[:4])
def test_binary_variants_and_gradients(name, dtype):
    a = torch.linspace(.125, 2, 6, dtype=dtype).reshape(3, 2).t()
    b = torch.tensor([[.25], [1.]], dtype=dtype)
    function = getattr(torch, name)
    expected = function(a, b)
    left, right = a.to('ruda'), b.to('ruda')
    close(function(left, right), expected)
    out = torch.empty((3, 2), dtype=dtype, device='ruda').t()
    close(function(left, right, out=out), expected)
    close(getattr(left.clone(), name + '_')(right), expected)
    close(function(left, torch.tensor(.5, dtype=dtype)), function(a, torch.tensor(.5, dtype=dtype)))
    close(function(torch.tensor(.5, dtype=dtype), right), function(torch.tensor(.5, dtype=dtype), b))
    for shape in ((), (0, 3)):
        value = torch.ones(shape, dtype=dtype)
        close(function(value.to('ruda'), value.to('ruda')), function(value, value))
    if dtype in (torch.float32, torch.float64):
        def gradients(x, y):
            x = x.detach().requires_grad_(); y = y.detach().requires_grad_()
            first = torch.autograd.grad(function(x, y).sum(), (x, y), create_graph=True)
            second = torch.autograd.grad(sum(v.square().sum() for v in first), (x, y))
            return first + second
        for actual, reference in zip(gradients(left, right), gradients(a, b)):
            close(actual, reference)


@pytest.mark.parametrize('name', ['atan2', 'hypot'])
def test_binary_promotion_specials_and_errors(name):
    function = getattr(torch, name)
    for dtype in (torch.bool, torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float64):
        a = torch.tensor([0, 1, 2], dtype=dtype)
        b = torch.tensor(.5, dtype=torch.float32)
        close(function(a.to('ruda'), b.to('ruda')), function(a, b))
    values = [0., -0., .5, -.5, math.inf, -math.inf, math.nan, 1e300, 1e-300]
    a = torch.tensor(values, dtype=torch.float64).reshape(-1, 1)
    b = torch.tensor(values, dtype=torch.float64).reshape(1, -1)
    close(function(a.to('ruda'), b.to('ruda')), function(a, b), zeros=True)
    if name == 'atan2':
        a = torch.tensor([-1, 0, 1]); b = torch.tensor([0, 1, -1])
        close(function(a.to('ruda'), b.to('ruda')), function(a, b))
    for device in ('cpu', 'ruda'):
        a = torch.ones((2, 3), device=device, requires_grad=True)
        with pytest.raises(RuntimeError, match='automatic differentiation|autograd'):
            function(a, a, out=torch.empty_like(a))
        with pytest.raises(RuntimeError):
            function(a.detach().to(torch.complex64), a.detach().to(torch.complex64))
        if name == 'hypot':
            with pytest.raises((RuntimeError, NotImplementedError)):
                function(a.detach().long(), a.detach().long())


def test_default_float_dtype():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        a = torch.tensor([0, 1, 2])
        for name in UNARY:
            close(getattr(torch, name)(a.to('ruda')), getattr(torch, name)(a))
        close(torch.atan2(a.to('ruda'), a.to('ruda')), torch.atan2(a, a))
    finally:
        torch.set_default_dtype(previous)
