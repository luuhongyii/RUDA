import os

import pytest
import torch


OPERATIONS = ['sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh']
DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.complex64, torch.complex128]


@pytest.fixture(scope='module', autouse=True)
def backend():
    if os.environ.get('RUDA_REQUIRE_GPU') != '1':
        pytest.skip('requires native RUDA GPU backend')
    import ruda_torch
    assert ruda_torch._native.ruda_torch_abi_version() == 10


def close(actual, expected):
    tolerance = 2e-2 if expected.dtype == torch.bfloat16 else 3e-3 if expected.dtype == torch.float16 else 3e-6
    if expected.dtype in (torch.float64, torch.complex128):
        tolerance = 1e-12
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance, equal_nan=True)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('operation', OPERATIONS)
def test_hyperbolic_values_and_layout(dtype, operation):
    source = torch.linspace(-3, 3, 30, dtype=torch.float64).reshape(5, 6)
    if dtype.is_complex:
        source = torch.complex(source, source.flip(1) / 7 + .125)
    source = source.to(dtype).t()
    function = getattr(torch, operation)
    close(function(source.to('ruda')), function(source))
    if not dtype.is_complex:
        special = torch.tensor([0., -0., 1., -1., float('inf'), -float('inf'), float('nan'),
                                1e-300, -1e-300, 20., -20., 709., 710., 711.], dtype=dtype)
        close(function(special.to('ruda')), function(special))
    for shape in ((), (0, 3)):
        value = torch.ones(shape, dtype=dtype)
        close(function(value.to('ruda')), function(value))


@pytest.mark.parametrize('dtype', DTYPES[2:])
@pytest.mark.parametrize('operation', OPERATIONS)
def test_hyperbolic_first_and_second_gradients(dtype, operation):
    value = torch.linspace(.125, .625, 12, dtype=torch.float64).reshape(3, 4)
    if operation == 'acosh':
        value += 1
    if dtype.is_complex:
        value = torch.complex(value, value.flip(1) * .2)
    value = value.to(dtype).t()
    function = getattr(torch, operation)
    def derivatives(a):
        a = a.detach().requires_grad_()
        result = function(a)
        loss = result.real.sum() + (result.imag.sum() * .3 if result.is_complex() else 0)
        first = torch.autograd.grad(loss, a, create_graph=True)[0]
        second = torch.autograd.grad((first.real * a.real.detach()).sum(), a)[0]
        return first, second
    expected = derivatives(value)
    actual = derivatives(value.to('ruda'))
    for result, reference in zip(actual, expected):
        close(result, reference)
