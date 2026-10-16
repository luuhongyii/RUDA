import math
import os

import pytest
import torch


FLOATS = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
COMPLEX = [torch.complex64, torch.complex128]
UNARY = ['exp2', 'expm1', 'log2', 'log10', 'entr']
CASES = [(name, dtype) for name in UNARY for dtype in FLOATS + (COMPLEX if name != 'entr' else [])]
BINARY = [('logaddexp', dtype) for dtype in FLOATS + COMPLEX] + [('logaddexp2', dtype) for dtype in FLOATS]


@pytest.fixture(scope='module', autouse=True)
def backend():
    if os.environ.get('RUDA_REQUIRE_GPU') != '1':
        pytest.skip('requires native RUDA GPU backend')
    import ruda_torch
    assert ruda_torch._scientific_available


def operation(name):
    return torch.special.entr if name == 'entr' else getattr(torch, name)


def close(actual, expected, *, dtype=None, tiny=False):
    dtype = expected.dtype if dtype is None else dtype
    tolerance = {torch.float16: .003, torch.bfloat16: .02, torch.float32: 3e-6,
                 torch.float64: 3e-12, torch.complex64: 5e-6, torch.complex128: 5e-12}[dtype]
    actual, expected = actual.cpu(), expected.cpu()
    if actual.is_complex():
        actual = torch.view_as_real(actual.resolve_conj().resolve_neg())
        expected = torch.view_as_real(expected.resolve_conj().resolve_neg())
    torch.testing.assert_close(actual, expected, rtol=tolerance,
                               atol=0 if tiny else tolerance, equal_nan=True)


@pytest.mark.parametrize('name,dtype', CASES)
def test_elementary_unary_variants(name, dtype):
    function = operation(name)
    a = torch.linspace(.125, 3., 42).reshape(7, 6).t()[::2].to(dtype)
    if dtype.is_complex:
        a = a + 1j * torch.linspace(-2., 2., 21).reshape(a.shape)
    value = a.to('ruda')
    expected = function(a.cuda())
    close(function(value), expected)
    for out in (torch.empty((7, 3), device='ruda', dtype=dtype).t(), torch.empty(0, device='ruda', dtype=dtype)):
        assert function(value, out=out) is out
        close(out, expected)
    alias = value.clone()
    close(function(alias, out=alias), expected)
    close(function(torch._neg_view(value)), function(torch._neg_view(a.cuda())))
    if dtype.is_complex:
        close(function(value.conj()), function(a.cuda().conj()))
        out = torch.empty_like(value).conj()
        close(function(value, out=out), expected)
    if name != 'entr':
        close(getattr(value.clone(), name + '_')(), expected)
    for shape in ((), (0, 3)):
        a = torch.full(shape, .5, dtype=dtype)
        close(function(a.to('ruda')), function(a.cuda()))


@pytest.mark.parametrize('name', UNARY + ['logit'])
@pytest.mark.parametrize('dtype', [torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64])
def test_elementary_integer_promotion(name, dtype):
    function = operation(name)
    a = torch.tensor([0, 1, 2, 3], dtype=dtype)
    expected = function(a.cuda())
    close(function(a.to('ruda')), expected)
    for target in FLOATS[2:]:
        out = torch.empty(4, device='ruda', dtype=target)
        reference = torch.empty(4, device='cuda', dtype=target)
        close(function(a.to('ruda'), out=out), function(a.cuda(), out=reference), dtype=torch.get_default_dtype())
    if name != 'entr':
        for device in ('cuda', 'ruda'):
            with pytest.raises(RuntimeError):
                getattr(a.to(device), name + '_')()


@pytest.mark.parametrize('name,dtype', CASES)
def test_elementary_boundaries(name, dtype):
    values = [0., -0., 1e-300, -1e-300, 1e-40, -1e-40, 1e-12, -1e-12,
              .25, -.25, .34657359027997265, 1., 2., 10., 20., -40., 88., 89.,
              709., 709.7827128933839, 710., -745., -1075., -1074., 1023., 1024.,
              1e30, -1e30, math.inf, -math.inf, math.nan]
    a = torch.tensor(values, dtype=torch.float64).to(dtype)
    close(operation(name)(a.to('ruda')), operation(name)(a.cuda()), tiny=True)
    if dtype.is_complex:
        values = [complex(x, y) for x in (0., -0., 1., -1., math.inf, -math.inf, math.nan)
                  for y in (0., -0., .5, -2., math.inf, -math.inf, math.nan)]
        a = torch.tensor(values, dtype=dtype)
        close(operation(name)(a.to('ruda')), operation(name)(a.cuda()))
    else:
        actual = operation(name)(torch.tensor([0., -0.], dtype=dtype, device='ruda')).cpu()
        expected = operation(name)(torch.tensor([0., -0.], dtype=dtype, device='cuda')).cpu()
        mask = (actual == 0) & (expected == 0)
        assert torch.equal(torch.signbit(actual[mask]), torch.signbit(expected[mask]))


@pytest.mark.parametrize('name,dtype', [(n, d) for n, d in CASES if d in FLOATS[2:] + COMPLEX])
def test_elementary_unary_gradients(name, dtype):
    source = torch.tensor([[.25, .5, 1.5], [2., .75, .125]], dtype=dtype).t()
    if dtype.is_complex:
        source = source + .3j
    def evaluate(device, mutate):
        x = source.to(device).detach().requires_grad_()
        y = getattr(x.clone(), name + '_')() if mutate else operation(name)(x)
        first = torch.autograd.grad(y.real.sum(), x, create_graph=True)[0]
        second = torch.autograd.grad(first.real.sum(), x)[0]
        return y, first, second
    for mutate in ([False] if name == 'entr' else [False, True]):
        for actual, expected in zip(evaluate('ruda', mutate), evaluate('cpu', mutate)):
            close(actual, expected)


@pytest.mark.parametrize('name,dtype', BINARY)
def test_logaddexp_variants(name, dtype):
    function = getattr(torch, name)
    a = torch.tensor([[-30., 0., 30.]], dtype=dtype)
    b = torch.tensor([[-.75], [1.25], [50.]], dtype=dtype)
    if dtype.is_complex:
        a = a + .3j
        b = b - 1.1j
    left, right = a.to('ruda'), b.to('ruda')
    expected = function(a.cuda(), b.cuda())
    close(function(left, right), expected)
    for out in (torch.empty((3, 3), device='ruda', dtype=dtype).t(), torch.empty(0, device='ruda', dtype=dtype)):
        assert function(left, right, out=out) is out
        close(out, expected)
    value = left.expand(3, 3).clone()
    close(function(value, right, out=value), expected)
    for a, b in ((a[:, :0], b), (torch.tensor(.125, dtype=dtype), b)):
        close(function(a.to('ruda'), b.to('ruda')), function(a.cuda(), b.cuda()))


@pytest.mark.parametrize('name,dtype', BINARY)
def test_logaddexp_boundaries(name, dtype):
    values = [-math.inf, -1000., -1e-30, 0., 1e-30, 1000., math.inf, math.nan]
    a = torch.tensor(values, dtype=dtype).reshape(-1, 1)
    b = a.t()
    close(getattr(torch, name)(a.to('ruda'), b.to('ruda')), getattr(torch, name)(a.cuda(), b.cuda()))
    if dtype.is_complex:
        values = [complex(x, y) for x in (-math.inf, -2., 0., math.inf, math.nan)
                  for y in (-math.inf, -math.pi, 0., .75, math.inf, math.nan)]
        a = torch.tensor(values, dtype=dtype).reshape(-1, 1)
        b = a.t()
        close(getattr(torch, name)(a.to('ruda'), b.to('ruda')), getattr(torch, name)(a.cuda(), b.cuda()))


@pytest.mark.parametrize('name,dtype', [(n, d) for n, d in BINARY if d in FLOATS[2:] + COMPLEX])
def test_logaddexp_gradients(name, dtype):
    def evaluate(device):
        a = torch.tensor([[-3., .1, 4.]], dtype=dtype, device=device, requires_grad=True)
        b = torch.tensor([[-.2], [1.3]], dtype=dtype, device=device, requires_grad=True)
        value = getattr(torch, name)(a + (.3j if dtype.is_complex else 0), b)
        first = torch.autograd.grad(value.real.sum(), (a, b), create_graph=True)
        second = torch.autograd.grad(sum(t.real.sum() for t in first), (a, b))
        return (value,) + first + second
    for actual, expected in zip(evaluate('ruda'), evaluate('cpu')):
        close(actual, expected)


@pytest.mark.parametrize('name', ['logaddexp', 'logaddexp2'])
@pytest.mark.parametrize('left,right', [(torch.float16, torch.float32), (torch.float32, torch.float64),
                                       (torch.int64, torch.float32), (torch.bool, torch.float16)])
def test_logaddexp_promotion(name, left, right):
    function = getattr(torch, name)
    a = torch.tensor([0, 1, 2], dtype=left)
    b = torch.tensor([[.25], [.75]], dtype=right)
    close(function(a.to('ruda'), b.to('ruda')), function(a.cuda(), b.cuda()))
    a = torch.tensor(1, dtype=left)
    for device in ('cuda', 'ruda'):
        with pytest.raises(RuntimeError):
            function(a, b.to(device))
        with pytest.raises(RuntimeError):
            function(b.to(device), a)


@pytest.mark.parametrize('dtype', FLOATS)
@pytest.mark.parametrize('eps', [None, -.1, 0., 1e-6, .2, .5, .75, math.nan])
def test_logit_variants_backward(dtype, eps):
    a = torch.tensor([-math.inf, -.1, -0., 0., 1e-6, .2, .5, .8, 1., 1.1, math.inf, math.nan], dtype=dtype).reshape(3, 4).t()
    value = a.to('ruda')
    expected = torch.logit(a.cuda(), eps)
    close(torch.logit(value, eps), expected)
    close(value.clone().logit_(eps), expected)
    out = torch.empty(0, device='ruda', dtype=dtype)
    close(torch.logit(value, eps, out=out), expected)
    grad = torch.tensor([1., -1., 0.], dtype=dtype).reshape(1, 3)
    function = torch.ops.aten.logit_backward
    reference = function(grad.cuda(), a.cuda(), eps)
    close(function(grad.to('ruda'), value, eps), reference)
    output = torch.empty((3, 4), device='ruda', dtype=dtype).t()
    close(function.grad_input(grad.to('ruda'), value, eps, grad_input=output), reference)
    for device in ('cuda', 'ruda'):
        with pytest.raises(RuntimeError):
            function(torch.tensor(1., dtype=dtype), a.to(device), eps)
        with pytest.raises(RuntimeError):
            function(a.to(device), torch.tensor(.5, dtype=dtype), eps)


@pytest.mark.parametrize('dtype', FLOATS[2:])
@pytest.mark.parametrize('eps', [None, -.1, 0., .2])
def test_logit_gradients(dtype, eps):
    def evaluate(device):
        x = torch.tensor([.1, .3, .7, .9], device=device, dtype=dtype, requires_grad=True)
        y = torch.logit(x, eps)
        first = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), x)[0]
        return y, first, second
    for actual, expected in zip(evaluate('ruda'), evaluate('cpu')):
        close(actual, expected)


@pytest.mark.parametrize('name', UNARY + ['logit', 'logaddexp', 'logaddexp2'])
def test_elementary_contracts(name):
    function = operation(name)
    for device in ('cuda', 'ruda'):
        a = torch.tensor([.1, .2, .3, .4], device=device)
        def apply(value, **kwargs):
            return function(value, value, **kwargs) if name.startswith('logaddexp') else function(value, **kwargs)
        with pytest.raises(RuntimeError):
            apply(a, out=torch.empty(4, dtype=torch.int64, device=device))
        with pytest.raises(RuntimeError):
            apply(a, out=a[:1].expand(4))
        with pytest.raises(RuntimeError):
            apply(a[:-1], out=a[1:])
        with pytest.raises(RuntimeError):
            apply(a.detach().requires_grad_(), out=torch.empty_like(a))
        if name in ('entr', 'logit', 'logaddexp2'):
            with pytest.raises(RuntimeError):
                apply(a.to(torch.complex64))
        if name.startswith('logaddexp'):
            with pytest.raises(RuntimeError):
                apply(a.to(torch.int64))


def test_elementary_aliases_default_dtype():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        for name in ('exp2', 'expm1', 'entr', 'logit'):
            x = torch.tensor([0, 1, 2], device='ruda')
            close(getattr(torch.special, name)(x), getattr(torch.special, name)(x.cpu().cuda()))
        x = torch.tensor([1, 2, 8], device='ruda')
        for name in ('log2', 'log10'):
            close(getattr(torch, name)(x), getattr(torch, name)(x.cpu().cuda()))
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize('dtype', FLOATS[2:])
def test_elementary_probability_integration(dtype):
    def evaluate(device):
        x = torch.tensor([[-2., -.5, .25, 2.]], dtype=dtype, device=device, requires_grad=True)
        p = x.sigmoid()
        mixture = torch.logaddexp(torch.logit(p), -torch.logit(p))
        loss = (mixture + torch.expm1(-p) + torch.special.entr(p) + torch.log2(1+p)).sum()
        first = torch.autograd.grad(loss, x, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), x)[0]
        return loss, first, second
    for actual, expected in zip(evaluate('ruda'), evaluate('cpu')):
        close(actual, expected)
