import math
import os

import pytest
import torch


FLOATS = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
INTS = [torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
NAMES = ['frexp', 'ldexp', 'copysign', 'nextafter', 'frac', 'round']


@pytest.fixture(scope='module', autouse=True)
def backend():
    if os.environ.get('RUDA_REQUIRE_GPU') != '1':
        pytest.skip('requires native RUDA GPU backend')
    import ruda_torch
    assert ruda_torch._scientific_available


def exact(actual, expected, *, payload=False):
    actual, expected = actual.cpu().resolve_neg().resolve_conj(), expected.cpu().resolve_neg().resolve_conj()
    if payload:
        assert actual.dtype == expected.dtype
        integer = {2: torch.int16, 4: torch.int32, 8: torch.int64}[actual.element_size()]
        assert torch.equal(actual.contiguous().view(integer), expected.contiguous().view(integer))
    else:
        if actual.is_complex():
            actual, expected = torch.view_as_real(actual), torch.view_as_real(expected)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
        if actual.is_floating_point():
            zeros = (actual == 0) & (expected == 0)
            assert torch.equal(torch.signbit(actual[zeros]), torch.signbit(expected[zeros]))


def samples(dtype):
    info = torch.finfo(dtype)
    return torch.tensor([0., -0., .5, -.5, 1., -1., 1.5, -1.5, 2.5, 125., -125.,
                         info.tiny, -info.tiny, info.tiny * info.eps, -info.tiny * info.eps,
                         info.max, -info.max, math.inf, -math.inf, math.nan], dtype=dtype)


@pytest.mark.parametrize('dtype', FLOATS)
def test_frexp_variants(dtype):
    a = samples(dtype).reshape(4, 5).t()
    value = a.to('ruda')
    reference = torch.frexp(a.cuda())
    result = torch.frexp(value)
    assert result.mantissa.dtype == dtype and result.exponent.dtype == torch.int32
    for x, y in zip(result, reference):
        exact(x, y)
    for m, e in ((torch.empty((4, 5), device='ruda', dtype=dtype).t(), torch.empty((4, 5), device='ruda', dtype=torch.int32).t()),
                 (torch.empty(0, device='ruda', dtype=dtype), torch.empty(0, device='ruda', dtype=torch.int32)),
                 (value.clone(), torch.empty_like(value, dtype=torch.int32))):
        result = torch.frexp(value if m is not value else m, out=(m, e))
        assert result.mantissa is m and result.exponent is e
        for x, y in zip(result, reference):
            exact(x, y)
    alias = value.clone()
    exact(torch.frexp(alias, out=(alias, torch.empty_like(alias, dtype=torch.int32))).mantissa, reference.mantissa)
    for a in (torch.tensor(-0., dtype=dtype), torch.empty((2, 0, 3), dtype=dtype), torch._neg_view(a)):
        for x, y in zip(torch.frexp(a.to('ruda')), torch.frexp(a.cuda())):
            exact(x, y)


@pytest.mark.parametrize('dtype', FLOATS[:2])
def test_all_reduced_float_bit_patterns(dtype):
    a = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype)
    value, reference = a.to('ruda'), a.cuda()
    for x, y in zip(torch.frexp(value), torch.frexp(reference)):
        exact(x, y)
    exact(torch.frac(value), torch.frac(reference))
    for direction in (0., -0., 1., -1., math.inf, -math.inf, math.nan):
        b = torch.full_like(a, direction)
        exact(torch.nextafter(value, b.to('ruda')), torch.nextafter(reference, b.cuda()))
        exact(torch.copysign(value, b.to('ruda')), torch.copysign(reference, b.cuda()), payload=True)
    for n in (-200, -25, -1, 0, 1, 25, 200):
        b = torch.full(a.shape, n, dtype=torch.int32)
        exact(torch.ldexp(value, b.to('ruda')), torch.ldexp(reference, b.cuda()))


@pytest.mark.parametrize('dtype', FLOATS)
@pytest.mark.parametrize('exponent_dtype', INTS)
def test_ldexp_integral_exponent(dtype, exponent_dtype):
    a = samples(dtype).reshape(5, 4).t()
    ns = [0, 1, 2, 3] if exponent_dtype == torch.bool else [0, 1, 20, 100] if exponent_dtype == torch.uint8 else [-100, -1, 0, 100]
    b = torch.tensor(ns, dtype=exponent_dtype).reshape(4, 1)
    value, exponent = a.to('ruda'), b.to('ruda')
    reference = torch.ldexp(a.cuda(), b.cuda())
    exact(torch.ldexp(value, exponent), reference)
    exact(value.clone().ldexp_(exponent), reference)
    for out in (torch.empty(0, dtype=dtype, device='ruda'), torch.empty((5, 4), dtype=dtype, device='ruda').t()):
        assert torch.ldexp(value, exponent, out=out) is out
        exact(out, reference)
    alias = value.clone()
    exact(torch.ldexp(alias, exponent, out=alias), reference)


@pytest.mark.parametrize('dtype', FLOATS)
def test_ldexp_boundaries(dtype):
    a = samples(dtype).reshape(-1, 1)
    b = torch.tensor([-2**63, -2**31-1, -2**31, -4096, -1075, -1022, -150, -149, -134, -133,
                      -127, -126, -25, -24, -14, -1, 0, 1, 10, 127, 128, 1023, 1024,
                      4096, 2**31-1, 2**31, 2**32, 2**32+1, 2**63-1], dtype=torch.int64)
    exact(torch.ldexp(a.to('ruda'), b.to('ruda')), torch.ldexp(a.cuda(), b.cuda()))
    for shape in ((), (0, 3)):
        a = torch.full(shape, .75, dtype=dtype)
        b = torch.ones(shape, dtype=torch.int32)
        exact(torch.ldexp(a.to('ruda'), b.to('ruda')), torch.ldexp(a.cuda(), b.cuda()))
    for device in ('cuda', 'ruda'):
        with pytest.raises(RuntimeError):
            torch.ldexp(torch.ones(2, dtype=dtype, device=device), torch.tensor(1))


@pytest.mark.parametrize('a_dtype,b_dtype', [(a, b) for a in FLOATS + [torch.int64, torch.bool, torch.complex64, torch.complex128]
                                           for b in [torch.float32, torch.float64, torch.complex64]])
def test_ldexp_general_exponents(a_dtype, b_dtype):
    a = torch.tensor([[0, 1, 2]], dtype=a_dtype)
    b = torch.tensor([[-.5], [1.25]], dtype=b_dtype)
    if a_dtype.is_complex:
        a += .25j
    if b_dtype.is_complex:
        b += .5j
    actual = torch.ldexp(a.to('ruda'), b.to('ruda')).cpu()
    expected = torch.ldexp(a.cuda(), b.cuda()).cpu()
    def power(device):
        exponent = b.to(device)
        return torch.pow(2.0, exponent) if a_dtype in INTS or a_dtype == torch.float32 else torch.full((), 2., dtype=a_dtype, device=device).pow(exponent)
    ruda_power, cuda_power = power('ruda').cpu(), power('cuda').cpu()
    assert ruda_power.dtype == cuda_power.dtype
    def tolerance_for(dtype):
        return .02 if dtype == torch.bfloat16 else .003 if dtype == torch.float16 else 3e-6 if dtype in (torch.float32, torch.complex64) else 3e-12
    power_tolerance = tolerance_for(cuda_power.dtype)
    torch.testing.assert_close(ruda_power, cuda_power, atol=power_tolerance, rtol=power_tolerance)
    final_tolerance = tolerance_for(actual.dtype)
    torch.testing.assert_close(actual, a * ruda_power, atol=final_tolerance, rtol=final_tolerance)
    tolerance = max(power_tolerance, final_tolerance)
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('dtype', FLOATS)
@pytest.mark.parametrize('name', ['copysign', 'nextafter'])
def test_sign_neighbor_variants(name, dtype):
    function = getattr(torch, name)
    a = samples(dtype).reshape(5, 4).t()
    b = torch.tensor([[0.], [-0.], [math.inf], [-math.inf]], dtype=dtype)
    value, other = a.to('ruda'), b.to('ruda')
    reference = function(a.cuda(), b.cuda())
    exact(function(value, other), reference, payload=name == 'copysign')
    exact(getattr(value.clone(), name + '_')(other), reference, payload=name == 'copysign')
    for out in (torch.empty(0, device='ruda', dtype=dtype), torch.empty((5, 4), device='ruda', dtype=dtype).t()):
        exact(function(value, other, out=out), reference, payload=name == 'copysign')
    alias = value.clone()
    exact(function(alias, other, out=alias), reference)
    for a in (torch.tensor(-0., dtype=dtype), torch.empty((0, 3), dtype=dtype)):
        b = torch.zeros_like(a)
        exact(function(a.to('ruda'), b.to('ruda')), function(a.cuda(), b.cuda()))


@pytest.mark.parametrize('a_dtype,b_dtype', [(torch.float16, torch.float64), (torch.float32, torch.bfloat16),
                                           (torch.int64, torch.float32), (torch.bool, torch.float16),
                                           (torch.int32, torch.int64)])
@pytest.mark.parametrize('name', ['copysign', 'nextafter'])
def test_sign_neighbor_promotion_devices(name, a_dtype, b_dtype):
    a = torch.tensor([0, 1, 3], dtype=a_dtype)
    b = torch.tensor([[0], [-1]], dtype=b_dtype)
    fn = getattr(torch, name)
    if name == 'nextafter' and a_dtype in INTS and b_dtype in INTS:
        for device in ('cuda', 'ruda'):
            with pytest.raises(RuntimeError):
                fn(a.to(device), b.to(device))
    else:
        exact(fn(a.to('ruda'), b.to('ruda')), fn(a.cuda(), b.cuda()))
    scalar = torch.tensor(-0., dtype=torch.float64)
    if name == 'copysign':
        for x, y in ((a, scalar), (scalar, a)):
            if x is scalar:
                exact(fn(x, y.to('ruda')), fn(x, y.cuda()))
            else:
                exact(fn(x.to('ruda'), y), fn(x.cuda(), y))
        exact(fn(a.to('ruda'), -0.), fn(a.cuda(), -0.))
    else:
        for device in ('cuda', 'ruda'):
            with pytest.raises(RuntimeError):
                fn(a.to(device), scalar)


@pytest.mark.parametrize('dtype', FLOATS)
@pytest.mark.parametrize('decimals', [None, 0, 1, 2, 3, 4, -1, -3, 20, 100, 400])
def test_fraction_round_variants(dtype, decimals):
    a = samples(dtype).reshape(4, 5).t()
    def fn(x, **kwargs):
        return torch.frac(x, **kwargs) if decimals is None else torch.round(x, decimals=decimals, **kwargs)
    value = a.to('ruda')
    reference = fn(a.cuda())
    exact(fn(value), reference)
    alias = value.clone()
    exact(alias.frac_() if decimals is None else alias.round_(decimals=decimals), reference)
    for out in (torch.empty(0, dtype=dtype, device='ruda'), torch.empty((4, 5), dtype=dtype, device='ruda').t()):
        exact(fn(value, out=out), reference)
    if decimals is not None:
        exact(torch.special.round(value, decimals=decimals), reference)
    for a in (torch.tensor(-0., dtype=dtype), torch.empty((0, 3), dtype=dtype)):
        exact(fn(a.to('ruda')), fn(a.cuda()))


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('dtype', FLOATS[2:])
def test_floating_gradients(name, dtype):
    def evaluate(device, mutate):
        x = torch.tensor([-.75, -.25, .5, 1.25], dtype=dtype, device=device, requires_grad=True)
        b = torch.tensor([-.5, 1., 2., -1.], dtype=dtype, device=device, requires_grad=True)
        if name == 'frexp':
            y = torch.frexp(x).mantissa
        elif name == 'round':
            y = x.clone().round_(decimals=1) if mutate else torch.round(x, decimals=1)
        elif name == 'frac':
            y = x.clone().frac_() if mutate else x.frac()
        else:
            y = getattr(x.clone(), name + '_')(b) if mutate else getattr(torch, name)(x, b)
        first = torch.autograd.grad(y.square().sum(), x, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), x)[0] if first.requires_grad else None
        return y, first, second
    for mutate in ([False] if name == 'frexp' else [False, True]):
        for actual, expected in zip(evaluate('ruda', mutate), evaluate('cpu', mutate)):
            if expected is None:
                assert actual is None
            else:
                torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=3e-6 if dtype == torch.float32 else 3e-12,
                                           atol=3e-6 if dtype == torch.float32 else 3e-12)


@pytest.mark.parametrize('name', NAMES)
def test_floating_contracts(name):
    for device in ('cuda', 'ruda'):
        a = torch.tensor([.1, .2, .3, .4], device=device)
        def call(x, **kwargs):
            if name in ('ldexp', 'copysign', 'nextafter'):
                return getattr(torch, name)(x, torch.ones_like(x), **kwargs)
            if name == 'round':
                return torch.ops.aten.round.decimals(x, decimals=2, **kwargs) if not kwargs else torch.ops.aten.round.decimals_out(x, decimals=2, **kwargs)
            return getattr(torch, name)(x, **kwargs)
        if name == 'frexp':
            with pytest.raises(RuntimeError):
                call(a, out=(torch.empty(4, dtype=torch.float64, device=device), torch.empty(4, dtype=torch.int32, device=device)))
            with pytest.raises(RuntimeError):
                call(a, out=(torch.empty_like(a), torch.empty(4, dtype=torch.int64, device=device)))
        else:
            for out in (torch.empty(4, dtype=torch.int64, device=device), a[:1].expand(4)):
                with pytest.raises(RuntimeError):
                    call(a, out=out)
            with pytest.raises(RuntimeError):
                call(a[:-1], out=a[1:])
            with pytest.raises(RuntimeError):
                call(a.detach().requires_grad_(), out=torch.empty_like(a))
        if name in ('frexp', 'frac', 'round', 'nextafter', 'copysign'):
            with pytest.raises(RuntimeError):
                call(a.to(torch.complex64))
        if name in ('frexp', 'frac', 'round', 'nextafter'):
            with pytest.raises(RuntimeError):
                call(a.to(torch.int64))


@pytest.mark.parametrize('dtype', FLOATS[2:])
def test_binary_scale_roundtrip_and_integration(dtype):
    x = torch.tensor([.125, -.75, 1.25, -100.], dtype=dtype)
    m, e = torch.frexp(x.to('ruda'))
    exact(torch.ldexp(m, e), x)
    def evaluate(device):
        a = x.to(device).requires_grad_()
        mantissa, exponent = torch.frexp(a)
        reconstructed = torch.ldexp(mantissa, exponent)
        y = torch.copysign(reconstructed.frac(), a)
        loss = (y.square() + mantissa.square()).sum()
        first = torch.autograd.grad(loss, a, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), a)[0]
        return loss, first, second
    for actual, expected in zip(evaluate('ruda'), evaluate('cpu')):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=3e-6 if dtype == torch.float32 else 3e-12,
                                   atol=3e-6 if dtype == torch.float32 else 3e-12, equal_nan=True)


@pytest.mark.parametrize('dtype', INTS)
def test_integral_power_dependency(dtype):
    exponents = ([0, 1, 2, 3, 7, 31, 63, 64] if dtype in (torch.bool, torch.uint8)
                 else [torch.iinfo(dtype).min, -3, -2, -1, 0, 1, 2, 3, 7, 31, 63, 64])
    exponent = torch.tensor(exponents, dtype=dtype)
    for base in (0, 1, -1, 2, -2, 3):
        expected = torch.pow(base, exponent.cuda())
        value = exponent.to('ruda')
        torch.testing.assert_close(torch.pow(base, value).cpu(), expected.cpu(), rtol=0, atol=0)
        out = torch.empty((exponent.numel(), 2), dtype=expected.dtype, device='ruda')[:, 0]
        torch.testing.assert_close(torch.pow(base, value, out=out).cpu(), expected.cpu(), rtol=0, atol=0)
    if dtype != torch.bool:
        a = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=dtype).t()
        b = torch.tensor([[0, 2]], dtype=dtype)
        torch.testing.assert_close(torch.pow(a.to('ruda'), b.to('ruda')).cpu(), torch.pow(a.cuda(), b.cuda()).cpu(), rtol=0, atol=0)
        for n in (0, 1, 2, 3, 6):
            torch.testing.assert_close(a.to('ruda').pow_(n).cpu(), a.cuda().pow_(n).cpu(), rtol=0, atol=0)
        for device in ('cuda', 'ruda'):
            with pytest.raises(RuntimeError):
                a.to(device).pow_(-1)
    for shape in ((), (0, 3)):
        b = torch.ones(shape, dtype=dtype)
        torch.testing.assert_close(torch.pow(2, b.to('ruda')).cpu(), torch.pow(2, b.cuda()).cpu(), rtol=0, atol=0)


@pytest.mark.parametrize('dtype', FLOATS[2:])
def test_where_scalar_second_gradient(dtype):
    for scalar_first in (False, True):
        def evaluate(device):
            x = torch.tensor([[-1., 2.]], dtype=dtype, device=device, requires_grad=True)
            condition = torch.tensor([[True], [False]], device=device)
            y = torch.where(condition, 0., x) if scalar_first else torch.where(condition, x, 0.)
            first = torch.autograd.grad(y.square().sum(), x, create_graph=True)[0]
            second = torch.autograd.grad(first.sum(), x)[0]
            return y, first, second
        for actual, expected in zip(evaluate('ruda'), evaluate('cpu')):
            torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
