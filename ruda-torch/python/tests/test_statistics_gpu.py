import math
import os
import warnings

import pytest
import torch


@pytest.fixture(scope='module', autouse=True)
def backend():
    if os.environ.get('RUDA_REQUIRE_GPU') != '1':
        pytest.skip('requires native RUDA GPU backend')
    import ruda_torch
    assert ruda_torch._C.abi_version == 10
    if os.environ.get('RUDA_REQUIRE_STATISTICS_NATIVE') == '1':
        assert ruda_torch._statistics_available
    if os.environ.get('RUDA_STATISTICS_COMPOSITION_ONLY') == '1':
        assert not ruda_torch._statistics_available
        assert not ruda_torch._ordering_available


DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.complex64, torch.complex128]


def equal(actual, expected):
    if isinstance(expected, tuple):
        for a, b in zip(actual, expected):
            equal(a, b)
        return
    tolerance = 2e-2 if expected.dtype == torch.bfloat16 else 3e-3 if expected.dtype == torch.float16 else 3e-6
    if expected.dtype in (torch.float64, torch.complex128):
        tolerance = 1e-10
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance, equal_nan=True)


def data(dtype, length=17):
    a = torch.arange(2*3*length, dtype=torch.float64).reshape(2, 3, length) / 31 - 1
    if dtype.is_complex:
        a = torch.complex(a, a.flip(-1) * .3)
    return a.to(dtype).transpose(0, 2)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('correction', [0, 1, .5, -2, 80])
def test_moments_dimensions_correction_and_layout(dtype, correction):
    a = data(dtype)
    b = a.to('ruda')
    for operation in (torch.var, torch.std, torch.var_mean, torch.std_mean):
        for dims, keep in ((None, False), ((), True), ((0, -1), False), ((1,), True)):
            equal(operation(b, dim=dims, correction=correction, keepdim=keep),
                  operation(a, dim=dims, correction=correction, keepdim=keep))
        equal(operation(b, unbiased=False), operation(a, unbiased=False))
    if dtype.is_complex:
        equal(torch.var_mean(b.conj(), dim=0), torch.var_mean(a.conj(), dim=0))


@pytest.mark.parametrize('dtype', DTYPES[:4])
@pytest.mark.parametrize('length', [1, 3, 255, 256, 257, 513, 65537])
def test_moments_tiles_and_stability(dtype, length):
    a = (torch.arange(2*length, dtype=torch.float64).reshape(2, length) % 7 + 1000).to(dtype)
    equal(torch.var_mean(a.to('ruda'), dim=-1, correction=0), torch.var_mean(a, dim=-1, correction=0))


@pytest.mark.parametrize('dtype', DTYPES)
def test_moments_empty_scalar_outputs(dtype):
    for shape, dim in (((), None), ((2, 0, 3), 1), ((0, 3), 1), ((2, 0, 3), (0, 2))):
        a = torch.ones(shape, dtype=dtype)
        for operation in (torch.var_mean, torch.std_mean):
            equal(operation(a.to('ruda'), dim=dim, correction=0), operation(a, dim=dim, correction=0))
    a = data(dtype)
    expected = torch.var_mean(a, dim=0)
    outs = tuple(torch.empty((2, 3), dtype=x.dtype, device='ruda').t() for x in expected)
    returned = torch.ops.aten.var_mean.correction_out(a.to('ruda'), [0], out0=outs[0], out1=outs[1])
    assert all(x is y for x, y in zip(returned, outs))
    equal(returned, expected)
    out = torch.empty(0, dtype=expected[0].dtype, device='ruda')
    equal(torch.std(a.to('ruda'), dim=0, out=out), torch.std(a, dim=0))


@pytest.mark.parametrize('dtype', DTYPES[:4])
def test_moments_nonfinite(dtype):
    a = torch.tensor([[1., float('nan')], [float('inf'), 1.], [-float('inf'), float('inf')]], dtype=dtype)
    for operation in (torch.var, torch.std):
        equal(operation(a.to('ruda'), dim=1), operation(a, dim=1))


@pytest.mark.parametrize('dtype', DTYPES[:4])
def test_nan_reductions(dtype):
    a = torch.tensor([[[float('nan'), 1., 2.], [float('nan'), float('nan'), float('nan')]],
                      [[3., -4., 9.], [float('inf'), -float('inf'), 1.]]], dtype=dtype).transpose(0, 2)
    for operation in (torch.nansum, torch.nanmean):
        for dims in (None, (), (0, 2), (1,)):
            equal(operation(a.to('ruda'), dim=dims, keepdim=True), operation(a, dim=dims, keepdim=True))
        equal(operation(a.to('ruda'), dim=0, dtype=torch.float64), operation(a, dim=0, dtype=torch.float64))
        out = torch.empty(0, dtype=torch.float64, device='ruda')
        equal(operation(a.to('ruda'), dim=0, out=out), operation(a, dim=0, out=torch.empty(0, dtype=torch.float64)))
        for shape in ((0, 3), (3, 0), ()):
            value = torch.ones(shape, dtype=dtype)
            equal(operation(value.to('ruda'), dim=0), operation(value, dim=0))
    large = torch.tensor([60000., 60000.], dtype=dtype)
    equal(torch.nanmean(large.to('ruda')), torch.nanmean(large))


@pytest.mark.parametrize('dtype', [torch.bool, torch.int8, torch.int32, torch.int64])
def test_integer_nansum(dtype):
    a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype)
    equal(torch.nansum(a.to('ruda'), dim=0), torch.nansum(a, dim=0))


@pytest.mark.parametrize('dtype', DTYPES + [torch.int32])
def test_logsumexp(dtype):
    a = data(dtype) * 100
    for dims in ((0,), (0, 2), ()):
        for keep in (False, True):
            if not dims and not keep:
                with pytest.raises(RuntimeError):
                    torch.logsumexp(a, dims, keep)
                with pytest.raises(RuntimeError):
                    torch.logsumexp(a.to('ruda'), dims, keep)
                continue
            equal(torch.logsumexp(a.to('ruda'), dims, keep), torch.logsumexp(a, dims, keep))
    for shape in ((0, 3), (3, 0), ()):
        value = torch.ones(shape, dtype=dtype)
        equal(torch.logsumexp(value.to('ruda'), (0,)), torch.logsumexp(value, (0,)))
    if not dtype.is_complex:
        a = torch.tensor([[float('inf'), 1.], [-float('inf'), -float('inf')], [1., float('nan')]])
        equal(torch.logsumexp(a.to('ruda'), 1), torch.logsumexp(a, 1))


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('order', [0, 1, 2, 3.5, -.5, -2, math.inf, -math.inf, math.nan])
def test_norm_orders(dtype, order):
    a = data(dtype)
    for dim, keep in ((None, False), ((0, 2), True), ((1,), False)):
        equal(torch.linalg.vector_norm(a.to('ruda'), order, dim, keep), torch.linalg.vector_norm(a, order, dim, keep))
        equal(torch.norm(a.to('ruda'), order, dim, keep), torch.norm(a, order, dim, keep))
    singleton = torch.ones((2, 1), dtype=dtype)
    equal(torch.linalg.vector_norm(singleton.to('ruda'), order, dim=1), torch.linalg.vector_norm(singleton, order, dim=1))


@pytest.mark.parametrize('dtype', DTYPES)
def test_norm_empty_dtype_and_out(dtype):
    a = data(dtype)
    target = torch.complex128 if dtype.is_complex else torch.float64
    expected = torch.linalg.vector_norm(a, dtype=target, dim=0)
    out = torch.empty((2, 3), device='ruda', dtype=expected.dtype).t()
    equal(torch.linalg.vector_norm(a.to('ruda'), dim=0, dtype=target, out=out), expected)
    equal(torch.norm(a.to('ruda'), dim=0, dtype=target), torch.norm(a, dim=0, dtype=target))
    for shape, dim in (((0, 3), 1), ((3, 0), 1), ((), 0)):
        value = torch.ones(shape, dtype=dtype)
        for order in (0, 1, 2):
            equal(torch.linalg.vector_norm(value.to('ruda'), order, dim), torch.linalg.vector_norm(value, order, dim))
    with pytest.raises(RuntimeError):
        torch.linalg.vector_norm(torch.empty((3, 0), device='ruda', dtype=dtype), math.inf, 1)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.complex64, torch.complex128])
@pytest.mark.parametrize('operation', ['var', 'std', 'var_mean', 'std_mean', 'norm', 'logsumexp', 'nanmean', 'nansum'])
def test_gradients(dtype, operation):
    if dtype.is_complex and operation in ('nanmean', 'nansum'):
        pytest.skip('PyTorch NaN reductions reject complex input')
    source = data(dtype).detach()
    a = source.clone().requires_grad_()
    b = source.to('ruda').requires_grad_()
    function = torch.linalg.vector_norm if operation == 'norm' else getattr(torch, operation)
    def loss(value):
        result = function(value, dim=0)
        if isinstance(result, tuple):
            return result[0].sum() + result[1].real.sum()
        return result.real.sum()
    loss(a).backward()
    loss(b).backward()
    equal(b.grad, a.grad)
    if operation in ('nanmean', 'nansum'):
        source[0, 0, 0] = float('nan')
        source[:, 1, 1] = float('nan')
        a = source.clone().requires_grad_()
        b = source.to('ruda').requires_grad_()
        loss(a).backward()
        loss(b).backward()
        equal(b.grad, a.grad)


def test_invalid_axes_and_types():
    a = torch.ones((2, 3), device='ruda')
    for function in (torch.var, torch.std, torch.nansum, torch.nanmean, torch.logsumexp, torch.linalg.vector_norm):
        with pytest.raises((RuntimeError, IndexError)):
            function(a, dim=(0, 0))
        with pytest.raises((RuntimeError, IndexError)):
            function(a, dim=2)
    for function in (torch.var, torch.std, torch.nanmean, torch.linalg.vector_norm):
        with pytest.raises(RuntimeError):
            function(a.long())


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.complex64, torch.complex128])
@pytest.mark.parametrize('operation', ['var', 'std', 'var_mean', 'std_mean'])
@pytest.mark.parametrize('overload', ['default', 'dim', 'correction'])
@pytest.mark.parametrize('constant', [False, True])
def test_moments_canonical_autograd(dtype, operation, overload, constant):
    source = data(dtype, length=3).detach()
    if constant:
        source = torch.ones_like(source)
    host = source.clone().requires_grad_()
    device = source.to('ruda').requires_grad_()
    op = getattr(getattr(torch.ops.aten, operation), overload)
    def derivatives(a):
        if overload == 'default':
            outputs = op(a, False)
        elif overload == 'dim':
            outputs = op(a, [0, 2], False, True)
        else:
            outputs = op(a, [0, 2], correction=.5, keepdim=False)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        loss = sum(value.real.sum() + (value.imag.sum() * .3 if value.is_complex() else 0) for value in outputs)
        first = torch.autograd.grad(loss, a, create_graph=True)[0]
        second = None if constant else torch.autograd.grad((first.real * a.real.detach()).sum(), a)[0]
        return outputs, first, second
    expected = derivatives(host)
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='.*autograd kernel was not registered.*')
        actual = derivatives(device)
    equal(actual[0], expected[0])
    equal(actual[1], expected[1])
    if not constant:
        equal(actual[2], expected[2])


@pytest.mark.parametrize('operation', ['var', 'std', 'var_mean', 'std_mean'])
def test_moments_empty_gradients_and_out_rejection(operation):
    function = getattr(torch, operation)
    for shape, dim in (((2, 0, 3), 1), ((0, 3), 1), ((), 0)):
        host = torch.ones(shape, dtype=torch.float64, requires_grad=True)
        device = host.detach().to('ruda').requires_grad_()
        def gradient(a):
            result = function(a, dim=dim, correction=0)
            loss = sum(x.sum() for x in result) if isinstance(result, tuple) else result.sum()
            return torch.autograd.grad(loss, a)[0]
        equal(gradient(device), gradient(host))
    for device in ('cpu', 'ruda'):
        value = torch.arange(6., device=device).reshape(2, 3).requires_grad_()
        outputs = [torch.empty(2, device=device) for _ in range(2 if operation.endswith('_mean') else 1)]
        if operation.endswith('_mean'):
            result = getattr(torch.ops.aten, operation).correction_out(value, [1], out0=outputs[0], out1=outputs[1])
            with pytest.raises(RuntimeError, match='derivative.*not implemented'):
                sum(output.sum() for output in result).backward()
        else:
            with pytest.raises(RuntimeError, match='automatic differentiation|autograd'):
                function(value, dim=1, out=outputs[0])
