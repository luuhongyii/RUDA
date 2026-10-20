import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F


FLOATS = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
CASES = [('elu', (1.3, .7, 1.1)), ('leaky_relu', (.13,)), ('hardsigmoid', ()),
         ('hardswish', ()), ('softplus', (1.3, 12.)), ('hardshrink', (.13,)),
         ('softshrink', (.13,)), ('mish', ()), ('gelu', ('none',)), ('gelu', ('tanh',)),
         ('silu', ()), ('hardtanh', (-.7, .9)), ('threshold', (.13, -.25))]
INPLACE = {'elu', 'leaky_relu', 'hardsigmoid', 'hardswish', 'mish', 'gelu', 'silu', 'hardtanh', 'threshold'}


def invoke(name, x, args, *, out=None, inplace=False):
    op = getattr(torch.ops.aten, name + ('_' if inplace else ''))
    kw = {'approximate': args[0]} if name == 'gelu' else {}
    positional = () if name == 'gelu' else args
    return op.out(x, *positional, out=out, **kw) if out is not None else op.default(x, *positional, **kw)


def gradient_result(name, args, dtype, device, inplace):
    x = torch.tensor([[-2.3, -.41, .23], [.73, 2.1, 4.2]], device=device, dtype=dtype, requires_grad=True)
    if name in ('celu', 'selu', 'logsigmoid'):
        fn = getattr(F, name)
        y = fn(x.clone(), *args, inplace=inplace) if name != 'logsigmoid' else fn(x)
    else:
        y = invoke(name, x.clone(), args, inplace=inplace)
    first = torch.autograd.grad(y.square().sum(), x, create_graph=True)[0]
    try:
        second = torch.autograd.grad(first.sum(), x)[0] if first.requires_grad else None
    except RuntimeError as error:
        if name != 'hardsigmoid' or str(error) != 'derivative for aten::hardsigmoid_backward is not implemented':
            raise
        second = str(error)
    return tuple(v.detach().cpu() if isinstance(v, torch.Tensor) else v for v in (y, first, second))


@pytest.fixture(scope='module', autouse=True)
def backend():
    if os.environ.get('RUDA_REQUIRE_GPU') != '1':
        pytest.skip('requires native RUDA GPU backend')
    import ruda_torch
    assert ruda_torch._scientific_available


@pytest.fixture(scope='module')
def cuda_gradients(tmp_path_factory):
    path = tmp_path_factory.mktemp('activation-reference') / 'gradients.pt'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), str(path)], env=env, check=True)
    return torch.load(path, weights_only=True)


def equal(actual, expected, *, exact=False):
    actual, expected = actual.cpu(), expected.cpu()
    dtype = actual.dtype
    tol = 0 if exact else .02 if dtype == torch.bfloat16 else .003 if dtype == torch.float16 else 5e-6 if dtype in (torch.float32, torch.complex64) else 4e-12
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol, equal_nan=True)


@pytest.mark.parametrize('name,args', CASES)
@pytest.mark.parametrize('dtype', FLOATS)
def test_activation_forward_variants(name, args, dtype):
    values = [-math.inf, -1000., -100., -20., -4., -3., -1., -.7, -.13, -1e-7, -0., 0.,
              1e-7, .13, .9, 1., 3., 20., 100., 1000., math.inf, math.nan, torch.finfo(dtype).tiny, 2.5]
    x = torch.tensor(values, dtype=dtype).reshape(4, 6).t()
    a = x.to('ruda');reference = invoke(name, x.cuda(), args)
    equal(invoke(name, a, args), reference)
    if name in INPLACE:
        equal(invoke(name, a.clone(), args, inplace=True), reference)
    for out in (torch.empty(0, device='ruda', dtype=dtype), torch.empty((4, 6), device='ruda', dtype=dtype).t(), a.clone()):
        assert invoke(name, a, args, out=out) is out
        equal(out, reference)
    for x in (torch.tensor(-0., dtype=dtype), torch.empty((0, 2), dtype=dtype)):
        equal(invoke(name, x.to('ruda'), args), invoke(name, x.cuda(), args))


def direct_backward(name, grad, x, args, *, out=None):
    kw = {}
    if name == 'elu':
        operands = (grad, *args, False, x)
    elif name == 'leaky_relu':
        operands = (grad, x, *args, False)
    elif name == 'threshold':
        operands = (grad, x, args[0])
    elif name == 'gelu':
        operands = (grad, x);kw['approximate'] = args[0]
    else:
        operands = (grad, x, *args)
    packet = getattr(torch.ops.aten, name + '_backward')
    if out is not None:
        return packet.out(*operands, out=out, **kw) if name == 'hardswish' else packet.grad_input(*operands, grad_input=out, **kw)
    return packet.default(*operands, **kw)


@pytest.mark.parametrize('name,args', CASES)
@pytest.mark.parametrize('dtype', FLOATS)
def test_activation_direct_backward(name, args, dtype):
    x = torch.tensor([-math.inf, -100., -3., -.7, -.13, -0., 0., .13, .9, 3., math.inf, math.nan], dtype=dtype).reshape(3, 4).t()
    grad = torch.tensor([[1., -1., .5]], dtype=dtype).expand_as(x)
    a, g = x.to('ruda'), grad.to('ruda')
    reference = direct_backward(name, grad.cuda(), x.cuda(), args)
    equal(direct_backward(name, g, a, args), reference)
    if name != 'mish':
        out = torch.empty((3, 4), device='ruda', dtype=dtype).t()
        equal(direct_backward(name, g, a, args, out=out), reference)


@pytest.mark.parametrize('case', range(len(CASES)))
@pytest.mark.parametrize('dtype', FLOATS[2:])
def test_activation_first_second_gradients(case, dtype, cuda_gradients):
    name, args = CASES[case]
    for inplace in ([False, True] if name in INPLACE else [False]):
        reference = cuda_gradients[(name, args, dtype, inplace)]
        actual = gradient_result(name, args, dtype, 'ruda', inplace)
        for a, b in zip(actual, reference):
            if isinstance(b, str):
                assert a == b
                continue
            assert (a is None) == (b is None)
            if a is not None:
                equal(a, b)


@pytest.mark.parametrize('name,args', [('celu', (.7,)), ('selu', ()), ('logsigmoid', ())])
@pytest.mark.parametrize('dtype', FLOATS)
def test_composite_activation(name, args, dtype, cuda_gradients):
    x = torch.tensor([-math.inf, -100., -10., -1e-7, -0., 0., 1e-7, 3., math.inf, math.nan], dtype=dtype)
    equal(getattr(F, name)(x.to('ruda'), *args), getattr(F, name)(x.cuda(), *args))
    if dtype in FLOATS[2:]:
        for inplace in ([False, True] if name != 'logsigmoid' else [False]):
            for actual, expected in zip(gradient_result(name, args, dtype, 'ruda', inplace), cuda_gradients[(name, args, dtype, inplace)]):
                if expected is not None:
                    equal(actual, expected)


@pytest.mark.parametrize('dtype', FLOATS)
def test_prelu_and_log_sigmoid_outputs(dtype):
    x = torch.tensor([[-2., -.5, 0.], [.2, 1., 4.]], dtype=dtype)
    w = torch.tensor([.1, .3, -.2], dtype=dtype)
    a, weight = x.to('ruda'), w.to('ruda')
    equal(F.prelu(a, weight), F.prelu(x.cuda(), w.cuda()))
    grad = torch.ones_like(x)
    for actual, expected in zip(torch.ops.aten._prelu_kernel_backward(grad.to('ruda'), a, weight), torch.ops.aten._prelu_kernel_backward(grad.cuda(), x.cuda(), w.cuda())):
        equal(actual, expected)
    output, buffer = torch.ops.aten.log_sigmoid_forward(a)
    reference, ref_buffer = torch.ops.aten.log_sigmoid_forward(x.cuda())
    equal(output, reference);assert buffer.shape == ref_buffer.shape == (0,)
    out = torch.empty((3, 2), dtype=dtype, device='ruda').t()
    buf = torch.empty((0,), dtype=dtype, device='ruda')
    actual = torch.ops.aten.log_sigmoid_forward.output(a, output=out, buffer=buf)
    assert actual[0] is out and actual[1] is buf
    equal(out, reference)
    equal(torch.ops.aten.log_sigmoid_backward(grad.to('ruda'), a, buf), torch.ops.aten.log_sigmoid_backward(grad.cuda(), x.cuda(), ref_buffer))
    equal(torch.ops.aten.log_sigmoid_backward.grad_input(grad.to('ruda'), a, buf, grad_input=out), torch.ops.aten.log_sigmoid_backward(grad.cuda(), x.cuda(), ref_buffer))


@pytest.mark.parametrize('dtype', [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64])
def test_integer_threshold_hardtanh(dtype):
    x = torch.tensor([0, 1, 2, 3, 5, 7], dtype=dtype).reshape(2, 3).t()
    if dtype == torch.int64:
        x += 2**54
    lower = int(x.min())+1;upper = int(x.max())-1
    for name, args in [('threshold', (lower, upper)), ('hardtanh', (lower, upper)), ('hardtanh', (upper, lower))]:
        a = x.to('ruda');reference = invoke(name, x.cuda(), args)
        equal(invoke(name, a, args), reference, exact=True)
        equal(invoke(name, a.clone(), args, inplace=True), reference, exact=True)
        out = torch.empty(0, dtype=dtype, device='ruda');equal(invoke(name, a, args, out=out), reference, exact=True)
    equal(torch.ops.aten.threshold_backward(torch.ones_like(x).to('ruda'), x.to('ruda'), lower), torch.ops.aten.threshold_backward(torch.ones_like(x).cuda(), x.cuda(), lower), exact=True)


@pytest.mark.parametrize('name,args', CASES)
def test_activation_contracts(name, args):
    for device in ('cuda', 'ruda'):
        x = torch.ones(4, device=device)
        for out in (torch.empty(4, dtype=torch.int64, device=device), x[:1].expand(4)):
            if name == 'threshold' and out.stride() == (0,):
                equal(invoke(name, x, args, out=out), invoke(name, x, args), exact=True)
                continue
            with pytest.raises(RuntimeError):
                invoke(name, x, args, out=out)
        with pytest.raises(RuntimeError):
            invoke(name, x.requires_grad_(), args, out=torch.empty_like(x))
        with pytest.raises(RuntimeError):
            invoke(name, x.detach().to(torch.bool), args)
        if name != 'silu':
            with pytest.raises(RuntimeError):
                invoke(name, x.detach().to(torch.complex64), args)


@pytest.mark.parametrize('dtype', FLOATS)
def test_parameter_edge_cases(dtype):
    for device in ('cuda', 'ruda'):
        x = torch.tensor([-.5, 0., .5], dtype=dtype, device=device)
        with pytest.raises(RuntimeError):F.celu(x, alpha=0)
        with pytest.raises(RuntimeError):F.softshrink(x, lambd=-.1)
        with pytest.raises(RuntimeError):F.gelu(x, approximate='invalid')
        with pytest.raises(RuntimeError):torch.ops.aten.elu_backward(x, -.1, 1., 1., True, x)
        with pytest.raises(RuntimeError):torch.ops.aten.leaky_relu_backward(x, x, -.1, True)
    x = torch.tensor([-2., -.1, 0., .1, 2.], dtype=dtype)
    for args in [(0., 1., 1.), (-.5, 2., .7), (1., -.7, 2.), (2., .5, -1.)]:
        equal(torch.ops.aten.elu(x.to('ruda'), *args), torch.ops.aten.elu(x.cuda(), *args))
    for args in [(0., 20.), (-1., 20.), (2., -1.), (1., math.inf)]:
        equal(torch.ops.aten.softplus(x.to('ruda'), *args), torch.ops.aten.softplus(x.cuda(), *args))


@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_complex_silu(dtype):
    x = torch.tensor([[-2.+.3j, 0j, .1-1j], [4j, 1.-.5j, 3.+2j]], dtype=dtype).t()
    equal(F.silu(x.to('ruda')), F.silu(x.cuda()))
    equal(F.silu(x.to('ruda'), inplace=True), F.silu(x.cuda(), inplace=True))


@pytest.mark.parametrize('dtype', FLOATS + [torch.int64])
def test_threshold_overlapping_output(dtype):
    for device in ('cuda', 'ruda'):
        x = torch.ones(4, dtype=dtype, device=device)
        out = torch.empty(1, dtype=dtype, device=device).expand(4)
        equal(torch.ops.aten.threshold.out(x, 2, 3, out=out), torch.full((4,), 3, dtype=dtype), exact=True)
        equal(torch.ops.aten.threshold_backward.grad_input(x, x, 0, grad_input=out), torch.ones(4, dtype=dtype), exact=True)
        neg = torch._neg_view(torch.empty_like(x))
        equal(torch.ops.aten.threshold.out(x, 2, 3, out=neg), torch.full((4,), 3, dtype=dtype), exact=True)


@pytest.mark.parametrize('dtype', FLOATS[2:])
def test_prelu_parameter_gradients_and_backward_contract(dtype):
    results = []
    for device in ('cpu', 'ruda'):
        x = torch.tensor([[-1., -.4, .3], [1.2, -.7, 2.]], dtype=dtype, device=device, requires_grad=True)
        w = torch.tensor([.1, -.2, .4], dtype=dtype, device=device, requires_grad=True)
        y = F.prelu(x, w)
        first = torch.autograd.grad(y.square().sum(), (x, w), create_graph=True)
        second = torch.autograd.grad(first[0].sum()+first[1].sum(), (x, w))
        results.append([v.detach().cpu() for v in (y, *first, *second)])
        for name in ('silu_backward', 'mish_backward'):
            direct = getattr(torch.ops.aten, name)(torch.ones_like(x), x)
            with pytest.raises(RuntimeError, match=f'derivative for aten::{name} is not implemented'):
                torch.autograd.grad(direct.sum(), x)
    for a, b in zip(*results):
        equal(a, b)


if __name__ == '__main__':
    torch.set_num_threads(1)
    result = {}
    for dtype in FLOATS[2:]:
        for name, args in CASES + [('celu', (.7,)), ('selu', ()), ('logsigmoid', ())]:
            for inplace in ([False, True] if name in INPLACE or name in ('celu', 'selu') else [False]):
                result[(name, args, dtype, inplace)] = gradient_result(name, args, dtype, 'cuda', inplace)
    torch.save(result, sys.argv[1])
