import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F


FLOATS = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
LOSSES = ['mse_loss', 'smooth_l1_loss', 'huber_loss', 'binary_cross_entropy', 'soft_margin_loss']


@pytest.fixture(scope='module', autouse=True)
def backend():
    if os.environ.get('RUDA_REQUIRE_GPU') != '1':
        pytest.skip('requires native RUDA GPU backend')
    import ruda_torch
    assert ruda_torch._scientific_available


def equal(a, b, *, exact=False, tol=None):
    a, b = a.cpu(), b.cpu()
    if tol is None:
        tol = 0 if exact else .025 if a.dtype == torch.bfloat16 else .004 if a.dtype == torch.float16 else 8e-6 if a.dtype == torch.float32 else 8e-12
    torch.testing.assert_close(a, b, rtol=tol, atol=tol, equal_nan=True)


def operands(dtype, name):
    x = torch.linspace(-1.7, 2.3, 24, dtype=dtype).reshape(4, 6).t()
    y = torch.linspace(.9, -.8, 24, dtype=dtype).reshape(4, 6).t()
    if name == 'binary_cross_entropy':
        x, y = x.sigmoid(), y.sigmoid()
    return x, y


def args(name, reduction):
    return (None, reduction) if name == 'binary_cross_entropy' else (reduction, .7) if name in ('smooth_l1_loss', 'huber_loss') else (reduction,)


def forward(name, x, y, reduction, out=None):
    op = getattr(torch.ops.aten, name)
    return op.default(x, y, *args(name, reduction)) if out is None else op.out(x, y, *args(name, reduction), out=out)


def backward(name, g, x, y, reduction, out=None):
    op = getattr(torch.ops.aten, name + '_backward')
    if out is None:
        return op.default(g, x, y, *args(name, reduction))
    fn = op.out if name == 'huber_loss' else op.grad_input
    return fn(g, x, y, *args(name, reduction), grad_input=out)


@pytest.mark.parametrize('name', LOSSES)
@pytest.mark.parametrize('dtype', FLOATS)
@pytest.mark.parametrize('reduction', [0, 1, 2])
def test_loss_native_forward_backward_out(name, dtype, reduction):
    x, y = operands(dtype, name)
    a, b, cx, cy = x.to('ruda'), y.to('ruda'), x.cuda(), y.cuda()
    expected = forward(name, cx, cy, reduction)
    value = forward(name, a, b, reduction)
    equal(value, expected)
    out = torch.empty((4, 6), dtype=dtype, device='ruda').t() if reduction == 0 else torch.empty(0, dtype=dtype, device='ruda')
    assert forward(name, a, b, reduction, out) is out
    equal(out, expected)
    g = torch.full(expected.shape, .7, dtype=dtype)
    expected_grad = backward(name, g.cuda(), cx, cy, reduction)
    equal(backward(name, g.to('ruda'), a, b, reduction), expected_grad)
    out = torch.empty((4, 6), dtype=dtype, device='ruda').t()
    assert backward(name, g.to('ruda'), a, b, reduction, out) is out
    equal(out, expected_grad)


@pytest.mark.parametrize('name', LOSSES)
@pytest.mark.parametrize('dtype', FLOATS)
def test_loss_scalar_empty_special(name, dtype):
    values = [0., 1., 1e-6, .5, .9, .2] if name == 'binary_cross_entropy' else [-math.inf, -100., -0., 0., math.inf, math.nan]
    for x in (torch.tensor(values, dtype=dtype), torch.tensor(.2, dtype=dtype), torch.empty((2, 0), dtype=dtype)):
        y = torch.full_like(x, .3)
        for reduction in (0, 1, 2):
            expected = forward(name, x.cuda(), y.cuda(), reduction)
            equal(forward(name, x.to('ruda'), y.to('ruda'), reduction), expected)
            g = torch.ones_like(expected).cpu()
            equal(backward(name, g.to('ruda'), x.to('ruda'), y.to('ruda'), reduction), backward(name, g.cuda(), x.cuda(), y.cuda(), reduction))


def nll_data(dtype, rank, weighted, device, *, empty=False, ignored=False, zero_weight=False):
    shape = {1: (5,), 2: (7, 5), 4: (2, 5, 3, 4)}[rank]
    if empty:
        shape = (0, 5) if rank == 2 else (2, 5, 0, 4)
    x = torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape).mul(-.017).to(dtype)
    target_shape = () if rank == 1 else (shape[0], *shape[2:])
    target = (torch.arange(math.prod(target_shape)).reshape(target_shape) % 5).long()
    if target.numel() > 1:
        target.flatten()[::3] = -100
    if ignored:
        target.fill_(-100)
    weight = torch.tensor([0., .3, 1.7, -.2, .7], dtype=dtype) if weighted else None
    if weight is not None and zero_weight:
        weight.zero_()
    return x.to(device), target.to(device), None if weight is None else weight.to(device)


def nll_forward(rank, x, t, w, reduction, out=None):
    op = torch.ops.aten.nll_loss2d_forward if rank == 4 else torch.ops.aten.nll_loss_forward
    return op.default(x, t, w, reduction, -100) if out is None else op.output(x, t, w, reduction, -100, output=out[0], total_weight=out[1])


def nll_backward(rank, g, x, t, w, reduction, total, out=None):
    op = torch.ops.aten.nll_loss2d_backward if rank == 4 else torch.ops.aten.nll_loss_backward
    return op.default(g, x, t, w, reduction, -100, total) if out is None else op.grad_input(g, x, t, w, reduction, -100, total, grad_input=out)


@pytest.mark.parametrize('dtype', FLOATS)
@pytest.mark.parametrize('rank', [1, 2, 4])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('reduction', [0, 1, 2])
def test_nll_native_forward_backward(dtype, rank, weighted, reduction):
    a, t, w = nll_data(dtype, rank, weighted, 'ruda')
    cx, ct, cw = nll_data(dtype, rank, weighted, 'cuda')
    expected, etotal = nll_forward(rank, cx, ct, cw, reduction)
    value, total = nll_forward(rank, a, t, w, reduction)
    equal(value, expected);equal(total, etotal)
    outs = (a.new_empty((0,)), a.new_empty((0,)))
    actual = nll_forward(rank, a, t, w, reduction, outs)
    assert actual[0] is outs[0] and actual[1] is outs[1]
    equal(outs[0], expected);equal(outs[1], etotal)
    g = torch.full(expected.shape, .7, dtype=dtype)
    expected_grad = nll_backward(rank, g.cuda(), cx, ct, cw, reduction, etotal)
    equal(nll_backward(rank, g.to('ruda'), a, t, w, reduction, total), expected_grad)
    out = a.new_empty((0,))
    assert nll_backward(rank, g.to('ruda'), a, t, w, reduction, total, out) is out
    equal(out, expected_grad)


@pytest.mark.parametrize('dtype', FLOATS)
@pytest.mark.parametrize('kind', ['empty', 'ignored', 'zero_weight', 'strided', 'uint8'])
def test_nll_edge_contracts(dtype, kind):
    for rank in ([1, 2] if kind == 'uint8' else [2, 4] if kind == 'empty' else [1, 2, 4]):
        x, t, w = nll_data(dtype, rank, True, 'cpu', **({kind: True} if kind in ('empty', 'ignored', 'zero_weight') else {}))
        if kind == 'uint8':
            t = t.clamp_min(0).to(torch.uint8)
        if kind == 'strided' and rank > 1:
            x = x.transpose(0, -1).contiguous().transpose(0, -1)
            if t.ndim > 1:
                t = t.transpose(0, -1).contiguous().transpose(0, -1)
        for reduction in (0, 1, 2):
            expected, total = nll_forward(rank, x.cuda(), t.cuda(), w.cuda(), reduction)
            actual, atotal = nll_forward(rank, x.to('ruda'), t.to('ruda'), w.to('ruda'), reduction)
            equal(actual, expected);equal(atotal, total)


def differentiable_case(case, dtype, device):
    x, target = operands(dtype, 'binary_cross_entropy' if case == 'binary_cross_entropy' else 'mse_loss')
    x = x.to(device).requires_grad_();target = target.to(device)
    if case in LOSSES:
        value = forward(case, x, target, 1)
    elif case == 'binary_cross_entropy_with_logits':
        target = target.sigmoid()
        value = F.binary_cross_entropy_with_logits(x, target, weight=torch.full((4,), .7, dtype=dtype, device=device), pos_weight=torch.full((4,), 1.3, dtype=dtype, device=device))
    elif case == 'cross_entropy':
        target = torch.tensor([0, 1, -100, 3, 0, 2], device=device)
        value = F.cross_entropy(x, target, label_smoothing=.2)
    elif case == 'l1_loss':
        value = F.l1_loss(x, target)
    else:
        value = F.kl_div(x.log_softmax(-1), target.softmax(-1), reduction='batchmean')
    first = torch.autograd.grad(value.square(), x, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), x)[0]
    return tuple(v.detach().cpu() for v in (value, first, second))


DIFFERENTIABLE = LOSSES + ['binary_cross_entropy_with_logits', 'cross_entropy', 'l1_loss', 'kl_div']


@pytest.fixture(scope='module')
def cuda_gradients(tmp_path_factory):
    path = tmp_path_factory.mktemp('loss-reference') / 'gradients.pt'
    subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), str(path)], env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1'), check=True)
    return torch.load(path, weights_only=True)


@pytest.mark.parametrize('case', DIFFERENTIABLE)
@pytest.mark.parametrize('dtype', FLOATS[2:])
def test_loss_first_second_derivatives(case, dtype, cuda_gradients):
    for a, b in zip(differentiable_case(case, dtype, 'ruda'), cuda_gradients[(case, dtype)]):
        equal(a, b)


@pytest.mark.parametrize('dtype', FLOATS)
@pytest.mark.parametrize('rank', [2, 3, 4])
@pytest.mark.parametrize('smooth', [0., .2])
def test_cross_entropy_composition(dtype, rank, smooth):
    shape = (2, 5, *([3] * (rank - 2)))
    x = torch.linspace(-1.2, 2.3, math.prod(shape), dtype=dtype).reshape(shape)
    indices = torch.arange(x.numel() // 5).reshape(shape[:1] + shape[2:]).long() % 5
    indices.flatten()[0] = -100
    weight = torch.tensor([.7, .2, 1.3, .9, 0.], dtype=dtype)
    soft_target = x.softmax(1)
    for target in (indices, soft_target):
        for reduction in ('none', 'mean', 'sum'):
            expected = F.cross_entropy(x.cuda(), target.cuda(), weight.cuda(), reduction=reduction, label_smoothing=smooth)
            equal(F.cross_entropy(x.to('ruda'), target.to('ruda'), weight.to('ruda'), reduction=reduction, label_smoothing=smooth), expected)


@pytest.mark.parametrize('dtype', FLOATS)
@pytest.mark.parametrize('weighted', [False, True])
def test_bce_logits_composition(dtype, weighted):
    x = torch.tensor([[-100., -20., -1e-7, 0., 1e-7], [.13, 4., 20., 100., math.inf]], dtype=dtype)
    target = torch.linspace(0., 1., 10, dtype=dtype).reshape_as(x)
    weight = torch.tensor([.7, .2, 0., -.3, 1.3], dtype=dtype) if weighted else None
    for reduction in ('none', 'mean', 'sum'):
        expected = F.binary_cross_entropy_with_logits(x.cuda(), target.cuda(), None if weight is None else weight.cuda(), pos_weight=torch.tensor(1.3, dtype=dtype, device='cuda'), reduction=reduction)
        equal(F.binary_cross_entropy_with_logits(x.to('ruda'), target.to('ruda'), None if weight is None else weight.to('ruda'), pos_weight=torch.tensor(1.3, dtype=dtype, device='ruda'), reduction=reduction), expected)


@pytest.mark.parametrize('logarithmic', [False, True])
@pytest.mark.parametrize('shape,dim', [((), 0), ((0, 3), 1), ((3, 1), 1), ((2, 33), -1), ((3, 4, 5), 1), ((2, 4097), 1)])
def test_float64_softmax_dependency(logarithmic, shape, dim):
    x = torch.linspace(-12., 17., math.prod(shape), dtype=torch.float64).reshape(shape)
    if len(shape) == 3:
        x = x.transpose(0, 2)
    name = '_log_softmax' if logarithmic else '_softmax'
    op = getattr(torch.ops.aten, name)
    a = x.to('ruda');expected = op.default(x.cuda(), dim, False)
    actual = op.default(a, dim, False);equal(actual, expected)
    out = torch.empty_like(a)
    equal(op.out(a, dim, False, out=out), expected)
    grad = torch.linspace(-.7, 1.2, x.numel(), dtype=torch.float64).reshape(x.shape)
    back = getattr(torch.ops.aten, name + '_backward_data')
    egrad = back.default(grad.cuda(), expected, dim, torch.float64)
    equal(back.default(grad.to('ruda'), actual, dim, torch.float64), egrad)
    kw = {'out' if logarithmic else 'grad_input': out}
    equal(back.out(grad.to('ruda'), actual, dim, torch.float64, **kw), egrad)


@pytest.mark.parametrize('name', ['mse_loss', 'smooth_l1_loss', 'huber_loss'])
def test_loss_promotion_and_fp64_precision(name):
    x = torch.tensor([[1.00000000001], [2.00000000003]], dtype=torch.float64)
    y = torch.tensor([[1., 2., 3.]], dtype=torch.float32)
    for reduction in (0, 1, 2):
        equal(forward(name, x.to('ruda'), y.to('ruda'), reduction), forward(name, x.cuda(), y.cuda(), reduction))
        for dtype in FLOATS:
            actual = torch.empty(0, dtype=dtype, device='ruda')
            expected = torch.empty(0, dtype=dtype, device='cuda')
            equal(forward(name, x.to('ruda'), y.to('ruda'), reduction, actual), forward(name, x.cuda(), y.cuda(), reduction, expected))
    g = torch.tensor(.7, dtype=torch.float32)
    if name != 'huber_loss':
        equal(backward(name, g.to('ruda'), x.float().to('ruda'), y.double().to('ruda'), 1), backward(name, g.cuda(), x.float().cuda(), y.double().cuda(), 1))
    a = torch.tensor([1.00000000001], dtype=torch.float64)
    b = torch.ones_like(a)
    equal(forward(name, a.to('ruda'), b.to('ruda'), 0), forward(name, a.cuda(), b.cuda(), 0), tol=1e-25)


def test_loss_rejections_and_boundary_parameters():
    x = torch.tensor([0., .5, 1.], device='ruda')
    for value in (-.1, 1.1, math.nan):
        with pytest.raises(RuntimeError, match='between 0 and 1'):
            F.binary_cross_entropy(torch.tensor([value], device='ruda'), torch.tensor([.3], device='ruda'))
    for beta in (0., .001, 1e5):
        a = torch.tensor([-.1, 0., .1], dtype=torch.float64)
        b = torch.zeros_like(a)
        equal(torch.ops.aten.smooth_l1_loss(a.to('ruda'), b.to('ruda'), 0, beta), torch.ops.aten.smooth_l1_loss(a.cuda(), b.cuda(), 0, beta))
        equal(torch.ops.aten.smooth_l1_loss_backward(torch.ones_like(a).to('ruda'), a.to('ruda'), b.to('ruda'), 0, beta), torch.ops.aten.smooth_l1_loss_backward(torch.ones_like(a).cuda(), a.cuda(), b.cuda(), 0, beta))
    with pytest.raises(RuntimeError):
        torch.ops.aten.smooth_l1_loss(x, x, 0, -.1)
    with pytest.raises(RuntimeError):
        torch.ops.aten.huber_loss(x, x, 0, 0.)
    for target in (-1, 5, 2**40):
        with pytest.raises(RuntimeError, match='out of bounds'):
            F.nll_loss(torch.zeros((1, 5), device='ruda'), torch.tensor([target], device='ruda'))
    with pytest.raises(RuntimeError):
        F.nll_loss(torch.zeros((1, 5), device='ruda'), torch.tensor([0], device='ruda', dtype=torch.int32))


@pytest.mark.parametrize('dtype', FLOATS)
def test_large_nll_reductions(dtype):
    for shape in ((4099, 7), (2, 7, 257, 513)):
        x = torch.full(shape, -.125, dtype=dtype)
        t = torch.arange(x.numel() // 7).reshape(shape[:1] + shape[2:]).long() % 7
        rank = len(shape)
        for reduction in (1, 2):
            actual = nll_forward(rank, x.to('ruda'), t.to('ruda'), None, reduction)
            expected = nll_forward(rank, x.cuda(), t.cuda(), None, reduction)
            for a, b in zip(actual, expected):
                equal(a, b)


if __name__ == '__main__':
    torch.set_num_threads(1)
    result = {(case, dtype): differentiable_case(case, dtype, 'cuda') for case in DIFFERENTIABLE for dtype in FLOATS[2:]}
    torch.save(result, sys.argv[1])
