import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F


FLOATS = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
FAMILIES = ['layer', 'rms', 'group', 'batch']
SHAPES = {'layer': [(3, 17), (2, 3, 5), (0, 7), (2, 0)],
          'rms': [(3, 17), (2, 3, 5), (0, 7), (2, 0)],
          'group': [(2, 6, 5), (2, 6), (0, 6, 3), (2, 6, 2, 3)],
          'batch': [(3, 4), (2, 4, 3), (2, 4, 3, 5), (2, 4, 2, 3, 5)]}


@pytest.fixture(scope='module', autouse=True)
def backend():
    if os.environ.get('RUDA_REQUIRE_GPU') != '1':
        pytest.skip('requires native RUDA GPU backend')
    import ruda_torch
    assert ruda_torch._scientific_available
    with torch.backends.cudnn.flags(enabled=False):
        yield


def equal(a, b, *, tol=None, dtype=True):
    if a is None or b is None:
        assert a is None and b is None
        return
    a, b = a.cpu(), b.cpu()
    if tol is None:
        tol = .035 if a.dtype == torch.bfloat16 else .006 if a.dtype == torch.float16 else 3e-5 if a.dtype == torch.float32 else 3e-11
    torch.testing.assert_close(a, b, rtol=tol, atol=tol, equal_nan=True, check_dtype=dtype)


def data(family, case, dtype, affine, device):
    shape = SHAPES[family][case]
    x = torch.sin(torch.arange(math.prod(shape), dtype=torch.float64) * .13).reshape(shape).to(dtype)
    parameter_shape = shape[-2:] if family in ('layer', 'rms') and case == 1 else shape[-1:] if family in ('layer', 'rms') else (shape[1],)
    n = math.prod(parameter_shape)
    weight = torch.linspace(.3, 1.3, n, dtype=dtype).reshape(parameter_shape) if affine else None
    bias = torch.linspace(-.2, .3, n, dtype=dtype).reshape(parameter_shape) if affine and family != 'rms' else None
    return tuple(v.to(device) if v is not None else None for v in (x, weight, bias)), parameter_shape


def native(family, x, w, b, shape, *, train=True, eps=1e-5, running=None, outputs=None):
    if family == 'layer':
        op, args = torch.ops.aten.native_layer_norm, (x, list(shape), w, b, eps)
    elif family == 'rms':
        return torch.ops.aten._fused_rms_norm(x, list(shape), w, eps)
    elif family == 'group':
        op, args = torch.ops.aten.native_group_norm, (x, w, b, x.shape[0], x.shape[1], math.prod(x.shape[2:]), 3, eps)
    else:
        op, args = torch.ops.aten.native_batch_norm, (x, w, b, *running, train, .2, eps)
    if outputs is None:
        return op.default(*args)
    names = ('out', 'save_mean', 'save_invstd') if family == 'batch' else ('out0', 'out1', 'out2')
    return op.out(*args, **dict(zip(names, outputs)))


def native_backward(family, g, x, w, b, shape, stats, mask, *, train=True, running=None, outputs=None):
    if family == 'layer':
        op, args = torch.ops.aten.native_layer_norm_backward, (g, x, list(shape), *stats, w, b, mask)
    elif family == 'rms':
        return torch.ops.aten._fused_rms_norm_backward(g, x, list(shape), stats[0], w, mask[:2])
    elif family == 'group':
        op, args = torch.ops.aten.native_group_norm_backward, (g, x, *stats, w, x.shape[0], x.shape[1], math.prod(x.shape[2:]), 3, mask)
    else:
        op, args = torch.ops.aten.native_batch_norm_backward, (g, x, w, *running, *stats, train, 1e-5, mask)
    return op.default(*args) if outputs is None else op.out(*args, **dict(zip(('out0', 'out1', 'out2'), outputs)))


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('case', range(4))
@pytest.mark.parametrize('dtype', FLOATS)
@pytest.mark.parametrize('affine', [False, True])
def test_normalization_native_forward_backward_out(family, case, dtype, affine):
    (x, w, b), shape = data(family, case, dtype, affine, 'ruda')
    (cx, cw, cb), _ = data(family, case, dtype, affine, 'cuda')
    for train in ([False, True] if family == 'batch' else [True]):
        run = (torch.zeros(x.shape[1], dtype=dtype, device='ruda'), torch.ones(x.shape[1], dtype=dtype, device='ruda')) if family == 'batch' else None
        crun = tuple(t.cpu().cuda() for t in run) if run else None
        actual = native(family, x, w, b, shape, train=train, running=run)
        expected = native(family, cx, cw, cb, shape, train=train, running=crun)
        for a, e in zip(actual, expected):
            equal(a, e)
        if run:
            for a, e in zip(run, crun):
                equal(a, e)
        if family != 'rms':
            outs = tuple(torch.empty(0, dtype=e.dtype, device='ruda') for e in expected)
            direct = native(family, x, w, b, shape, train=train, running=run, outputs=outs)
            reference = native(family, cx, cw, cb, shape, train=train, running=crun)
            for a, out, e in zip(direct, outs, reference):
                assert a is out;equal(a, e)
        g = torch.cos(torch.arange(x.numel(), dtype=torch.float64) * .17).reshape(x.shape).to(dtype)
        mask = [True, affine, affine and family != 'rms']
        gradients = native_backward(family, g.to('ruda'), x, w, b, shape, actual[1:], mask, train=train, running=run)
        expected_gradients = native_backward(family, g.cuda(), cx, cw, cb, shape, expected[1:], mask, train=train, running=crun)
        for a, e in zip(gradients, expected_gradients):
            equal(a, e)
        if affine and family != 'rms':
            outs = tuple(torch.empty(0, dtype=e.dtype, device='ruda') for e in expected_gradients)
            for a, e in zip(native_backward(family, g.to('ruda'), x, w, b, shape, actual[1:], mask, train=train, running=run, outputs=outs), expected_gradients):
                equal(a, e)


def public(family, x, w, b, *, training=True, eps=1e-5):
    if family == 'layer':
        return F.layer_norm(x, x.shape[-2:], w, b, eps)
    if family == 'rms':
        return F.rms_norm(x, x.shape[-2:], w, eps)
    if family == 'group':
        return F.group_norm(x, 2, w, b, eps)
    return F.batch_norm(x, torch.zeros(x.shape[1], dtype=w.dtype, device=x.device), torch.ones(x.shape[1], dtype=w.dtype, device=x.device), w, b, training, .3, eps)


def gradient_case(family, dtype, training, device):
    x = torch.linspace(-1.2, 2.1, 48, dtype=dtype).reshape(2, 4, 6).to(device).requires_grad_()
    shape = (4, 6) if family in ('layer', 'rms') else (4,)
    weight = torch.linspace(.3, 1.3, math.prod(shape), dtype=dtype).reshape(shape).to(device).requires_grad_()
    bias = None if family == 'rms' else torch.linspace(-.3, .2, math.prod(shape), dtype=dtype).reshape(shape).to(device).requires_grad_()
    inputs = (x, weight) if bias is None else (x, weight, bias)
    value = public(family, x, weight, bias, training=training)
    first = torch.autograd.grad(value.sin().square().sum(), inputs, create_graph=True)
    second = torch.autograd.grad(sum(v.square().sum() for v in first), inputs)
    return tuple(v.detach().cpu() for v in (value, *first, *second))


GRADIENTS = [(family, train) for family in FAMILIES for train in ([False, True] if family == 'batch' else [True])]


@pytest.fixture(scope='module')
def cuda_gradients(tmp_path_factory):
    path = tmp_path_factory.mktemp('norm-reference') / 'gradients.pt'
    subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), str(path)], env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1'), check=True)
    return torch.load(path, weights_only=True)


@pytest.mark.parametrize('family,training', GRADIENTS)
@pytest.mark.parametrize('dtype', FLOATS[2:])
def test_normalization_first_second_gradients(family, training, dtype, cuda_gradients):
    for a, e in zip(gradient_case(family, dtype, training, 'ruda'), cuda_gradients[(family, dtype, training)]):
        equal(a, e)


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('dtype', FLOATS)
def test_normalization_public_layouts(family, dtype):
    x = torch.sin(torch.arange(2 * 4 * 3 * 5, dtype=torch.float64)).reshape(2, 4, 3, 5).to(dtype)
    for x in (x, x.to(memory_format=torch.channels_last), x.transpose(2, 3), x[:1].expand(2, 4, 3, 5)):
        shape = x.shape[-2:] if family in ('layer', 'rms') else (x.shape[1],)
        w, b = torch.ones(shape, dtype=dtype), torch.full(shape, .2, dtype=dtype)
        for eps in (0., 1e-5, .1):
            if family == 'batch' and eps == 0.:
                for device in ('cuda', 'ruda'):
                    with pytest.raises(ValueError, match='eps must be positive'):
                        public(family, x.to(device), w.to(device), b.to(device), eps=eps)
                continue
            expected = public(family, x.cuda(), w.cuda(), b.cuda(), eps=eps)
            actual = public(family, x.to('ruda'), w.to('ruda'), b.to('ruda'), eps=eps)
            equal(actual, expected)
            if family == 'batch':
                assert actual.is_contiguous(memory_format=torch.channels_last) == expected.is_contiguous(memory_format=torch.channels_last)


@pytest.mark.parametrize('dtype', FLOATS)
def test_normalization_constant_special_default_epsilon(dtype):
    for value in (0., .5, 100., math.inf, math.nan):
        x = torch.full((2, 4, 3), value, dtype=dtype)
        for family in FAMILIES:
            shape = (4, 3) if family in ('layer', 'rms') else (4,)
            w, b = torch.ones(shape, dtype=dtype), torch.zeros(shape, dtype=dtype)
            for eps in (0., 1e-5):
                if family == 'batch' and eps == 0.:
                    for device in ('cuda', 'ruda'):
                        with pytest.raises(ValueError, match='eps must be positive'):
                            public(family, x.to(device), w.to(device), b.to(device), eps=eps)
                    equal(torch.ops.aten.native_batch_norm(x.to('ruda'), w.to('ruda'), b.to('ruda'), None, None, True, .1, eps)[0],
                          torch.ops.aten.native_batch_norm(x.cuda(), w.cuda(), b.cuda(), None, None, True, .1, eps)[0])
                else:
                    equal(public(family, x.to('ruda'), w.to('ruda'), b.to('ruda'), eps=eps), public(family, x.cuda(), w.cuda(), b.cuda(), eps=eps))
    tiny = torch.tensor([[1e-5, -1e-5, 0., 2e-5]], dtype=dtype)
    equal(F.rms_norm(tiny.to('ruda'), (4,)), F.rms_norm(tiny.cuda(), (4,)))


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('family', ['rms', 'batch'])
def test_mixed_precision_normalization(family, dtype):
    x = torch.linspace(-1., 1.3, 48, dtype=dtype).reshape(2, 4, 6)
    shape = (4, 6) if family == 'rms' else (4,)
    w = torch.linspace(.3, 1.3, math.prod(shape)).reshape(shape)
    b = torch.zeros_like(w)
    equal(public(family, x.to('ruda'), w.to('ruda'), b.to('ruda')), public(family, x.cuda(), w.cuda(), b.cuda()))


@pytest.mark.parametrize('dtype', FLOATS)
def test_batch_norm_statistics_and_aliases(dtype):
    x = torch.sin(torch.arange(120, dtype=torch.float64)).reshape(2, 4, 3, 5).to(dtype)
    a, c = x.to('ruda'), x.cuda()
    stats = torch.ops.aten.batch_norm_stats(a, .01)
    expected = torch.ops.aten.batch_norm_stats(c, .01)
    for v, e in zip(stats, expected):
        equal(v, e)
    out_stats = tuple(torch.empty(0, device='ruda', dtype=t.dtype) for t in stats)
    for v, e in zip(torch.ops.aten.batch_norm_stats.out(a, .01, out0=out_stats[0], out1=out_stats[1]), expected):
        equal(v, e)
    equal(torch.ops.aten.batch_norm_elemt(a, None, None, *stats, .01), torch.ops.aten.batch_norm_elemt(c, None, None, *expected, .01))
    out = torch.empty_like(a)
    equal(torch.ops.aten.batch_norm_elemt.out(a, None, None, *stats, .01, out=out), torch.ops.aten.batch_norm_elemt(c, None, None, *expected, .01))
    rm, rv = torch.zeros(4, dtype=dtype), torch.ones(4, dtype=dtype)
    am, av, cm, cv = rm.to('ruda'), rv.to('ruda'), rm.cuda(), rv.cuda()
    for v, e in zip(torch.ops.aten.batch_norm_update_stats(a, am, av, .3), torch.ops.aten.batch_norm_update_stats(c, cm, cv, .3)):
        equal(v, e)
    equal(am, cm);equal(av, cv)
    for op in (torch.ops.aten._native_batch_norm_legit, torch.ops.aten.native_batch_norm):
        for v, e in zip(op(a, None, None, am, av, True, .2, .01), op(c, None, None, cm, cv, True, .2, .01)):
            equal(v, e)
    for v, e in zip(torch.ops.aten._native_batch_norm_legit.no_stats(a, None, None, True, .2, .01), torch.ops.aten._native_batch_norm_legit.no_stats(c, None, None, True, .2, .01)):
        equal(v, e)
    for v, e in zip(torch.ops.aten._native_batch_norm_legit_no_training(a, None, None, am, av, .2, .01), torch.ops.aten._native_batch_norm_legit_no_training(c, None, None, cm, cv, .2, .01)):
        equal(v, e)
    x = torch.linspace(-1., 1.3, 120, dtype=dtype).reshape(2, 4, 3, 5)
    results = []
    for device in ('ruda', 'cuda'):
        value = x.to(device); weight = torch.linspace(.4, 1.1, 4, dtype=dtype, device=device)
        mean = torch.zeros_like(weight); variance = torch.ones_like(weight)
        _, saved_mean, saved_inv, reserve = torch.ops.aten._batch_norm_with_update(value, weight, torch.zeros_like(weight), mean, variance, .2, .01)
        results.append(torch.ops.aten.batch_norm_backward(torch.ones_like(value), value, weight, mean, variance, saved_mean, saved_inv, True, .01, [True, True, True], reserve))
    for v, e in zip(*results):
        equal(v, e)


@pytest.mark.parametrize('family', FAMILIES)
def test_normalization_masks_and_errors(family):
    (x, w, b), shape = data(family, 0, torch.float64, True, 'ruda')
    (cx, cw, cb), _ = data(family, 0, torch.float64, True, 'cuda')
    running = (x.new_zeros((x.shape[1],)), x.new_ones((x.shape[1],))) if family == 'batch' else None
    crun = tuple(v.cpu().cuda() for v in running) if running else None
    result = native(family, x, w, b, shape, running=running)
    reference = native(family, cx, cw, cb, shape, running=crun)
    for mask in ([False, False, False], [True, False, False], [False, True, False], [False, False, True]):
        if family == 'rms' and mask[2]:
            continue
        for v, e in zip(native_backward(family, torch.ones_like(x), x, w, b, shape, result[1:], mask, running=running), native_backward(family, torch.ones_like(cx), cx, cw, cb, shape, reference[1:], mask, running=crun)):
            equal(v, e)
    if family in ('layer', 'rms'):
        with pytest.raises(RuntimeError):
            native(family, x, w, b, (999,))
    else:
        with pytest.raises(RuntimeError):
            native(family, x, w[:-1], b, shape, running=running)


@pytest.mark.parametrize('dtype', FLOATS)
def test_large_normalization_rows_and_fp64_precision(dtype):
    for width in (1536, 4096, 4097):
        x = torch.sin(torch.arange(2 * width, dtype=torch.float64) * .31).reshape(2, width).to(dtype)
        for family in ('layer', 'rms'):
            fn = F.layer_norm if family == 'layer' else F.rms_norm
            equal(fn(x.to('ruda'), (width,)), fn(x.cuda(), (width,)))
        x = torch.full((2, width), 100., dtype=dtype)
        equal(F.layer_norm(x.to('ruda'), (width,)), F.layer_norm(x.cuda(), (width,)))
    if dtype == torch.float64:
        x = torch.tensor([[1.000000001, 1.000000003, 1.000000007, 1.000000011]], dtype=dtype)
        equal(F.layer_norm(x.to('ruda'), (4,), eps=1e-20), F.layer_norm(x.cuda(), (4,), eps=1e-20), tol=3e-7)


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.backends.cudnn.enabled = False
    values = {(family, dtype, training): gradient_case(family, dtype, training, 'cuda') for family, training in GRADIENTS for dtype in FLOATS[2:]}
    torch.save(values, sys.argv[1])
