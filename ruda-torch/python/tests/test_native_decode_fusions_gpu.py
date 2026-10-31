"""Native small-M weight reuse and last-arrival paged split reduction."""
import os
import pytest
import torch
from paged_selected_reference import SPEC, data, dense


@pytest.fixture(scope='module')
def backend():
    if os.environ.get('RUDA_REQUIRE_GPU') != '1':
        pytest.skip('explicit native GPU execution required')
    import ruda_torch as r
    assert os.environ.get('RUDA_CUDA_COMPILER') == 'ptx'
    return r


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('rows', [1, 2, 3, 4])
@pytest.mark.parametrize('k,n,padding', [(33, 7, 3), (256, 129, 0), (1024, 512, 8)])
def test_small_m_forward_backward(backend, dtype, rows, k, n, padding):
    torch.manual_seed(542)
    a = torch.randn(rows, k + padding, dtype=dtype)[:, :k]
    w = torch.randn(n, k + padding, dtype=dtype)[:, :k]
    ad = a.to('ruda').detach().requires_grad_()
    wd = w.to('ruda').detach().requires_grad_()
    # Recreate padded rows on-device; a .to() is allowed to make a packed copy.
    if padding:
        ad = torch.cat((ad, torch.zeros(rows, padding, dtype=dtype, device='ruda')), 1)[:, :k]
        wd = torch.cat((wd, torch.zeros(n, padding, dtype=dtype, device='ruda')), 1)[:, :k]
        ad.retain_grad(); wd.retain_grad()
    ac = a.cuda().detach().requires_grad_(); wc = w.cuda().detach().requires_grad_()
    y = ad @ wd.t(); expected = ac @ wc.t()
    tolerance = 0.04 if dtype == torch.bfloat16 else 0.005
    torch.testing.assert_close(y.cpu(), expected.cpu(), rtol=tolerance, atol=tolerance)
    go = torch.randn(rows, n, dtype=dtype)
    y.backward(go.to('ruda')); expected.backward(go.cuda())
    for result, reference in ((ad.grad, ac.grad), (wd.grad, wc.grad)):
        torch.testing.assert_close(result.cpu(), reference.cpu(), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('rows', [1, 2, 3, 4])
def test_small_m_compiled(backend, dtype, rows):
    a = torch.randn(rows, 256, dtype=dtype).to('ruda')
    w = torch.randn(129, 256, dtype=dtype).to('ruda')
    fn = backend.compile(lambda x, weight: x @ weight.t(), native='required',
                         min_native_ops=1, fullgraph=True)
    for offset in (0., .125):
        a.add_(offset)
        expected = a.cpu().float() @ w.cpu().float().t()
        torch.testing.assert_close(fn(a, w).cpu().float(), expected,
                                   rtol=.04 if dtype == torch.bfloat16 else .005,
                                   atol=.04 if dtype == torch.bfloat16 else .005)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('splits', [2, 8, 32])
@pytest.mark.parametrize('mla', [False, True])
@pytest.mark.parametrize('strategy', ['atomic', 'ordered'])
def test_split_merge_reuse_and_gradients(backend, dtype, splits, mla, strategy):
    plan = backend.PagedAttentionPlan(**SPEC, splits=splits, backward_strategy=strategy)
    ts = data(mla, dtype=dtype)
    dev = [x.to('ruda').requires_grad_() for x in ts]
    call = plan.mla if mla else plan.attention
    tolerance = {torch.float32: 3e-5, torch.float16: .005, torch.bfloat16: .04}[dtype]
    for repetition in range(12):
        # Different contents, same buffers and counter workspace on every replay.
        values = [x + repetition * .01 for x in ts]
        with torch.no_grad():
            for dst, src in zip(dev, values): dst.copy_(src)
        ref = [x.float().cuda().requires_grad_() for x in values]
        expected = dense(ref, mla, causal=bool(repetition % 2))
        before = backend.execution_stats()
        actual = call(*dev, scale=.37, causal=bool(repetition % 2))
        after = backend.execution_stats()
        assert after['kernel_launches'] - before['kernel_launches'] == 1
        torch.testing.assert_close(actual.cpu().float(), expected.cpu(), rtol=tolerance, atol=tolerance)
        if repetition == 11:
            go = torch.randn_like(expected).to(dtype)
            actual.backward(go.cpu().to('ruda')); expected.backward(go.float())
            for result, reference in zip(dev, ref):
                torch.testing.assert_close(result.grad.cpu().float(), reference.grad.cpu(),
                                           rtol=tolerance, atol=tolerance)
