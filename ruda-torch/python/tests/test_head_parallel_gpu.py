"""Native zero-copy split-K at the actual Qwen LM-head gradient geometry."""
import pytest
import torch
import ruda_torch as r


@pytest.fixture(scope='module')
def head_case():
    torch.manual_seed(313)
    a=torch.randn(33,152064,dtype=torch.bfloat16)
    b=torch.randn(152065,3584,dtype=torch.bfloat16)
    expected=(a[1:].cuda().double()@b[1:].cuda().double()).cpu()
    return a,b,expected


@pytest.mark.parametrize('rows',[31,32])
def test_parallel_head_gradient_and_aligned_storage_offsets(head_case,rows,monkeypatch):
    monkeypatch.setenv('RUDA_TORCH_BF16_PRECISION','cuda')
    a,b,reference=head_case
    x=a.to('ruda')[1:rows+1];y=b.to('ruda')[1:]
    actual=(x@y).cpu();expected=reference[:rows].to(torch.bfloat16)
    assert torch.isfinite(actual).all()
    aa,bb=actual.double(),expected.double()
    assert (aa-bb).norm()/bb.norm() < 1e-3
    torch.testing.assert_close(actual,expected,rtol=.025,atol=.025)


@pytest.mark.parametrize('rows',[31,32])
def test_head_geometry_preserves_fp32_output(head_case,rows,monkeypatch):
    monkeypatch.setenv('RUDA_TORCH_BF16_PRECISION','cuda')
    a,b,reference=head_case
    target=torch.empty(rows,3584,device='ruda',dtype=torch.float32)
    r._C.execute(7,a.to('ruda')[1:rows+1],b.to('ruda')[1:],target,0.0)
    actual=target.cpu().double()
    assert torch.isfinite(actual).all()
    expected=reference[:rows]
    assert (actual-expected).norm()/expected.norm() < 1e-4
