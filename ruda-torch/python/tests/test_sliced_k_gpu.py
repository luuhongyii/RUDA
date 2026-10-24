"""Native interleaved K reduction for the real Qwen K/V projection geometry."""
import pytest
import torch
from torch.nn import functional as F
import ruda_torch


@pytest.mark.parametrize('seed',[251,257])
@pytest.mark.parametrize('offset',[False,True])
def test_bf16_kv_sliced_reduction(seed,offset):
    torch.manual_seed(seed)
    width=3586 if offset else 3584
    a=torch.randn(512,width,dtype=torch.bfloat16)
    w=torch.randn(512,width,dtype=torch.bfloat16)
    x,y=a.to('ruda'),w.to('ruda')
    if offset:
        a,w=a[:,1:3585],w[:,1:3585]
        x,y=x[:,1:3585],y[:,1:3585]
    bias=torch.randn(512,dtype=torch.bfloat16)
    expected=F.linear(a.contiguous().cuda(),w.contiguous().cuda(),bias.cuda()).cpu()
    actual=F.linear(x,y,bias.to('ruda')).cpu()
    aa,bb=actual.double(),expected.double()
    assert torch.isfinite(actual).all()
    assert (aa-bb).norm()/bb.norm() < 1e-4
    torch.testing.assert_close(actual,expected,rtol=.01,atol=.01)


@pytest.mark.parametrize('seed',[263,269])
@pytest.mark.parametrize('offset',[False,True])
def test_bf16_down_projection_serial_writeback(seed,offset):
    torch.manual_seed(seed)
    width=18946 if offset else 18944
    a=torch.randn(512,width,dtype=torch.bfloat16)
    w=torch.randn(3584,width,dtype=torch.bfloat16)
    x,y=a.to('ruda'),w.to('ruda')
    if offset:
        a,w=a[:,1:18945],w[:,1:18945]
        x,y=x[:,1:18945],y[:,1:18945]
    expected=F.linear(a.contiguous().cuda(),w.contiguous().cuda()).cpu()
    actual=F.linear(x,y).cpu()
    aa,bb=actual.double(),expected.double()
    assert torch.isfinite(actual).all()
    assert (aa-bb).norm()/bb.norm() < 1e-4
    torch.testing.assert_close(actual,expected,rtol=.01,atol=.01)
