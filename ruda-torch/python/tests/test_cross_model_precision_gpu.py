"""A100 CUDA-reference split-K regressions, independent of model names."""
import pytest
import torch
import ruda_torch as r


@pytest.fixture(autouse=True)
def precision(monkeypatch):
    monkeypatch.setenv('RUDA_TORCH_BF16_PRECISION','cuda')
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction


@pytest.mark.parametrize('inner',[4096,12288,14336])
@pytest.mark.parametrize('transposed',[False,True])
def test_serial_three_part_forward_and_input_gradient(inner,transposed):
    torch.manual_seed(20261026)
    a=torch.randn(512,inner,dtype=torch.bfloat16)
    b=torch.randn(4096,inner,dtype=torch.bfloat16).t() if transposed else torch.randn(inner,4096,dtype=torch.bfloat16)
    expected=(a.cuda()@b.cuda()).cpu()
    actual=(a.to('ruda')@b.to('ruda')).cpu()
    assert torch.isfinite(actual).all()
    assert (actual.double()-expected.double()).norm()/expected.double().norm()<1e-4


@pytest.mark.parametrize('shape',[(512,4096,1024),(31,32768,4096),(32,32768,4096),
                                  (31,151936,4096),(32,151936,4096)])
def test_rounded_parallel_partials_match_cuda(shape):
    torch.manual_seed(20261026)
    m,k,n=shape
    a=torch.randn(m,k,dtype=torch.bfloat16)
    b=torch.randn(n,k,dtype=torch.bfloat16).t() if m==512 else torch.randn(k,n,dtype=torch.bfloat16)
    expected=(a.cuda()@b.cuda()).cpu()
    actual=(a.to('ruda')@b.to('ruda')).cpu()
    assert torch.isfinite(actual).all()
    assert (actual.double()-expected.double()).norm()/expected.double().norm()<1e-4


@pytest.mark.parametrize('offset',[0,1])
@pytest.mark.parametrize('shape,parts,parallel',[
    ((512,4096,4096),3,False),((512,12288,4096),3,False),((512,14336,4096),3,False),
    ((512,4096,1024),3,True),((31,32768,4096),5,True),((32,32768,4096),5,True)])
def test_split_k_rounding_tail_and_canaries(shape,parts,parallel,offset):
    m,k,n=shape
    block=((k+parts*64-1)//(parts*64))*64
    a=torch.zeros(m,k+offset,dtype=torch.bfloat16)
    b=torch.zeros(n,k+offset,dtype=torch.bfloat16) if n==1024 else torch.zeros(k,n+offset,dtype=torch.bfloat16)
    for index,value in [(block-1,256),(2*block-1,1),(k-1,-256)]:
        a[:,index+offset]=1
        if n==1024: b[:,index+offset]=value
        else: b[index,offset:]=value
    x=a.to('ruda')[:,offset:]
    y=b.to('ruda')[:,offset:].t() if n==1024 else b.to('ruda')[:,offset:]
    storage=torch.full((m*n+2,),-320.,device='ruda',dtype=torch.bfloat16)
    target=storage[1:-1].view(m,n)
    r._C.execute(7,x,y,target,0.)
    values=storage.cpu()
    assert values[0]==values[-1]==-320.
    torch.testing.assert_close(values[1:-1],torch.full_like(values[1:-1],1 if parallel else 0),rtol=0,atol=0)


def test_fp32_output_keeps_unrounded_accumulation():
    a=torch.zeros(512,4096,dtype=torch.bfloat16)
    b=torch.zeros(1024,4096,dtype=torch.bfloat16)
    for index,value in [(0,256),(1,1),(4095,-256)]:
        a[:,index]=1;b[:,index]=value
    target=torch.empty(512,1024,device='ruda',dtype=torch.float32)
    r._C.execute(7,a.to('ruda'),b.to('ruda').t(),target,0.)
    torch.testing.assert_close(target.cpu(),torch.ones_like(target.cpu()),rtol=0,atol=0)
