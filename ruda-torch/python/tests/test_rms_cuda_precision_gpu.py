"""Opt-in CUDA reduction order with an independent CUDA RMSNorm oracle."""
import subprocess
import sys
import pytest
import torch
import ruda_torch as r


@pytest.fixture(scope='module')
def cuda_rms_cases(tmp_path_factory):
    directory=tmp_path_factory.mktemp('cuda_rms_precision')
    cases=[]
    for rows in (16,512):
        for exponent in (-4,0,4):
            torch.manual_seed(293)
            x=(torch.randn(rows,3584)*2.**exponent).to(torch.bfloat16)
            w=torch.randn(3584,dtype=torch.bfloat16)
            cases.append(dict(x=x,w=w,dy=torch.randn_like(x)))
    torch.save(cases,directory/'inputs.pt')
    code='''
import sys,torch
from pathlib import Path
directory=Path(sys.argv[1]);results=[]
for case in torch.load(directory/'inputs.pt',weights_only=True):
    x=case['x'].cuda().requires_grad_();w=case['w'].cuda();value=x.float()
    stats=(value.square().mean(-1,keepdim=True)+1e-6).rsqrt()
    y=w*(value*stats).to(x.dtype);y.backward(case['dy'].cuda())
    results.append(dict(y=y.detach().cpu(),stats=stats.detach().cpu().flatten(),dx=x.grad.cpu()))
torch.save(results,directory/'expected.pt')
'''
    subprocess.run([sys.executable,'-c',code,str(directory)],check=True,capture_output=True,text=True)
    return cases,torch.load(directory/'expected.pt',weights_only=True)


@pytest.mark.parametrize('index',range(6))
def test_cuda_rms_reduction_and_bf16_boundaries(cuda_rms_cases,index,monkeypatch):
    monkeypatch.setenv('RUDA_TORCH_BF16_PRECISION','cuda')
    cases,references=cuda_rms_cases;case=cases[index];reference=references[index]
    x=case['x'].to('ruda');w=case['w'].to('ruda')
    y,stats=r._C.training_hf_rms_forward(x,w,1e-6)
    assert torch.equal(stats.cpu().view(torch.int32),reference['stats'].view(torch.int32))
    assert torch.equal(y.cpu().view(torch.int16),reference['y'].view(torch.int16))
    x.requires_grad_();output=r.rms_norm(x,w,eps=1e-6,round_normalized=True)
    output.backward(case['dy'].to('ruda'))
    actual,expected=x.grad.cpu().double(),reference['dx'].double()
    assert torch.isfinite(actual).all()
    assert (actual-expected).norm()/expected.norm() < 1e-3


def test_cuda_rms_small_row_geometry_preserves_existing_path(monkeypatch):
    torch.manual_seed(307)
    x=torch.randn(8,3584,dtype=torch.bfloat16).to('ruda')
    w=torch.randn(3584,dtype=torch.bfloat16).to('ruda')
    monkeypatch.setenv('RUDA_TORCH_BF16_PRECISION','fp32')
    expected=r._C.training_hf_rms_forward(x,w,1e-6)
    monkeypatch.setenv('RUDA_TORCH_BF16_PRECISION','cuda')
    actual=r._C.training_hf_rms_forward(x,w,1e-6)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a.cpu(),b.cpu(),rtol=0,atol=0)
