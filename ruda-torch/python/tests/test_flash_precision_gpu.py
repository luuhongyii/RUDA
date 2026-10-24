"""Native BF16 Flash boundaries versus a separate-process CUDA Flash oracle."""
import subprocess
import sys
import pytest
import torch
import ruda_torch as r


@pytest.fixture(scope='module')
def flash_cases(tmp_path_factory):
    directory=tmp_path_factory.mktemp('flash_precision')
    cases=[]
    for causal in (False,True):
        for batch,heads,kv in ((1,2,2),(1,14,2),(2,4,2)):
            torch.manual_seed(239)
            q=torch.randn(batch,heads,512,128,dtype=torch.bfloat16)
            k=torch.randn(batch,kv,512,128,dtype=torch.bfloat16)
            v=torch.randn_like(k);dy=torch.randn_like(q)
            cases.append(dict(q=q,k=k,v=v,dy=dy,causal=causal))
    torch.save(cases,directory/'inputs.pt')
    code='''
import sys,torch
from pathlib import Path
from torch.nn import functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel
directory=Path(sys.argv[1]);results=[]
for case in torch.load(directory/'inputs.pt',weights_only=True):
    values=[case[name].cuda().requires_grad_() for name in ('q','k','v')]
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        y=F.scaled_dot_product_attention(*values,is_causal=case['causal'],enable_gqa=values[0].shape[1]!=values[1].shape[1])
        y.backward(case['dy'].cuda())
    results.append(dict(output=y.detach().cpu(),gradients=[v.grad.cpu() for v in values]))
torch.save(results,directory/'expected.pt')
'''
    subprocess.run([sys.executable,'-c',code,str(directory)],check=True,capture_output=True,text=True)
    return cases,torch.load(directory/'expected.pt',weights_only=True)


@pytest.mark.parametrize('index',range(6))
def test_flash_precision_forward_backward(flash_cases,index):
    cases,references=flash_cases;case=cases[index];reference=references[index]
    values=[case[name].to('ruda').requires_grad_() for name in ('q','k','v')]
    y=r.scaled_dot_product_attention(*values,is_causal=case['causal'],
        enable_gqa=values[0].shape[1]!=values[1].shape[1],precision='flash')
    y.backward(case['dy'].to('ruda'))
    actual=[y.cpu(),*(value.grad.cpu() for value in values)]
    expected=[reference['output'],*reference['gradients']]
    for i,(a,b) in enumerate(zip(actual,expected)):
        a,b=a.double(),b.double()
        assert torch.isfinite(a).all()
        assert (a-b).norm()/b.norm() < (.003 if i==0 else .008)


def test_flash_precision_rejects_unsupported_calls():
    q=torch.randn(1,2,32,128,dtype=torch.bfloat16).to('ruda')
    with pytest.raises(ValueError,match='flash precision requires'):
        r.scaled_dot_product_attention(q,q,q,precision='flash')
