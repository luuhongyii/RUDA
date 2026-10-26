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
        for batch,heads,kv in ((1,2,2),(1,14,2),(2,4,2),(1,28,4)):
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


@pytest.mark.parametrize('index',range(8))
def test_flash_precision_forward_backward(flash_cases,index):
    cases,references=flash_cases;case=cases[index];reference=references[index]
    values=[case[name].to('ruda').requires_grad_() for name in ('q','k','v')]
    y=r.scaled_dot_product_attention(*values,is_causal=case['causal'],
        enable_gqa=values[0].shape[1]!=values[1].shape[1],precision='flash')
    y.backward(case['dy'].to('ruda'))
    actual=[y.cpu(),*(value.grad.cpu() for value in values)]
    expected=[reference['output'],*reference['gradients']]
    if case['q'].shape[1]==28:
        assert torch.equal(actual[0].view(torch.int16),expected[0].view(torch.int16))
    for i,(a,b) in enumerate(zip(actual,expected)):
        a,b=a.double(),b.double()
        assert torch.isfinite(a).all()
        assert (a-b).norm()/b.norm() < (.003 if i==0 else .008)


def test_flash_precision_rejects_unsupported_calls():
    q=torch.randn(1,2,32,128,dtype=torch.bfloat16).to('ruda')
    with pytest.raises(ValueError,match='flash precision requires'):
        r.scaled_dot_product_attention(q,q,q,precision='flash')


@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('offset', [1, 3, 7, 8])
def test_flash_precision_storage_offsets(causal, offset):
    torch.manual_seed(261026)
    values = []
    for heads in (4, 2, 2, 4):
        count = heads * 512 * 128
        flat = torch.randn(count + offset, dtype=torch.bfloat16).to('ruda')
        value = flat[offset:].view(1, heads, 512, 128).detach()
        assert value.storage_offset() == offset
        values.append(value)
    qkv = [x.requires_grad_() for x in values[:3]]
    aligned = [x.detach().clone().requires_grad_() for x in qkv]
    actual = r.scaled_dot_product_attention(*qkv, is_causal=causal, enable_gqa=True, precision='flash')
    expected = r.scaled_dot_product_attention(*aligned, is_causal=causal, enable_gqa=True, precision='flash')
    actual.backward(values[3])
    expected.backward(values[3].clone())
    for a, b in zip([actual, *[x.grad for x in qkv]], [expected, *[x.grad for x in aligned]]):
        assert torch.equal(a.cpu().view(torch.int16), b.cpu().view(torch.int16))


@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('batch,heads,kv_heads', [(1, 7, 1), (2, 6, 2)])
def test_flash_grouped_gradients_preserve_per_head_rounding(causal, batch, heads, kv_heads):
    torch.manual_seed(261028)
    host = [torch.randn(batch, h, 512, 128, dtype=torch.bfloat16) for h in (heads, kv_heads, kv_heads)]
    grad = torch.randn_like(host[0])
    values = [x.to('ruda').requires_grad_() for x in host]
    y = r.scaled_dot_product_attention(*values, is_causal=causal, scale=.2, enable_gqa=True, precision='flash')
    y.backward(grad.to('ruda'))
    group = heads // kv_heads
    expected_k, expected_v = [torch.zeros_like(x, dtype=torch.float32) for x in host[1:]]
    expected_q = torch.empty_like(host[0])
    for head in range(heads):
        kh = head // group
        single = [host[0][:, head:head+1].contiguous().to('ruda').requires_grad_(),
                  *[x[:, kh:kh+1].contiguous().to('ruda').requires_grad_() for x in host[1:]]]
        result = r.scaled_dot_product_attention(*single, is_causal=causal, scale=.2, precision='flash')
        result.backward(grad[:, head:head+1].contiguous().to('ruda'))
        expected_q[:, head:head+1] = single[0].grad.cpu()
        expected_k[:, kh:kh+1] += single[1].grad.cpu().float()
        expected_v[:, kh:kh+1] += single[2].grad.cpu().float()
    for actual, expected in zip(values, [expected_q, expected_k.bfloat16(), expected_v.bfloat16()]):
        torch.testing.assert_close(actual.grad.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize('causal', [False, True])
def test_flash_backward_explicit_scale_and_accumulation(tmp_path, causal):
    torch.manual_seed(261027)
    case = {name: torch.randn(1, heads, 512, 128, dtype=torch.bfloat16)
            for name, heads in [('q', 4), ('k', 2), ('v', 2), ('dy', 4)]}
    torch.save(case, tmp_path / 'case.pt')
    code = '''
import sys,torch
from pathlib import Path
from torch.nn import functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel
p=Path(sys.argv[1]);case=torch.load(p/'case.pt',weights_only=True)
values=[case[n].cuda().requires_grad_() for n in ('q','k','v')]
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    y=F.scaled_dot_product_attention(*values,is_causal=sys.argv[2]=='True',scale=.2,enable_gqa=True)
    y.backward(case['dy'].cuda(),retain_graph=True)
    y.backward(case['dy'].cuda())
torch.save([v.grad.cpu() for v in values],p/'expected.pt')
'''
    subprocess.run([sys.executable, '-c', code, str(tmp_path), str(causal)],
                   check=True, capture_output=True, text=True)
    values = [case[n].to('ruda').requires_grad_() for n in ('q', 'k', 'v')]
    y = r.scaled_dot_product_attention(*values, is_causal=causal, scale=.2,
                                      enable_gqa=True, precision='flash')
    dy = case['dy'].to('ruda')
    y.backward(dy, retain_graph=True)
    y.backward(dy)
    expected = torch.load(tmp_path / 'expected.pt', weights_only=True)
    for value, ref in zip(values, expected):
        a, b = value.grad.cpu().double(), ref.double()
        assert torch.isfinite(a).all()
        assert float((a - b).norm() / b.norm()) < .008


@pytest.mark.parametrize('scale',[0.0,-.125])
def test_transformers_keeps_fp32_path_for_nonpositive_scale(scale):
    from types import SimpleNamespace
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from ruda_torch.transformers_attention import register_transformers_attention
    name='ruda_nonpositive_scale_'+('zero' if scale==0 else 'negative')
    register_transformers_attention(name,implementation='native')
    torch.manual_seed(241)
    q=torch.randn(1,2,512,128,dtype=torch.bfloat16).to('ruda')
    expected=r.scaled_dot_product_attention(q,q,q,scale=scale)
    actual,_=ALL_ATTENTION_FUNCTIONS[name](SimpleNamespace(is_causal=False),q,q,q,None,scaling=scale)
    torch.testing.assert_close(actual.cpu(),expected.transpose(1,2).contiguous().cpu(),rtol=0,atol=0)
