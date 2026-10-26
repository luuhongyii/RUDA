import os
import io
import subprocess
import sys

import pytest
import torch


@pytest.fixture(scope='module')
def backend():
    if os.environ.get('RUDA_REQUIRE_GPU')!='1':pytest.skip('requires explicit native GPU execution')
    import ruda_torch
    return ruda_torch


def _attention_inputs(dtype,mask_kind):
    generator=torch.Generator().manual_seed(772)
    shape=(2,4,65,32);kv_shape=(2,2,83 if mask_kind!='causal' else 65,32)
    values=[torch.randn(s,generator=generator).to(dtype) for s in (shape,kv_shape,kv_shape)]
    mask=None
    if mask_kind in ('bool','additive'):
        allowed=torch.rand((1,1,65,kv_shape[2]),generator=generator)>0.2;allowed[...,3,:]=False
        mask=allowed if mask_kind=='bool' else torch.where(allowed,0.,float('-inf'))
    return values,mask,torch.randn(shape,generator=generator).to(dtype)


@pytest.fixture(scope='module')
def cuda_gradients(backend):
    requests=[]
    for kind in ['none','causal','bool','additive']:
        for dtype in [torch.float16,torch.bfloat16]:requests.append((*_attention_inputs(dtype,kind),kind=='causal'))
    payload=io.BytesIO();torch.save(requests,payload)
    code='''import io,sys,torch
requests=torch.load(io.BytesIO(sys.stdin.buffer.read()),weights_only=False)
results=[]
for values,mask,grad,causal in requests:
    tensors=[v.cuda().requires_grad_() for v in values]
    out=torch.nn.functional.scaled_dot_product_attention(*tensors,attn_mask=None if mask is None else mask.cuda(),is_causal=causal,scale=.17,enable_gqa=True)
    out.backward(grad.cuda())
    results.append((out.detach().cpu(),[t.grad.cpu() for t in tensors]))
buffer=io.BytesIO();torch.save(results,buffer);sys.stdout.buffer.write(buffer.getvalue())
'''
    process=subprocess.run([sys.executable,'-c',code],input=payload.getvalue(),capture_output=True,timeout=120)
    assert process.returncode==0,process.stderr.decode(errors='replace')
    return torch.load(io.BytesIO(process.stdout),weights_only=False)


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('shape',[(1,65,129),(7,64,80),(67,129,97),(256,512,1024),(2,65,64,96)])
def test_generic_eager_and_graph_matmul(backend,dtype,shape):
    generator=torch.Generator().manual_seed(771)
    m,k,n=shape[-3:];batch=shape[:-3]
    left=torch.randn((*batch,m,k),generator=generator,dtype=torch.float32).to(dtype)
    right=torch.randn((*batch,k,n),generator=generator,dtype=torch.float32).to(dtype)
    expected=left.cuda()@right.cuda()
    a,b=left.to('ruda'),right.to('ruda')
    tolerance=0.04 if dtype==torch.bfloat16 else 0.006
    with torch.inference_mode():
        torch.testing.assert_close((a@b).cpu(),expected.cpu(),rtol=tolerance,atol=tolerance)
        nodes=[backend.GraphOp('bmm' if batch else 'mm','out','a','b')]
        with backend.StaticGraph({'a':a,'b':b},nodes) as graph:
            torch.testing.assert_close(graph.replay()['out'].cpu(),expected.cpu(),rtol=tolerance,atol=tolerance)
            b.copy_(right.mul(0.5).to('ruda'))
            torch.testing.assert_close(graph.replay()['out'].cpu(),(left.cuda()@right.mul(0.5).cuda()).cpu(),rtol=tolerance,atol=tolerance)


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('mask_kind',['none','causal','bool','additive'])
def test_generic_sdpa_prefill_and_gradients(backend,cuda_gradients,dtype,mask_kind):
    values,mask,grad=_attention_inputs(dtype,mask_kind)
    native=[t.to('ruda').requires_grad_() for t in values]
    scale=.17
    index=['none','causal','bool','additive'].index(mask_kind)*2+[torch.float16,torch.bfloat16].index(dtype)
    expected,gradients=cuda_gradients[index]
    output=backend.scaled_dot_product_attention(*native,attn_mask=None if mask is None else mask.to('ruda'),
        is_causal=mask_kind=='causal',scale=scale,enable_gqa=True)
    tolerance=.04 if dtype==torch.bfloat16 else .008
    torch.testing.assert_close(output.cpu(),expected,rtol=tolerance,atol=tolerance)
    output.backward(grad.to('ruda'))
    for actual,reference in zip(native,gradients,strict=True):
        torch.testing.assert_close(actual.grad.cpu(),reference,rtol=tolerance,atol=tolerance)


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_interleaved_bmm_and_empty_attention(backend,dtype):
    a=torch.randn(4,4,32).to(dtype);b=torch.randn(4,4,32).to(dtype)
    left=a.to('ruda').permute(1,0,2);right=b.to('ruda').permute(1,2,0)
    expected=(a.permute(1,0,2).float()@b.permute(1,2,0).float()).to(dtype)
    torch.testing.assert_close((left@right).cpu(),expected,rtol=.03,atol=.03)
    q=torch.empty((0,4,65,32),dtype=dtype,device='ruda')
    kv=torch.empty((0,2,65,32),dtype=dtype,device='ruda')
    assert backend.scaled_dot_product_attention(q,kv,kv,enable_gqa=True).shape==q.shape


def test_compiled_public_view_keeps_alias(backend):
    def function(x):
        y=x+x
        return y,y.view(2,6)
    compiled=backend.compile(function,native='required',fullgraph=True)
    try:
        with torch.inference_mode():
            x=torch.randn(3,4).to('ruda');a,b=compiled(x)
            assert a.untyped_storage().data_ptr()==b.untyped_storage().data_ptr()
            torch.testing.assert_close(b.cpu(),(x.cpu()+x.cpu()).view(2,6))
    finally:compiled.close()


def test_generic_compiled_normalization_and_parameter_mutation(backend):
    class DenseBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.randn(96,96)*.02)
            self.gain=torch.nn.Parameter(torch.ones(96))
        def forward(self,x):
            h=x@self.weight
            f=h.float()
            normalized=(f*torch.rsqrt(f.square().mean(-1,keepdim=True)+1e-5)).to(h.dtype)*self.gain
            return torch.nn.functional.silu(normalized)*h
    torch.manual_seed(773)
    model=DenseBlock().eval().to('ruda')
    compiled=backend.compile(model,native='required',fullgraph=True)
    x=torch.randn(8,96).to('ruda')
    try:
        with torch.inference_mode():
            first=compiled(x);snapshot=first.cpu().clone()
            torch.testing.assert_close(first.cpu(),model(x).cpu(),rtol=4e-5,atol=4e-5)
            model.weight.add_(.01)
            second=compiled(x)
            torch.testing.assert_close(second.cpu(),model(x).cpu(),rtol=4e-5,atol=4e-5)
            model.weight.data.mul_(.5)
            torch.testing.assert_close(compiled(x).cpu(),model(x).cpu(),rtol=4e-5,atol=4e-5)
            torch.testing.assert_close(first.cpu(),snapshot)
            assert compiled.info['native_replays']>=3
    finally:compiled.close()


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_compiled_generic_attention_graph(backend,dtype):
    generator=torch.Generator().manual_seed(774)
    values=[torch.randn(s,generator=generator).to(dtype) for s in [(1,4,65,64),(1,2,65,64),(1,2,65,64)]]
    def attention(q,k,v):return torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True)
    compiled=backend.compile(attention,native='required',fullgraph=True)
    try:
        with torch.inference_mode():
            expected=attention(*(t.cuda() for t in values)).cpu()
            actual=compiled(*(t.to('ruda') for t in values))
            torch.testing.assert_close(actual.cpu(),expected,rtol=.04,atol=.04)
            assert compiled.info['native_replays']>0
    finally:compiled.close()
