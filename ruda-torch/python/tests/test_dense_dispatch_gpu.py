import pytest
import torch
import ruda_torch as r
from ruda_torch import _dense_ops as ops
from test_fused_dispatch_gpu import same


def test_dense_dispatch_has_native_entrypoints():
    for direction in ('forward','backward'):
        assert torch._C._dispatch_has_kernel_for_dispatch_key('ruda::dense_attention_'+direction,'PrivateUse1')


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('policy',['plain','causal','mask','dropout','all_dropped','empty'])
@torch.no_grad()
def test_dense_native_dispatch_matches_original_body(dtype,policy):
    torch.manual_seed(261101)
    q=torch.randn(1,4,0 if policy=='empty' else 17,32,dtype=dtype).to('ruda')
    k,v=[torch.randn(1,2,17,32,dtype=dtype).to('ruda') for _ in range(2)]
    mask=torch.randn(1,1,1,17).to('ruda') if policy=='mask' else torch.zeros(1,device='ruda')
    shape=list(q.shape[:-1])+[k.shape[2]]
    kind=2 if policy=='mask' else 0
    p=.25 if policy=='dropout' else 1. if policy=='all_dropped' else 0.
    args=(q,k,v,mask,shape,kind,policy=='causal',.125,p)
    r.manual_seed(271)
    actual=ops.forward(*args)
    r.manual_seed(271)
    expected=ops.forward._init_fn(*args)
    same(actual,expected)
    grad=torch.randn_like(actual[0])
    back=(q,k,v,mask,grad,actual[1],actual[2],shape,kind,policy=='causal',.125,p,kind==2)
    same(ops.backward(*back),ops.backward._init_fn(*back))


@pytest.mark.parametrize('causal',[False,True])
@torch.no_grad()
def test_dense_native_flash_dispatch_matches_original_body(causal):
    torch.manual_seed(261102)
    q=torch.randn(1,4,512,128,dtype=torch.bfloat16).to('ruda')
    k,v=[torch.randn(1,2,512,128,dtype=torch.bfloat16).to('ruda') for _ in range(2)]
    mask=torch.zeros(1,device='ruda');shape=[1,4,512,512]
    args=(q,k,v,mask,shape,0,causal,128**-.5,0.,True)
    actual=ops.forward(*args)
    same(actual,ops.forward._init_fn(*args))
    grad=torch.randn_like(q)
    back=(q,k,v,mask,grad,actual[1],actual[2],shape,0,causal,128**-.5,0.,False)
    same(ops.backward(*back),ops.backward._init_fn(*back))


def test_dense_native_dispatch_fake_aot_and_stream():
    stream=r.Stream()
    with r.stream(stream):
        q=torch.randn(1,2,17,32,device='ruda',requires_grad=True)
        k=torch.randn(1,1,17,32,device='ruda',requires_grad=True)
        v=torch.randn_like(k,requires_grad=True)
        mask=torch.zeros(1,device='ruda')
        args=(q,k,v,mask,[1,2,17,17],0,True,.125,0.,False)
        y=ops.forward(*args)[0]
        y.sum().backward()
    stream.synchronize()
    assert all(x.grad is not None and torch.isfinite(x.grad.cpu()).all() for x in (q,k,v))
    torch.library.opcheck(ops.forward,args,test_utils=('test_schema','test_faketensor','test_aot_dispatch_dynamic'))


@pytest.mark.parametrize('invalid',[torch.zeros(3,dtype=torch.int64),torch.tensor([-1,0,0,0]),torch.tensor([2**32,0,0,0])])
def test_dense_native_dispatch_preserves_rng_validation(invalid):
    q=torch.ones(1,1,1,1,device='ruda');mask=torch.zeros(1,device='ruda')
    stats=ops.forward(q,q,q,mask,[1,1,1,1],0,False,1.,0.)[1]
    with pytest.raises(ValueError):
        ops.backward(q,q,q,mask,q,stats,invalid,[1,1,1,1],0,False,1.,0.,False)
