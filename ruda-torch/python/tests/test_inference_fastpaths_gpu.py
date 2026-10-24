import os
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope='module')
def backend():
    if os.environ.get('RUDA_REQUIRE_GPU') != '1':
        pytest.skip('requires explicit native GPU execution')
    import ruda_torch
    return ruda_torch


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('k,n', [(33, 7), (257, 129), (3584, 512), (18944, 129)])
def test_gemv_and_widened_addmm(backend, dtype, k, n):
    generator = torch.Generator().manual_seed(640)
    x = torch.randn(1, k, generator=generator).to(dtype)
    storage = torch.randn(n + 2, k, generator=generator).to(dtype)
    weight = storage[1:n+1].t()
    bias = torch.randn(n, generator=generator).to(dtype)
    a, b, c = x.to('ruda'), storage.to('ruda')[1:n+1].t(), bias.to('ruda')
    expected = (x.float().cuda() @ weight.float().cuda()).cpu()
    tolerance = .035 if dtype == torch.bfloat16 else .004
    actual = torch.mm(a, b).cpu()
    torch.testing.assert_close(actual, expected.to(dtype), rtol=tolerance, atol=tolerance)
    actual = torch.addmm(c, a, b, alpha=.75, beta=.5).cpu()
    target = (expected * .75 + bias.float() * .5).to(dtype)
    torch.testing.assert_close(actual, target, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('q_tokens,k_tokens,mask_kind', [(33,33,'none'), (1,257,'none'),
                                                        (5,37,'bool'), (5,37,'additive')])
def test_transformers_native_attention_inference(backend, q_tokens, k_tokens, mask_kind):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from ruda_torch.transformers_attention import register_transformers_attention
    name = 'ruda_inference_test'
    if name not in ALL_ATTENTION_FUNCTIONS:
        register_transformers_attention(name)
    generator = torch.Generator().manual_seed(641)
    dtype = torch.bfloat16
    q = torch.randn(1,4,q_tokens,32,generator=generator).to(dtype)
    k = torch.randn(1,2,k_tokens,32,generator=generator).to(dtype)
    v = torch.randn(1,2,k_tokens,32,generator=generator).to(dtype)
    mask = None
    if mask_kind != 'none':
        mask = torch.arange(k_tokens)[None,:] <= torch.arange(q_tokens)[:,None]+k_tokens-q_tokens
        mask = mask[None,None]
        if mask_kind == 'additive':
            mask = torch.zeros_like(mask,dtype=torch.float32).masked_fill(~mask,float('-inf'))
    causal = mask is None and q_tokens > 1
    expected = torch.nn.functional.scaled_dot_product_attention(q.cuda(),k.cuda(),v.cuda(),
        attn_mask=None if mask is None else mask.cuda(),is_causal=causal,enable_gqa=True).cpu()
    with torch.inference_mode():
        actual,_ = ALL_ATTENTION_FUNCTIONS[name](SimpleNamespace(is_causal=True),
            q.to('ruda'),k.to('ruda'),v.to('ruda'),None if mask is None else mask.to('ruda'))
        actual = actual.transpose(1,2).cpu()
    torch.testing.assert_close(actual,expected,rtol=.025,atol=.025)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_qwen2_fused_rounding_and_strided_rope(backend, dtype):
    generator = torch.Generator().manual_seed(642)
    x = (torch.randn(2,3,3584,generator=generator) * 3).to(dtype)
    weight = torch.randn(3584,generator=generator).to(dtype)
    epsilon = 1e-6
    expected = weight * (x.float() * (x.float().square().mean(-1,keepdim=True)+epsilon).rsqrt()).to(dtype)
    actual = torch.empty_like(x,device='ruda')
    backend._C.execute(109,x.to('ruda'),weight.to('ruda'),actual,epsilon)
    torch.testing.assert_close(actual.cpu(),expected,rtol=2e-5 if dtype==torch.float32 else .008,
                               atol=2e-6 if dtype==torch.float32 else .008)
    q = torch.randn(2,7,4,128,generator=generator).to(dtype).transpose(1,2)
    cos = torch.randn(2,7,128,generator=generator).to(dtype)
    sin = torch.randn(2,7,128,generator=generator).to(dtype)
    rotation = torch.cat((-q[...,64:],q[...,:64]),-1)
    expected = q * cos[:,None] + rotation * sin[:,None]
    source = q.transpose(1,2).contiguous().to('ruda').transpose(1,2)
    actual = torch.empty(q.shape,device='ruda',dtype=dtype)
    backend._C.execute(110,source,torch.stack((cos,sin)).to('ruda'),actual,0.)
    torch.testing.assert_close(actual.cpu(),expected,rtol=0,atol=0)


@pytest.mark.parametrize('batch', [1,2])
def test_qwen2_adapter_inference_and_training_boundary(backend, batch):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from ruda_torch.transformers_qwen2 import enable_transformers_qwen2
    torch.manual_seed(643)
    config = Qwen2Config(vocab_size=257,hidden_size=128,intermediate_size=384,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
        max_position_embeddings=512,attn_implementation='eager')
    model = Qwen2ForCausalLM(config).to(dtype=torch.bfloat16,device='ruda').eval()
    ids = torch.tensor([[2,5,7,3]]*batch,device='ruda')
    with torch.inference_mode():
        expected = model(ids,use_cache=True)
        expected_next = model(ids[:,-1:],past_key_values=expected.past_key_values,use_cache=True).logits.cpu()
    original = {name:value.detach().cpu() for name,value in model.state_dict().items()}
    enable_transformers_qwen2(model)
    enable_transformers_qwen2(model)
    with torch.inference_mode():
        actual = model(ids,use_cache=True)
        actual_next = model(ids[:,-1:],past_key_values=actual.past_key_values,use_cache=True).logits.cpu()
    torch.testing.assert_close(actual.logits.cpu(),expected.logits.cpu(),rtol=.02,atol=.02)
    torch.testing.assert_close(actual_next,expected_next,rtol=.02,atol=.02)
    assert original.keys() == model.state_dict().keys()
    for name,value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(),original[name],rtol=0,atol=0)
    model.train()
    norm = model.model.layers[0].input_layernorm
    x = torch.randn(1,2,128,device='ruda',requires_grad=True)
    torch.testing.assert_close(norm(x),norm._ruda_qwen2_original_forward(x),rtol=0,atol=0)
