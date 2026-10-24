import copy
import os

import pytest
import torch


@pytest.fixture(scope='module')
def backend():
    if os.environ.get('RUDA_REQUIRE_GPU') != '1':
        pytest.skip('requires explicit native GPU execution')
    import ruda_torch
    return ruda_torch


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('length,dim,causal',[(33,32,True),(64,128,True),(17,32,False)])
def test_tensor_core_prefill(backend,dtype,length,dim,causal):
    from ruda_torch.transformers_qwen2 import _initialize_native
    generator=torch.Generator().manual_seed(820)
    q=torch.randn(1,4,length,dim,generator=generator).to(dtype)
    k=torch.randn(1,2,length,dim,generator=generator).to(dtype).repeat_interleave(2,dim=1)
    v=torch.randn(1,2,length,dim,generator=generator).to(dtype).repeat_interleave(2,dim=1)
    expected=torch.nn.functional.scaled_dot_product_attention(q.cuda(),k.cuda(),v.cuda(),is_causal=causal).cpu()
    out=torch.empty_like(q,device='ruda')
    with torch.inference_mode():
        _initialize_native().qwen_flash([q.to('ruda'),k.to('ruda'),v.to('ruda'),out],causal,dim**-.5)
    torch.testing.assert_close(out.cpu(),expected,rtol=.03,atol=.03)


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('length,keys,dim,causal',[(65,65,32,True),(129,129,128,True),(64,91,64,False),(33,33,32,True)])
def test_prefill_gqa_block_softmax(backend,dtype,length,keys,dim,causal):
    from ruda_torch.transformers_qwen2 import _initialize_native
    generator=torch.Generator().manual_seed(824)
    q=torch.randn(1,8,length,dim,generator=generator).to(dtype)
    k=torch.randn(1,2,keys,dim,generator=generator).to(dtype)
    v=torch.randn(1,2,keys,dim,generator=generator).to(dtype)
    expected=torch.nn.functional.scaled_dot_product_attention(q.cuda(),k.cuda(),v.cuda(),is_causal=causal,enable_gqa=True).cpu()
    out=torch.empty_like(q,device='ruda')
    with torch.inference_mode():
        _initialize_native().qwen_flash([q.to('ruda'),k.to('ruda'),v.to('ruda'),out],causal,dim**-.5)
    torch.testing.assert_close(out.cpu(),expected,rtol=.03,atol=.03)


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('m,k,n',[(64,128,256),(129,257,259),(128,3584,512),(256,1024,1024)])
def test_pipelined_projection(backend,dtype,m,k,n):
    torch.manual_seed(821)
    x=torch.randn(m,k,dtype=dtype)
    weight=torch.randn(n,k,dtype=dtype)
    expected=torch.nn.functional.linear(x.float().cuda(),weight.float().cuda()).cpu().to(dtype)
    actual=torch.nn.functional.linear(x.to('ruda'),weight.to('ruda')).cpu()
    torch.testing.assert_close(actual,expected,rtol=.04,atol=.04)
    bias=torch.randn(n,dtype=dtype)
    expected=(torch.nn.functional.linear(x.float().cuda(),weight.float().cuda()).cpu()+bias.float()).to(dtype)
    actual=torch.nn.functional.linear(x.to('ruda'),weight.to('ruda'),bias.to('ruda')).cpu()
    torch.testing.assert_close(actual,expected,rtol=.04,atol=.04)


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_qwen_graph_cache_replay_crop_growth_and_weights(backend,dtype):
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from ruda_torch.transformers_attention import register_transformers_attention
    from ruda_torch.transformers_qwen2 import enable_transformers_qwen2
    from ruda_torch._qwen_cache import Qwen2InferenceCache
    name='ruda_qwen_graph_test'
    if name not in ALL_ATTENTION_FUNCTIONS:
        register_transformers_attention(name)
    torch.manual_seed(822)
    config=Qwen2Config(vocab_size=257,hidden_size=128,intermediate_size=384,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,attn_implementation=name)
    reference=Qwen2ForCausalLM(config).eval().requires_grad_(False).to(dtype=dtype,device='ruda')
    model=copy.deepcopy(reference)
    enable_transformers_qwen2(model,cache_capacity=17)
    ids=torch.tensor([[2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53]],device='ruda')
    with torch.inference_mode():
        expected=reference(ids,use_cache=True,logits_to_keep=1)
        actual=model(ids,use_cache=True,logits_to_keep=1)
        assert isinstance(actual.past_key_values,Qwen2InferenceCache)
        torch.testing.assert_close(actual.logits.cpu(),expected.logits.cpu(),rtol=.04,atol=.04)
        retained=actual.logits
        saved=retained.cpu()
        cache=actual.past_key_values
        for token in [59,61,67,71]:
            x=ids.new_full((1,1),token)
            expected=reference(x,past_key_values=expected.past_key_values,use_cache=True,logits_to_keep=1)
            actual=model(x,past_key_values=cache,use_cache=True,logits_to_keep=1)
            torch.testing.assert_close(actual.logits.cpu(),expected.logits.cpu(),rtol=.04,atol=.04)
        assert cache.graph_replays==4 and cache.layers[0].capacity>=20
        assert cache._ruda_graph.kernel_count>20
        cache.crop(-2);expected.past_key_values.crop(-2)
        x=ids.new_full((1,1),73)
        expected=reference(x,past_key_values=expected.past_key_values,use_cache=True)
        actual=model(x,past_key_values=cache,use_cache=True)
        torch.testing.assert_close(actual.logits.cpu(),expected.logits.cpu(),rtol=.04,atol=.04)
        for target in [reference,model]:
            target.model.layers[0].mlp.gate_proj.weight.add_(.001)
        x=ids.new_full((1,1),79)
        expected=reference(x,past_key_values=expected.past_key_values,use_cache=True)
        actual=model(x,past_key_values=cache,use_cache=True)
        torch.testing.assert_close(actual.logits.cpu(),expected.logits.cpu(),rtol=.04,atol=.04)
        torch.testing.assert_close(retained.cpu(),saved,rtol=0,atol=0)
        cache.close_graph()


def test_cache_batch_selection_and_reset(backend):
    from ruda_torch._qwen_cache import Qwen2InferenceCache
    cache=Qwen2InferenceCache(1,4)
    values=torch.arange(16,dtype=torch.float32).reshape(1,2,2,4).to('ruda')
    cache.update(values,values,0)
    cache.batch_repeat_interleave(2)
    expected=values.cpu().repeat_interleave(2,0)
    extra=values.repeat_interleave(2,0)
    keys,_=cache.update(extra,extra,0)
    torch.testing.assert_close(keys.cpu(),torch.cat((expected,expected),-2),rtol=0,atol=0)
    cache.batch_select_indices(torch.tensor([1],device='ruda'))
    cache.crop(-1)
    keys,_=cache.update(values,values,0)
    assert keys.shape==(1,2,5,4) and cache.layers[0].capacity>=5
    cache.reset()
    assert cache.get_seq_length()==0


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('hidden,heads,kv_heads,length',[(128,4,2,17),(3584,28,4,1025)])
def test_parallel_decode_rms_and_split_attention(backend,dtype,hidden,heads,kv_heads,length):
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from ruda_torch.transformers_attention import register_transformers_attention
    from ruda_torch.transformers_qwen2 import enable_transformers_qwen2
    torch.manual_seed(823)
    config=Qwen2Config(vocab_size=257,hidden_size=hidden,intermediate_size=384,num_hidden_layers=1,
        num_attention_heads=heads,num_key_value_heads=kv_heads,max_position_embeddings=2048,
        attn_implementation='sdpa')
    reference=Qwen2ForCausalLM(config).eval().requires_grad_(False).to(dtype=dtype)
    model=copy.deepcopy(reference)
    reference=reference.cuda()
    name='ruda_parallel_decode_test'
    if name not in ALL_ATTENTION_FUNCTIONS:register_transformers_attention(name)
    model.config._attn_implementation=name
    model=model.to('ruda');enable_transformers_qwen2(model,cache_capacity=length+4)
    ids=(torch.arange(length)%251+2)[None]
    with torch.inference_mode():
        expected=reference(ids.cuda(),use_cache=True,logits_to_keep=1)
        actual=model(ids.to('ruda'),use_cache=True,logits_to_keep=1)
        for token in [67,71]:
            step=torch.tensor([[token]])
            expected=reference(step.cuda(),past_key_values=expected.past_key_values,use_cache=True,logits_to_keep=1)
            actual=model(step.to('ruda'),past_key_values=actual.past_key_values,use_cache=True,logits_to_keep=1)
            torch.testing.assert_close(actual.logits.cpu(),expected.logits.cpu(),rtol=.04,atol=.04)
        assert actual.past_key_values.graph_replays==2
        actual.past_key_values.close_graph()
