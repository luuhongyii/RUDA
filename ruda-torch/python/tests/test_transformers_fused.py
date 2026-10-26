"""Real HF module compatibility on CPU; not a GPU speed benchmark."""
import copy
import importlib

import pytest
import torch
from architecture_test_utils import NAME


tf = importlib.import_module(NAME + '.transformers_fused')


def dense_model(family):
    transformers = pytest.importorskip('transformers')
    config = getattr(transformers, family + 'Config')(vocab_size=37, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, max_position_embeddings=32, attention_dropout=0., hidden_act='silu')
    config._attn_implementation = 'eager'
    return getattr(transformers, family + 'ForCausalLM')(config)


@pytest.mark.parametrize('family', ['Qwen2', 'Qwen3', 'Llama', 'Mistral'])
def test_actual_hf_cpu_forward_backward_and_parameter_identity(family):
    torch.manual_seed(97)
    original = dense_model(family).train()
    actual = copy.deepcopy(original)
    parameters = {name: id(value) for name, value in actual.named_parameters()}
    paths = tuple(actual.state_dict())
    assert tf.enable_transformers_fused(actual) is actual
    assert tf.enable_transformers_fused(actual) is actual
    assert parameters == {name: id(value) for name, value in actual.named_parameters()}
    assert paths == tuple(actual.state_dict())
    assert hasattr(actual.model.layers[0].input_layernorm, '_ruda_fused_original_forward')
    assert hasattr(actual.model.layers[0].mlp, '_ruda_fused_original_forward')
    tokens = torch.randint(0, 37, (2, 9))
    reference = original(tokens, labels=tokens, use_cache=False).loss
    result = actual(tokens, labels=tokens, use_cache=False).loss
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    result.backward(); reference.backward()
    for left, right in zip(actual.parameters(), original.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


@pytest.mark.parametrize('family', ['Qwen2', 'Qwen3', 'Llama', 'Mistral'])
def test_custom_forwards_and_activations_are_not_replaced(family):
    model = dense_model(family)
    custom = lambda x: x
    model.model.layers[0].input_layernorm.forward = custom
    model.model.layers[0].mlp.act_fn = torch.nn.ReLU()
    tf.enable_transformers_fused(model)
    assert model.model.layers[0].input_layernorm.forward is custom
    assert not hasattr(model.model.layers[0].mlp, '_ruda_fused_original_forward')


@pytest.mark.parametrize('family', ['Qwen2', 'Qwen3', 'Llama', 'Mistral'])
def test_native_attention_is_explicit_and_preserves_cuda_sdpa_interface(family):
    model = dense_model(family)
    tf.enable_transformers_fused(model, attention='native')
    assert model.config._attn_implementation == 'ruda_fused_training'
    tokens = torch.randint(0, 37, (1, 7))
    result = model(tokens, labels=tokens, use_cache=False)
    assert torch.isfinite(result.loss)
    result.loss.backward()
