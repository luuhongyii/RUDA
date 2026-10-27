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


@pytest.mark.parametrize('family', ['Granite', 'Qwen3Moe', 'Mixtral'])
def test_additional_hf_architectures_preserve_model_and_cache(family):
    transformers = pytest.importorskip('transformers')
    config = getattr(transformers, family + 'Config')(
        vocab_size=37, hidden_size=32, intermediate_size=64, moe_intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        num_experts=2, num_local_experts=2, num_experts_per_tok=1,
        max_position_embeddings=32, attention_dropout=0., hidden_act='silu')
    config._attn_implementation = 'eager'
    original = getattr(transformers, family + 'ForCausalLM')(config).eval()
    model = copy.deepcopy(original)
    parameters = {name: id(value) for name, value in model.named_parameters()}
    paths = tuple(model.state_dict())
    tf.enable_transformers_fused(model)
    assert parameters == {name: id(value) for name, value in model.named_parameters()}
    assert paths == tuple(model.state_dict())
    for module in model.modules():
        if type(module).__name__ in tf._NORMS | tf._MLPS.keys():
            assert hasattr(module, '_ruda_fused_original_forward')
    tokens = torch.randint(0, 37, (1, 7))
    actual = model(tokens, labels=tokens, use_cache=False).loss
    expected = original(tokens, labels=tokens, use_cache=False).loss
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward(); expected.backward()
    for a, b in zip(model.parameters(), original.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    with torch.no_grad():
        first = model(tokens, use_cache=True)
        reference = original(tokens, use_cache=True)
        next_token = tokens[:, :1]
        actual = model(next_token, past_key_values=first.past_key_values).logits
        expected = original(next_token, past_key_values=reference.past_key_values).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_activation_hooks_and_modified_forward_are_not_bypassed():
    activation = torch.nn.SiLU()
    assert tf._plain_silu(activation)
    hook = activation.register_forward_hook(lambda module, args, output: output + 1)
    assert not tf._plain_silu(activation)
    gate, up = torch.randn(2, 7), torch.randn(2, 7)
    torch.testing.assert_close(tf._activation_mul(activation, gate, up), activation(gate) * up)
    hook.remove()
    assert tf._plain_silu(activation)
    activation.forward = lambda x: x
    assert not tf._plain_silu(activation)
