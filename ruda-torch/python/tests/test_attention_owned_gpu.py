"""Real-device checks for the native cached-query/output-owned KV backward."""
import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import ruda_torch as r


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('batch,heads,kv_heads', [(1, 2, 2), (1, 14, 2), (2, 4, 2)])
@pytest.mark.parametrize('mask_kind', ['none', 'causal', 'boolean', 'additive'])
def test_owned_attention_matches_fp32_math(dtype, batch, heads, kv_heads, mask_kind):
    torch.manual_seed(211)
    host = [torch.randn(batch, h, 512, 128, dtype=dtype) for h in (heads, kv_heads, kv_heads)]
    actual_inputs = [value.to('ruda').requires_grad_() for value in host]
    reference_inputs = [value.float().requires_grad_() for value in host]
    mask = None
    if mask_kind == 'boolean':
        mask = torch.rand(1, 1, 512, 512) > .2
        mask[..., 0, :] = False
    elif mask_kind == 'additive':
        mask = torch.randn(1, 1, 512, 512) * .1
        mask[..., 0, :] = float('-inf')
        mask[..., :, 7] = float('-inf')
    options = dict(dropout_p=0., is_causal=mask_kind == 'causal', scale=.07)
    actual = r.scaled_dot_product_attention(*actual_inputs,
        attn_mask=None if mask is None else mask.to('ruda'), enable_gqa=heads != kv_heads, **options)
    q, k, v = reference_inputs
    group = heads // kv_heads
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with sdpa_kernel(SDPBackend.MATH):
            expected = F.scaled_dot_product_attention(q, k.repeat_interleave(group, 1),
                v.repeat_interleave(group, 1), attn_mask=mask, **options)
        gradient = torch.randn_like(host[0])
        actual.backward(gradient.to('ruda'))
        expected.backward(gradient.float())
        tolerance = 3e-2 if dtype == torch.bfloat16 else 5e-3 if dtype == torch.float16 else 2e-4
        absolute = 3e-3 if dtype != torch.float32 else 2e-5
        torch.testing.assert_close(actual.cpu().float(), expected.cpu(), rtol=tolerance, atol=absolute)
        for value, reference in zip(actual_inputs, reference_inputs):
            torch.testing.assert_close(value.grad.cpu().float(), reference.grad.cpu(), rtol=tolerance, atol=absolute)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


@pytest.mark.parametrize('poison', ['query', 'key', 'value', 'gradient'])
def test_owned_attention_nonfinite_matches_atomic_path(poison):
    torch.manual_seed(223)
    host = [torch.randn(1, h, 512, 128) for h in (2, 1, 1)]
    gradient = torch.randn_like(host[0])
    if poison == 'gradient':
        gradient[0, 0, 9, 3] = float('inf')
    else:
        host[['query', 'key', 'value'].index(poison)][0, 0, 7, 0] = float('nan')
    bias = torch.zeros(1, 2, 512, 512)
    bias[..., 0, :] = float('-inf')
    results = []
    for mask_gradient in (False, True):
        values = [value.to('ruda').requires_grad_() for value in host]
        mask = bias.to('ruda').requires_grad_(mask_gradient)
        result = r.scaled_dot_product_attention(*values, attn_mask=mask, enable_gqa=True, scale=.07)
        result.backward(gradient.to('ruda'))
        results.append((result.cpu(), [value.grad.cpu() for value in values]))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=5e-4, atol=1e-4, equal_nan=True)
    for actual, expected in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(actual, expected, rtol=5e-4, atol=1e-4, equal_nan=True)
