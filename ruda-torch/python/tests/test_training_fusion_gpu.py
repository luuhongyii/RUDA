"""Requires the real rebuilt RUDA backend; no simulator or successful skips."""
import pytest
import torch
from torch.nn import functional as F
import ruda_torch as r


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('width', [1, 47, 1024, 2053])
def test_cross_entropy_full_vocabulary_forward_backward(dtype, width):
    torch.manual_seed(29)
    host = torch.randn(5, width, dtype=dtype)
    target = torch.tensor([0, width - 1, -100, width // 2, 0])
    if width > 1024:
        host[0, 1:1025] = float('-inf')
    reference = host.clone().requires_grad_()
    actual = host.to('ruda').requires_grad_()
    loss = r.cross_entropy(actual, target.to('ruda'), reduction='none')
    expected = F.cross_entropy(reference.float(), target, reduction='none')
    torch.testing.assert_close(loss.cpu(), expected, rtol=3e-5, atol=3e-5)
    gradient = torch.tensor([1., -.5, 2., .25, 1.5])
    loss.backward(gradient.to('ruda')); expected.backward(gradient)
    tolerance = 3e-3 if dtype != torch.float32 else 3e-5
    torch.testing.assert_close(actual.grad.cpu(), reference.grad, rtol=tolerance, atol=tolerance)


def test_cross_entropy_large_common_offset_and_ignored_negative_infinity():
    host = torch.full((2, 2053), 1e8)
    host[1] = float('-inf')
    target = torch.tensor([7, -100])
    x = host.to('ruda').requires_grad_()
    result = r.cross_entropy(x, target.to('ruda'), reduction='none')
    expected = F.cross_entropy(host, target, reduction='none')
    torch.testing.assert_close(result.cpu(), expected, rtol=2e-6, atol=2e-6)
    result.sum().backward()
    assert torch.isfinite(x.grad.cpu()).all()
    assert torch.count_nonzero(x.grad.cpu()[1]) == 0


def test_cross_entropy_invalid_target_is_rejected():
    x = torch.ones(2, 17).to('ruda')
    with pytest.raises(RuntimeError, match='out of bounds'):
        r.cross_entropy(x, torch.tensor([17, 0]).to('ruda'))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('need_x,need_w', [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize('width', [129, 1024, 3584, 8192, 8193])
def test_hf_rms_norm_storage_boundaries_and_gradients(dtype, need_x, need_w, width):
    torch.manual_seed(71)
    host = torch.randn(7, width, dtype=dtype)
    weight = torch.randn(width, dtype=dtype)
    reference = host.clone().requires_grad_(need_x)
    ref_weight = weight.clone().requires_grad_(need_w)
    x = host.to('ruda').requires_grad_(need_x)
    w = weight.to('ruda').requires_grad_(need_w)
    result = r.rms_norm(x, w, eps=1e-6, round_normalized=True)
    value = reference.float()
    expected = ref_weight * (value * (value.square().mean(-1, keepdim=True) + 1e-6).rsqrt()).to(dtype)
    tolerance = 4e-2 if dtype == torch.bfloat16 else 5e-3 if dtype == torch.float16 else 4e-5
    torch.testing.assert_close(result.cpu(), expected, rtol=tolerance, atol=tolerance)
    gradient = torch.randn_like(host)
    result.backward(gradient.to('ruda')); expected.backward(gradient)
    if need_x:
        torch.testing.assert_close(x.grad.cpu(), reference.grad, rtol=tolerance, atol=tolerance)
    if need_w:
        torch.testing.assert_close(w.grad.cpu(), ref_weight.grad, rtol=tolerance, atol=tolerance)


def test_fused_operators_survive_real_aot_forward_backward():
    def function(gate, up, weight):
        return r.rms_norm(r.silu_mul(gate, up), weight, eps=1e-6).sum()
    from torch._dynamo.backends.common import aot_autograd
    graphs = []
    def capture(graph, inputs):
        graphs.append(graph)
        return graph.forward
    compiled = torch.compile(function, backend=aot_autograd(fw_compiler=capture, bw_compiler=capture),
                             fullgraph=True)
    values = [torch.randn(4, 129).to('ruda').requires_grad_(),
              torch.randn(4, 129).to('ruda').requires_grad_(),
              torch.ones(129).to('ruda').requires_grad_()]
    compiled(*values).backward()
    targets = {str(node.target) for graph in graphs for node in graph.graph.nodes}
    assert any('training_rms_forward' in name for name in targets)
    assert any('training_rms_backward' in name for name in targets)
    assert any('training_silu_forward' in name for name in targets)
    assert any('training_silu_backward' in name for name in targets)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('need_q,need_k', [(True, True), (True, False), (False, True)])
def test_strided_gqa_rotary_forward_backward(dtype, need_q, need_k):
    torch.manual_seed(81)
    q_base = torch.randn(2, 17, 4, 130, dtype=dtype)
    k_base = torch.randn(2, 17, 2, 130, dtype=dtype)
    q_ref = q_base.transpose(1, 2).detach().requires_grad_(need_q)
    k_ref = k_base.transpose(1, 2).detach().requires_grad_(need_k)
    q = q_base.to('ruda').transpose(1, 2).detach().requires_grad_(need_q)
    k = k_base.to('ruda').transpose(1, 2).detach().requires_grad_(need_k)
    angles = torch.randn(2, 17, 130)
    cos, sin = angles.cos().to(dtype), angles.sin().to(dtype)
    def reference(x):
        rotated = torch.cat((-x[..., 65:], x[..., :65]), dim=-1)
        return x * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)
    expected_q, expected_k = reference(q_ref), reference(k_ref)
    actual_q, actual_k = r.apply_rotary_pos_emb(q, k, cos.to('ruda'), sin.to('ruda'))
    tolerance = 2e-5 if dtype == torch.float32 else 2e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(actual_q.cpu(), expected_q, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(actual_k.cpu(), expected_k, rtol=tolerance, atol=tolerance)
    if need_q:
        gradient = torch.randn_like(expected_q)
        actual_q.backward(gradient.to('ruda')); expected_q.backward(gradient)
        torch.testing.assert_close(q.grad.cpu(), q_ref.grad, rtol=tolerance, atol=tolerance)
    if need_k:
        gradient = torch.randn_like(expected_k)
        actual_k.backward(gradient.to('ruda')); expected_k.backward(gradient)
        torch.testing.assert_close(k.grad.cpu(), k_ref.grad, rtol=tolerance, atol=tolerance)
