"""Requires the real rebuilt RUDA backend; no simulator or successful skips."""
import pytest
import torch
from torch.nn import functional as F
import ruda_torch as r


@pytest.mark.parametrize('column_major', [False, True])
@pytest.mark.parametrize('shape', [(512, 3584, 16), (512, 18944, 16), (32, 152064, 16),
                                  (31, 1537, 7), (3, 513, 1), (2, 512, 17), (1, 1025, 32)])
def test_skinny_fp32_matmul_split_k(shape, column_major):
    m, k, n = shape
    torch.manual_seed(179)
    a = torch.randn(m, k)
    # Nonzero storage offsets and padded strides must survive native binding.
    backing = torch.randn(n, k + 3) if column_major else torch.randn(k, n + 3)
    b = backing[:, 1:k + 1].t() if column_major else backing[:, 1:n + 1]
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        expected = a.cuda() @ b.cuda()
        actual = a.to('ruda') @ b.to('ruda')
        host = actual.cpu()
        torch.testing.assert_close(host, expected.cpu(), rtol=1e-4, atol=3e-3)
        relative = (host.double() - expected.cpu().double()).norm() / expected.double().norm().cpu()
        assert relative < 4e-6
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def test_skinny_fp32_matmul_nonfinite_values():
    a = torch.zeros(3, 513)
    b = torch.ones(513, 7)
    a[0, 0] = float('nan')
    b[1, 1] = float('inf')
    b[2, 2] = float('-inf')
    torch.testing.assert_close((a.to('ruda') @ b.to('ruda')).cpu(), a @ b, equal_nan=True)


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
        actual_q.backward(gradient.to('ruda'), retain_graph=need_k); expected_q.backward(gradient)
        torch.testing.assert_close(q.grad.cpu(), q_ref.grad, rtol=tolerance, atol=tolerance)
    if need_k:
        gradient = torch.randn_like(expected_k)
        actual_k.backward(gradient.to('ruda')); expected_k.backward(gradient)
        torch.testing.assert_close(k.grad.cpu(), k_ref.grad, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('scale', [1., 1. / 3., 1.75])
@pytest.mark.parametrize('need_base,need_update', [(True, True), (True, False), (False, True)])
def test_lora_epilogue_strided_forward_backward_and_storage_rounding(dtype, scale, need_base, need_update):
    torch.manual_seed(203)
    base_host = torch.randn(7, 3, dtype=dtype)
    update_host = torch.randn(7, 3, dtype=torch.float32)
    reference = base_host.transpose(0, 1).detach().requires_grad_(need_base)
    ref_update = update_host.transpose(0, 1).detach().requires_grad_(need_update)
    base = base_host.to('ruda').transpose(0, 1).detach().requires_grad_(need_base)
    update = update_host.to('ruda').transpose(0, 1).detach().requires_grad_(need_update)
    expected = reference + (ref_update * scale).to(dtype)
    actual = r.lora_epilogue(base, update, scale)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    gradient = torch.randn(1, 7, dtype=dtype).expand(3, 7)
    actual.backward(gradient.to('ruda')); expected.backward(gradient)
    if need_base:
        torch.testing.assert_close(base.grad.cpu(), reference.grad, rtol=0, atol=0)
    if need_update:
        torch.testing.assert_close(update.grad.cpu(), ref_update.grad, rtol=0, atol=0)


def test_lora_epilogue_keeps_the_intermediate_low_precision_cast():
    base = torch.tensor([-1., -1.], dtype=torch.bfloat16).to('ruda')
    update = torch.tensor([1.001, 1.003], dtype=torch.float32).to('ruda')
    result = r.lora_epilogue(base, update, 1.)
    torch.testing.assert_close(result.cpu(), torch.zeros(2, dtype=torch.bfloat16), rtol=0, atol=0)


def test_cross_entropy_accepts_zero_stride_backward_gradient():
    torch.manual_seed(207)
    host = torch.randn(5, 2053)
    labels = torch.tensor([0, -100, 2052, 3, 17])
    reference = host.clone().requires_grad_()
    actual = host.to('ruda').requires_grad_()
    result = r.cross_entropy(actual, labels.to('ruda'), reduction='none')
    expected = F.cross_entropy(reference, labels, reduction='none')
    gradient = torch.tensor(.25).to('ruda').expand(5)
    assert gradient.stride() == (0,)
    result.backward(gradient); expected.backward(torch.full((5,), .25))
    torch.testing.assert_close(actual.grad.cpu(), reference.grad, rtol=3e-5, atol=3e-5)


def test_ignored_cross_entropy_rows_have_zero_loss_and_gradient():
    host = torch.randn(4, 2053)
    host[1] = float('nan'); host[3] = float('-inf')
    labels = torch.tensor([0, -100, 4, -100]).to('ruda')
    actual = host.to('ruda').requires_grad_()
    losses = r.cross_entropy(actual, labels, reduction='none')
    losses.sum().backward()
    assert torch.isfinite(losses.cpu()).all()
    assert torch.count_nonzero(losses.cpu()[[1, 3]]) == 0
    assert torch.count_nonzero(actual.grad.cpu()[[1, 3]]) == 0


def test_lora_epilogue_survives_aot_forward_and_backward():
    from torch._dynamo.backends.common import aot_autograd
    graphs = []
    def capture(graph, inputs):
        graphs.append(graph)
        return graph.forward
    function = torch.compile(lambda base, update: r.lora_epilogue(base, update, .5).sum(),
        backend=aot_autograd(fw_compiler=capture, bw_compiler=capture), fullgraph=True)
    base = torch.ones(3, 7, dtype=torch.bfloat16).to('ruda').requires_grad_()
    update = torch.ones(3, 7, dtype=torch.float32).to('ruda').requires_grad_()
    function(base, update).backward()
    targets = {str(node.target) for graph in graphs for node in graph.graph.nodes}
    assert any('lora_epilogue_forward' in name for name in targets)
    assert any('lora_epilogue_backward' in name for name in targets)
