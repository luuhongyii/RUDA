"""Requires the real rebuilt RUDA backend; no simulator or successful skips."""
import pytest
import torch
from torch.nn import functional as F
import ruda_torch as r


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_launch_metadata_reuse_keeps_shapes_strides_and_streams(dtype):
    streams = [r.Stream(), r.Stream()]
    host = torch.arange(512, dtype=torch.float32).reshape(16, 32).to(dtype)
    pending = []
    for iteration in range(8):
        for index, stream in enumerate(streams):
            with r.stream(stream):
                base = host.to('ruda')
                source = base.transpose(0, 1) if (iteration + index) % 2 else base[:, ::2]
                expected = host.T if (iteration + index) % 2 else host[:, ::2]
                output = torch.empty(source.shape, device='ruda', dtype=dtype)
                output.copy_(source)
                pending.append((output, expected.clone()))
    for output, expected in pending:
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


def test_launch_metadata_eviction_preserves_inflight_outputs():
    width = 2064
    host = torch.arange(2 * (width + 4), dtype=torch.float32).reshape(2, width + 4)
    source = host.to('ruda')
    pending = []
    for length in list(range(1, width + 1)) + [1, 17, 64, width]:
        output = torch.empty((2, length), dtype=torch.float32, device='ruda')
        output.copy_(source[:, :length])
        pending.append((output, length))
    for output, length in pending:
        torch.testing.assert_close(output.cpu(), host[:, :length], rtol=0, atol=0)


def test_idempotent_stream_exchange_preserves_tls_and_rejects_invalid_ids():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    streams = [r.Stream(), r.Stream()]
    barrier = Barrier(2)
    previous = r.current_stream().stream_id
    def worker(stream):
        initial = r.current_stream().stream_id
        with r.stream(stream):
            barrier.wait()
            for _ in range(16):
                assert r._C.stream_command(2, stream.stream_id) == stream.stream_id
                assert r.current_stream().stream_id == stream.stream_id
            with pytest.raises(RuntimeError, match='invalid stream'):
                r._C.stream_command(2, (1 << 63) - 1)
            assert r.current_stream().stream_id == stream.stream_id
            output = torch.ones(257, device='ruda') * 3
        assert r.current_stream().stream_id == initial
        return output
    with ThreadPoolExecutor(max_workers=2) as executor:
        outputs = list(executor.map(worker, streams))
    assert r.current_stream().stream_id == previous
    assert r._C.stream_command(2, previous) == previous
    for output in outputs:
        torch.testing.assert_close(output.cpu(), torch.full((257,), 3.0), rtol=0, atol=0)


@pytest.mark.parametrize('source_dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('target_dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('length,offset', [(512, 0), (512, 4), (513, 1), (512, 1)])
def test_contiguous_conversion_preserves_strided_reference(source_dtype, target_dtype, length, offset):
    torch.manual_seed(261032)
    host = torch.randn(length + offset, dtype=source_dtype)
    host[offset:offset + 8] = torch.tensor([0., -0., float('nan'), float('inf'), float('-inf'),
                                          1.00390625, -1.00390625, 2. ** -24], dtype=source_dtype)
    packed = host.to('ruda')[offset:]
    strided = host[:, None].expand(-1, 2).contiguous().to('ruda')[offset:, 0]
    actual = torch.empty(length + offset, device='ruda', dtype=target_dtype)[offset:]
    expected = torch.empty_like(actual)
    actual.copy_(packed); expected.copy_(strided)
    a, b = actual.cpu(), expected.cpu()
    torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(torch.signbit(a[~torch.isnan(a)]), torch.signbit(b[~torch.isnan(b)]))


@pytest.mark.parametrize('source_dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('target_dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('layout', ['transpose', 'padded', 'broadcast', 'offset', 'tail', 'unaligned_rows'])
def test_row_vector_conversion_preserves_layout_and_rounding(source_dtype, target_dtype, layout):
    torch.manual_seed(261035)
    width = {'padded': 24, 'offset': 20, 'tail': 15, 'unaligned_rows': 19}.get(layout, 16)
    host = torch.randn(1 if layout == 'broadcast' else 2, 3, 7, width, dtype=source_dtype)
    host.flatten()[:8] = torch.tensor([0., -0., float('nan'), float('inf'), float('-inf'),
                                     1.00390625, -1.00390625, 2. ** -24], dtype=source_dtype)
    def view(x):
        if layout == 'transpose':
            return x.transpose(1, 2)
        if layout == 'padded':
            return x[..., 4:20]
        if layout == 'broadcast':
            return x.expand(2, 3, 7, 16)
        if layout == 'offset':
            return x[..., 1:17]
        if layout == 'unaligned_rows':
            return x[..., :16]
        return x
    source = view(host.to('ruda'))
    expected = view(host).to(target_dtype)
    output = torch.empty(source.shape, dtype=target_dtype, device='ruda')
    output.copy_(source)
    actual = output.cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(torch.signbit(actual[~torch.isnan(actual)]), torch.signbit(expected[~torch.isnan(expected)]))
    fallback = view(host).unsqueeze(-1).expand(*source.shape, 2).contiguous().to('ruda')[..., 0]
    output.copy_(fallback)
    torch.testing.assert_close(actual, output.cpu(), rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('operation', ['add', 'mul', 'neg'])
@pytest.mark.parametrize('length,offset', [(512, 0), (512, 4), (513, 1), (512, 1)])
def test_contiguous_arithmetic_preserves_strided_reference(dtype, operation, length, offset):
    torch.manual_seed(261033)
    hosts = [torch.randn(length + offset, dtype=dtype) for _ in range(2)]
    hosts[0][offset:offset + 6] = torch.tensor([0., -0., float('nan'), float('inf'), float('-inf'), 2. ** -24], dtype=dtype)
    values = [x.to('ruda')[offset:] for x in hosts]
    strided = [x[:, None].expand(-1, 2).contiguous().to('ruda')[offset:, 0] for x in hosts]
    def run(x, y, out=None):
        if operation == 'add':
            return torch.add(x, y, alpha=1. / 3., out=out)
        return torch.mul(x, y, out=out) if operation == 'mul' else torch.neg(x, out=out)
    a, b = run(*values).cpu(), run(*strided).cpu()
    torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(torch.signbit(a[~torch.isnan(a)]), torch.signbit(b[~torch.isnan(b)]))
    a, b = run(*values, out=values[0]).cpu(), run(*strided, out=strided[0]).cpu()
    torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize('column_major', [False, True])
@pytest.mark.parametrize('transpose_a', [False, True])
@pytest.mark.parametrize('shape', [(512, 3584, 16), (512, 18944, 16), (32, 152064, 16),
                                  (31, 1537, 7), (3, 513, 1), (2, 512, 17), (1, 1025, 32),
                                  (16, 512, 3584), (16, 512, 18944), (7, 1025, 129), (1, 512, 3584)])
def test_skinny_fp32_matmul_split_k(shape, column_major, transpose_a):
    m, k, n = shape
    torch.manual_seed(179)
    a_backing = torch.randn(k + 3, m) if transpose_a else torch.randn(m, k)
    a = a_backing[1:k + 1].t() if transpose_a else a_backing
    native_a_backing = a_backing.to('ruda')
    native_a = native_a_backing[1:k + 1].t() if transpose_a else native_a_backing
    # Nonzero storage offsets and padded strides must survive native binding.
    backing = torch.randn(n, k + 3) if column_major else torch.randn(k, n + 3)
    b = backing[:, 1:k + 1].t() if column_major else backing[:, 1:n + 1]
    native_backing = backing.to('ruda')
    native_b = native_backing[:, 1:k + 1].t() if column_major else native_backing[:, 1:n + 1]
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        expected = a.cuda() @ b.cuda()
        actual = native_a @ native_b
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


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('transpose_a,transpose_b', [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize('shape', [(512, 3584, 18944), (512, 18944, 3584), (32, 512, 3584),
                                  (31, 513, 129), (64, 257, 129)])
def test_pipelined_half_gemm(shape, dtype, transpose_a, transpose_b):
    m, k, n = shape
    torch.manual_seed(191)
    a = torch.randn(k, m, dtype=dtype) if transpose_a else torch.randn(m, k, dtype=dtype)
    b = torch.randn(n, k, dtype=dtype) if transpose_b else torch.randn(k, n, dtype=dtype)
    x, y = a.to('ruda'), b.to('ruda')
    if transpose_a:
        a, x = a.t(), x.t()
    if transpose_b:
        b, y = b.t(), y.t()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        expected = (a.cuda().float() @ b.cuda().float()).to(dtype).cpu()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    actual = (x @ y).cpu()
    tolerance = 2.5e-2 if dtype == torch.bfloat16 else 5e-3
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_pipelined_addmm_fp32_epilogue(dtype):
    torch.manual_seed(193)
    a = torch.randn(64, 512, dtype=dtype)
    b = torch.randn(512, 1024, dtype=dtype)
    bias = torch.randn(1024, dtype=dtype)
    expected = (a.float() @ b.float() + bias.float()).to(dtype)
    actual = torch.addmm(bias.to('ruda'), a.to('ruda'), b.to('ruda')).cpu()
    tolerance = 2.5e-2 if dtype == torch.bfloat16 else 5e-3
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


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
