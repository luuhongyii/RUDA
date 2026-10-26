import itertools
import pytest
import torch
from torch.nn import functional as F
import ruda_torch as r
from ruda_torch.lora_epilogue import _rank16_residual, rank16_residual_eligible


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('shape', [(16, 128), (1, 32, 256), (2, 256, 512)])
@pytest.mark.parametrize('scale', [1., 1. / 3., 1.75])
def test_rank16_fusion_preserves_native_forward_and_gradients(dtype, shape, scale):
    assert r._lora_projection_available
    torch.manual_seed(261030)
    hosts = [torch.randn(shape, dtype=dtype), torch.randn((*shape[:-1], 16)),
             torch.randn(shape[-1], 16)]
    fused = [x.to('ruda').requires_grad_() for x in hosts]
    reference = [x.to('ruda').requires_grad_() for x in hosts]
    actual = _rank16_residual(*fused, scale)
    expected = r.lora_epilogue(reference[0], F.linear(reference[1], reference[2]), scale)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    gradient = torch.randn(shape[-1], dtype=dtype).to('ruda').expand(shape)
    actual.backward(gradient); expected.backward(gradient)
    for a, b in zip(fused, reference):
        torch.testing.assert_close(a.grad.cpu(), b.grad.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize('needs', list(itertools.product([False, True], repeat=3)))
def test_rank16_fusion_selected_gradients(needs):
    hosts = [torch.ones(16, 128, dtype=torch.bfloat16), torch.ones(16, 16), torch.ones(128, 16)]
    inputs = [x.to('ruda').requires_grad_(need) for x, need in zip(hosts, needs)]
    result = _rank16_residual(*inputs, .5)
    assert result.requires_grad == any(needs)
    if any(needs):
        result.sum().backward()
        for index, (x, need) in enumerate(zip(inputs, needs)):
            if need:
                expected = torch.full(x.shape, [1., 64., 8.][index], dtype=x.dtype)
                torch.testing.assert_close(x.grad.cpu(), expected, rtol=0, atol=0)
            else:
                assert x.grad is None


def test_rank16_fusion_rounding_and_nonfinite_values():
    base = torch.full((16, 128), -1., dtype=torch.bfloat16)
    rank = torch.zeros(16, 16); rank[:, 0] = 1.
    weight = torch.zeros(128, 16); weight[:, 0] = 1.003
    args = [x.to('ruda') for x in (base, rank, weight)]
    torch.testing.assert_close(_rank16_residual(*args, 1.).cpu(), torch.zeros_like(base), rtol=0, atol=0)
    rank[0, 1] = float('nan'); weight[1, 2] = float('inf'); base[2, 2] = float('-inf')
    args = [x.to('ruda') for x in (base, rank, weight)]
    expected = r.lora_epilogue(args[0], F.linear(args[1], args[2]), 1.)
    torch.testing.assert_close(_rank16_residual(*args, 1.).cpu(), expected.cpu(), rtol=0, atol=0, equal_nan=True)


def test_rank16_fusion_rejects_unaligned_weight_and_row_tail():
    base = torch.ones(32, 128, dtype=torch.bfloat16).to('ruda')
    rank = torch.ones(32, 16).to('ruda')
    weight = torch.ones(128 * 16 + 1).to('ruda')[1:].reshape(128, 16)
    assert not rank16_residual_eligible(base, rank, weight)
    assert not rank16_residual_eligible(base[:31], rank[:31], weight.clone())


def test_rank16_fusion_preserves_second_derivatives():
    inputs = [torch.ones(16, 128, dtype=torch.bfloat16), torch.ones(16, 16), torch.ones(128, 16)]
    derivatives = []
    for fused in (False, True):
        base, rank, weight = [x.to('ruda').requires_grad_() for x in inputs]
        out = _rank16_residual(base, rank, weight, .5) if fused else r.lora_epilogue(base, F.linear(rank, weight), .5)
        first = torch.autograd.grad(out.float().square().sum(), rank, create_graph=True)[0]
        derivatives.append(torch.autograd.grad(first.sum(), weight)[0].cpu())
    torch.testing.assert_close(*derivatives, rtol=0, atol=0)


def test_rank16_fusion_aot_forward_and_backward():
    from torch._dynamo.backends.common import aot_autograd
    graphs = []
    def capture(graph, inputs):
        graphs.append(graph)
        return graph.forward
    compiled = torch.compile(lambda base, rank, weight: _rank16_residual(base, rank, weight, .5).sum(),
                             backend=aot_autograd(fw_compiler=capture, bw_compiler=capture), fullgraph=True)
    inputs = [torch.ones(16, 128, dtype=torch.bfloat16).to('ruda').requires_grad_(),
              torch.ones(16, 16).to('ruda').requires_grad_(),
              torch.ones(128, 16).to('ruda').requires_grad_()]
    compiled(*inputs).backward()
    assert all(x.grad is not None for x in inputs)
    assert any('lora_rank16_residual' in str(node.target) for graph in graphs for node in graph.graph.nodes)


def test_lora_dispatch_uses_native_kernels():
    for name in ('lora_epilogue_forward', 'lora_epilogue_backward', 'lora_rank16_residual'):
        assert torch._C._dispatch_has_kernel_for_dispatch_key('ruda::' + name, 'PrivateUse1')


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('offset', [0, 4])
def test_rank16_dispatch_preserves_offsets_and_input_versions(dtype, offset):
    torch.manual_seed(261047)
    shapes = [(2, 16, 256), (2, 16, 16), (256, 16)]
    inputs = [torch.randn(torch.Size(shape).numel() + offset, dtype=dt).to('ruda')[offset:].reshape(shape)
              for shape, dt in zip(shapes, (dtype, torch.float32, torch.float32))]
    snapshots = [x.cpu() for x in inputs]
    versions = [x._version for x in inputs]
    actual = _rank16_residual(*inputs, .375)
    expected = r.lora_epilogue(inputs[0], F.linear(inputs[1], inputs[2]), .375)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    assert [x._version for x in inputs] == versions
    for x, before in zip(inputs, snapshots):
        torch.testing.assert_close(x.cpu(), before, rtol=0, atol=0)
        assert actual.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()


@pytest.mark.parametrize('case', ['row_tail', 'column_tail', 'rank_width', 'weight_shape',
                                 'unaligned_weight', 'transposed_rank', 'base_dtype', 'rank_dtype'])
def test_rank16_native_dispatch_rejects_ineligible_geometry(case):
    base = torch.ones(16, 128, dtype=torch.bfloat16).to('ruda')
    rank = torch.ones(16, 16).to('ruda')
    weight = torch.ones(128, 16).to('ruda')
    if case == 'row_tail': base, rank = base[:15], rank[:15]
    if case == 'column_tail': base, weight = base[:, :127].contiguous(), weight[:127]
    if case == 'rank_width': rank = rank[:, :15].contiguous()
    if case == 'weight_shape': weight = weight[:127]
    if case == 'unaligned_weight': weight = torch.ones(2049).to('ruda')[1:].reshape(128, 16)
    if case == 'transposed_rank': rank = rank.t()
    if case == 'base_dtype': base = base.float()
    if case == 'rank_dtype': rank = rank.to(torch.bfloat16)
    assert not rank16_residual_eligible(base, rank, weight)
    with pytest.raises(ValueError, match='unsupported RUDA rank-16 residual projection geometry'):
        _rank16_residual(base, rank, weight, .5)


def test_rank16_native_dispatch_preserves_saved_tensor_version_checks():
    base = torch.ones(16, 128, dtype=torch.bfloat16).to('ruda').requires_grad_()
    rank = torch.ones(16, 16).to('ruda').requires_grad_()
    weight = torch.ones(128, 16).to('ruda').requires_grad_()
    output = _rank16_residual(base, rank, weight, .5)
    with torch.no_grad(): weight.add_(1.)
    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        output.sum().backward()


def test_rank16_native_dispatch_is_visible_to_torch_dispatch_modes():
    from torch.utils._python_dispatch import TorchDispatchMode
    seen = []
    class Record(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            seen.append(str(func))
            return func(*args, **(kwargs or {}))
    args = [torch.ones(16, 128, dtype=torch.bfloat16).to('ruda'),
            torch.ones(16, 16).to('ruda'), torch.ones(128, 16).to('ruda')]
    with Record(): output = _rank16_residual(*args, .5)
    assert 'ruda.lora_rank16_residual.default' in seen
    torch.testing.assert_close(output.cpu(), torch.full((16, 128), 9., dtype=torch.bfloat16), rtol=0, atol=0)


@pytest.mark.parametrize('op', ['projection', 'epilogue'])
def test_lora_native_dispatch_schema_fake_and_aot(op):
    from ruda_torch.lora_epilogue import _forward
    base = torch.randn(16, 128, dtype=torch.bfloat16).to('ruda').requires_grad_()
    if op == 'projection':
        fn = _rank16_residual
        args = (base, torch.randn(16, 16).to('ruda').requires_grad_(),
                torch.randn(128, 16).to('ruda').requires_grad_(), .375)
    else:
        fn = _forward
        args = (base, torch.randn(16, 128).to('ruda').requires_grad_(), .375)
    assert torch._C._dispatch_has_kernel_for_dispatch_key(fn._opoverload.name(), 'Autograd')
    torch.library.opcheck(fn, args, test_utils=('test_schema', 'test_faketensor', 'test_aot_dispatch_dynamic'))


def test_rank16_native_dispatch_vmap_matches_individual_calls():
    torch.manual_seed(261048)
    bases = torch.randn(3, 16, 128, dtype=torch.bfloat16).to('ruda')
    ranks = torch.randn(3, 16, 16).to('ruda')
    weight = torch.randn(128, 16).to('ruda')
    actual = torch.vmap(lambda base, rank: _rank16_residual(base, rank, weight, .375))(bases, ranks)
    expected = torch.stack([_rank16_residual(bases[i], ranks[i], weight, .375) for i in range(3)])
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize('reentrant', [False, True])
def test_rank16_native_dispatch_checkpoint_and_nondefault_stream(reentrant):
    from torch.utils.checkpoint import checkpoint
    producer, consumer = r.Stream(), r.Stream()
    with r.stream(producer):
        args = [torch.ones(16, 128, dtype=torch.bfloat16).to('ruda').requires_grad_(),
                torch.ones(16, 16).to('ruda').requires_grad_(),
                torch.ones(128, 16).to('ruda').requires_grad_()]
        ready = producer.record_event()
    with r.stream(consumer):
        consumer.wait_event(ready)
        output = checkpoint(lambda *values: _rank16_residual(*values, .5), *args, use_reentrant=reentrant)
        output.sum().backward()
        for value in args: r.record_stream(value, consumer)
    consumer.synchronize()
    torch.testing.assert_close(output.cpu(), torch.full((16, 128), 9., dtype=torch.bfloat16), rtol=0, atol=0)
    for x, value in zip(args, [1., 64., 8.]):
        torch.testing.assert_close(x.grad.cpu(), torch.full(x.shape, value, dtype=x.dtype), rtol=0, atol=0)
    ready.close()
