"""Native factory/storage regressions. Host tensors are independent references only."""
import copy
import os

import pytest
import torch


@pytest.fixture(scope='module')
def r():
    assert os.environ.get('RUDA_CUDA_COMPILER') == 'ptx'
    import ruda_torch
    assert ruda_torch._factory_available
    return ruda_torch


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16,
                                   torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8])
@pytest.mark.parametrize('start,end,step', [(0, 17, 1), (13, 0, -3), (5, 5, 2)])
def test_arange_native_strides_and_no_transfers(r, dtype, start, end, step):
    expected = torch.arange(start, end, step, dtype=dtype)
    storage = torch.empty(expected.numel()*2+3, device='ruda', dtype=dtype).fill_(7)
    view = storage[1:1+2*expected.numel():2]
    before = r.execution_stats()
    actual = torch.arange(start, end, step, device='ruda', dtype=dtype)
    result = torch.arange(start, end, step, out=view)
    r.synchronize()
    after = r.execution_stats()
    assert result is view and view.storage_offset() == 1 and view.stride() == (2,)
    for key in ('host_to_device_bytes', 'device_to_host_bytes'):
        assert before[key] == after[key]
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(view.cpu(), expected, rtol=0, atol=0)
    assert torch.all(storage[::2].cpu() == 7)


def test_arange_large_integer_resize_and_errors(r):
    out = torch.empty(0, device='ruda', dtype=torch.int64)
    start = 2**60+3
    torch.arange(start, start+35, 3, out=out)
    torch.testing.assert_close(out.cpu(), torch.arange(start, start+35, 3), rtol=0, atol=0)
    for args in ((0, 5, 0), (5, 0, 1), (0, 5, -1), (float('nan'), 5, 1)):
        with pytest.raises(RuntimeError):
            torch.arange(*args, out=out)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_normal_generator_replay_strides_and_no_transfers(r, dtype):
    generator = torch.Generator(device='ruda').manual_seed(129)
    state = generator.get_state()
    out = torch.empty((17, 33), device='ruda', dtype=dtype).t()
    before = r.execution_stats()
    out.normal_(mean=.25, std=.5, generator=generator)
    generator.set_state(state)
    replay = torch.randn(out.shape, device='ruda', dtype=dtype, generator=generator)*.5+.25
    r.synchronize()
    after = r.execution_stats()
    for key in ('host_to_device_bytes', 'device_to_host_bytes'):
        assert before[key] == after[key]
    # The kernel applies the affine transform before storage narrowing.
    generator.set_state(state)
    exact = torch.empty(out.shape, device='ruda', dtype=dtype).normal_(.25, .5, generator=generator)
    torch.testing.assert_close(out.cpu(), exact.cpu(), rtol=0, atol=0)
    assert torch.isfinite(replay.cpu()).all()
    values = out.cpu().float()
    assert abs(values.mean().item()-.25) < .12 and abs(values.std().item()-.5) < .12
    r.manual_seed_all(731)
    saved = r.get_rng_state()
    a = torch.randn(29, device='ruda', dtype=dtype)
    r.set_rng_state(saved)
    b = torch.randn(29, device='ruda', dtype=dtype)
    torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)


def test_deepcopy_preserves_aliases_offsets_and_independent_storage(r):
    base = torch.arange(35, device='ruda', dtype=torch.float32).reshape(5, 7)
    before = r.execution_stats()
    cloned = copy.deepcopy({'a': base[:, 1::2], 'b': base.t()})
    r.synchronize()
    after = r.execution_stats()
    for key in ('host_to_device_bytes', 'device_to_host_bytes'):
        assert before[key] == after[key]
    assert cloned['a'].untyped_storage().data_ptr() == cloned['b'].untyped_storage().data_ptr()
    assert cloned['b'].untyped_storage().data_ptr() != base.untyped_storage().data_ptr()
    assert cloned['a'].stride() == base[:, 1::2].stride() and cloned['a'].storage_offset() == 1
    torch.testing.assert_close(cloned['a'].cpu(), base[:, 1::2].cpu(), rtol=0, atol=0)
    cloned['b'].zero_()
    assert torch.count_nonzero(cloned['a'].cpu()) == 0
    assert torch.count_nonzero(base.cpu()) == 34


@pytest.mark.parametrize('dtype', [torch.bool, torch.float32, torch.float16, torch.int64])
@pytest.mark.parametrize('layout', [0, 1, 2])
def test_advanced_integer_index_broadcast_strides_negative_and_out(r, dtype, layout):
    assert r._indexing_available
    data = torch.arange(60).reshape(3, 4, 5)
    data = (data % 3 != 0) if dtype == torch.bool else data.to(dtype)
    source = data.permute(2, 1, 0)[:, 1:, :]
    native = data.to('ruda').permute(2, 1, 0)[:, 1:, :]
    indices = ([torch.tensor([[0], [-1]], dtype=torch.int32), torch.tensor([0, 1, -1])]
               if layout == 0 else [torch.tensor([[0], [-1]]), None, torch.tensor([0, -1])]
               if layout == 1 else [None, torch.tensor(-1)])
    expected = torch.ops.aten.index.Tensor(source, indices)
    device_indices = [None if item is None else item.to('ruda') for item in indices]
    storage = torch.empty(expected.numel()*2+3, device='ruda', dtype=dtype).fill_(1)
    out = storage[1:1+2*expected.numel():2].view(expected.shape)
    before = r.execution_stats()
    actual = torch.ops.aten.index.Tensor(native, device_indices)
    result = torch.ops.aten.index.Tensor_out(native, device_indices, out=out)
    r.synchronize()
    after = r.execution_stats()
    assert result is out and out.storage_offset() == 1
    assert after['kernel_launches'] > before['kernel_launches']
    assert after['host_to_device_bytes'] == before['host_to_device_bytes']
    assert after['device_to_host_bytes']-before['device_to_host_bytes'] == 8*sum(i is not None for i in indices)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    assert torch.all(storage[::2].cpu() == 1)


def test_advanced_index_empty_resize_and_bounds(r):
    source = torch.arange(12, device='ruda').reshape(3, 4)
    empty = torch.empty(0, device='ruda', dtype=torch.int64)
    result = torch.ops.aten.index.Tensor(source, [empty])
    assert result.shape == (0, 4)
    out = torch.empty(0, device='ruda', dtype=torch.int64)
    index = torch.tensor([-3, -1]).to('ruda')
    torch.ops.aten.index.Tensor_out(source, [index], out=out)
    torch.testing.assert_close(out.cpu(), torch.arange(12).reshape(3, 4)[[-3, -1]], rtol=0, atol=0)
    for value in (-4, 3):
        invalid = torch.tensor([value]).to('ruda')
        with pytest.raises((RuntimeError, IndexError), match='out of bounds'):
            torch.ops.aten.index.Tensor(source, [invalid])


@pytest.mark.parametrize('compiled', [False, True])
def test_advanced_index_nested_vmap_attention_mask_and_fullgraph(r, compiled):
    def mask(padding, batch, heads, queries, keys):
        def rule(b, h, q, k):
            return (q >= k) & padding[b, k]
        fn = torch.vmap(rule, in_dims=(None, None, None, 0))
        fn = torch.vmap(fn, in_dims=(None, None, 0, None))
        fn = torch.vmap(fn, in_dims=(None, 0, None, None))
        fn = torch.vmap(fn, in_dims=(0, None, None, None))
        return fn(batch, heads, queries, keys)
    inputs = [torch.tensor([[True]*6, [True]*4+[False]*2]), torch.arange(2),
              torch.arange(1), torch.arange(6), torch.arange(6)]
    expected = mask(*inputs)
    device_inputs = [value.to('ruda') for value in inputs]
    executable = r.compile(mask, fullgraph=True) if compiled else mask
    try:
        before = r.execution_stats()
        actual = executable(*device_inputs)
        r.synchronize()
        after = r.execution_stats()
        assert after['kernel_launches'] > before['kernel_launches']
        assert after['host_to_device_bytes'] == before['host_to_device_bytes']
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        if compiled:
            assert any(graph['phase'] == 'forward' for graph in executable.info['graphs'])
    finally:
        if compiled:
            executable.close()
