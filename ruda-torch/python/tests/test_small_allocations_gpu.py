import gc
import os

import pytest
import torch

if os.environ.get('RUDA_REQUIRE_GPU') != '1':
    pytest.skip('requires real native GPU allocations', allow_module_level=True)

import ruda_torch as r


def test_small_allocation_rollover_keeps_live_storage_and_views():
    saved = []
    for cycle in range(3):
        for index in range(160):
            value = torch.full((1024,), cycle * 160 + index, device='ruda', dtype=torch.float32)
            if index % 31 == 0:
                saved.append((value[3:19], cycle * 160 + index))
            value.add_(.5)
        gc.collect()
    pointers = [view.data_ptr() for view, _ in saved]
    assert len(set(pointers)) == len(pointers)
    for view, value in saved:
        torch.testing.assert_close(view.cpu(), torch.full((16,), value + .5), rtol=0, atol=0)


def test_small_allocation_cross_stream_and_resize():
    first, second = r.Stream(), r.Stream()
    with r.stream(first):
        x = torch.arange(513, device='ruda', dtype=torch.float32)
        expected = torch.arange(513, dtype=torch.float32)
        ready = r.Event();ready.record()
    with r.stream(second):
        second.wait_event(ready)
        r.record_stream(x, second)
        y = x * 2.
        for _ in range(160):
            torch.empty(2048, device='ruda').fill_(-7.)
        y.resize_(1024)
        y[513:].fill_(11.)
    r.synchronize()
    torch.testing.assert_close(y[:513].cpu(), expected * 2, rtol=0, atol=0)
    torch.testing.assert_close(y[513:].cpu(), torch.full((511,), 11.), rtol=0, atol=0)
    torch.testing.assert_close(x.cpu(), expected, rtol=0, atol=0)


def test_small_and_large_bindings_can_share_graph_inputs_without_overlap():
    x = torch.ones((1, 256), device='ruda', dtype=torch.bfloat16)
    y = torch.full((1, 256), 2., device='ruda', dtype=torch.bfloat16)
    fn = r.compile(lambda a, b: (a + b) * b, native='required', fullgraph=True, dynamic=False)
    try:
        first = fn(x, y)
        y.fill_(3.)
        second = fn(x, y)
        torch.testing.assert_close(first.cpu(), torch.full((1, 256), 6., dtype=torch.bfloat16), rtol=0, atol=0)
        torch.testing.assert_close(second.cpu(), torch.full((1, 256), 12., dtype=torch.bfloat16), rtol=0, atol=0)
    finally:
        fn.close()


def test_medium_allocation_rollover_and_size_boundaries():
    saved = []
    for index in range(24):
        count = (16384, 16385, 262144, 524288, 524289)[index % 5]
        value = torch.full((count,), float(index), device='ruda')
        value[-7:].add_(.5)
        if index % 4 == 0:
            saved.append((value[-7:], index))
    gc.collect()
    assert len({view.data_ptr() for view, _ in saved}) == len(saved)
    for view, index in saved:
        torch.testing.assert_close(view.cpu(), torch.full((7,), index + .5), rtol=0, atol=0)


def test_medium_allocation_stream_switch_preserves_live_storage():
    first, second = r.Stream(), r.Stream()
    with r.stream(first):
        x = torch.arange(262144, device='ruda', dtype=torch.float32)
        ready = r.Event();ready.record()
    with r.stream(second):
        second.wait_event(ready)
        r.record_stream(x, second)
        y = x + 2
        for _ in range(20):
            torch.empty(262144, device='ruda').fill_(-3)
        consumed = r.Event();consumed.record()
    with r.stream(first):
        first.wait_event(consumed)
        x.add_(1)
    r.synchronize()
    torch.testing.assert_close(y.cpu(), torch.arange(262144, dtype=torch.float32) + 2, rtol=0, atol=0)
    torch.testing.assert_close(x.cpu(), torch.arange(262144, dtype=torch.float32) + 1, rtol=0, atol=0)
