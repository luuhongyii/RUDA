import os
import pytest
import torch


@pytest.fixture(scope='module', autouse=True)
def backend():
    if os.environ.get('RUDA_REQUIRE_GPU') != '1':
        pytest.skip('requires the native RUDA GPU backend')
    import ruda_torch
    assert ruda_torch._C.abi_version == 10


REAL = [torch.float32, torch.float16, torch.bfloat16, torch.float64,
        torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool]
NUMERIC = REAL + [torch.complex64, torch.complex128]


def equal(actual, expected):
    if isinstance(expected, (tuple, list)):
        for value, reference in zip(actual, expected):
            equal(value, reference)
    else:
        tolerance = 3e-2 if expected.dtype == torch.bfloat16 else 3e-3 if expected.dtype == torch.float16 else 2e-6
        if not (expected.is_floating_point() or expected.is_complex()):
            tolerance = 0
        torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance, equal_nan=True)


@pytest.mark.parametrize('dtype', REAL)
def test_pointwise_ordering_and_clamp(dtype):
    a = torch.tensor([[3, 1, 4], [0, 2, 1]], dtype=dtype).t()
    b = torch.tensor([[2, 0]], dtype=dtype)
    for operation in (torch.maximum, torch.minimum, torch.fmax, torch.fmin):
        equal(operation(a.to('ruda'), b.to('ruda')), operation(a, b))
    if dtype != torch.bool:
        equal(torch.clamp(a.to('ruda'), min=1, max=2), torch.clamp(a, min=1, max=2))
        equal(torch.clamp(a.to('ruda'), min=b.to('ruda')), torch.clamp(a, min=b))
        equal(torch.clamp(a.to('ruda'), min=4, max=1), torch.clamp(a, min=4, max=1))
        actual = a.to('ruda')
        assert actual.clamp_(min=1, max=2) is actual
        equal(actual, a.clamp(1, 2))
    equal(torch.clamp(a.to('ruda'), min=.5, max=2.5), torch.clamp(a, min=.5, max=2.5))


@pytest.mark.parametrize('dtype', REAL)
def test_extrema_reductions_and_scans(dtype):
    source = torch.tensor([[[2, 2, 1, 4], [0, 3, 3, 1]], [[2, 0, 1, 1], [2, 2, 4, 0]]], dtype=dtype).transpose(0, 2)
    actual = source.to('ruda')
    for operation in (torch.min, torch.max):
        equal(operation(actual), operation(source))
        for dim in (0, -1):
            equal(operation(actual, dim=dim, keepdim=True), operation(source, dim=dim, keepdim=True))
    for operation in (torch.amin, torch.amax):
        for dims in ((), (0, 2), (1,)):
            equal(operation(actual, dim=dims, keepdim=True), operation(source, dim=dims, keepdim=True))
    for operation in (torch.cummin, torch.cummax):
        equal(operation(actual, dim=0), operation(source, dim=0))
        equal(operation(actual, dim=-1), operation(source, dim=-1))
    if dtype != torch.bool:
        for operation in (torch.argmin, torch.argmax):
            equal(operation(actual, keepdim=True), operation(source, keepdim=True))
            equal(operation(actual, dim=1), operation(source, dim=1))
    equal(torch.aminmax(actual), torch.aminmax(source))
    equal(torch.aminmax(actual, dim=1, keepdim=True), torch.aminmax(source, dim=1, keepdim=True))


@pytest.mark.parametrize('dtype', REAL[:4])
def test_nan_infinity_zero_and_tie_rules(dtype):
    source = torch.tensor([[2., 2., 1., float('nan'), float('nan'), 3.],
                           [-float('inf'), float('inf'), float('inf'), 0., -0., 0.]], dtype=dtype)
    for operation in (torch.min, torch.max, torch.cummin, torch.cummax):
        equal(operation(source.to('ruda'), dim=1), operation(source, dim=1))
    a = torch.tensor([0., -0., float('nan'), 1., float('nan')], dtype=dtype)
    b = torch.tensor([-0., 0., 1., float('nan'), float('nan')], dtype=dtype)
    for operation in (torch.maximum, torch.minimum, torch.fmax, torch.fmin):
        actual, expected = operation(a.to('ruda'), b.to('ruda')).cpu(), operation(a, b)
        equal(actual, expected)
        equal(torch.signbit(actual[:2]), torch.signbit(expected[:2]))
    for operation in (torch.argmin, torch.argmax):
        equal(operation(source.to('ruda'), dim=1), operation(source, dim=1))


@pytest.mark.parametrize('dtype', REAL[:-1])
@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('right', [False, True])
def test_searchsorted_bucketize_sorter_and_output(dtype, batched, right):
    sequence = torch.tensor([[1, 1, 3, 5], [0, 2, 2, 4]], dtype=dtype)
    values = torch.tensor([[0, 1, 2, 5, 6], [0, 1, 2, 3, 5]], dtype=dtype)
    if not batched:
        sequence = sequence[0]
    actual = torch.searchsorted(sequence.to('ruda'), values.to('ruda'), right=right)
    equal(actual, torch.searchsorted(sequence, values, right=right))
    out = torch.empty(0, device='ruda', dtype=torch.int32)
    assert torch.searchsorted(sequence.to('ruda'), values.to('ruda'), side='right' if right else 'left', out_int32=True, out=out) is out
    equal(out, torch.searchsorted(sequence, values, right=right, out_int32=True))
    reverse = sequence.flip(-1)
    sorter = reverse.argsort(dim=-1, stable=True)
    equal(torch.searchsorted(reverse.to('ruda'), values.to('ruda'), sorter=sorter.to('ruda'), right=right),
          torch.searchsorted(reverse, values, sorter=sorter, right=right))
    if not batched:
        equal(torch.bucketize(values.to('ruda'), sequence.to('ruda'), right=right), torch.bucketize(values, sequence, right=right))
        equal(torch.searchsorted(sequence.to('ruda'), 2., right=right), torch.searchsorted(sequence, 2., right=right))


@pytest.mark.parametrize('dtype', NUMERIC)
@pytest.mark.parametrize('reduce', ['sum', 'prod', 'mean', 'amin', 'amax'])
@pytest.mark.parametrize('include_self', [False, True])
def test_scatter_reduce_modes_layout_and_duplicates(dtype, reduce, include_self):
    source = torch.tensor([[2, 1, 4], [1, 3, 2]], dtype=dtype).t()
    values = torch.tensor([[0, 2], [3, 1], [2, 0], [1, 2]], dtype=dtype)
    index = torch.tensor([[0, 1], [2, 1], [0, 0], [2, 1]])
    if dtype.is_complex and reduce in ('amin', 'amax') or dtype == torch.bool and reduce == 'mean':
        with pytest.raises(RuntimeError):
            torch.scatter_reduce(source.to('ruda'), 0, index.to('ruda'), values.to('ruda'), reduce, include_self=include_self)
        return
    expected = torch.scatter_reduce(source, 0, index, values, reduce, include_self=include_self)
    equal(torch.scatter_reduce(source.to('ruda'), 0, index.to('ruda'), values.to('ruda'), reduce, include_self=include_self), expected)
    actual = source.to('ruda')
    assert actual.scatter_reduce_(0, index.to('ruda'), values.to('ruda'), reduce, include_self=include_self) is actual
    equal(actual, expected)


@pytest.mark.parametrize('dtype', REAL[:-1])
@pytest.mark.parametrize('reduce', ['prod', 'mean', 'amin', 'amax'])
@pytest.mark.parametrize('include_self', [False, True])
def test_index_reduce_modes_and_output(dtype, reduce, include_self):
    source = torch.tensor([[2, 1, 4], [1, 3, 2]], dtype=dtype).t()
    values = torch.tensor([[0, 2, 1, 2], [3, 1, 2, 1], [2, 0, 2, 0]], dtype=dtype)
    index = torch.tensor([1, 0, 1, 1], dtype=torch.int32)
    expected = torch.index_reduce(source, 1, index, values, reduce, include_self=include_self)
    actual = source.to('ruda')
    out = torch.empty(0, dtype=dtype, device='ruda')
    assert torch.index_reduce(actual, 1, index.to('ruda'), values.to('ruda'), reduce, include_self=include_self, out=out) is out
    equal(out, expected)
    assert actual.index_reduce_(1, index.to('ruda'), values.to('ruda'), reduce, include_self=include_self) is actual
    equal(actual, expected)


@pytest.mark.parametrize('mode', ['sum', 'prod', 'mean', 'amin', 'amax'])
@pytest.mark.parametrize('include_self', [False, True])
def test_scatter_reduce_gradients_with_zeros_and_ties(mode, include_self):
    results = []
    for device in ('ruda', 'cpu'):
        a = torch.tensor([[2., 0., 1., 7.], [1., 2., 0., 8.]], dtype=torch.float64, device=device, requires_grad=True)
        b = torch.tensor([[0., 0., 2., 3.], [2., 2., 1., 0.]], dtype=a.dtype, device=device, requires_grad=True)
        index = torch.tensor([[0, 0, 1, 2], [1, 1, 0, 2]], device=device)
        result = a.scatter_reduce(1, index, b, mode, include_self=include_self)
        weights = torch.arange(1, 9, dtype=a.dtype, device=device).reshape(2, 4)
        (result * weights).sum().backward()
        results.append((result, a.grad, b.grad))
    equal(*results)


@pytest.mark.parametrize('mode', ['prod', 'mean', 'amin', 'amax'])
@pytest.mark.parametrize('include_self', [False, True])
def test_index_reduce_gradients(mode, include_self):
    results = []
    for device in ('ruda', 'cpu'):
        a = torch.tensor([[2., 0., 1.], [1., 2., 0.]], dtype=torch.float64, device=device, requires_grad=True)
        b = torch.tensor([[0., 0., 2., 3.], [2., 2., 1., 0.]], dtype=a.dtype, device=device, requires_grad=True)
        index = torch.tensor([0, 0, 1, 2], device=device)
        result = a.index_reduce(1, index, b, mode, include_self=include_self)
        result.sum().backward()
        results.append((result, a.grad, b.grad))
    equal(*results)


@pytest.mark.parametrize('operation', ['minimum', 'maximum', 'clamp', 'min', 'max', 'amin', 'amax', 'cummin', 'cummax'])
def test_ordering_gradients(operation):
    results = []
    for device in ('ruda', 'cpu'):
        a = torch.tensor([[2., 2., 1., 0.], [0., 1., 1., 2.]], dtype=torch.float64, device=device, requires_grad=True)
        b = torch.tensor([[1., 2., 0., 0.]], dtype=a.dtype, device=device, requires_grad=True)
        if operation in ('minimum', 'maximum'):
            result = getattr(torch, operation)(a, b)
        elif operation == 'clamp':
            result = torch.clamp(a, min=b, max=b + 1)
        else:
            result = getattr(torch, operation)(a, dim=1)
            if operation in ('min', 'max', 'cummin', 'cummax'):
                result = result.values
        weights = torch.arange(1, result.numel()+1, dtype=a.dtype, device=device).reshape(result.shape)
        (result * weights).sum().backward()
        results.append((result, a.grad, b.grad))
    for actual, expected in zip(*results):
        if expected is None:
            assert actual is None
        else:
            equal(actual, expected)


def test_empty_scalar_out_and_invalid_contracts():
    a = torch.empty((0, 3), device='ruda')
    equal(torch.amax(a, dim=1), torch.empty(0))
    equal(torch.cummax(a, 0), torch.cummax(torch.empty((0, 3)), 0))
    equal(torch.cummin(a, 1), torch.cummin(torch.empty((0, 3)), 1))
    equal(torch.max(a, dim=1, keepdim=True), torch.max(torch.empty((0, 3)), dim=1, keepdim=True))
    with pytest.raises((RuntimeError, IndexError)):
        torch.amax(a)
    with pytest.raises(RuntimeError):
        torch.amax(torch.ones((2, 3), device='ruda'), dim=(0, 0))
    scalar = torch.tensor(3., device='ruda')
    equal(torch.max(scalar, dim=0), torch.max(torch.tensor(3.), dim=0))
    equal(torch.cummin(scalar, 0), torch.cummin(torch.tensor(3.), 0))
    values = torch.tensor([-float('inf'), 0., float('inf'), float('nan')])
    boundaries = torch.tensor([-float('inf'), 0., float('inf')])
    equal(torch.searchsorted(boundaries.to('ruda'), values.to('ruda')), torch.searchsorted(boundaries, values))
    equal(torch.searchsorted(torch.empty(0, device='ruda'), values.to('ruda')), torch.zeros(4, dtype=torch.int64))
    with pytest.raises(RuntimeError):
        torch.searchsorted(boundaries.to('ruda'), values.to('ruda'), right=True, side='left')
    with pytest.raises(RuntimeError):
        torch.searchsorted(boundaries.to('ruda'), values.to('ruda'), sorter=torch.tensor([0, 1, 3], device='ruda'))
    source = torch.tensor([[3., 1., 2.], [2., 4., 0.]], device='ruda')
    out = torch.empty(0, device='ruda')
    idx = torch.empty(0, dtype=torch.int64, device='ruda')
    result = torch.max(source, dim=1, out=(out, idx))
    assert result.values is out and result.indices is idx
    equal(result, torch.max(source.cpu(), dim=1))
    result = torch.cummin(source, dim=1, out=(out, idx))
    equal(result, torch.cummin(source.cpu(), 1))
    out = torch.empty(0, dtype=torch.float64, device='ruda')
    assert torch.maximum(source, source + 1, out=out) is out
    equal(out, (source.cpu() + 1).double())
    empty = torch.empty((2, 0), dtype=torch.int64, device='ruda')
    equal(source.scatter_reduce(1, empty, source[:, :0], 'prod', include_self=False), source.cpu())
    with pytest.raises(RuntimeError):
        source.scatter_reduce(1, torch.tensor([[3], [0]], device='ruda'), source[:, :1], 'sum')


def test_legacy_scatter_and_integer_mean():
    a = torch.tensor([[3, 1, 2], [2, 4, 0]], dtype=torch.int64)
    index = torch.tensor([[0, 1, 1], [2, 0, 0]])
    values = torch.tensor([[-4, 2, -5], [-1, -7, 2]], dtype=a.dtype)
    for reduce in ('add', 'multiply'):
        equal(torch.scatter(a.to('ruda'), 1, index.to('ruda'), values.to('ruda'), reduce=reduce),
              torch.scatter(a, 1, index, values, reduce=reduce))
        equal(torch.scatter(a.to('ruda'), 1, index.to('ruda'), 2, reduce=reduce), torch.scatter(a, 1, index, 2, reduce=reduce))
    equal(torch.scatter_reduce(a.to('ruda'), 1, index.to('ruda'), values.to('ruda'), 'mean'),
          torch.scatter_reduce(a, 1, index, values, 'mean'))
    out = torch.empty(0, dtype=a.dtype, device='ruda')
    assert torch.scatter(a.to('ruda'), 1, index.to('ruda'), values.to('ruda'), out=out) is out
    equal(out, torch.scatter(a, 1, index, values))
