import os

import pytest
import torch


@pytest.fixture(scope='module', autouse=True)
def backend():
    if os.environ.get('RUDA_REQUIRE_GPU') != '1':
        pytest.skip('requires the native RUDA GPU backend')
    import ruda_torch
    assert ruda_torch._C.abi_version == 10


DTYPES = [torch.float32, torch.float16, torch.bfloat16, torch.float64,
          torch.complex64, torch.complex128, torch.int64, torch.int32,
          torch.int16, torch.int8, torch.uint8, torch.bool]


def data(dtype, shape=(3, 4)):
    value = torch.arange(torch.Size(shape).numel()).reshape(shape) % 7
    if dtype.is_complex:
        value = torch.complex(value.float(), (value % 3).float())
    return value.to(dtype)


def equal(actual, expected):
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize('dtype', DTYPES)
def test_nonzero_and_masked_selection(dtype):
    source = data(dtype).t()
    mask = torch.tensor([[True], [False], [True], [False]])
    device = source.to('ruda')
    equal(torch.nonzero(device), torch.nonzero(source))
    for actual, expected in zip(torch.nonzero(device, as_tuple=True), torch.nonzero(source, as_tuple=True)):
        equal(actual, expected)
    equal(torch.masked_select(device, mask.to('ruda')), torch.masked_select(source, mask))
    equal(device[mask[:, 0].to('ruda')], source[mask[:, 0]])
    equal(device[mask[:, 0]], source[mask[:, 0]])
    for value in (source.reshape(-1)[0], source.reshape(-1)[1], source[:0]):
        equal(value.to('ruda').nonzero(), value.nonzero())
    zeros = torch.zeros_like(device)
    equal(zeros.nonzero(), torch.zeros_like(source).nonzero())


@pytest.mark.parametrize('dtype', DTYPES)
def test_take_and_put_strided_negative_duplicate(dtype):
    source = data(dtype).t()
    device = source.to('ruda')
    indices = torch.tensor([[0, -1, 3], [3, -6, 2]])
    values = data(dtype, (6,))
    equal(torch.take(device, indices.to('ruda')), torch.take(source, indices))
    equal(torch.put(device, indices.to('ruda'), values.to('ruda'), accumulate=True),
          torch.put(source, indices, values, accumulate=True))
    unique = indices.flatten()[:3]
    expected = source.clone().put_(unique, values[:3])
    pointer = device.data_ptr()
    assert device.put_(unique.to('ruda'), values[:3].to('ruda')) is device
    assert device.data_ptr() == pointer
    equal(device, expected)


@pytest.mark.parametrize('dtype', DTYPES)
def test_index_copy_and_fill_strided(dtype):
    source = data(dtype).t()
    device = source.to('ruda')
    index = torch.tensor([2, 0])
    updates = data(dtype, (4, 2))
    equal(torch.index_copy(device, 1, index.to('ruda'), updates.to('ruda')),
          torch.index_copy(source, 1, index, updates))
    negative = torch.tensor([-1, 0])
    equal(torch.index_fill(device, -1, negative.to('ruda'), 1),
          torch.index_fill(source, -1, negative, 1))
    assert device.index_copy_(1, index.to('ruda'), updates.to('ruda')) is device
    expected = source.clone().index_copy_(1, index, updates)
    equal(device, expected)
    assert device.index_fill_(0, torch.tensor([1]).to('ruda'), torch.tensor(0)) is device
    equal(device, expected.index_fill_(0, torch.tensor([1]), torch.tensor(0)))


@pytest.mark.parametrize('dtype', DTYPES)
def test_masked_scatter_broadcast_and_inplace(dtype):
    source = data(dtype, (1, 3))
    mask = torch.tensor([[True, False, True], [False, True, False]])
    values = data(dtype, (5,))
    equal(torch.masked_scatter(source.to('ruda'), mask.to('ruda'), values.to('ruda')),
          torch.masked_scatter(source, mask, values))
    expected = data(dtype, (3, 2)).t()
    actual = expected.to('ruda')
    assert actual.masked_scatter_(mask.to('ruda'), values.to('ruda')) is actual
    equal(actual, expected.masked_scatter_(mask, values))


@pytest.mark.parametrize('dtype', DTYPES)
def test_index_put_broadcast_separated_axes_and_masks(dtype):
    source = data(dtype, (3, 4, 5)).transpose(0, 2)
    left = torch.tensor([[0], [-1]])
    right = torch.tensor([[0, 1]])
    values = data(dtype, (2, 2, 4))
    for accumulate in (False, True):
        expected = torch.ops.aten.index_put(source, [left, None, right], values, accumulate)
        actual = torch.ops.aten.index_put(source.to('ruda'), [left, None, right.to('ruda')], values.to('ruda'), accumulate)
        equal(actual, expected)
    mask = torch.tensor([[True, False, False, True], [False, True, False, False],
                         [False, False, False, False], [True, False, True, False],
                         [False, False, False, False]])
    expected = source.clone()
    actual = source.to('ruda')
    expected[mask] = 1
    actual[mask.to('ruda')] = 1
    equal(actual, expected)
    repeated = torch.tensor([0, 0, -1])
    repeated_values = data(dtype, (3, 4, 3))
    equal(torch.index_put(source.to('ruda'), (repeated.to('ruda'),), repeated_values.to('ruda'), accumulate=True),
          torch.index_put(source, (repeated,), repeated_values, accumulate=True))


@pytest.mark.parametrize('operation', ['take', 'put', 'copy', 'fill', 'select', 'scatter', 'index', 'index_put'])
def test_autograd_matches_independent_pytorch(operation):
    def run(device):
        a = torch.arange(12, dtype=torch.float64).reshape(3, 4).to(device).requires_grad_()
        b = torch.arange(6, dtype=torch.float64).reshape(3, 2).to(device).requires_grad_()
        index = torch.tensor([0, 2], device=device)
        mask = torch.tensor([[True, False, True, False]], device=device)
        if operation == 'take':
            result = a.take(index)
        elif operation == 'put':
            result = a.put(index, b[0], accumulate=True)
        elif operation == 'copy':
            result = a.index_copy(1, index, b)
        elif operation == 'fill':
            result = a.index_fill(1, index, b[0, 0])
        elif operation == 'select':
            result = a.masked_select(mask)
        elif operation == 'scatter':
            result = a.masked_scatter(mask, b)
        elif operation == 'index':
            result = a[:, mask[0]]
        else:
            result = torch.ops.aten.index_put(a, [None, index], b, True)
        weights = torch.arange(result.numel(), dtype=result.dtype, device=device).reshape(result.shape) + 1
        (result * weights).sum().backward()
        return result, a.grad, b.grad
    actual, expected = run('ruda'), run('cpu')
    for value, reference in zip(actual, expected):
        assert (value is None) == (reference is None)
        if value is not None:
            equal(value.detach(), reference.detach())


def test_out_storage_empty_scalar_and_errors():
    a = torch.arange(12).reshape(3, 4).to('ruda')
    indices = torch.tensor([0, -1], device='ruda')
    out = torch.empty(0, dtype=a.dtype, device='ruda')
    assert torch.take(a, indices, out=out) is out
    equal(out, torch.tensor([0, 11]))
    storage = torch.full((4,), -1, dtype=a.dtype, device='ruda')
    torch.take(a, indices, out=storage[::2])
    equal(storage, torch.tensor([0, -1, 11, -1]))
    assert torch.nonzero(a, out=out) is out
    equal(out, torch.arange(12).reshape(3, 4).nonzero())
    mask = a > 3
    out = torch.empty(0, dtype=a.dtype, device='ruda')
    assert torch.masked_select(a, mask, out=out) is out
    equal(out, torch.arange(4, 12))
    empty = torch.empty(0, dtype=torch.int64, device='ruda')
    equal(a.take(empty), torch.empty(0, dtype=a.dtype))
    equal(a.index_put((empty,), torch.empty((0, 4), dtype=a.dtype, device='ruda')), a.cpu())
    scalar = torch.tensor(5, device='ruda')
    equal(scalar.index_fill(0, torch.tensor([-1], device='ruda'), 9), torch.tensor(9))
    for actual, expected in zip(scalar.nonzero(as_tuple=True), torch.tensor(5).nonzero(as_tuple=True)):
        equal(actual, expected)
    for invalid in (-13, 12):
        with pytest.raises(RuntimeError):
            a.take(torch.tensor([invalid], device='ruda'))
    with pytest.raises(RuntimeError):
        a.masked_scatter_(mask, torch.empty(1, dtype=a.dtype, device='ruda'))
    with pytest.raises(RuntimeError):
        a.put_(torch.tensor([0], device='ruda'), a.reshape(-1)[:1])
    with pytest.raises((RuntimeError, IndexError)):
        a[torch.ones(2, dtype=torch.bool, device='ruda')]


def test_logical_complex_views_and_large_integers():
    a = torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex128).to('ruda')
    conjugate = a.conj()
    conjugate.put_(torch.tensor([0], device='ruda'), torch.tensor([5 + 6j], dtype=a.dtype, device='ruda'))
    equal(a, torch.tensor([5 - 6j, 3 - 4j], dtype=a.dtype))
    source = torch.tensor([2**53 + 1, 2**60 + 7, -(2**60) - 9])
    index = torch.tensor([2, 0, 1])
    equal(source.to('ruda').take(index.to('ruda')), source.take(index))


@pytest.mark.parametrize('dtype', [dtype for dtype in DTYPES if not dtype.is_complex])
@pytest.mark.parametrize('consecutive', [False, True])
def test_unique_values_inverse_counts_and_dimensions(dtype, consecutive):
    operation = torch.unique_consecutive if consecutive else torch.unique
    source = torch.tensor([[3, 3, 1, 2], [1, 1, 2, 2], [3, 3, 1, 2]], dtype=dtype).t()
    for dim in (None, 0, 1, -1):
        actual = operation(source.to('ruda'), dim=dim, return_inverse=True, return_counts=True)
        expected = operation(source, dim=dim, return_inverse=True, return_counts=True)
        for value, reference in zip(actual, expected):
            equal(value, reference)
    for value in (source[:0], torch.tensor(1, dtype=dtype)):
        actual = operation(value.to('ruda'), return_inverse=True, return_counts=True)
        expected = operation(value, return_inverse=True, return_counts=True)
        for result, reference in zip(actual, expected):
            equal(result, reference)


@pytest.mark.parametrize('operation', ['put', 'copy', 'fill', 'scatter', 'index_put', 'unique'])
def test_update_and_unique_out_variants(operation):
    source = torch.arange(12).reshape(3, 4)
    index = torch.tensor([0, 2])
    values = torch.arange(6).reshape(3, 2)
    mask = source % 2 == 0
    def run(a, i, v, m, out):
        if operation == 'put':
            return torch.ops.aten.put.out(a, i, v[0], out=out)
        if operation == 'copy':
            return torch.index_copy(a, 1, i, v, out=out)
        if operation == 'fill':
            return torch.ops.aten.index_fill.int_Scalar_out(a, 1, i, 5, out=out)
        if operation == 'scatter':
            return torch.ops.aten.masked_scatter.out(a, m, v, out=out)
        if operation == 'index_put':
            return torch.ops.aten.index_put.out(a, [None, i], v, out=out)
        extra = [torch.empty(0, dtype=torch.int64, device=a.device) for _ in range(2)]
        return torch.ops.aten._unique2.out(a % 5, True, True, True, out0=out, out1=extra[0], out2=extra[1])
    out = torch.empty(0, dtype=source.dtype, device='ruda')
    actual = run(source.to('ruda'), index.to('ruda'), values.to('ruda'), mask.to('ruda'), out)
    expected = run(source, index, values, mask, torch.empty(0, dtype=source.dtype))
    if operation == 'unique':
        assert actual[0] is out
        for value, reference in zip(actual, expected):
            equal(value, reference)
    else:
        assert actual is out
        equal(actual, expected)


def test_unique_special_values_and_duplicate_index_fill_gradient():
    values = torch.tensor([float('nan'), -0., 0., float('inf'), -float('inf'), float('nan'), 2., 2.], dtype=torch.float64)
    for operation in (torch.unique, torch.unique_consecutive):
        actual = operation(values.to('ruda'), return_inverse=True, return_counts=True)
        expected = operation(values, return_inverse=True, return_counts=True)
        for value, reference in zip(actual, expected):
            equal(value, reference)
    gradients = []
    for device in ('ruda', 'cpu'):
        a = torch.arange(12, dtype=torch.float64).reshape(3, 4).to(device).requires_grad_()
        value = torch.tensor(9., dtype=a.dtype, device=device, requires_grad=True)
        a.index_fill(1, torch.tensor([0, 0, 2], device=device), value).sum().backward()
        gradients.append((a.grad, value.grad))
    for value, reference in zip(*gradients):
        equal(value, reference)
