import os
import pytest
import torch

from test_ordering_reduce_gpu import REAL, equal


@pytest.fixture(scope='module', autouse=True)
def native_backend():
    if os.environ.get('RUDA_REQUIRE_GPU') != '1':
        pytest.skip('requires the native RUDA GPU backend')
    import ruda_torch
    assert ruda_torch._ordering_available, 'build and load the Rust/C++ ordering API 1'


@pytest.mark.parametrize('dtype', REAL)
@pytest.mark.parametrize('length', [1, 3, 4, 5, 8, 31, 32, 33, 255, 256, 257, 513])
def test_native_ordering_boundaries(dtype, length, monkeypatch):
    from ruda_torch import _C
    original = _C.ordering_call
    calls = []

    def dispatch(kind, tensors, integers):
        calls.append(kind)
        return original(kind, tensors, integers)

    monkeypatch.setattr(_C, 'ordering_call', dispatch)
    source = (torch.arange(length * 3).reshape(length, 3) % 17).to(dtype)
    if dtype.is_floating_point and length > 4:
        source[1, 0] = float('nan')
        source[-1, 0] = float('nan')
    if dtype == torch.int64:
        source += 2**53 + 3
    device = source.to('ruda')
    for operation in (torch.min, torch.max, torch.cummin, torch.cummax):
        equal(operation(device, dim=0), operation(source, dim=0))
    for operation in (torch.amin, torch.amax):
        equal(operation(device, dim=(0, 1)), operation(source, dim=(0, 1)))
    assert calls.count(0) == 4 and calls.count(1) == 2
    if dtype != torch.bool:
        boundaries = source.t().sort(dim=-1).values.contiguous()
        values = source[:min(length, 7)].t().contiguous()
        for right in (False, True):
            for int32 in (False, True):
                equal(torch.searchsorted(boundaries.to('ruda'), values.to('ruda'), right=right, out_int32=int32),
                      torch.searchsorted(boundaries, values, right=right, out_int32=int32))
        assert calls.count(2) == 4


def test_native_empty_scalar_and_strided_outputs():
    for shape, axis in (((0, 3), 1), ((3, 0), 0), ((2, 0, 3), 2)):
        a = torch.empty(shape)
        for operation in (torch.min, torch.max, torch.cummin, torch.cummax):
            equal(operation(a.to('ruda'), dim=axis), operation(a, dim=axis))
    for shape in ((0,), (3, 0), (0, 4)):
        boundaries = torch.empty(shape)
        values = torch.zeros((7,) if len(shape) == 1 else (shape[0], 7))
        equal(torch.searchsorted(boundaries.to('ruda'), values.to('ruda')), torch.searchsorted(boundaries, values))
    a = torch.tensor([[2., 2., 0.], [0., -0., 3.]], device='ruda').t()
    values = torch.empty((3, 4), device='ruda')[:, ::2]
    indices = torch.empty((3, 4), device='ruda', dtype=torch.int64)[:, ::2]
    equal(torch.cummax(a, 0, out=(values, indices)), torch.cummax(a.cpu(), 0))
    equal(torch.searchsorted(torch.tensor([1., 3.], device='ruda'), 2.), torch.tensor(1))


@pytest.mark.parametrize('length', [0, 1, 33, 256, 257])
def test_native_reduce_identities_and_integer_overflow(length):
    from ruda_torch import _C
    source = ((torch.arange(3 * length).reshape(3, length) % 5) - 2).long()
    if length:
        source[:, 0] = 2**62 + 7
    device = source.to('ruda')
    for code, operation in [(6, torch.any), (7, torch.all), (8, torch.sum), (9, torch.prod)]:
        expected = operation(source, dim=1)
        out = torch.empty(expected.shape, dtype=expected.dtype, device='ruda')
        _C.ordering_call(0, [device, out], [code, 2, 0])
        equal(out, expected)


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex64, torch.complex128])
@pytest.mark.parametrize('integer', [torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool])
def test_mixed_true_division(dtype, integer):
    numerator = torch.tensor([2., -4., 7.], dtype=dtype)
    denominator = torch.tensor([1, 2, 3], dtype=integer)
    a, b = numerator.to('ruda'), denominator.to('ruda')
    equal(a / b, numerator / denominator)
    assert a.div_(b) is a
    equal(a, numerator / denominator)
    equal(b / numerator.to('ruda'), denominator / numerator)
