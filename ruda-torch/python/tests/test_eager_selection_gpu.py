import os

import pytest
import torch

if os.environ.get('RUDA_REQUIRE_GPU') != '1':
    pytest.skip('requires the native RUDA GPU backend', allow_module_level=True)

import ruda_torch as ruda
from ruda_torch import (_additive_reduce, _arithmetic, _bitwise, _conditional_select,
                        _division, _exponential, _ops, _ordering, _predicates)


def same(actual, expected):
    if isinstance(expected, (tuple, list)):
        for a, b in zip(actual, expected):
            same(a, b)
        return
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.stride() == expected.stride()
    a, b = (value.cpu().contiguous().reshape(-1).view(torch.uint8) for value in (actual, expected))
    assert torch.equal(a, b)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16, torch.float64,
                                  torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool])
@pytest.mark.parametrize('layout', ['contiguous', 'transpose', 'offset', 'empty', 'scalar'])
def test_eager_selection_preserves_original_storage_and_bits(dtype, layout):
    source = (torch.arange(49).reshape(7, 7) % 11).to(dtype)
    if dtype.is_floating_point:
        source[0, :4] = torch.tensor([float('nan'), float('inf'), -float('inf'), -0.], dtype=dtype)
    native = source.to('ruda')
    if layout == 'transpose':
        native = native.t()
    elif layout == 'offset':
        native = native[1:, 1:]
    elif layout == 'empty':
        native = native[:0]
    elif layout == 'scalar':
        native = native[1, 1]
    mask = (torch.arange(native.numel()).reshape(native.shape) % 3 == 0).to('ruda')
    same(native.clone(), _ops.clone(native))
    same(native.masked_fill(mask, 1), _conditional_select.masked_fill(native, mask, 1))
    for operation, code in [(torch.eq, 78), (torch.ne, 79), (torch.lt, 80),
                            (torch.le, 81), (torch.gt, 82), (torch.ge, 83)]:
        same(operation(native, native), _predicates.predicate(native, native, op=code))
    if dtype.is_floating_point:
        same(native.exp(), _exponential.unary(native, op=0))
        same(native.sum(dim=-1), _additive_reduce.sum_reduce(native, -1))
    else:
        same(~native, _bitwise.elementwise(native, op=92))
    if native.numel():
        for operation, maximum in [(torch.max, True), (torch.min, False)]:
            for keepdim in (False, True):
                same(operation(native, dim=-1, keepdim=keepdim),
                     _ordering.extreme_dim(native, -1, keepdim, maximum=maximum))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16, torch.float64,
                                  torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8])
@pytest.mark.parametrize('mode', [None, 'floor', 'trunc'])
def test_eager_division_matches_original(dtype, mode):
    host = torch.tensor([[-7, -3, 0, 8], [11, 2, 1, 9]], dtype=torch.int64).to(dtype).t()
    a = host.to('ruda')
    b = torch.tensor([[2, 3]], dtype=dtype).to('ruda')
    for divisor in (b, 3, torch.tensor(3)):
        same(torch.div(a, divisor, rounding_mode=mode), _division.divide(a, divisor, rounding_mode=mode))
    same(torch.floor_divide(a, 3), _division.divide(a, 3, rounding_mode='floor'))


@pytest.mark.parametrize('dtype', [torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64])
def test_integer_scalar_cpp_dispatch_preserves_fallback(dtype, monkeypatch):
    a = (torch.arange(12).reshape(3, 4) % 7).to(dtype).to('ruda')
    scalar = True if dtype == torch.bool else 3
    expected = [_arithmetic.arithmetic(a, scalar, op=op) for op in (0, 2)]
    def forbidden(*args, **kwargs):
        raise AssertionError('common integer scalar path entered Python')
    with monkeypatch.context() as patch:
        patch.setattr(_arithmetic, 'arithmetic', forbidden)
        same(torch.add(a, scalar), expected[0])
        same(torch.mul(a, scalar), expected[1])
        same(torch.add(torch.tensor(scalar), a), expected[0])
    original = _arithmetic.arithmetic
    calls = []
    def observed(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(_arithmetic, 'arithmetic', observed)
    ruda._C.initialize_eager_integer_scalar(False)
    try:
        same(torch.add(a, scalar), expected[0])
        assert calls
    finally:
        ruda._C.initialize_eager_integer_scalar(True)


def test_eager_selection_common_paths_avoid_python(monkeypatch):
    a = torch.arange(24, dtype=torch.float32).reshape(3, 8).div(8).to('ruda')
    b = torch.full((3, 1), 2., device='ruda')
    mask = (torch.arange(24).reshape(3, 8) % 3 == 0).to('ruda')
    indices = torch.arange(8, dtype=torch.int64).to('ruda')
    lengths = torch.tensor([[3], [5], [7]], dtype=torch.int32).to('ruda')
    operations = [lambda: a.clone(), lambda: a.exp(), lambda: a.masked_fill(mask, -float('inf')),
                  lambda: ~mask, lambda: a.sum(dim=1, keepdim=True), lambda: a.max(dim=1),
                  lambda: a.min(dim=1), lambda: a / b, lambda: indices < lengths, lambda: lengths // 3]
    expected = [operation() for operation in operations]
    def forbidden(*args, **kwargs):
        raise AssertionError('common eager path entered Python')
    for module, name in [(_ops, 'clone'), (_exponential, 'unary'), (_conditional_select, 'masked_fill'),
                         (_bitwise, 'elementwise'), (_additive_reduce, 'sum_reduce'), (_ordering, 'extreme_dim'),
                         (_division, 'divide'), (_predicates, 'predicate')]:
        monkeypatch.setattr(module, name, forbidden)
    before = ruda.execution_stats()
    actual = [operation() for operation in operations]
    after = ruda.execution_stats()
    assert after['host_to_device_bytes'] == before['host_to_device_bytes']
    for value, reference in zip(actual, expected):
        same(value, reference)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_eager_weighted_reduction_gradients(dtype):
    torch.manual_seed(3261030)
    values = torch.randn(2, 3, 17, 11, dtype=dtype)
    logits = torch.randn(2, 3, 17, dtype=dtype)
    mask = torch.rand(2, 1, 17) > .25
    def evaluate(device):
        x, lse = [value.to(device).requires_grad_() for value in (values, logits)]
        shifted = lse.masked_fill(~mask.to(device), -float('inf'))
        weights = (shifted - shifted.max(-1, keepdim=True).values).exp()
        output = (x * weights.unsqueeze(-1)).sum(2) / weights.sum(2, keepdim=True)
        gradients = torch.autograd.grad(output.float().square().sum(), (x, lse))
        return (output, *gradients)
    expected, actual = evaluate('cpu'), evaluate('ruda')
    tolerance = {torch.float32: 2e-5, torch.float16: .008, torch.bfloat16: .05}[dtype]
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value.cpu(), reference, rtol=tolerance, atol=tolerance, equal_nan=True)
