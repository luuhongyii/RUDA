import os

import pytest
import torch

if os.environ.get('RUDA_REQUIRE_GPU') != '1':
    pytest.skip('requires the native RUDA GPU backend', allow_module_level=True)

import ruda_torch as ruda


DTYPES = [torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]


def values(dtype):
    if dtype == torch.bool:
        return torch.tensor([[True, False, True], [False, True, False]])
    limits = torch.iinfo(dtype)
    return torch.tensor([[limits.min, limits.max, 0], [1, 7, 11]], dtype=dtype)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('transposed', [False, True])
@pytest.mark.parametrize('reverse', [False, True])
def test_integer_arithmetic_immediate_preserves_values_without_upload(dtype, transposed, reverse):
    source = values(dtype)
    if transposed:
        source = source.t()
    native = source.to('ruda')
    scalar = True if dtype == torch.bool else 3
    operations = [torch.add, torch.mul] if dtype == torch.bool else [torch.add, torch.sub, torch.mul]
    for operation in operations:
        left, right = (scalar, native) if reverse else (native, scalar)
        expected_left, expected_right = (scalar, source) if reverse else (source, scalar)
        before = ruda.execution_stats()
        actual = operation(left, right)
        after = ruda.execution_stats()
        assert after['host_to_device_bytes'] == before['host_to_device_bytes']
        assert after['device_to_host_bytes'] == before['device_to_host_bytes']
        torch.testing.assert_close(actual.cpu(), operation(expected_left, expected_right), rtol=0, atol=0)


@pytest.mark.parametrize('dtype', DTYPES[1:])
@pytest.mark.parametrize('divisor', [3, -3])
@pytest.mark.parametrize('operation', ['floor', 'trunc', 'remainder', 'fmod'])
def test_integer_division_immediate_preserves_rounding_without_upload(dtype, divisor, operation):
    if dtype == torch.uint8 and divisor < 0:
        divisor = 251
    source = values(dtype).t()
    native = source.to('ruda')
    def invoke(value):
        if operation in ('floor', 'trunc'):
            return torch.div(value, divisor, rounding_mode=operation)
        return getattr(torch, operation)(value, divisor)
    before = ruda.execution_stats()
    actual = invoke(native)
    after = ruda.execution_stats()
    assert after['host_to_device_bytes'] == before['host_to_device_bytes']
    assert after['device_to_host_bytes'] == before['device_to_host_bytes']
    torch.testing.assert_close(actual.cpu(), invoke(source), rtol=0, atol=0)


@pytest.mark.parametrize('dtype', DTYPES[1:])
def test_integer_immediate_inplace_and_strided_output(dtype):
    source = values(dtype)
    native = source.to('ruda')
    native.add_(3, alpha=2)
    source.add_(3, alpha=2)
    native.mul_(3)
    source.mul_(3)
    torch.testing.assert_close(native.cpu(), source, rtol=0, atol=0)
    output = torch.empty(3, 2, dtype=dtype, device='ruda').t()
    torch.sub(native, 2, out=output)
    torch.testing.assert_close(output.cpu(), torch.sub(source, 2), rtol=0, atol=0)
    native.div_(3, rounding_mode='floor')
    source.div_(3, rounding_mode='floor')
    torch.testing.assert_close(native.cpu(), source, rtol=0, atol=0)
