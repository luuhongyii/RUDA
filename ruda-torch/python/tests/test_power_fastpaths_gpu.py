import os

import pytest
import torch


@pytest.fixture(scope='module')
def backend():
    if os.environ.get('RUDA_REQUIRE_GPU') != '1':
        pytest.skip('requires explicit native GPU execution')
    import ruda_torch
    return ruda_torch


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize('operation', ['sqrt', 'rsqrt', 'reciprocal', 'square', 'cube', 'inverse_square', 'fraction', 'tensor', 'broadcast'])
def test_power_packed_preserves_strided_bits(backend, dtype, operation):
    bits_dtype = {torch.float16: torch.int16, torch.bfloat16: torch.int16,
                  torch.float32: torch.int32, torch.float64: torch.int64}[dtype]
    if dtype in (torch.float16, torch.bfloat16):
        values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype)
    else:
        values = torch.linspace(-8, 8, 65536, dtype=dtype)
        values[:8] = torch.tensor([0., -0., float('inf'), -float('inf'), float('nan'), 1e-30, -1e-30, 1e30], dtype=dtype)

    for length in [17, 1024, 1031, 65536]:
        cpu = values[:length]
        packed = torch.cat((cpu[:1], cpu)).to('ruda')[1:]
        strided = cpu.repeat_interleave(2).to('ruda')[::2]
        exponent = torch.linspace(.25, 3.5, length).to(dtype).to('ruda')

        def invoke(x, out=None):
            if operation in ('sqrt', 'rsqrt', 'reciprocal', 'square'):
                return getattr(torch, operation)(x, out=out)
            power = {'cube': 3., 'inverse_square': -2., 'fraction': .7}.get(operation)
            if operation == 'tensor':
                power = exponent
            elif operation == 'broadcast':
                power = exponent[:1].expand(length)
            return torch.pow(x, power, out=out)

        expected = invoke(strided).cpu().view(bits_dtype)
        actual = invoke(packed).cpu().view(bits_dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        output = torch.empty(length * 2, dtype=dtype, device='ruda')[::2]
        invoke(packed, out=output)
        torch.testing.assert_close(output.cpu().view(bits_dtype), expected, rtol=0, atol=0)
        invoke(packed, out=packed)
        torch.testing.assert_close(packed.cpu().view(bits_dtype), expected, rtol=0, atol=0)
