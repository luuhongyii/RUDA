"""CUDA SiLU rounding over every BF16 input encoding and fused training boundaries."""
import subprocess
import sys

import pytest
import torch
import ruda_torch as r


@pytest.fixture(scope='module')
def cuda_silu_cases(tmp_path_factory):
    directory = tmp_path_factory.mktemp('cuda_silu_precision')
    torch.manual_seed(20261026)
    encodings = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    cases = []
    for scale in (0.125, 1., 8.):
        gate = encodings.repeat(4)
        up = (torch.randn(gate.numel()) * scale).to(torch.bfloat16)
        dy = (torch.randn(gate.numel()) / scale).to(torch.bfloat16)
        cases.append(dict(gate=gate, up=up, dy=dy))
    cases.append(dict(gate=torch.randn(3, 257, dtype=torch.bfloat16),
                      up=torch.randn(3, 257, dtype=torch.bfloat16),
                      dy=torch.randn(3, 257, dtype=torch.bfloat16)))
    torch.save(cases, directory/'inputs.pt')
    code = '''
import sys,torch
from pathlib import Path
directory=Path(sys.argv[1]);results=[]
for case in torch.load(directory/'inputs.pt',weights_only=True):
    gate=case['gate'].cuda().requires_grad_();up=case['up'].cuda().requires_grad_()
    y=torch.nn.functional.silu(gate)*up;y.backward(case['dy'].cuda())
    results.append(dict(y=y.detach().cpu(),dg=gate.grad.cpu(),du=up.grad.cpu()))
torch.save(results,directory/'expected.pt')
'''
    subprocess.run([sys.executable, '-B', '-c', code, str(directory)],
                   check=True, capture_output=True, text=True)
    return cases, torch.load(directory/'expected.pt', weights_only=True)


def exact_except_nan_payload(actual, expected):
    actual = actual.cpu()
    nan = torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), nan)
    assert torch.equal(actual[~nan].view(torch.int16), expected[~nan].view(torch.int16))


@pytest.mark.parametrize('index', range(4))
@pytest.mark.parametrize('needs', [(True, True), (True, False), (False, True)])
def test_silu_cuda_bf16_exact(cuda_silu_cases, index, needs, monkeypatch):
    monkeypatch.setenv('RUDA_TORCH_BF16_PRECISION', 'cuda')
    cases, expected = cuda_silu_cases
    case, oracle = cases[index], expected[index]
    gate = case['gate'].to('ruda').requires_grad_(needs[0])
    up = case['up'].to('ruda').requires_grad_(needs[1])
    y = r.silu_mul(gate, up)
    y.backward(case['dy'].to('ruda'))
    exact_except_nan_payload(y, oracle['y'])
    if needs[0]: exact_except_nan_payload(gate.grad, oracle['dg'])
    if needs[1]: exact_except_nan_payload(up.grad, oracle['du'])


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
def test_silu_cuda_option_preserves_other_dtypes(dtype, monkeypatch):
    torch.manual_seed(307)
    values = [torch.randn(257, dtype=dtype) for _ in range(3)]
    outputs = []
    for mode in ('fp32', 'cuda'):
        monkeypatch.setenv('RUDA_TORCH_BF16_PRECISION', mode)
        gate, up = [x.to('ruda').requires_grad_() for x in values[:2]]
        y = r.silu_mul(gate, up); y.backward(values[2].to('ruda'))
        outputs.append([x.detach().cpu() for x in (y, gate.grad, up.grad)])
    for actual, expected in zip(*outputs):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
