"""Large-vocabulary BF16 CE against an independent FP32 CUDA loss oracle."""
import subprocess
import sys

import pytest
import torch
from ruda_torch import fused_cross_entropy as ce


@pytest.fixture(scope='module')
def cases(tmp_path_factory):
    directory = tmp_path_factory.mktemp('cross_entropy_cuda_precision')
    inputs = []
    for width in (32768, 32772, 65536, 128256, 151936):
        torch.manual_seed(261027 + width)
        x = (torch.randn(7, width) * 4).bfloat16()
        target = torch.tensor([0, width - 1, -100, 17, width // 2, 3, 117])
        grad = torch.tensor([1 / 511, -.375, .7, .03125, 1 / 511, .123, -.25])
        inputs.append(dict(x=x, target=target, grad=grad))
    for scale, shift in ((60., 0.), (1., 4096.)):
        x = (torch.randn(3, 32768) * scale + shift).bfloat16()
        inputs.append(dict(x=x, target=torch.tensor([0, 19, 32767]),
                           grad=torch.full((3,), 1 / 511)))
    x = torch.zeros(4, 32768, dtype=torch.bfloat16)
    x[0, 0] = float('inf')
    x[1].fill_(float('-inf'))
    x[2, 7] = float('nan')
    inputs.append(dict(x=x, target=torch.tensor([0, 7, 3, -100]), grad=torch.ones(4)))
    torch.save(inputs, directory / 'inputs.pt')
    code = '''
import sys, torch
from pathlib import Path
from torch.nn import functional as F
torch.set_num_threads(4)
directory = Path(sys.argv[1]); results = []
for case in torch.load(directory / 'inputs.pt', weights_only=True):
    x = case['x'].cuda().requires_grad_()
    target = case['target'].cuda()
    loss = F.cross_entropy(x.float(), target, reduction='none')
    loss.backward(case['grad'].cuda())
    maximum = x.float().amax(-1)
    lp = torch.log_softmax(x.float(), -1)
    logsum = -lp.gather(1, x.float().argmax(-1, keepdim=True)).flatten() + 0.
    stats = torch.stack((maximum, logsum), dim=1)
    stats[target == -100] = 0
    results.append(dict(loss=loss.detach().cpu(), dx=x.grad.cpu(), stats=stats.cpu()))
torch.save(results, directory / 'expected.pt')
'''
    subprocess.run([sys.executable, '-B', '-c', code, str(directory)], check=True)
    return inputs, torch.load(directory / 'expected.pt', weights_only=True)


@pytest.mark.parametrize('index', range(8))
def test_cuda_precision_loss_and_gradients(cases, index, monkeypatch):
    monkeypatch.setenv('RUDA_TORCH_BF16_PRECISION', 'cuda')
    inputs, references = cases
    case, reference = inputs[index], references[index]
    storage = torch.empty(case['x'].numel() + 3, dtype=torch.bfloat16, device='ruda')
    x = storage[3:].view_as(case['x'])
    x.copy_(case['x'].to('ruda'))
    target = case['target'].to('ruda')
    loss, stats = ce._forward(x, target, -100)
    dx = ce._backward(x, target, stats, case['grad'].to('ruda'), -100)
    torch.testing.assert_close(loss.cpu(), reference['loss'], rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(dx.cpu(), reference['dx'], rtol=0, atol=0, equal_nan=True)
    if index < 7:
        assert torch.equal(stats.cpu().view(torch.int32), reference['stats'].view(torch.int32))


@pytest.mark.parametrize('target', [-1, 32768, 2**40])
def test_cuda_precision_target_bounds(target, monkeypatch):
    monkeypatch.setenv('RUDA_TORCH_BF16_PRECISION', 'cuda')
    x = torch.zeros(1, 32768, dtype=torch.bfloat16, device='ruda')
    with pytest.raises(RuntimeError, match='out of bounds'):
        ce._forward(x, torch.tensor([target], device='ruda'), -100)


def test_cuda_precision_ignored_rows_and_broadcast_grad(monkeypatch):
    monkeypatch.setenv('RUDA_TORCH_BF16_PRECISION', 'cuda')
    x = torch.randn(7, 32768, dtype=torch.bfloat16).to('ruda')
    target = torch.full((7,), -23, device='ruda', dtype=torch.int64)
    loss, stats = ce._forward(x, target, -23)
    dx = ce._backward(x, target, stats, torch.tensor(.375, device='ruda').expand(7), -23)
    for value in (loss, stats, dx):
        assert torch.count_nonzero(value.cpu()) == 0
