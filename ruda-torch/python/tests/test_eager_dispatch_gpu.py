import os

import pytest
import torch

if os.environ.get('RUDA_REQUIRE_GPU') != '1':
    pytest.skip('requires a real GPU', allow_module_level=True)

import ruda_torch


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_native_matrix_dispatch_without_python_wrapper(dtype, monkeypatch):
    from ruda_torch import _ops

    def forbidden(*args, **kwargs):
        raise AssertionError('eligible matrix operation entered its Python wrapper')

    for name in ['mm', 'bmm', 'addmm']:
        monkeypatch.setattr(_ops, name, forbidden)
    torch.manual_seed(994)
    a, w, bias = [torch.randn(shape).to(dtype) for shape in [(2, 17, 32), (2, 29, 32), (29,)]]
    x, weight, b = [value.to('ruda') for value in [a, w, bias]]
    ca, cw, cb = [value.cuda() for value in [a, w, bias]]
    tolerance = .04 if dtype == torch.bfloat16 else .004 if dtype == torch.float16 else 2e-5
    with torch.inference_mode():
        torch.testing.assert_close(torch.mm(x[0], weight[0].t()).cpu(), torch.mm(ca[0], cw[0].t()).cpu(), atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(torch.bmm(x, weight.transpose(1, 2)).cpu(), torch.bmm(ca, cw.transpose(1, 2)).cpu(), atol=tolerance, rtol=tolerance)
        for alpha, beta in [(1., 1.), (.75, -.25), (0., 1.), (1., 0.)]:
            actual = torch.addmm(b, x[0], weight[0].t(), alpha=alpha, beta=beta)
            expected = torch.addmm(cb, ca[0], cw[0].t(), alpha=alpha, beta=beta)
            torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('strided', [False, True])
def test_native_activation_dispatch_without_python_wrapper(dtype, strided, monkeypatch):
    from ruda_torch import _activation

    def forbidden(*args, **kwargs):
        raise AssertionError('eligible activation entered its Python wrapper')

    monkeypatch.setattr(_activation, '_run', forbidden)
    values = torch.linspace(-8, 8, 192).reshape(8, 24).to(dtype)
    x, expected = values.to('ruda'), values.cuda()
    if strided:
        x, expected = x.t()[::2], expected.t()[::2]
    tolerance = .02 if dtype == torch.bfloat16 else .002 if dtype == torch.float16 else 2e-6
    with torch.inference_mode():
        for function in [torch.nn.functional.silu,
                         lambda value: torch.nn.functional.gelu(value, approximate='none'),
                         lambda value: torch.nn.functional.gelu(value, approximate='tanh')]:
            torch.testing.assert_close(function(x).cpu(), function(expected).cpu(), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('shape', [(2, 3, 16), (3, 17)])
@pytest.mark.parametrize('affine', ['both', 'weight', 'bias', 'none'])
def test_native_layer_norm_preserves_python_results(dtype, shape, affine, monkeypatch):
    from ruda_torch import _normalization
    torch.manual_seed(995)
    x = torch.randn(shape).to(dtype).to('ruda').requires_grad_()
    normalized = shape[-2:] if len(shape) == 3 else shape[-1:]
    weight = torch.randn(normalized).to(dtype).to('ruda').requires_grad_() if affine in ('both', 'weight') else None
    bias = torch.randn(normalized).to(dtype).to('ruda').requires_grad_() if affine in ('both', 'bias') else None
    expected = _normalization.layer_norm(x, normalized, weight, bias, 1e-5)

    def forbidden(*args, **kwargs):
        raise AssertionError('eligible layer normalization entered its Python wrapper')

    monkeypatch.setattr(_normalization, 'layer_norm', forbidden)
    actual = torch.ops.aten.native_layer_norm(x, normalized, weight, bias, 1e-5)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value.detach().cpu(), reference.detach().cpu(), rtol=0, atol=0)
    gradient = torch.randn(shape).to(dtype).to('ruda')
    parameters = tuple(value for value in (x, weight, bias) if value is not None)
    actual_gradients = torch.autograd.grad(actual[0], parameters, gradient)
    expected_gradients = _normalization.layer_backward(gradient, x, normalized, expected[1], expected[2], weight, bias,
                                                      (True, weight is not None, bias is not None))
    for value, reference in zip(actual_gradients, (value for value in expected_gradients if value is not None)):
        torch.testing.assert_close(value.cpu(), reference.cpu(), rtol=0, atol=0)


def test_native_matrix_dispatch_preserves_context_backend(monkeypatch):
    from contextvars import copy_context
    from ruda_torch import _vendor_blas
    from ruda_torch.math_backend import math_backend
    a, b = torch.ones(3, 4, device='ruda'), torch.ones(4, 2, device='ruda')
    output = torch.zeros(3, 2, device='ruda')
    calls = []

    def vendor(*args, **kwargs):
        calls.append((args, kwargs))
        return output

    monkeypatch.setattr(_vendor_blas, 'gemm', vendor)
    with math_backend('nvidia'):
        context = copy_context()
        assert torch.mm(a, b) is output
        assert torch.addmm(output, a, b) is output
        assert torch.bmm(a.unsqueeze(0), b.unsqueeze(0)) is output
    assert context.run(torch.mm, a, b) is output
    with math_backend('ruda'):
        torch.testing.assert_close(torch.mm(a, b).cpu(), torch.full((3, 2), 4.))
    assert len(calls) == 4


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex64])
def test_native_matrix_dispatch_preserves_wide_fallback(dtype):
    a = torch.arange(12).reshape(3, 4).to(dtype) / 8
    b = torch.arange(8).reshape(4, 2).to(dtype) / 4
    torch.testing.assert_close(torch.mm(a.to('ruda'), b.to('ruda')).cpu(), a @ b)


@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('singleton', [False, True])
def test_embedding_checked_readback_on_selected_stream(dtype, singleton, monkeypatch):
    monkeypatch.delenv('RUDA_TORCH_INDEX_CHECK', raising=False)
    stream = ruda_torch.Stream()
    with ruda_torch.stream(stream):
        weight = torch.arange(35, dtype=torch.float32).reshape(7, 5).to('ruda')
        indices = torch.tensor([6] if singleton else [0, 6, 2], dtype=dtype).to('ruda')
        expected = torch.nn.functional.embedding(indices.cpu(), weight.cpu())
        torch.testing.assert_close(torch.nn.functional.embedding(indices, weight).cpu(), expected)
        for invalid in [-1, 7]:
            indices.fill_(invalid)
            with pytest.raises(RuntimeError, match='index out of bounds'):
                torch.nn.functional.embedding(indices, weight)
        indices.fill_(3)
        torch.testing.assert_close(torch.nn.functional.embedding(indices, weight).cpu(),
                                   torch.arange(15, 20, dtype=torch.float32).expand(indices.numel(), 5))


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize('offset', [0, 1])
@pytest.mark.parametrize('length', [1024, 1027])
def test_contiguous_activation_addressing(dtype, offset, length):
    values = torch.linspace(-9, 9, length + offset).to(dtype)
    device = values.to('ruda')[offset:]
    reference = values[offset:].cuda()
    assert device.is_contiguous()
    tolerance = .02 if dtype == torch.bfloat16 else .002 if dtype == torch.float16 else 3e-6 if dtype == torch.float32 else 4e-12
    for operation in [torch.nn.functional.silu,
                      lambda x: torch.nn.functional.gelu(x, approximate='none'),
                      lambda x: torch.nn.functional.gelu(x, approximate='tanh')]:
        torch.testing.assert_close(operation(device).cpu(), operation(reference).cpu(), atol=tolerance, rtol=tolerance)
