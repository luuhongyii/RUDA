import itertools
import os

import pytest
import torch

if not os.environ.get('RUDA_REQUIRE_GPU'):
    pytest.skip('set RUDA_REQUIRE_GPU=1 for real GPU coverage', allow_module_level=True)

import ruda_torch
import ruda_torch as r
from ruda_torch import _arithmetic, _ops

DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('layout', ['broadcast', 'transpose', 'strided_output', 'rank_five'])
def test_fast_broadcast_addressing(dtype, layout):
    a=(torch.arange(3*5*257).reshape(3,5,257)%17-8).to(dtype)/8
    b=(a+1).clone()
    x,y=a.to('ruda'),b.to('ruda')
    ca,cb=a.cuda(),b.cuda()
    if layout=='broadcast':
        x,ca=x[:,:1,:],ca[:,:1,:]
        y,cb=y[:1,:,:],cb[:1,:,:]
    elif layout=='transpose':
        x,y,ca,cb=[value.transpose(0,2) for value in [x,y,ca,cb]]
    elif layout=='rank_five':
        x,y,ca,cb=[value.reshape(3,1,5,1,257).expand(3,7,5,2,257) for value in [x,y,ca,cb]]
    for operation in [torch.add,torch.sub,torch.mul]:
        if layout=='strided_output':
            out=torch.empty(257,5,3,dtype=dtype,device='ruda').transpose(0,2)
            actual=operation(x,y,out=out)
        else:
            actual=operation(x,y)
        torch.testing.assert_close(actual.cpu(),operation(ca,cb).cpu(),rtol=0,atol=0)


_CONVERSION_DTYPES=(*DTYPES,torch.int8,torch.uint8,torch.int16,torch.int32,torch.int64)
@pytest.mark.parametrize('source,destination',[(a,b) for a in _CONVERSION_DTYPES for b in _CONVERSION_DTYPES if a!=b and (a in DTYPES or b in DTYPES)])
@pytest.mark.parametrize('offset',[0,1])
def test_vector_conversion_addressing(source,destination,offset):
    values=((torch.arange(1024+offset)%97)/8).to(source)
    device=values.to('ruda')[offset:]
    expected=values[offset:].cuda().to(destination)
    torch.testing.assert_close(device.to(destination).cpu(),expected.cpu(),rtol=0,atol=0)


@pytest.mark.parametrize('source,destination', itertools.product(DTYPES, repeat=2))
@pytest.mark.parametrize('offset', [0, 1])
@pytest.mark.parametrize('width', [260, 261])
def test_row_vector_strided_conversion(source, destination, offset, width):
    values=((torch.arange(3*7*width+offset)%97)-48).to(source)/8
    native=values.to('ruda')[offset:].reshape(3,7,width).transpose(0,1)[...,:256]
    expected=values[offset:].reshape(3,7,width).transpose(0,1)[...,:256].cuda().to(destination)
    out=torch.empty_strided(native.shape,(width,7*width,1),dtype=destination,device='ruda')
    out.copy_(native)
    torch.testing.assert_close(out.cpu(),expected.cpu(),rtol=0,atol=0)
    torch.testing.assert_close(native.to(destination).cpu(),expected.cpu(),rtol=0,atol=0)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_row_vector_copy_preserves_all_half_bits(dtype):
    values=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(dtype).reshape(256,256)
    native=values.to('ruda').reshape(16,16,256).transpose(0,1)
    expected=values.reshape(16,16,256).transpose(0,1).contiguous()
    actual=native.contiguous().cpu()
    assert torch.equal(actual.view(torch.int16),expected.view(torch.int16))


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('axis', [0,1,2])
@pytest.mark.parametrize('width', [128,129])
def test_row_vector_cat_layouts_and_tails(dtype, axis, width):
    values=((torch.arange(3*97*width+1)%17)-8).to(dtype)/8
    source=values[1:].reshape(3,97,width).transpose(0,1)
    native=values.to('ruda')[1:].reshape(3,97,width).transpose(0,1)
    left=native[...,:1].expand_as(native) if axis!=2 else native
    expected_left=source[...,:1].expand_as(source) if axis!=2 else source
    torch.testing.assert_close(torch.cat([left,native],dim=axis).cpu(),torch.cat([expected_left,source],dim=axis),rtol=0,atol=0)


@pytest.mark.parametrize('left_dtype,right_dtype', list(itertools.product(DTYPES, repeat=2)))
@pytest.mark.parametrize('offset', [0, 1])
def test_vector_arithmetic_addressing(left_dtype, right_dtype, offset):
    values = (torch.arange(1024 + offset, dtype=torch.float32) % 17 - 8) / 8
    a, b = values.to(left_dtype), values.flip(0).to(right_dtype)
    a[offset:offset+4] = torch.tensor([float('inf'), -float('inf'), float('nan'), -0.])
    x, y = a.to('ruda')[offset:], b.to('ruda')[offset:]
    a, b = a[offset:].cuda(), b[offset:].cuda()
    for fn in (torch.add, torch.sub, torch.mul):
        for actual, expected in [(fn(x, y), fn(a, b)), (fn(x, .125), fn(a, .125)),
                                 (fn(.125, x), fn(.125, a))]:
            torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close((-x).cpu(), (-a).cpu(), rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(torch.add(x, y, alpha=.75).cpu(), torch.add(a, b, alpha=.75).cpu(), rtol=0, atol=0, equal_nan=True)
    x.mul_(2.)
    torch.testing.assert_close(x.cpu(), (a * 2).cpu(), rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize('left_dtype,right_dtype', list(itertools.product(DTYPES, repeat=2)))
@pytest.mark.parametrize('layout', ['same', 'broadcast', 'strided'])
def test_floating_tensor_dispatch(left_dtype, right_dtype, layout):
    a = (torch.arange(24).reshape(2, 3, 4) - 11).to(left_dtype) / 8
    b = (torch.arange(24).reshape(2, 3, 4) - 7).to(right_dtype) / 4
    if layout == 'broadcast':
        b = b[:, :1, :]
    elif layout == 'strided':
        a, b = a.transpose(1, 2), b.transpose(1, 2)
    x, y = a.to('ruda'), b.to('ruda')
    for fn in (torch.add, torch.sub, torch.mul):
        actual = fn(x, y)
        expected = fn(a, b)
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('value', [0., -0., .125, 7, True, float('inf'), float('-inf'), float('nan')])
def test_immediate_scalar_dispatch(dtype, value):
    a = torch.tensor([[-2., -0., 0., 1.5], [2., 4., -4., 8.]], dtype=dtype).t()
    x = a.to('ruda')
    for fn in (torch.add, torch.mul):
        for reverse in (False, True):
            before = ruda_torch.execution_stats()
            actual = fn(value, x) if reverse else fn(x, value)
            after = ruda_torch.execution_stats()
            assert after['host_to_device_bytes'] == before['host_to_device_bytes']
            expected = fn(value, a) if reverse else fn(a, value)
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    if not isinstance(value, bool):
        torch.testing.assert_close(torch.sub(x, value).cpu(), torch.sub(a, value), rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize('dtype', DTYPES)
def test_direct_destination_and_public_aliases(dtype):
    a = torch.arange(12, dtype=dtype).reshape(3, 4).t()
    x = a.to('ruda'); alias = x.view_as(x)
    x.add_(.5)
    torch.testing.assert_close(alias.cpu(), a + .5, rtol=0, atol=0)
    destination = torch.empty((3, 4), dtype=dtype, device='ruda').t()
    returned = torch.mul(x, 2., out=destination)
    assert returned.data_ptr() == destination.data_ptr()
    torch.testing.assert_close(destination.cpu(), (a + .5) * 2, rtol=0, atol=0)
    old = destination.clone()
    x.mul_(3.)
    torch.testing.assert_close(destination.cpu(), old.cpu(), rtol=0, atol=0)
    resized = torch.empty((0,), dtype=torch.float64, device='ruda')
    with pytest.raises(RuntimeError, match='Scalar_out'):
        torch.add(x, .25, out=resized)
    torch.add(x, torch.full(x.shape, .25, dtype=dtype, device='ruda'), out=resized)
    torch.testing.assert_close(resized.cpu(), ((a + .5) * 3 + .25).double(), rtol=0, atol=0)
    with pytest.raises(RuntimeError):
        torch.add(x, 1., out=x[:1].expand_as(x))


def test_scalar_tensor_integer_and_autograd_compatibility():
    a = torch.tensor([-2., 0., 3.], device='ruda', requires_grad=True)
    b = a * .5 + 2.
    b.sum().backward()
    torch.testing.assert_close(a.grad.cpu(), torch.full((3,), .5), rtol=0, atol=0)
    scalar = torch.tensor(.25)
    torch.testing.assert_close((a.detach() + scalar).cpu(), torch.tensor([-1.75, .25, 3.25]), rtol=0, atol=0)
    integers = torch.tensor([1, -2, 3], dtype=torch.int64, device='ruda')
    torch.testing.assert_close((integers * 3 + 2).cpu(), torch.tensor([5, -4, 11]), rtol=0, atol=0)
    torch.testing.assert_close((a.detach()[:0] + 1).cpu(), torch.empty(0), rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_zero_dimensional_tensor_keeps_storage_conversion(dtype):
    a = torch.tensor([1., -1.], dtype=dtype)
    scalar = torch.tensor(.10001, dtype=torch.float32)
    x, y = a.to('ruda'), scalar.to('ruda')
    torch.testing.assert_close((x * y).cpu(), (a.float() * scalar.to(dtype).float()).to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('offset', [0, 1])
def test_flat_vector_arguments_preserve_rank_and_storage_offset(dtype, offset):
    source=((torch.arange(1536+offset+4,dtype=torch.float32)%31)/16+.125).to(dtype)
    storage=source.to('ruda')
    value=storage[offset:offset+1536].reshape(2,3,16,16)
    reference=source[offset:offset+1536].reshape(2,3,16,16)
    for result,expected in [(value*.5+.125,reference*.5+.125),
                            (value.square(),reference.square()),
                            (value.float(),reference.float())]:
        assert result.shape==reference.shape
        torch.testing.assert_close(result.cpu(),expected,rtol=0,atol=0)
    actual=torch.nn.functional.silu(value)
    expected=torch.nn.functional.silu(reference.cuda()).cpu()
    tolerance=.02 if dtype==torch.bfloat16 else .002 if dtype==torch.float16 else 3e-6
    torch.testing.assert_close(actual.cpu(),expected,rtol=tolerance,atol=tolerance)
    value.mul_(.5)
    reference.mul_(.5)
    torch.testing.assert_close(storage.cpu(),source,rtol=0,atol=0)


NAMES = ['add.Tensor', 'add_.Tensor', 'sub.Tensor', 'sub_.Tensor',
         'mul.Tensor', 'mul_.Tensor', 'neg', 'neg_']


def operator(name):
    base, _, overload = name.partition('.')
    return getattr(getattr(torch.ops.aten, base), overload or 'default')


def original(name, a, b, alpha=.5):
    op = {'add': 0, 'sub': 1, 'mul': 2, 'neg': 3}[name.split('.')[0].rstrip('_')]
    return _arithmetic.arithmetic(a, b, op=op, alpha=alpha, inplace=name.split('.')[0].endswith('_'))


def invoke(name, a, b, alpha=.5):
    args = (a,) if name.startswith('neg') else (a, b)
    kwargs = {'alpha': alpha} if name.startswith(('add', 'sub')) else {}
    return operator(name)(*args, **kwargs)


def same(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.stride() == expected.stride() and actual.device == expected.device
    assert torch.equal(actual.cpu().contiguous().reshape(-1).view(torch.uint8),
                       expected.cpu().contiguous().reshape(-1).view(torch.uint8))


@pytest.mark.parametrize('name', NAMES)
def test_arithmetic_dispatch_has_cpp_entrypoint(name):
    assert _ops._cpp_arithmetic_dispatch
    table = torch._C._dispatch_dump_table('aten::' + name)
    entry = next(line for line in table.splitlines() if line.startswith('PrivateUse1:'))
    assert 'backend.cpp' in entry


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize('layout', ['contiguous', 'transpose', 'offset'])
@pytest.mark.parametrize('name', NAMES)
def test_arithmetic_dispatch_matches_original_bits_and_cpu(name, layout, dtype):
    source = torch.arange(41, dtype=torch.float64).sub(20).div(8).to(dtype)
    def tensor():
        value = source.to('ruda')
        value = value[1:] if layout == 'offset' else value[:40]
        value = value.reshape(5, 8)
        return value.t() if layout == 'transpose' else value
    a, b, old_a, old_b = (tensor() for _ in range(4))
    cpu_a, cpu_b = a.cpu(), b.cpu()
    version = a._version
    actual, expected = invoke(name, a, b), original(name, old_a, old_b)
    same(actual, expected)
    torch.testing.assert_close(actual.cpu(), invoke(name, cpu_a, cpu_b), rtol=0, atol=0)
    if name.split('.')[0].endswith('_'):
        assert actual.data_ptr() == a.data_ptr() and a._version == version + 1
    else:
        assert actual.data_ptr() != a.data_ptr() and a._version == version


@pytest.mark.parametrize('name', NAMES)
def test_arithmetic_dispatch_common_path_never_enters_python(name, monkeypatch):
    a = torch.ones(5, 8, device='ruda')
    b = torch.full((5, 8), 2., device='ruda')
    expected = invoke(name, torch.ones(5, 8), torch.full((5, 8), 2.))
    def unexpected(*args, **kwargs): raise AssertionError('entered Python arithmetic')
    monkeypatch.setattr(_arithmetic, 'arithmetic', unexpected)
    torch.testing.assert_close(invoke(name, a, b).cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize('name', NAMES)
def test_arithmetic_dispatch_empty_scalar_alias_and_special_values(name):
    for cpu in (torch.empty(2, 0, 3), torch.tensor(2.),
                torch.tensor([0., -0., float('inf'), -float('inf'), float('nan'), 1.])):
        a, old_a = cpu.to('ruda'), cpu.to('ruda')
        same(invoke(name, a, a), original(name, old_a, old_a))


@pytest.mark.parametrize('name', ['add.Tensor', 'sub.Tensor', 'mul.Tensor'])
@pytest.mark.parametrize('case', ['broadcast', 'mixed', 'cpu_scalar', 'integral', 'complex', 'negative_view'])
def test_arithmetic_dispatch_preserves_fallback(name, case, monkeypatch):
    a = torch.arange(8, dtype=torch.float32).reshape(2, 4).to('ruda')
    b = a.clone()
    if case == 'broadcast': b = b[:1]
    elif case == 'mixed': b = b.to(torch.float16)
    elif case == 'cpu_scalar': b = torch.tensor(2.)
    elif case == 'integral': a, b = a.to(torch.int64), b.to(torch.int64)
    elif case == 'complex': a, b = a.to(torch.complex64), b.to(torch.complex64)
    else: a = torch._neg_view(a)
    expected = original(name, a, b, alpha=1)
    body = _arithmetic.arithmetic
    calls = []
    def observed(*args, **kwargs):
        calls.append(kwargs)
        return body(*args, **kwargs)
    monkeypatch.setattr(_arithmetic, 'arithmetic', observed)
    same(invoke(name, a, b, alpha=1), expected)
    if case in ('integral', 'complex'): assert calls
    elif case != 'negative_view': assert not calls


@pytest.mark.parametrize('name', ['add_.Tensor', 'sub_.Tensor', 'mul_.Tensor'])
def test_arithmetic_dispatch_partial_overlap_errors_match_original(name):
    for run in (original, invoke):
        a = torch.ones(9, device='ruda')
        with pytest.raises(RuntimeError) as error: run(name, a[:8], a[1:])
        if run is original: expected = str(error.value)
        else: assert str(error.value) == expected


@pytest.mark.parametrize('name', ['add.Tensor', 'sub.Tensor'])
def test_arithmetic_dispatch_alpha_matches_original(name):
    a = torch.tensor([1., -0., .75, 32.], device='ruda')
    b = torch.tensor([-.25, 2., 3., 4.], device='ruda')
    for alpha in (0, -1, -.125, 2**40, float('inf'), float('nan')):
        same(invoke(name, a, b, alpha), original(name, a, b, alpha))
    for alpha in (True, 1j):
        with pytest.raises(Exception) as expected: original(name, a, b, alpha)
        with pytest.raises(type(expected.value)) as actual: invoke(name, a, b, alpha)
        assert str(actual.value) == str(expected.value)


def test_arithmetic_dispatch_first_and_second_gradients_and_stream():
    def gradients(device):
        a = torch.arange(1, 9, dtype=torch.float32).to(device).requires_grad_()
        b = torch.arange(9, 17, dtype=torch.float32).to(device).requires_grad_()
        loss = (a * b + a - b).neg().square().sum()
        first = torch.autograd.grad(loss, (a, b), create_graph=True)
        second = torch.autograd.grad(first[0].sum() + first[1].sum(), (a, b))
        return first + second
    expected = gradients('cpu')
    stream = r.Stream()
    with r.stream(stream): actual = gradients('ruda')
    stream.synchronize()
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), b, rtol=0, atol=0)


def test_arithmetic_dispatch_aot_dynamic_autograd():
    from torch._dynamo.backends.common import aot_autograd
    graphs = []
    def capture(graph, inputs):
        graphs.append(graph)
        return graph.forward
    def function(a, b): return ((a + b) * (a - b)).neg().sum()
    compiled = torch.compile(function, backend=aot_autograd(fw_compiler=capture, bw_compiler=capture),
                             fullgraph=True, dynamic=True)
    for rows in (3, 7):
        args = [torch.randn(rows, 5).to('ruda').requires_grad_() for _ in range(2)]
        expected = torch.autograd.grad(function(*args), args)
        actual = torch.autograd.grad(compiled(*args), args)
        for a, b in zip(actual, expected): same(a, b)
    assert len(graphs) >= 2
