from functools import partial

import torch
from . import _C
from ._ops import _dtypes, _integral_dtypes, _typed_value
from ._ordering import _output, _outputs


def _run(a, *, op, grad=None, weight=None, params=(0., 0., 0.), flag=False, out=None, inplace=False):
    if a.dtype not in _dtypes:
        raise RuntimeError('activation requires real floating inputs')
    original = (a,) + tuple(t for t in (grad, weight) if t is not None)
    if any(t.device != a.device or t.dtype != a.dtype for t in original):
        raise RuntimeError('activation inputs must have the same device and dtype')
    if inplace:
        out = a
    if out is not None:
        for t in original:
            _C.check_inplace(out, t)
    tensors = [t.resolve_conj().resolve_neg() for t in original]
    shape = torch.broadcast_shapes(*(t.shape for t in tensors))
    if inplace and shape != a.shape:
        raise RuntimeError('in-place activation cannot change shape')
    x = tensors[0].expand(shape)
    g = (tensors[1] if grad is not None else tensors[0]).expand(shape)
    w = (tensors[-1] if weight is not None else tensors[0]).expand(shape)
    result = torch.empty_like(x)
    _C.scientific_call(op, [x, g, w, result], [int(flag)], [float(p) for p in params])
    return _output(result, out, original, cast=True)


def elu(a, alpha=1, scale=1, input_scale=1, *, out=None, inplace=False):
    return _run(a, op=265, params=(alpha, scale, input_scale), out=out, inplace=inplace)


def elu_backward(grad, alpha, scale, input_scale, is_result, self_or_result, *, grad_input=None):
    if is_result and not alpha >= 0:
        raise RuntimeError('in-place ELU backward does not support a negative slope')
    return _run(self_or_result, op=266, grad=grad, params=(alpha, scale, input_scale), flag=is_result, out=grad_input)


def leaky_relu(a, negative_slope=.01, *, out=None, inplace=False):
    return _run(a, op=267, params=(negative_slope, 0, 0), out=out, inplace=inplace)


def leaky_relu_backward(grad, a, negative_slope, self_is_result, *, grad_input=None):
    if self_is_result and not negative_slope >= 0:
        raise RuntimeError('in-place leaky ReLU backward does not support a negative slope')
    return _run(a, op=268, grad=grad, params=(negative_slope, 0, 0), out=grad_input)


def backward(grad, a, *, op, grad_input=None, out=None):
    return _run(a, op=op, grad=grad, out=grad_input if grad_input is not None else out)


def softplus(a, beta=1, threshold=20, *, out=None):
    return _run(a, op=273, params=(beta, threshold, 0), out=out)


def softplus_backward(grad, a, beta, threshold, *, grad_input=None):
    return _run(a, op=274, grad=grad, params=(beta, threshold, 0), out=grad_input)


def shrink(a, lambd=.5, *, op, out=None):
    if op == 276 and (a.dtype not in _dtypes or not 0 <= lambd <= torch.finfo(a.dtype).max):
        raise RuntimeError('softshrink lambda must lie in the nonnegative input dtype range')
    if a.dtype in _dtypes:
        _typed_value(lambd, a.dtype, a.device)
    return _run(a, op=op, params=(lambd, 0, 0), out=out)


def shrink_backward(grad, a, lambd, *, grad_input=None):
    if a.dtype in _dtypes:
        _typed_value(lambd, a.dtype, a.device)
    return _run(a, op=277, grad=grad, params=(lambd, 0, 0), out=grad_input)


def log_sigmoid_forward(a, *, output=None, buffer=None):
    result = _run(a, op=280, out=output)
    if buffer is None:
        buffer = a.new_empty((0,))
    return result, buffer


def log_sigmoid_backward(grad, a, buffer, *, grad_input=None):
    if buffer.device != a.device:
        raise RuntimeError('log_sigmoid buffer must share the input device')
    return _run(a, op=281, grad=grad, out=grad_input)


def gelu(a, *, approximate='none', out=None, inplace=False):
    if approximate not in ('none', 'tanh'):
        raise RuntimeError('gelu approximate must be none or tanh')
    return _run(a, op=282, flag=approximate == 'tanh', out=out, inplace=inplace)


def gelu_backward(grad, a, *, approximate='none', grad_input=None):
    if approximate not in ('none', 'tanh'):
        raise RuntimeError('gelu approximate must be none or tanh')
    return _run(a, op=283, grad=grad, flag=approximate == 'tanh', out=grad_input)


def silu(a, *, out=None, inplace=False):
    if a.is_complex():
        if inplace:
            out = a
        return _output(a / (1 + (-a).exp()), out, (a,), cast=True)
    return _run(a, op=284, out=out, inplace=inplace)


def prelu(a, weight):
    return _run(a, op=286, weight=weight)


def prelu_backward(grad, a, weight):
    return (_run(a, op=287, grad=grad, weight=weight), _run(a, op=288, grad=grad, weight=weight))


def _bounded(a, lower, upper, *, op, other=None, out=None, inplace=False):
    if a.dtype not in _dtypes + _integral_dtypes or a.dtype == torch.bool:
        raise RuntimeError('activation requires real non-bool input')
    if other is None:
        other = a
    if a.device != other.device:
        raise RuntimeError('activation inputs must share a device')
    dtype = torch.result_type(a, other)
    x, y = torch.broadcast_tensors(a.to(dtype).resolve_neg(), other.to(dtype).resolve_neg())
    if inplace:
        out = a
        if x.shape != a.shape:
            raise RuntimeError('in-place activation cannot change shape')
    low, high = (_typed_value(v, dtype, a.device) for v in (lower, upper))
    result = torch.empty_like(x)
    _C.scientific_call(op, [x, y, low, high, result], [], [])
    if out is not None and op != 289:
        if out.device != result.device or not torch.can_cast(result.dtype, out.dtype):
            raise RuntimeError('threshold output dtype or device does not match')
        if out.shape != result.shape:
            _C.prepare_index_output(out, list(result.shape))
        value = result.to(out.dtype)
        if out.is_conj() or out.is_neg():
            _C.scientific_call(207, [value, out], [int(out.is_conj()), int(out.is_neg())], [])
        else:
            _C.scientific_call(0, [value, value, out], [], [1.])
        return out
    return _output(result, out, (a, other) if op == 289 else (), cast=True)


def hardtanh(a, min_val=-1, max_val=1, *, out=None, inplace=False):
    if a.dtype in _integral_dtypes:
        min_val, max_val = int(min_val), int(max_val)
        if a.dtype == torch.uint8 and (min_val < 0 or max_val < 0):
            raise RuntimeError('unsigned hardtanh cannot have negative limits')
    return _bounded(a, min_val, max_val, op=289, out=out, inplace=inplace)


def hardtanh_backward(grad, a, min_val, max_val, *, grad_input=None):
    return _run(a, op=290, grad=grad, params=(min_val, max_val, 0), out=grad_input)


def threshold(a, threshold, value, *, out=None, inplace=False):
    return _bounded(a, threshold, value, op=291, out=out, inplace=inplace)


def threshold_backward(grad, a, threshold, *, grad_input=None):
    return _bounded(a, threshold, 0, op=292, other=grad, out=grad_input)


def register(library):
    for name, fn in {'elu': elu, 'leaky_relu': leaky_relu, 'hardsigmoid': partial(_run, op=269),
                     'hardswish': partial(_run, op=271), 'softplus': softplus,
                     'hardshrink': partial(shrink, op=275), 'softshrink': partial(shrink, op=276),
                     'mish': partial(_run, op=278), 'gelu': gelu, 'silu': silu,
                     'hardtanh': hardtanh, 'threshold': threshold}.items():
        library.impl(name, fn)
        library.impl(name + '.out', fn)
        if name not in ('softplus', 'hardshrink', 'softshrink'):
            library.impl(name + '_', partial(fn, inplace=True))
    for name, fn in {'elu_backward': elu_backward, 'leaky_relu_backward': leaky_relu_backward,
                     'hardsigmoid_backward': partial(backward, op=270), 'hardswish_backward': partial(backward, op=272),
                     'softplus_backward': softplus_backward, 'hardshrink_backward': shrink_backward,
                     'softshrink_backward': shrink_backward, 'mish_backward': partial(backward, op=279),
                     'gelu_backward': gelu_backward, 'silu_backward': partial(backward, op=285),
                     'hardtanh_backward': hardtanh_backward, 'threshold_backward': threshold_backward}.items():
        library.impl(name, fn)
        if name != 'mish_backward':
            library.impl(name + ('.out' if name == 'hardswish_backward' else '.grad_input'), fn)
    library.impl('log_sigmoid_forward', log_sigmoid_forward)
    library.impl('log_sigmoid_forward.output', log_sigmoid_forward)
    library.impl('log_sigmoid_backward', log_sigmoid_backward)
    library.impl('log_sigmoid_backward.grad_input', log_sigmoid_backward)
    library.impl('_prelu_kernel', prelu)
    library.impl('_prelu_kernel_backward', prelu_backward)
