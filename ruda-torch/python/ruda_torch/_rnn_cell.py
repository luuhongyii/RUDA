import torch

from . import _C


_floating = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _matching(reference, *tensors):
    if reference.dtype not in _floating:
        raise RuntimeError('fused RNN cells require real floating tensors')
    for value in (reference, *tensors):
        if value is not None and (value.layout != torch.strided or value.device.type != 'ruda'
                                  or value.device != reference.device or value.dtype != reference.dtype):
            raise RuntimeError('fused RNN tensors must share strided RUDA storage dtype and device')


def _forward(input_gates, hidden_gates, previous, input_bias, hidden_bias, factor):
    _matching(input_gates, hidden_gates, previous, input_bias, hidden_bias)
    if input_gates.ndim != 2 or hidden_gates.shape != input_gates.shape or previous.ndim != 2:
        raise RuntimeError('fused RNN cells require matching 2D gates and a 2D previous state')
    if previous.numel() != input_gates.numel() // factor:
        raise RuntimeError('fused RNN previous state size does not match the gates')
    if input_bias is not None:
        if (input_bias.ndim != 1 or input_bias.numel() != input_gates.size(1)
                or hidden_bias is None or hidden_bias.shape != input_bias.shape):
            raise RuntimeError('fused RNN biases must both match the gate width')
        if previous.numel() and input_bias.numel() < previous.size(1) * factor:
            raise RuntimeError('fused RNN bias storage is smaller than the state gate width')


def _empty(a):
    return torch.empty_like(a, memory_format=torch.contiguous_format)


def _launch(op, inputs, outputs, flags):
    _C.scientific_call(op, [*(value.resolve_neg() for value in inputs), *outputs], [flags], [])


def lstm_cell(input_gates, hidden_gates, cx, input_bias=None, hidden_bias=None):
    _forward(input_gates, hidden_gates, cx, input_bias, hidden_bias, 4)
    hy, cy, workspace = _empty(cx), _empty(cx), _empty(input_gates)
    has_bias = input_bias is not None
    _launch(443, [input_gates, hidden_gates, cx,
                  input_bias if has_bias else cx, hidden_bias if has_bias else cx],
            [hy, cy, workspace], int(has_bias))
    return hy, cy, workspace


def lstm_cell_backward(grad_hy, grad_cy, cx, cy, workspace, has_bias):
    if grad_hy is None and grad_cy is None:
        return None, None, None
    reference = grad_hy if grad_hy is not None else grad_cy
    _matching(workspace, grad_hy, grad_cy, cx, cy)
    if reference.ndim != 2 or any(value is not None and value.shape != reference.shape
                                  for value in (grad_hy, grad_cy, cx, cy)):
        raise RuntimeError('LSTM cell gradients and cell states must have matching 2D shapes')
    if workspace.ndim != 2 or workspace.numel() != reference.numel() * 4:
        raise RuntimeError('LSTM cell workspace must have four saved gates per state element')
    grad_gates, grad_cx = _empty(workspace), _empty(cx)
    flags = int(grad_hy is not None) | (int(grad_cy is not None) << 1)
    _launch(444, [grad_hy if grad_hy is not None else cx,
                  grad_cy if grad_cy is not None else cx, cx, cy, workspace],
            [grad_gates, grad_cx], flags)
    grad_bias = grad_gates.sum(0) if has_bias else None
    return grad_gates, grad_cx, grad_bias


def gru_cell(input_gates, hidden_gates, hx, input_bias=None, hidden_bias=None):
    _forward(input_gates, hidden_gates, hx, input_bias, hidden_bias, 3)
    hy = _empty(hx)
    workspace = torch.empty((hx.size(0), hx.size(1) * 5), dtype=hx.dtype, device=hx.device)
    has_bias = input_bias is not None
    _launch(445, [input_gates, hidden_gates, hx,
                  input_bias if has_bias else hx, hidden_bias if has_bias else hx],
            [hy, workspace], int(has_bias))
    return hy, workspace


def gru_cell_backward(grad_hy, workspace, has_bias):
    _matching(grad_hy, workspace)
    if grad_hy.ndim != 2 or workspace.shape != (grad_hy.size(0), grad_hy.size(1) * 5):
        raise RuntimeError('GRU cell workspace must have five saved values per state element')
    shape = (grad_hy.size(0), grad_hy.size(1) * 3)
    grad_input = torch.empty(shape, dtype=workspace.dtype, device=workspace.device)
    grad_hidden = torch.empty_like(grad_input)
    grad_hx = _empty(grad_hy)
    _launch(446, [grad_hy, workspace], [grad_input, grad_hidden, grad_hx], 0)
    grad_input_bias = grad_input.sum(0) if has_bias else None
    grad_hidden_bias = grad_hidden.sum(0) if has_bias else None
    return grad_input, grad_hidden, grad_hx, grad_input_bias, grad_hidden_bias


def register(library):
    library.impl('_thnn_fused_lstm_cell', lstm_cell)
    library.impl('_thnn_fused_lstm_cell_backward_impl', lstm_cell_backward)
    library.impl('_thnn_fused_gru_cell', gru_cell)
    library.impl('_thnn_fused_gru_cell_backward', gru_cell_backward)
