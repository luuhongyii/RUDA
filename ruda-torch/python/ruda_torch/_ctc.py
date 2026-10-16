import math
import operator

import torch

from . import _C
from ._indexing import _logical
from ._ordering import _output, _outputs
from ._pooling import _nondeterministic


def _lengths(value, batch, name):
    if isinstance(value, torch.Tensor):
        if value.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
                               torch.uint16, torch.uint32, torch.uint64):
            raise RuntimeError(name + ' must be integral')
        if value.layout != torch.strided or value.numel() != batch:
            raise RuntimeError(name + ' must contain batch_size entries')
        return value.to(device='cpu', dtype=torch.int64).contiguous().reshape(-1).tolist()
    if len(value) != batch:
        raise RuntimeError(name + ' must contain batch_size entries')
    return [operator.index(n) for n in value]


def _inputs(log_probs, targets, input_lengths, target_lengths, blank):
    if log_probs.layout != torch.strided or log_probs.ndim != 3 or not log_probs.numel():
        raise RuntimeError('CTC log_probs must be a nonempty strided (T,N,C) tensor')
    if log_probs.dtype not in (torch.float32, torch.float64):
        raise RuntimeError('native CTC requires float32 or float64 log_probs')
    if (targets.device != log_probs.device or targets.layout != torch.strided
            or targets.dtype not in (torch.int32, torch.int64) or targets.ndim not in (1, 2)):
        raise RuntimeError('CTC targets must be a 1D/2D int32/int64 tensor on the log_probs device')
    time, batch, classes = log_probs.shape
    if not 0 <= blank < classes:
        raise RuntimeError('CTC blank must be in label range')
    inputs = _lengths(input_lengths, batch, 'input_lengths')
    lengths = _lengths(target_lengths, batch, 'target_lengths')
    if any(n < 0 or n > time for n in inputs):
        raise RuntimeError('CTC input_lengths must be between zero and T')
    if any(n < 0 for n in lengths):
        raise RuntimeError('CTC target_lengths must be nonnegative')
    maximum = max(lengths)
    if targets.ndim == 1:
        if sum(lengths) != targets.numel():
            raise RuntimeError('concatenated CTC targets must contain sum(target_lengths) entries')
        offsets, offset = [], 0
        for n in lengths:
            offsets.append(offset)
            offset += n
    else:
        if targets.shape[0] != batch or targets.shape[1] < maximum:
            raise RuntimeError('padded CTC targets must have shape (N,S) with S >= max(target_lengths)')
        offsets = [b * targets.shape[1] for b in range(batch)]
    shape = (batch, time, 2 * maximum + 1)
    if max(log_probs.numel(), targets.numel(), math.prod(shape), batch * 3) > 2**32 - 1:
        raise RuntimeError('CTC tensors exceed native U32 indexing')
    metadata = torch.tensor(list(zip(inputs, lengths, offsets)), dtype=torch.int64, device=log_probs.device)
    return _logical(log_probs), _logical(targets).reshape(-1), metadata, shape


def forward(log_probs, targets, input_lengths, target_lengths, blank=0, zero_infinity=False, *, out0=None, out1=None):
    a, t, metadata, shape = _inputs(log_probs, targets, input_lengths, target_lengths, blank)
    nll = torch.empty((a.shape[1],), dtype=a.dtype, device=a.device)
    alpha = torch.empty(shape, dtype=a.dtype, device=a.device)
    _C.scientific_call(356, [a, t, metadata, nll, alpha], [blank], [])
    if (out0 is None) != (out1 is None):
        raise RuntimeError('CTC requires both out0 and out1 outputs')
    outputs = None if out0 is None else (out0, out1)
    return _outputs((nll, alpha), outputs, (log_probs, targets, input_lengths, target_lengths))


def backward(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood,
             log_alpha, blank, zero_infinity=False, *, out=None):
    _nondeterministic('ctc_loss_backward_gpu')
    a, t, metadata, shape = _inputs(log_probs, targets, input_lengths, target_lengths, blank)
    time, batch, classes = a.shape
    for value, expected, name in ((grad, (batch,), 'grad'),
                                  (neg_log_likelihood, (batch,), 'neg_log_likelihood'),
                                  (log_alpha, shape, 'log_alpha')):
        if (value.device != a.device or value.dtype != a.dtype or value.layout != torch.strided
                or tuple(value.shape) != expected):
            raise RuntimeError('CTC ' + name + ' has incorrect shape, dtype or device')
    g, nll, alpha = _logical(grad), _logical(neg_log_likelihood), _logical(log_alpha)
    beta = torch.empty(shape, dtype=a.dtype, device=a.device)
    _C.scientific_call(357, [a, t, metadata, beta], [blank], [])
    result = torch.empty_like(a, memory_format=torch.contiguous_format)
    tensors = [a, t, metadata, nll, alpha, beta, g, result]
    large = 2 * time + (24 * batch) // 10 + (2 * classes) // 10 > 450
    if large:
        torch.exp(a, out=result)
        blank_grad = result.narrow(2, blank, 1)
        blank_mass = torch.logsumexp(alpha[:, :, ::2] + beta[:, :, ::2], 2, keepdim=True).permute(1, 0, 2)
        blank_mass.add_(nll.reshape(1, batch, 1)).sub_(a.narrow(2, blank, 1)).exp_()
        blank_grad.sub_(blank_mass)
        result.mul_(g.reshape(1, batch, 1))
        if zero_infinity:
            result = torch.where(nll.reshape(1, batch, 1) == math.inf,
                                 torch.zeros((), dtype=a.dtype, device=a.device), result)
            tensors[-1] = result
        _C.scientific_call(359, tensors, [blank, int(zero_infinity)], [])
        _C.scientific_call(360, [result, metadata], [], [])
    else:
        _C.scientific_call(358, tensors, [blank, int(zero_infinity)], [])
    return _output(result, out, (grad, log_probs, targets, neg_log_likelihood, log_alpha))


def register(library):
    for name in ('_ctc_loss', '_ctc_loss.out', '_ctc_loss.Tensor', '_ctc_loss.Tensor_out'):
        library.impl(name, forward)
    for name in ('_ctc_loss_backward', '_ctc_loss_backward.out', '_ctc_loss_backward.Tensor'):
        library.impl(name, backward)
