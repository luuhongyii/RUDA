import math
import operator
from functools import partial

import torch

from . import _C
from ._indexing import _logical
from ._ops import _dimension
from ._sparse_coo import _make, _sorted, _sparse_output


def _coo(value):
    if value.layout != torch.sparse_coo or value.device.type != torch._C._get_privateuse1_backend_name():
        raise RuntimeError('sparse softmax requires COO storage on RUDA')


def _finish(result, out):
    if out is None:
        return result
    if out.dtype != result.dtype:
        raise RuntimeError('sparse softmax output dtype must equal the functional result dtype')
    return _sparse_output(result, out)


def _empty(value):
    indices = torch.empty((value.sparse_dim(), 0), dtype=torch.int64, device=value.device)
    values = torch.empty((0, *value.shape[value.sparse_dim():]), dtype=value.dtype, device=value.device)
    return _make(indices, values, value.shape, True)


def _pools(indices, dim):
    keys = torch.cat((indices[:dim], indices[dim + 1:]), dim=0)
    return _sorted(keys)


def softmax(self, dim, half_to_float, *, out=None, logarithmic=False):
    _coo(self)
    if half_to_float:
        raise RuntimeError('sparse softmax with half_to_float conversion is not supported')
    value = self.coalesce()
    dim = _dimension(operator.index(dim), value.ndim)
    if value.numel() == 0:
        return _finish(_empty(value), out)
    if value.dtype not in (torch.float32, torch.float64):
        raise RuntimeError('sparse softmax requires float32 or float64 values; cast with the public dtype argument')
    indices = _logical(value._indices()).clone(memory_format=torch.contiguous_format)
    values = _logical(value._values())
    if dim >= value.sparse_dim():
        operation = torch.ops.aten._log_softmax.default if logarithmic else torch.ops.aten._softmax.default
        result_values = operation(values, dim - value.sparse_dim() + 1, False)
    else:
        result_values = torch.empty(values.shape, dtype=value.dtype, device=value.device)
        if result_values.numel():
            order, starts, count = _pools(indices, dim)
            width = math.prod(values.shape[1:])
            _C.scientific_call(530, [values.reshape(value._nnz(), width), order, starts, count,
                                      result_values.reshape(value._nnz(), width)], [int(logarithmic)], [])
    return _finish(_make(indices, result_values, value.shape, True), out)


def backward(grad_output, output, dim, self, *, out=None, logarithmic=False):
    for value in (grad_output, output, self):
        _coo(value)
    if grad_output.shape != output.shape:
        raise RuntimeError('sparse softmax gradient and output shapes must match')
    if grad_output.device != output.device or self.device != output.device:
        raise RuntimeError('sparse softmax backward tensors must share a device')
    dim = _dimension(operator.index(dim), output.ndim)
    grad, value = grad_output.coalesce(), output.coalesce()
    if grad.sparse_dim() != value.sparse_dim():
        raise RuntimeError('sparse softmax gradient and output sparse dimensions must match')
    if value.numel() == 0:
        return _finish(_empty(value), out)
    if value.dtype not in (torch.float32, torch.float64) or grad.dtype != value.dtype or self.dtype != value.dtype:
        raise RuntimeError('sparse softmax backward requires matching float32 or float64 dtypes')
    indices = _logical(value._indices()).clone(memory_format=torch.contiguous_format)
    grad_indices = _logical(grad._indices())
    values, grad_values = _logical(value._values()), _logical(grad._values())
    nnz = value._nnz(); width = math.prod(values.shape[1:])
    result_values = torch.empty(values.shape, dtype=value.dtype, device=value.device)
    if result_values.numel():
        if dim >= value.sparse_dim():
            aligned = result_values
            present = torch.empty((nnz,), dtype=torch.bool, device=value.device)
            _C.scientific_call(532, [indices, grad_indices, grad_values.reshape(grad._nnz(), width),
                                      aligned.reshape(nnz, width), present], [], [])
            operation = (torch.ops.aten._log_softmax_backward_data.default if logarithmic
                         else torch.ops.aten._softmax_backward_data.default)
            result_values = operation(aligned, values, dim - value.sparse_dim() + 1, self.dtype)
            result_values = torch.where(present.reshape((nnz,) + (1,) * value.dense_dim()), result_values, 0)
        else:
            order, starts, count = _pools(indices, dim)
            _C.scientific_call(531, [indices, grad_indices, values.reshape(nnz, width),
                                      grad_values.reshape(grad._nnz(), width), order, starts, count,
                                      result_values.reshape(nnz, width)], [int(logarithmic)], [])
    return _finish(_make(indices, result_values, value.shape, True), out)


def register(library):
    for name, logarithmic in (('_sparse_softmax', False), ('_sparse_log_softmax', True)):
        forward = partial(softmax, logarithmic=logarithmic)
        reverse = partial(backward, logarithmic=logarithmic)
        library.impl(name, forward)
        library.impl(name + '.out', forward)
        library.impl(name + '_backward_data', reverse)
        library.impl(name + '_backward_data.out', reverse)
