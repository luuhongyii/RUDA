import torch
from . import _C
from ._ops import _registry, _numeric_dtypes, _integral_dtypes, _dimension, _typed_value
from ._indexing import (_logical, _indices, _nonempty_rank, _prefix, _target, _finish_target,
                        _index_out, scatter)


def scatter_reduce(a, dim, index, source, reduce, *, include_self=True, inplace=False, out=None):
    if reduce not in ('sum', 'prod', 'mean', 'amin', 'amax'):
        raise RuntimeError('scatter_reduce expects sum, prod, mean, amin or amax')
    if a.dtype not in _numeric_dtypes + _integral_dtypes:
        raise RuntimeError('scatter_reduce requires a supported numeric dtype')
    if a.is_complex() and reduce in ('amin', 'amax') or a.dtype == torch.bool and reduce == 'mean':
        raise RuntimeError('scatter_reduce mode is not supported for the input dtype')
    axis = _dimension(dim, a.ndim)
    _indices(a, index, empty_dtype=True)
    if source.dtype != a.dtype or source.device != a.device:
        raise RuntimeError('scatter_reduce source must match the input dtype and device')
    if inplace:
        _C.check_index_output(a, source, index)
    result = _target(a, inplace)
    if index.numel():
        target, indices, value = _nonempty_rank(a), _nonempty_rank(_logical(index)), _nonempty_rank(_logical(source))
        if indices.ndim != target.ndim or indices.ndim != value.ndim:
            raise RuntimeError('scatter_reduce input, source and indices must have matching ranks')
        if any(indices.shape[d] > value.shape[d] or d != axis and indices.shape[d] > target.shape[d] for d in range(target.ndim)):
            raise RuntimeError('scatter_reduce index shape exceeds the source or destination')
        mode = ('sum', 'prod', 'mean', 'amin', 'amax').index(reduce)
        _C.scientific_call(467, [_prefix(value, indices.shape), indices, _nonempty_rank(result)],
                           [axis, mode, int(include_self)], [])
    result = _finish_target(a, result, inplace)
    return _index_out(result, out, (a, index, source), allow_self=True)


def index_reduce(a, dim, index, source, reduce, *, include_self=True, inplace=False, out=None):
    if reduce not in ('prod', 'mean', 'amin', 'amax'):
        raise RuntimeError('index_reduce expects prod, mean, amin or amax')
    if a.is_complex() or a.dtype == torch.bool:
        raise RuntimeError('index_reduce requires a real numeric dtype')
    axis = _dimension(dim, a.ndim)
    _indices(a, index)
    if index.ndim > 1:
        raise RuntimeError('index_reduce requires a scalar or vector index')
    value, target = _nonempty_rank(source), _nonempty_rank(a)
    if value.ndim != target.ndim or value.shape[axis] != index.numel():
        raise RuntimeError('index_reduce source dimension must match the number of indices')
    if any(value.shape[d] != target.shape[d] for d in range(target.ndim) if d != axis):
        raise RuntimeError('index_reduce source shape must match outside the indexed dimension')
    shape = [1] * target.ndim
    shape[axis] = index.numel()
    indices = _logical(index).reshape(shape).expand(value.shape)
    result = scatter_reduce(target, axis, indices, value, reduce, include_self=include_self, inplace=inplace)
    if a.ndim == 0:
        result = result.view(())
    if inplace:
        result = a
    return _index_out(result, out, (a, index, source), allow_self=True)


def legacy_scatter(a, dim, index, source, *, reduce, inplace=False, out=None):
    if reduce not in ('add', 'multiply'):
        raise RuntimeError('scatter reduce expects add or multiply')
    values = source if isinstance(source, torch.Tensor) else _typed_value(source, a.dtype, a.device).expand(index.shape)
    return scatter_reduce(a, dim, index, values, 'sum' if reduce == 'add' else 'prod', inplace=inplace, out=out)


def scatter_out(a, dim, index, source, *, out):
    result = scatter(a, dim, index, source)
    inputs = (a, index, source) if isinstance(source, torch.Tensor) else (a, index)
    return _index_out(result, out, inputs, allow_self=True)


for name, function in {
    'scatter_reduce.two': scatter_reduce, 'scatter_reduce.two_out': scatter_reduce,
    'scatter_reduce_.two': lambda a, dim, index, src, reduce, *, include_self=True: scatter_reduce(a, dim, index, src, reduce, include_self=include_self, inplace=True),
    'index_reduce': index_reduce, 'index_reduce.out': index_reduce,
    'index_reduce_': lambda a, dim, index, source, reduce, *, include_self=True: index_reduce(a, dim, index, source, reduce, include_self=include_self, inplace=True),
    'scatter.reduce': legacy_scatter, 'scatter.value_reduce': legacy_scatter,
    'scatter.reduce_out': legacy_scatter, 'scatter.value_reduce_out': legacy_scatter,
    'scatter_.reduce': lambda a, dim, index, source, *, reduce: legacy_scatter(a, dim, index, source, reduce=reduce, inplace=True),
    'scatter_.value_reduce': lambda a, dim, index, value, *, reduce: legacy_scatter(a, dim, index, value, reduce=reduce, inplace=True),
    'scatter.src_out': scatter_out, 'scatter.value_out': scatter_out,
}.items():
    _registry.impl(name, function)
