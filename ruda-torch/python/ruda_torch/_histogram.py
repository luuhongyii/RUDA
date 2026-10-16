import math
import struct

import torch

from . import _C
from ._indexing import _logical
from ._ops import _dtypes
from ._ordering import _output, _outputs
from ._pooling import _nondeterministic


def _extent(size):
    if not 0 <= size <= 2**32 - 1:
        raise RuntimeError('histogram extent exceeds native U32 indexing')
    return size


def _bounds_scalar(value, dtype):
    if isinstance(value, complex):
        raise RuntimeError('histc range must be real-valued')
    if dtype == torch.float64:
        return float(value)
    if dtype == torch.float32:
        try:
            return struct.unpack('f', struct.pack('f', float(value)))[0]
        except OverflowError:
            return math.copysign(math.inf, float(value))
    result = int(value)
    if result < -(2**63) or result >= 2**63:
        raise RuntimeError('histc integer range exceeds int64')
    return result


def histc(a, bins=100, min=0, max=0, *, out=None):
    allowed = (torch.float32, torch.float64, torch.int64, torch.int32,
               torch.int16, torch.int8, torch.uint8)
    if a.layout != torch.strided or a.dtype not in allowed:
        raise RuntimeError('histc requires float32/float64 or integral input other than bool')
    if a.is_floating_point():
        _nondeterministic('_histc_cuda with floating point input')
    if bins <= 0:
        raise RuntimeError('histc bins must be positive')
    _extent(bins)
    left, right = _bounds_scalar(min, a.dtype), _bounds_scalar(max, a.dtype)
    value = _logical(a)
    if left == right and a.numel():
        low, high = torch.aminmax(value)
        left, right = low.item(), high.item()
    if left == right:
        left, right = _bounds_scalar(left - 1, a.dtype), _bounds_scalar(right + 1, a.dtype)
    if not math.isfinite(left) or not math.isfinite(right) or not left < right:
        raise RuntimeError('histc range must be finite with max greater than min')
    dtype = a.dtype if a.is_floating_point() else torch.int64
    counts = torch.zeros((bins,), dtype=dtype, device=a.device)
    if a.numel():
        integers, scalars = ([], [left, right]) if a.is_floating_point() else ([left, right], [])
        _C.scientific_call(350, [value, counts], integers, scalars)
    result = counts.to(a.dtype)
    if out is None:
        return result
    if out.device != a.device:
        raise RuntimeError('histc output must be on the input device')
    _C.check_inplace(out, a)
    if out.shape != result.shape:
        _C.prepare_index_output(out, list(result.shape))
    out.copy_(result)
    return out


def _shape(a):
    if a.layout != torch.strided or a.ndim < 2:
        raise RuntimeError('histogramdd input must be strided with rank >= 2')
    return math.prod(a.shape[:-1]), a.shape[-1]


def _check(a, bins, weight):
    samples, dimensions = _shape(a)
    if a.dtype not in _dtypes:
        raise RuntimeError('histogram requires float16, bfloat16, float32 or float64 input')
    if len(bins) != dimensions:
        raise RuntimeError('histogramdd requires one bin-edge tensor per input coordinate')
    shape = []
    for edge in bins:
        if (edge.device != a.device or edge.dtype != a.dtype or edge.layout != torch.strided
                or edge.ndim != 1 or edge.numel() < 2):
            raise RuntimeError('histogram bin edges must be matching one-dimensional tensors with at least two edges')
        shape.append(edge.numel() - 1)
    if weight is not None:
        weight_shape = tuple(weight.shape) if weight.ndim else (1,)
        if (weight.dtype != a.dtype or weight.device != a.device or weight.layout != torch.strided
                or weight_shape != tuple(a.shape[:-1])):
            raise RuntimeError('histogram weight must match input dtype/device and shape excluding the final dimension')
    _extent(samples)
    _extent(math.prod(shape))
    _extent(sum(n + 1 for n in shape))
    return samples, shape


def _make_edges(a, bins, range):
    samples, dimensions = _shape(a)
    if len(bins) != dimensions:
        raise RuntimeError('histogramdd bin counts must match the final input dimension')
    for count in bins:
        if count < -1:
            raise RuntimeError('histogram bin edge count cannot be negative')
        _extent(count + 1)
    _extent(sum(count + 1 for count in bins))
    if range is not None:
        if len(range) != 2 * dimensions:
            raise RuntimeError('histogramdd range requires two values per dimension')
        limits = [(float(range[2 * d]), float(range[2 * d + 1])) for d, _ in enumerate(bins)]
    elif a.numel():
        if a.dtype not in (torch.float32, torch.float64):
            raise RuntimeError('histogram automatic nonempty range requires float32/float64 input')
        low, high = torch.aminmax(_logical(a).reshape(samples, dimensions), dim=0)
        limits = [(float(low[d].item()), float(high[d].item())) for d, _ in enumerate(bins)]
    else:
        limits = [(0.0, 1.0)] * dimensions
    result = []
    for count, (left, right) in zip(bins, limits):
        if not math.isfinite(left) or not math.isfinite(right) or left > right:
            raise RuntimeError('histogram range must be finite with min <= max')
        if left == right:
            left, right = left - 0.5, right + 0.5
        result.append(torch.linspace(left, right, count + 1, dtype=a.dtype, device=a.device))
    return result


def _histogram(a, bins, weight, density, *, linear):
    samples, shape = _check(a, bins, weight)
    hist = a.new_zeros(shape)
    edges = torch.cat([_logical(edge) for edge in bins]) if bins else a.new_empty((0,))
    if samples and bins:
        keys = torch.empty((samples,), device=a.device, dtype=torch.int64)
        source = _logical(a).reshape(samples, len(bins))
        _C.scientific_call(351, [source, edges, keys], [int(linear), *shape], [])
        ordered = torch.are_deterministic_algorithms_enabled()
        if ordered:
            keys, order = torch.sort(keys, stable=True)
        else:
            order = keys.new_empty((0,))
        weights = a.new_empty((0,)) if weight is None else _logical(weight).reshape(samples)
        _C.scientific_call(352, [keys, order, weights, hist.reshape(-1)],
                           [int(weight is not None), int(ordered)], [])
    if density:
        total = hist.sum()
        _C.scientific_call(353, [hist.reshape(-1), total, edges], shape, [])
    return hist


def from_bin_tensors(a, bins, *, weight=None, density=False, out=None):
    result = _histogram(a, bins, weight, density, linear=False)
    return _output(result, out, (a, weight, *bins))


def from_bin_counts(a, bins, *, range=None, weight=None, density=False, out=None):
    if any(count <= 0 for count in bins):
        raise RuntimeError('histogram bin counts must be positive')
    _extent(math.prod(bins))
    edges = _make_edges(a, bins, range)
    result = _histogram(a, edges, weight, density, linear=True)
    return _output(result, out, (a, weight))


def bin_edges(a, bins, *, range=None, weight=None, density=False):
    return _make_edges(a, bins, range)


def bin_edges_out(a, bins, *, range=None, weight=None, density=False, out):
    if len(out) != len(bins):
        raise RuntimeError('histogram bin edge outputs must match the dimension count')
    _outputs(_make_edges(a, bins, range), out, (a, weight))


def _one_dimensional(a, bins, range, weight, density, hist, bin_edges):
    original = a, weight, bins if isinstance(bins, torch.Tensor) else None
    source = a.reshape(a.numel(), 1)
    weights = None if weight is None else weight.reshape(-1)
    if isinstance(bins, torch.Tensor):
        edges = [_logical(bins).clone()]
        linear = False
    else:
        if bins <= 0:
            raise RuntimeError('histogram bins must be positive')
        edges = _make_edges(source, [bins], range)
        linear = True
    result = _histogram(source, edges, weights, density, linear=linear)
    outputs = None
    if hist is not None or bin_edges is not None:
        if hist is None or bin_edges is None:
            raise RuntimeError('histogram out requires both hist and bin_edges')
        outputs = hist, bin_edges
    return _outputs((result, edges[0]), outputs, original)


def histogram_bins(a, bins, *, weight=None, density=False, hist=None, bin_edges=None):
    return _one_dimensional(a, bins, None, weight, density, hist, bin_edges)


def histogram_counts(a, bins=100, *, range=None, weight=None, density=False, hist=None, bin_edges=None):
    return _one_dimensional(a, bins, range, weight, density, hist, bin_edges)


def register(library):
    for name, function in (('histc', histc), ('histc.out', histc),
                           ('histogram.bins_tensor', histogram_bins),
                           ('histogram.bins_tensor_out', histogram_bins),
                           ('histogram.bin_ct', histogram_counts),
                           ('histogram.bin_ct_out', histogram_counts),
                           ('_histogramdd_from_bin_tensors', from_bin_tensors),
                           ('_histogramdd_from_bin_tensors.out', from_bin_tensors),
                           ('_histogramdd_from_bin_cts', from_bin_counts),
                           ('_histogramdd_from_bin_cts.out', from_bin_counts),
                           ('_histogramdd_bin_edges', bin_edges),
                           ('_histogramdd_bin_edges.out', bin_edges_out)):
        library.impl(name, function)
