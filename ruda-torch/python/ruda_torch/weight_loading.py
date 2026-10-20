"""Streaming safetensors loading with explicit, model-independent layout mappings."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
import math
import torch


@dataclass(frozen=True)
class TensorSource:
    name: str
    selection: tuple = ()

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('source tensor name must be nonempty')
        if not isinstance(self.selection, tuple) or any(type(index) is not int and not isinstance(index, slice) for index in self.selection):
            raise ValueError('source selection must be a tuple of integer/slice indices')


@dataclass(frozen=True)
class WeightMapping:
    target: str
    sources: tuple
    combine: str = 'single'
    axis: int = 0
    permute: tuple[int, ...] | None = None
    reshape: tuple[int, ...] | None = None

    def __post_init__(self):
        if not isinstance(self.target, str) or not self.target or not isinstance(self.sources, tuple) or not self.sources:
            raise ValueError('weight mapping needs a target and a nonempty tuple of sources')
        if any(not isinstance(source, (str, TensorSource)) for source in self.sources):
            raise TypeError('sources must be tensor names or TensorSource selections')
        if self.combine not in ('single', 'concat', 'stack') or (self.combine == 'single' and len(self.sources) != 1):
            raise ValueError('combine must be single, concat or stack with a valid source count')
        if type(self.axis) is not int:
            raise TypeError('combine axis must be an integer')


class SafetensorWeights(Mapping):
    """Index local shards without loading their payloads; open one source at a time."""
    def __init__(self, path):
        from safetensors import safe_open
        root = Path(path)
        if root.is_file():
            if root.suffix != '.safetensors':
                raise ValueError('supply a safetensors file or an indexed checkpoint directory')
            with safe_open(str(root), framework='pt', device='cpu') as source:
                self._files = {name: root for name in source.keys()}
        else:
            from .quantization_interop import _safetensor_index
            self._files = _safetensor_index(root)
        self._shapes = {}

    def __iter__(self):
        return iter(self._files)

    def __len__(self):
        return len(self._files)

    def __contains__(self, name):
        return name in self._files

    def __getitem__(self, name):
        return self.read(TensorSource(name))

    def shape(self, name):
        from safetensors import safe_open
        if name not in self._shapes:
            with safe_open(str(self._files[name]), framework='pt', device='cpu') as handle:
                self._shapes[name] = tuple(handle.get_slice(name).get_shape())
        return self._shapes[name]

    def read(self, source):
        from safetensors import safe_open
        source = TensorSource(source) if isinstance(source, str) else source
        _slice_shape(self.shape(source.name), source.selection)
        with safe_open(str(self._files[source.name]), framework='pt', device='cpu') as handle:
            if not source.selection:
                return handle.get_tensor(source.name)
            return handle.get_slice(source.name)[source.selection]


def _slice_shape(shape, selection):
    if len(selection) > len(shape):
        raise ValueError('too many source tensor indices')
    output = []
    for size, index in zip(shape, selection + (slice(None),) * (len(shape) - len(selection))):
        if type(index) is int:
            if not -size <= index < size:
                raise ValueError('source tensor index is out of range')
        else:
            if index.step is not None and (type(index.step) is not int or index.step <= 0):
                raise ValueError('source slices require positive strides')
            start, stop, step = index.indices(size)
            output.append(len(range(start, stop, step)))
    return tuple(output)


def _sources(mapping):
    return tuple(TensorSource(source) if isinstance(source, str) else source for source in mapping.sources)


def _mapped_shape(mapping, reader):
    shapes = [_slice_shape(reader.shape(source.name), source.selection) for source in _sources(mapping)]
    shape = shapes[0]
    if mapping.combine != 'single':
        rank = len(shape)
        dimensions = rank + int(mapping.combine == 'stack')
        if not -dimensions <= mapping.axis < dimensions:
            raise ValueError('combine axis is outside the tensor rank')
        axis = mapping.axis % dimensions
        if mapping.combine == 'stack':
            if any(other != shape for other in shapes):
                raise ValueError('stack sources must have identical shapes')
            shape = (*shape[:axis], len(shapes), *shape[axis:])
        else:
            if any(len(other) != rank or any(other[i] != shape[i] for i in range(rank) if i != axis) for other in shapes):
                raise ValueError('concat sources differ outside the concatenation axis')
            shape = (*shape[:axis], sum(other[axis] for other in shapes), *shape[axis+1:])
    if mapping.permute is not None:
        if len(mapping.permute) != len(shape) or sorted(mapping.permute) != list(range(len(shape))):
            raise ValueError('permute must contain each tensor axis exactly once')
        shape = tuple(shape[axis] for axis in mapping.permute)
    if mapping.reshape is not None:
        requested = mapping.reshape
        if (any(type(size) is not int or size < -1 for size in requested) or requested.count(-1) > 1):
            raise ValueError('invalid reshape dimensions')
        size, known = math.prod(shape), math.prod(d for d in requested if d != -1)
        if -1 in requested:
            if known == 0 or size % known:
                raise ValueError('reshape has an ambiguous or incompatible inferred dimension')
            shape = tuple(size // known if d == -1 else d for d in requested)
        else:
            if known != size:
                raise ValueError('reshape changes the tensor element count')
            shape = tuple(requested)
    return shape


@dataclass(frozen=True)
class WeightLoadResult:
    loaded: tuple[str, ...]
    missing: tuple[str, ...]
    unused_sources: tuple[str, ...]
    tensor_bytes: int


def load_safetensor_weights(model, path, *, mappings=None, device=None, strict=True):
    """Load into existing parameters/buffers, preserving ties and optimizer identities.

    With no mappings, checkpoint names must match state_dict names. Explicit
    mappings support sliced/split weights, concatenated QKV, stacked experts,
    transposition, reshape and tensor-parallel slices without family inference.
    Destination tensor dtype controls conversion. Meta tensors require device;
    materialized tensors remain on their existing device. Quantized formats need
    their own decoder and are not inferred. Peak payload memory scales with one
    mapped destination, rather than the whole checkpoint. No network access.
    """
    reader = path if isinstance(path, SafetensorWeights) else SafetensorWeights(path)
    state = model.state_dict(keep_vars=True)
    if any(not isinstance(tensor, torch.Tensor) for tensor in state.values()):
        raise ValueError('load non-tensor module extra state through its own explicit loader')
    aliases = {}
    for name, tensor in state.items():
        aliases.setdefault(id(tensor), []).append(name)
    if mappings is None:
        mappings = [WeightMapping(name, (name,)) for name in state if name in reader]
    else:
        mappings = list(mappings)
    seen_targets, used_sources, operations = set(), set(), []
    for mapping in mappings:
        if not isinstance(mapping, WeightMapping) or mapping.target not in state:
            raise ValueError('mapping target is not a persistent model tensor')
        target = state[mapping.target]
        if id(target) in seen_targets:
            raise ValueError('multiple mappings target the same tied parameter; map one canonical alias')
        seen_targets.add(id(target))
        sources = _sources(mapping)
        absent = [source.name for source in sources if source.name not in reader]
        if absent:
            raise ValueError(f'missing checkpoint tensors: {absent}')
        shape = _mapped_shape(mapping, reader)
        if shape != tuple(target.shape):
            raise ValueError(f'{mapping.target}: mapped shape {shape} differs from {tuple(target.shape)}')
        destination = torch.device(device) if device is not None else target.device
        if destination.type == 'meta' or (target.device.type != 'meta' and destination != target.device):
            raise ValueError('meta weights require a real device; existing tensors cannot be implicitly moved')
        used_sources.update(source.name for source in sources)
        operations.append((mapping, target, destination))
    missing = tuple(name for name, tensor in state.items() if id(tensor) not in seen_targets)
    unused = tuple(sorted(set(reader) - used_sources))
    if strict and (missing or unused):
        raise ValueError(f'checkpoint mapping mismatch: missing={missing}, unused={unused}')
    loaded, total_bytes = [], 0
    with torch.no_grad():
        for mapping, target, destination in operations:
            pieces = [reader.read(source) for source in _sources(mapping)]
            value = pieces[0] if mapping.combine == 'single' else (
                torch.cat(pieces, dim=mapping.axis) if mapping.combine == 'concat' else torch.stack(pieces, dim=mapping.axis))
            del pieces
            if mapping.permute is not None:
                value = value.permute(mapping.permute)
            if mapping.reshape is not None:
                value = value.reshape(mapping.reshape)
            value = value.to(device=destination, dtype=target.dtype)
            if target.device.type == 'meta':
                replacement = torch.nn.Parameter(value, requires_grad=target.requires_grad) if isinstance(target, torch.nn.Parameter) else value
                for alias in aliases[id(target)]:
                    parent_name, _, leaf = alias.rpartition('.')
                    parent = model.get_submodule(parent_name) if parent_name else model
                    if leaf in parent._parameters:
                        parent._parameters[leaf] = replacement
                    elif leaf in parent._buffers:
                        parent._buffers[leaf] = replacement
                    else:
                        raise ValueError(f'meta target {alias} is not a registered parameter or buffer')
            else:
                target.copy_(value)
            loaded.extend(aliases[id(target)])
            total_bytes += target.numel() * target.element_size()
            del value
    return WeightLoadResult(tuple(loaded), missing, unused, total_bytes)
