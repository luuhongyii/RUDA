"""Typed, device-preserving caches for attention and recurrent sequence layers."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from collections.abc import Mapping
import torch


class _LayerState:
    def _map(self, fn):
        def visit(value):
            if isinstance(value, torch.Tensor):
                return fn(value)
            if isinstance(value, tuple):
                return tuple(visit(item) for item in value)
            return value
        return replace(self, **{field.name: visit(getattr(self, field.name)) for field in fields(self)})

    def detach(self):
        return self._map(lambda tensor: tensor.detach())

    def clone(self):
        return self._map(lambda tensor: tensor.clone())

    def to(self, device=None):
        return self._map(lambda tensor: tensor.to(device=device))

    def reorder(self, batch_indices):
        if batch_indices.ndim != 1 or batch_indices.dtype != torch.int64:
            raise ValueError('cache batch indices must be one-dimensional int64')
        return self._map(lambda tensor: tensor.index_select(0, batch_indices.to(device=tensor.device)))

    @property
    def tensor_bytes(self):
        total = 0
        def count(tensor):
            nonlocal total
            total += tensor.numel() * tensor.element_size()
            return tensor
        self._map(count)
        return total

    def state_dict(self):
        snapshot = self.detach().clone()
        return {'kind': type(self).__name__,
                'values': {field.name: getattr(snapshot, field.name) for field in fields(self)}}


def _seen(value):
    if type(value) is not int or value < 0:
        raise ValueError('seen_tokens must be a nonnegative integer')


@dataclass(frozen=True)
class KVCache(_LayerState):
    key: torch.Tensor
    value: torch.Tensor
    seen_tokens: int

    def __post_init__(self):
        _seen(self.seen_tokens)
        if (self.key.ndim != 4 or self.value.ndim != 4 or
                self.key.shape[:3] != self.value.shape[:3] or
                self.key.device != self.value.device or self.key.dtype != self.value.dtype or
                self.seen_tokens < self.key.shape[-2]):
            raise ValueError('KV cache requires matching [batch,heads,tokens,features] tensors')

    def append(self, key, value, *, max_length=None):
        if (key.ndim != 4 or value.ndim != 4 or key.shape[:3] != value.shape[:3] or
                key.shape[:2] != self.key.shape[:2] or key.shape[-1] != self.key.shape[-1] or
                value.shape[-1] != self.value.shape[-1] or
                any(t.device != self.key.device or t.dtype != self.key.dtype for t in (key, value))):
            raise ValueError('appended KV tensors differ from the cache layout, dtype or device')
        if max_length is not None and (type(max_length) is not int or max_length <= 0):
            raise ValueError('max_length must be positive')
        keys, values = torch.cat((self.key, key), -2), torch.cat((self.value, value), -2)
        if max_length is not None and keys.shape[-2] > max_length:
            keys, values = keys[..., -max_length:, :].clone(), values[..., -max_length:, :].clone()
        return KVCache(keys, values, self.seen_tokens + key.shape[-2])

    @property
    def start_position(self):
        return self.seen_tokens - self.key.shape[-2]


@dataclass(frozen=True)
class KDAState(_LayerState):
    recurrent: torch.Tensor
    convolution: tuple[torch.Tensor, ...] = ()
    seen_tokens: int = 0

    def __post_init__(self):
        _seen(self.seen_tokens)
        if self.recurrent.ndim != 4 or not self.recurrent.is_floating_point():
            raise ValueError('KDA state must be floating [batch,value_heads,key_dim,value_dim]')
        _convolution(self.recurrent, self.convolution)


@dataclass(frozen=True)
class MambaState(_LayerState):
    recurrent: torch.Tensor
    convolution: torch.Tensor | None = None
    seen_tokens: int = 0

    def __post_init__(self):
        _seen(self.seen_tokens)
        if self.recurrent.ndim != 4 or not self.recurrent.is_floating_point():
            raise ValueError('Mamba state must be floating [batch,heads,head_dim,state_dim]')
        _convolution(self.recurrent, () if self.convolution is None else (self.convolution,))


def _convolution(state, tensors):
    if any(t.ndim != 3 or t.shape[0] != state.shape[0] or
           t.device != state.device or not t.is_floating_point() for t in tensors):
        raise ValueError('convolution caches must be same-device [batch,channels,history] tensors')


class ArchitectureCache(Mapping):
    """Named layer states; updates return a new cache and never detach gradients."""
    def __init__(self, layers=None):
        self._layers = dict(layers or {})
        if any(not isinstance(name, str) or not name or not isinstance(state, (KVCache, KDAState, MambaState))
               for name, state in self._layers.items()):
            raise ValueError('cache entries require nonempty layer names and typed states')

    def __getitem__(self, name):
        return self._layers[name]

    def __iter__(self):
        return iter(self._layers)

    def __len__(self):
        return len(self._layers)

    def updated(self, name, state):
        if name in self and type(self[name]) is not type(state):
            raise ValueError('a layer cannot change its cache kind')
        return ArchitectureCache({**self._layers, name: state})

    def reorder(self, batch_indices):
        return ArchitectureCache({name: state.reorder(batch_indices) for name, state in self.items()})

    def detach(self):
        return ArchitectureCache({name: state.detach() for name, state in self.items()})

    def to(self, device=None):
        return ArchitectureCache({name: state.to(device) for name, state in self.items()})

    @property
    def tensor_bytes(self):
        return sum(state.tensor_bytes for state in self.values())

    def state_dict(self):
        return {'format': 'ruda-architecture-cache', 'version': 1,
                'layers': {name: state.state_dict() for name, state in self.items()}}

    @classmethod
    def from_state_dict(cls, state):
        if state.get('format') != 'ruda-architecture-cache' or state.get('version') != 1:
            raise ValueError('unsupported architecture cache format')
        kinds = {kind.__name__: kind for kind in (KVCache, KDAState, MambaState)}
        layers = {}
        for name, item in state['layers'].items():
            if item['kind'] not in kinds:
                raise ValueError('unknown layer cache kind')
            layers[name] = kinds[item['kind']](**item['values']).clone()
        return cls(layers)
