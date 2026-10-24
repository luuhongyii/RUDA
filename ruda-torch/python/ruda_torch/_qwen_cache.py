"""Growable, preallocated KV storage with the Transformers Cache interface."""
from __future__ import annotations

import torch
from transformers.cache_utils import Cache, DynamicLayer


class _Layer(DynamicLayer):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity
        self.key_storage = self.value_storage = None

    def lazy_initialization(self, keys, values):
        self.dtype, self.device = keys.dtype, keys.device
        self.key_storage = keys.new_empty((*keys.shape[:2], self.capacity, keys.shape[-1]))
        self.value_storage = values.new_empty((*values.shape[:2], self.capacity, values.shape[-1]))
        self.keys = self.key_storage[..., :0, :]
        self.values = self.value_storage[..., :0, :]
        self.is_initialized = True

    def ensure_storage(self, required=0):
        if not self.is_initialized:
            return
        length = self.get_seq_length()
        owned = (self.key_storage is not None and self.value_storage is not None
            and self.keys.untyped_storage().data_ptr() == self.key_storage.untyped_storage().data_ptr()
            and self.values.untyped_storage().data_ptr() == self.value_storage.untyped_storage().data_ptr()
            and self.keys.shape[:2] == self.key_storage.shape[:2])
        if owned and required <= self.capacity:
            return
        capacity = max(required, self.capacity * 2 if required > self.capacity else self.capacity)
        keys = self.keys.new_empty((*self.keys.shape[:2], capacity, self.keys.shape[-1]))
        values = self.values.new_empty((*self.values.shape[:2], capacity, self.values.shape[-1]))
        if length:
            keys[..., :length, :].copy_(self.keys)
            values[..., :length, :].copy_(self.values)
        self.capacity, self.key_storage, self.value_storage = capacity, keys, values
        self.publish(length)

    def publish(self, length):
        self.keys = self.key_storage[..., :length, :]
        self.values = self.value_storage[..., :length, :]

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        begin = self.get_seq_length()
        end = begin + key_states.shape[-2]
        self.ensure_storage(end)
        self.key_storage[..., begin:end, :].copy_(key_states)
        self.value_storage[..., begin:end, :].copy_(value_states)
        self.publish(end)
        return self.keys, self.values

    def reset(self):
        super().reset()
        self.key_storage = self.value_storage = None


class Qwen2InferenceCache(Cache):
    def __init__(self, layers, capacity):
        super().__init__(layers=[_Layer(capacity) for _ in range(layers)])
        self._ruda_graph = None
        self._ruda_graph_key = None
        self._ruda_graph_length = None
        self._ruda_graph_logits = None
        self.graph_replays = 0

    def prepare_decode(self):
        length = self.get_seq_length()
        for layer in self.layers:
            if not layer.is_initialized or layer.get_seq_length() != length:
                return None
            layer.ensure_storage(length + 1)
        return length

    def publish_decode(self, length):
        for layer in self.layers:
            layer.publish(length)
        self._ruda_graph_length = length
        self.graph_replays += 1

    def close_graph(self):
        if self._ruda_graph is not None:
            self._ruda_graph.close()
        self._ruda_graph = self._ruda_graph_key = self._ruda_graph_length = None
        self._ruda_graph_logits = None

    def reset(self):
        self.close_graph()
        return super().reset()

    def crop(self, tokens_to_remove):
        self.close_graph()
        return super().crop(tokens_to_remove)

    def reorder_cache(self, beam_idx):
        self.close_graph()
        return super().reorder_cache(beam_idx)

    def batch_repeat_interleave(self, repeats):
        self.close_graph()
        return super().batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices):
        self.close_graph()
        return super().batch_select_indices(indices)
