"""Bounded attention dropout tiles with logical, rather than tile-local, indices."""
from __future__ import annotations

import ctypes
from threading import RLock
import torch

_lock = RLock()
_ready = False


def _native():
    global _ready
    from . import _C, _native as library
    with _lock:
        if not _ready:
            version = getattr(library, 'ruda_torch_attention_dropout_api_version', None)
            if version is None or getattr(_C, 'attention_dropout_api_version', 0) != 1:
                raise RuntimeError('indexed dropout tiles require matching rebuilt native components')
            version.restype = ctypes.c_uint32
            if version() != 1:
                raise RuntimeError('attention dropout API mismatch')
            _C.initialize_attention_dropout(ctypes.cast(library.ruda_torch_attention_dropout, ctypes.c_void_p).value)
            _ready = True
    return _C


@torch.library.custom_op('ruda::attention_dropout_mask', mutates_args=())
def mask(anchor: torch.Tensor, rng: torch.Tensor, shape: list[int], global_shape: list[int],
         offsets: list[int], dropout: float) -> torch.Tensor:
    from ._dense_ops import rng_words
    if len(shape) != 4 or len(global_shape) != 4 or len(offsets) != 3:
        raise ValueError('dropout tiles require [B,H,Q,K] sizes and head/query/key offsets')
    result = torch.empty(shape, device=anchor.device, dtype=anchor.dtype)
    seed, counter = rng_words(rng)
    _native().attention_dropout([anchor, result], [*global_shape, *offsets], dropout, seed, counter)
    return result


@mask.register_fake
def _mask_fake(anchor, rng, shape, global_shape, offsets, dropout):
    return torch.empty(shape, device=anchor.device, dtype=anchor.dtype)


def tile(probability, state, global_shape, head_start, row, column, dropout):
    shape = [global_shape[0], probability.shape[-3], probability.shape[-2], probability.shape[-1]]
    result = mask(probability.new_empty(0), state, shape, list(global_shape), [head_start, row, column], dropout)
    return result.reshape(probability.shape)
