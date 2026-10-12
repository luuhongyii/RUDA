"""Quantized physical-page caches, with device-side writes and direct attention."""
from __future__ import annotations
import math
import torch
from torch import nn
from .packed_quantization import PackedQuantizedTensor, quantize_packed, FORMATS, _bits


class QuantizedKVCache(nn.Module):
    """Inference cache; only newly supplied tokens are calibrated and packed.

    Unwritten slots represent zero. Schedules must not expose unwritten history.
    Writes require distinct physical slots, like the nonquantized paged cache.
    """
    def __init__(self, *, num_pages, page_size, kv_heads, key_dim, value_dim=None,
                 format='int8', value_format=None, group_size=32, symmetric=True, device='ruda:0'):
        super().__init__()
        value_dim = key_dim if value_dim is None else value_dim
        value_format = format if value_format is None else value_format
        dims = (num_pages, page_size, kv_heads, key_dim, value_dim, group_size)
        if any(type(n) is not int or n <= 0 for n in dims) or format not in FORMATS or value_format not in FORMATS:
            raise ValueError('invalid quantized KV geometry/format')
        self.num_pages, self.page_size, self.kv_heads = num_pages, page_size, kv_heads
        self.key_dim, self.value_dim = key_dim, value_dim
        self.format, self.value_format = format, value_format
        self.group_size, self.symmetric = group_size, symmetric
        rows = num_pages * page_size * kv_heads
        for prefix, width, fmt in (('k', key_dim, format), ('v', value_dim, value_format)):
            self.register_buffer(prefix+'_codes', torch.zeros((rows, (width*_bits(fmt)+7)//8), dtype=torch.uint8, device=device))
            self.register_buffer(prefix+'_scales', torch.zeros((rows, (width+group_size-1)//group_size), dtype=torch.float32, device=device))
            self.register_buffer(prefix+'_offsets', torch.zeros((rows, (width+group_size-1)//group_size), dtype=torch.float32, device=device))

    def _weight(self, prefix, width, fmt):
        return PackedQuantizedTensor(*(getattr(self, prefix+'_'+name) for name in ('codes','scales','offsets')),
            (self.num_pages, self.page_size, self.kv_heads, width), fmt, self.group_size)

    @property
    def key(self):
        return self._weight('k', self.key_dim, self.format)

    @property
    def value(self):
        return self._weight('v', self.value_dim, self.value_format)

    def _apply(self, fn, recurse=True):
        metadata = {name: buf for name, buf in self.named_buffers() if not name.endswith('_codes')}
        super()._apply(fn, recurse)
        for name, original in metadata.items():
            setattr(self, name, original.to(device=getattr(self, name).device, dtype=torch.float32))
        return self

    def get_extra_state(self):
        return {name: getattr(self, name) for name in ('num_pages','page_size','kv_heads','key_dim','value_dim',
                                                       'format','value_format','group_size','symmetric')}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('quantized KV cache checkpoint geometry differs')

    @torch.no_grad()
    def update_(self, slots, key, value):
        """Write [tokens,Hkv,D] at unique flat physical `page*page_size+offset` slots."""
        if slots.ndim != 1 or slots.dtype not in (torch.int32, torch.int64) or slots.device != self.k_codes.device:
            raise ValueError('KV slots must be a device integer vector')
        if key.shape != (slots.numel(), self.kv_heads, self.key_dim) or value.shape != (slots.numel(), self.kv_heads, self.value_dim):
            raise ValueError('quantized KV update shape mismatch')
        if key.device != self.k_codes.device or value.device != key.device:
            raise ValueError('quantized KV update device mismatch')
        if not slots.numel():
            return self
        for prefix, data, fmt in (('k', key, self.format), ('v', value, self.value_format)):
            packed = quantize_packed(data, format=fmt, group_size=self.group_size, symmetric=self.symmetric)
            for name in ('codes', 'scales', 'offsets'):
                target = getattr(self, prefix+'_'+name)
                source = getattr(packed, name)
                target.reshape(self.num_pages*self.page_size, -1).index_copy_(0, slots.long(), source.reshape(slots.numel(), -1))
        return self

    @torch.no_grad()
    def copy_pages_(self, source_pages, destination_pages):
        """Copy-on-write page duplication, including all quantization metadata."""
        if source_pages.shape != destination_pages.shape or source_pages.ndim != 1:
            raise ValueError('page index vectors must have matching shapes')
        for name, tensor in self.named_buffers():
            pages = tensor.reshape(self.num_pages, -1)
            pages.index_copy_(0, destination_pages.long(), pages.index_select(0, source_pages.long()))
        return self

    def attention(self, plan, query, *, scale, causal=True):
        return quantized_paged_attention(plan, query, self.key, self.value, scale=scale, causal=causal)


def quantized_paged_attention(plan, q, k, v, *, scale, causal=True, position_query=None, position_cache=None):
    """MHA/GQA, or MLA when positional Q/cache are supplied. No dense KV expansion."""
    if not isinstance(k, PackedQuantizedTensor) or not isinstance(v, PackedQuantizedTensor):
        raise TypeError('quantized attention expects packed K and V')
    if (position_query is None) != (position_cache is None):
        raise ValueError('quantized MLA needs both positional operands')
    if torch.is_grad_enabled() and (q.requires_grad or position_query is not None and position_query.requires_grad):
        raise RuntimeError('quantized KV attention is inference-only; use floating caches for training')
    if q.device != k.codes.device or q.device != v.codes.device or q.device.type != 'ruda':
        raise ValueError('quantized attention device mismatch')
    if len(k.shape) != 4 or len(v.shape) != 4 or k.shape[:3] != v.shape[:3] or k.shape[:2] != (plan._spec[1], plan._spec[0]):
        raise ValueError('quantized KV geometry does not match the page schedule')
    if position_cache is not None:
        if not isinstance(position_cache, PackedQuantizedTensor) or k.shape[2] != 1 or position_cache.shape[:3] != k.shape[:3]:
            raise ValueError('quantized MLA cache geometry mismatch')
        p, qp = position_cache, position_query.contiguous()
    else:
        p, qp = k, q
    for cache in (k, v, p):
        if cache.group_index.numel():
            raise ValueError('KV cache quantization requires contiguous feature groups')
    layout = [FORMATS[k.format], k.group_size, k.shape[-1], k.shape[2],
              FORMATS[v.format], v.group_size, v.shape[-1], FORMATS[p.format], p.group_size,
              p.shape[-1] if position_cache is not None else 0]
    caches = [t for cache in (k, v, p) for t in (cache.codes, cache.scales, cache.offsets)]
    return plan._native(q).run_quantized(q.contiguous(), qp.contiguous(), caches, layout, float(scale), bool(causal))
