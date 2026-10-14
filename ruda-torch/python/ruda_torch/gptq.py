"""Standard GPTQ v1/v2 safetensors, retaining original codes and g_idx."""
from __future__ import annotations
import json
import math
import re
import sys
from pathlib import Path
import torch
from torch import nn
from .packed_quantization import PackedQuantizedTensor, PackedLinear, _FLOATS
from .finetuning import _selected, _replace
from .weight_loading import SafetensorWeights


def _unpack_words(words, count, bits):
    """Little-endian bit stream, including GPTQ's split 3-bit words."""
    width = words.element_size() * 8
    packed = words.to(torch.int64) & ((1 << width) - 1)
    columns = torch.arange(count, dtype=torch.int64)
    word, shift = columns * bits // width, columns * bits % width
    if word.numel() and int(word[-1]) >= packed.shape[-1]:
        raise ValueError('truncated GPTQ packed metadata')
    value = packed[:, word] >> shift
    cross = shift + bits > width
    if cross.any():
        next_word = word[cross] + 1
        if int(next_word[-1]) >= packed.shape[-1]:
            raise ValueError('truncated crossing GPTQ word')
        value[:, cross] |= packed[:, next_word] << (width - shift[cross])
    return value & ((1 << bits) - 1)


def gptq_linear(state, prefix, *, in_features, out_features, bits, group_size,
                checkpoint_format='gptq', desc_act=False, zero_point_wrap=False,
                device='ruda:0', dtype=torch.float16, chunk_rows=128, strategy='scalar'):
    """Import standard packed GPTQ (not Marlin), without requantizing weights.

    v1 stores zero-1, v2 stores zero directly. `zero_point_wrap=True` selects
    decoders that mask v1 zero+1 modulo 2**bits; this is explicit for boundary
    zero points because legacy GPTQ implementations disagree on that case.
    """
    if strategy not in ('scalar', 'tensor_core'):
        raise ValueError('packed Linear strategy must be scalar or tensor_core')
    if checkpoint_format not in ('gptq', 'gptq_v2') or bits not in (2, 3, 4, 8):
        raise ValueError('expected standard GPTQ/v2 with 2/3/4/8-bit codes')
    if sys.byteorder != 'little':
        raise NotImplementedError('GPTQ byte repacking requires a little-endian host')
    if any(type(n) is not int or n < 1 for n in (in_features, out_features, chunk_rows)) or dtype not in _FLOATS:
        raise ValueError('invalid GPTQ linear dimensions/dtype')
    group_size = in_features if group_size == -1 else group_size
    if type(group_size) is not int or group_size < 1:
        raise ValueError('GPTQ group_size must be positive or -1')
    key = lambda suffix: prefix + '.' + suffix if prefix else suffix
    weights, zeros, scales = (state[key(s)] for s in ('qweight', 'qzeros', 'scales'))
    groups = math.ceil(in_features / group_size)
    for tensor in (weights, zeros, scales):
        if tensor.device.type != 'cpu' or tensor.ndim != 2:
            raise ValueError('GPTQ source tensors must be CPU matrices')
    if weights.dtype not in (torch.int8, torch.int16, torch.int32) or zeros.dtype != weights.dtype:
        raise ValueError('GPTQ qweight/qzeros must share an integer packing dtype')
    word_bits = weights.element_size() * 8
    if weights.shape != (math.ceil(in_features * bits / word_bits), out_features):
        raise ValueError('GPTQ qweight is not the standard [packed_K,N] layout')
    if zeros.shape != (groups, math.ceil(out_features * bits / word_bits)) or scales.shape != (groups, out_features):
        raise ValueError('GPTQ scales/qzeros geometry mismatch')
    if scales.dtype not in _FLOATS or not torch.isfinite(scales).all() or (scales < 0).any():
        raise ValueError('GPTQ scales must be finite nonnegative floating values')
    index = state.get(key('g_idx'))
    if index is None:
        if desc_act:
            raise ValueError('act-order GPTQ requires its original g_idx tensor')
        index = torch.arange(in_features, dtype=torch.int32) // group_size
    if index.device.type != 'cpu' or index.dtype not in (torch.int32, torch.int64) or index.shape != (in_features,):
        raise ValueError('GPTQ g_idx must be an integer [in_features] vector')
    if (index < 0).any() or (index >= groups).any():
        raise ValueError('GPTQ g_idx points outside the scale groups')
    # Reorder only byte lanes. At no point are full floating weights materialized.
    byte_width = math.ceil(in_features * bits / 8)
    packed = torch.empty((out_features, byte_width), dtype=torch.uint8, device=device)
    for start in range(0, out_features, chunk_rows):
        raw = weights[:, start:start+chunk_rows].t().contiguous().view(torch.uint8)
        packed[start:start+raw.shape[0]].copy_(raw[:, :byte_width])
    zero = _unpack_words(zeros, out_features, bits)
    if checkpoint_format == 'gptq':
        zero = zero + 1
        if zero_point_wrap:
            zero = zero & ((1 << bits) - 1)
    scale = scales.t().float().contiguous()
    offset = -zero.t().float() * scale
    weight = PackedQuantizedTensor(packed, scale.to(device), offset.contiguous().to(device),
                                  (out_features, in_features), 'int'+str(bits), group_size, index.int().contiguous().to(device))
    bias = state.get(key('bias'))
    return PackedLinear(weight, None if bias is None else bias.to(device=device, dtype=dtype), strategy=strategy)


def _config(directory):
    root = Path(directory)
    if root.is_file():
        root = root.parent
    for filename in ('quantize_config.json', 'quant_config.json', 'config.json'):
        path = root / filename
        if path.is_file():
            data = json.loads(path.read_text(encoding='utf-8'))
            candidate = data.get('quantization_config', data if filename != 'config.json' else None)
            if candidate is not None:
                return candidate
    raise ValueError('GPTQ quantization config is missing; supply config explicitly')


def load_gptq_safetensors(model, directory, *, target_modules, config=None, device='ruda:0',
                          dtype=torch.float16, zero_point_wrap=False, chunk_rows=128, strategy='scalar'):
    """Stream standard GPTQ and remaining dense tensors into a constructed model.

    No architecture, key renaming, remote code or download is inferred. Dynamic
    per-module bits/group_size/desc_act config overrides are resolved explicitly.
    """
    if strategy not in ('scalar', 'tensor_core'):
        raise ValueError('packed Linear strategy must be scalar or tensor_core')
    config = dict(_config(directory) if config is None else config)
    if config.get('quant_method', 'gptq').lower() != 'gptq' or config.get('is_marlin_format', False):
        raise ValueError('checkpoint is not standard GPTQ')
    checkpoint_format = config.get('checkpoint_format', config.get('format', 'gptq'))
    if checkpoint_format not in ('gptq', 'gptq_v2'):
        raise ValueError('Marlin/other packed GPTQ formats require conversion before this importer')
    source = SafetensorWeights(directory)
    selected = _selected(model, target_modules, (nn.Linear,))
    originals = list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False))
    names = {name for name, _ in selected}
    selected_ids = {id(layer.weight) for _, layer in selected}
    for name, tensor in originals:
        if id(tensor) in selected_ids and name.rsplit('.', 1)[0] not in names:
            raise ValueError('selected GPTQ weight is tied to an unselected consumer')
    replaced = {}
    for name, layer in selected:
        cfg = dict(config)
        for pattern, overrides in config.get('dynamic', {}).items():
            negative = pattern.startswith('-:')
            expression = pattern[2:] if pattern.startswith(('-:', '+:')) else pattern
            if re.match(expression, name):
                if negative:
                    raise ValueError(f'{name} is selected but excluded by the GPTQ config')
                cfg.update(overrides)
                break
        if id(layer) not in replaced:
            replacement = gptq_linear(source, name, in_features=layer.in_features, out_features=layer.out_features,
                bits=cfg['bits'], group_size=cfg['group_size'], checkpoint_format=checkpoint_format,
                desc_act=cfg.get('desc_act', False), zero_point_wrap=zero_point_wrap,
                device=device, dtype=dtype, chunk_rows=chunk_rows, strategy=strategy)
            if (layer.bias is None) != (replacement.bias is None):
                raise ValueError(f'GPTQ bias presence differs from model: {name}')
            replacement.train(layer.training)
            replaced[id(layer)] = replacement
        _replace(model, [(name, replaced[id(layer)])])
    loaded = {}
    for name, original in originals:
        if any(name in (target+'.weight', target+'.bias') for target in names):
            continue
        if id(original) not in loaded:
            if name not in source and not isinstance(original, nn.Parameter) and original.device.type != 'meta':
                value = original.detach()
            else:
                value = source[name]
            if value.shape != original.shape:
                raise ValueError(f'checkpoint shape differs: {name}')
            value = value.to(device=device, dtype=dtype if value.is_floating_point() and isinstance(original, nn.Parameter) else value.dtype)
            loaded[id(original)] = nn.Parameter(value, requires_grad=False) if isinstance(original, nn.Parameter) else value
        parent, _, leaf = name.rpartition('.')
        setattr(model.get_submodule(parent), leaf, loaded[id(original)])
    return model
