"""Tensor-parallel packed projections retaining imported codes and affine tables."""
from __future__ import annotations

import torch
from torch import nn
from .packed_quantization import PackedLinear, PackedEmbedding, PackedQuantizedTensor, _bits
from .parallel_training import (copy_to_tensor_parallel, reduce_from_tensor_parallel,
    scatter_to_tensor_parallel, gather_from_tensor_parallel)


def _geometry(shape, axis, rank, world):
    if axis not in (0, 1) or type(world) is not int or world < 1 or type(rank) is not int or not 0 <= rank < world:
        raise ValueError('invalid packed TP axis or topology')
    rows, width = shape
    if shape[axis] % world:
        raise ValueError('packed TP feature dimension must divide the group size')
    return (rows // world, width) if axis == 0 else (rows, width // world)


def _tile_rows(byte_count):
    return max(1, (1 << 20) // max(1, byte_count * 16))


def _slice_codes(codes, start, width, bits):
    rows, stride = codes.shape
    count = (width * bits + 7) // 8
    byte, shift = divmod(start * bits, 8)
    if not shift:
        result = codes[:, byte:byte + count].contiguous().clone()
    else:
        result = torch.empty((rows, count), dtype=torch.uint8, device=codes.device)
        positions = torch.arange(count, device=codes.device, dtype=torch.int64) + byte
        high_positions = (positions + 1).clamp(max=stride - 1)
        valid = (positions + 1 < stride).to(torch.int32).unsqueeze(0)
        step = _tile_rows(count)
        for row in range(0, rows, step):
            source = codes[row:row + step]
            low = source.index_select(1, positions).to(torch.int32)
            high = source.index_select(1, high_positions).to(torch.int32) * valid
            result[row:row + step].copy_(((low >> shift) | (high << (8 - shift))).to(torch.uint8))
    tail = (width * bits) % 8
    if tail:
        result[:, -1:].copy_(result[:, -1:] & ((1 << tail) - 1))
    return result


def _join_codes(values, width, bits):
    rows = values[0].shape[0]
    columns = width // len(values)
    stride = (width * bits + 7) // 8
    result = values[0].new_zeros((rows, stride))
    count = (columns * bits + 7) // 8
    tail = (columns * bits) % 8
    step = _tile_rows(count)
    for rank, value in enumerate(values):
        byte, shift = divmod(rank * columns * bits, 8)
        for row in range(0, rows, step):
            source = value[row:row + step].to(torch.int32)
            if tail:
                source[:, -1:] &= (1 << tail) - 1
            target = result[row:row + step, byte:byte + count]
            target.copy_(target | ((source << shift) & 255).to(torch.uint8))
            if shift:
                high = min(count, stride - byte - 1)
                target = result[row:row + step, byte + 1:byte + 1 + high]
                target.copy_(target | (source[:, :high] >> (8 - shift)).to(torch.uint8))
    return result


def _packed_state(shape, format, group_size, groups, strategy):
    state = {'version': 1, 'shape': tuple(shape), 'format': format, 'group_size': group_size}
    if strategy != 'scalar':
        state.update(version=2, strategy=strategy)
    if groups != (shape[1] + group_size - 1) // group_size:
        state.update(version=3, parameter_groups=groups)
    return state


@torch.no_grad()
def shard_packed_linear(layer, group, *, axis):
    if type(layer) is not PackedLinear:
        raise TypeError('packed TP requires a PackedLinear, not an embedding or custom projection')
    rows, width = _geometry((layer.out_features, layer.in_features), axis, group.rank, group.world_size)
    if axis == 0:
        start = group.rank * rows
        codes, scales, offsets = (value[start:start + rows].contiguous().clone()
            for value in (layer.codes, layer.scales, layer.offsets))
        index = layer.group_index.clone()
        bias = None if layer.bias is None else layer.bias[start:start + rows].clone()
    else:
        start = group.rank * width
        codes = _slice_codes(layer.codes, start, width, _bits(layer.format))
        scales, offsets = layer.scales.clone(), layer.offsets.clone()
        index = (layer.group_index[start:start + width].contiguous().clone() if layer.group_index.numel()
            else torch.arange(start, start + width, dtype=torch.int32, device=layer.codes.device) // layer.group_size)
        bias = None
    weight = PackedQuantizedTensor(codes, scales, offsets, (rows, width), layer.format, layer.group_size, index)
    return PackedLinear(weight, bias, strategy=layer.strategy).train(layer.training)


class _ParallelPacked(nn.Module):
    axis = None

    def _setup(self, layer, group):
        self.local = shard_packed_linear(layer, group, axis=self.axis)
        self.group = group
        self.in_features, self.out_features = layer.in_features, layer.out_features
        self.format, self.group_size, self.strategy = layer.format, layer.group_size, layer.strategy
        self.parameter_groups = layer.scales.shape[1]
        self.source_indexed = bool(layer.group_index.numel())

    def get_extra_state(self):
        return {'version': 1, 'kind': 'ruda-packed-tp', 'axis': self.axis,
            'shape': (self.out_features, self.in_features), 'format': self.format,
            'group_size': self.group_size, 'parameter_groups': self.parameter_groups,
            'indexed': self.source_indexed, 'strategy': self.strategy}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('packed TP checkpoint geometry differs')


class ColumnParallelPackedLinear(_ParallelPacked):
    axis = 0

    def __init__(self, layer, group, *, gather_output=True):
        super().__init__()
        self._setup(layer, group)
        self.gather_output = bool(gather_output)
        self.local_out_features = self.local.out_features
        self.train(layer.training)

    def forward(self, value):
        output = self.local(copy_to_tensor_parallel(value, self.group))
        return gather_from_tensor_parallel(output, self.group) if self.gather_output else output


class RowParallelPackedLinear(_ParallelPacked):
    axis = 1

    def __init__(self, layer, group, *, input_is_parallel=False):
        super().__init__()
        self._setup(layer, group)
        self.input_is_parallel = bool(input_is_parallel)
        self.local_in_features = self.local.in_features
        self.register_buffer('bias', None if layer.bias is None else layer.bias.detach().clone())
        self.train(layer.training)

    def forward(self, value):
        value = value if self.input_is_parallel else scatter_to_tensor_parallel(value, self.group)
        output = reduce_from_tensor_parallel(self.local(value), self.group)
        return output if self.bias is None else output + self.bias.to(output.dtype)


class VocabParallelPackedEmbedding(_ParallelPacked):
    axis = 0

    def __init__(self, embedding, group):
        super().__init__()
        if type(embedding) is not PackedEmbedding:
            raise TypeError('packed vocabulary parallelism requires a PackedEmbedding')
        self._setup(PackedLinear(embedding.packed_weight, strategy=embedding.strategy), group)
        self.num_embeddings, self.embedding_dim = embedding.num_embeddings, embedding.embedding_dim
        self.local_vocab = self.local.out_features
        self.start = group.rank * self.local_vocab
        self.padding_idx = embedding.padding_idx
        padding = (self.padding_idx - self.start if self.padding_idx is not None
            and self.start <= self.padding_idx < self.start + self.local_vocab else None)
        self.local = PackedEmbedding(self.local.packed_weight, padding_idx=padding, output_dtype=embedding.output_dtype)
        self.local.strategy = embedding.strategy
        self.train(embedding.training)

    def get_extra_state(self):
        state = super().get_extra_state()
        state.update(embedding=True, padding_idx=self.padding_idx, output_dtype=str(self.local.output_dtype))
        return state

    def forward(self, tokens):
        outside = (tokens < self.start) | (tokens >= self.start + self.local_vocab)
        local = (tokens - self.start).masked_fill(outside, 0)
        local = local.masked_fill((tokens < 0) | (tokens >= self.num_embeddings), self.local_vocab)
        output = self.local(local).masked_fill(outside.unsqueeze(-1), 0)
        return reduce_from_tensor_parallel(output, self.group)


def packed_layout(module, prefix):
    common = dict(module.get_extra_state(), aliases=[])
    result = {prefix + 'local.' + name: dict(common, name=prefix + name, transform='packed-' + name)
        for name in ('codes', 'scales', 'offsets', 'group_index')}
    for local, transform in (('local._extra_state', 'packed-extra'), ('_extra_state', 'packed-wrapper-extra')):
        result[prefix + local] = dict(common, name=prefix + '_extra_state', transform=transform)
    if module.axis == 0 and module.local.bias is not None:
        result[prefix + 'local.bias'] = {'name': prefix + 'bias', 'aliases': [], 'shape': [module.out_features], 'axis': 0}
    if module.axis == 1 and module.bias is not None:
        result[prefix + 'bias'] = {'name': prefix + 'bias', 'aliases': [], 'shape': [module.out_features], 'replicated': True}
    return result


def _wrapper_state(spec):
    result = {key: spec[key] for key in ('version', 'kind', 'axis', 'shape', 'format',
        'group_size', 'parameter_groups', 'indexed', 'strategy')}
    result['shape'] = tuple(result['shape'])
    if spec.get('embedding'):
        result.update(embedding=True, padding_idx=spec['padding_idx'], output_dtype=spec['output_dtype'])
    return result


def _state(spec, shape, rank=None):
    result = _packed_state(shape, spec['format'], spec['group_size'], spec['parameter_groups'], spec['strategy'])
    if spec.get('embedding'):
        padding = spec['padding_idx']
        if rank is not None:
            start = rank * shape[0]
            padding = padding - start if padding is not None and start <= padding < start + shape[0] else None
        result.update(padding_idx=padding, output_dtype=spec['output_dtype'])
    return result


def _expect_tensor(value, shape, dtype):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape) or value.dtype != dtype:
        raise ValueError('packed TP checkpoint buffer shape/dtype differs')


def _same(values):
    first = values[0].contiguous().view(torch.uint8)
    if any(not torch.equal(first, value.contiguous().view(torch.uint8)) for value in values[1:]):
        raise ValueError('replicated packed TP parameter tables differ')
    return values[0]


def join_packed_field(values, spec):
    shape = tuple(spec['shape'])
    local = _geometry(shape, spec['axis'], 0, len(values))
    kind = spec['transform']
    if kind in ('packed-extra', 'packed-wrapper-extra'):
        for rank, value in enumerate(values):
            expected = _state(spec, local, rank) if kind == 'packed-extra' else _wrapper_state(spec)
            if value != expected:
                raise ValueError('packed TP checkpoint metadata differs')
        return _state(spec, shape)
    rows, width = local
    groups = spec['parameter_groups']
    if kind == 'packed-codes':
        bits = _bits(spec['format'])
        for value in values:
            _expect_tensor(value, (rows, (width * bits + 7) // 8), torch.uint8)
        return torch.cat(values, 0) if spec['axis'] == 0 else _join_codes(values, shape[1], bits)
    if kind in ('packed-scales', 'packed-offsets'):
        for value in values:
            _expect_tensor(value, (rows, groups), torch.float32)
        return torch.cat(values, 0) if spec['axis'] == 0 else _same(values)
    if kind != 'packed-group_index':
        raise ValueError('unknown packed TP checkpoint transform')
    for rank, value in enumerate(values):
        _expect_tensor(value, (width if spec['axis'] == 1 or spec['indexed'] else 0,), torch.int32)
        if value.numel() and (bool((value < 0).any()) or bool((value >= groups).any())):
            raise ValueError('packed TP group index exceeds the original table')
        if spec['axis'] == 1 and not spec['indexed']:
            expected = torch.arange(rank * width, (rank + 1) * width, dtype=torch.int32, device=value.device) // spec['group_size']
            if not torch.equal(value, expected):
                raise ValueError('packed TP contiguous-group mapping differs')
    if not spec['indexed']:
        return values[0].new_empty((0,))
    return _same(values) if spec['axis'] == 0 else torch.cat(values)


def slice_packed_field(value, spec, rank, world):
    shape = tuple(spec['shape'])
    rows, width = _geometry(shape, spec['axis'], rank, world)
    kind = spec['transform']
    if kind in ('packed-extra', 'packed-wrapper-extra'):
        if value != _state(spec, shape):
            raise ValueError('full packed checkpoint metadata differs')
        return _state(spec, (rows, width), rank) if kind == 'packed-extra' else _wrapper_state(spec)
    if kind == 'packed-codes':
        bits = _bits(spec['format'])
        _expect_tensor(value, (shape[0], (shape[1] * bits + 7) // 8), torch.uint8)
        return (value[rank * rows:(rank + 1) * rows].contiguous().clone() if spec['axis'] == 0
            else _slice_codes(value, rank * width, width, bits))
    if kind in ('packed-scales', 'packed-offsets'):
        _expect_tensor(value, (shape[0], spec['parameter_groups']), torch.float32)
        return value[rank * rows:(rank + 1) * rows].contiguous().clone() if spec['axis'] == 0 else value.clone()
    if kind != 'packed-group_index':
        raise ValueError('unknown packed TP checkpoint transform')
    _expect_tensor(value, (shape[1] if spec['indexed'] else 0,), torch.int32)
    if value.numel() and (bool((value < 0).any()) or bool((value >= spec['parameter_groups']).any())):
        raise ValueError('packed checkpoint group index exceeds the parameter table')
    if spec['axis'] == 0:
        return value.clone()
    if not spec['indexed']:
        return torch.arange(rank * width, (rank + 1) * width, dtype=torch.int32, device=value.device) // spec['group_size']
    return value[rank * width:(rank + 1) * width].contiguous().clone()


def full_packed_state(module):
    result = {}
    local = module.state_dict()
    for key, spec in packed_layout(module, '').items():
        if spec.get('transform') in ('packed-extra', 'packed-wrapper-extra'):
            continue
        value = local[key]
        if value.numel():
            gathered = module.group.all_gather(value, axis=0).cpu()
            pieces = list(gathered.chunk(module.group.world_size, 0))
        else:
            pieces = [value.detach().cpu().clone() for _ in range(module.group.world_size)]
        if 'transform' in spec:
            value = join_packed_field(pieces, spec)
        else:
            value = _same(pieces) if spec.get('replicated') else torch.cat(pieces, spec['axis'])
        result[spec['name']] = value
    result['_extra_state'] = _state(module.get_extra_state(), (module.out_features, module.in_features))
    return result


def local_packed_state(module, state, prefix):
    result = {}
    for key, spec in packed_layout(module, prefix).items():
        if spec['name'] not in state:
            continue
        value = state[spec['name']]
        if 'transform' in spec:
            value = slice_packed_field(value, spec, module.group.rank, module.group.world_size)
        elif not spec.get('replicated'):
            _expect_tensor(value, spec['shape'], module.local.bias.dtype)
            value = value.chunk(module.group.world_size, spec['axis'])[module.group.rank].contiguous()
        result[key] = value
    return result
