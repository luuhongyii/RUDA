"""GGUF v2/v3 tensor import, with lossless repacking of common GGML quantizers."""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
import math
import struct
import torch
from torch import nn
from .packed_quantization import PackedQuantizedTensor, PackedLinear, _FLOATS

# ggml type -> (name, elements/block, bytes/block, integer bits, parameter group).
_QUANTS = {
    2: ('Q4_0',32,18,4,32), 3: ('Q4_1',32,20,4,32),
    6: ('Q5_0',32,22,5,32), 7: ('Q5_1',32,24,5,32),
    8: ('Q8_0',32,34,8,32), 9: ('Q8_1',32,36,8,32),
    10: ('Q2_K',256,84,2,16), 11: ('Q3_K',256,110,3,16),
    12: ('Q4_K',256,144,4,32), 13: ('Q5_K',256,176,5,32),
    14: ('Q6_K',256,210,6,16), 15: ('Q8_K',256,292,8,256),
}
_DENSE = {0:torch.float32,1:torch.float16,24:torch.int8,25:torch.int16,
          26:torch.int32,27:torch.int64,28:torch.float64,30:torch.bfloat16}
_META = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}


@dataclass(frozen=True)
class GGUFTensorInfo:
    path: Path
    name: str
    shape: tuple[int, ...]
    type: int
    offset: int
    byte_length: int
    endian: str


class _Reader:
    def __init__(self, handle, endian):
        self.file, self.endian = handle, endian
        self.size = handle.seek(0, 2)
        handle.seek(0)

    def read(self, size):
        if size < 0 or size > self.size - self.file.tell():
            raise ValueError('truncated GGUF field')
        value = self.file.read(size)
        if len(value) != size:
            raise ValueError('truncated GGUF read')
        return value

    def number(self, fmt):
        return struct.unpack(self.endian + fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self):
        return self.read(self.number('Q')).decode('utf-8')

    def value(self, kind, depth=0):
        if depth > 32:
            raise ValueError('excessively nested GGUF metadata')
        if kind in _META:
            if kind == 7:
                value = self.number('B')
                if value not in (0, 1):
                    raise ValueError('invalid GGUF boolean')
                return bool(value)
            return self.number(_META[kind])
        if kind == 8:
            return self.string()
        if kind == 9:
            element_type, length = self.number('I'), self.number('Q')
            if length > self.size - self.file.tell():
                raise ValueError('invalid GGUF array length')
            return tuple(self.value(element_type, depth+1) for _ in range(length))
        raise ValueError(f'unknown GGUF metadata type {kind}')


def _index(path):
    with path.open('rb') as handle:
        if handle.read(4) != b'GGUF':
            raise ValueError(f'not a GGUF file: {path}')
        raw = handle.read(4)
        if len(raw) != 4:
            raise ValueError('truncated GGUF version')
        endian = '<' if int.from_bytes(raw, 'little') in (2,3) else '>'
        reader = _Reader(handle, endian)
        reader.read(4)
        version = reader.number('I')
        if version not in (2,3):
            raise ValueError(f'unsupported GGUF version: {version}')
        tensor_count, metadata_count = reader.number('Q'), reader.number('Q')
        if tensor_count > reader.size // 24 or metadata_count > reader.size // 12:
            raise ValueError('invalid GGUF header counts')
        metadata = {}
        for _ in range(metadata_count):
            key = reader.string()
            if key in metadata:
                raise ValueError('duplicate GGUF metadata key')
            metadata[key] = reader.value(reader.number('I'))
        alignment = metadata.get('general.alignment', 32)
        if type(alignment) is not int or alignment < 8 or alignment % 8:
            raise ValueError('invalid GGUF data alignment')
        tensors = []
        for _ in range(tensor_count):
            name, rank = reader.string(), reader.number('I')
            if not 1 <= rank <= 4:
                raise ValueError('GGUF tensor rank must be 1..4')
            shape = tuple(reversed(tuple(reader.number('Q') for _ in range(rank))))
            kind, offset = reader.number('I'), reader.number('Q')
            if any(n < 1 for n in shape) or offset % alignment:
                raise ValueError('invalid GGUF tensor shape/offset')
            if kind in _QUANTS:
                _, block, size, _, _ = _QUANTS[kind]
                if shape[-1] % block:
                    raise ValueError('GGUF quantized row is not block-aligned')
                length = math.prod(shape) // block * size
            elif kind in _DENSE:
                length = math.prod(shape) * torch.empty((), dtype=_DENSE[kind]).element_size()
            else:
                raise NotImplementedError(f'GGUF type {kind} is outside Q4/Q5/Q8 and K-quants support')
            tensors.append((name, shape, kind, offset, length))
        base = (handle.tell() + alignment - 1) // alignment * alignment
        result = {}
        intervals = []
        for name, shape, kind, offset, length in tensors:
            if name in result or base+offset+length > reader.size:
                raise ValueError('duplicate or truncated GGUF tensor')
            result[name] = GGUFTensorInfo(path, name, shape, kind, base+offset, length, endian)
            intervals.append((offset, offset+length))
        intervals.sort()
        if any(a[1] > b[0] for a,b in zip(intervals, intervals[1:])):
            raise ValueError('overlapping GGUF tensor payloads')
        return metadata, result


def _float_field(raw, start, size=2, endian='<'):
    field = raw[:, start:start+size].contiguous()
    if endian == '>':
        field = field.flip(1).contiguous()
    return field.view(torch.float16 if size == 2 else torch.float32).float()


def _codes_and_params(raw, kind, endian):
    """Decode only integer codes and small parameter blocks, not dense weights."""
    n = raw.shape[0]
    data = raw.to(torch.int64)
    if kind in (2,3,6,7):
        five, affine = kind in (6,7), kind in (3,7)
        d = _float_field(raw,0,endian=endian)
        start = 4 if affine else 2
        offset = _float_field(raw,2,endian=endian) if affine else -d*(16 if five else 8)
        low_start = start+4 if five else start
        low = data[:,low_start:low_start+16]
        codes = torch.cat((low&15,low>>4),1)
        if five:
            high = data[:,start:start+4]
            columns = torch.arange(32)
            # qh is a byte bit mask in GGML, not an endian-dependent integer.
            codes |= ((high[:,columns//8]>>(columns%8))&1)<<4
        return codes,d,offset
    if kind in (8,9,15):
        start = 2 if kind == 8 else 4
        count = 256 if kind == 15 else 32
        d = _float_field(raw,0,4 if kind == 15 else 2,endian)
        # Bias signed storage into unsigned codes without losing -128.
        codes = (data[:,start:start+count]+128)&255
        return codes,d,-128*d
    columns = torch.arange(256)
    if kind == 10:
        scales = data[:,:16]
        codes = (data[:,16+columns//128*32+columns%32] >> ((columns%128)//32*2))&3
        d = _float_field(raw,80,endian=endian)*(scales&15).float()
        offset = -_float_field(raw,82,endian=endian)*(scales>>4).float()
        return codes,d,offset
    if kind == 11:
        group = torch.arange(16)
        scale_bits = data[:,96+group%8] >> (group//8*4)
        high_scale = (data[:,104+group%4] >> (group//4*2))&3
        scales = ((scale_bits&15)|(high_scale<<4))-32
        d = _float_field(raw,108,endian=endian)*scales.float()
        low = (data[:,32+columns//128*32+columns%32] >> ((columns%128)//32*2))&3
        high = (data[:,columns%32] >> (columns//32))&1
        codes = low | (high<<2)
        return codes,d,-4*d
    if kind in (12,13):
        s = data[:,4:16]
        scale = torch.cat((s[:,:4]&63,(s[:,8:12]&15)|((s[:,:4]>>2)&48)),1)
        minimum = torch.cat((s[:,4:8]&63,(s[:,8:12]>>4)|((s[:,4:8]>>2)&48)),1)
        d = _float_field(raw,0,endian=endian)*scale.float()
        offset = -_float_field(raw,2,endian=endian)*minimum.float()
        start = 16 if kind == 12 else 48
        codes = (data[:,start+columns//64*32+columns%32] >> ((columns%64)//32*4))&15
        if kind == 13:
            codes |= ((data[:,16+columns%32] >> (columns//32))&1)<<4
        return codes,d,offset
    if kind == 14:
        low = (data[:,columns//128*64+columns%64] >> ((columns%128)//64*4))&15
        high = (data[:,128+columns//128*32+columns%32] >> ((columns%128)//32*2))&3
        scales = raw[:,192:208].contiguous().view(torch.int8).float()
        d = _float_field(raw,208,endian=endian)*scales
        return low|(high<<4),d,-32*d
    raise ValueError('unknown GGUF quantization type')


def _pack_codes(codes, bits):
    rows, width = codes.shape
    count = (width*bits+7)//8
    byte = torch.arange(count, dtype=torch.int64)
    output = torch.zeros((rows,count),dtype=torch.int64)
    for lane in range(8):
        positions = byte*8+lane
        valid = positions < width*bits
        column = (positions//bits).clamp_max(width-1)
        output |= (((codes[:,column] >> (positions%bits))&1)*valid)<<lane
    return output.to(torch.uint8)


class GGUFWeights(Mapping):
    """Index local single/split GGUF files, reading bounded row chunks on demand.

    Supply every shard explicitly, or the directory containing only that model's
    shards. Tensor names and row permutations stay caller-owned.
    """
    def __init__(self, paths):
        if isinstance(paths,(str,Path)):
            path = Path(paths)
            paths = sorted(path.glob('*.gguf')) if path.is_dir() else [path]
        self._info, self.metadata, shard_metadata = {}, {}, []
        for path in paths:
            metadata, tensors = _index(Path(path))
            if self._info.keys() & tensors.keys():
                raise ValueError('duplicate tensor across GGUF shards')
            if self.metadata and metadata.get('general.architecture') != self.metadata.get('general.architecture'):
                raise ValueError('GGUF shards belong to different architectures')
            if not self.metadata:
                self.metadata = metadata
            self._info.update(tensors)
            shard_metadata.append(metadata)
        if not self._info:
            raise ValueError('no GGUF tensors found')
        expected = self.metadata.get('split.count',1)
        if expected != len(shard_metadata):
            raise ValueError('GGUF shard set is incomplete')
        if expected > 1:
            if {m.get('split.no') for m in shard_metadata} != set(range(expected)):
                raise ValueError('GGUF shard numbers are missing or repeated')
            total = self.metadata.get('split.tensors.count')
            if total is not None and total != len(self._info):
                raise ValueError('GGUF tensor count differs from split metadata')

    def __iter__(self):
        return iter(self._info)

    def __len__(self):
        return len(self._info)

    def __getitem__(self,name):
        return self.packed(name,device='cpu') if self._info[name].type in _QUANTS else self.tensor(name)

    def info(self,name):
        return self._info[name]

    def tensor(self,name,*,device='cpu',dtype=None):
        info = self._info[name]
        if info.type not in _DENSE:
            raise ValueError('quantized GGUF tensors use packed(), not implicit full dequantization')
        with info.path.open('rb') as handle:
            handle.seek(info.offset)
            raw = bytearray(handle.read(info.byte_length))
        if len(raw) != info.byte_length:
            raise ValueError('GGUF tensor was truncated after indexing')
        value = torch.frombuffer(raw,dtype=torch.uint8)
        target = _DENSE[info.type]
        size = torch.empty((),dtype=target).element_size()
        if info.endian == '>' and size > 1:
            value = value.reshape(-1,size).flip(1).contiguous().reshape(-1)
        return value.view(target).reshape(info.shape).to(device=device,dtype=dtype or target)

    def packed(self,name,*,device='ruda:0',chunk_rows=64,row_permutation=None):
        info = self._info[name]
        if info.type not in _QUANTS or type(chunk_rows) is not int or chunk_rows < 1:
            raise ValueError('expected a supported quantized GGUF tensor and positive chunk_rows')
        _,block,size,bits,group = _QUANTS[info.type]
        width,rows = info.shape[-1],math.prod(info.shape[:-1])
        packed = torch.empty((rows,(width*bits+7)//8),dtype=torch.uint8,device=device)
        scales = torch.empty((rows,width//group),dtype=torch.float32,device=device)
        offsets = torch.empty_like(scales)
        source_bytes = width//block*size
        with info.path.open('rb') as handle:
            handle.seek(info.offset)
            for start in range(0,rows,chunk_rows):
                count = min(chunk_rows,rows-start)
                raw = bytearray(handle.read(count*source_bytes))
                if len(raw) != count*source_bytes:
                    raise ValueError('GGUF tensor was truncated after indexing')
                codes,s,o = _codes_and_params(torch.frombuffer(raw,dtype=torch.uint8).reshape(-1,size),info.type,info.endian)
                packed[start:start+count].copy_(_pack_codes(codes.reshape(count,width),bits))
                scales[start:start+count].copy_(s.reshape(count,-1))
                offsets[start:start+count].copy_(o.reshape(count,-1))
        if row_permutation is not None:
            permutation = torch.as_tensor(row_permutation,device='cpu',dtype=torch.int64)
            if permutation.shape != (rows,) or not torch.equal(permutation.sort().values,torch.arange(rows)):
                raise ValueError('GGUF row_permutation must be a bijection over logical rows')
            permutation = permutation.to(device)
            packed,scales,offsets = (t.index_select(0,permutation) for t in (packed,scales,offsets))
        return PackedQuantizedTensor(packed,scales,offsets,info.shape,'int'+str(bits),group)


def load_gguf(model, paths, *, name_map, device='ruda:0', dtype=torch.float16,
              row_permutations=None, chunk_rows=64):
    """Load a constructed model using explicit model-parameter -> GGUF-name maps.

    Quantized Linear/Embedding weights stay packed. Q/K RoPE row permutations,
    when the model's layout differs from GGUF, are supplied in row_permutations.
    Extra GGUF tensors are allowed; every model parameter requires a mapping.
    """
    from .packed_quantization import PackedEmbedding
    source = GGUFWeights(paths)
    row_permutations = {} if row_permutations is None else row_permutations
    originals = list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False))
    replacements, handled, loaded, packed_weights = {}, set(), {}, {}
    aliases = {}
    for name,original in originals:
        if name not in name_map:
            continue
        previous = aliases.get(id(original))
        if previous is not None:
            same_rows = torch.equal(torch.as_tensor(row_permutations.get(previous,()),dtype=torch.int64),
                                    torch.as_tensor(row_permutations.get(name,()),dtype=torch.int64))
            if name_map[previous] != name_map[name] or not same_rows:
                raise ValueError('tied model tensors require the same GGUF source and row permutation')
        aliases[id(original)] = name
    for name,original in originals:
        if name in handled:
            continue
        parent,_,leaf = name.rpartition('.')
        module = model.get_submodule(parent)
        if name not in name_map:
            if isinstance(original,nn.Parameter) or original.device.type == 'meta':
                raise ValueError(f'missing explicit GGUF mapping: {name}')
            setattr(module,leaf,original.to(device))
            continue
        source_name = name_map[name]
        info = source.info(source_name)
        if info.shape != tuple(original.shape):
            raise ValueError(f'GGUF/model shape mismatch: {name}')
        if info.type in _QUANTS:
            if leaf != 'weight' or not isinstance(module,(nn.Linear,nn.Embedding)):
                raise ValueError(f'quantized GGUF weight needs a Linear/Embedding consumer: {name}')
            if id(module) not in replacements:
                if id(original) not in packed_weights:
                    packed_weights[id(original)] = source.packed(source_name,device=device,chunk_rows=chunk_rows,row_permutation=row_permutations.get(name))
                weight = packed_weights[id(original)]
                if isinstance(module,nn.Linear):
                    bias = None
                    if module.bias is not None:
                        bias_name = parent+'.bias' if parent else 'bias'
                        bias = source.tensor(name_map[bias_name],device=device,dtype=dtype)
                        handled.add(bias_name)
                    replacement = PackedLinear(weight,bias)
                else:
                    replacement = PackedEmbedding(weight,padding_idx=module.padding_idx,output_dtype=dtype)
                replacement.train(module.training)
                replacements[id(module)] = replacement
            if not parent:
                model = replacements[id(module)]
            else:
                owner,_,child = parent.rpartition('.')
                setattr(model.get_submodule(owner),child,replacements[id(module)])
            handled.add(name)
        else:
            if id(original) not in loaded:
                value = source.tensor(source_name,device=device,dtype=dtype if original.is_floating_point() and isinstance(original,nn.Parameter) else None)
                if name in row_permutations:
                    value = value.index_select(0,torch.as_tensor(row_permutations[name],dtype=torch.int64,device=device))
                loaded[id(original)] = nn.Parameter(value,requires_grad=False) if isinstance(original,nn.Parameter) else value
            setattr(module,leaf,loaded[id(original)])
    return model
