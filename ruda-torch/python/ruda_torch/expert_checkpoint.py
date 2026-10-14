"""Stream only owned experts from ordinary floating safetensors checkpoints."""
from __future__ import annotations

import torch
from torch import nn
from .expert_parallel import ExpertParallelMLP


def load_expert_parallel_safetensors(model, directory, *, device, tile_rows=128):
    """Load an EP-partitioned model, including a model constructed on meta.

    Router/trunk tensors keep their declared dtype; tied tensors keep sharing.
    Stacked expert tensors are sliced by expert ID and separate expert matrices
    are read only for this owner. CPU staging is bounded to tile_rows of one
    matrix. Non-checkpoint buffers must already contain their derived values,
    as in the existing streaming TP loader. No network download is performed.
    """
    from safetensors import safe_open
    from .quantization_interop import _safetensor_index
    if type(tile_rows) is not int or tile_rows < 1:
        raise ValueError('expert checkpoint tile_rows must be positive')
    device = torch.device(device)
    files = _safetensor_index(directory)
    from .distributed_checkpoint import model_shard_layout
    from .parallel_training import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
    layout = model_shard_layout(model)
    tensor_groups = {}
    for path, module in model.named_modules(remove_duplicate=False):
        if isinstance(module, (ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding)):
            prefix = path + '.' if path else ''
            for field in ('weight', 'bias'):
                if getattr(module, field, None) is not None:
                    tensor_groups[prefix + field] = module.group
    banks = [(name, module) for name, module in model.named_modules() if isinstance(module, ExpertParallelMLP)]
    if not banks:
        raise ValueError('streaming EP load requires partitioned expert banks')
    if any(bank.group.device_type != device.type for _, bank in banks):
        raise ValueError('expert load device differs from the communicator device')
    expert_parameters = {id(p) for _, bank in banks for p in bank.parameters()}
    entries = [('parameter', name, value) for name, value in model.named_parameters(remove_duplicate=False)]
    entries += [('buffer', name, value) for name, value in model.named_buffers(remove_duplicate=False)]
    aliases = {}
    for kind, name, value in entries:
        aliases.setdefault(id(value), []).append(name)

    def source(key, shape):
        if key not in files:
            raise ValueError(f'missing expert checkpoint tensor: {key}')
        handle = safe_open(str(files[key]), framework='pt', device='cpu')
        handle.__enter__()
        try:
            sliced = handle.get_slice(key)
            if tuple(sliced.get_shape()) != tuple(shape):
                raise ValueError(f'expert checkpoint shape differs: {key}')
            if sliced.get_dtype() not in ('F16', 'BF16', 'F32', 'F64'):
                raise ValueError(f'floating expert checkpoint required: {key}')
            return handle, sliced
        except BaseException:
            handle.__exit__(None, None, None)
            raise

    with torch.no_grad():
        for path, bank in banks:
            prefix = path if bank.source_prefix is None else bank.source_prefix
            prefix = prefix + '.' if prefix else ''
            gate = torch.empty(bank.gate_up_proj.shape, dtype=bank.gate_up_proj.dtype, device=device)
            down = torch.empty(bank.down_proj.shape, dtype=bank.down_proj.dtype, device=device)
            width = bank.local_intermediate_size
            begin = 0 if bank.tensor_group is None else width * bank.tensor_group.rank
            finish = begin + width
            targets = (gate[:, :width], gate[:, width:], down)
            kind, fields = bank.source_kind, bank.source_names
            if kind == 'hf':
                if all(prefix + field in files for field in fields):
                    kind = 'stacked'
                else:
                    kind, fields = 'separate', ('gate_proj', 'up_proj', 'down_proj')
            if kind == 'stacked':
                for part, (field, target) in enumerate(zip(fields, (gate, down))):
                    shape = ((bank.num_experts, 2 * bank.intermediate_size, bank.hidden_size) if part == 0
                             else (bank.num_experts, bank.hidden_size, bank.intermediate_size))
                    handle, sliced = source(prefix + field, shape)
                    try:
                        for local, global_id in enumerate(range(bank.partition.start, bank.partition.end)):
                            if part == 0:
                                for half in range(2):
                                    for row in range(0, width, tile_rows):
                                        end = min(row + tile_rows, width)
                                        offset = half * bank.intermediate_size + begin
                                        target[local, half*width+row:half*width+end].copy_(
                                            sliced[global_id, offset+row:offset+end, :].to(device=device, dtype=target.dtype))
                            else:
                                for row in range(0, bank.hidden_size, tile_rows):
                                    end = min(row + tile_rows, bank.hidden_size)
                                    target[local, row:end].copy_(sliced[global_id, row:end, begin:finish].to(device=device, dtype=target.dtype))
                    finally:
                        handle.__exit__(None, None, None)
            elif kind == 'separate':
                for local, global_id in enumerate(range(bank.partition.start, bank.partition.end)):
                    for part, (field, target) in enumerate(zip(fields, targets)):
                        key = f'{prefix}{global_id}.{field}.weight'
                        shape = ((bank.intermediate_size, bank.hidden_size) if part < 2
                                 else (bank.hidden_size, bank.intermediate_size))
                        handle, sliced = source(key, shape)
                        try:
                            for row in range(0, target.shape[1], tile_rows):
                                end = min(row + tile_rows, target.shape[1])
                                piece = sliced[begin+row:begin+end, :] if part < 2 else sliced[row:end, begin:finish]
                                target[local, row:end].copy_(piece.to(device=device, dtype=target.dtype))
                        finally:
                            handle.__exit__(None, None, None)
            else:
                raise ValueError('unknown expert checkpoint layout')
            bank.gate_up_proj = nn.Parameter(gate, requires_grad=bank.gate_up_proj.requires_grad)
            bank.down_proj = nn.Parameter(down, requires_grad=bank.down_proj.requires_grad)
            bank._mark_parameters()
        loaded = {}
        for kind, name, value in entries:
            identity = id(value)
            if identity in expert_parameters:
                continue
            if identity not in loaded:
                key = next((alias for alias in aliases[identity] if alias in files), None)
                if key is None:
                    if kind == 'parameter' or value.device.type == 'meta':
                        raise ValueError(f'missing checkpoint value or unmaterialized derived buffer: {name}')
                    tensor = value.detach().to(device=device)
                else:
                    with safe_open(str(files[key]), framework='pt', device='cpu') as handle:
                        sliced = handle.get_slice(key)
                        spec = layout.get(name, {})
                        expected = tuple(spec.get('shape', value.shape))
                        if tuple(sliced.get_shape()) != expected:
                            raise ValueError(f'checkpoint shape differs: {name}')
                        axis = None if spec.get('replicated') else spec.get('axis')
                        peer = tensor_groups.get(name)
                        if axis is not None and peer is None:
                            raise ValueError(f'checkpoint tensor lacks its TP projection communicator: {name}')
                        if spec.get('transform'):
                            raise ValueError('floating EP loader does not reinterpret packed projection formats')
                        selection = [slice(None)] * value.ndim
                        offset = 0 if axis is None else peer.rank * value.shape[axis]
                        if axis is not None:
                            selection[axis] = slice(offset, offset + value.shape[axis])
                        tensor = torch.empty(value.shape, dtype=value.dtype, device=device)
                        if tensor.ndim == 0:
                            tensor.copy_(handle.get_tensor(key).to(device=device, dtype=value.dtype))
                        else:
                            for row in range(0, tensor.shape[0], tile_rows):
                                end = min(row + tile_rows, tensor.shape[0])
                                selection[0] = slice(row + (offset if axis == 0 else 0), end + (offset if axis == 0 else 0))
                                tensor[row:end].copy_(sliced[tuple(selection)].to(device=device, dtype=value.dtype))
                loaded[identity] = nn.Parameter(tensor, requires_grad=value.requires_grad) if kind == 'parameter' else tensor
                for attribute in ('_ruda_tp_sharded', '_ruda_sp_replicated', '_ruda_ep_sharded'):
                    if hasattr(value, attribute):
                        setattr(loaded[identity], attribute, getattr(value, attribute))
            parent, _, field = name.rpartition('.')
            module = model.get_submodule(parent) if parent else model
            if kind == 'parameter':
                module._parameters[field] = loaded[identity]
            else:
                module._buffers[field] = loaded[identity]
    return model
