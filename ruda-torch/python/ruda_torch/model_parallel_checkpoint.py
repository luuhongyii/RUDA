"""Logical checkpoint tensors across data, context, expert and tensor axes."""
from __future__ import annotations

import math
import torch


def _geometry(spec):
    data, pipeline, context, expert, tensor = spec['model_mesh']
    if pipeline != 1 or min(data, context, expert, tensor) < 1:
        raise ValueError('a model checkpoint field belongs to one pipeline stage')
    return data, context, expert, tensor


def _plain(spec):
    return {key: value for key, value in spec.items()
            if key not in ('model_mesh', 'data_flat', 'model_extra')}


def _tensor_spec(spec, expert_rank, expert_world):
    result = _plain(spec)
    if result.get('partition') == 'expert-owned':
        result.pop('partition')
        result['axis'] = result.pop('expert_tensor_axis', None)
        result['replicated'] = result['axis'] is None
    elif result.get('partition') == 'expert':
        result['shape'] = list(result['shape'])
        count = result['shape'][0]
        result['shape'][0] = count * (expert_rank + 1) // expert_world - count * expert_rank // expert_world
        result.pop('partition')
        axis = result.pop('expert_tensor_axis', None)
        result['axis'] = axis
        result['replicated'] = axis is None
    return result


def local_storage_shape(spec, expert_rank, tensor_rank):
    from .sharded_mesh import _logical_shape
    _, _, experts, tensors = _geometry(spec)
    field = _tensor_spec(spec, expert_rank, experts)
    shape = list(field['shape'])
    if not field.get('replicated'):
        axis = field['axis']
        replicas = field.get('tp_replicas', 1)
        if tensors % replicas:
            raise ValueError('KV replicas do not divide the tensor topology')
        shards = tensors // replicas
        if axis is None or shape[axis] % shards:
            raise ValueError('tensor partition does not divide the logical checkpoint field')
        shape[axis] //= shards
    field['shape'] = shape
    return tuple(_logical_shape(field))


def _join_tensor(values, spec):
    from .distributed_checkpoint import _join
    if spec.get('transform') == 'expert-gate-up':
        if spec.get('replicated'):
            return _join(values, None)
        gate, up = zip(*(value.chunk(2, dim=1) for value in values))
        return torch.cat((torch.cat(gate, dim=1), torch.cat(up, dim=1)), dim=1)
    return _join(values, spec)


def _slice_tensor(value, spec, rank, world):
    from .distributed_checkpoint import _slice
    if spec.get('transform') == 'expert-gate-up' and not spec.get('replicated'):
        if value.shape[1] % (2 * world):
            raise ValueError('expert intermediate width must divide the new tensor group')
        gate, up = value.chunk(2, dim=1)
        return torch.cat((gate.chunk(world, dim=1)[rank], up.chunk(world, dim=1)[rank]), dim=1).contiguous()
    return _slice(value, spec, rank, world)


def join_model_mesh_field(values, spec):
    from .distributed_checkpoint import _equal
    from ._expert_ownership import owned, owning_ranks
    data, context, experts, tensors = _geometry(spec)
    if len(values) != data * context * experts * tensors:
        raise ValueError('model checkpoint rank count differs from its mesh')
    if spec.get('model_extra') is not None:
        return _join_extra(values, spec)

    def identical(copies):
        if any(not _equal(copies[0], value) for value in copies[1:]):
            raise ValueError('replicated model checkpoint fields differ: ' + spec['name'])
        return copies[0]

    selected = list(range(experts))
    if owned(spec):
        owners = owning_ranks(spec, len(values))
        if any(value is not None for rank, value in enumerate(values) if rank not in owners):
            raise ValueError('an unowned expert checkpoint contains storage')
        selected = [owners[0] // tensors % experts]
    pieces = []
    for expert in selected:
        tensor_pieces = []
        for tensor in range(tensors):
            contexts = []
            for cp in range(context):
                copies = [values[((dp * context + cp) * experts + expert) * tensors + tensor]
                          for dp in range(data)]
                if spec.get('data_flat'):
                    shape = local_storage_shape(spec, expert, tensor)
                    size = math.ceil(math.prod(shape) / data)
                    if any(not isinstance(value, torch.Tensor) or value.shape != (size,) for value in copies):
                        raise ValueError('data-sharded checkpoint storage differs from its logical field')
                    contexts.append(torch.cat(copies)[:math.prod(shape)].view(shape))
                else:
                    contexts.append(identical(copies))
            tensor_pieces.append(identical(contexts))
        pieces.append(_join_tensor(tensor_pieces, _tensor_spec(spec, expert, experts)))
    if spec.get('partition') == 'expert':
        result = torch.cat(pieces, dim=0)
        if tuple(result.shape) != tuple(spec['shape']):
            raise ValueError('expert checkpoint reconstruction shape differs')
        return result
    return identical(pieces)


def slice_model_mesh_field(value, spec, rank, world):
    from ._expert_ownership import owned, owning_ranks
    data, context, experts, tensors = _geometry(spec)
    if world != data * context * experts * tensors or not 0 <= rank < world:
        raise ValueError('target checkpoint rank count differs from its mesh')
    tensor = rank % tensors
    expert = rank // tensors % experts
    dp = rank // (context * experts * tensors)
    if owned(spec) and rank not in owning_ranks(spec, world):
        raise ValueError('target rank does not own this expert checkpoint field')
    if spec.get('model_extra') is not None:
        return _slice_extra(value, spec, rank, world)
    if spec.get('partition') == 'expert':
        count = value.shape[0]
        value = value[count * expert // experts:count * (expert + 1) // experts]
    local = _slice_tensor(value, _tensor_spec(spec, expert, experts), tensor, tensors)
    if not spec.get('data_flat'):
        return local
    size = math.ceil(local.numel() / data)
    result = local.new_zeros(size)
    begin, end = dp * size, min((dp + 1) * size, local.numel())
    if end > begin:
        result[:end-begin].copy_(local.reshape(-1)[begin:end])
    return result


def model_mesh_layout(model, mesh):
    from .distributed_checkpoint import model_shard_layout
    from .sharded_training import FullyShardedModule
    geometry = tuple(mesh.shape)
    if geometry[1] != 1:
        raise ValueError('use the current pipeline stage mesh for checkpoint fields')

    def field(name, shape, source=None, *, flat=False):
        result = dict(source or {'name': name, 'aliases': [], 'shape': list(shape),
                                 'axis': None, 'replicated': True})
        result['model_mesh'] = geometry
        if flat:
            result['data_flat'] = True
        return result

    layout = {name: field(name, spec.get('shape'), spec) for name, spec in model_shard_layout(model).items()}
    for path, unit in model.named_modules(remove_duplicate=False):
        if not isinstance(unit, FullyShardedModule):
            continue
        prefix = path + '.' if path else ''
        fields = dict(unit._template_layout)
        for name, _, shape, _, _ in unit._schema:
            fields.setdefault(name, {'name': name, 'aliases': [], 'shape': list(shape),
                                     'axis': None, 'replicated': True})
        for name, key in unit._buffer_names:
            shape = unit._buffer_shards.get(key, tuple(unit.get_buffer(key).shape))
            fields.setdefault(name, {'name': name, 'aliases': [], 'shape': list(shape),
                                     'axis': None, 'replicated': True})
        seen = set()

        def install(name, key, flat):
            local = prefix + key
            source = fields[name]
            logical = prefix + source['name']
            if local in seen:
                if logical != layout[local]['name'] and logical not in layout[local]['aliases']:
                    layout[local]['aliases'].append(logical)
            else:
                spec = dict(source, name=logical, aliases=[prefix + alias for alias in source.get('aliases', [])])
                layout[local] = field(logical, None, spec, flat=flat)
                seen.add(local)

        for name, index, _, _, _ in unit._schema:
            install(name, unit._shard_names[index], True)
        for name, key in unit._buffer_names:
            install(name, key, key in unit._buffer_shards)
        # Keep a rank-independent template description. Local EP widths are
        # reconstructed from logical fields, not copied from rank zero.
        schema = [(name, index, dtype, trainable) for name, index, _, dtype, trainable in unit._schema]
        buffers = {key: next(name for name, physical in unit._buffer_names if physical == key)
                   for key in unit._buffer_shards}
        extra = dict(schema=schema, buffers=buffers, fields=fields,
                     modules=list(unit.get_extra_state()['module_extra']))
        layout[prefix + '_extra_state'] = {'name': prefix + '_extra_state', 'aliases': [],
                                           'model_mesh': geometry, 'model_extra': extra}
    for name, value in list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False)):
        if name not in layout:
            layout[name] = field(name, value.shape)
    for path, module in model.named_modules(remove_duplicate=False):
        if type(module).get_extra_state is not torch.nn.Module.get_extra_state:
            name = path + '._extra_state' if path else '_extra_state'
            layout.setdefault(name, {'name': name, 'aliases': [], 'replicated': True, 'model_mesh': geometry})
    return layout


def _extra_field(spec, name):
    source = spec['model_extra']['fields'].get(name)
    if source is None:
        source = {'name': name, 'aliases': [], 'replicated': True}
    return dict(source, model_mesh=spec['model_mesh'])


def _extra_schema_contract(spec):
    from .sharded_mesh import _logical_shape
    extra = spec['model_extra']
    canonical, schema = {}, {}
    for name, index, dtype, trainable in extra['schema']:
        canonical.setdefault(index, name)
        field = _extra_field(spec, name)
        schema[name] = dict(field, dtype=dtype, trainable=trainable,
                            canonical=canonical[index], shape=list(_logical_shape(field)))
    buffers = {name: dict(_extra_field(spec, name), shape=list(_logical_shape(_extra_field(spec, name))))
               for name in extra['buffers'].values()}
    return schema, buffers


def merge_extra_contracts(specs):
    from ._expert_ownership import merge_layouts
    first = specs[0]
    header = {key: value for key, value in first.items() if key != 'model_extra'}
    if any({key: value for key, value in spec.items() if key != 'model_extra'} != header for spec in specs):
        raise ValueError('FSDP checkpoint mesh headers differ')
    contracts = [_extra_schema_contract(spec) for spec in specs]
    schema = merge_layouts([entry[0] for entry in contracts])
    buffers = merge_layouts([entry[1] for entry in contracts])
    fields = merge_layouts([{name: _extra_field(spec, name) for name in spec['model_extra']['fields']}
                           for spec in specs])
    modules = first['model_extra']['modules']
    if any(spec['model_extra']['modules'] != modules for spec in specs):
        raise ValueError('FSDP module extra-state ownership differs between expert ranks')
    return dict(first, model_extra_ranked=[spec['model_extra'] for spec in specs],
                model_extra_contract=dict(schema=schema, buffers=buffers, fields=fields))


def _local_extra_schema(spec, rank):
    _, _, experts, tensors = _geometry(spec)
    expert, tensor = rank // tensors % experts, rank % tensors
    extra = spec['model_extra']
    schema = tuple((name, index, local_storage_shape(_extra_field(spec, name), expert, tensor), dtype, trainable)
                   for name, index, dtype, trainable in extra['schema'])
    buffers = {key: local_storage_shape(_extra_field(spec, name), expert, tensor)
               for key, name in extra['buffers'].items()}
    return schema, buffers


def _join_extra(values, spec):
    extra = spec['model_extra']
    for rank, value in enumerate(values):
        local = dict(spec, model_extra=spec['model_extra_ranked'][rank]) if 'model_extra_ranked' in spec else spec
        schema, buffers = _local_extra_schema(local, rank)
        if value['version'] != 1 or value['schema'] != schema or value['buffer_shards'] != buffers:
            raise ValueError('FSDP rank template does not match the model mesh')
        if set(value['module_extra']) != set(extra['modules']):
            raise ValueError('FSDP module extra-state paths differ')
    modules = {}
    for path in extra['modules']:
        key = path + '._extra_state' if path else '_extra_state'
        modules[path] = join_model_mesh_field([value['module_extra'][path] for value in values], _extra_field(spec, key))
    if 'model_extra_contract' in spec:
        contract = spec['model_extra_contract']
        schema = {name: {key: value for key, value in field.items()
                         if key in ('shape', 'dtype', 'trainable', 'canonical')}
                  for name, field in contract['schema'].items()}
        buffers = {name: field['shape'] for name, field in contract['buffers'].items()}
        return dict(version=4, schema=schema, buffers=buffers, module_extra=modules)
    from .sharded_mesh import _logical_shape
    schema = tuple((name, index, _logical_shape(extra['fields'][name]), dtype, trainable)
                   for name, index, dtype, trainable in extra['schema'])
    buffers = {key: _logical_shape(extra['fields'][name]) for key, name in extra['buffers'].items()}
    return {'version': 3, 'schema': schema, 'buffers': buffers, 'module_extra': modules}


def _slice_extra(value, spec, rank, world):
    from .sharded_mesh import _logical_shape
    extra = spec['model_extra']
    if value['version'] == 4:
        schema_contract, buffer_contract = _extra_schema_contract(spec)
        for name, field in schema_contract.items():
            expected = {key: item for key, item in field.items() if key in ('shape', 'dtype', 'trainable', 'canonical')}
            if value['schema'].get(name) != expected:
                raise ValueError('consolidated independent-expert FSDP parameter differs: ' + name)
        for name, field in buffer_contract.items():
            if value['buffers'].get(name) != field['shape']:
                raise ValueError('consolidated independent-expert FSDP buffer differs: ' + name)
        schema, local_buffers = _local_extra_schema(spec, rank)
        if set(value['module_extra']) != set(extra['modules']):
            raise ValueError('consolidated FSDP module extra-state paths differ')
        modules = {path: slice_model_mesh_field(value['module_extra'][path],
                    _extra_field(spec, path + '._extra_state' if path else '_extra_state'), rank, world)
                   for path in extra['modules']}
        return dict(version=1, schema=schema, buffer_shards=local_buffers, module_extra=modules)
    expected = tuple((name, index, _logical_shape(extra['fields'][name]), dtype, trainable)
                     for name, index, dtype, trainable in extra['schema'])
    buffers = {key: _logical_shape(extra['fields'][name]) for key, name in extra['buffers'].items()}
    if value['version'] != 3 or value['schema'] != expected or value['buffers'] != buffers:
        raise ValueError('consolidated FSDP model template differs')
    schema, local_buffers = _local_extra_schema(spec, rank)
    modules = {path: slice_model_mesh_field(value['module_extra'][path],
                _extra_field(spec, path + '._extra_state' if path else '_extra_state'), rank, world)
               for path in extra['modules']}
    return {'version': 1, 'schema': schema, 'buffer_shards': local_buffers, 'module_extra': modules}
