"""Committed all-rank checkpoints and explicit topology-changing tensor resharding."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
import uuid
import torch


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    return value


def _equal(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def _write_json(path, value):
    temporary = path.with_name(path.name+'.next')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def model_shard_layout(model):
    """Describe actual FSDP/TP storage, retaining full logical names and ties."""
    from .sharded_training import FullyShardedModule
    from .parallel_training import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
    from .parallel_adapters import _ParallelNF4,_ParallelLoRA,nf4_layout
    from .packed_parallel import _ParallelPacked,packed_layout
    from .expert_parallel import ExpertParallelMLP, _SeparateExpertParallelMLP
    layout = {}
    for path, module in model.named_modules(remove_duplicate=False):
        prefix = path+'.' if path else ''
        if isinstance(module, FullyShardedModule):
            for name, index, shape, dtype, trainable in module._schema:
                local = prefix+module._shard_names[index]
                if local in layout:
                    layout[local]['aliases'].append(prefix+name)
                else:
                    layout[local] = {'name': prefix+name, 'aliases': [], 'shape': list(shape), 'axis': None}
            for name, key in module._buffer_names:
                if key in module._buffer_shards:
                    spec={'name':prefix+name,'aliases':[],'shape':list(module._buffer_shards[key]),'axis':None}
                else:
                    spec={'name':prefix+name,'aliases':[],'shape':list(module.get_buffer(key).shape),'replicated':True}
                if prefix+key in layout:
                    layout[prefix+key]['aliases'].append(prefix+name)
                else:
                    layout[prefix+key]=spec
        elif isinstance(module, _SeparateExpertParallelMLP):
            for expert_id, column, name, layer in module.projection_items():
                key = f'{prefix}{expert_id}.{name}.weight'
                shape = ([module.intermediate_size, module.hidden_size] if column < 2
                         else [module.hidden_size, module.intermediate_size])
                layout[key] = {'name': key, 'aliases': [], 'shape': shape, 'axis': None,
                               'partition': 'expert-owned', 'expert_id': expert_id,
                               'num_experts': module.num_experts}
                if module.tensor_group is not None:
                    layout[key]['expert_tensor_axis'] = 0 if column < 2 else 1
        elif isinstance(module, ExpertParallelMLP):
            for name in ('gate_up_proj', 'down_proj'):
                shape = ([module.num_experts, 2 * module.intermediate_size, module.hidden_size]
                         if name == 'gate_up_proj' else [module.num_experts, module.hidden_size, module.intermediate_size])
                layout[prefix+name] = {'name': prefix+name, 'aliases': [], 'shape': shape,
                                      'axis': 0, 'partition': 'expert'}
                if getattr(module, 'tensor_group', None) is not None:
                    layout[prefix+name]['expert_tensor_axis'] = 1 if name == 'gate_up_proj' else 2
                    if name == 'gate_up_proj':
                        layout[prefix+name]['transform'] = 'expert-gate-up'
        elif isinstance(module,_ParallelLoRA):
            for name,shape,axis in (('lora_A',[module.rank,module.in_features],1 if module.axis==1 else None),
                                    ('lora_B',[module.out_features,module.rank],0 if module.axis==0 else None)):
                spec={'name':prefix+name,'aliases':[],'shape':shape,'axis':axis}
                if axis is None:spec['replicated']=True
                layout[prefix+name]=spec
        elif isinstance(module,_ParallelPacked):
            layout.update(packed_layout(module,prefix))
        elif isinstance(module,_ParallelNF4):
            layout.update(nf4_layout(module,prefix))
        elif isinstance(module, (ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding)):
            axis = 1 if isinstance(module, RowParallelLinear) else 0
            shape = list(module.weight.shape)
            replicas = getattr(module, 'replicas', 1)
            shape[axis] *= module.group.world_size // replicas
            layout[prefix+'weight'] = {'name': prefix+'weight', 'aliases': [], 'shape': shape, 'axis': axis}
            if isinstance(module, ColumnParallelLinear) and module.bias is not None:
                layout[prefix+'bias'] = {'name': prefix+'bias', 'aliases': [], 'shape': [module.out_features], 'axis': 0}
        replicas = getattr(module, 'replicas', 1)
        if replicas > 1:
            for name, spec in layout.items():
                if name.startswith(prefix) and spec.get('axis') == 0 and not spec.get('replicated'):
                    spec['tp_replicas'] = replicas
    return layout


def _checkpoint_layout(model,group):
    layout=model_shard_layout(model)
    return group.checkpoint_layout(layout) if hasattr(group,'checkpoint_layout') else layout


def _optimizer_description(model, optimizer, layout):
    if optimizer is None:
        return {'groups':[],'states':{},'extra':{},'layout':{},'class':None}
    from .sharded_training import Zero2Optimizer
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    if isinstance(optimizer, Zero2Optimizer):
        names = {id(shard): names[id(parameter)] for parameter, shard in optimizer.entries}
        optimizer_layout = {names[id(shard)]: {'name': names[id(shard)], 'aliases': [],
                            'shape': list(parameter.shape), 'axis': None}
                            for parameter, shard in optimizer.entries}
        for parameter,shard in optimizer.entries:
            name=names[id(shard)]
            spec=layout.get(name)
            if spec is not None and 'model_mesh' in spec:
                optimizer_layout[name] = dict(spec, data_flat=True)
            elif spec is not None and 'mesh' in spec:
                optimizer_layout[name]=dict(spec,transform='zero2-tp-flat',local_shape=list(parameter.shape))
            elif any('mesh' in entry for entry in layout.values()):
                mesh=next(entry['mesh'] for entry in layout.values() if 'mesh' in entry)
                optimizer_layout[name]=dict(name=name,aliases=[],shape=list(parameter.shape),axis=None,
                    replicated=True,mesh=mesh,transform='zero2-tp-flat',local_shape=list(parameter.shape))
        actual = optimizer.optimizer
    else:
        optimizer_layout, actual = layout, optimizer
    state = actual.state_dict()
    groups, states = [], {}
    for live, saved in zip(actual.param_groups, state['param_groups'], strict=True):
        group_names = [names[id(parameter)] for parameter in live['params']]
        groups.append(dict((key, value) for key, value in saved.items() if key != 'params') | {'params': group_names})
        for name, identity in zip(group_names, saved['params'], strict=True):
            if identity in state['state']:
                states[name] = state['state'][identity]
    extra = {key: value for key, value in state.items() if key not in ('param_groups', 'state')}
    return {'groups': groups, 'states': states, 'extra': extra, 'layout': optimizer_layout,
            'class': type(actual).__module__+'.'+type(actual).__qualname__}


def _logical_checkpoint(state):
    from ._expert_ownership import logical_layout
    layout = state['layout']
    model = {layout.get(name, {}).get('name', name): value for name, value in state['model'].items()}
    description = state['optimizer']
    optimizer_layout = description['layout']
    names = lambda name: optimizer_layout.get(name, {}).get('name', name)
    groups = [dict(group, params=[names(name) for name in group['params']]) for group in description['groups']]
    description = dict(description, groups=groups,
        states={names(name): value for name, value in description['states'].items()},
        layout=logical_layout(optimizer_layout))
    return dict(state, model=model, layout=logical_layout(layout), optimizer=description)


class DistributedCheckpoint:
    """Shared-filesystem, all-rank commit at a completed optimizer boundary.

    The latest pointer moves only after every rank has fsynced and reread its
    checkpoint and root has published a complete manifest. A failed write leaves
    the previous committed checkpoint intact. Application/sampler/RNG positions
    are caller-owned state, not inferred from the number of processed batches.
    """
    def __init__(self, group, directory):
        self.group, self.directory = group, Path(directory)

    def save(self, model, optimizer, *, step, application_state=None, scheduler=None, scaler=None):
        from .float8_parallel import recipe_contract
        from ._expert_ownership import merge_layouts
        layout = _checkpoint_layout(model,self.group)
        self.group.validate_training_options(step)
        merge_layouts(self.group.gather_metadata(layout))
        if type(step) is not int or step < 0:
            raise ValueError('checkpoint step must be a nonnegative integer')
        generations = self.group.gather_metadata(uuid.uuid4().hex if self.group.rank == 0 else None)
        generation = f'step-{step}-{generations[0]}'
        folder = self.directory/generation
        error, path = None, folder/f'rank-{self.group.rank}.pt'
        started = time.monotonic()
        try:
            folder.mkdir(parents=True, exist_ok=True)
            state = {'version': 1, 'step': step, 'rank': self.group.rank, 'world_size': self.group.world_size,
                     'float8_recipe': recipe_contract(model),
                     'model': model.state_dict(), 'layout': layout,
                     'optimizer': _optimizer_description(model, optimizer, layout),
                     'application': application_state,
                     'scheduler': None if scheduler is None else scheduler.state_dict(),
                     'scaler': None if scaler is None else scaler.state_dict(),
                     'torch_rng': torch.get_rng_state()}
            if self.group.device_type == 'ruda':
                state['cuda_rng'] = torch.cuda.get_rng_state(self.group.cuda_index)
                from . import get_rng_state
                state['ruda_rng'] = get_rng_state()
            temporary = path.with_suffix('.next')
            with temporary.open('wb') as stream:
                torch.save(_cpu(state), stream)
                stream.flush()
                os.fsync(stream.fileno())
            checked = torch.load(temporary, map_location='cpu', weights_only=True)
            if (checked['step'], checked['rank'], checked['world_size']) != (step, self.group.rank, self.group.world_size):
                raise RuntimeError('rank checkpoint readback differs')
            os.replace(temporary, path)
        except Exception as failure:
            error = f'{type(failure).__name__}: {failure}'
        receipts = self.group.gather_metadata((str(path.resolve()), error))
        for saved, failure in receipts:
            if failure:
                raise RuntimeError(f'checkpoint not committed: {failure}')
        committed = None
        if self.group.rank == 0:
            try:
                for rank, (saved, _) in enumerate(receipts):
                    checked = torch.load(saved, map_location='cpu', weights_only=True)
                    if (checked['rank'], checked['step'], checked['world_size']) != (rank, step, self.group.world_size):
                        raise RuntimeError('shared filesystem rank files differ')
                manifest = {'version': 1, 'step': step, 'world_size': self.group.world_size,
                            'files': [Path(saved).name for saved, _ in receipts]}
                _write_json(folder/'manifest.json', manifest)
                _write_json(self.directory/'latest.json', {'generation': generation, 'step': step})
            except Exception as failure:
                committed = f'{type(failure).__name__}: {failure}'
        errors = self.group.gather_metadata(committed)
        if errors[0]:
            raise RuntimeError(f'checkpoint not committed: {errors[0]}')
        return {'directory': str(folder.resolve()), 'step': step, 'seconds': time.monotonic()-started}

    def load(self, model, optimizer, *, generation=None, scheduler=None, scaler=None,validate_application=None):
        folder,manifest,error=None,None,None
        try:
            folder,manifest=self._manifest(generation)
        except Exception as failure:
            error=f'{type(failure).__name__}: {failure}'
        for failure in self.group.gather_metadata(error):
            if failure:raise ValueError(failure)
        self.group.validate_training_options((str(folder.resolve()), manifest))
        if manifest['world_size'] != self.group.world_size:
            raise ValueError('world size changed; use consolidate() and load_consolidated() with an explicit data resharder')
        state, error = None, None
        try:
            state = torch.load(folder/manifest['files'][self.group.rank], map_location='cpu', weights_only=True)
            from .float8_parallel import validate_recipe_contract
            validate_recipe_contract(model, state.get('float8_recipe'))
            if (state['step'], state['rank'], state['world_size'], state['layout']) != (
                    manifest['step'], self.group.rank, self.group.world_size, _checkpoint_layout(model,self.group)):
                raise ValueError('rank checkpoint topology or layout differs')
            if (state['scheduler'] is None) != (scheduler is None) or (state['scaler'] is None) != (scaler is None):
                raise ValueError('scheduler/scaler presence differs')
            current=model.state_dict()
            if set(current)!=set(state['model']):raise ValueError('checkpoint model keys differ')
            for name,target in current.items():
                value=state['model'][name]
                if isinstance(target,torch.Tensor) and (not isinstance(value,torch.Tensor) or target.shape!=value.shape or target.dtype!=value.dtype):
                    raise ValueError(f'checkpoint model shape/dtype differs: {name}')
            live=_optimizer_description(model,optimizer,state['layout'])
            if live['class']!=state['optimizer']['class'] or [g['params'] for g in live['groups']]!=[g['params'] for g in state['optimizer']['groups']]:
                raise ValueError('checkpoint optimizer class/parameter layout differs')
            if validate_application is not None:
                validate_application(state['application'])
        except Exception as failure:
            error = str(failure)
        for failure in self.group.gather_metadata(error):
            if failure:
                raise ValueError(failure)
        model.load_state_dict(state['model'])
        _restore_optimizer(model, optimizer, state['optimizer'], state['layout'], self.group, consolidated=False)
        if scheduler is not None:
            scheduler.load_state_dict(state['scheduler'])
        if scaler is not None:
            scaler.load_state_dict(state['scaler'])
        torch.set_rng_state(state['torch_rng'])
        if 'cuda_rng' in state:
            torch.cuda.set_rng_state(state['cuda_rng'], self.group.cuda_index)
        if 'ruda_rng' in state:
            from . import set_rng_state
            set_rng_state(state['ruda_rng'])
        return state['step'], state['application']

    def _manifest(self, generation):
        if generation is None:
            generation = json.loads((self.directory/'latest.json').read_text(encoding='utf-8'))['generation']
        if Path(generation).name != generation:
            raise ValueError('generation must be a checkpoint directory name')
        folder = self.directory/generation
        manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if manifest['version'] != 1 or len(manifest['files']) != manifest['world_size']:
            raise ValueError('invalid checkpoint commit manifest')
        return folder, manifest

    def consolidate(self, *, generation=None):
        """CPU operation: reconstruct weights and elementwise optimizer states.

        Memory is the full checkpoint plus the rank-local input checkpoints.
        This is explicit, never automatically run inside a training step.
        """
        from ._expert_ownership import owned, owning_ranks, merge_layouts
        folder, manifest = self._manifest(generation)
        ranks = [torch.load(folder/name, map_location='cpu', weights_only=True) for name in manifest['files']]
        first = ranks[0]
        for rank, state in enumerate(ranks):
            if state.get('float8_recipe') != first.get('float8_recipe'):
                raise ValueError('rank FP8 recipes differ')
            if (state['version'], state['rank'], state['world_size'], state['step']) != (
                    1, rank, manifest['world_size'], manifest['step']):
                raise ValueError('rank checkpoint identities differ')
        ranks = [_logical_checkpoint(state) for state in ranks]
        first = ranks[0]
        layout = merge_layouts([state['layout'] for state in ranks])
        model = {}
        for key in dict.fromkeys(key for state in ranks for key in state['model']):
            spec = layout.get(key)
            expected = owning_ranks(spec, len(ranks))
            if [rank for rank, state in enumerate(ranks) if key in state['model']] != expected:
                raise ValueError('model checkpoint tensor ownership differs: ' + key)
            value = _join([state['model'].get(key) for state in ranks], spec)
            name = key if spec is None else spec['name']
            model[name] = value
            for alias in () if spec is None else spec['aliases']:
                model[alias] = value
        descriptions = [state['optimizer'] for state in ranks]
        base = descriptions[0]
        if any(not _equal((d['extra'], d['class']), (base['extra'], base['class']))
               or len(d['groups']) != len(base['groups']) for d in descriptions):
            raise ValueError('optimizer rank descriptions differ')
        optimizer_layout = merge_layouts([description['layout'] for description in descriptions])
        groups = []
        for index, group in enumerate(base['groups']):
            options = {key: value for key, value in group.items() if key != 'params'}
            peers = [description['groups'][index] for description in descriptions]
            if any(not _equal(options, {key: value for key, value in peer.items() if key != 'params'}) for peer in peers):
                raise ValueError('optimizer group options differ between ranks')
            members = merge_layouts([{name: optimizer_layout.get(name, {'replicated': True})
                                     for name in peer['params']} for peer in peers])
            groups.append(dict(options, params=[optimizer_layout.get(name, {}).get('name', name) for name in members]))
        states = {}
        for name in set().union(*(d['states'] for d in descriptions)):
            values = [d['states'].get(name) for d in descriptions]
            spec = optimizer_layout.get(name)
            owners = owning_ranks(spec, len(ranks))
            if [rank for rank, value in enumerate(values) if value is not None] != owners:
                raise ValueError('optimizer state missing on some shards')
            global_name = name if spec is None else spec['name']
            local_parameter = ranks[owners[0]]['model'].get(name)
            fields = {}
            source = values[owners[0]]
            if any(entry.keys() != source.keys() for entry in values if entry is not None):
                raise ValueError('optimizer state fields differ between owning ranks')
            for key, value in source.items():
                shape = None if spec is None else spec['shape']
                is_parameter_state = isinstance(value, torch.Tensor) and shape is not None and key != 'step' and (
                    (spec.get('transform')=='zero2-tp-flat' or spec.get('data_flat')) and value.ndim==1 or
                    local_parameter is not None and value.shape == local_parameter.shape or
                    spec.get('axis') is None and value.ndim == 1 and value.numel() == math.ceil(math.prod(shape)/len(ranks)))
                if is_parameter_state:
                    fields[key] = _join([None if entry is None else entry[key] for entry in values], spec)
                else:
                    fields[key] = _join([values[rank][key] for rank in owners], None)
            states[global_name] = fields
        owned_parameters = [spec['name'] for spec in optimizer_layout.values() if owned(spec)]
        return {'version': 1, 'step': first['step'], 'model': model,
                'float8_recipe': first.get('float8_recipe'),
                'optimizer': dict(base, groups=groups, states=states, layout={}, owned_parameters=owned_parameters),
                'rank_states': [{'application': state['application'], 'torch_rng': state['torch_rng'],
                                 'cuda_rng': state.get('cuda_rng'),'ruda_rng':state.get('ruda_rng')} for state in ranks],
                'scheduler': first['scheduler'], 'scaler': first['scaler']}

    def load_consolidated(self, model, optimizer, state, *, reshard_application, scheduler=None, scaler=None):
        """Restore tensors on a new topology; caller explicitly repartitions data/RNG.

        reshard_application(old_rank_states, new_rank, new_world) returns the
        application state and must explicitly restore any run-specific RNG.
        """
        from .float8_parallel import validate_recipe_contract
        validate_recipe_contract(model, state.get('float8_recipe'))
        layout = _checkpoint_layout(model,self.group)
        local = {}
        for name, target in model.state_dict().items():
            spec = layout.get(name)
            logical = name if spec is None else spec['name']
            local[name] = _slice(state['model'][logical], spec, self.group.rank, self.group.world_size)
            if isinstance(target,torch.Tensor) and (not isinstance(local[name],torch.Tensor) or local[name].shape != target.shape):
                raise ValueError(f'resharded model shape differs: {name}')
        model.load_state_dict(local)
        _restore_optimizer(model, optimizer, state['optimizer'], layout, self.group, consolidated=True)
        if (scheduler is None) != (state['scheduler'] is None) or (scaler is None) != (state['scaler'] is None):
            raise ValueError('scheduler/scaler presence differs')
        if scheduler is not None:
            scheduler.load_state_dict(state['scheduler'])
        if scaler is not None:
            scaler.load_state_dict(state['scaler'])
        application = reshard_application(state['rank_states'], self.group.rank, self.group.world_size)
        return state['step'], application


def _join(values, spec):
    if spec is not None and 'model_mesh' in spec:
        from .model_parallel_checkpoint import join_model_mesh_field
        return join_model_mesh_field(values, spec)
    if spec is not None and spec.get('partition') == 'expert-owned':
        from ._expert_ownership import owning_ranks
        owners = owning_ranks(spec, len(values))
        if any(value is not None for rank, value in enumerate(values) if rank not in owners):
            raise ValueError('an unowned expert checkpoint contains storage')
        result = values[owners[0]]
        if not isinstance(result, torch.Tensor) or tuple(result.shape) != tuple(spec['shape']):
            raise ValueError('separate expert checkpoint shape differs')
        return result
    if spec is not None and 'mesh' in spec:
        from .hybrid_parallel import join_mesh_field
        return join_mesh_field(values,spec)
    if spec is not None and spec.get('tp_replicas', 1) > 1:
        replicas = spec['tp_replicas']
        if len(values) % replicas:
            raise ValueError('checkpoint TP replicas do not divide the tensor ranks')
        unique = []
        for start in range(0, len(values), replicas):
            copies = values[start:start+replicas]
            if any(not _equal(copies[0], value) for value in copies[1:]):
                raise ValueError('KV head replica checkpoint fields differ: ' + spec['name'])
            unique.append(copies[0])
        return _join(unique, {key: value for key, value in spec.items() if key != 'tp_replicas'})
    if spec is not None and spec.get('transform','').startswith('packed-'):
        from .packed_parallel import join_packed_field
        return join_packed_field(values,spec)
    if spec is not None and spec.get('transform','').startswith('nf4-'):
        from .parallel_adapters import join_nf4_field
        return join_nf4_field(values,spec)
    if spec is None or spec.get('replicated') or not isinstance(values[0], torch.Tensor):
        if any(not _equal(values[0], value) for value in values[1:]):
            raise ValueError('replicated checkpoint values differ')
        return values[0]
    axis = spec['axis']
    if axis is None:
        return torch.cat([value.reshape(-1) for value in values])[:math.prod(spec['shape'])].view(spec['shape'])
    return torch.cat(values, dim=axis)


def _slice(value, spec, rank, world):
    if spec is not None and 'model_mesh' in spec:
        from .model_parallel_checkpoint import slice_model_mesh_field
        return slice_model_mesh_field(value, spec, rank, world)
    if spec is not None and spec.get('partition') == 'expert-owned':
        from ._expert_ownership import owning_ranks
        if rank not in owning_ranks(spec, world):
            raise ValueError('target rank does not own this expert checkpoint field')
        return value
    if spec is not None and 'mesh' in spec:
        from .hybrid_parallel import slice_mesh_field
        return slice_mesh_field(value,spec,rank,world)
    if spec is not None and spec.get('tp_replicas', 1) > 1:
        replicas = spec['tp_replicas']
        if world % replicas:
            raise ValueError('target TP replicas do not divide the tensor ranks')
        return _slice(value, {key: item for key, item in spec.items() if key != 'tp_replicas'},
                      rank // replicas, world // replicas)
    if spec is not None and spec.get('transform','').startswith('packed-'):
        from .packed_parallel import slice_packed_field
        return slice_packed_field(value,spec,rank,world)
    if spec is not None and spec.get('transform','').startswith('nf4-'):
        from .parallel_adapters import slice_nf4_field
        return slice_nf4_field(value,spec,rank,world)
    if spec is None or spec.get('replicated'):
        return value
    if spec.get('partition') == 'expert':
        count = value.shape[0]
        return value[count * rank // world:count * (rank + 1) // world].contiguous()
    if spec['axis'] is None:
        size = math.ceil(math.prod(spec['shape'])/world)
        result = value.new_zeros(size)
        start, end = rank*size, min((rank+1)*size, value.numel())
        if end > start:
            result[:end-start].copy_(value.reshape(-1)[start:end])
        return result
    if value.shape[spec['axis']] % world:
        raise ValueError('TP checkpoint dimension does not divide the new world size')
    return value.chunk(world, dim=spec['axis'])[rank].contiguous()


def _restore_optimizer(model, optimizer, description, layout, group, *, consolidated):
    if optimizer is None:
        if description['class'] is not None:raise ValueError('checkpoint has an optimizer but this stage does not')
        return
    from .sharded_training import Zero2Optimizer
    live_description = _optimizer_description(model, optimizer, layout)
    if description['class'] != live_description['class']:
        raise ValueError('optimizer class differs')
    actual = optimizer.optimizer if isinstance(optimizer, Zero2Optimizer) else optimizer
    live = actual.state_dict()
    saved_groups = description['groups']
    groups, states = [], {}
    if len(saved_groups) != len(live['param_groups']):
        raise ValueError('optimizer parameter groups differ')
    for named, indexed, saved in zip(live_description['groups'], live['param_groups'], saved_groups, strict=True):
        specs = live_description['layout']
        logical = [specs.get(name, {}).get('name', name) for name in named['params']]
        expected = saved['params']
        if consolidated:
            owned_names = set(description.get('owned_parameters', ()))
            expected = [name for name in expected if name not in owned_names or name in logical]
        if (logical if consolidated else named['params']) != expected:
            raise ValueError('optimizer parameter paths differ')
        groups.append(dict(saved, params=indexed['params']))
        for name, global_name, identity, parameter in zip(named['params'], logical, indexed['params'],
                                                         actual.param_groups[len(groups)-1]['params'], strict=True):
            entry = description['states'].get(global_name if consolidated else name)
            if entry is None:
                continue
            fields = {}
            spec = specs.get(name)
            for key, value in entry.items():
                if consolidated and spec is not None and (not spec.get('replicated') or 'fsdp_shape' in spec or spec.get('data_flat') or spec.get('transform')=='zero2-tp-flat') and isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(spec['shape']) and key != 'step':
                    value = _slice(value, spec, group.rank, group.world_size)
                fields[key] = value
            states[identity] = fields
    actual.load_state_dict(dict(description['extra'], param_groups=groups, state=states))
    if isinstance(optimizer, Zero2Optimizer):
        with torch.no_grad():
            for parameter, shard in optimizer.entries:
                value = _slice(parameter.detach(), {'axis': None, 'shape': list(parameter.shape)}, optimizer.group.rank, optimizer.group.world_size)
                shard.copy_(value)
