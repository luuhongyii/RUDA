"""Portable LoRA weights over FSDP storage without gathering frozen base tensors."""
from __future__ import annotations

import math
import torch
from .sharded_training import FullyShardedModule
from .finetuning import LoRALinear
from ._parameter_lora import (LoRAParameter,adapter_shapes,parameter_adapter_metadata,
                             parameter_weight,full_parameter_adapter,local_parameter_adapter)
from .parallel_adapters import _ParallelLoRA,adapter_base_kind,local_adapter_matrix


def has_sharded_units(model):
    return any(isinstance(module,FullyShardedModule) for module in model.modules())


def parameter_records(model):
    """Original parameter names, adapter templates and live FSDP storage paths."""
    from .expert_parallel import ExpertParallelMLP, _SeparateExpertParallelMLP
    result={}

    def visit(root,prefix='',unit=None):
        for path,owner in root.named_modules(remove_duplicate=False):
            if isinstance(owner,FullyShardedModule):
                visit(owner._template,'.'.join(part for part in (prefix,path) if part),owner)
                continue
            if 'parametrizations' in path.split('.') or isinstance(owner,(LoRALinear,LoRAParameter,_ParallelLoRA)):
                continue
            fields=list(owner._parameters)
            if hasattr(owner,'parametrizations'):
                fields.extend(field for field,chain in owner.parametrizations.items()
                              if len(chain)==1 and isinstance(chain[0],LoRAParameter) and field not in fields)
            order=getattr(owner,'_ruda_parameter_order',tuple(fields))
            for field in fields:
                weight,layer=parameter_weight(owner,field)
                if weight is None or not isinstance(weight,torch.nn.Parameter):continue
                owner_path='.'.join(part for part in (prefix,path) if part)
                logical_field=field
                if isinstance(owner,ExpertParallelMLP) and not isinstance(owner,_SeparateExpertParallelMLP):
                    if owner.source_prefix is not None:owner_path=owner.source_prefix
                    if field in ('gate_up_proj','down_proj'):
                        logical_field=owner.source_names[('gate_up_proj','down_proj').index(field)]
                name='.'.join(part for part in (owner_path,logical_field) if part)
                adapter_path='.'.join(part for part in (owner_path,'parametrizations',logical_field,'0') if part)
                storage_path='.'.join(part for part in (path,'parametrizations',field,'0') if part)
                result[name]=dict(owner=owner,field=field,weight=weight,layer=layer,unit=unit,
                                  adapter_path=adapter_path,storage_path=storage_path,
                                  order=order.index(field) if field in order else len(order))
    visit(model)
    return result


def adapter_layers(model,*,remove_duplicate=True):
    """Actual adapter templates and their live storage owner, by logical path."""
    result={}
    for path,module in model.named_modules(remove_duplicate=remove_duplicate):
        if isinstance(module,(LoRALinear,_ParallelLoRA)):
            result[path]=(module,None,path)
        if isinstance(module,FullyShardedModule):
            for local,layer in module._template.named_modules(remove_duplicate=remove_duplicate):
                if isinstance(layer,(LoRALinear,_ParallelLoRA)):
                    name='.'.join(part for part in (path,local) if part)
                    result[name]=(layer,module,local)
    seen=set()
    for record in parameter_records(model).values():
        layer,unit=record['layer'],record['unit']
        if layer is None:continue
        identity=(id(layer),id(unit))
        if remove_duplicate and identity in seen:continue
        seen.add(identity)
        result[record['adapter_path']]=(layer,unit,record['storage_path'])
    return result


def _storage(layer,unit,path,field):
    if unit is None:
        parameter=getattr(layer,field)
        return parameter,tuple(parameter.shape)
    logical=path+'.'+field if path else field
    entry=next((item for item in unit._schema if item[0]==logical),None)
    if entry is None:raise ValueError(f'adapter is missing from its FSDP parameter schema: {logical}')
    name,index,shape,dtype,trainable=entry
    return unit._local_shards()[index],tuple(shape)


def _metadata(layer,version):
    if isinstance(layer,LoRAParameter):
        if version!=3:raise ValueError('parameter adapters require checkpoint version 3')
        return parameter_adapter_metadata(layer)
    metadata={'rank':layer.rank,'alpha':layer.alpha,'in_features':layer.in_features,
              'out_features':layer.out_features,'base':adapter_base_kind(layer)}
    if version>=2:metadata.update(dropout=layer.dropout,use_rslora=layer.use_rslora)
    elif layer.dropout or layer.use_rslora:raise ValueError('legacy checkpoint lacks dropout/RSLoRA semantics')
    return metadata


def _axis(layer,field):
    if not isinstance(layer,_ParallelLoRA):return None
    return 1 if field=='lora_A' and layer.axis==1 else 0 if field=='lora_B' and layer.axis==0 else None


@torch.no_grad()
def sharded_adapter_state_dict(model):
    """Collective adapter-only export, including TP reconstruction when needed.

    Each FSDP/TP participant enters the same layer/field order. Only trainable
    A/B matrices are materialized; frozen dense or packed NF4 bases stay sharded.
    The ordinary ruda-lora record is independent of both data and tensor topology.
    """
    layers=adapter_layers(model)
    if not layers:raise ValueError('model has no LoRA adapters')
    version=(3 if any(isinstance(layer,LoRAParameter) for layer,unit,path in layers.values()) else
             2 if any(layer.dropout or layer.use_rslora for layer,unit,path in layers.values()) else 1)
    values,cache={},{}
    for name,(layer,unit,path) in layers.items():
        entry=_metadata(layer,version)
        for field in ('lora_A','lora_B'):
            parameter,shape=_storage(layer,unit,path,field)
            axis=_axis(layer,field)
            key=(id(parameter),None if unit is None else id(unit.group.group),
                 None if not isinstance(layer,_ParallelLoRA) else id(layer.group.group),axis)
            if key not in cache:
                value=parameter.detach()
                if unit is not None:value=unit.group.all_gather(value)[:math.prod(shape)].view(shape)
                if isinstance(layer,LoRAParameter):value=full_parameter_adapter(layer,field,value)
                elif axis is not None:
                    from .distributed_checkpoint import _join
                    pieces=list(layer.group.all_gather(value,axis=axis).cpu().chunk(layer.group.world_size,dim=axis))
                    value=_join(pieces,{'name':field,'axis':axis,'tp_replicas':getattr(layer,'replicas',1)})
                cache[key]=value.cpu().clone()
            entry[field]=cache[key]
        values[name]=entry
    return {'format':'ruda-lora','version':version,'layers':values}


@torch.no_grad()
def load_sharded_adapter_state_dict(model,state):
    """Validate full logical adapters, then split TP axes and data elements.

    Existing parameter objects, ties, storage dtype and trainable flags remain
    unchanged. Optimizer masters are not silently rewritten by a weights loader.
    """
    if not isinstance(state,dict) or set(state)!={'format','version','layers'} or state['format']!='ruda-lora' or state['version'] not in (1,2,3):
        raise ValueError('unsupported adapter checkpoint')
    layers=adapter_layers(model)
    if not layers or not isinstance(state['layers'],dict) or set(layers)!=set(state['layers']):
        raise ValueError('adapter target names mismatch')
    copies={}
    for name,(layer,unit,path) in layers.items():
        metadata=_metadata(layer,state['version'])
        entry=state['layers'][name]
        if not isinstance(entry,dict) or set(entry)!=set(metadata)|{'lora_A','lora_B'} or any(entry[key]!=value for key,value in metadata.items()):
            raise ValueError(f'adapter configuration mismatch: {name}')
        for field in ('lora_A','lora_B'):
            source=entry[field]
            shape=adapter_shapes(layer)[field]
            if not isinstance(source,torch.Tensor) or tuple(source.shape)!=shape or source.dtype not in (torch.float32,torch.float16,torch.bfloat16):
                raise ValueError(f'adapter tensor mismatch: {name}.{field}')
            if source.device.type!='cpu':source=source.detach().cpu()
            value=(local_parameter_adapter(layer,field,source) if isinstance(layer,LoRAParameter) else
                   local_adapter_matrix(layer,field,source) if isinstance(layer,_ParallelLoRA) else source)
            parameter,local_shape=_storage(layer,unit,path,field)
            if tuple(value.shape)!=local_shape:raise ValueError('FSDP/TP adapter shape differs from template storage')
            if unit is not None:
                size=parameter.numel()
                begin=unit.group.rank*size
                end=min(begin+size,value.numel())
                local=value.new_zeros(parameter.shape)
                if end>begin:local[:end-begin].copy_(value.reshape(-1)[begin:end])
                value=local
            if id(parameter) in copies and not torch.equal(copies[id(parameter)][1],value):
                raise ValueError('shared adapter storage has different checkpoint values')
            copies[id(parameter)]=(parameter,value)
    for parameter,value in copies.values():parameter.copy_(value)


def load_sharded_peft_adapter(model,pairs,config,*,module_map=None,parameter_pairs=()):
    """Import PEFT matrices into adapters injected before FSDP construction."""
    from .adapter_interop import _options,_finite_pair
    layers=adapter_layers(model)
    aliases=adapter_layers(model,remove_duplicate=False)
    canonical={(id(layer),id(unit)):name for name,(layer,unit,path) in layers.items()}
    entries={}
    version=3 if any(isinstance(layer,LoRAParameter) for layer,unit,path in layers.values()) else 2
    for source,pair in pairs.items():
        target=source if module_map is None else module_map.get(source,source)
        if target not in aliases or set(pair)!={'A','B'}:raise ValueError('inject matching LoRA targets before sharding their base model')
        layer,unit,path=aliases[target]
        if isinstance(layer,LoRAParameter):raise ValueError('a parameter adapter must be selected by target_parameters')
        target=canonical[(id(layer),id(unit))]
        expected=_options(config,source)
        if (layer.rank,layer.alpha,layer.dropout,layer.use_rslora)!=expected:
            raise ValueError(f'existing sharded adapter configuration differs: {target}')
        _finite_pair(pair)
        entry=dict(_metadata(layer,version),lora_A=pair['A'],lora_B=pair['B'])
        if target in entries and any(not torch.equal(entries[target][field],entry[field]) for field in ('lora_A','lora_B')):
            raise ValueError('shared PEFT target paths have different adapter matrices')
        entries[target]=entry
    for item in parameter_pairs:
        record,pair=item['record'],item['pair']
        layer,unit=record['layer'],record['unit']
        if layer is None:raise ValueError('inject parameter adapters before constructing FSDP storage')
        target=canonical[(id(layer),id(unit))]
        if (layer.rank,layer.alpha,layer.dropout,layer.use_rslora)!=item['options']:
            raise ValueError(f'existing sharded parameter adapter configuration differs: {target}')
        entry=dict(_metadata(layer,version),lora_A=pair['A'],lora_B=pair['B'])
        if target in entries and any(not torch.equal(entries[target][field],entry[field]) for field in ('lora_A','lora_B')):
            raise ValueError('shared PEFT parameter paths have different adapter matrices')
        entries[target]=entry
    load_sharded_adapter_state_dict(model,{'format':'ruda-lora','version':version,'layers':entries})
    return model
