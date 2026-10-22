"""PEFT safetensors interchange for dense/NF4 LoRA and rank-stabilized LoRA."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
import torch
from .finetuning import LoRALinear, NF4Linear, _replace,adapter_state_dict


def _pattern(patterns,name,default):
    for pattern,value in patterns.items():
        if re.match(r'(.*\.)?'+pattern+r'$',name):
            return value
    return default


def _options(config,source):
    rank=_pattern(config.get('rank_pattern',{}),source,config['r'])
    alpha=_pattern(config.get('alpha_pattern',{}),source,config['lora_alpha'])
    dropout=config.get('lora_dropout',0.)
    rslora=config.get('use_rslora',False)
    if type(rank) is not int or rank<=0 or type(alpha) not in (int,float) or not math.isfinite(alpha) or alpha<=0:
        raise ValueError('adapter rank/alpha must be positive and finite')
    if type(dropout) not in (int,float) or not 0<=dropout<=1 or type(rslora) is not bool:
        raise ValueError('invalid adapter dropout/RSLoRA policy')
    return rank,float(alpha),float(dropout),rslora


def _finite_pair(pair):
    if set(pair)!={'A','B'} or any(not tensor.is_floating_point() or not torch.isfinite(tensor).all()
                                  for tensor in pair.values()):
        raise ValueError('adapter A/B matrices must contain finite floating values')


def load_peft_adapter(model,directory,*,module_map=None,prefix='base_model.model.',adapter_dtype=torch.float32,
                      parameter_layout=None):
    """Install explicit checkpoint A/B matrices without invoking model-family heuristics.

    module_map maps original module or parameter paths to exact model paths.
    Adapter target paths come from the actual safetensors, not a guessed
    target_modules mapping. Base-model identity remains the caller's selection.
    parameter_layout can specify 'out_in'/'in_out', globally or by source
    parameter name, for ambiguous legacy square expert tensors.
    """
    from safetensors.torch import load_file
    from .parallel_adapters import _ParallelLoRA,_ParallelNF4,local_adapter_matrix,lora_from_parallel_base
    from .parallel_training import ColumnParallelLinear,RowParallelLinear
    from ._parameter_lora import LoRAParameter,local_parameter_adapter,register_parameter_lora
    from ._peft_parameter import split_parameter_pairs,parameter_pair_from_peft
    from .sharded_adapter_interop import parameter_records
    parallel_types=(ColumnParallelLinear,RowParallelLinear,_ParallelNF4)
    directory=Path(directory)
    config=json.loads((directory/'adapter_config.json').read_text(encoding='utf-8'))
    if config.get('peft_type')!='LORA' or config.get('bias','none')!='none':
        raise ValueError('this adapter loader requires LoRA with bias=none')
    unsupported=('use_dora','lora_bias','modules_to_save','layer_replication',
                 'trainable_token_indices','alora_invocation_tokens','arrow_config','kasa_config')
    if any(config.get(key) for key in unsupported) or config.get('fan_in_fan_out',False):
        raise ValueError('checkpoint requests adapter/base transformations outside linear LoRA semantics')
    tensors=load_file(str(directory/'adapter_model.safetensors'),device='cpu')
    pairs={}
    for key,value in tensors.items():
        if not key.startswith(prefix):
            raise ValueError(f'adapter key lacks the explicit prefix: {key}')
        match=re.fullmatch(r'(.+)\.lora_([AB])(?:\.[^.]+)?\.weight',key[len(prefix):])
        if match is None:
            raise ValueError(f'unsupported adapter tensor: {key}')
        source,side=match.groups()
        entry=pairs.setdefault(source,{})
        if side in entry:
            raise ValueError('multiple named adapters are not one adapter checkpoint')
        entry[side]=value
    if not pairs:
        raise ValueError('adapter checkpoint has no linear A/B matrices')
    pairs,parameter_pairs=split_parameter_pairs(model,pairs,config,module_map)
    if parameter_layout is not None:
        layouts=parameter_layout.values() if isinstance(parameter_layout,dict) else (parameter_layout,)
        if any(layout not in ('out_in','in_out') for layout in layouts):
            raise ValueError('parameter_layout must select out_in or in_out for source expert tensors')
    for entry in parameter_pairs:
        options=_options(config,entry['source'])
        if options[2]:raise ValueError('PEFT parameter adapters require dropout=0')
        _finite_pair(entry['pair'])
        layout=parameter_layout.get(entry['source']) if isinstance(parameter_layout,dict) else parameter_layout
        entry['pair']=parameter_pair_from_peft(entry['record'],entry['pair'],options[0],config,layout)
        entry['options']=options
    from .sharded_adapter_interop import has_sharded_units,load_sharded_peft_adapter
    if has_sharded_units(model):
        return load_sharded_peft_adapter(model,pairs,config,module_map=module_map,parameter_pairs=parameter_pairs)
    if adapter_dtype not in (torch.float32,torch.float16,torch.bfloat16):
        raise ValueError('adapter dtype must be FP32/FP16/BF16')
    replacements,aliases={},{}
    plans={}
    modules=dict(model.named_modules(remove_duplicate=False))
    for source,pair in pairs.items():
        path=source if module_map is None else module_map.get(source,source)
        base=modules[path]
        if set(pair)!={'A','B'} or not isinstance(base,(torch.nn.Linear,NF4Linear,LoRALinear,_ParallelLoRA)+parallel_types):
            raise ValueError('adapter target or A/B matrices are missing')
        rank,alpha,dropout,rslora=_options(config,source)
        _finite_pair(pair)
        if pair['A'].shape!=(rank,base.in_features) or pair['B'].shape!=(base.out_features,rank):
            raise ValueError(f'adapter dimensions differ: {source}')
        if isinstance(base,(LoRALinear,_ParallelLoRA)):
            expected=(rank,float(alpha),float(dropout),rslora)
            if (base.rank,base.alpha,base.dropout,base.use_rslora)!=expected:
                raise ValueError('existing adapter configuration differs')
        identity=id(base)
        if identity in plans:
            old_base,old_pair,old_options=plans[identity]
            if not all(torch.equal(old_pair[side],pair[side]) for side in ('A','B')):
                raise ValueError('shared target paths have different adapter matrices')
            if old_options!=(rank,alpha,dropout,rslora):
                raise ValueError('shared adapter paths have different configurations')
        plans[identity]=(base,pair,(rank,alpha,dropout,rslora))
    parameter_plans={}
    records=parameter_records(model)
    for entry in parameter_pairs:
        record,pair,options=entry['record'],entry['pair'],entry['options']
        weight,layer=record['weight'],record['layer']
        owner_path=entry['target'].rpartition('.')[0]
        if any(owner_path==path or owner_path.startswith(path+'.') for path,module in modules.items() if id(module) in plans):
            raise ValueError('parameter and module LoRA targets overlap')
        if layer is not None and (layer.rank,layer.alpha,layer.dropout,layer.use_rslora)!=options:
            raise ValueError('existing parameter adapter configuration differs')
        identity=id(weight)
        if identity in parameter_plans:
            old_record,old_pair,old_options=parameter_plans[identity]
            if old_options!=options or any(not torch.equal(old_pair[side],pair[side]) for side in ('A','B')):
                raise ValueError('shared parameter targets have different adapters')
        parameter_plans[identity]=(record,pair,options)
    parameter_aliases={}
    flags=[(parameter,parameter.requires_grad,parameter.grad) for parameter in model.parameters()]
    try:
        for identity,(base,pair,(rank,alpha,dropout,rslora)) in plans.items():
            factory=lora_from_parallel_base if isinstance(base,parallel_types) else LoRALinear
            layer=base if isinstance(base,(LoRALinear,_ParallelLoRA)) else factory(base,rank=rank,alpha=alpha,
                adapter_dtype=adapter_dtype,dropout=dropout,use_rslora=rslora)
            aliases[identity]=(pair,layer)
            replacements[identity]=layer
        for identity,(record,pair,(rank,alpha,dropout,rslora)) in parameter_plans.items():
            layer=record['layer']
            if layer is None:
                layer=LoRAParameter(record['weight'],rank=rank,alpha=alpha,adapter_dtype=adapter_dtype,
                    use_rslora=rslora,is_transposed=record['weight'].ndim==3 and
                    getattr(record['owner'],'is_transposed',False)).train(record['owner'].training)
            for alias in records.values():
                if id(alias['weight'])==identity and alias['layer'] is not None and alias['layer'] is not layer:
                    raise ValueError('shared parameter targets already have different parametrizations')
            parameter_aliases[identity]=(pair,layer)
    except Exception:
        for parameter,trainable,gradient in flags:
            parameter.requires_grad_(trainable)
            parameter.grad=gradient
        raise
    # Freeze the selected base model only after the entire checkpoint preflight.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad=None
    with torch.no_grad():
        for pair,layer in aliases.values():
            a=local_adapter_matrix(layer,'lora_A',pair['A']) if isinstance(layer,_ParallelLoRA) else pair['A']
            b=local_adapter_matrix(layer,'lora_B',pair['B']) if isinstance(layer,_ParallelLoRA) else pair['B']
            layer.lora_A.copy_(a.to(layer.lora_A))
            layer.lora_B.copy_(b.to(layer.lora_B))
            layer.lora_A.requires_grad_(True)
            layer.lora_B.requires_grad_(True)
        for pair,layer in parameter_aliases.values():
            layer.lora_A.copy_(local_parameter_adapter(layer,'lora_A',pair['A']))
            layer.lora_B.copy_(local_parameter_adapter(layer,'lora_B',pair['B']))
            layer.lora_A.requires_grad_(True)
            layer.lora_B.requires_grad_(True)
    _replace(model,[(name,replacements[id(module)]) for name,module in modules.items()
                    if name and id(module) in replacements])
    registered=set()
    for record in records.values():
        if id(record['weight']) in parameter_aliases and record['layer'] is None:
            key=(id(record['owner']),record['field'])
            if key in registered:continue
            register_parameter_lora(record['owner'],record['field'],parameter_aliases[id(record['weight'])][1])
            registered.add(key)
    return model


def save_peft_adapter(model,directory,*,base_model_name_or_path,prefix='base_model.model.',
                      task_type=None,module_map=None,replica_group=None):
    """Write full, PEFT-compatible matrices into a new directory.

    All DP/TP ranks participate in reconstruction; only the coordinator root
    writes files. FSDP storage requires its actual replica_group coordinator.
    An export spans one model/stage group, not separate pipeline namespaces.
    """
    from safetensors.torch import save_file
    from .parallel_adapters import _ParallelLoRA
    from .sharded_adapter_interop import adapter_layers,has_sharded_units,parameter_records
    from ._parameter_lora import LoRAParameter
    from ._peft_parameter import parameter_wrapper_paths,parameter_pair_to_peft,wrapped_module_path
    storage=adapter_layers(model)
    layers={name:entry[0] for name,entry in storage.items()}
    if has_sharded_units(model) and replica_group is None:
        raise ValueError('FSDP PEFT export requires the actual training coordinator')
    if replica_group is not None:replica_group.validate_model(model)
    if not layers:
        raise ValueError('model contains no LoRA layers')
    first=next(iter(layers.values()))
    if any((layer.dropout,layer.use_rslora)!=(first.dropout,first.use_rslora) for layer in layers.values()):
        raise ValueError('PEFT config requires one dropout and RSLoRA policy')
    if not isinstance(base_model_name_or_path,str) or not base_model_name_or_path:
        raise ValueError('identify the actual base checkpoint')
    rank_pattern,alpha_pattern,tensors={},{},{}
    groups={id(layer.group):layer.group for layer in layers.values() if isinstance(layer,_ParallelLoRA)}
    experts={}
    for layer in layers.values():
        if isinstance(layer,LoRAParameter):
            if layer.tensor_group is not None:groups[id(layer.tensor_group)]=layer.tensor_group
            if layer.expert_group is not None:experts[id(layer.expert_group)]=layer.expert_group
    if len(groups)>1 or len(experts)>1:raise ValueError('PEFT export requires one TP and one EP group per participant')
    if replica_group is None and any(group.world_size>1 for group in groups.values()) and any(group.world_size>1 for group in experts.values()):
        raise ValueError('combined TP/EP PEFT export requires the actual model-parallel coordinator')
    candidates=[group for group in (*groups.values(),*experts.values()) if group.world_size>1]
    group=candidates[0] if candidates else None
    if replica_group is not None:group=replica_group
    names,parameter_names=[],[]
    paths,parameter_targets,parameter_sources={},{},{}
    for name,layer in layers.items():
        if isinstance(layer,LoRAParameter):continue
        target=name if module_map is None else module_map.get(name,name)
        if target in names:
            raise ValueError('adapter export path mapping is not one-to-one')
        names.append(target)
        paths[name]=target
        if layer.rank!=first.rank:rank_pattern[target]=layer.rank
        if layer.alpha!=first.alpha:alpha_pattern[target]=layer.alpha
    canonical={(id(layer),id(unit)):name for name,(layer,unit,path) in storage.items()}
    seen=set()
    for raw,record in parameter_records(model).items():
        layer=record['layer']
        if layer is None:continue
        identity=(id(record['owner']),record['field'],id(record['unit']))
        if identity in seen:continue
        seen.add(identity)
        name=canonical[(id(layer),id(record['unit']))]
        target=raw if module_map is None else module_map.get(raw,module_map.get(name,raw))
        if target in parameter_targets or target in names:
            raise ValueError('adapter export path mapping is not one-to-one')
        parameter_names.append(target)
        parameter_targets[target]=record
        parameter_sources[target]=name
        if layer.rank!=first.rank:rank_pattern[target]=layer.rank
        if layer.alpha!=first.alpha:alpha_pattern[target]=layer.alpha
    wrappers,counts=parameter_wrapper_paths(parameter_targets)
    config={'peft_type':'LORA','base_model_name_or_path':base_model_name_or_path,'task_type':task_type,
            'r':first.rank,'lora_alpha':first.alpha,'lora_dropout':first.dropout,'use_rslora':first.use_rslora,
            'bias':'none','fan_in_fan_out':False,'inference_mode':True,'target_modules':names,
            'rank_pattern':rank_pattern,'alpha_pattern':alpha_pattern}
    if parameter_names:config['target_parameters']=parameter_names
    if parameter_names:config['peft_version']='0.21.2'
    if group is not None:
        group.validate_training_options((str(Path(directory).resolve()),prefix,paths,parameter_sources,config))
    state=adapter_state_dict(model)
    for name,entry in state['layers'].items():
        if isinstance(layers[name],LoRAParameter):continue
        target=wrapped_module_path(paths[name],counts)
        tensors[prefix+target+'.lora_A.weight']=entry['lora_A'].contiguous().clone()
        tensors[prefix+target+'.lora_B.weight']=entry['lora_B'].contiguous().clone()
    for target,record in parameter_targets.items():
        pair=parameter_pair_to_peft(record['layer'],state['layers'][parameter_sources[target]])
        for side,value in pair.items():
            key=prefix+wrappers[target]+'.lora_'+side+'.weight'
            if key in tensors:raise ValueError('PEFT module and parameter wrapper paths collide')
            tensors[key]=value.clone()
    directory=Path(directory)
    error=None
    if group is None or group.rank==0:
        try:
            directory.mkdir(parents=True,exist_ok=False)
            save_file(tensors,str(directory/'adapter_model.safetensors'),metadata={'format':'pt'})
            (directory/'adapter_config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
        except Exception as failure:
            if group is None:raise
            error=f'{type(failure).__name__}: {failure}'
    if group is not None:
        for failure in group.gather_metadata(error):
            if failure:raise RuntimeError(f'PEFT export failed: {failure}')
    return directory
