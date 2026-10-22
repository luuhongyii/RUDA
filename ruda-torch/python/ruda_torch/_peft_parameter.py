"""PEFT ParamWrapper paths and flattened expert LoRA factors."""
from __future__ import annotations

from collections import defaultdict
import re

from .sharded_adapter_interop import parameter_records


def _wrapped_path(path, counts):
    parts, result, logical = path.split('.'), [], []
    for index, part in enumerate(parts):
        logical.append(part)
        result.append(part)
        if index < len(parts)-1:
            result.extend(['base_layer'] * counts.get('.'.join(logical), 0))
    return '.'.join(result)


def parameter_wrapper_paths(targets):
    """Map qualified original parameter names to nested ParamWrapper paths."""
    owners=defaultdict(list)
    for name,record in targets.items():
        owner,_,field=name.rpartition('.')
        if not owner:
            raise ValueError('PEFT cannot target a parameter on the top-level model')
        owners[owner].append((name,record))
    counts={owner:len(values) for owner,values in owners.items()}
    paths={}
    for owner,values in owners.items():
        values.sort(key=lambda item:item[1]['order'])
        if len({record['order'] for _,record in values})!=len(values):
            raise ValueError('parameter registration order is ambiguous for PEFT wrappers')
        for index,(name,_) in enumerate(values):
            paths[name]=_wrapped_path(owner,counts)+'.base_layer'*(len(values)-index-1)
    return paths,counts


def _unwrapped_path(path,counts):
    parts,result,index=path.split('.'),[],0
    while index<len(parts):
        result.append(parts[index])
        index+=1
        count=counts.get('.'.join(result),0)
        if index<len(parts) and count:
            if parts[index:index+count]!=['base_layer']*count:
                raise ValueError('PEFT module path does not match its parameter wrappers')
            index+=count
    return '.'.join(result)


def split_parameter_pairs(model,pairs,config,module_map):
    patterns=config.get('target_parameters') or []
    if not patterns:
        return pairs,[]
    if (not isinstance(patterns,list) or any(not isinstance(name,str) or not name for name in patterns)
            or len(set(patterns))!=len(patterns)):
        raise ValueError('PEFT target_parameters must be a list of unique parameter names')
    records=parameter_records(model)
    candidates={name:(name,record) for name,record in records.items()}
    if module_map is not None:
        mapped={target for target in module_map.values() if target in records}
        candidates={name:value for name,value in candidates.items() if name not in mapped}
        for source,target in module_map.items():
            if target in records:candidates[source]=(target,records[target])
    selected,seen={},set()
    for source,value in candidates.items():
        record=value[1]
        identity=(id(record['owner']),record['field'],id(record['unit']))
        if identity not in seen and any(source==pattern or source.endswith('.'+pattern) for pattern in patterns):
            selected[source]=value
            seen.add(identity)
    if any(not any(source==pattern or source.endswith('.'+pattern) for source in selected) for pattern in patterns):
        raise ValueError('PEFT parameter target missing; map original qualified parameter names with module_map')
    paths,counts=parameter_wrapper_paths({source:record for source,(_,record) in selected.items()})
    remaining=dict(pairs)
    parameters=[]
    for source,(target,record) in selected.items():
        key=paths[source]
        if key not in remaining or set(remaining[key])!={'A','B'}:
            raise ValueError(f'PEFT parameter A/B matrices missing: {source}')
        parameters.append(dict(source=source,target=target,record=record,pair=remaining.pop(key)))
    modules={}
    for source,pair in remaining.items():
        logical=_unwrapped_path(source,counts)
        if logical in modules:raise ValueError('multiple PEFT adapters map to one module')
        modules[logical]=pair
    return modules,parameters


def parameter_pair_from_peft(record,pair,rank,config,parameter_layout=None):
    layer=record['layer']
    weight=record['weight']
    shape=tuple(layer.weight_shape if layer is not None else
                getattr(weight,'_ruda_parameter_layout',{}).get('shape',weight.shape))
    transposed=(layer.is_transposed if layer is not None else
                weight.ndim==3 and getattr(record['owner'],'is_transposed',False))
    if len(shape) not in (2,3):raise ValueError('PEFT parameters must be 2-D or 3-D')
    experts=shape[0] if len(shape)==3 else 1
    source_transposed=transposed
    if len(shape)==3:
        matches=[]
        for candidate in (False,True):
            outputs,inputs=shape[-2:][::-1] if candidate else shape[-2:]
            if pair['A'].shape==(experts*rank,inputs) and pair['B'].shape==(outputs,rank*experts):
                matches.append(candidate)
        match=re.match(r'^(\d+)\.(\d+)\.(\d+)',config.get('peft_version') or '')
        if parameter_layout is not None:
            source_transposed=parameter_layout=='in_out'
        elif experts==1:
            source_transposed=False
        elif len(matches)==1:
            source_transposed=matches[0]
        elif match is not None and '.dev' not in config['peft_version']:
            version=tuple(map(int,match.groups()))
            if version<(0,19,0):source_transposed=True
            elif version>=(0,20,0):source_transposed=transposed
            else:
                raise ValueError('PEFT 0.19 square experts require explicit parameter_layout=out_in or in_out')
        else:
            raise ValueError('ambiguous PEFT expert layout; identify peft_version or pass explicit parameter_layout')
        # PEFT's single-expert branch uses Linear's B@A without a transpose.
        if experts==1:
            if parameter_layout=='in_out':
                raise ValueError('PEFT single-expert adapters require out_in delta factors')
            source_transposed=False
    outputs,inputs=shape[-2:][::-1] if source_transposed else shape[-2:]
    if pair['A'].shape!=(experts*rank,inputs) or pair['B'].shape!=(outputs,rank*experts):
        raise ValueError('PEFT parameter factors differ from the logical weight shape')
    if len(shape)==2:return pair
    a=pair['A'].reshape(experts,rank,inputs)
    b=pair['B'].reshape(outputs,rank,experts).permute(2,0,1)
    if source_transposed!=transposed:a,b=b.transpose(-2,-1),a.transpose(-2,-1)
    return {'A':a.contiguous(),'B':b.contiguous()}


def parameter_pair_to_peft(layer,pair):
    a,b=pair['lora_A'],pair['lora_B']
    if len(layer.weight_shape)==3:
        if layer.weight_shape[0]==1 and layer.is_transposed:
            if layer.in_features!=layer.out_features:
                raise ValueError('PEFT does not transpose its single-expert delta; this parameter layout cannot be exported')
            a,b=b.transpose(-2,-1),a.transpose(-2,-1)
        a=a.reshape(-1,layer.in_features)
        b=b.permute(1,2,0).reshape(layer.out_features,-1)
    return {'A':a.contiguous(),'B':b.contiguous()}


def wrapped_module_path(path,counts):
    return _wrapped_path(path,counts)
