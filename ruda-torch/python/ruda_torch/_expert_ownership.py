def owned(spec):
    return spec is not None and spec.get('partition') == 'expert-owned'


def owning_ranks(spec, world):
    if not owned(spec):
        return list(range(world))
    if 'model_mesh' in spec:
        data, pipeline, context, experts, tensors = spec['model_mesh']
        if pipeline != 1 or data * context * experts * tensors != world:
            raise ValueError('expert ownership mesh differs from the checkpoint ranks')
    else:
        experts, tensors = world, 1
    count, identity = spec['num_experts'], spec['expert_id']
    if type(count) is not int or type(identity) is not int or not 0 <= identity < count or count < experts:
        raise ValueError('invalid separate expert checkpoint ownership')
    owner = ((identity + 1) * experts - 1) // count
    return [rank for rank in range(world) if rank // tensors % experts == owner]


def logical_layout(layout):
    result = {}
    for physical, spec in layout.items():
        name = spec.get('name', physical)
        if name in result and result[name] != spec:
            raise ValueError('local checkpoint aliases have inconsistent contracts: ' + name)
        result[name] = spec
    return result


def merge_layouts(layouts):
    layouts = [logical_layout(layout) for layout in layouts]
    result = {}
    for layout in layouts:
        for name, spec in layout.items():
            if name in result and result[name] != spec:
                if 'model_extra' in spec and 'model_extra' in result[name]:
                    continue
                raise ValueError('rank checkpoint field contracts differ: ' + name)
            result[name] = spec
    for name, spec in result.items():
        actual = [rank for rank, layout in enumerate(layouts) if name in layout]
        if actual != owning_ranks(spec, len(layouts)):
            raise ValueError('rank checkpoint field ownership differs: ' + name)
        if 'model_extra' in spec and any(layout[name] != spec for layout in layouts):
            from .model_parallel_checkpoint import merge_extra_contracts
            result[name] = merge_extra_contracts([layout[name] for layout in layouts])
    return result
