"""Explicit DP/PP/CP/EP/TP Cartesian communicators."""
from __future__ import annotations

from datetime import timedelta
from itertools import product
from math import prod
import torch.distributed as dist
from .distributed_training import ReplicaGroup, gather_process_metadata


class ModelParallelMesh:
    axes = ('data', 'pipeline', 'context', 'expert', 'tensor')

    def __init__(self, *, data_parallel=1, pipeline_parallel=1, context_parallel=1,
                 expert_parallel=1, tensor_parallel=1, device_type='ruda',
                 transport='ruccl', timeout_seconds=1800):
        self.shape = (data_parallel, pipeline_parallel, context_parallel, expert_parallel, tensor_parallel)
        if any(type(size) is not int or size < 1 for size in self.shape) or timeout_seconds <= 0:
            raise ValueError('model parallel dimensions and timeout must be positive')
        if not dist.is_initialized() or prod(self.shape) != dist.get_world_size():
            raise ValueError('DP/PP/CP/EP/TP dimensions must cover the initialized world')
        if device_type not in ('ruda', 'cpu') or transport not in ('ruccl', 'nccl'):
            raise ValueError('select a RUDA device/transport or explicit CPU reference')
        if device_type == 'cpu' and transport != 'nccl':
            raise ValueError('CPU reference uses Gloo, not ruCCL')
        if transport == 'ruccl':
            from .process_group import register_ruccl_backend
            register_ruccl_backend()
        self.device_type, self.transport = device_type, transport
        self.rank = dist.get_rank()
        remainder = self.rank
        coordinates = [0] * len(self.shape)
        for axis in reversed(range(len(self.shape))):
            coordinates[axis] = remainder % self.shape[axis]
            remainder //= self.shape[axis]
        self.coordinates = dict(zip(self.axes, coordinates))
        contract = self.shape, device_type, transport, timeout_seconds
        if any(other != contract for other in gather_process_metadata(contract)):
            raise ValueError('ranks disagree about the model parallel mesh')
        self.groups, self._owned, self._cache = {}, [], {}
        backend = transport if device_type == 'ruda' else 'gloo'
        timeout = timedelta(seconds=timeout_seconds)
        world = tuple(range(prod(self.shape)))
        if str(dist.get_backend()).lower() == backend:
            self._cache[world] = None

        def create(ranks):
            ranks = tuple(ranks)
            if ranks not in self._cache:
                group = dist.new_group(list(ranks), backend=backend, timeout=timeout)
                self._cache[ranks] = group
                if self.rank in ranks:
                    self._owned.append(group)
            if self.rank in ranks:
                return ReplicaGroup(self._cache[ranks], device_type=device_type, transport=transport)
            return None

        self.world = create(world)
        definitions = {name: (index,) for index, name in enumerate(self.axes)}
        definitions.update(dense_replica=(0, 2, 3), expert_replica=(0, 2),
                           training=(0, 2, 3), stage=(0, 2, 3, 4))
        for name, varying in definitions.items():
            fixed_axes = tuple(axis for axis in range(len(self.shape)) if axis not in varying)
            for fixed in product(*(range(self.shape[axis]) for axis in fixed_axes)):
                coordinate = [0] * len(self.shape)
                for axis, value in zip(fixed_axes, fixed):
                    coordinate[axis] = value
                ranks = []
                for values in product(*(range(self.shape[axis]) for axis in varying)):
                    for axis, value in zip(varying, values):
                        coordinate[axis] = value
                    rank = 0
                    for size, value in zip(self.shape, coordinate):
                        rank = rank * size + value
                    ranks.append(rank)
                group = create(ranks)
                if group is not None:
                    self.groups[name] = group

    @property
    def data(self): return self.groups['data']

    @property
    def pipeline(self): return self.groups['pipeline']

    @property
    def context(self): return self.groups['context']

    @property
    def expert(self): return self.groups['expert']

    @property
    def tensor(self): return self.groups['tensor']

    @property
    def dense_replica(self): return self.groups['dense_replica']

    @property
    def expert_replica(self): return self.groups['expert_replica']

    @property
    def training(self): return self.groups['training']

    @property
    def stage(self): return self.groups['stage']

    def stage_mesh(self):
        return _ModelStageMesh(self)

    def close(self):
        for group in reversed(self._owned):
            dist.destroy_process_group(group)
        self._owned.clear()
        self._cache.clear()


class _ModelStageMesh:
    axes = ModelParallelMesh.axes

    def __init__(self, parent):
        self.parent = parent
        data, _, context, expert, tensor = parent.shape
        self.shape = (data, 1, context, expert, tensor)
        self.device_type, self.transport = parent.device_type, parent.transport
        self.world = self.stage = parent.stage
        self.rank = self.world.rank
        self.coordinates = dict(parent.coordinates, pipeline=0)
        for name in ('data', 'context', 'expert', 'tensor', 'dense_replica', 'expert_replica', 'training'):
            setattr(self, name, getattr(parent, name))

    def stage_mesh(self):
        return self


def pipeline_rank_layout(mesh):
    """Return logical replica/PP/TP indexing without changing real process ranks."""
    if len(mesh.shape) == 3:
        return mesh.shape, mesh.coordinates, tuple(range(prod(mesh.shape)))
    data, pipeline, context, expert, tensor = mesh.shape
    coordinate = mesh.coordinates
    replica = (coordinate['data'] * context + coordinate['context']) * expert + coordinate['expert']
    order = tuple((((d * pipeline + p) * context + c) * expert + e) * tensor + t
                  for d in range(data) for c in range(context) for e in range(expert)
                  for p in range(pipeline) for t in range(tensor))
    return ((data * context * expert, pipeline, tensor),
            (replica, coordinate['pipeline'], coordinate['tensor']), order)
