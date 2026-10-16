"""One loss and gradient contract for DP/CP/EP/TP, including DP-sharded storage."""
from __future__ import annotations

import torch

from .model_parallel_checkpoint import model_mesh_layout


class ModelParallelTrainingGroup:
    """Coordinate a complete model or one pipeline stage on ModelParallelMesh.

    Tokens are distinct on data, context and expert axes; tensor ranks execute
    the same local tokens. Losses passed to backward are local token sums,
    divided by total_weight exactly once. Expert backward already returns the
    contributions routed from all EP ranks, and FSDP backward already performs
    its data reduce-scatter. Neither contribution is reduced a second time.
    """
    supports_gradient_overlap = False

    def __init__(self, model, mesh, *, initialize=True, broadcast_buffers=True):
        from .sharded_training import FullyShardedModule
        from .expert_parallel import ExpertParallelMLP
        if len(mesh.shape) != 5:
            raise TypeError('ModelParallelTrainingGroup requires ModelParallelMesh')
        self.model = model
        self.mesh = mesh.stage_mesh() if mesh.shape[1] != 1 else mesh
        self.transport = self.mesh.world
        self.data, self.context = self.mesh.data, self.mesh.context
        self.expert, self.tensor = self.mesh.expert, self.mesh.tensor
        self.rank, self.world_size = self.transport.rank, self.transport.world_size
        self.device_type, self.cuda_index, self.group = self.transport.device_type, self.transport.cuda_index, self.transport.group
        self._parameters = list(model.named_parameters())
        self._specs = self._parameter_contract(model)
        self._layout = model_mesh_layout(model, self.mesh)
        self._fields = {id(parameter): self._layout[name] for name, parameter in self._parameters}
        self._fsdp = [unit for unit in model.modules() if isinstance(unit, FullyShardedModule)]
        error = None
        if any(unit.group.group is not self.data.group for unit in self._fsdp):
            error = 'FSDP units must shard over the actual mesh data group'
        roots = [model] + [unit._template for unit in self._fsdp]
        for root in roots:
            for unit in root.modules():
                if isinstance(unit, ExpertParallelMLP):
                    if unit.group.group is not self.expert.group:
                        error = 'expert banks must use the actual mesh expert group'
                    tensor = getattr(unit, 'tensor_group', None)
                    if tensor is not None and tensor.group is not self.tensor.group:
                        error = 'tensor-sharded experts must use the actual mesh tensor group'
                elif any(getattr(p, '_ruda_tp_sharded', False) for p in unit.parameters(recurse=False)):
                    peer = getattr(unit, 'group', None)
                    if peer is not None and peer.group is not self.tensor.group:
                        error = 'tensor projections must use the actual mesh tensor group'
        if any(p.device.type != self.device_type or not p.is_contiguous() for _, p in self._parameters):
            error = 'model mesh parameters must be contiguous on the communicator device'
        for failure in self.gather_metadata(error):
            if failure:
                raise ValueError(failure)
        from ._expert_ownership import merge_layouts
        self.validate_training_options((self.mesh.shape, initialize, broadcast_buffers))
        merge_layouts(self.gather_metadata(self._layout))
        schema = {name: dict(self._layout[name], dtype=str(p.dtype), trainable=p.requires_grad)
                  for name, p in self._parameters}
        merge_layouts(self.gather_metadata(schema))
        if initialize:
            self.initialize(broadcast_buffers=broadcast_buffers)

    @staticmethod
    def _parameter_contract(model):
        return [(name, id(p), tuple(p.shape), p.dtype, p.device, p.requires_grad)
                for name, p in model.named_parameters()]

    def gather_metadata(self, value):
        return self.transport.gather_metadata(value)

    def validate_training_options(self, value):
        return self.transport.validate_training_options(value)

    def validate_model(self, model):
        if model is not self.model or self._parameter_contract(model) != self._specs:
            raise ValueError('model mesh parameters changed after coordinator construction')

    @staticmethod
    def _expert_field(spec):
        return spec.get('partition') in ('expert', 'expert-owned')

    @staticmethod
    def _tensor_field(spec):
        if spec.get('partition') in ('expert', 'expert-owned'):
            return spec.get('expert_tensor_axis') is not None
        return not spec.get('replicated', False)

    @torch.no_grad()
    def initialize(self, *, broadcast_buffers=True):
        values = list(self.model.named_parameters())
        if broadcast_buffers:
            values += list(self.model.named_buffers())
        for name, value in values:
            spec = self._layout[name]
            if not spec.get('data_flat'):
                self.data.broadcast_(value, 0)
            self.context.broadcast_(value, 0)
            if not self._expert_field(spec):
                self.expert.broadcast_(value, 0)
            if not self._tensor_field(spec):
                self.tensor.broadcast_(value, 0)
            elif spec.get('tp_replicas', 1) > 1:
                from ._tp_replicas import replica_broadcast
                from ._collective_ops import group_key
                value.copy_(replica_broadcast(value, group_key(self.tensor), spec['tp_replicas']))

    def total_weight(self, local_weight):
        weights = self.gather_metadata(local_weight)
        tensor = self.tensor.world_size
        if any(type(weight) is not int or weight < 0 for weight in weights):
            raise ValueError('model mesh token weights must be nonnegative integers')
        if any(len(set(weights[start:start+tensor])) != 1 for start in range(0, len(weights), tensor)):
            raise ValueError('tensor ranks must agree on their local logical token count')
        total = sum(weights[::tensor])
        if not total:
            raise ValueError('global effective token count must be positive')
        return total

    def validate_microbatches(self, batches):
        # Inputs are caller-supplied CPU token batches, never GPU activations.
        prepared = [{key: value.tolist() if isinstance(value, torch.Tensor) else value
                     for key, value in batch.items()} for batch in batches]
        peers = self.tensor.gather_metadata(prepared)
        error = None if all(peer == prepared for peer in peers) else 'tensor ranks must execute identical local token batches'
        schedules = (self.context, self.expert, self.tensor) + ((self.data,) if self._fsdp else ())
        for group in schedules:
            counts = group.gather_metadata(len(batches))
            if not batches or len(set(counts)) != 1:
                error = 'collective-bearing peers must execute the same nonempty microbatch schedule'
        # Context shards may be ragged, but ring Q/K/V batches and document
        # lanes must describe the same logical sequences on all CP ranks.
        lanes = [(tuple(batch['input_ids'].shape[:-1]),
                  tuple(batch['position_ids'].shape[:-1]) if 'position_ids' in batch else None)
                 for batch in batches]
        if any(peer != lanes for peer in self.context.gather_metadata(lanes)):
            error = 'context ranks must share the logical batch/document lanes'
        for failure in self.gather_metadata(error):
            if failure:
                raise ValueError(failure)

    def sum_(self, value):
        self.data.sum_(value)
        self.context.sum_(value)
        self.expert.sum_(value)
        return value

    def _gradient_presence(self, parameters, *, missing):
        if missing not in ('zero', 'error'):
            raise ValueError('missing gradients must use zero or error')
        error = None
        names = {id(parameter): self._fields[id(parameter)]['name'] for _, parameter in self._parameters}
        flags = {}
        for parameter in parameters:
            gradient = parameter.grad
            flags[names[id(parameter)]] = gradient is not None
            if gradient is not None and (gradient.is_sparse or gradient.shape != parameter.shape
                    or gradient.device != parameter.device or gradient.dtype != parameter.dtype):
                error = 'model mesh gradient storage differs from its parameter'
            if missing == 'error' and parameter.requires_grad and gradient is None:
                error = 'a model mesh parameter has no gradient'
        peers = self.gather_metadata((error, flags))
        for failure, present in peers:
            if failure:
                raise ValueError(failure)
        tensor_world, expert_world = self.tensor.world_size, self.expert.world_size
        expert_rank = self.rank // tensor_world % expert_world
        result = []
        for parameter in parameters:
            name, spec = names[id(parameter)], self._fields[id(parameter)]
            ranks = (range(len(peers)) if not self._expert_field(spec) else
                     [rank for rank in range(len(peers)) if rank // tensor_world % expert_world == expert_rank])
            result.append(any(peers[rank][1].get(name, False) for rank in ranks))
        return result

    @torch.no_grad()
    def _reduce(self, *, local_weight, normalized, missing, skip_data=False):
        total = self.total_weight(local_weight)
        self.validate_training_options((normalized, missing, skip_data))
        parameters = [p for _, p in self._parameters]
        active = self._gradient_presence(parameters, missing=missing)
        for parameter, used in zip(parameters, active, strict=True):
            if not used or not parameter.requires_grad:
                continue
            spec = self._fields[id(parameter)]
            value = (torch.zeros_like(parameter, dtype=torch.float32) if parameter.grad is None
                     else parameter.grad.detach().float().contiguous())
            if not skip_data and not spec.get('data_flat'):
                self.data.sum_(value)
            self.context.sum_(value)
            if not self._expert_field(spec):
                self.expert.sum_(value)
            if getattr(parameter, '_ruda_sp_replicated', False):
                self.tensor.sum_(value)
            if not normalized:
                value.div_(total)
            parameter.grad = value.to(parameter.dtype)
        return total

    def synchronize_gradients(self, *, local_weight, normalized=False, missing='zero'):
        return self._reduce(local_weight=local_weight, normalized=normalized, missing=missing)

    @torch.no_grad()
    def synchronize_optimizer_gradients(self, optimizer, *, local_weight, normalized=False, missing='zero'):
        from .sharded_training import Zero2Optimizer
        if not isinstance(optimizer, Zero2Optimizer):
            return self.synchronize_gradients(local_weight=local_weight, normalized=normalized, missing=missing)
        if self._fsdp or optimizer.group.group is not self.data.group:
            raise ValueError('ZeRO-2 uses the mesh data axis and cannot re-shard FSDP storage')
        total = self._reduce(local_weight=local_weight, normalized=normalized, missing=missing, skip_data=True)
        for parameter, shard in optimizer.entries:
            if parameter.grad is None:
                shard.grad = None
                continue
            padded = torch.zeros(shard.numel() * self.data.world_size, device=shard.device, dtype=torch.float32)
            padded[:parameter.numel()].copy_(parameter.grad.reshape(-1).float())
            shard.grad = self.data.reduce_scatter(padded).to(shard.dtype)
            parameter.grad = None
        optimizer._weight = total
        return total

    def replication_factors(self, parameters):
        data, _, context, expert, tensor = self.mesh.shape
        return {id(p): (1 if self._fields[id(p)].get('data_flat') else data) * context
                * (1 if self._expert_field(self._fields[id(p)]) else expert)
                * (self._fields[id(p)].get('tp_replicas',1) if self._tensor_field(self._fields[id(p)]) else tensor) for p in parameters}

    def prepare_optimizer_step(self, optimizer, *, loss_scale=1.):
        from .sharded_training import Zero2Optimizer
        from .sharded_optim import distributed_grad_norm
        from .optim import Muon
        zero = isinstance(optimizer, Zero2Optimizer)
        actual = optimizer.optimizer if zero else optimizer
        parameters = [p for options in actual.param_groups for p in options['params']]
        mapping = {id(shard): original for original, shard in optimizer.entries} if zero else {}
        originals = [mapping[id(p)] for p in parameters] if zero else parameters
        names = {id(p): name for name, p in self._parameters}
        from ._expert_ownership import merge_layouts
        merge_layouts(self.gather_metadata({names[id(p)]: self._fields[id(p)] for p in originals}))
        if isinstance(actual, Muon) and any(self._tensor_field(self._fields[id(p)]) or
                self._fields[id(p)].get('data_flat') for p in originals):
            raise ValueError('partitioned matrices require the complete-matrix distributed Muon optimizer')
        if getattr(actual, 'complete_tensor_parallel_matrices', False):
            actual.consensus_group = self.transport
        if not hasattr(actual, 'last_step_skipped'):
            return
        device = parameters[0].device if parameters else torch.device('ruda:0' if self.device_type == 'ruda' else 'cpu')
        bad = torch.zeros(1, dtype=torch.float32, device=device)
        for parameter in parameters:
            if parameter.grad is not None:
                bad.add_((~torch.isfinite(parameter.grad.float() / loss_scale)).any().float())
        self.transport.max_(bad)
        if bad.item():
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.fill_(float('inf'))
        elif getattr(actual, 'max_grad_norm', None) is not None and hasattr(actual, '_distributed_grad_norm'):
            factors = self.replication_factors(originals)
            factors = {id(p): factors[id(original)] // (self.data.world_size if zero else 1)
                       for p, original in zip(parameters, originals, strict=True)}
            norm = distributed_grad_norm(parameters, self.transport, loss_scale=loss_scale, replication_factors=factors)
            if zero:
                optimizer._mesh_grad_norm = norm
            else:
                actual._distributed_grad_norm = norm

    def checkpoint_layout(self, layout):
        return model_mesh_layout(self.model, self.mesh)

    def advance_float8_scales(self, model):
        from .float8_scaling import iter_float8_states
        # Each state already reduces over its projection TP group. The replica
        # group covers DP/CP/EP, but not other experts with different weights.
        for path, state in iter_float8_states(model):
            owner = path.removesuffix('.scale_state') if path != 'scale_state' else ''
            prefix = owner + '.' if owner else ''
            expert = any(spec['name'].startswith(prefix) and self._expert_field(spec)
                         for spec in self._layout.values())
            state.advance(group=self.mesh.expert_replica if expert else self.mesh.dense_replica)
