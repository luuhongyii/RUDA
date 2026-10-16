"""Expert-sharded parameters and replicated trunk gradient coordination."""
from __future__ import annotations

import torch
from .expert_parallel import ExpertParallelMLP


class ExpertParallelGroup:
    """SFT coordinator for distinct token batches on an expert process group.

    Expert gradients already contain contributions received from every rank.
    Only the replicated trunk/router gradients are reduced over the EP group.
    normalized=True means each local loss sum used the GLOBAL token divisor,
    matching SFTTrainer/ReplicaGroup, not independent per-rank mean losses.
    """
    def __init__(self, model, group):
        self.model, self.transport = model, group
        self.rank, self.world_size = group.rank, group.world_size
        self.group, self.cuda_index = group.group, group.cuda_index
        self.device_type, self.backend = group.device_type, group.backend
        self._parameters = list(model.named_parameters())
        self._specs = [(name, id(p), tuple(p.shape), p.dtype, p.device, p.requires_grad)
                       for name, p in self._parameters]
        self._expert_ids = set()
        banks = []
        for path, module in model.named_modules():
            if isinstance(module, ExpertParallelMLP):
                if module.group is not group:
                    raise ValueError('expert layers and coordinator must share one communicator')
                self._expert_ids.update(id(p) for p in module.parameters())
                banks.append((path, module.num_experts, module.hidden_size, module.intermediate_size,
                              module.combine, module.activation))
        if not banks:
            raise ValueError('expert coordinator requires partitioned expert modules')
        self._replicated = [(name, p) for name, p in self._parameters if id(p) not in self._expert_ids]
        schema = [(name, tuple(p.shape), str(p.dtype), p.requires_grad) for name, p in self._replicated]
        self.validate_training_options((banks, schema))

    def gather_metadata(self, value):
        return self.transport.gather_metadata(value)

    def validate_training_options(self, value):
        return self.transport.validate_training_options(value)

    def total_weight(self, local_weight):
        return self.transport.total_weight(local_weight)

    def sum_(self, tensor):
        return self.transport.sum_(tensor)

    def validate_model(self, model):
        specs = [(name, id(p), tuple(p.shape), p.dtype, p.device, p.requires_grad)
                 for name, p in model.named_parameters()]
        if model is not self.model or specs != self._specs:
            raise ValueError('expert model parameters changed after coordinator creation')

    def initialize(self, *, root=0, broadcast_buffers=True):
        """Initialize replicas without overwriting distinct expert-owned weights."""
        self.validate_training_options((root, broadcast_buffers))
        with torch.no_grad():
            for _, parameter in self._replicated:
                self.transport.broadcast_(parameter, root)
            if broadcast_buffers:
                for _, buffer in self.model.named_buffers():
                    self.transport.broadcast_(buffer, root)
        return self.model

    def validate_microbatches(self, batches):
        # EP collectives execute in every microbatch, including an empty local shard.
        self.validate_training_options(len(batches))

    def synchronize_gradients(self, *, local_weight, normalized=False, missing='zero', bucket_bytes=None):
        self.validate_model(self.model)
        if missing not in ('error', 'zero') or type(normalized) is not bool:
            raise ValueError('invalid expert gradient reduction policy')
        if bucket_bytes is not None and (type(bucket_bytes) is not int or bucket_bytes < 4):
            raise ValueError('gradient bucket_bytes must be at least four')
        count = self.total_weight(local_weight)
        replicated = [(name, p) for name, p in self._replicated if p.requires_grad]
        flags = tuple(p.grad is not None for _, p in replicated)
        failure = None
        for name, parameter in self._parameters:
            gradient = parameter.grad
            if gradient is not None and (gradient.shape != parameter.shape or gradient.is_sparse
                    or gradient.dtype != parameter.dtype or gradient.device != parameter.device):
                failure = f'invalid expert-model gradient: {name}'
            if (missing == 'error' and parameter.requires_grad and gradient is None
                    and (local_weight > 0 or id(parameter) in self._expert_ids)):
                failure = f'missing expert-model gradient: {name}'
        requests = self.gather_metadata((normalized, missing, bucket_bytes, flags, failure))
        for norm, policy, capacity, other_flags, error in requests:
            if error:
                raise ValueError(error)
            if (norm, policy, capacity, len(other_flags)) != (normalized, missing, bucket_bytes, len(flags)):
                raise ValueError('expert peers supplied different gradient reduction contracts')
        with torch.no_grad():
            for index, (_, parameter) in enumerate(replicated):
                if not any(request[3][index] for request in requests):
                    parameter.grad = None
                    continue
                gradient = torch.zeros_like(parameter) if parameter.grad is None else parameter.grad
                flat = gradient.reshape(-1)
                capacity = flat.numel() if bucket_bytes is None else max(1, bucket_bytes // 4)
                result = torch.empty_like(flat)
                for begin in range(0, flat.numel(), max(1, capacity)):
                    end = min(begin + capacity, flat.numel())
                    work = flat[begin:end].float().clone()
                    self.transport.sum_(work)
                    if not normalized:
                        work.div_(count)
                    result[begin:end].copy_(work.to(parameter.dtype))
                parameter.grad = result.reshape(parameter.shape)
            if not normalized:
                for _, parameter in self._parameters:
                    if id(parameter) in self._expert_ids and parameter.grad is not None:
                        parameter.grad.div_(count)
        return count

    def prepare_optimizer_step(self, optimizer, *, loss_scale=1.):
        parameters = [p for options in optimizer.param_groups for p in options['params']]
        if not parameters:
            return
        if hasattr(optimizer, 'last_step_skipped'):
            overflow = torch.zeros(1, dtype=torch.float32, device=parameters[0].device)
            for parameter in parameters:
                if parameter.grad is not None:
                    overflow.add_((~torch.isfinite(parameter.grad.float() / loss_scale)).any().float())
            self.transport.max_(overflow)
            if overflow.item():
                for parameter in parameters:
                    if parameter.grad is not None:
                        parameter.grad.fill_(float('inf'))
                return
        if getattr(optimizer, 'max_grad_norm', None) is not None and hasattr(optimizer, '_distributed_grad_norm'):
            from .sharded_optim import distributed_grad_norm
            optimizer._distributed_grad_norm = distributed_grad_norm(parameters, self.transport,
                loss_scale=loss_scale, replicated_ids={id(p) for _, p in self._replicated})

    def checkpoint_layout(self, layout):
        for spec in layout.values():
            if spec.get('expert_tensor_axis') is not None:
                raise ValueError('EP/TP expert checkpoints use ModelParallelTrainingGroup')
            if spec.get('partition') not in ('expert', 'expert-owned') and not spec.get('replicated'):
                raise ValueError('combined EP/TP/FSDP checkpointing requires an explicit multi-axis mesh layout')
        return layout
