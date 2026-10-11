"""Explicit replicated training with selectable NCCL or ruCCL GPU communication.

One process owns one RUDA device. CPU/Gloo is a separately selected reference,
not an automatic fallback. Gradient synchronization occurs at accumulation boundaries.
"""
from __future__ import annotations

import ctypes
import torch
import torch.distributed as dist


class ReplicaGroup:
    """A caller-initialized process group, with explicit device and collective order.

    Set RUDA_TORCH_CUDA_DEVICE before the first native operation, set the matching
    torch.cuda device. NCCL uses an NCCL process group; ruCCL uses a Gloo
    process group for metadata only and its own single-host tensor transport.
    Each rank keeps a full model and optimizer. This is data parallelism, not FSDP.
    """
    def __init__(self, process_group=None, *, device_type='ruda', transport='nccl', ruccl_algorithm='auto'):
        if not dist.is_initialized():
            raise RuntimeError('initialize torch.distributed before constructing ReplicaGroup')
        if transport not in ('nccl', 'ruccl'):
            raise ValueError('select nccl or ruccl transport')
        if device_type not in ('ruda', 'cpu', 'cuda', 'mps') or (device_type in ('cuda', 'mps') and transport != 'ruccl'):
            raise ValueError('native PyTorch GPU tensors require explicit ruCCL transport')
        if device_type == 'cpu' and transport == 'ruccl':
            raise ValueError('ruCCL transport requires a GPU device')
        self.group = process_group
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        self.device_type = device_type
        self.cuda_index = None
        if self.rank < 0 or self.world_size < 1:
            raise ValueError('this process must belong to the group')
        expected = 'nccl' if device_type == 'ruda' and transport == 'nccl' else 'gloo'
        if str(dist.get_backend(process_group)).lower() != expected:
            raise ValueError(f'{device_type} training requires the {expected} process-group backend')
        if device_type == 'ruda':
            from . import _C, _native
            if not hasattr(_C, '_cuda_alias') or not hasattr(_native, 'ruda_torch_cuda_device_index'):
                raise RuntimeError('rebuild matching native components with distributed interop')
            query = _native.ruda_torch_cuda_device_index
            query.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
            query.restype = ctypes.c_int32
            index = ctypes.c_uint32()
            if query(ctypes.byref(index)) != 0:
                raise RuntimeError('native CUDA device selection failed')
            self.cuda_index = index.value
            if torch.cuda.current_device() != self.cuda_index:
                raise ValueError('PyTorch NCCL and RUDA must select the same visible CUDA ordinal')
        self._schema = None
        self._parameters = []
        self._parameter_specs = []
        self._ruccl = None
        if transport == 'ruccl':
            from .ruccl_transport import RucclTransport
            if device_type == 'mps':
                self.cuda_index = 0
            elif self.cuda_index is None:
                self.cuda_index = torch.cuda.current_device()
            self._ruccl = RucclTransport(self.group, self.cuda_index, algorithm=ruccl_algorithm,
                backend='metal' if device_type == 'mps' else 'cuda' if device_type == 'ruda' else None)

    def gather_metadata(self, value):
        """Gather rank-ordered, trusted application metadata, not tensor payloads."""
        result = [None] * self.world_size
        dist.all_gather_object(result, value, group=self.group)
        return result

    def _alias(self, tensor):
        if tensor.device.type != self.device_type or not tensor.is_contiguous():
            raise ValueError('collective tensor must be contiguous on the selected device')
        if self.device_type in ('cpu', 'cuda', 'mps'):
            return tensor.detach()
        from . import _C
        _C.stream_command(11)
        return _C._cuda_alias(tensor.detach(), self.cuda_index)

    def _complete(self, tensors):
        # Cross-runtime ownership is fenced explicitly; no CPU copy of the tensor payload.
        if self.device_type == 'ruda':
            torch.cuda.synchronize(self.cuda_index)
        for tensor in tensors:
            torch.autograd.graph.increment_version(tensor)

    def sum_(self, tensor):
        """All-reduce a dense tensor in place; valid only outside an active autograd graph."""
        if self._ruccl is not None:
            self._ruccl.collective(self._alias(tensor), 0)
            self._complete([tensor])
            return tensor
        alias = self._alias(tensor)
        dist.all_reduce(alias, op=dist.ReduceOp.SUM, group=self.group)
        self._complete([tensor])
        return tensor

    def max_(self, tensor):
        """Dense in-place maximum reduction, using the same allocation fence."""
        if self._ruccl is not None:
            self._ruccl.collective(self._alias(tensor), 1)
            self._complete([tensor])
            return tensor
        dist.all_reduce(self._alias(tensor),op=dist.ReduceOp.MAX,group=self.group)
        self._complete([tensor])
        return tensor

    def min_(self, tensor):
        """Dense in-place minimum reduction, using the same allocation fence."""
        if self._ruccl is not None:
            self._ruccl.collective(self._alias(tensor), 2)
            self._complete([tensor])
            return tensor
        dist.all_reduce(self._alias(tensor),op=dist.ReduceOp.MIN,group=self.group)
        self._complete([tensor])
        return tensor

    def broadcast_(self, tensor, root=0):
        """Broadcast from a group-relative rank, preserving RUDA allocation ownership."""
        if not isinstance(root, int) or isinstance(root, bool) or not 0 <= root < self.world_size:
            raise ValueError('invalid group-relative root')
        if self._ruccl is not None:
            self._ruccl.collective(self._alias(tensor), 3, root=root)
            self._complete([tensor])
            return tensor
        alias = self._alias(tensor)
        source = root if self.group is None else dist.get_global_rank(self.group, root)
        dist.broadcast(alias, src=source, group=self.group)
        self._complete([tensor])
        return tensor

    def all_gather(self, tensor, *, axis=0):
        """Concatenate equally shaped rank-local tensors along an explicit axis."""
        if tensor.ndim == 0 or not -tensor.ndim <= axis < tensor.ndim:
            raise ValueError('all-gather axis is out of range')
        axis %= tensor.ndim
        source = tensor.movedim(axis, 0).contiguous()
        shape = list(source.shape)
        shape[0] *= self.world_size
        output = source.new_empty(shape)
        if self._ruccl is not None:
            self._ruccl.collective(self._alias(source), 4, output=self._alias(output))
        else:
            dist.all_gather_into_tensor(self._alias(output), self._alias(source), group=self.group)
        self._complete([output])
        return output.movedim(0, axis).contiguous()

    def reduce_scatter(self, tensor, *, axis=0):
        """Sum and scatter equal slices; no averaging or implicit dtype conversion."""
        if tensor.ndim == 0 or not -tensor.ndim <= axis < tensor.ndim:
            raise ValueError('reduce-scatter axis is out of range')
        axis %= tensor.ndim
        if tensor.shape[axis] % self.world_size:
            raise ValueError('reduce-scatter dimension must divide the group size')
        source = tensor.movedim(axis, 0).contiguous()
        shape = list(source.shape)
        shape[0] //= self.world_size
        output = source.new_empty(shape)
        if self._ruccl is not None:
            self._ruccl.collective(self._alias(source), 5, output=self._alias(output))
        else:
            dist.reduce_scatter_tensor(self._alias(output), self._alias(source), group=self.group)
        self._complete([output])
        return output.movedim(0, axis).contiguous()

    def begin_gradient_overlap(self, *, global_weight, normalized=False, bucket_bytes=25*1024*1024):
        """Start bucket reductions during the FINAL backward of an accumulation window.

        Earlier microbatches accumulate normally. Finish before any optimizer step.
        Bucket order is fixed across ranks, including locally unused parameters.
        """
        from .parallel_training import GradientOverlap
        return GradientOverlap(self, global_weight=global_weight, normalized=normalized,
                               bucket_bytes=bucket_bytes)

    def initialize(self, model, *, root=0, broadcast_buffers=True):
        """Validate replicas/aliases collectively and broadcast state before optimizer creation."""
        schema = []
        aliases = {}
        parameters = []
        error = None
        for name, parameter in model.named_parameters(remove_duplicate=False):
            identity = id(parameter)
            alias = aliases.setdefault(identity, len(aliases))
            schema.append(('parameter', name, tuple(parameter.shape), str(parameter.dtype), parameter.requires_grad, alias))
            if parameter.device.type != self.device_type or not parameter.is_contiguous():
                error = 'replica parameters must be contiguous on the selected device'
            if parameter.requires_grad and parameter.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                error = 'trainable replica parameters require FP32, FP16 or BF16'
            if identity not in {id(p) for _, p in parameters}:
                parameters.append((name, parameter))
        buffers = []
        buffer_aliases = {}
        if not isinstance(broadcast_buffers, bool):
            error = 'broadcast_buffers must be a bool'
        if broadcast_buffers:
            for name, buffer in model.named_buffers(remove_duplicate=False):
                identity = id(buffer)
                alias = buffer_aliases.setdefault(identity, len(buffer_aliases))
                schema.append(('buffer', name, tuple(buffer.shape), str(buffer.dtype), False, alias))
                if buffer.device.type != self.device_type or not buffer.is_contiguous():
                    error = 'replica buffers must be contiguous on the selected device'
                if identity not in {id(p) for p in buffers}:
                    buffers.append(buffer)
        if not isinstance(root, int) or isinstance(root, bool) or not 0 <= root < self.world_size:
            error = 'invalid replica root'
        requests = self.gather_metadata((schema, root, broadcast_buffers, error))
        for other_schema, other_root, other_buffers, error in requests:
            if error:
                raise ValueError(error)
            if (other_schema, other_root, other_buffers) != (schema, root, broadcast_buffers):
                raise ValueError('replica structure, aliases, root or buffer policy differs')
        with torch.no_grad():
            for _, parameter in parameters:
                self.broadcast_(parameter, root)
            for buffer in buffers:
                self.broadcast_(buffer, root)
        self._schema = schema
        self._parameters = parameters
        self._parameter_specs = [(tuple(p.shape), p.dtype, p.device, p.requires_grad) for _, p in parameters]
        return model

    def validate_model(self, model):
        """Require the same parameter objects used at initialization, before an update."""
        current = list(model.named_parameters())
        if len(current) != len(self._parameters) or any(
                name != saved_name or parameter is not saved_parameter
                for (name, parameter), (saved_name, saved_parameter) in zip(current, self._parameters)):
            raise ValueError('initialize the final model before optimizer/trainer creation; parameters changed')
        if any((tuple(p.shape), p.dtype, p.device, p.requires_grad) != spec
               for (_, p), spec in zip(current, self._parameter_specs)):
            raise ValueError('replica shape/dtype/device/frozen configuration changed after initialization')

    def total_weight(self, local_weight):
        """Exact global token/sample count; every rank participates, including weight-zero ranks."""
        counts = self.gather_metadata(local_weight)
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in counts):
            raise ValueError('effective weights must be nonnegative integers')
        total = sum(counts)
        if total == 0:
            raise ValueError('global effective weight must be positive')
        return total

    def synchronize_gradients(self, *, local_weight, normalized=False, missing='error', bucket_bytes=None):
        """Reduce accumulated loss-sum gradients into a global token mean.

        Set normalized=True only when each local loss sum was already divided by
        total_weight before backward, as SFTTrainer does. A globally unused parameter
        retains grad=None, so optimizers do not apply decay to it.
        """
        if self._schema is None:
            raise RuntimeError('initialize a replica before reducing gradients')
        if bucket_bytes is not None and (not isinstance(bucket_bytes, int) or isinstance(bucket_bytes, bool) or bucket_bytes < 4):
            raise ValueError('gradient bucket_bytes must be at least four')
        present = []
        error = None
        if missing not in ('error', 'zero') or not isinstance(normalized, bool):
            error = 'invalid reduction policy'
        for _, parameter in self._parameters:
            if not parameter.requires_grad:
                continue
            gradient = parameter.grad
            present.append(gradient is not None)
            if gradient is not None and (gradient.shape != parameter.shape or gradient.dtype != parameter.dtype
                    or gradient.device != parameter.device or gradient.is_sparse or not gradient.is_contiguous()):
                error = 'gradient shape/dtype/device/layout differs from the parameter'
        requests = self.gather_metadata((local_weight, normalized, missing, present, error, bucket_bytes))
        count = 0
        globally_present = [False] * len(present)
        for weight, other_normalized, other_missing, flags, error, other_bucket in requests:
            if error:
                raise ValueError(error)
            if not isinstance(weight, int) or isinstance(weight, bool) or weight < 0:
                raise ValueError('effective weight must be a nonnegative integer')
            if (other_normalized, other_missing, len(flags), other_bucket) != (normalized, missing, len(present), bucket_bytes):
                raise ValueError('ranks disagree on gradient reduction policy')
            if weight and missing == 'error' and not all(flags):
                raise ValueError('a positive-weight rank is missing a trainable gradient')
            count += weight
            if weight:
                globally_present = [a or b for a, b in zip(globally_present, flags)]
        if count == 0:
            raise ValueError('global effective weight must be positive')
        index = 0
        with torch.no_grad():
            pending = []
            pending_elements = 0
            capacity = None if bucket_bytes is None else bucket_bytes // 4
            def flush():
                nonlocal pending_elements
                if not pending:
                    return
                work = torch.cat([piece for piece, _, _ in pending])
                self.sum_(work)
                if not normalized:
                    work.div_(count)
                position = 0
                for piece, gradient, offset in pending:
                    gradient.view(-1)[offset:offset+piece.numel()].copy_(work[position:position+piece.numel()])
                    position += piece.numel()
                pending.clear()
                pending_elements = 0
            for _, parameter in self._parameters:
                if not parameter.requires_grad:
                    continue
                active = globally_present[index]
                index += 1
                if not active:
                    parameter.grad = None
                    continue
                gradient = parameter.grad if local_weight and parameter.grad is not None else torch.zeros_like(parameter)
                if capacity is not None:
                    parameter.grad = gradient
                    flat = gradient.view(-1)
                    offset = 0
                    while offset < flat.numel():
                        length = min(capacity-pending_elements, flat.numel()-offset)
                        pending.append((flat[offset:offset+length].float(), gradient, offset))
                        pending_elements += length
                        offset += length
                        if pending_elements == capacity:
                            flush()
                    continue
                work = gradient.float()
                self.sum_(work)
                if not normalized:
                    work.div_(count)
                parameter.grad = work.to(parameter.dtype)
            flush()
        return count

    def validate_training_options(self, value):
        """Require identical trusted run/scaler/step metadata before a distributed update."""
        values = self.gather_metadata(value)
        if any(item != value for item in values):
            raise ValueError('rank training/scaler options differ')
