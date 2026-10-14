"""Native C++ communication programs; no Python tensor-operation replay."""
from __future__ import annotations

import ctypes
from contextlib import ExitStack
from threading import RLock
import torch
from torch.fx.node import map_arg
from ._expert_exchange import _group


_SCHEMAS = {
    'ruda::parallel_region': (4, 1),
    'ruda::collective_reduce': (3, 2),
    'ruda::tp_replica_sum': (3, 1),
    'ruda::tp_unique_gather': (3, 1),
    'ruda::tp_replica_broadcast': (3, 1),
    'ruda::sequence_all_gather': (3, 2),
    'ruda::sequence_reduce_scatter': (3, 2),
    'ruda::context_shift': (3, 2),
    'ruda::context_initial_state': (2, 1),
    'ruda::context_initial_state_backward': (2, 1),
    'ruda::context_max_position': (2, 1),
    'ruda::context_sum_objective': (2, 1),
    'ruda::context_loss_reduce': (3, 2),
    'ruda::expert_exchange': (4, 3),
    'ruda::expert_exchange_backward': (5, 4),
    'ruda::expert_return': (5, 4),
    'ruda::expert_return_backward': (5, 4),
    'ruda::ring_attention_forward': (17, 9),
    'ruda::indexed_ring_attention_forward': (20, 9),
    'ruda::ring_attention_backward': (16, 9),
    'ruda::attention_rng_state': (5, 2),
}


def definition(node):
    if node.op != 'call_function' or not isinstance(node.target, torch._ops.OpOverload) or node.kwargs:
        return None
    schema = node.target._schema
    spec = _SCHEMAS.get(schema.name)
    if spec is None or schema.overload_name or len(node.args) != spec[0]:
        return None
    return schema.name, spec[1]


class _Communicator:
    def __init__(self, group):
        from . import _C
        import torch.distributed as dist
        if getattr(_C, 'collective_program_api_version', 0) != 1:
            raise ValueError('native collective program API 1 unavailable; rebuild the C++ extension')
        if group.device_type != 'ruda':
            raise ValueError('native collective programs require the RUDA device')
        self.owner, self.transport, self.pg = group, None, None
        self.lock = RLock()
        self.handle = None
        pg = group.group if group.group is not None else dist.distributed_c10d._get_default_group()
        if group.transport == 'nccl' or group.world_size == 1:
            if group.world_size != 1 and group.backend != 'nccl':
                raise ValueError('native process-group program requires NCCL, not a CPU payload transport')
            self.native = _C.NativeCollectiveProgram(pg, group.cuda_index)
            return
        transport = group._ruccl
        if transport is None:
            if not hasattr(pg, '_transport'):
                pg = pg._get_backend(torch.device('cuda', group.cuda_index))
            self.pg = pg
            transport = pg._transport
        if getattr(transport, '_ucx', None) is not None:
            transport = transport._ucx
        self.transport = transport
        self.lock = transport._lock
        library = transport.library
        ucx = hasattr(transport, '_next') and hasattr(transport, 'timeout_ms')
        if not ucx and not getattr(transport, 'ipc', False):
            raise ValueError('native ruCCL programs require CUDA IPC or UCX device payload transport')
        names = ('ruccl_ucx_exchange', 'ruccl_ucx_reduce') if ucx else ('ruccl_ipc_collective', 'ruccl_ipc_all_to_all')
        if any(not hasattr(library, name) for name in names):
            raise ValueError('matching ruCCL communication program entry points are unavailable')
        self.handle = transport.handle.value
        if not self.handle:
            raise ValueError('native ruCCL communicator is closed')
        addresses = [self.handle, ctypes.cast(library.ruccl_error, ctypes.c_void_p).value]
        addresses.extend(ctypes.cast(getattr(library, name), ctypes.c_void_p).value for name in names)
        self.ucx = ucx
        self.native = _C.NativeCollectiveProgram(2 if ucx else 1, group.rank, group.world_size,
            group.cuda_index, addresses, transport.timeout_ms if ucx else 0)

    def _run(self, call, reservations):
        with ExitStack() as stack:
            if self.pg is not None:
                stack.enter_context(self.pg._submit_lock)
                # Drain earlier asynchronous process-group work before using
                # the same native communicator on this RUDA execution queue.
                self.pg._executor.submit(lambda: None).result()
            stack.enter_context(self.lock)
            sequence = 0
            if self.transport is not None:
                if self.transport.handle.value != self.handle:
                    raise RuntimeError('native collective communicator was closed or replaced')
                if self.ucx:
                    if self.transport._sequence > (1 << 31) - reservations:
                        raise OverflowError('UCX collective tag space exhausted')
                    sequence = self.transport._sequence
                    self.transport._sequence += reservations
            result = call(sequence)
            return result

    def run(self, op, tensors, option):
        reservations = max(8, 2 * (option - 1).bit_length() + 1) if op in (16,18) else 8
        return self._run(lambda sequence: self.native.run(op, tensors, [option], sequence), reservations)

    def ring(self, backward, tensors, queries, keys, policy, scalars, words):
        return self._run(lambda sequence: self.native.ring(backward, tensors, queries, keys,
            policy, scalars, words, sequence), 5 * self.owner.world_size + 8)


class CollectiveRegion(torch.nn.Module):
    def __init__(self, node, state, phase, options):
        super().__init__()
        self.target, self.template = node.target, node.args
        self.name, self.group_position = definition(node)
        dependencies = []
        def dependency(node):
            if node not in dependencies:
                dependencies.append(node)
            return node
        map_arg(self.template, dependency)
        self.dependencies = tuple(dependencies)
        self.indices = {node: i for i, node in enumerate(dependencies)}
        self.state, self.phase, self.options = state, phase, options
        self._communicator, self._key, self._closed = None, None, False
        self._lock = RLock()

    def forward(self, *values):
        from ._compile_native import NativeCoverageError
        with self._lock:
            self.state.check_open()
            if self._closed:
                raise RuntimeError('native collective program is closed')
            args = map_arg(self.template, lambda node: values[self.indices[node]])
            key = args[self.group_position]
            reason = None
            if self.name in ('ruda::tp_replica_sum', 'ruda::tp_unique_gather', 'ruda::tp_replica_broadcast'):
                from . import _C
                if getattr(_C, 'replicated_tp_program_api_version', 0) != 1:
                    reason = 'native replicated TP program API 1 unavailable'
            if self._communicator is None or key != self._key:
                try:
                    self._communicator = _Communicator(_group(key))
                    self._key = key
                except (ValueError, AttributeError) as exc:
                    reason = str(exc)
            if reason is not None:
                if self.options.native == 'required':
                    raise NativeCoverageError(f'{self.phase}: {reason}')
                self.state.add('region_reference_runs')
                self.state.reason(self.phase, reason)
                return self.target(*args)
            name, option = self.name, 0
            tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
            if name == 'ruda::parallel_region':
                op, option = args[2:]
            elif name == 'ruda::collective_reduce':
                op, option = 9, args[1]
            elif name in ('ruda::tp_replica_sum', 'ruda::tp_unique_gather', 'ruda::tp_replica_broadcast'):
                op, option = {'ruda::tp_replica_sum':16, 'ruda::tp_unique_gather':17, 'ruda::tp_replica_broadcast':18}[name], args[2]
            elif name in ('ruda::sequence_all_gather', 'ruda::sequence_reduce_scatter', 'ruda::context_shift'):
                op = {'ruda::sequence_all_gather': 4, 'ruda::sequence_reduce_scatter': 5, 'ruda::context_shift': 6}[name]
                option = args[1]
            else:
                op = {'ruda::context_initial_state': 7, 'ruda::context_initial_state_backward': 8,
                    'ruda::context_max_position': 10, 'ruda::context_sum_objective': 11,
                    'ruda::context_loss_reduce': 12, 'ruda::expert_exchange': 13,
                    'ruda::expert_exchange_backward': 14, 'ruda::expert_return': 14,
                    'ruda::expert_return_backward': 15}[name]
            # Once native execution starts, a failed collective is never retried
            # through Python: doing so would change every peer's call order.
            with torch.no_grad():
                result = self._communicator.run(op, tensors, option)
            self.state.add('native_collective_program_calls')
            return tuple(result) if op in (12, 13) else result[0]

    def close(self):
        with self._lock:
            self._communicator = None
            self._closed = True


class AttentionRngRegion(CollectiveRegion):
    def forward(self, *values):
        from ._parallel_rng import attention_state
        with self._lock:
            self.state.check_open()
            if self._closed:
                raise RuntimeError('attention RNG metadata region is closed')
            anchor, shape, groups, dropout, requested = map_arg(self.template, lambda node: values[self.indices[node]])
            result = attention_state(tuple(shape), dropout, groups, requested)
            self.state.add('host_attention_rng_reservations')
            return result


class RingRegion(CollectiveRegion):
    def forward(self, *values):
        from ._compile_native import NativeCoverageError
        from ._ring_ops import _native
        with self._lock:
            self.state.check_open()
            if self._closed:
                raise RuntimeError('native ring program is closed')
            args = map_arg(self.template, lambda node: values[self.indices[node]])
            key = args[9]
            indexed = self.name == 'ruda::indexed_ring_attention_forward'
            indexed_backward = self.name == 'ruda::ring_attention_backward' and args[6].shape[1] == 4
            reason = None
            try:
                _native(indexed or indexed_backward)
                if self._communicator is None or key != self._key:
                    self._communicator = _Communicator(_group(key))
                    self._key = key
            except (ValueError, AttributeError, RuntimeError) as exc:
                reason = str(exc)
            if reason is not None:
                if self.options.native == 'required':
                    raise NativeCoverageError(f'{self.phase}: {reason}')
                self.state.add('region_reference_runs')
                self.state.reason(self.phase, reason)
                return self.target(*args)
            group = self._communicator.owner
            backward = self.name == 'ruda::ring_attention_backward'
            causal, left, right, scale, dropout, softcap = args[10:16]
            q, k, v = args[:3]
            batch, query_length, heads, dimension = q.shape
            native_policy = [int(causal), left, right]
            if backward:
                state = args[8]
                if state.device.type != 'cpu' or state.dtype != torch.int64 or state.shape != (5 + group.world_size,):
                    raise ValueError('native ring backward requires the saved CPU partition/RNG metadata')
                saved = state.tolist()
                key_lengths, words, query_lengths = saved[5:], saved[1:5], []
                if saved[0] != sum(key_lengths):
                    raise ValueError('saved ring global key count differs')
                tensors = list(args[:8])
            else:
                requested = args[16]
                if requested.device.type != 'cpu' or requested.dtype != torch.int64 or requested.shape not in ((0,), (4,)):
                    raise ValueError('native ring RNG request requires CPU int64[0 or 4]')
                words = requested.tolist()
                global_heads, head_start, tensor_key = args[17:20] if indexed else (0, 0, '')
                contract = (batch, heads, k.shape[2], dimension, v.shape[3], str(q.dtype),
                    (causal, left, right, scale, dropout, softcap), words, global_heads, head_start)
                peers = group.gather_metadata((query_length, k.shape[1], contract))
                if any(other != contract for _, _, other in peers):
                    raise ValueError('ring peers disagree on head geometry, dtype, RNG request or policy')
                query_lengths, key_lengths = [peer[0] for peer in peers], [peer[1] for peer in peers]
                total_queries, total_keys = sum(query_lengths), sum(key_lengths)
                if total_keys > 0xffffffff or not indexed and batch * total_queries * heads > 0xffffffff:
                    raise ValueError('ring global positions exceed native Philox indexing')
                if indexed:
                    if global_heads <= 0 or head_start < 0 or head_start + heads > global_heads:
                        raise ValueError('ring local query heads lie outside the global head interval')
                    from ._parallel_rng import attention_state
                    words = attention_state((batch, global_heads, total_queries, total_keys), dropout,
                        (key, tensor_key), requested).tolist()
                    native_policy.extend((global_heads, head_start))
                elif not words:
                    local = None
                    if group.rank == 0:
                        if 0 < dropout < 1:
                            from ._varlen_ops import reserve
                            local = reserve(batch * total_queries * heads * ((total_keys + 3) // 4))
                        else:
                            local = [0, 0, 0, 0]
                    words = group.gather_metadata(local)[0]
                tensors = list(args[:9])
            if len(words) != 4 or any(type(word) is not int or not 0 <= word <= 0xffffffff for word in words):
                raise ValueError('ring dropout requires four unsigned 32-bit words')
            with torch.no_grad():
                result = self._communicator.ring(backward, tensors, query_lengths, key_lengths,
                    native_policy, [scale, dropout, softcap], words)
            self.state.add('native_ring_program_calls')
            return tuple(result)


def make_region(node, state, phase, options):
    if definition(node)[0] == 'ruda::attention_rng_state':
        return AttentionRngRegion(node, state, phase, options)
    constructor = RingRegion if definition(node)[0].startswith(('ruda::ring_attention_', 'ruda::indexed_ring_attention_')) else CollectiveRegion
    return constructor(node, state, phase, options)
