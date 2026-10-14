import ctypes
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import threading

import torch
import torch.distributed as dist

from .ruccl_transport import RucclTransport


class RuCCLWork(dist.Work):
    def __init__(self, future, timeout):
        super().__init__()
        self._future = future
        self._timeout = timeout
        self._complete = threading.Event()
        future.then(lambda _: self._complete.set())

    def wait(self, timeout=None):
        seconds = self._timeout.total_seconds()
        if timeout is not None and timeout.total_seconds() > 0:
            seconds = timeout.total_seconds()
        if not self._complete.wait(seconds):
            raise TimeoutError('ruCCL Work did not complete before the process-group timeout')
        self._future.wait()
        return True

    def get_future(self):
        return self._future

    def is_completed(self):
        return self._future.done()


class ProcessGroupRuCCL(dist.ProcessGroup):
    def __init__(self, store, rank, world_size, timeout=timedelta(minutes=5)):
        super().__init__(rank, world_size)
        self._rank, self._world = rank, world_size
        self._group_name = ''
        self._group_desc = ''
        self._timeout = timeout
        self._store = store
        self._metadata_sequence = 0
        self._metadata_lock = threading.Lock()
        self._device = torch.cuda.current_device()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ruccl-process-group')
        self._submit_lock = threading.Lock()
        self._p2p_executors = {}
        self._stream = torch.cuda.Stream(device=self._device)
        library = ctypes.CDLL(os.environ['RUCCL_TORCH_LIBRARY'])
        library.ruccl_error.restype = ctypes.c_char_p
        from ._ruccl_ucx import UcxTransport, local_host_id
        requested = os.environ.get('RUCCL_CUDA_TRANSPORT', 'auto')
        topology = self.gather_metadata((local_host_id(), requested))
        if any(path != requested for _, path in topology) or requested not in ('auto', 'ipc', 'ucx'):
            raise ValueError('ruCCL ProcessGroup needs the same auto/ipc/ucx transport on every rank')
        same_host = len({host for host, _ in topology}) == 1
        transport = ('ipc' if same_host else 'ucx') if requested == 'auto' else requested
        if transport == 'ipc' and not same_host:
            raise ValueError('CUDA IPC cannot cross hosts; select RUCCL_CUDA_TRANSPORT=ucx')
        if transport == 'ucx':
            self._transport = UcxTransport(library, self._device, rank, world_size,
                self.gather_metadata, max(1, int(timeout.total_seconds()*1000)))
            return
        library.ruccl_server.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        if rank == 0:
            uid, port = (ctypes.c_uint8*16)(), ctypes.c_uint32()
            if library.ruccl_server(world_size, uid, ctypes.byref(port)):
                raise RuntimeError(library.ruccl_error().decode())
            store.set('ruccl/rendezvous', json.dumps((bytes(uid).hex(), port.value)))
        uid, port = json.loads(store.get('ruccl/rendezvous'))
        self._transport = RucclTransport(None, self._device, backend='cuda', rank=rank,
            world_size=world_size, rendezvous=(bytes.fromhex(uid), port), cuda_transport='ipc',
            timeout_ms=max(1, int(timeout.total_seconds()*1000)))

    def getBackendName(self):
        return 'ruccl'

    def gather_metadata(self, value):
        with self._metadata_lock:
            sequence = self._metadata_sequence
            self._metadata_sequence += 1
            prefix = f'ruccl/metadata/{sequence}'
            keys = [f'{prefix}/value/{rank}' for rank in range(self._world)]
            self._store.set(keys[self._rank], pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
            self._store.wait(keys, self._timeout)
            values = [pickle.loads(self._store.get(key)) for key in keys]
            acknowledgements = [f'{prefix}/read/{rank}' for rank in range(self._world)]
            self._store.set(acknowledgements[self._rank], b'1')
            if self._rank == 0:
                self._store.wait(acknowledgements, self._timeout)
                for key in keys + acknowledgements:
                    self._store.delete_key(key)
            return values

    def _set_group_name(self, name):
        self._group_name = name

    @property
    def group_name(self):
        return self._group_name

    def _set_group_desc(self, description):
        self._group_desc = description

    @property
    def group_desc(self):
        return self._group_desc

    getGroupName = lambda self: self._group_name
    setGroupName = _set_group_name
    getGroupDesc = lambda self: self._group_desc
    setGroupDesc = _set_group_desc

    def _alias(self, tensor):
        if tensor.device.type == 'cuda':
            if tensor.device.index != self._device:
                raise ValueError('each ruCCL process group owns one CUDA device')
            return tensor.detach()
        if tensor.device.type == 'ruda':
            from ruda_torch import _C, _native
            index = ctypes.c_uint32()
            query = _native.ruda_torch_cuda_device_index
            query.argtypes, query.restype = [ctypes.POINTER(ctypes.c_uint32)], ctypes.c_int32
            if query(ctypes.byref(index)) or index.value != self._device:
                raise ValueError('RUDA and this process group must select the same CUDA ordinal')
            if tensor.numel() == 0:
                return torch.empty(tensor.shape, dtype=tensor.dtype, device=f'cuda:{self._device}')
            return _C._cuda_alias(tensor.detach(), self._device)
        raise ValueError('ruCCL DDP requires CUDA or RUDA device tensors; there is no CPU fallback')

    def _submit(self, tensors, operation, executor=None):
        tensors = list(tensors)
        for tensor in tensors:
            if tensor.device.type not in ('cuda', 'ruda'):
                raise ValueError('ruCCL process-group tensors must be on CUDA or RUDA')
        ruda_event = None
        if any(t.device.type == 'ruda' for t in tensors):
            from ruda_torch import current_stream
            ruda_event = current_stream().record_event()
        event = torch.cuda.current_stream(self._device).record_event()
        future = torch.futures.Future()

        def execute():
            try:
                with torch.cuda.device(self._device), torch.no_grad(), torch.cuda.stream(self._stream if executor is None else torch.cuda.Stream(device=self._device)):
                    event.synchronize()
                    if ruda_event is not None:
                        ruda_event.synchronize()
                    operation()
                    torch.cuda.current_stream(self._device).synchronize()
                    if ruda_event is not None:
                        from ruda_torch import current_stream
                        current_stream().synchronize()
                future.set_result(tensors)
            except Exception as error:
                future.set_exception(error)

        with self._submit_lock:
            (self._executor if executor is None else executor).submit(execute)
        return RuCCLWork(future, self._timeout)

    def _collect(self, tensor, op, output=None, root=0):
        if not tensor.is_contiguous() or (output is not None and not output.is_contiguous()):
            raise ValueError('ruCCL collectives require contiguous tensors')
        self._transport.collective(self._alias(tensor), op,
            output=None if output is None else self._alias(output), root=root)
        torch.autograd.graph.increment_version(tensor if output is None else output)

    def allreduce(self, tensors, opts=dist.AllreduceOptions()):
        code = next((code for op,code in [(dist.ReduceOp.SUM,0), (dist.ReduceOp.MAX,1),
            (dist.ReduceOp.MIN,2), (dist.ReduceOp.AVG,0)] if opts.reduceOp == op), None)
        if code is None:
            raise ValueError('ruCCL all-reduce supports SUM, MAX, MIN and floating AVG')
        if opts.reduceOp == dist.ReduceOp.AVG and any(not t.is_floating_point() for t in tensors):
            raise ValueError('ruCCL AVG requires floating tensors')
        def run():
            for tensor in tensors:
                self._collect(tensor, code)
                if opts.reduceOp == dist.ReduceOp.AVG:
                    tensor.div_(self._world)
        return self._submit(tensors, run)

    def allreduce_coalesced(self, tensors, opts=dist.AllreduceCoalescedOptions()):
        return self.allreduce(tensors, opts)

    def broadcast(self, tensors, opts=dist.BroadcastOptions()):
        if opts.rootTensor != 0:
            raise ValueError('ruCCL exposes one device per process; rootTensor must be zero')
        def run():
            for tensor in tensors:
                self._collect(tensor, 3, root=opts.rootRank)
        return self._submit(tensors, run)

    def allgather(self, output_tensors, input_tensors, opts=None):
        if len(output_tensors) != len(input_tensors):
            raise ValueError('all-gather input/output list lengths differ')
        for outputs, tensor in zip(output_tensors, input_tensors):
            if len(outputs) != self._world or any(o.shape != tensor.shape or o.dtype != tensor.dtype or o.device != tensor.device for o in outputs):
                raise ValueError('all-gather requires equal rank-ordered output shapes/dtypes/devices')
        def run():
            for outputs, tensor in zip(output_tensors, input_tensors):
                joined = torch.empty((self._world*tensor.numel(),), dtype=tensor.dtype, device=tensor.device)
                self._collect(tensor.contiguous(), 4, joined)
                for output, part in zip(outputs, joined.view(self._world, *tensor.shape)):
                    output.copy_(part)
        return self._submit([t for group in output_tensors for t in group], run)

    def _allgather_base(self, output, tensor, opts=None):
        return self._submit([output], lambda: self._collect(tensor, 4, output))

    all_gather_single = _allgather_base

    def allgather_into_tensor_coalesced(self, outputs, tensors, opts=None):
        if len(outputs) != len(tensors):
            raise ValueError('all-gather input/output list lengths differ')
        def run():
            for output, tensor in zip(outputs, tensors):
                self._collect(tensor, 4, output)
        return self._submit(outputs, run)

    all_gather_single_coalesced = allgather_into_tensor_coalesced

    def reduce_scatter(self, output_tensors, input_tensors, opts=dist.ReduceScatterOptions()):
        if not (opts.reduceOp == dist.ReduceOp.SUM or opts.reduceOp == dist.ReduceOp.AVG):
            raise ValueError('ruCCL reduce-scatter supports SUM and floating AVG')
        if opts.reduceOp == dist.ReduceOp.AVG and any(not t.is_floating_point() for t in output_tensors):
            raise ValueError('ruCCL AVG requires floating tensors')
        if len(output_tensors) != len(input_tensors):
            raise ValueError('reduce-scatter input/output list lengths differ')
        for output, inputs in zip(output_tensors, input_tensors):
            if len(inputs) != self._world or any(t.shape != output.shape or t.dtype != output.dtype or t.device != output.device for t in inputs):
                raise ValueError('reduce-scatter requires equal rank-ordered input shapes/dtypes/devices')
        def run():
            for output, inputs in zip(output_tensors, input_tensors):
                self._collect(torch.cat([t.reshape(-1) for t in inputs]), 5, output)
                if opts.reduceOp == dist.ReduceOp.AVG:
                    output.div_(self._world)
        return self._submit(output_tensors, run)

    def _reduce_scatter_base(self, output, tensor, opts=dist.ReduceScatterOptions()):
        return self.reduce_scatter_tensor_coalesced([output], [tensor], opts)

    reduce_scatter_single = _reduce_scatter_base

    def reduce_scatter_tensor_coalesced(self, outputs, tensors, opts=dist.ReduceScatterOptions()):
        if not (opts.reduceOp == dist.ReduceOp.SUM or opts.reduceOp == dist.ReduceOp.AVG) or len(outputs) != len(tensors):
            raise ValueError('reduce-scatter requires SUM/AVG and matching input/output lists')
        if opts.reduceOp == dist.ReduceOp.AVG and any(not t.is_floating_point() for t in outputs):
            raise ValueError('ruCCL AVG requires floating tensors')
        def run():
            for output, tensor in zip(outputs, tensors):
                self._collect(tensor, 5, output)
                if opts.reduceOp == dist.ReduceOp.AVG:
                    output.div_(self._world)
        return self._submit(outputs, run)

    reduce_scatter_single_coalesced = reduce_scatter_tensor_coalesced

    def alltoall_base(self, output, tensor, output_split_sizes, input_split_sizes, opts=None):
        if tensor.ndim < 1 or output.ndim < 1 or tensor.shape[1:] != output.shape[1:]:
            raise ValueError('all-to-all tensors require matching non-split dimensions')
        def splits(requested, rows):
            if requested:
                if len(requested) != self._world or any(n < 0 for n in requested) or sum(requested) != rows:
                    raise ValueError('all-to-all split sizes must cover dimension zero')
                return list(requested)
            if rows % self._world:
                raise ValueError('equal all-to-all split requires divisible dimension zero')
            return [rows//self._world]*self._world
        row_items = tensor.numel()//tensor.shape[0] if tensor.shape[0] else output.numel()//output.shape[0] if output.shape[0] else 1
        sends = [n*row_items for n in splits(input_split_sizes, tensor.shape[0])]
        receives = [n*row_items for n in splits(output_split_sizes, output.shape[0])]
        def run():
            self._transport.all_to_all(self._alias(tensor), self._alias(output), sends, receives)
            if output.device.type == 'ruda':
                torch.autograd.graph.increment_version(output)
        return self._submit([output], run)

    def alltoall(self, outputs, tensors, opts=None):
        if len(outputs) != self._world or len(tensors) != self._world:
            raise ValueError('all-to-all lists must have one tensor per rank')
        reference = tensors[0]
        if any(t.dtype != reference.dtype or t.device != reference.device for t in list(tensors)+list(outputs)):
            raise ValueError('all-to-all input/output dtype/device mismatch')
        def run():
            joined = torch.cat([t.reshape(-1) for t in tensors])
            output = torch.empty(sum(t.numel() for t in outputs), dtype=reference.dtype, device=reference.device)
            self._transport.all_to_all(self._alias(joined), self._alias(output),
                [t.numel() for t in tensors], [t.numel() for t in outputs])
            offset = 0
            for tensor in outputs:
                tensor.copy_(output[offset:offset+tensor.numel()].view(tensor.shape))
                offset += tensor.numel()
        return self._submit(outputs, run)

    def _point_to_point(self, tensors, peer, tag, receive):
        if len(tensors) != 1:
            raise ValueError('ruCCL point-to-point supports one tensor per operation')
        key = (peer, receive)
        with self._submit_lock:
            executor = self._p2p_executors.get(key)
            if executor is None:
                executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f'ruccl-p2p-{peer}-{int(receive)}')
                self._p2p_executors[key] = executor
        tensor = tensors[0]
        def run():
            self._transport.p2p(self._alias(tensor), peer, tag, receive=receive)
            if receive and tensor.device.type == 'ruda':
                torch.autograd.graph.increment_version(tensor)
        work = self._submit(tensors, run, executor)
        return work

    def send(self, tensors, dst_rank, tag=0):
        return self._point_to_point(tensors, dst_rank, tag, False)

    def recv(self, tensors, src_rank, tag=0):
        return self._point_to_point(tensors, src_rank, tag, True)

    def reduce(self, tensors, opts=dist.ReduceOptions()):
        code = next((code for op,code in [(dist.ReduceOp.SUM,0), (dist.ReduceOp.MAX,1),
            (dist.ReduceOp.MIN,2)] if opts.reduceOp == op), None)
        if code is None:
            raise ValueError('ruCCL reduce supports SUM, MAX and MIN')
        def run():
            for tensor in tensors:
                work = tensor.clone()
                self._collect(work, code)
                if self._rank == opts.rootRank:
                    tensor.copy_(work)
        return self._submit(tensors, run)

    def barrier(self, opts=dist.BarrierOptions()):
        def run():
            empty = torch.empty(0, device=f'cuda:{self._device}')
            self._transport.collective(empty, 0)
        return self._submit([], run)

    def stats(self):
        return self._transport.stats()

    def shutdown(self):
        for executor in self._p2p_executors.values():
            executor.shutdown(wait=True)
        self._p2p_executors.clear()
        self._executor.shutdown(wait=True)
        self._transport.close()


def register_ruccl_backend():
    if 'RUCCL' not in dist.Backend._plugins:
        dist.Backend.register_backend('ruccl', ProcessGroupRuCCL,
            devices=['cuda', torch._C._get_privateuse1_backend_name()])
