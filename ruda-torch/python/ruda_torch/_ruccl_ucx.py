"""ruCCL ring/tree collectives over CUDA-aware UCX device-pointer transfers."""
import ctypes
import os
from pathlib import Path
import socket
import threading
import time

import torch


def local_host_id():
    if 'RUCCL_HOST_ID' in os.environ:
        return os.environ['RUCCL_HOST_ID']
    boot = Path('/proc/sys/kernel/random/boot_id')
    return boot.read_text().strip() if boot.is_file() else socket.gethostname()


class _Transfer(ctypes.Structure):
    _fields_ = [('peer', ctypes.c_uint32), ('receive', ctypes.c_uint32),
                ('tag', ctypes.c_uint64), ('pointer', ctypes.c_uint64), ('bytes', ctypes.c_size_t)]


class UcxTransport:
    def __init__(self, library, device, rank, world_size, gather_metadata, timeout_ms):
        if not 1 <= world_size <= 65535 or not 0 <= rank < world_size:
            raise ValueError('UCX ruCCL needs 1..65535 ranks')
        if not hasattr(library, 'ruccl_ucx_api_version') or library.ruccl_ucx_api_version() != 1:
            raise RuntimeError('cross-node ruCCL requires ruccl-torch-bridge built with --features ucx')
        library.ruccl_error.restype = ctypes.c_char_p
        library.ruccl_ucx_create.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        library.ruccl_ucx_address.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        library.ruccl_ucx_connect.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
        library.ruccl_ucx_exchange.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Transfer), ctypes.c_size_t, ctypes.c_uint64]
        library.ruccl_ucx_reduce.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
                                           ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
        library.ruccl_ucx_destroy.argtypes = [ctypes.c_void_p]
        library.ruccl_ucx_destroy.restype = None
        self.library, self.device, self.rank, self.world_size = library, device, rank, world_size
        self.timeout_ms, self.handle = timeout_ms, ctypes.c_void_p()
        self._lock, self._stats_lock = threading.Lock(), threading.Lock()
        self._sequence = 0
        self._calls = self._bytes = self._reductions = 0
        self._seconds = 0.
        self._marker = torch.empty(1, dtype=torch.uint8, device=f'cuda:{device}')
        self.transports = os.environ.get('RUCCL_UCX_TLS', 'rc,cuda_ipc')
        # No automatic TCP/host-staging transport is configured here.
        error, address = None, None
        try:
            self._check(library.ruccl_ucx_create(world_size, device, self.transports.encode(), ctypes.byref(self.handle)))
            size = ctypes.c_size_t()
            self._check(library.ruccl_ucx_address(self.handle, None, ctypes.byref(size)))
            data = (ctypes.c_uint8 * size.value)()
            self._check(library.ruccl_ucx_address(self.handle, data, ctypes.byref(size)))
            address = bytes(data)
        except Exception as exc:
            error = str(exc)
        try:
            peers = gather_metadata((address, error, self.transports))
            failures = [f'rank {r}: {item[1]}' for r, item in enumerate(peers) if item[1]]
            if failures:
                raise RuntimeError('UCX initialization failed: ' + '; '.join(failures))
            if any(item[2] != self.transports for item in peers):
                raise ValueError('RUCCL_UCX_TLS differs across ranks')
            error = None
            try:
                for peer, (data, _, _) in enumerate(peers):
                    if peer != rank:
                        buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
                        self._check(library.ruccl_ucx_connect(self.handle, peer, buf))
            except Exception as exc:
                error = str(exc)
            failures = [f'rank {r}: {value}' for r, value in enumerate(gather_metadata(error)) if value]
            if failures:
                raise RuntimeError('UCX endpoint setup failed: ' + '; '.join(failures))
        except Exception:
            self.close()
            raise

    def _check(self, status):
        if status:
            raise RuntimeError(self.library.ruccl_error().decode('utf-8', errors='replace'))

    def _tensor(self, tensor):
        if tensor.device.type != 'cuda' or tensor.device.index != self.device or not tensor.is_contiguous():
            raise ValueError('UCX payloads must be contiguous tensors on the rank CUDA device')
        if tensor.is_conj() or tensor.is_neg():
            raise ValueError('UCX payloads cannot contain unresolved view bits')

    def _next(self):
        if self._sequence >= (1 << 31):
            raise OverflowError('UCX collective tag space exhausted; recreate the process group')
        result = self._sequence
        self._sequence += 1
        return result

    @staticmethod
    def _tag(sequence, step, sender):
        return (sequence << 32) | (step << 16) | sender

    def _exchange(self, entries):
        if not self.handle.value:
            raise RuntimeError('UCX communicator is closed')
        # Keep every source/destination view alive until the native requests drain.
        descriptors = (_Transfer * len(entries))(*[
            _Transfer(peer, int(receive), tag, tensor.data_ptr() if tensor.numel() else self._marker.data_ptr(),
                      tensor.numel() * tensor.element_size())
            for peer, receive, tag, tensor in entries])
        started = time.perf_counter()
        self._check(self.library.ruccl_ucx_exchange(self.handle, descriptors, len(entries), self.timeout_ms))
        with self._stats_lock:
            self._bytes += sum(t.numel() * t.element_size() for _, recv, _, t in entries if not recv)
            self._seconds += time.perf_counter() - started

    def _ring_exchange(self, sent, received, sequence, step):
        previous, following = (self.rank - 1) % self.world_size, (self.rank + 1) % self.world_size
        self._exchange([(previous, True, self._tag(sequence, step, previous), received),
                        (following, False, self._tag(sequence, step, self.rank), sent)])

    def _reduce(self, destination, source, operation):
        codes = {torch.float32: 0, torch.float64: 1, torch.int32: 2, torch.int64: 3}
        self._check(self.library.ruccl_ucx_reduce(self.handle, destination.data_ptr(), source.data_ptr(),
                    source.numel(), codes[source.dtype], operation))
        with self._stats_lock:
            self._reductions += int(source.numel() != 0)

    def _scatter_ring(self, chunks, operation):
        sequence = self._next()
        scratch = torch.empty(max((chunk.numel() for chunk in chunks), default=0),
                              dtype=chunks[0].dtype, device=chunks[0].device)
        for step in range(self.world_size - 1):
            send = (self.rank - 1 - step) % self.world_size
            receive = (self.rank - 2 - step) % self.world_size
            target = scratch[:chunks[receive].numel()]
            self._ring_exchange(chunks[send], target, sequence, step)
            self._reduce(chunks[receive], target, operation)

    def _gather_ring(self, chunks):
        sequence = self._next()
        for step in range(self.world_size - 1):
            send = (self.rank - step) % self.world_size
            receive = (self.rank - step - 1) % self.world_size
            self._ring_exchange(chunks[send], chunks[receive], sequence, step)

    @torch.no_grad()
    def collective(self, tensor, operation, *, output=None, root=0):
        self._tensor(tensor)
        if operation not in range(6) or not 0 <= root < self.world_size:
            raise ValueError('invalid UCX collective/root')
        if output is not None:
            self._tensor(output)
            if output.dtype != tensor.dtype:
                raise ValueError('UCX input/output dtypes differ')
        if operation in (4, 5):
            if operation == 5 and tensor.numel() % self.world_size:
                raise ValueError('reduce-scatter input must divide evenly across ranks')
            expected = tensor.numel() * self.world_size if operation == 4 else tensor.numel() // self.world_size
            if output is None or output.numel() != expected:
                raise ValueError('UCX gather/scatter output size differs')
        elif output is not None:
            raise ValueError('UCX all-reduce/broadcast operate in-place')
        reducing = operation in (0, 1, 2, 5)
        if reducing and tensor.dtype not in (torch.float32, torch.float64, torch.int32, torch.int64, torch.float16, torch.bfloat16):
            raise ValueError('UCX reductions support F32/F64/I32/I64 and GPU-promoted F16/BF16')
        with self._lock:
            promote = reducing and tensor.dtype in (torch.float16, torch.bfloat16)
            work = tensor.detach().float() if promote else tensor.detach()
            if operation == 5:
                work = work.clone() if not promote else work
            work = work.reshape(-1)
            destination = tensor if output is None else output
            target = torch.empty_like(output, dtype=work.dtype) if output is not None and promote else output
            torch.cuda.current_stream(self.device).synchronize()
            if operation in (0, 1, 2, 5):
                quotient, remainder = divmod(work.numel(), self.world_size)
                sizes = [quotient + int(r < remainder) for r in range(self.world_size)]
                chunks = list(work.split(sizes))
                self._scatter_ring(chunks, 0 if operation == 5 else operation)
                if operation == 5:
                    target.reshape(-1).copy_(chunks[self.rank])
                else:
                    self._gather_ring(chunks)
            elif operation == 4:
                chunks = list(target.reshape(-1).split([work.numel()] * self.world_size))
                chunks[self.rank].copy_(work)
                torch.cuda.current_stream(self.device).synchronize()
                self._gather_ring(chunks)
            else:
                sequence, step, mask = self._next(), 0, 1
                relative = (self.rank - root) % self.world_size
                while mask < self.world_size:
                    if relative < mask and relative + mask < self.world_size:
                        peer = (relative + mask + root) % self.world_size
                        self._exchange([(peer, False, self._tag(sequence, step, self.rank), work)])
                    elif mask <= relative < 2 * mask:
                        peer = (relative - mask + root) % self.world_size
                        self._exchange([(peer, True, self._tag(sequence, step, peer), work)])
                    mask *= 2
                    step += 1
            source = work.reshape(tensor.shape) if target is None else target
            if source.data_ptr() != destination.data_ptr():
                destination.copy_(source)
            torch.cuda.current_stream(self.device).synchronize()
            torch.autograd.graph.increment_version(destination)
            with self._stats_lock:
                self._calls += 1
            return destination

    @torch.no_grad()
    def all_to_all(self, tensor, output, sends, receives):
        self._tensor(tensor)
        self._tensor(output)
        if tensor.dtype != output.dtype or len(sends) != self.world_size or len(receives) != self.world_size:
            raise ValueError('invalid UCX all-to-all dtype/split count')
        if any(type(n) is not int or n < 0 for n in list(sends) + list(receives)):
            raise ValueError('UCX all-to-all splits must be nonnegative integers')
        if sum(sends) != tensor.numel() or sum(receives) != output.numel() or sends[self.rank] != receives[self.rank]:
            raise ValueError('UCX all-to-all split sizes differ')
        with self._lock:
            if tensor.untyped_storage().data_ptr() == output.untyped_storage().data_ptr():
                tensor = tensor.clone()
            source = list(tensor.reshape(-1).split(list(sends)))
            target = list(output.reshape(-1).split(list(receives)))
            target[self.rank].copy_(source[self.rank])
            torch.cuda.current_stream(self.device).synchronize()
            sequence, entries = self._next(), []
            for peer in range(self.world_size):
                if peer != self.rank:
                    entries.extend([(peer, True, self._tag(sequence, 0, peer), target[peer]),
                                    (peer, False, self._tag(sequence, 0, self.rank), source[peer])])
            self._exchange(entries)
            torch.autograd.graph.increment_version(output)
            with self._stats_lock:
                self._calls += 1
        return output

    def p2p(self, tensor, peer, tag, *, receive=False):
        self._tensor(tensor)
        if not 0 <= peer < self.world_size or peer == self.rank or not 0 <= tag <= 0x7fffffff:
            raise ValueError('invalid UCX point-to-point peer/tag')
        sender, receiver = (peer, self.rank) if receive else (self.rank, peer)
        message = (1 << 63) | (tag << 32) | (sender << 16) | receiver
        torch.cuda.current_stream(self.device).synchronize()
        self._exchange([(peer, receive, message, tensor)])
        if receive:
            torch.autograd.graph.increment_version(tensor)
        return tensor

    def stats(self):
        with self._stats_lock:
            return dict(provider='ruCCL', backend='cuda', transport='UcxDevicePointer', ucx_tls=self.transports,
                        calls=self._calls, device_payload_bytes=self._bytes,
                        reduction_kernel_launches=self._reductions, wall_seconds=self._seconds)

    def close(self):
        if self.handle.value:
            self.library.ruccl_ucx_destroy(self.handle)
            self.handle = ctypes.c_void_p()
