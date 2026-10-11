"""Single-host ruCCL GPU payload transport; Gloo carries control metadata only."""
import ctypes
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.distributed as dist


class RucclTransport:
    def __init__(self, group, device, *, algorithm='auto', backend=None):
        if algorithm not in ('auto', 'ring'):
            raise ValueError('ruCCL algorithm must be auto or ring')
        self.library = ctypes.CDLL(os.environ['RUCCL_TORCH_LIBRARY'])
        library = self.library
        self.backend = backend or ('rocm' if torch.version.hip else 'cuda')
        if self.backend not in ('cuda', 'rocm', 'metal'):
            raise ValueError('PyTorch interop supports cuda/rocm/metal; Vulkan uses ruccl_bridge GPU buffers')
        library.ruccl_error.restype = ctypes.c_char_p
        library.ruccl_server.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        if self.backend == 'cuda':
            library.ruccl_create.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32,
                                           ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
            library.ruccl_collective.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64,
                                               ctypes.c_size_t, ctypes.c_uint64, ctypes.c_uint32]
            library.ruccl_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
            library.ruccl_destroy.argtypes = [ctypes.c_void_p]
        self.handle = ctypes.c_void_p()
        self.rank, self.world_size = dist.get_rank(group), dist.get_world_size(group)
        self.device, self.wall_seconds = device, 0.0
        self._tensor = None
        self.interop_seconds = 0.0
        self._executor = None
        self._lock = threading.Lock()
        metadata = None
        if self.rank == 0:
            uid = (ctypes.c_uint8*16)()
            port = ctypes.c_uint32()
            self._check(library.ruccl_server(self.world_size, uid, ctypes.byref(port)))
            metadata = (bytes(uid), port.value)
        gathered = [None]*self.world_size
        dist.all_gather_object(gathered, metadata, group=group)
        uid_bytes, port = gathered[0]
        uid = (ctypes.c_uint8*16).from_buffer_copy(uid_bytes)
        if self.backend == 'cuda':
            self._check(library.ruccl_create(self.rank, self.world_size, device, port, uid,
                                           int(algorithm == 'ring'), ctypes.byref(self.handle)))
        else:
            from ruccl_bridge import Communicator
            self._tensor = Communicator(self.rank, self.world_size, (uid_bytes, port), backend=self.backend,
                device=device, algorithm=algorithm)

    def _synchronize(self):
        if self.backend == 'metal':
            torch.mps.synchronize()
        else:
            torch.cuda.synchronize(self.device)

    def _check(self, status):
        if status:
            raise RuntimeError(self.library.ruccl_error().decode('utf-8', errors='replace'))

    def collective(self, tensor, operation, *, output=None, root=0):
        with self._lock:
            return self._collective(tensor, operation, output=output, root=root)

    def _collective(self, tensor, operation, *, output=None, root=0):
        if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError('ruCCL bridge supports floating tensor payloads only')
        kind = 'mps' if self.backend == 'metal' else 'cuda'
        if tensor.device.type != kind or tensor.device.index not in (None, self.device) or not tensor.is_contiguous():
            raise ValueError('ruCCL requires contiguous tensors on the selected rank GPU')
        if output is not None and (output.device != tensor.device or not output.is_contiguous()):
            raise ValueError('ruCCL output device/layout mismatch')
        if self._tensor is not None:
            started = time.perf_counter()
            host = tensor.detach().float().cpu().contiguous()
            buffer = self._tensor.upload_address(host.data_ptr(), host.shape)
            uploaded = time.perf_counter()-started
            try:
                result = self._tensor.collective(buffer, operation, root=root)
                try:
                    started = time.perf_counter()
                    destination = tensor if output is None else output
                    target = torch.empty(destination.shape, dtype=torch.float32, device='cpu')
                    self._tensor.download_address(result, target.data_ptr())
                    destination.copy_(target.to(device=destination.device, dtype=destination.dtype))
                    self._synchronize()
                    self.interop_seconds += uploaded + time.perf_counter()-started
                    return destination
                finally:
                    if result is not buffer: result.close()
            finally:
                buffer.close()
        work = tensor.detach().float().contiguous()
        target = None if output is None else torch.empty_like(output, dtype=torch.float32)
        self._synchronize()
        started = time.perf_counter()
        self._check(self.library.ruccl_collective(self.handle, operation, work.data_ptr(), work.numel(),
                                                0 if target is None else target.data_ptr(), root))
        self.wall_seconds += time.perf_counter()-started
        if target is None:
            if work.data_ptr() != tensor.data_ptr():
                tensor.copy_(work)
            torch.autograd.graph.increment_version(tensor)
            return tensor
        output.copy_(target)
        return output

    def sum_async(self, tensor):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ruccl-ordered')
        # Fence the producing stream before another host thread reads this tensor.
        self._synchronize()
        future = self._executor.submit(self.collective, tensor, 0)
        class Work:
            def wait(self):
                future.result()
                return True
        return Work()

    def stats(self):
        if self._tensor is not None:
            with self._lock:
                result = self._tensor.stats()
                result['interop_seconds'] = self.interop_seconds
                return result
        values = (ctypes.c_uint64*8)()
        with self._lock:
            self._check(self.library.ruccl_stats(self.handle, values))
        result = dict(zip(('calls', 'protocol_bytes', 'reduction_kernel_launches', 'dtoh_bytes',
                          'htod_bytes', 'allocated_bytes', 'dtoh_ns', 'htod_ns'), values))
        result.update(wall_seconds=self.wall_seconds, provider='ruCCL', backend=self.backend, transport='TcpHostStaged')
        return result

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        if self._tensor is not None:
            self._tensor.close()
            self._tensor = None
        if self.handle.value:
            self.library.ruccl_destroy(self.handle)
            self.handle = ctypes.c_void_p()
