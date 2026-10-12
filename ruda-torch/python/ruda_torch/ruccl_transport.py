"""Single-host ruCCL GPU payload transport; Gloo carries control metadata only."""
import ctypes
import os
import time
import threading
import sys
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.distributed as dist


class RucclTransport:
    def __init__(self, group, device, *, algorithm='auto', backend=None,
                 rank=None, world_size=None, rendezvous=None, cuda_transport=None, timeout_ms=300000):
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
        self.rank = dist.get_rank(group) if rank is None else rank
        self.world_size = dist.get_world_size(group) if world_size is None else world_size
        self.device, self.wall_seconds = device, 0.0
        self._tensor = None
        self.interop_seconds = 0.0
        self._executor = None
        self._lock = threading.Lock()
        self.ipc = False
        if self.backend == 'cuda':
            path = cuda_transport or os.environ.get('RUCCL_CUDA_TRANSPORT', 'auto')
            if path not in ('auto', 'ipc', 'host'):
                raise ValueError('RUCCL_CUDA_TRANSPORT must be auto, ipc or host')
            self.ipc = path == 'ipc' or (path == 'auto' and sys.platform.startswith('linux'))
            if self.ipc:
                if not hasattr(library, 'ruccl_ipc_api_version') or library.ruccl_ipc_api_version() != 1:
                    raise RuntimeError('rebuild ruCCL bridge with CUDA IPC API 1')
                library.ruccl_ipc_create.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int32,
                    ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_void_p)]
                library.ruccl_ipc_collective.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64,
                    ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint32]
                library.ruccl_ipc_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
                library.ruccl_ipc_destroy.argtypes = [ctypes.c_void_p]
                if hasattr(library, 'ruccl_ipc_all_to_all'):
                    library.ruccl_ipc_all_to_all.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                        ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
                if hasattr(library, 'ruccl_ipc_p2p'):
                    library.ruccl_ipc_p2p.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_uint32]
        metadata = None
        if rendezvous is None and self.rank == 0:
            uid = (ctypes.c_uint8*16)()
            port = ctypes.c_uint32()
            self._check(library.ruccl_server(self.world_size, uid, ctypes.byref(port)))
            metadata = (bytes(uid), port.value)
        if rendezvous is None:
            gathered = [None]*self.world_size
            dist.all_gather_object(gathered, metadata, group=group)
            rendezvous = gathered[0]
        uid_bytes, port = rendezvous
        uid = (ctypes.c_uint8*16).from_buffer_copy(uid_bytes)
        if self.backend == 'cuda':
            if self.ipc:
                self._check(library.ruccl_ipc_create(self.rank, self.world_size, device, port, uid,
                    timeout_ms, ctypes.byref(self.handle)))
            else:
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
        if self.ipc:
            return self._ipc_collective(tensor, operation, output=output, root=root)
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

    def _ipc_collective(self, tensor, operation, *, output=None, root=0):
        if tensor.device.type != 'cuda' or tensor.device.index != self.device or not tensor.is_contiguous():
            raise ValueError('CUDA IPC requires contiguous tensors on the rank GPU')
        if output is not None and (output.dtype != tensor.dtype or output.device != tensor.device or not output.is_contiguous()):
            raise ValueError('CUDA IPC output dtype/device/layout mismatch')
        reducing = operation in (0, 1, 2, 5)
        convert = reducing and tensor.dtype in (torch.float16, torch.bfloat16)
        work = tensor.detach().float() if convert else tensor.detach()
        codes = {torch.float32:0, torch.float64:1, torch.int32:2, torch.int64:3,
                 torch.uint8:4, torch.int8:4, torch.bool:4,
                 torch.int16:5, torch.float16:5, torch.bfloat16:5}
        if work.dtype not in codes or (reducing and codes[work.dtype] > 3):
            raise ValueError('CUDA IPC reduction supports F32/F64/I32/I64 and GPU-promoted F16/BF16')
        expected = work.numel()*self.world_size if operation == 4 else work.numel()//self.world_size if operation == 5 else work.numel()
        if operation in (4, 5) and (output is None or output.numel() != expected):
            raise ValueError('CUDA IPC gather/scatter output size mismatch')
        target = None if output is None else torch.empty_like(output, dtype=work.dtype) if convert else output
        torch.cuda.current_stream(self.device).synchronize()
        started = time.perf_counter()
        self._check(self.library.ruccl_ipc_collective(self.handle, operation, work.data_ptr(), work.numel(),
            codes[work.dtype], 0 if target is None else target.data_ptr(), root))
        self.wall_seconds += time.perf_counter()-started
        destination = tensor if output is None else output
        source = work if target is None else target
        if source.data_ptr() != destination.data_ptr():
            destination.copy_(source)
            torch.cuda.current_stream(self.device).synchronize()
        torch.autograd.graph.increment_version(destination)
        return destination

    @staticmethod
    def _copy_dtype(tensor):
        codes = {torch.float32:0, torch.float64:1, torch.int32:2, torch.int64:3,
            torch.uint8:4, torch.int8:4, torch.bool:4, torch.int16:5, torch.float16:5, torch.bfloat16:5}
        if tensor.dtype not in codes:
            raise ValueError('CUDA IPC copy requires a supported real/integer/bool dtype')
        return codes[tensor.dtype]

    def all_to_all(self, tensor, output, sends, receives):
        if not self.ipc or not hasattr(self.library, 'ruccl_ipc_all_to_all'):
            raise RuntimeError('rebuild ruCCL bridge with CUDA IPC AllToAll')
        if (tensor.device != output.device or tensor.dtype != output.dtype or tensor.device.type != 'cuda'
            or tensor.device.index != self.device or not tensor.is_contiguous() or not output.is_contiguous()):
            raise ValueError('all-to-all requires contiguous same-dtype rank GPU tensors')
        if len(sends) != self.world_size or len(receives) != self.world_size or any(n < 0 for n in sends+receives):
            raise ValueError('all-to-all splits require one nonnegative count per rank')
        if sum(sends) != tensor.numel() or sum(receives) != output.numel():
            raise ValueError('all-to-all split counts do not cover input/output')
        send_counts = (ctypes.c_size_t*self.world_size)(*sends)
        receive_counts = (ctypes.c_size_t*self.world_size)(*receives)
        torch.cuda.current_stream(self.device).synchronize()
        with self._lock:
            self._check(self.library.ruccl_ipc_all_to_all(self.handle, tensor.data_ptr(), output.data_ptr(),
                self._copy_dtype(tensor), send_counts, receive_counts))
        torch.autograd.graph.increment_version(output)
        return output

    def p2p(self, tensor, peer, tag, *, receive=False):
        if not self.ipc or not hasattr(self.library, 'ruccl_ipc_p2p'):
            raise RuntimeError('rebuild ruCCL bridge with CUDA IPC P2P')
        if tensor.device.type != 'cuda' or tensor.device.index != self.device or not tensor.is_contiguous():
            raise ValueError('point-to-point requires a contiguous rank GPU tensor')
        if not 0 <= peer < self.world_size or peer == self.rank or not 0 <= tag <= 0x7fffffff:
            raise ValueError('invalid point-to-point peer/tag')
        torch.cuda.current_stream(self.device).synchronize()
        self._check(self.library.ruccl_ipc_p2p(self.handle, int(receive), peer, tag,
            tensor.data_ptr(), tensor.numel(), self._copy_dtype(tensor)))
        if receive:
            torch.autograd.graph.increment_version(tensor)
        return tensor

    def sum_async(self, tensor):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ruccl-ordered')
        if self.ipc:
            event = torch.cuda.current_stream(self.device).record_event()
            if not hasattr(self, '_async_stream'):
                self._async_stream = torch.cuda.Stream(device=self.device)
            def run():
                with torch.cuda.device(self.device), torch.no_grad(), torch.cuda.stream(self._async_stream):
                    event.synchronize()
                    result = self.collective(tensor, 0)
                    self._async_stream.synchronize()
                    return result
            future = self._executor.submit(run)
        else:
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
            self._check((self.library.ruccl_ipc_stats if self.ipc else self.library.ruccl_stats)(self.handle, values))
        result = dict(zip(('calls', 'protocol_bytes', 'reduction_kernel_launches', 'dtoh_bytes',
                          'htod_bytes', 'allocated_bytes', 'dtoh_ns', 'htod_ns'), values))
        if self.ipc:
            result['peer_bytes'] = result.pop('protocol_bytes')
        result.update(wall_seconds=self.wall_seconds, provider='ruCCL', backend=self.backend,
                      transport='CudaIpcRing' if self.ipc else 'TcpHostStaged')
        return result

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        if self._tensor is not None:
            self._tensor.close()
            self._tensor = None
        if self.handle.value:
            (self.library.ruccl_ipc_destroy if self.ipc else self.library.ruccl_destroy)(self.handle)
            self.handle = ctypes.c_void_p()
