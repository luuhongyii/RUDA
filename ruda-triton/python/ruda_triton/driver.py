"""Triton driver for the Ruda backend.

Kernels are PTX produced by Ruda's direct PTX compiler and are loaded and launched through the
CUDA driver API on the device that owns the PyTorch tensors.
"""
from __future__ import annotations

import ctypes
import os
import struct
import sys

from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver

from .compiler import BACKEND

_SCALAR_FORMATS = {"i1": "?", "i8": "b", "i16": "h", "i32": "i", "i64": "q", "u8": "B", "u16": "H", "u32": "I",
                   "u64": "Q", "f16": "e", "f32": "f", "f64": "d"}

CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0
CU_FUNC_ATTRIBUTE_NUM_REGS = 4
CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3
CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8


class CudaApi:
    _lib = None

    def __init__(self):
        if CudaApi._lib is None:
            CudaApi._lib = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
        self.lib = CudaApi._lib

    def check(self, result, what):
        if result != 0:
            name = ctypes.c_char_p()
            self.lib.cuGetErrorName(result, ctypes.byref(name))
            raise RuntimeError(f"{what} failed: {name.value.decode() if name.value else result}")

    def ensure_context(self, device):
        ctx = ctypes.c_void_p()
        self.check(self.lib.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
        if ctx.value:
            return
        self.check(self.lib.cuInit(0), "cuInit")
        dev = ctypes.c_int()
        self.check(self.lib.cuDeviceGet(ctypes.byref(dev), device), "cuDeviceGet")
        self.check(self.lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "cuDevicePrimaryCtxRetain")
        self.check(self.lib.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")


class RudaUtils:

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self):
        self.cuda = CudaApi()

    def load_binary(self, name, kernel, shared, device):
        cuda = self.cuda
        cuda.ensure_context(device)
        image = kernel if isinstance(kernel, bytes) else kernel.encode()
        if not image.endswith(b"\0"):
            image += b"\0"
        module = ctypes.c_void_p()
        log = ctypes.create_string_buffer(8192)
        options = (ctypes.c_int * 2)(5, 6)  # CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
        values = (ctypes.c_void_p * 2)(ctypes.cast(log, ctypes.c_void_p), ctypes.c_void_p(len(log)))
        result = cuda.lib.cuModuleLoadDataEx(ctypes.byref(module), image, 2, options, values)
        if result != 0:
            raise RuntimeError(f"cuModuleLoadDataEx failed for {name} ({result}): {log.value.decode(errors='replace')}")
        function = ctypes.c_void_p()
        cuda.check(cuda.lib.cuModuleGetFunction(ctypes.byref(function), module, name.encode()), "cuModuleGetFunction")
        if shared > 48 * 1024:
            cuda.check(cuda.lib.cuFuncSetAttribute(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared),
                       "cuFuncSetAttribute")
        attr = ctypes.c_int()
        cuda.check(cuda.lib.cuFuncGetAttribute(ctypes.byref(attr), CU_FUNC_ATTRIBUTE_NUM_REGS, function), "attr")
        n_regs = attr.value
        cuda.check(cuda.lib.cuFuncGetAttribute(ctypes.byref(attr), CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function),
                   "attr")
        n_spills = attr.value // 4
        cuda.check(cuda.lib.cuFuncGetAttribute(ctypes.byref(attr), CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function),
                   "attr")
        return module.value, function.value, n_regs, n_spills, attr.value

    def unload_module(self, module):
        if module:
            self.cuda.lib.cuModuleUnload(ctypes.c_void_p(module))

    def get_device_properties(self, device):
        import torch
        props = torch.cuda.get_device_properties(device)
        return {
            "max_shared_mem": getattr(props, "shared_memory_per_block_optin", 48 * 1024),
            "multiprocessor_count": props.multi_processor_count,
            "sm_clock_rate": 0,
            "mem_clock_rate": 0,
            "mem_bus_width": 0,
            "warpSize": 32,
        }


def _pointer(arg) -> int:
    if arg is None:
        return 0
    if isinstance(arg, int):
        return arg
    if hasattr(arg, "data_ptr"):
        return arg.data_ptr()
    raise TypeError(f"cannot pass {type(arg).__name__} as a pointer argument")


class RudaLauncher:

    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        signature = list(src.signature.values())
        if any(isinstance(t, tuple) for t in signature):
            raise NotImplementedError("tuple kernel arguments")
        self.runtime_args = [i for i, t in enumerate(signature) if t != "constexpr"]
        self.arg_kinds = [tuple(k) for k in metadata.ruda_arg_kinds]
        if len(self.runtime_args) != len(self.arg_kinds):
            raise RuntimeError(f"signature has {len(self.runtime_args)} runtime arguments, kernel has "
                               f"{len(self.arg_kinds)}")
        self.threads = metadata.ruda_threads
        self.shared = metadata.shared
        self.info_size = metadata.ruda_info_size
        self.scalar_offsets = metadata.ruda_scalar_offsets
        self.num_ptrs = sum(1 for k, _ in self.arg_kinds if k == "ptr")
        self.cuda = CudaApi()

    def __call__(self, gridX, gridY, gridZ, stream, function, kernel_metadata, launch_metadata, launch_enter_hook,
                 launch_exit_hook, *args):
        if gridX * gridY * gridZ == 0:
            return
        if launch_enter_hook is not None:
            launch_enter_hook(launch_metadata)
        ptrs = []
        info = bytearray(self.info_size)
        scalar = 0
        for index, (kind, ty) in zip(self.runtime_args, self.arg_kinds):
            arg = args[index]
            if kind == "ptr":
                ptrs.append(ctypes.c_uint64(_pointer(arg)))
            else:
                if ty == "bf16":
                    raw = struct.pack("<f", float(arg))[2:]
                    info[self.scalar_offsets[scalar]:self.scalar_offsets[scalar] + 2] = raw
                else:
                    value = bool(arg) if ty == "i1" else (float(arg) if ty.startswith("f") else int(arg))
                    struct.pack_into("<" + _SCALAR_FORMATS[ty], info, self.scalar_offsets[scalar], value)
                scalar += 1
        params = [ctypes.cast(ctypes.byref(p), ctypes.c_void_p) for p in ptrs]
        info_buffer = None
        if self.info_size:
            info_buffer = (ctypes.c_char * self.info_size).from_buffer(info)
            params.append(ctypes.cast(info_buffer, ctypes.c_void_p))
        array = (ctypes.c_void_p * max(1, len(params)))(*params)
        self.cuda.check(
            self.cuda.lib.cuLaunchKernel(ctypes.c_void_p(function), gridX, gridY, gridZ, self.threads, 1, 1,
                                         self.shared, ctypes.c_void_p(stream), array, None), "cuLaunchKernel")
        if launch_exit_hook is not None:
            launch_exit_hook(launch_metadata)


class RudaDriver(GPUDriver):

    def __init__(self):
        self.utils = RudaUtils()
        self.launcher_cls = RudaLauncher
        super().__init__()

    def get_current_target(self):
        major, minor = self.get_device_capability(self.get_current_device())
        return GPUTarget(BACKEND, major * 10 + minor, 32)

    def get_active_torch_device(self):
        import torch
        return torch.device("cuda", self.get_current_device())

    def get_device_interface(self):
        import torch
        return torch.cuda

    @staticmethod
    def is_active():
        if os.environ.get("TRITON_DEFAULT_BACKEND") != BACKEND:
            return False
        torch = sys.modules.get("torch")
        if torch is None:
            import torch
        return torch.cuda.is_available() and torch.version.hip is None

    def map_python_to_cpp_type(self, ty: str) -> str:
        return {"i1": "bool", "i8": "int8_t", "i16": "int16_t", "i32": "int32_t", "i64": "int64_t", "u8": "uint8_t",
                "u16": "uint16_t", "u32": "uint32_t", "u64": "uint64_t", "fp16": "double", "bf16": "double",
                "fp32": "double", "fp64": "double"}.get(ty, "void*")

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        import torch
        return torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int, device="cuda")

    def clear_cache(self, cache):
        cache.zero_()
