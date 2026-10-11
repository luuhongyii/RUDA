"""Triton driver for the Ruda backend.

Kernels are PTX produced by Ruda's direct PTX compiler, loaded through the CUDA driver API and
launched by a small compiled C launcher.

Two device modes:

* ``cuda`` (default): tensors are ``torch.cuda`` tensors and kernels run on PyTorch's current
  CUDA stream.
* ``ruda`` (``RUDA_TRITON_DEVICE=ruda``): tensors live on the RUDA PrivateUse1 device
  (``ruda_torch``). Kernels run on the native stream of RUDA's current stream; RUDA work
  queued on that stream is flushed first, so Triton kernels are ordered with RUDA kernels.
* ``gx1`` (``RUDA_TRITON_DEVICE=gx1``): kernels are compiled to GX-ISA and run on the GX1
  functional model; tensors are CPU tensors, copied to the model around every launch.
"""
from __future__ import annotations

import ctypes
import functools
import os
import struct
import sys
from pathlib import Path

from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver

BACKEND = "ruda"

_SCALAR_CODES = {"i1": "?", "i8": "b", "i16": "h", "i32": "i", "i64": "q", "u1": "?", "u8": "B", "u16": "H",
                 "u32": "I", "u64": "Q", "f16": "e", "bf16": "E", "f32": "f", "f64": "d"}

CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0
CU_FUNC_ATTRIBUTE_NUM_REGS = 4
CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3
CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8


def device_mode() -> str:
    mode = os.environ.get("RUDA_TRITON_DEVICE", "cuda")
    if mode not in ("cuda", "ruda", "gx1"):
        raise RuntimeError(f"RUDA_TRITON_DEVICE must be 'cuda', 'ruda' or 'gx1', not {mode!r}")
    return mode


@functools.lru_cache(None)
def _ruda_torch():
    import ruda_torch
    return ruda_torch


def ruda_native_stream() -> int:
    """`CUstream` of RUDA's current stream, after flushing RUDA work queued on it."""
    c = _ruda_torch()._C
    return c.stream_command(13, c.stream_command(0))


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

    @property
    def launch_kernel_address(self) -> int:
        return ctypes.cast(self.lib.cuLaunchKernel, ctypes.c_void_p).value


@functools.lru_cache(None)
def _launcher_module():
    from triton.runtime.build import compile_module_from_src
    src = (Path(__file__).parent / "launcher.c").read_text()
    return compile_module_from_src(src, "ruda_launcher")


class RudaUtils:

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self):
        self.cuda = CudaApi()

    def load_binary(self, name, kernel, shared, device):
        cuda = self.cuda
        if device_mode() == "ruda":
            _ruda_torch()
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
        cuda = self.cuda
        cuda.ensure_context(device)
        dev = ctypes.c_int()
        cuda.check(cuda.lib.cuDeviceGet(ctypes.byref(dev), device), "cuDeviceGet")

        def attribute(code):
            value = ctypes.c_int()
            cuda.check(cuda.lib.cuDeviceGetAttribute(ctypes.byref(value), code, dev), "cuDeviceGetAttribute")
            return value.value

        return {
            "max_shared_mem": attribute(97),  # MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
            "multiprocessor_count": attribute(16),
            "sm_clock_rate": attribute(13),
            "mem_clock_rate": attribute(36),
            "mem_bus_width": attribute(37),
            "warpSize": 32,
        }


class RudaLauncher:

    def __init__(self, src, metadata):
        signature = list(src.signature.values())
        if any(isinstance(t, tuple) for t in signature):
            raise NotImplementedError("tuple kernel arguments")
        kinds = [tuple(k) for k in metadata.ruda_arg_kinds]
        runtime = [t for t in signature if t != "constexpr"]
        if len(runtime) != len(kinds):
            raise RuntimeError(f"signature has {len(runtime)} runtime arguments, kernel has {len(kinds)}")
        codes = iter("p" if kind == "ptr" else _SCALAR_CODES[ty] for kind, ty in kinds)
        self.spec = "".join("c" if t == "constexpr" else next(codes) for t in signature).encode()
        self.offsets = struct.pack(f"<{len(metadata.ruda_scalar_offsets)}i", *metadata.ruda_scalar_offsets)
        self.threads = metadata.ruda_threads
        self.shared = metadata.shared
        self.info_size = metadata.ruda_info_size
        self.launch_kernel = CudaApi().launch_kernel_address
        self.launch = _launcher_module().launch

    def __call__(self, gridX, gridY, gridZ, stream, function, kernel_metadata, launch_metadata, launch_enter_hook,
                 launch_exit_hook, *args):
        if launch_enter_hook is not None:
            launch_enter_hook(launch_metadata)
        self.launch(self.launch_kernel, function, gridX, gridY, gridZ, self.threads, self.shared, stream, self.spec,
                    self.offsets, self.info_size, args)
        if launch_exit_hook is not None:
            launch_exit_hook(launch_metadata)


def _scalar_bits(value, ty: str) -> int:
    if ty in ("f32", "fp32"):
        return struct.unpack("<I", struct.pack("<f", value))[0]
    if ty in ("f16", "fp16"):
        return struct.unpack("<H", struct.pack("<e", value))[0]
    if ty in ("bf16", ):
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) & 0xffff
    if ty in ("f64", "fp64"):
        raise NotImplementedError("GX-ISA gap: f64 kernel arguments")
    return int(value) & 0xFFFF_FFFF_FFFF_FFFF


class GxUtils:

    def load_binary(self, name, kernel, shared, device):
        from . import gx1
        source = kernel.decode() if isinstance(kernel, bytes) else kernel
        handle, info = gx1.compile_kernel(source)
        return None, handle, info["registers"], info["spilled_values"], 1024

    def unload_module(self, module):
        pass

    def get_device_properties(self, device):
        return {"max_shared_mem": 128 * 1024, "multiprocessor_count": 1, "sm_clock_rate": 0, "mem_clock_rate": 0,
                "mem_bus_width": 0, "warpSize": 32}


class GxLauncher:
    """Copies the storage of every tensor argument to the functional model, launches, and copies
    the storages back."""

    def __init__(self, src, metadata):
        signature = list(src.signature.values())
        kinds = iter(tuple(k) for k in metadata.ruda_arg_kinds)
        self.plan = [None if t == "constexpr" else next(kinds) for t in signature]

    def __call__(self, gridX, gridY, gridZ, stream, function, kernel_metadata, launch_metadata, launch_enter_hook,
                 launch_exit_hook, *args):
        from . import gx1
        if launch_enter_hook is not None:
            launch_enter_hook(launch_metadata)
        storages, pointers, scalars = {}, [], []
        try:
            for plan, arg in zip(self.plan, args):
                if plan is None:
                    continue
                kind, ty = plan
                if kind != "ptr":
                    scalars.append(_scalar_bits(arg, ty))
                    continue
                storage = arg.untyped_storage()
                base, nbytes = storage.data_ptr(), storage.nbytes()
                if base not in storages:
                    address = gx1.alloc(nbytes)
                    storages[base] = (address, nbytes)
                    gx1.write(address, base, nbytes)
                pointers.append(storages[base][0] + arg.data_ptr() - base)
            gx1.launch(function, (gridX, gridY, gridZ), pointers, scalars)
            for base, (address, nbytes) in storages.items():
                gx1.read(address, base, nbytes)
        finally:
            for address, _ in storages.values():
                gx1.free(address)
        if launch_exit_hook is not None:
            launch_exit_hook(launch_metadata)


class _GxDeviceInterface:

    @staticmethod
    def synchronize(*args, **kwargs):
        pass

    @staticmethod
    def current_device():
        return 0

    @staticmethod
    def device_count():
        return 1


class RudaDriver(GPUDriver):

    def __init__(self):
        if device_mode() == "gx1":
            self.utils = GxUtils()
            self.launcher_cls = GxLauncher
            self.get_current_device = lambda: 0
            self.set_current_device = lambda device: None
            self.get_current_stream = lambda device=None: 0
            self.get_device_capability = lambda device=0: (0, 0)
            return
        self.utils = RudaUtils()
        self.launcher_cls = RudaLauncher
        super().__init__()
        if device_mode() == "ruda":
            self.get_current_device = lambda: 0
            self.set_current_device = lambda device: None
            self.get_current_stream = lambda device=None: ruda_native_stream()
            self.get_device_capability = self._capability

    def _capability(self, device=0):
        cuda = self.utils.cuda
        cuda.ensure_context(device)
        dev, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        cuda.check(cuda.lib.cuDeviceGet(ctypes.byref(dev), device), "cuDeviceGet")
        cuda.check(cuda.lib.cuDeviceGetAttribute(ctypes.byref(major), 75, dev), "cuDeviceGetAttribute")
        cuda.check(cuda.lib.cuDeviceGetAttribute(ctypes.byref(minor), 76, dev), "cuDeviceGetAttribute")
        return major.value, minor.value

    def get_current_target(self):
        if device_mode() == "gx1":
            return GPUTarget(BACKEND, "gx1", 32)
        major, minor = self.get_device_capability(self.get_current_device())
        return GPUTarget(BACKEND, major * 10 + minor, 32)

    def get_active_torch_device(self):
        import torch
        if device_mode() == "gx1":
            return torch.device("cpu")
        if device_mode() == "ruda":
            _ruda_torch()
            return torch.device("ruda", 0)
        return torch.device("cuda", self.get_current_device())

    def get_device_interface(self):
        if device_mode() == "gx1":
            return _GxDeviceInterface
        if device_mode() == "ruda":
            return _ruda_torch()
        import torch
        return torch.cuda

    @staticmethod
    def is_active():
        if os.environ.get("TRITON_DEFAULT_BACKEND") != BACKEND:
            return False
        if device_mode() in ("ruda", "gx1"):
            return True
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
        return torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int, device=self.get_active_torch_device())

    def clear_cache(self, cache):
        cache.zero_()
