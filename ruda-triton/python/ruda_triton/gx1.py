"""ctypes binding of the GX1 functional model (``gx1_triton`` cdylib from the GX1 monorepo).

``RUDA_TRITON_GX1_LIB`` names the library (``cargo build -p gx1-triton --release``).
"""
from __future__ import annotations

import ctypes
import functools
import json
import os

TARGET = "gx1"


def library_path() -> str:
    path = os.environ.get("RUDA_TRITON_GX1_LIB")
    if not path:
        raise RuntimeError("RUDA_TRITON_DEVICE=gx1 needs RUDA_TRITON_GX1_LIB set to the gx1_triton library")
    return path


@functools.lru_cache(None)
def lib():
    handle = ctypes.CDLL(library_path())
    u64, i64, i32, usize, u32 = ctypes.c_uint64, ctypes.c_int64, ctypes.c_int32, ctypes.c_size_t, ctypes.c_uint32
    buf = ctypes.c_char_p
    for name, res, args in [
        ("gx_last_error", usize, [buf, usize]),
        ("gx_last_stats", usize, [buf, usize]),
        ("gx_compile", i64, [buf, usize]),
        ("gx_kernel_info", usize, [i64, buf, usize]),
        ("gx_alloc", u64, [u64]),
        ("gx_free", i32, [u64]),
        ("gx_write", i32, [u64, ctypes.c_void_p, u64]),
        ("gx_read", i32, [u64, ctypes.c_void_p, u64]),
        ("gx_launch", i32, [i64, u32, u32, u32, ctypes.POINTER(u64), usize, ctypes.POINTER(u64), usize]),
    ]:
        fn = getattr(handle, name)
        fn.restype, fn.argtypes = res, args
    return handle


def _text(fn, *args) -> str:
    size = fn(*args, None, 0)
    out = ctypes.create_string_buffer(size + 1)
    fn(*args, out, size + 1)
    return out.value.decode(errors="replace")


class GxError(RuntimeError):
    pass


def _check(ok: bool, what: str):
    if not ok:
        raise GxError(f"{what}: {_text(lib().gx_last_error)}")


def compile_kernel(simt_json: str) -> tuple[int, dict]:
    data = simt_json.encode()
    handle = lib().gx_compile(data, len(data))
    _check(handle >= 0, "GX1 compile")
    return handle, json.loads(_text(lib().gx_kernel_info, handle))


def last_stats() -> str:
    return _text(lib().gx_last_stats)


def alloc(nbytes: int) -> int:
    address = lib().gx_alloc(nbytes)
    _check(address != 0, "GX1 alloc")
    return address


def free(address: int):
    _check(lib().gx_free(address) == 0, "GX1 free")


def write(address: int, host_ptr: int, nbytes: int):
    _check(lib().gx_write(address, host_ptr, nbytes) == 0, "GX1 write")


def read(address: int, host_ptr: int, nbytes: int):
    _check(lib().gx_read(address, host_ptr, nbytes) == 0, "GX1 read")


def launch(handle: int, grid: tuple[int, int, int], pointers: list[int], scalars: list[int]):
    ptrs = (ctypes.c_uint64 * max(len(pointers), 1))(*pointers)
    vals = (ctypes.c_uint64 * max(len(scalars), 1))(*scalars)
    _check(lib().gx_launch(handle, *grid, ptrs, len(pointers), vals, len(scalars)) == 0, "GX1 launch")
