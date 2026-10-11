from __future__ import annotations

import functools
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Tuple

from triton._C.libtriton import ir, passes
from triton.backends.compiler import BaseBackend, GPUTarget, Language

from .lower import lower

BACKEND = "ruda"
PTX_VERSION = (8, 0)


def compiler_binary() -> str:
    explicit = os.environ.get("RUDA_TRITON_PTX")
    if explicit:
        return explicit
    bundled = Path(__file__).parent / "bin" / ("ruda-triton-ptx" + (".exe" if os.name == "nt" else ""))
    if bundled.exists():
        return str(bundled)
    found = shutil.which("ruda-triton-ptx")
    if found:
        return found
    raise RuntimeError("ruda-triton-ptx not found: build `cargo build -p ruda-triton --release` "
                       "and set RUDA_TRITON_PTX to the binary")


@functools.lru_cache(None)
def _file_hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class RudaOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 1
    warp_size: int = 32
    maxnreg: Optional[int] = None
    ptx_version: Optional[int] = None
    ir_override: Optional[str] = None
    enable_fp_fusion: bool = True
    launch_cooperative_grid: bool = False
    launch_pdl: bool = False
    supported_fp8_dtypes: Tuple[str, ...] = ()
    deprecated_fp8_dot_operand_dtypes: Tuple[str, ...] = ()
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str, ...] = ("ieee", "tf32")
    max_num_imprecise_acc_default: int = 0
    extern_libs: Optional[tuple] = None
    debug: bool = False
    backend_name: str = BACKEND
    sanitize_overflow: bool = False
    arch: Optional[str] = None
    instrumentation_mode: str = ""

    def __post_init__(self):
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, "num_warps must be a power of 2"
        assert self.num_warps * self.warp_size <= 1024, "a program is one CTA of at most 1024 threads"

    def hash(self):
        key = "_".join(f"{k}-{v}" for k, v in sorted(self.__dict__.items()))
        return hashlib.sha256(key.encode()).hexdigest()


def takes_over_cuda() -> bool:
    """With TRITON_DEFAULT_BACKEND=ruda, kernels compiled for the "cuda" target (e.g. by Inductor) go through Ruda."""
    return os.environ.get("TRITON_DEFAULT_BACKEND") == BACKEND


def _yield_cuda_target():
    try:
        from triton.backends.nvidia.compiler import CUDABackend
    except ImportError:
        return
    original = CUDABackend.supports_target
    if getattr(original, "_ruda_wrapped", False):
        return

    def supports_target(target: GPUTarget):
        return not (target.backend == "cuda" and takes_over_cuda()) and original(target)

    supports_target._ruda_wrapped = True
    CUDABackend.supports_target = staticmethod(supports_target)


_yield_cuda_target()


class RudaBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == BACKEND or (target.backend == "cuda" and takes_over_cuda())

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "ptx"

    def get_target_name(self, options) -> str:
        return f"{BACKEND}:{self.target.arch}"

    def parse_options(self, opts) -> Any:
        args = {"arch": f"sm{self.target.arch}"}
        args.update({k: opts[k] for k in RudaOptions.__dataclass_fields__ if k in opts and opts[k] is not None})
        args["debug"] = False
        args["sanitize_overflow"] = False
        args.pop("supported_fp8_dtypes", None)
        if "enable_fp_fusion" not in args:
            from triton import knobs
            args["enable_fp_fusion"] = knobs.language.default_fp_fusion
        return RudaOptions(**args)

    def pack_metadata(self, metadata):
        return (metadata.num_warps, metadata.num_ctas, metadata.shared)

    def get_codegen_implementation(self, options):
        import triton.language.extra.cuda as cuda
        return {"convert_custom_types": cuda.convert_custom_float8_sm80, "min_dot_size": lambda a, b: (1, 1, 1)}

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.cuda import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        pass

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, "make_ttir")
        return mod

    @staticmethod
    def make_simt(mod, metadata, opt):
        kernel, arg_kinds = lower(mod.str_nodebug(), opt.num_warps, opt.enable_fp_fusion)
        metadata["name"] = kernel["name"]
        metadata["ruda_arg_kinds"] = arg_kinds
        return json.dumps(kernel)

    def make_ptx(self, src, metadata, opt):
        kernel = json.loads(src)
        kernel["sm"] = self.target.arch
        kernel["ptx_version"] = list(PTX_VERSION)
        proc = subprocess.run([compiler_binary()], input=json.dumps(kernel).encode(), capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ruda-triton-ptx failed for {kernel['name']}:\n{proc.stderr.decode(errors='replace')}")
        compiled = json.loads(proc.stdout)
        metadata["shared"] = compiled["shared"]
        metadata["ruda_threads"] = compiled["threads"]
        metadata["ruda_info_size"] = compiled["info_size"]
        metadata["ruda_scalar_offsets"] = compiled["scalar_offsets"]
        metadata["global_scratch_size"] = 0
        metadata["global_scratch_align"] = 1
        metadata["profile_scratch_size"] = 0
        metadata["profile_scratch_align"] = 1
        return compiled["ptx"]

    def add_stages(self, stages, options, language=Language.TRITON):
        if language != Language.TRITON:
            raise NotImplementedError("the Ruda backend compiles Triton (not Gluon) kernels")
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["simt"] = lambda src, metadata: self.make_simt(src, metadata, options)
        stages["ptx"] = lambda src, metadata: self.make_ptx(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        from . import __version__
        here = Path(__file__).parent
        sources = "".join(_file_hash(str(here / f)) for f in ("lower.py", "ttir.py", "compiler.py"))
        return f"ruda-{__version__}-{_file_hash(compiler_binary())}-{hashlib.sha256(sources.encode()).hexdigest()}-{self.target.arch}"
