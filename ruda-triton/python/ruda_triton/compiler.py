from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
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


_BLOCKED = re.compile(r"^(#\w+) = #ttg\.blocked<\{sizePerThread = \[([\d, ]*)\], threadsPerWarp = \[([\d, ]*)\], "
                      r"warpsPerCTA = \[([\d, ]*)\]", re.M)
_SLICE = re.compile(r"#ttg\.slice<\{dim = (\d+), parent = (#\w+)\}>")


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def official_reduce_layouts(ttir: str, num_warps: int, num_stages: int, arch: int) -> Optional[list]:
    """Blocked layout (sizePerThread, threadsPerWarp, warpsPerCTA) that the official NVIDIA backend
    assigns to the first operand of every ``tt.reduce``, in program order; None entries where the
    layout is not a blocked or sliced blocked layout. Reproducing the official reduction order
    from these layouts makes reductions bit-identical to official Triton."""
    try:
        from triton.backends.nvidia.compiler import CUDABackend
    except ImportError:
        return None
    global _PROBING_OFFICIAL
    _PROBING_OFFICIAL = True
    try:
        backend = CUDABackend(GPUTarget("cuda", arch, 32))
    finally:
        _PROBING_OFFICIAL = False
    options = backend.parse_options({"num_warps": num_warps, "num_stages": num_stages})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    fd, path = tempfile.mkstemp(suffix=".ttir")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(ttir)
        module = ir.parse_mlir_module(path, context)
        module.context = context
        CUDABackend.make_ttgir(module, {}, options, arch)
        ttgir = module.str_nodebug()
    except Exception:
        return None
    finally:
        os.unlink(path)
    blocked = {m.group(1): [_ints(m.group(i)) for i in (2, 3, 4)] for m in _BLOCKED.finditer(ttgir)}
    layouts, stack = [], []
    for line in ttgir.splitlines():
        line = line.strip()
        start = re.search(r'"(tt\.\w+)"\(.*\(\{$', line)
        if start:
            stack.append(start.group(1))
            continue
        if line.startswith("})") and stack:
            if stack.pop() != "tt.reduce":
                continue
            operand = re.search(r"\}\) : \((tensor<([\dx]+)x\w+, (.*?)>)[,)]", line)
            layout = None
            if operand:
                encoding = operand.group(3)
                sliced = _SLICE.fullmatch(encoding)
                if encoding in blocked:
                    layout = blocked[encoding]
                elif sliced and sliced.group(2) in blocked:
                    dim = int(sliced.group(1))
                    layout = [[v for i, v in enumerate(part) if i != dim] for part in blocked[sliced.group(2)]]
                if layout is not None:
                    layout = {"shape": _ints(operand.group(2).replace("x", ",")), "spt": layout[0],
                              "tpw": layout[1], "wpc": layout[2]}
            layouts.append(layout)
    return layouts


_MLIR_TYPES = {"f32": "f32", "f64": "f64", "i32": "i32", "u32": "i32", "i64": "i64", "u64": "i64"}


@functools.lru_cache(None)
def libdevice_functions(externs: tuple, arch: int, libdevice_path: str) -> str:
    """PTX ``.func`` definitions for ``(wrapper, symbol, ret, args)`` entries, compiled from libdevice
    by Triton's own LLVM with the official options (nvvm-reflect ftz, O3, fp fusion)."""
    from triton._C.libtriton import llvm, nvidia
    from triton.backends.nvidia.compiler import get_features, sm_arch_from_capability

    lines = ["module {"]
    for wrapper, symbol, ret, args in externs:
        params = ", ".join(f"%a{i}: {_MLIR_TYPES[t]}" for i, t in enumerate(args))
        arg_tys = ", ".join(_MLIR_TYPES[t] for t in args)
        lines.append(f"  llvm.func @{symbol}({arg_tys}) -> {_MLIR_TYPES[ret]}")
        lines.append(f"  llvm.func @{wrapper}({params}) -> {_MLIR_TYPES[ret]} {{")
        lines.append(f"    %r = llvm.call @{symbol}({', '.join(f'%a{i}' for i in range(len(args)))}) : "
                     f"({arg_tys}) -> {_MLIR_TYPES[ret]}")
        lines.append(f"    llvm.return %r : {_MLIR_TYPES[ret]}")
        lines.append("  }")
    lines.append("}")
    context = ir.context()
    ir.load_dialects(context)
    fd, path = tempfile.mkstemp(suffix=".mlir")
    try:
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(lines))
        module = ir.parse_mlir_module(path, context)
    finally:
        os.unlink(path)
    llvm.init_targets()
    llvm_context = llvm.context()
    llvm_mod = llvm.to_module(module, llvm_context)
    options = type("Options", (), {"ptx_version": None})()
    proc, triple = sm_arch_from_capability(arch), "nvptx64-nvidia-cuda"
    features = get_features(options, arch)
    nvidia.set_short_ptr()
    llvm.attach_datalayout(llvm_mod, triple, proc, features)
    nvidia.set_nvvm_reflect_ftz(llvm_mod)
    llvm.link_extern_libs(llvm_mod, [libdevice_path])
    llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)
    ptx = llvm.translate_to_asm(str(llvm_mod), triple, proc, features, ["nvptx-mad-wide-opt"], True, False, False)
    body = [line for line in ptx.splitlines()
            if not line.lstrip().startswith((".version", ".target", ".address_size", "//"))]
    return "\n".join(body).strip()


def _attach_externs(stmts: list, module: str) -> bool:
    for stmt in stmts:
        if stmt.get("k") == "asm" and stmt.get("extern"):
            stmt["module"] = module
            return True
        for key in ("then", "otherwise", "body"):
            if key in stmt and _attach_externs(stmt[key], module):
                return True
    return False


def _libdevice_path() -> str:
    from triton import knobs
    return knobs.nvidia.libdevice_path or str(
        Path(__import__("triton.backends.nvidia", fromlist=["x"]).__file__).parent / "lib" / "libdevice.10.bc")


@dataclass(frozen=True)
class RudaOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 3
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


_PROBING_OFFICIAL = False


def takes_over_cuda() -> bool:
    """With TRITON_DEFAULT_BACKEND=ruda, kernels compiled for the "cuda" target (e.g. by Inductor) go through Ruda."""
    return os.environ.get("TRITON_DEFAULT_BACKEND") == BACKEND and not _PROBING_OFFICIAL


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

    def make_simt(self, mod, metadata, opt):
        ttir = mod.str_nodebug()
        layouts = official_reduce_layouts(ttir, opt.num_warps, opt.num_stages, self.target.arch) \
            if "tt.reduce" in ttir else None
        kernel, arg_kinds = lower(ttir, opt.num_warps, opt.enable_fp_fusion, layouts)
        externs = kernel.pop("externs")
        if externs:
            entries = tuple(sorted((name, *sig) for name, sig in externs.items()))
            _attach_externs(kernel["body"], libdevice_functions(entries, self.target.arch, _libdevice_path()))
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
