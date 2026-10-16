import ctypes
import os
from pathlib import Path
import sys
from contextlib import contextmanager

import torch

_name = torch._C._get_privateuse1_backend_name()
if _name != "privateuseone":
    raise RuntimeError(f"PrivateUse1 is already registered as {_name}; use a separate process for RUDA")

_root = Path(__file__).resolve().parents[3]
_filename = "ruda_torch_native.dll" if sys.platform == "win32" else "libruda_torch_native.so"
_packaged_library = Path(__file__).resolve().with_name(_filename)
_default_library = _packaged_library if _packaged_library.is_file() else _root / "target" / "debug" / _filename
_library = Path(os.environ.get("RUDA_TORCH_LIBRARY", _default_library))
_dll_directories = []
if sys.platform == "win32":
    for directory in os.environ.get("RUDA_TORCH_DLL_DIR", "").split(os.pathsep):
        if directory:
            _dll_directories.append(os.add_dll_directory(directory))
_native = ctypes.CDLL(str(_library))

from . import _C
from .math_backend import get_math_backend, set_math_backend, math_backend

if not hasattr(_native, "ruda_torch_abi_version"):
    raise RuntimeError("RUDA native library is outdated; rebuild Rust and C++ extensions")
_native.ruda_torch_abi_version.restype = ctypes.c_uint32
if _native.ruda_torch_abi_version() != 10 or getattr(_C, "abi_version", None) != 10:
    raise RuntimeError("RUDA native ABI mismatch; rebuild Rust and C++ extensions")

_C.initialize([ctypes.cast(getattr(_native, "ruda_torch_" + name), ctypes.c_void_p).value
               for name in ("alloc", "free", "error", "execute", "transfer", "sync", "fill", "spatial", "addmm", "layer_norm", "rms_norm", "stream", "paged")])
torch.utils.rename_privateuse1_backend("ruda")

from . import _scientific
_scientific_available = _scientific.initialize(_native)
_blas_available = False
if hasattr(_native, "ruda_torch_blas_api_version") and hasattr(_C, "initialize_blas"):
    _native.ruda_torch_blas_api_version.restype = ctypes.c_uint32
    if _native.ruda_torch_blas_api_version() != 1 or _C.blas_api_version != 1:
        raise RuntimeError("RUDA BLAS API mismatch; rebuild both native libraries")
    _C.initialize_blas(ctypes.cast(_native.ruda_torch_blas, ctypes.c_void_p).value)
    _blas_available = True

_indexing_available = False
if hasattr(_native, "ruda_torch_indexing_api_version") and hasattr(_C, "initialize_indexing"):
    _native.ruda_torch_indexing_api_version.restype = ctypes.c_uint32
    if _native.ruda_torch_indexing_api_version() != 1 or getattr(_C, "indexing_api_version", None) != 1:
        raise RuntimeError("RUDA indexing API mismatch; rebuild Rust and C++ extensions")
    _C.initialize_indexing(ctypes.cast(_native.ruda_torch_advanced_index, ctypes.c_void_p).value)
    _indexing_available = True

_sorting_available = False
if hasattr(_native,'ruda_torch_sorting_api_version') and hasattr(_C,'initialize_sorting'):
    _native.ruda_torch_sorting_api_version.restype = ctypes.c_uint32
    if _native.ruda_torch_sorting_api_version()!=1 or getattr(_C,'sorting_api_version',0)!=1:
        raise RuntimeError('RUDA sorting API mismatch; rebuild Rust and C++ extensions')
    _C.initialize_sorting(ctypes.cast(_native.ruda_torch_sorting,ctypes.c_void_p).value)
    _sorting_available=True

_ordering_available = False
if hasattr(_native, 'ruda_torch_ordering_api_version') and hasattr(_C, 'initialize_ordering'):
    _native.ruda_torch_ordering_api_version.restype = ctypes.c_uint32
    if _native.ruda_torch_ordering_api_version() != 1 or getattr(_C, 'ordering_api_version', 0) != 1:
        raise RuntimeError('RUDA ordering API mismatch; rebuild Rust and C++ extensions')
    _C.initialize_ordering(ctypes.cast(_native.ruda_torch_ordering, ctypes.c_void_p).value)
    _ordering_available = True

_statistics_available = False
if hasattr(_native, 'ruda_torch_statistics_api_version') and hasattr(_C, 'initialize_statistics'):
    _native.ruda_torch_statistics_api_version.restype = ctypes.c_uint32
    if _native.ruda_torch_statistics_api_version() != 1 or getattr(_C, 'statistics_api_version', 0) != 1:
        raise RuntimeError('RUDA statistics API mismatch; rebuild Rust and C++ extensions')
    _C.initialize_statistics(ctypes.cast(_native.ruda_torch_statistics, ctypes.c_void_p).value)
    _statistics_available = True

_factory_available = False
if hasattr(_native, "ruda_torch_factory_api_version") and hasattr(_C, "initialize_factories"):
    _native.ruda_torch_factory_api_version.restype = ctypes.c_uint32
    if _native.ruda_torch_factory_api_version() != 1 or getattr(_C, "factory_api_version", None) != 1:
        raise RuntimeError("RUDA factory API mismatch; rebuild Rust and C++ extensions")
    _C.initialize_factories([ctypes.cast(getattr(_native, "ruda_torch_" + name), ctypes.c_void_p).value
                            for name in ("arange", "normal")])
    _factory_available = True


if hasattr(_C, "default_generator"):
    def manual_seed_all(seed):
        _C.default_generator().manual_seed(seed)

    def manual_seed(seed):
        manual_seed_all(seed)

    def get_rng_state(device=0):
        if device not in (0, "ruda", "ruda:0", torch.device("ruda:0")):
            raise ValueError("RUDA currently exposes only ruda:0")
        return _C.default_generator().get_state()

    def set_rng_state(state, device=0):
        if device not in (0, "ruda", "ruda:0", torch.device("ruda:0")):
            raise ValueError("RUDA currently exposes only ruda:0")
        _C.default_generator().set_state(state)


def is_available():
    return True

def is_initialized():
    """The native bridge is initialized before the device module is registered."""
    return True

def _lazy_init():
    # Importing this module already performs native ABI checks/initialization.
    return None

def device_count():
    return 1

def current_device():
    return 0

def _is_in_bad_fork():
    return False

def get_amp_supported_dtype():
    """Dtypes accepted by torch.autocast(device_type="ruda")."""
    return [torch.float16, torch.bfloat16]

def synchronize(device=None):
    if device not in (None, 0, "ruda", "ruda:0", torch.device("ruda:0")):
        raise ValueError("RUDA currently exposes only ruda:0")
    _C.synchronize()

torch._register_device_module("ruda", sys.modules[__name__])
torch.utils.generate_methods_for_privateuse1_backend()

_initialized = True
_math_available = False
if hasattr(_native, 'ruda_torch_math_api_version'):
    _native.ruda_torch_math_api_version.restype = ctypes.c_uint32
    if _native.ruda_torch_math_api_version() != 1:
        raise RuntimeError('RUDA math API mismatch; rebuild the native library')
    _math_available = True
_random_available = False
if hasattr(_native,'ruda_torch_random_api_version') and hasattr(_C,'initialize_random'):
    _native.ruda_torch_random_api_version.restype = ctypes.c_uint32
    if _native.ruda_torch_random_api_version()!=1 or getattr(_C,'random_api_version',0)!=1:
        raise RuntimeError('RUDA random API mismatch; rebuild Rust and C++ extensions')
    _C.initialize_random(ctypes.cast(_native.ruda_torch_random_fill,ctypes.c_void_p).value)
    _random_available = True
from .random import Generator, default_generator, uniform_, normal_, bernoulli_

def manual_seed_all(seed):
    default_generator.manual_seed(seed)

def manual_seed(seed):
    return default_generator.manual_seed(seed)

def get_rng_state(device=None):
    return default_generator.get_state()

def set_rng_state(state, device=None):
    default_generator.set_state(state)

def get_rng_state_all():
    return [get_rng_state()]

def set_rng_state_all(states):
    if len(states)!=1:
        raise ValueError('RUDA exposes one process-local random generator')
    set_rng_state(states[0])

@contextmanager
def device(index=None):
    if index not in (None,0,'ruda','ruda:0',torch.device('ruda:0')):
        raise ValueError('RUDA exposes only process-local ruda:0')
    yield

_native.ruda_torch_counter.argtypes = [ctypes.c_uint32]
_native.ruda_torch_counter.restype = ctypes.c_uint64

def execution_stats():
    return {name: _native.ruda_torch_counter(index) for index, name in enumerate(
        ("kernel_launches", "host_to_device_bytes", "device_to_host_bytes",
         "rublas_calls", "scalar_matmul_calls", "direct_pointwise_calls",
         "addmm_epilogues", "addmm_workspace_bytes_total", "legacy_fp32_temp_bytes_total",
         "warp_softmax_calls", "scalar_softmax_calls", "fused_layer_norm_calls",
         "trusted_index_calls", "storage_reduction_calls", "warp_reduction_calls",
         "fused_rms_norm_calls", "async_dispatches", "paged_split_calls",
         "paged_workspace_allocations", "paged_workspace_bytes_total",
         "static_graph_builds", "static_graph_replays", "static_graph_eager_runs",
         "paged_backward_calls", "paged_backward_history_workspace_bytes_total",
         "paged_backward_placeholder_bytes_total", "paged_backward_ordered_calls",
         "paged_backward_ordered_workspace_allocations", "paged_backward_ordered_workspace_bytes_total"))}

from . import _ops
if _scientific_available:
    _scientific.register()
from .blas import gemm, fused_gemm, grouped_gemm, triangular_solve
from . import _grouped_ops
from . import fft, linalg
if hasattr(_native, 'ruda_torch_fft_api_version') and hasattr(_C, 'initialize_fft'):
    fft.register()
_linalg_available = linalg.initialize(_native)
if _linalg_available:
    linalg.install(_ops._registry)
from .random import register_random_ops
register_random_ops()
from ._random_extra import register as _register_random_extra
_register_random_extra(_ops._registry)

from ._streams import Stream, Event, stream, current_stream, default_stream, record_stream, get_stream_priority_range
from ._paged import PagedAttentionPlan

# Additive extension API: the old eager tensor ABI still loads without it.
_graph_available = False
if hasattr(_native, "ruda_torch_graph_api_version") and hasattr(_C, "initialize_graph"):
    _native.ruda_torch_graph_api_version.restype = ctypes.c_uint32
    if _native.ruda_torch_graph_api_version() != 3 or getattr(_C,"graph_api_version",None) != 3:
        raise RuntimeError("RUDA static graph API mismatch; rebuild Rust and C++ extensions")
    _C.initialize_graph(ctypes.cast(_native.ruda_torch_graph,ctypes.c_void_p).value)
    _graph_available = True
_graph_layout_available = False
if _graph_available and hasattr(_native,'ruda_torch_graph_layout_api_version'):
    _native.ruda_torch_graph_layout_api_version.restype = ctypes.c_uint32
    _graph_layout_available = (_native.ruda_torch_graph_layout_api_version()==1
                               and getattr(_C,'graph_layout_api_version',0)==1)
_graph_math_available = False
if _graph_available and hasattr(_native,'ruda_torch_graph_math_api_version'):
    _native.ruda_torch_graph_math_api_version.restype = ctypes.c_uint32
    _graph_math_available = (_native.ruda_torch_graph_math_api_version()==1
                             and getattr(_C,'graph_math_api_version',0)==1)
from ._graph import StaticGraph, GraphOp
_graph_index_available = False
if _graph_available and hasattr(_native,'ruda_torch_graph_index_api_version'):
    _native.ruda_torch_graph_index_api_version.restype = ctypes.c_uint32
    _graph_index_available = (_native.ruda_torch_graph_index_api_version()==1
                              and getattr(_C,'graph_index_api_version',0)==1)
_graph_factory_available = False
if _graph_available and hasattr(_native,'ruda_torch_graph_factory_api_version'):
    _native.ruda_torch_graph_factory_api_version.restype = ctypes.c_uint32
    _graph_factory_available = (_native.ruda_torch_graph_factory_api_version()==1
                                and getattr(_C,'graph_factory_api_version',0)==1)
_graph_empty_available = False
if _graph_available and hasattr(_native,'ruda_torch_graph_empty_api_version'):
    _native.ruda_torch_graph_empty_api_version.restype = ctypes.c_uint32
    _graph_empty_available = (_native.ruda_torch_graph_empty_api_version()==1
                              and getattr(_C,'graph_empty_api_version',0)==1)
_graph_root_available = False
if _graph_available and hasattr(_native,'ruda_torch_graph_root_api_version'):
    _native.ruda_torch_graph_root_api_version.restype = ctypes.c_uint32
    _graph_root_available = (_native.ruda_torch_graph_root_api_version()==1
                             and getattr(_C,'graph_root_api_version',0)==1)
_operation_graph_available = False
if hasattr(_native,'ruda_torch_operation_graph_api_version') and hasattr(_C,'initialize_operation_graph'):
    _native.ruda_torch_operation_graph_api_version.restype = ctypes.c_uint32
    if _native.ruda_torch_operation_graph_api_version()==1 and getattr(_C,'operation_graph_api_version',0)==1:
        _C.initialize_operation_graph(ctypes.cast(_native.ruda_torch_operation_graph,ctypes.c_void_p).value)
        _operation_graph_available=True
_sequence_graph_available = False
if _operation_graph_available and hasattr(_native,'ruda_torch_sequence_graph_api_version'):
    _native.ruda_torch_sequence_graph_api_version.restype = ctypes.c_uint32
    _sequence_graph_available = (_native.ruda_torch_sequence_graph_api_version()==1
                                 and getattr(_C,'sequence_graph_api_version',0)==1)
_pointwise_graph_available = False
if _operation_graph_available and hasattr(_native,'ruda_torch_pointwise_graph_api_version'):
    _native.ruda_torch_pointwise_graph_api_version.restype = ctypes.c_uint32
    _pointwise_graph_available = (_native.ruda_torch_pointwise_graph_api_version()==1
                                  and getattr(_C,'pointwise_graph_api_version',0)==1)
_transform_graph_available = False
if _operation_graph_available and hasattr(_native,'ruda_torch_transform_graph_api_version'):
    _native.ruda_torch_transform_graph_api_version.restype = ctypes.c_uint32
    _transform_graph_available = (_native.ruda_torch_transform_graph_api_version()==1
                                 and getattr(_C,'transform_graph_api_version',0)==1)
_reduction_graph_available = False
if _operation_graph_available and hasattr(_native,'ruda_torch_reduction_graph_api_version'):
    _native.ruda_torch_reduction_graph_api_version.restype = ctypes.c_uint32
    _reduction_graph_available = (_native.ruda_torch_reduction_graph_api_version()==1
                                 and getattr(_C,'reduction_graph_api_version',0)==1)
_router_graph_available = False
if _operation_graph_available and hasattr(_native,'ruda_torch_router_graph_api_version'):
    _native.ruda_torch_router_graph_api_version.restype = ctypes.c_uint32
    _router_graph_available = (_native.ruda_torch_router_graph_api_version()==1
                              and getattr(_C,'router_graph_api_version',0)==1)
_grouped_graph_available = False
if _operation_graph_available and hasattr(_native,'ruda_torch_grouped_graph_api_version'):
    _native.ruda_torch_grouped_graph_api_version.restype = ctypes.c_uint32
    _grouped_graph_available = (_native.ruda_torch_grouped_graph_api_version()==1
                               and getattr(_C,'grouped_graph_api_version',0)==1)
_scatter_graph_available = False
if _operation_graph_available and hasattr(_native,'ruda_torch_scatter_graph_api_version'):
    _native.ruda_torch_scatter_graph_api_version.restype = ctypes.c_uint32
    _scatter_graph_available = (_native.ruda_torch_scatter_graph_api_version()==1
                               and getattr(_C,'scatter_graph_api_version',0)==1)

_sorting_graph_available = (_operation_graph_available and _sorting_available
                            and getattr(_C,'sorting_graph_api_version',0)==1)
_block_float8_graph_available = False
if _operation_graph_available and hasattr(_native, 'ruda_torch_float8_block_api_version'):
    _native.ruda_torch_float8_block_api_version.restype = ctypes.c_uint32
    _block_float8_graph_available = (_native.ruda_torch_float8_block_api_version() == 1
                                    and getattr(_C, 'block_float8_graph_api_version', 0) == 1)

_delayed_float8_graph_available = False
_shard_float8_graph_available = False
_indexed_attention_graph_available = False
_attention_dropout_graph_available = False
if _operation_graph_available and hasattr(_native, 'ruda_torch_attention_dropout_api_version'):
    _native.ruda_torch_attention_dropout_api_version.restype = ctypes.c_uint32
    _attention_dropout_graph_available = (_native.ruda_torch_attention_dropout_api_version() == 1
                                         and getattr(_C, 'attention_dropout_api_version', 0) == 1)
if _operation_graph_available and hasattr(_native, 'ruda_torch_indexed_dense_attention_api_version'):
    _native.ruda_torch_indexed_dense_attention_api_version.restype = ctypes.c_uint32
    _indexed_attention_graph_available = (_native.ruda_torch_indexed_dense_attention_api_version() == 1
                                         and getattr(_C, 'indexed_dense_attention_api_version', 0) == 1)
if _operation_graph_available and hasattr(_native, 'ruda_torch_float8_shard_api_version'):
    _native.ruda_torch_float8_shard_api_version.restype = ctypes.c_uint32
    _shard_float8_graph_available = (_native.ruda_torch_float8_shard_api_version() == 1
                                    and getattr(_C, 'float8_shard_api_version', 0) == 1)

if _operation_graph_available and hasattr(_native, 'ruda_torch_delayed_float8_graph_api_version'):
    _native.ruda_torch_delayed_float8_graph_api_version.restype = ctypes.c_uint32
    _delayed_float8_graph_available = (_native.ruda_torch_delayed_float8_graph_api_version() == 1
                                      and getattr(_C, 'delayed_float8_graph_api_version', 0) == 1)

_indexed_expert_available = False
if hasattr(_native, 'ruda_torch_indexed_expert_api_version'):
    _native.ruda_torch_indexed_expert_api_version.restype = ctypes.c_uint32
    _indexed_expert_available = (_native.ruda_torch_indexed_expert_api_version() == 1
                                 and getattr(_C, 'indexed_expert_api_version', 0) == 1)
_indexed_expert_graph_available = _operation_graph_available and _indexed_expert_available
_indexed_expert_training_available = False
if _indexed_expert_available and hasattr(_native,'ruda_torch_indexed_expert_training_api_version'):
    _native.ruda_torch_indexed_expert_training_api_version.restype = ctypes.c_uint32
    _indexed_expert_training_available = (_native.ruda_torch_indexed_expert_training_api_version() == 1
                                         and getattr(_C,'indexed_expert_training_api_version',0) == 1)

_indexed_block_float8_expert_available = False
if hasattr(_native,'ruda_torch_indexed_block_float8_expert_api_version'):
    _native.ruda_torch_indexed_block_float8_expert_api_version.restype = ctypes.c_uint32
    _indexed_block_float8_expert_available = (_native.ruda_torch_indexed_block_float8_expert_api_version() == 1
                                            and getattr(_C,'indexed_block_float8_expert_api_version',0) == 1)
_indexed_float8_expert_available = False
if hasattr(_native,'ruda_torch_indexed_float8_expert_api_version'):
    _native.ruda_torch_indexed_float8_expert_api_version.restype = ctypes.c_uint32
    _indexed_float8_expert_available = (_native.ruda_torch_indexed_float8_expert_api_version() == 1
                                      and getattr(_C,'indexed_float8_expert_api_version',0) == 1)
_indexed_quantized_expert_available = False
if hasattr(_native,'ruda_torch_indexed_quantized_expert_api_version'):
    _native.ruda_torch_indexed_quantized_expert_api_version.restype = ctypes.c_uint32
    _indexed_quantized_expert_available = (_native.ruda_torch_indexed_quantized_expert_api_version() == 1
                                          and getattr(_C,'indexed_quantized_expert_api_version',0) == 1)

# Training is a separately negotiated extension. Explicit StaticGraph training
# and model compilation remain opt-in. No CPU fallback is installed.
_training_available = False
_C.storage_mean_api = 0
if hasattr(_native, "ruda_torch_training_api_version") and hasattr(_C, "initialize_training"):
    _native.ruda_torch_training_api_version.restype = ctypes.c_uint32
    if _native.ruda_torch_training_api_version() != 4 or getattr(_C, "training_api_version", None) != 4:
        raise RuntimeError("RUDA training API mismatch; rebuild Rust and C++ extensions")
    _C.initialize_training(ctypes.cast(_native.ruda_torch_training, ctypes.c_void_p).value)
    _training_available = True
    _C.storage_mean_api = 1
from .training import LayerNorm, layer_norm, RMSNorm, rms_norm, silu_mul, AdamW, GradScaler

# Optional router-weight extension. Existing base ABI 10/training API 4 remain
# loadable without it; a router operation fails clearly instead of falling back.
_router_available = False
if hasattr(_native, "ruda_torch_router_api_version") and hasattr(_C, "initialize_router"):
    _native.ruda_torch_router_api_version.restype = ctypes.c_uint32
    if _native.ruda_torch_router_api_version() != 1 or getattr(_C, "router_api_version", None) != 1:
        raise RuntimeError("RUDA router API mismatch; rebuild Rust and C++ extensions")
    _C.initialize_router(ctypes.cast(_native.ruda_torch_router, ctypes.c_void_p).value)
    _router_available = True
from ._router import selected_router_weights
from . import _router_ops

# Selected paged backward has an additive capability check. Old eager tensor
# ABI 10 remains loadable, but the new autograd path never silently falls back
# to materializing all gradients with an older runtime.
_paged_backward_available = False
if hasattr(_native, "ruda_torch_paged_backward_api_version") and hasattr(_C, "initialize_paged_backward"):
    _native.ruda_torch_paged_backward_api_version.restype=ctypes.c_uint32
    version=_native.ruda_torch_paged_backward_api_version()
    if version not in (1,2) or getattr(_C,"paged_backward_api_version",0)<version:
        raise RuntimeError("RUDA paged backward API mismatch; rebuild Rust and C++ extensions")
    _C.initialize_paged_backward(version)
    _paged_backward_available=True

# General model capture is layered above the additive native graph API.
from .compiler import compile, make_backend, CompiledModel, CompiledFunction, NativeCoverageError, GraphExecutionError

from .quantization import LearnedFakeQuantize, learned_fake_quantize

_packed_available = False
if hasattr(_native, 'ruda_torch_packed_api_version') and hasattr(_C, 'initialize_packed'):
    _native.ruda_torch_packed_api_version.restype = ctypes.c_uint32
    if _native.ruda_torch_packed_api_version() != 1 or getattr(_C, 'packed_api_version', None) != 1:
        raise RuntimeError('RUDA packed quantization API mismatch; rebuild both native libraries')
    _C.initialize_packed(ctypes.cast(_native.ruda_torch_packed, ctypes.c_void_p).value)
    _packed_available = True
_mxfp8_available = False
if hasattr(_native, 'ruda_torch_mxfp8_api_version') and hasattr(_C, 'initialize_mxfp8'):
    _native.ruda_torch_mxfp8_api_version.restype = ctypes.c_uint32
    if _native.ruda_torch_mxfp8_api_version() != 1 or getattr(_C, 'mxfp8_api_version', None) != 1:
        raise RuntimeError('RUDA MXFP8 API mismatch; rebuild both native libraries')
    _C.initialize_mxfp8(1)
    _mxfp8_available = True
_fp4_available = False
if hasattr(_native, 'ruda_torch_fp4_api_version') and hasattr(_C, 'initialize_fp4'):
    _native.ruda_torch_fp4_api_version.restype = ctypes.c_uint32
    if _native.ruda_torch_fp4_api_version() != 1 or getattr(_C, 'fp4_api_version', None) != 1:
        raise RuntimeError('RUDA FP4 API mismatch; rebuild both native libraries')
    _C.initialize_fp4(1)
    _fp4_available = True
_packed_indexed_groups_available = False
if hasattr(_native, 'ruda_torch_packed_indexed_groups_api_version'):
    _native.ruda_torch_packed_indexed_groups_api_version.restype = ctypes.c_uint32
    _packed_indexed_groups_available = (_packed_available
        and _native.ruda_torch_packed_indexed_groups_api_version() == 1
        and getattr(_C, 'packed_indexed_groups_api_version', None) == 1)
_paged_quantized_available = False
if hasattr(_native, 'ruda_torch_paged_quantized_api_version') and hasattr(_C, 'initialize_paged_quantized'):
    _native.ruda_torch_paged_quantized_api_version.restype = ctypes.c_uint32
    if _native.ruda_torch_paged_quantized_api_version() != 1 or getattr(_C, 'paged_quantized_api_version', None) != 1:
        raise RuntimeError('RUDA quantized paged API mismatch; rebuild both native libraries')
    _C.initialize_paged_quantized(ctypes.cast(_native.ruda_torch_paged_quantized, ctypes.c_void_p).value)
    _paged_quantized_available = True
from .packed_quantization import (PackedQuantizedTensor, PackedLinear, PackedEmbedding, quantize_packed, scaled_float8_mm)
from .float8_training import Float8Recipe, Float8Linear, float8_linear, convert_to_float8_training
from .float8_scaling import Float8ScaleState, advance_float8_scales
from .quantization_calibration import (calibrate_quantization, QuantizationObserver, affine_fake_quantize)
from .quantized_cache import QuantizedKVCache, quantized_paged_attention

# Trainable architecture components are composed from same-device tensor ops.
from .mhc import MHC, MHCResidual, MHCSequential, MHCCoefficients, sinkhorn
from .sparse_attention import (LightningIndexer, DSAIndexer, indexer_kl_loss,
    LearnedKVCompressor, CompressedSparseAttention, HeavilyCompressedAttention,
    CSA, HCA, AttentionOutput, IndexerOutput, CompressedAttentionCache, CompressionState, RotaryEmbedding)
from .optim import Muon, MuonAdamW, muon_orthogonalize
from .hybrid_model import (MHCTransformerBlock, HybridAttentionLanguageModel,
                          HybridAttentionBackbone, HybridAttentionCache, next_token_loss)

_sequence_available = False
if hasattr(_native, "ruda_torch_sequence_api_version") and hasattr(_C, "initialize_sequence"):
    _native.ruda_torch_sequence_api_version.restype = ctypes.c_uint32
    if _native.ruda_torch_sequence_api_version() != 1 or getattr(_C, "sequence_api_version", None) != 1:
        raise RuntimeError("RUDA sequence API mismatch; rebuild Rust and C++ libraries")
    _C.initialize_sequence([ctypes.cast(getattr(_native, name), ctypes.c_void_p).value
                            for name in ('ruda_torch_triangular_solve', 'ruda_torch_delta_forward')])
    _sequence_available = True
from .sequence_training import gated_delta_rule, solve_triangular
from . import _native_training_ops
from . import _sequence_ops

_kda_available = False
if hasattr(_native, "ruda_torch_kda_api_version") and hasattr(_C, "initialize_kda"):
    _native.ruda_torch_kda_api_version.restype = ctypes.c_uint32
    if _native.ruda_torch_kda_api_version() != 1 or getattr(_C, "kda_api_version", None) != 1:
        raise RuntimeError("RUDA KDA API mismatch; rebuild Rust and C++ libraries")
    _C.initialize_kda([ctypes.cast(getattr(_native, name), ctypes.c_void_p).value
                       for name in ('ruda_torch_kda_forward', 'ruda_torch_kda_backward')])
    _kda_available = True
from . import _kda_ops

_nf4_available = False
_nf4_matmul_available = False
if hasattr(_native, "ruda_torch_nf4_matmul_api_version") and hasattr(_C, "initialize_nf4_matmul"):
    _native.ruda_torch_nf4_matmul_api_version.restype = ctypes.c_uint32
    if _native.ruda_torch_nf4_matmul_api_version() != 1 or getattr(_C, "nf4_matmul_api_version", None) != 1:
        raise RuntimeError("RUDA NF4 matmul API mismatch; rebuild Rust and C++ libraries")
    _C.initialize_nf4_matmul(ctypes.cast(_native.ruda_torch_nf4_matmul, ctypes.c_void_p).value)
    _nf4_matmul_available = True
if hasattr(_native, "ruda_torch_nf4_api_version") and hasattr(_C, "initialize_nf4"):
    _native.ruda_torch_nf4_api_version.restype = ctypes.c_uint32
    if _native.ruda_torch_nf4_api_version() != 1 or getattr(_C, "nf4_api_version", None) != 1:
        raise RuntimeError("RUDA NF4 API mismatch; rebuild Rust and C++ libraries")
    _C.initialize_nf4(ctypes.cast(_native.ruda_torch_nf4_decode, ctypes.c_void_p).value)
    _nf4_available = True
from .finetuning import (LoRALinear, NF4Linear, inject_lora, quantize_nf4,
                         adapter_state_dict, load_adapter_state_dict, merge_lora,
                         load_nf4_safetensors, finetune_state_dict, load_finetune_state_dict)
from .causal_finetuning import (chunked_lm_cross_entropy, SFTCollator, CausalLMFinetuner,
                               SFTTrainer, load_hf_nf4_model, activation_checkpoint_modules)
from .distributed_training import ReplicaGroup
if torch.distributed.is_available():
    from .process_group import ProcessGroupRuCCL, register_ruccl_backend
from .parallel_training import (ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding,
    TensorParallelMLP, TensorParallelAttention,
    tensor_parallelize, tensor_parallel_state_dict, load_tensor_parallel_state_dict,
    copy_to_tensor_parallel, reduce_from_tensor_parallel, gather_from_tensor_parallel, scatter_to_tensor_parallel)
from .sharded_training import FullyShardedModule, fully_shard, Zero2Optimizer, ShardedReplicaGroup
from .pipeline_training import PipelineStage, PipelineTensorSpec
from .structured_pipeline import StructuredPipelineStage,PipelineTreeSpec,PipelineLeafSpec,PipelineConstant
from .distributed_checkpoint import DistributedCheckpoint, model_shard_layout
from .adapter_interop import load_peft_adapter, save_peft_adapter
from .attention import scaled_dot_product_attention
from . import _dense_ops
from .block_attention import block_scaled_dot_product_attention
from .parallel_loss import vocab_parallel_cross_entropy
from .transformers_attention import register_transformers_attention
from .parallel_mesh import ParallelMesh, TensorParallelGroup
from .expert_parallel import (ExpertPartition, ExpertParallelMLP, expert_parallelize_kimi,
                             expert_parallelize_qwen35, expert_parallelize_qwen3,
                             expert_parallelize_deepseek, expert_parallelize_moe)
from .expert_training import ExpertParallelGroup
from .expert_parallel import tensor_parallelize_experts
from .expert_checkpoint import load_expert_parallel_safetensors
from .model_parallel_mesh import ModelParallelMesh
from .model_parallel_training import ModelParallelTrainingGroup
from .context_parallel import RingAttention, ring_scaled_dot_product_attention
from .context_recurrent import (context_parallel_kda, context_parallel_delta,
    context_parallel_causal_conv1d, context_document_resets)
from .sequence_parallel import (gather_sequence_parallel, reduce_scatter_sequence_parallel,
    scatter_sequence_parallel, SequenceParallelColumnLinear, SequenceParallelRowLinear, sequence_parallelize)
from .transformers_sequence_parallel import sequence_parallelize_transformers
from .sharded_optim import ShardedMuon, distributed_grad_norm
from .quantization_interop import bnb_nf4_linear, load_bnb_nf4_safetensors
from .gptq import gptq_linear, load_gptq_safetensors
from .gguf import GGUFWeights, GGUFTensorInfo, load_gguf
from .parallel_adapters import (ColumnParallelLoRALinear,RowParallelLoRALinear,
    ColumnParallelNF4Linear,RowParallelNF4Linear,shard_nf4_linear)
from .packed_parallel import (ColumnParallelPackedLinear,RowParallelPackedLinear,
    VocabParallelPackedEmbedding,shard_packed_linear)
from .parallel_checkpoint_io import load_tensor_parallel_safetensors,load_tensor_parallel_bnb_nf4_safetensors
from .hybrid_parallel import DataTensorParallelGroup
from .tensor_parallel_optim import TensorParallelMuon,MeshShardedMuon
from .sharded_mesh import ShardedDataTensorParallelGroup
from .pipeline_trainer import PipelineTrainer
from .pipeline_ties import PipelineTiedParameters
from .training_data import StatefulShardSampler,TokenBudgetBatchSampler,collate_varlen_causal_lm
from .varlen_attention import varlen_scaled_dot_product_attention
from . import _varlen_ops
from .causal_finetuning import PackedCausalLMFinetuner,CausalLMTrainer
from .causal_finetuning import ContextParallelFinetuner
from .training_checkpoint import training_state_dict,validate_training_state_dict,load_training_state_dict
from .training_autotune import TrainingAutotuner,TrainingPlan,PipelineAutotuner
from .architecture_cache import KVCache, KDAState, MambaState, ArchitectureCache
from ._recurrent_ops import causal_depthwise_conv1d, CausalDepthwiseConv1d
from .kda import kda, KimiDeltaAttention
from .mamba import ssd_scan, Mamba2
from .block_attention import attention_softcap, cached_block_scaled_dot_product_attention
from .moe import (RoutingPolicy, RoutingResult, route_experts, router_load_balancing_loss,
                  router_z_loss, SwiGLUExperts, MixtureOfExperts, MoEOutput)
from .weight_loading import (TensorSource, WeightMapping, SafetensorWeights,
                             WeightLoadResult, load_safetensor_weights)


def register_transformers_kimi(*, exist_ok=False):
    from .transformers_kimi import register_transformers_kimi as register
    return register(exist_ok=exist_ok)


def tensor_parallelize_kimi(model, group, *, vocab_parallel=True):
    from .transformers_kimi import tensor_parallelize_kimi as parallelize
    return parallelize(model, group, vocab_parallel=vocab_parallel)


def enable_transformers_qwen35(model):
    from .transformers_qwen35 import enable_transformers_qwen35 as enable
    return enable(model)


def enable_transformers_context_parallel(model, group, **kwargs):
    from .transformers_context_parallel import enable_transformers_context_parallel as enable
    return enable(model, group, **kwargs)


from .transformers_context_parallel import (ContextParallelCausalLM, shard_context_batch,
    context_parallel_cross_entropy, context_parallel_router_loss, register_transformers_ring_attention)
from .deepseek_mtp import (DeepseekMtpForCausalLM,DeepseekMtpPredictor,DeepseekMtpTrainingOutput,DeepseekMtpTrainer,
                          load_deepseek_mtp,parallelize_deepseek_mtp)
