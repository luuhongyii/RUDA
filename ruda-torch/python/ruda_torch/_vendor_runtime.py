"""Dynamic symbol loading; typed native calls run on RUDA's device service."""
import ctypes as ct
import ctypes.util
import glob
import os
from pathlib import Path
import sys
from threading import RLock
import torch

_lock = RLock()
_libraries = {}
_initialized = False
_dll_dirs = []


def _load(name):
    with _lock:
        if name in _libraries:
            return _libraries[name]
        override = os.environ.get('RUDA_TORCH_' + name.upper() + '_LIBRARY')
        if sys.platform == 'win32':
            names = {'cuda': ['nvcuda.dll'], 'cublas': ['cublas64_13.dll', 'cublas64_12.dll'],
                     'cublaslt': ['cublasLt64_13.dll', 'cublasLt64_12.dll'],
                     'cufft': ['cufft64_12.dll', 'cufft64_11.dll'],
                     'cusolver': ['cusolver64_12.dll', 'cusolver64_11.dll']}[name]
            loader = ct.WinDLL
        else:
            names = {'cuda': ['libcuda.so.1'], 'cublas': ['libcublas.so.13', 'libcublas.so.12'],
                     'cublaslt': ['libcublasLt.so.13', 'libcublasLt.so.12'],
                     'cufft': ['libcufft.so.12', 'libcufft.so.11'],
                     'cusolver': ['libcusolver.so.12', 'libcusolver.so.11']}[name]
            loader = ct.CDLL
        roots = [Path(torch.__file__).resolve().parent / 'lib']
        roots.extend(Path(p) for p in os.environ.get('RUDA_TORCH_DLL_DIR', '').split(os.pathsep) if p)
        for key in ('CUDA_PATH', 'CUDA_HOME'):
            if os.environ.get(key):
                roots.extend(Path(os.environ[key]) / d for d in ('bin', 'bin/x64', 'lib64'))
        # CUDA Python wheels keep libraries outside torch/lib. No download or
        # filesystem-wide search is performed; only installed package locations.
        if name != 'cuda':
            roots.extend(Path(p) for p in glob.glob(str(Path(torch.__file__).resolve().parents[1] / 'nvidia' / '*' / 'lib')))
            roots.extend(Path(p) for p in glob.glob(str(Path(torch.__file__).resolve().parents[1] / 'nvidia' / '*' / 'bin')))
        if sys.platform == 'win32':
            for root in roots:
                if root.is_dir():
                    _dll_dirs.append(os.add_dll_directory(str(root)))
        found = ct.util.find_library(name)
        candidates = [override] if override else [*(str(root / item) for root in roots for item in names), *names]
        if found and not override:
            candidates.append(found)
        failures = []
        for path in candidates:
            try:
                library = loader(path)
                _libraries[name] = library
                return library
            except OSError as error:
                failures.append(str(error))
        raise RuntimeError(f'{name} shared library unavailable; set RUDA_TORCH_{name.upper()}_LIBRARY. ' + '; '.join(failures[-2:]))


def _address(library,name):
    with _lock:
        lib = _load(library)
        try:
            fn = getattr(lib, name)
        except AttributeError as error:
            raise RuntimeError(f'{library} does not export {name}') from error
        return ct.cast(fn,ct.c_void_p).value


def _initialize():
    global _initialized
    from . import _C, _native
    with _lock:
        if not _initialized:
            if not hasattr(_native, 'ruda_torch_vendor_api_version') or not hasattr(_C, 'initialize_vendor'):
                raise RuntimeError('direct NVIDIA backend requires rebuilding Rust and C++ libraries')
            _native.ruda_torch_vendor_api_version.restype = ct.c_uint32
            if _native.ruda_torch_vendor_api_version() != 1 or _C.vendor_api_version != 1:
                raise RuntimeError('RUDA vendor API mismatch')
            _C.initialize_vendor(ct.cast(_native.ruda_torch_vendor_scope,ct.c_void_p).value)
            _initialized = True
    return _C


def invoke(op,tensors,ints,scalars=(),*,prefix=None,stream=None):
    bridge = _initialize()
    if stream is not None and bridge.stream_command(0)!=stream:
        raise RuntimeError('NVIDIA plan must execute on its creation RUDA stream')
    symbols = {'cuda': ['cuStreamSynchronize','cuMemAlloc_v2','cuMemFree_v2']}
    if op in (0,1):
        symbols['cublas'] = ['cublasCreate_v2','cublasDestroy_v2','cublasSetStream_v2','cublasSetPointerMode_v2']
        symbols['cublas'].append('cublasGemmStridedBatchedEx' if op==0 else
                                ('cublasDtrsm_v2' if ints[4]==8 else 'cublasStrsm_v2'))
    elif op==15:
        dtype, conjugate = ints[3:5]
        symbol = {0:'cublasSdot_v2', 1:'cublasDdot_v2', 2:'cublasDotEx', 14:'cublasDotEx',
                  4:'cublasCdotc_v2' if conjugate else 'cublasCdotu_v2',
                  5:'cublasZdotc_v2' if conjugate else 'cublasZdotu_v2'}[dtype]
        symbols['cublas'] = ['cublasCreate_v2','cublasDestroy_v2','cublasSetStream_v2','cublasSetPointerMode_v2',symbol]
    elif op in (16,17):
        symbols['cublaslt'] = ['cublasLtCreate','cublasLtDestroy','cublasLtMatmulDescCreate','cublasLtMatmulDescDestroy',
                               'cublasLtMatrixLayoutCreate','cublasLtMatrixLayoutDestroy','cublasLtMatmul']
        if op==17:
            symbols['cublaslt'].extend(('cublasLtMatmulDescSetAttribute','cublasLtMatmulPreferenceCreate',
                                       'cublasLtMatmulPreferenceDestroy','cublasLtMatmulPreferenceSetAttribute',
                                       'cublasLtMatmulAlgoGetHeuristic'))
    elif 2<=op<=11:
        routine = {2:'getrf',3:'getrs',4:'potrf',5:'potrs',6:'geqrf',7:'orgqr',8:'ormqr',9:'syevd',10:'gesvdj',11:'gesvd'}[op]
        symbols['cusolver'] = ['cusolverDnCreate','cusolverDnDestroy','cusolverDnSetStream',prefix+routine]
        if op not in (3,5):
            symbols['cusolver'].append(prefix+routine+'_bufferSize')
        if op==10:
            symbols['cusolver'].extend(('cusolverDnCreateGesvdjInfo','cusolverDnDestroyGesvdjInfo'))
    elif op==12:
        symbols['cufft'] = ['cufftCreate','cufftDestroy','cufftSetAutoAllocation','cufftSetStream',
                            'cufftXtMakePlanMany' if ints[2]==2 else 'cufftMakePlanMany64','cufftSetWorkArea']
    elif op==13:
        kind,precision = ints[1:]
        if precision==2:
            symbols['cufft'] = ['cufftSetStream','cufftXtExec']
        else:
            suffix = ({0:'Z2Z',1:'Z2Z',2:'D2Z',3:'Z2D'} if precision else
                      {0:'C2C',1:'C2C',2:'R2C',3:'C2R'})[kind]
            symbols['cufft'] = ['cufftSetStream','cufftExec'+suffix]
    elif op==14:
        symbols['cufft'] = ['cufftDestroy']
    else:
        raise ValueError('unknown vendor command')
    addresses = {name:_address(library,name) for library,names in symbols.items() for name in names}
    return bridge.vendor_execute(op,tensors,list(ints),list(scalars),addresses)


def integer(value):
    if not 0 <= value <= 0x7fffffff:
        raise OverflowError('vendor routine dimension exceeds signed int32')
    return value


def column(tensor):
    tensor = tensor.resolve_conj().resolve_neg()
    return tensor.transpose(-2, -1).contiguous().transpose(-2, -1)


def empty_column(shape, like, dtype=None):
    return torch.empty((*shape[:-2], shape[-1], shape[-2]), device=like.device,
                       dtype=like.dtype if dtype is None else dtype).transpose(-2, -1)
