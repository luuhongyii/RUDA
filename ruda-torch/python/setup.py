import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="ruda-torch", version="0.1.0", packages=["ruda_torch"],
    package_data={"ruda_torch": ["ruda_torch_native.dll", "libruda_torch_native.so", "csrc/*.cpp", "csrc/*.inc"]},
    ext_modules=[CppExtension("ruda_torch._C", ["ruda_torch/csrc/backend.cpp"],
        depends=["ruda_torch/csrc/collective_program.inc", "ruda_torch/csrc/collective_program_bindings.inc", "ruda_torch/csrc/ring_attention.inc", "ruda_torch/csrc/ring_attention_bindings.inc", "ruda_torch/csrc/operation_graph.inc", "ruda_torch/csrc/dense_attention.inc", "ruda_torch/csrc/dense_attention_bindings.inc", "ruda_torch/csrc/varlen_attention.inc", "ruda_torch/csrc/varlen_attention_bindings.inc", "ruda_torch/csrc/float8.inc", "ruda_torch/csrc/float8_bindings.inc", "ruda_torch/csrc/sorting.inc", "ruda_torch/csrc/static_graph.inc", "ruda_torch/csrc/training.inc", "ruda_torch/csrc/router.inc", "ruda_torch/csrc/nf4.inc", "ruda_torch/csrc/sequence_training.inc", "ruda_torch/csrc/kda.inc", "ruda_torch/csrc/packed.inc", "ruda_torch/csrc/random.inc", "ruda_torch/csrc/storage.inc", "ruda_torch/csrc/generator.inc", "ruda_torch/csrc/factories.inc", "ruda_torch/csrc/indexing.inc", "ruda_torch/csrc/scientific_bridge.inc", "ruda_torch/csrc/scientific_bridge_bindings.inc", "ruda_torch/csrc/blas_bridge.inc", "ruda_torch/csrc/blas_bridge_bindings.inc", "ruda_torch/csrc/linalg_bridge.inc", "ruda_torch/csrc/linalg_bridge_bindings.inc", "ruda_torch/csrc/fft_bridge.inc", "ruda_torch/csrc/fft_bridge_bindings.inc", "ruda_torch/csrc/vendor_bridge.inc", "ruda_torch/csrc/vendor_bridge_bindings.inc"],
        extra_compile_args=["/std:c++20", "/O2", "/EHsc", "/utf-8"] if sys.platform == "win32" else ["-std=c++20", "-O2"])],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
