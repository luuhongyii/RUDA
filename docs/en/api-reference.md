# Full-stack crate and API index

[Documentation](README.md) · [中文](../zh/api-reference.md)

[Rust methods and types](rust-api.md) · [Python signatures](python-api-signatures.md) · [Python parallel training](parallel-and-data.md)

## Start at the right layer

| Task | First API | Guide |
| --- | --- | --- |
| Write a GPU kernel | `ruda`, kernel DSL, runtime client | [Programming guide](programming-guide.md) |
| Use Python/PyTorch on NVIDIA | `ruda_torch`, `torch.device("ruda:0")` | [PyTorch quickstart](native-pytorch-quickstart.md) and [API reference](native-pytorch-api.md) |
| Use typed tensors and gradients | `ruda_tensor::api::Tensor`, `Autodiff<B>` | [Tensor recipes](tensor-recipes.md) |
| Choose a tensor execution target | `Host`, `Cuda`, `Rocm`, `Wgpu`, Router, Remote | [Backend composition](backend-composition.md) |
| Call a domain operator | ruBLAS, ruDNN, ruPRIM, and other domain crates | [Compute libraries](libraries/README.md) |
| Build and train a model | `Module`, `ruda-nn`, `ruda-optim` | [Training](training.md) |
| Train with half storage and FP32 updates | `Module::to_dtype`, `Fp32MasterOptimizer`, `GradientsAccumulator::accumulate_with_dtype` | [Mixed-precision training](training.md#fp32-masters-and-mixed-parameter-storage) |
| Preserve mixed storage and pending gradients | `TrainingRecord::capture_with_dtypes`, `restore_with_dtypes` | [Training state](training.md#save-and-restore-training-state) |
| Synchronize replicas or differentiate collectives | `DataParallel`, `ruda_autodiff::collective` | [Distributed training](training.md#replicated-training-and-differentiable-tensor-collectives) |
| Load samples or weights | `Dataset`, `DataLoaderBuilder`, `ModuleSnapshot` | [Data and storage](data-and-storage.md) |
| Run local model inference | `rullm` | [Model inference](model-inference.md) |

## Package names and source entry points

The Cargo package name, repository directory, and Rust import name can differ. For example, ruFFT lives in `crates/ruFFT/`, is installed as `ruda-fft`, and is imported as `rufft`; ruSOLVER is installed as `ruda-solver` and imported as `rusolver`.

The tables cover the resolved workspace, including the path-dependent CANN driver and the test-support packages. Package links open their manifests; source links open the crate entry point. Features and versions are specified by each manifest, not by the directory name. Test fixtures and the native PyTorch extension are not ordinary crates.io installation targets.

## Kernel, compiler, and runtime

| Cargo package | Rust import / source | Responsibility |
| --- | --- | --- |
| [`ruda`](../../crates/ruda/Cargo.toml) | [`ruda`](../../crates/ruda/src/lib.rs) | Kernel facade and runtime re-exports; start here for GPU kernels |
| [`ruda-core`](../../crates/ruda-core/Cargo.toml) | [`ruda_core`](../../crates/ruda-core/src/lib.rs) | Shared IR, tensor data, dtype, device, and memory contracts |
| [`ruda-compiler`](../../crates/ruda-compiler/Cargo.toml) | [`ruda_compiler`](../../crates/ruda-compiler/src/lib.rs) | Kernel frontends and target code generation |
| [`ruda-triton`](../../crates/ruda-triton/Cargo.toml) | [`ruda_triton`](../../crates/ruda-triton/src/lib.rs) | Triton SIMT description → Ruda IR → PTX; source-build integration, not a crates.io package |
| [`ruda-kernel`](../../crates/ruda-kernel/Cargo.toml) | [`ruda_kernel`](../../crates/ruda-kernel/src/lib.rs) | Rust kernel DSL, launch interfaces, and device tensor utilities |
| [`ruda-runtime`](../../crates/ruda-runtime/Cargo.toml) | [`ruda_runtime`](../../crates/ruda-runtime/src/lib.rs) | Compute clients/servers, storage, streams, and scheduling |
| [`ruda-kernel-macros`](../../crates/ruda-kernel-macros/Cargo.toml) | [`ruda_kernel_macros`](../../crates/ruda-kernel-macros/src/lib.rs) | Procedural macros translating kernel Rust to IR |
| [`ruda-ir-macros`](../../crates/ruda-ir-macros/Cargo.toml) | [`ruda_ir_macros`](../../crates/ruda-ir-macros/src/lib.rs) | Derive helpers for IR implementations |

## Drivers and Ascend programs

| Cargo package | Rust import / source | Responsibility |
| --- | --- | --- |
| [`ruda-driver-cuda`](../../crates/ruda-driver-cuda/Cargo.toml) | [`ruda_driver_cuda`](../../crates/ruda-driver-cuda/src/lib.rs) | CUDA devices, allocation, compilation, and launch |
| [`ruda-driver-hip`](../../crates/ruda-driver-hip/Cargo.toml) | [`ruda_driver_hip`](../../crates/ruda-driver-hip/src/lib.rs) | AMD HIP runtime adapter |
| [`ruda-hip-sys`](../../crates/ruda-hip-sys/Cargo.toml) | [`ruda_hip_sys`](../../crates/ruda-hip-sys/src/lib.rs) | Low-level HIP runtime bindings |
| [`ruda-driver-wgpu`](../../crates/ruda-driver-wgpu/Cargo.toml) | [`ruda_driver_wgpu`](../../crates/ruda-driver-wgpu/src/lib.rs) | WGPU setup, storage, and compute execution |
| [`ruda-driver-cpu`](../../crates/ruda-driver-cpu/Cargo.toml) | [`ruda_driver_cpu`](../../crates/ruda-driver-cpu/src/lib.rs) | CPU kernel runtime driver, separate from Host tensor backend |
| [`ruda-driver-cann`](../../crates/ruda-driver-cann/Cargo.toml) | [`ruda_driver_cann`](../../crates/ruda-driver-cann/src/lib.rs) | Dynamically loaded CANN AscendCL interfaces |
| [`ruda-ascend-kernels`](../../crates/ruda-ascend-kernels/Cargo.toml) | [`ruda_ascend_kernels`](../../crates/ruda-ascend-kernels/src/lib.rs) | Rust Ascend device programs and instruction lowering |

## Compute libraries

| Cargo package | Rust import / source | Responsibility |
| --- | --- | --- |
| [`rublas`](../../crates/ruBLAS/Cargo.toml) | [`rublas`](../../crates/ruBLAS/src/lib.rs) | Matrix/vector operations and backend dispatch |
| [`ruDNN`](../../crates/ruDNN/Cargo.toml) | [`rudnn`](../../crates/ruDNN/src/lib.rs) | Attention, convolution, pooling, normalization, and MoE |
| [`ruPRIM`](../../crates/ruPRIM/Cargo.toml) | [`ruprim`](../../crates/ruPRIM/src/lib.rs) | Reductions, scans, indexing, and elementwise kernels |
| [`ruda-fft`](../../crates/ruFFT/Cargo.toml) | [`rufft`](../../crates/ruFFT/src/lib.rs) | Fourier transforms |
| [`ruRAND`](../../crates/ruRAND/Cargo.toml) | [`rurand`](../../crates/ruRAND/src/lib.rs) | Random sampling and distributions |
| [`ruSPARSE`](../../crates/ruSPARSE/Cargo.toml) | [`rusparse`](../../crates/ruSPARSE/src/lib.rs) | Sparse formats and operators |
| [`ruTENSOR`](../../crates/ruTENSOR/Cargo.toml) | [`rutensor`](../../crates/ruTENSOR/src/lib.rs) | Tensor contractions, permutations, and reductions |
| [`ruCCL`](../../crates/ruCCL/Cargo.toml) | [`ruccl`](../../crates/ruCCL/src/lib.rs) | Collective communication algorithms |
| [`ruda-solver`](../../crates/ruSOLVER/Cargo.toml) | [`rusolver`](../../crates/ruSOLVER/src/lib.rs) | Host scientific solvers and opt-in device solve kernels |
| [`ruintegrate`](../../crates/ruINTEGRATE/Cargo.toml) | [`ruintegrate`](../../crates/ruINTEGRATE/src/lib.rs) | Host quadrature, ODE integration, and event location |
| [`rublas-host`](../../crates/ruBLAS/host/Cargo.toml) | [`rublas_host`](../../crates/ruBLAS/host/src/lib.rs) | CPU strided and batched matrix multiplication |
| [`ruDNN-host`](../../crates/ruDNN/host/Cargo.toml) | [`rudnn_host`](../../crates/ruDNN/host/src/lib.rs) | CPU neural-network operator implementations |
| [`ruPRIM-host`](../../crates/ruPRIM/host/Cargo.toml) | [`ruprim_host`](../../crates/ruPRIM/host/src/lib.rs) | CPU tensor primitives and indexing |
| [`ruFFT-host`](../../crates/ruFFT/host/Cargo.toml) | [`rufft_host`](../../crates/ruFFT/host/src/lib.rs) | CPU real Fourier transforms |
| [`ruRAND-host`](../../crates/ruRAND/host/Cargo.toml) | [`rurand_host`](../../crates/ruRAND/host/src/lib.rs) | Host random-number generation |

## Tensors, differentiation, and execution composition

| Cargo package | Rust import / source | Responsibility |
| --- | --- | --- |
| [`ruda-tensor`](../../crates/ruda-tensor/Cargo.toml) | [`ruda_tensor`](../../crates/ruda-tensor/src/lib.rs) | Backend contracts and api::Tensor behind the api feature |
| [`ruda-tensor-config`](../../crates/ruda-tensor-config/Cargo.toml) | [`ruda_tensor_config`](../../crates/ruda-tensor-config/src/lib.rs) | Shared autodiff/fusion configuration |
| [`ruda-tensor-device`](../../crates/ruda-tensor-device/Cargo.toml) | [`ruda_tensor_device`](../../crates/ruda-tensor-device/src/lib.rs) | DeviceBackend, CUDA adapter, and domain-library dispatch |
| [`ruda-tensor-host`](../../crates/ruda-tensor-host/Cargo.toml) | [`ruda_tensor_host`](../../crates/ruda-tensor-host/src/lib.rs) | Host CPU tensor backend and strided layouts |
| [`ruda-tensor-wgpu`](../../crates/ruda-tensor-wgpu/Cargo.toml) | [`ruda_tensor_wgpu`](../../crates/ruda-tensor-wgpu/src/lib.rs) | WGPU tensor backend and device setup |
| [`ruda-tensor-rocm`](../../crates/ruda-tensor-rocm/Cargo.toml) | [`ruda_tensor_rocm`](../../crates/ruda-tensor-rocm/src/lib.rs) | ROCm tensor adapter |
| [`ruda-tensor-tch`](../../crates/ruda-tensor-tch/Cargo.toml) | [`ruda_tensor_tch`](../../crates/ruda-tensor-tch/src/lib.rs) | LibTorch tensor adapter |
| [`ruda-autodiff`](../../crates/ruda-autodiff/Cargo.toml) | [`ruda_autodiff`](../../crates/ruda-autodiff/src/lib.rs) | Gradient graph, backward pass, and recomputation |
| [`ruda-fusion`](../../crates/ruda-fusion/Cargo.toml) | [`ruda_fusion`](../../crates/ruda-fusion/src/lib.rs) | Tensor operation fusion planning and execution |
| [`ruda-tensor-router`](../../crates/ruda-tensor-router/Cargo.toml) | [`ruda_tensor_router`](../../crates/ruda-tensor-router/src/lib.rs) | Local multi-backend routing and byte bridges |
| [`ruda-tensor-remote`](../../crates/ruda-tensor-remote/Cargo.toml) | [`ruda_tensor_remote`](../../crates/ruda-tensor-remote/src/lib.rs) | Remote tensor client/server |
| [`ruda-communication`](../../crates/ruda-communication/Cargo.toml) | [`ruda_communication`](../../crates/ruda-communication/src/lib.rs) | Transport protocols, WebSocket, and tensor data service |

## Models, training, storage, and integrations

| Cargo package | Rust import / source | Responsibility |
| --- | --- | --- |
| [`ruda-model`](../../crates/ruda-model/Cargo.toml) | [`ruda_model`](../../crates/ruda-model/src/lib.rs) | Module parameters, configuration, records, and data loaders |
| [`ruda-model-macros`](../../crates/ruda-model-macros/Cargo.toml) | [`ruda_model_macros`](../../crates/ruda-model-macros/src/lib.rs) | Config, Module, and Record derive macros |
| [`ruda-model-codegen`](../../crates/ruda-model-codegen/Cargo.toml) | [`ruda_model_codegen`](../../crates/ruda-model-codegen/src/lib.rs) | Code generation used by the model derive macros |
| [`ruda-nn`](../../crates/ruda-nn/Cargo.toml) | [`ruda_nn`](../../crates/ruda-nn/src/lib.rs) | Neural-network layers, activations, and losses |
| [`ruda-optim`](../../crates/ruda-optim/Cargo.toml) | [`ruda_optim`](../../crates/ruda-optim/src/lib.rs) | Optimizers, gradient accumulation/clipping, and schedules |
| [`ruda-dataset`](../../crates/ruda-dataset/Cargo.toml) | [`ruda_dataset`](../../crates/ruda-dataset/src/lib.rs) | Indexed datasets, sources, and transforms |
| [`ruda-io`](../../crates/ruda-io/Cargo.toml) | [`ruda_io`](../../crates/ruda-io/src/lib.rs) | Host I/O and optional network downloads |
| [`ruda-store`](../../crates/ruda-store/Cargo.toml) | [`ruda_store`](../../crates/ruda-store/src/lib.rs) | Model snapshots, Rudapack, safetensors, and PyTorch import |
| [`ruda-llm`](../../crates/ruLLM/Cargo.toml) | [`rullm`](../../crates/ruLLM/src/lib.rs) | Model loading and autoregressive inference |
| [`ruda-torch-native`](../../crates/ruda-torch/Cargo.toml) | [`ruda_torch_native`](../../crates/ruda-torch/src/lib.rs) | Native cdylib for the Python ruda_torch package; source-build component |

## Test and consumer support

| Cargo package | Rust import / source | Responsibility |
| --- | --- | --- |
| [`ruda-test-runtime`](../../crates/ruda-test-runtime/Cargo.toml) | [`ruda_test_runtime`](../../crates/ruda-test-runtime/src/lib.rs) | Runtime/kernel test infrastructure |
| [`ruda-test-utils`](../../crates/ruda-test-utils/Cargo.toml) | [`ruda_test_utils`](../../crates/ruda-test-utils/src/lib.rs) | Kernel test helpers |
| [`ruda-facade-consumer`](../../crates/ruda/tests/consumer/Cargo.toml) | [`ruda_facade_consumer`](../../crates/ruda/tests/consumer/src/lib.rs) | External-consumer fixture for facade feature wiring |
| [`ruda-store-pytorch-tests`](../../crates/ruda-store/pytorch-tests/Cargo.toml) | [`ruda_store_pytorch_tests`](../../crates/ruda-store/pytorch-tests/src/lib.rs) | PyTorch format interoperability tests |
| [`ruda-store-safetensors-tests`](../../crates/ruda-store/safetensors-tests/Cargo.toml) | [`ruda_store_safetensors_tests`](../../crates/ruda-store/safetensors-tests/src/lib.rs) | Safetensors format interoperability tests |

## Find an individual method

Native Python methods are documented in the [PyTorch API reference](native-pytorch-api.md), [model compiler](model-compiler.md), [static graphs](static-pytorch-graphs.md) and [LoRA/NF4 fine-tuning guide](finetuning.md). Shared operator/application selection uses the [stack autotuning policy](stack-autotuning.md).

The [architecture guide](architecture-training.md) documents mHC, compressed attention/caches and Python Muon. The [distributed training guide](distributed-training.md) covers explicit devices, rendezvous, replica initialization, weighted reduction and rank-local recovery.

Use the typed tensor methods under [crates/ruda-tensor/src/api](../../crates/ruda-tensor/src/api), backend traits under [crates/ruda-tensor/src/backend](../../crates/ruda-tensor/src/backend), and domain-specific modules linked from each library guide. Enable the feature that exposes the module before using its symbols.

For device memory, submission, and synchronization, read the [Runtime API](runtime-api.md). For backend-specific initialization and launch contracts, read the [Driver API](driver-api.md). For model parameter and record types, start with [ruda-model exports](../../crates/ruda-model/src/lib.rs), rather than the compiler's similarly named IR types.
