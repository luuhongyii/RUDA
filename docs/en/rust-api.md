# Rust API reference

[Documentation](README.md) · [Python API](python-api-signatures.md)

Select the crate and version used by your application. Cargo features determine which modules are available; a published rustdoc build is not an all-feature backend support guarantee.

The table follows the manifests on the documented `main` snapshot. Versioned rustdoc describes the corresponding published package, not later `main`-only additions; use the [source/API index](api-reference.md) for those. `ruda-triton`, like the native PyTorch component, is source-build only.

| Package | Version | Rust import | API | Manifest |
| --- | --- | --- | --- | --- |
| `ruda` | `0.1.46` | `ruda` | [rustdoc](https://docs.rs/ruda/0.1.46/ruda/) | [Manifest](../../crates/ruda/Cargo.toml) |
| `ruda-core` | `0.1.10` | `ruda_core` | [rustdoc](https://docs.rs/ruda-core/0.1.10/ruda_core/) | [Manifest](../../crates/ruda-core/Cargo.toml) |
| `ruda-compiler` | `0.1.17` | `ruda_compiler` | [rustdoc](https://docs.rs/ruda-compiler/0.1.17/ruda_compiler/) | [Manifest](../../crates/ruda-compiler/Cargo.toml) |
| `ruda-triton` | `0.1.0` | `ruda_triton` | [Source](../../crates/ruda-triton/src/lib.rs) | [Manifest](../../crates/ruda-triton/Cargo.toml) |
| `ruda-kernel` | `0.1.45` | `ruda_kernel` | [rustdoc](https://docs.rs/ruda-kernel/0.1.45/ruda_kernel/) | [Manifest](../../crates/ruda-kernel/Cargo.toml) |
| `ruda-runtime` | `0.1.30` | `ruda_runtime` | [rustdoc](https://docs.rs/ruda-runtime/0.1.30/ruda_runtime/) | [Manifest](../../crates/ruda-runtime/Cargo.toml) |
| `ruda-kernel-macros` | `0.1.11` | `ruda_kernel_macros` | [rustdoc](https://docs.rs/ruda-kernel-macros/0.1.11/ruda_kernel_macros/) | [Manifest](../../crates/ruda-kernel-macros/Cargo.toml) |
| `ruda-ir-macros` | `0.1.2` | `ruda_ir_macros` | [rustdoc](https://docs.rs/ruda-ir-macros/0.1.2/ruda_ir_macros/) | [Manifest](../../crates/ruda-ir-macros/Cargo.toml) |
| `ruda-driver-cuda` | `0.1.98` | `ruda_driver_cuda` | [rustdoc](https://docs.rs/ruda-driver-cuda/0.1.98/ruda_driver_cuda/) | [Manifest](../../crates/ruda-driver-cuda/Cargo.toml) |
| `ruda-driver-hip` | `0.1.98` | `ruda_driver_hip` | [rustdoc](https://docs.rs/ruda-driver-hip/0.1.98/ruda_driver_hip/) | [Manifest](../../crates/ruda-driver-hip/Cargo.toml) |
| `ruda-hip-sys` | `7.14.6085000` | `ruda_hip_sys` | [rustdoc](https://docs.rs/ruda-hip-sys/7.14.6085000/ruda_hip_sys/) | [Manifest](../../crates/ruda-hip-sys/Cargo.toml) |
| `ruda-driver-wgpu` | `0.1.98` | `ruda_driver_wgpu` | [rustdoc](https://docs.rs/ruda-driver-wgpu/0.1.98/ruda_driver_wgpu/) | [Manifest](../../crates/ruda-driver-wgpu/Cargo.toml) |
| `ruda-driver-cpu` | `0.1.98` | `ruda_driver_cpu` | [rustdoc](https://docs.rs/ruda-driver-cpu/0.1.98/ruda_driver_cpu/) | [Manifest](../../crates/ruda-driver-cpu/Cargo.toml) |
| `ruda-driver-cann` | `0.1.98` | `ruda_driver_cann` | [rustdoc](https://docs.rs/ruda-driver-cann/0.1.98/ruda_driver_cann/) | [Manifest](../../crates/ruda-driver-cann/Cargo.toml) |
| `ruda-ascend-kernels` | `0.1.0` | `ruda_ascend_kernels` | [rustdoc](https://docs.rs/ruda-ascend-kernels/0.1.0/ruda_ascend_kernels/) | [Manifest](../../crates/ruda-ascend-kernels/Cargo.toml) |
| `rublas` | `0.1.98` | `rublas` | [rustdoc](https://docs.rs/rublas/0.1.98/rublas/) | [Manifest](../../crates/ruBLAS/Cargo.toml) |
| `ruDNN` | `0.1.98` | `rudnn` | [rustdoc](https://docs.rs/ruDNN/0.1.98/rudnn/) | [Manifest](../../crates/ruDNN/Cargo.toml) |
| `ruPRIM` | `0.1.98` | `ruprim` | [rustdoc](https://docs.rs/ruPRIM/0.1.98/ruprim/) | [Manifest](../../crates/ruPRIM/Cargo.toml) |
| `ruda-fft` | `0.1.98` | `rufft` | [rustdoc](https://docs.rs/ruda-fft/0.1.98/rufft/) | [Manifest](../../crates/ruFFT/Cargo.toml) |
| `ruRAND` | `0.1.98` | `rurand` | [rustdoc](https://docs.rs/ruRAND/0.1.98/rurand/) | [Manifest](../../crates/ruRAND/Cargo.toml) |
| `ruSPARSE` | `0.1.98` | `rusparse` | [rustdoc](https://docs.rs/ruSPARSE/0.1.98/rusparse/) | [Manifest](../../crates/ruSPARSE/Cargo.toml) |
| `ruTENSOR` | `0.1.98` | `rutensor` | [rustdoc](https://docs.rs/ruTENSOR/0.1.98/rutensor/) | [Manifest](../../crates/ruTENSOR/Cargo.toml) |
| `ruCCL` | `0.21.78` | `ruccl` | [rustdoc](https://docs.rs/ruCCL/0.21.78/ruccl/) | [Manifest](../../crates/ruCCL/Cargo.toml) |
| `ruda-solver` | `0.1.98` | `rusolver` | [rustdoc](https://docs.rs/ruda-solver/0.1.98/rusolver/) | [Manifest](../../crates/ruSOLVER/Cargo.toml) |
| `ruintegrate` | `0.1.98` | `ruintegrate` | [rustdoc](https://docs.rs/ruintegrate/0.1.98/ruintegrate/) | [Manifest](../../crates/ruINTEGRATE/Cargo.toml) |
| `rublas-host` | `0.1.98` | `rublas_host` | [rustdoc](https://docs.rs/rublas-host/0.1.98/rublas_host/) | [Manifest](../../crates/ruBLAS/host/Cargo.toml) |
| `ruDNN-host` | `0.1.98` | `rudnn_host` | [rustdoc](https://docs.rs/ruDNN-host/0.1.98/rudnn_host/) | [Manifest](../../crates/ruDNN/host/Cargo.toml) |
| `ruPRIM-host` | `0.1.98` | `ruprim_host` | [rustdoc](https://docs.rs/ruPRIM-host/0.1.98/ruprim_host/) | [Manifest](../../crates/ruPRIM/host/Cargo.toml) |
| `ruFFT-host` | `0.1.98` | `rufft_host` | [rustdoc](https://docs.rs/ruFFT-host/0.1.98/rufft_host/) | [Manifest](../../crates/ruFFT/host/Cargo.toml) |
| `ruRAND-host` | `0.1.98` | `rurand_host` | [rustdoc](https://docs.rs/ruRAND-host/0.1.98/rurand_host/) | [Manifest](../../crates/ruRAND/host/Cargo.toml) |
| `ruda-tensor` | `0.21.71` | `ruda_tensor` | [rustdoc](https://docs.rs/ruda-tensor/0.21.71/ruda_tensor/) | [Manifest](../../crates/ruda-tensor/Cargo.toml) |
| `ruda-tensor-config` | `0.21.8` | `ruda_tensor_config` | [rustdoc](https://docs.rs/ruda-tensor-config/0.21.8/ruda_tensor_config/) | [Manifest](../../crates/ruda-tensor-config/Cargo.toml) |
| `ruda-tensor-device` | `0.21.71` | `ruda_tensor_device` | [rustdoc](https://docs.rs/ruda-tensor-device/0.21.71/ruda_tensor_device/) | [Manifest](../../crates/ruda-tensor-device/Cargo.toml) |
| `ruda-tensor-host` | `0.21.68` | `ruda_tensor_host` | [rustdoc](https://docs.rs/ruda-tensor-host/0.21.68/ruda_tensor_host/) | [Manifest](../../crates/ruda-tensor-host/Cargo.toml) |
| `ruda-tensor-wgpu` | `0.21.70` | `ruda_tensor_wgpu` | [rustdoc](https://docs.rs/ruda-tensor-wgpu/0.21.70/ruda_tensor_wgpu/) | [Manifest](../../crates/ruda-tensor-wgpu/Cargo.toml) |
| `ruda-tensor-rocm` | `0.21.70` | `ruda_tensor_rocm` | [rustdoc](https://docs.rs/ruda-tensor-rocm/0.21.70/ruda_tensor_rocm/) | [Manifest](../../crates/ruda-tensor-rocm/Cargo.toml) |
| `ruda-tensor-tch` | `0.21.74` | `ruda_tensor_tch` | [rustdoc](https://docs.rs/ruda-tensor-tch/0.21.74/ruda_tensor_tch/) | [Manifest](../../crates/ruda-tensor-tch/Cargo.toml) |
| `ruda-autodiff` | `0.21.56` | `ruda_autodiff` | [rustdoc](https://docs.rs/ruda-autodiff/0.21.56/ruda_autodiff/) | [Manifest](../../crates/ruda-autodiff/Cargo.toml) |
| `ruda-fusion` | `0.21.70` | `ruda_fusion` | [rustdoc](https://docs.rs/ruda-fusion/0.21.70/ruda_fusion/) | [Manifest](../../crates/ruda-fusion/Cargo.toml) |
| `ruda-tensor-router` | `0.21.63` | `ruda_tensor_router` | [rustdoc](https://docs.rs/ruda-tensor-router/0.21.63/ruda_tensor_router/) | [Manifest](../../crates/ruda-tensor-router/Cargo.toml) |
| `ruda-tensor-remote` | `0.21.51` | `ruda_tensor_remote` | [rustdoc](https://docs.rs/ruda-tensor-remote/0.21.51/ruda_tensor_remote/) | [Manifest](../../crates/ruda-tensor-remote/Cargo.toml) |
| `ruda-communication` | `0.21.68` | `ruda_communication` | [rustdoc](https://docs.rs/ruda-communication/0.21.68/ruda_communication/) | [Manifest](../../crates/ruda-communication/Cargo.toml) |
| `ruda-model` | `0.21.60` | `ruda_model` | [rustdoc](https://docs.rs/ruda-model/0.21.60/ruda_model/) | [Manifest](../../crates/ruda-model/Cargo.toml) |
| `ruda-model-macros` | `0.21.1` | `ruda_model_macros` | [rustdoc](https://docs.rs/ruda-model-macros/0.21.1/ruda_model_macros/) | [Manifest](../../crates/ruda-model-macros/Cargo.toml) |
| `ruda-model-codegen` | `0.21.1` | `ruda_model_codegen` | [rustdoc](https://docs.rs/ruda-model-codegen/0.21.1/ruda_model_codegen/) | [Manifest](../../crates/ruda-model-codegen/Cargo.toml) |
| `ruda-nn` | `0.21.69` | `ruda_nn` | [rustdoc](https://docs.rs/ruda-nn/0.21.69/ruda_nn/) | [Manifest](../../crates/ruda-nn/Cargo.toml) |
| `ruda-optim` | `0.21.67` | `ruda_optim` | [rustdoc](https://docs.rs/ruda-optim/0.21.67/ruda_optim/) | [Manifest](../../crates/ruda-optim/Cargo.toml) |
| `ruda-dataset` | `0.21.17` | `ruda_dataset` | [rustdoc](https://docs.rs/ruda-dataset/0.21.17/ruda_dataset/) | [Manifest](../../crates/ruda-dataset/Cargo.toml) |
| `ruda-io` | `0.21.3` | `ruda_io` | [rustdoc](https://docs.rs/ruda-io/0.21.3/ruda_io/) | [Manifest](../../crates/ruda-io/Cargo.toml) |
| `ruda-store` | `0.21.59` | `ruda_store` | [rustdoc](https://docs.rs/ruda-store/0.21.59/ruda_store/) | [Manifest](../../crates/ruda-store/Cargo.toml) |
| `ruda-llm` | `0.1.98` | `rullm` | [rustdoc](https://docs.rs/ruda-llm/0.1.98/rullm/) | [Manifest](../../crates/ruLLM/Cargo.toml) |
| `ruda-torch-native` | `0.1.0` | `ruda_torch_native` | [Source](../../crates/ruda-torch/src/lib.rs) | [Manifest](../../crates/ruda-torch/Cargo.toml) |
| `ruda-test-runtime` | `0.1.98` | `ruda_test_runtime` | [rustdoc](https://docs.rs/ruda-test-runtime/0.1.98/ruda_test_runtime/) | [Manifest](../../crates/ruda-test-runtime/Cargo.toml) |
| `ruda-test-utils` | `0.1.98` | `ruda_test_utils` | [rustdoc](https://docs.rs/ruda-test-utils/0.1.98/ruda_test_utils/) | [Manifest](../../crates/ruda-test-utils/Cargo.toml) |
| `ruda-facade-consumer` | `0.0.0` | `ruda_facade_consumer` | [Source](../../crates/ruda/tests/consumer/src/lib.rs) | [Manifest](../../crates/ruda/tests/consumer/Cargo.toml) |
| `ruda-store-pytorch-tests` | `0.21.0` | `ruda_store_pytorch_tests` | [rustdoc](https://docs.rs/ruda-store-pytorch-tests/0.21.0/ruda_store_pytorch_tests/) | [Manifest](../../crates/ruda-store/pytorch-tests/Cargo.toml) |
| `ruda-store-safetensors-tests` | `0.21.0` | `ruda_store_safetensors_tests` | [rustdoc](https://docs.rs/ruda-store-safetensors-tests/0.21.0/ruda_store_safetensors_tests/) | [Manifest](../../crates/ruda-store/safetensors-tests/Cargo.toml) |

```sh
cargo doc --locked --no-deps -p ruda --features kernel
cargo doc --locked --no-deps -p ruda-tensor --features api
```
