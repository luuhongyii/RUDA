# Rust-API-Referenz

[Dokumentation](README.md) · [Python API](python-api-signatures.md)

Wählen Sie das tatsächlich verwendete Crate und die Version. Cargo-Features bestimmen verfügbare Module; veröffentlichte rustdoc-Seiten garantieren nicht die Unterstützung aller Features und Backends.

Die Tabelle folgt den Manifesten des dokumentierten `main`-Snapshots. Versionierte rustdoc-Seiten beschreiben das veröffentlichte Paket, nicht spätere APIs nur auf `main`; siehe dafür den [Quellcode-/API-Index](../en/api-reference.md). `ruda-triton` und die native PyTorch-Komponente werden aus Quellcode gebaut.

| Paket | Version | Rust-Import | API | Cargo-Konfiguration |
| --- | --- | --- | --- | --- |
| `ruda` | `0.1.46` | `ruda` | [rustdoc](https://docs.rs/ruda/0.1.46/ruda/) | [Cargo-Konfiguration](../../crates/ruda/Cargo.toml) |
| `ruda-core` | `0.1.10` | `ruda_core` | [rustdoc](https://docs.rs/ruda-core/0.1.10/ruda_core/) | [Cargo-Konfiguration](../../crates/ruda-core/Cargo.toml) |
| `ruda-compiler` | `0.1.17` | `ruda_compiler` | [rustdoc](https://docs.rs/ruda-compiler/0.1.17/ruda_compiler/) | [Cargo-Konfiguration](../../crates/ruda-compiler/Cargo.toml) |
| `ruda-triton` | `0.1.0` | `ruda_triton` | [Source](../../crates/ruda-triton/src/lib.rs) | [Manifest](../../crates/ruda-triton/Cargo.toml) |
| `ruda-kernel` | `0.1.45` | `ruda_kernel` | [rustdoc](https://docs.rs/ruda-kernel/0.1.45/ruda_kernel/) | [Cargo-Konfiguration](../../crates/ruda-kernel/Cargo.toml) |
| `ruda-runtime` | `0.1.30` | `ruda_runtime` | [rustdoc](https://docs.rs/ruda-runtime/0.1.30/ruda_runtime/) | [Cargo-Konfiguration](../../crates/ruda-runtime/Cargo.toml) |
| `ruda-kernel-macros` | `0.1.11` | `ruda_kernel_macros` | [rustdoc](https://docs.rs/ruda-kernel-macros/0.1.11/ruda_kernel_macros/) | [Cargo-Konfiguration](../../crates/ruda-kernel-macros/Cargo.toml) |
| `ruda-ir-macros` | `0.1.2` | `ruda_ir_macros` | [rustdoc](https://docs.rs/ruda-ir-macros/0.1.2/ruda_ir_macros/) | [Cargo-Konfiguration](../../crates/ruda-ir-macros/Cargo.toml) |
| `ruda-driver-cuda` | `0.1.98` | `ruda_driver_cuda` | [rustdoc](https://docs.rs/ruda-driver-cuda/0.1.98/ruda_driver_cuda/) | [Cargo-Konfiguration](../../crates/ruda-driver-cuda/Cargo.toml) |
| `ruda-driver-hip` | `0.1.98` | `ruda_driver_hip` | [rustdoc](https://docs.rs/ruda-driver-hip/0.1.98/ruda_driver_hip/) | [Cargo-Konfiguration](../../crates/ruda-driver-hip/Cargo.toml) |
| `ruda-hip-sys` | `7.14.6085000` | `ruda_hip_sys` | [rustdoc](https://docs.rs/ruda-hip-sys/7.14.6085000/ruda_hip_sys/) | [Cargo-Konfiguration](../../crates/ruda-hip-sys/Cargo.toml) |
| `ruda-driver-wgpu` | `0.1.98` | `ruda_driver_wgpu` | [rustdoc](https://docs.rs/ruda-driver-wgpu/0.1.98/ruda_driver_wgpu/) | [Cargo-Konfiguration](../../crates/ruda-driver-wgpu/Cargo.toml) |
| `ruda-driver-cpu` | `0.1.98` | `ruda_driver_cpu` | [rustdoc](https://docs.rs/ruda-driver-cpu/0.1.98/ruda_driver_cpu/) | [Cargo-Konfiguration](../../crates/ruda-driver-cpu/Cargo.toml) |
| `ruda-driver-cann` | `0.1.98` | `ruda_driver_cann` | [rustdoc](https://docs.rs/ruda-driver-cann/0.1.98/ruda_driver_cann/) | [Cargo-Konfiguration](../../crates/ruda-driver-cann/Cargo.toml) |
| `ruda-ascend-kernels` | `0.1.0` | `ruda_ascend_kernels` | [rustdoc](https://docs.rs/ruda-ascend-kernels/0.1.0/ruda_ascend_kernels/) | [Cargo-Konfiguration](../../crates/ruda-ascend-kernels/Cargo.toml) |
| `rublas` | `0.1.98` | `rublas` | [rustdoc](https://docs.rs/rublas/0.1.98/rublas/) | [Cargo-Konfiguration](../../crates/ruBLAS/Cargo.toml) |
| `ruDNN` | `0.1.98` | `rudnn` | [rustdoc](https://docs.rs/ruDNN/0.1.98/rudnn/) | [Cargo-Konfiguration](../../crates/ruDNN/Cargo.toml) |
| `ruPRIM` | `0.1.98` | `ruprim` | [rustdoc](https://docs.rs/ruPRIM/0.1.98/ruprim/) | [Cargo-Konfiguration](../../crates/ruPRIM/Cargo.toml) |
| `ruda-fft` | `0.1.98` | `rufft` | [rustdoc](https://docs.rs/ruda-fft/0.1.98/rufft/) | [Cargo-Konfiguration](../../crates/ruFFT/Cargo.toml) |
| `ruRAND` | `0.1.98` | `rurand` | [rustdoc](https://docs.rs/ruRAND/0.1.98/rurand/) | [Cargo-Konfiguration](../../crates/ruRAND/Cargo.toml) |
| `ruSPARSE` | `0.1.98` | `rusparse` | [rustdoc](https://docs.rs/ruSPARSE/0.1.98/rusparse/) | [Cargo-Konfiguration](../../crates/ruSPARSE/Cargo.toml) |
| `ruTENSOR` | `0.1.98` | `rutensor` | [rustdoc](https://docs.rs/ruTENSOR/0.1.98/rutensor/) | [Cargo-Konfiguration](../../crates/ruTENSOR/Cargo.toml) |
| `ruCCL` | `0.21.78` | `ruccl` | [rustdoc](https://docs.rs/ruCCL/0.21.78/ruccl/) | [Cargo-Konfiguration](../../crates/ruCCL/Cargo.toml) |
| `ruda-solver` | `0.1.98` | `rusolver` | [rustdoc](https://docs.rs/ruda-solver/0.1.98/rusolver/) | [Cargo-Konfiguration](../../crates/ruSOLVER/Cargo.toml) |
| `ruintegrate` | `0.1.98` | `ruintegrate` | [rustdoc](https://docs.rs/ruintegrate/0.1.98/ruintegrate/) | [Cargo-Konfiguration](../../crates/ruINTEGRATE/Cargo.toml) |
| `rublas-host` | `0.1.98` | `rublas_host` | [rustdoc](https://docs.rs/rublas-host/0.1.98/rublas_host/) | [Cargo-Konfiguration](../../crates/ruBLAS/host/Cargo.toml) |
| `ruDNN-host` | `0.1.98` | `rudnn_host` | [rustdoc](https://docs.rs/ruDNN-host/0.1.98/rudnn_host/) | [Cargo-Konfiguration](../../crates/ruDNN/host/Cargo.toml) |
| `ruPRIM-host` | `0.1.98` | `ruprim_host` | [rustdoc](https://docs.rs/ruPRIM-host/0.1.98/ruprim_host/) | [Cargo-Konfiguration](../../crates/ruPRIM/host/Cargo.toml) |
| `ruFFT-host` | `0.1.98` | `rufft_host` | [rustdoc](https://docs.rs/ruFFT-host/0.1.98/rufft_host/) | [Cargo-Konfiguration](../../crates/ruFFT/host/Cargo.toml) |
| `ruRAND-host` | `0.1.98` | `rurand_host` | [rustdoc](https://docs.rs/ruRAND-host/0.1.98/rurand_host/) | [Cargo-Konfiguration](../../crates/ruRAND/host/Cargo.toml) |
| `ruda-tensor` | `0.21.71` | `ruda_tensor` | [rustdoc](https://docs.rs/ruda-tensor/0.21.71/ruda_tensor/) | [Cargo-Konfiguration](../../crates/ruda-tensor/Cargo.toml) |
| `ruda-tensor-config` | `0.21.8` | `ruda_tensor_config` | [rustdoc](https://docs.rs/ruda-tensor-config/0.21.8/ruda_tensor_config/) | [Cargo-Konfiguration](../../crates/ruda-tensor-config/Cargo.toml) |
| `ruda-tensor-device` | `0.21.71` | `ruda_tensor_device` | [rustdoc](https://docs.rs/ruda-tensor-device/0.21.71/ruda_tensor_device/) | [Cargo-Konfiguration](../../crates/ruda-tensor-device/Cargo.toml) |
| `ruda-tensor-host` | `0.21.68` | `ruda_tensor_host` | [rustdoc](https://docs.rs/ruda-tensor-host/0.21.68/ruda_tensor_host/) | [Cargo-Konfiguration](../../crates/ruda-tensor-host/Cargo.toml) |
| `ruda-tensor-wgpu` | `0.21.70` | `ruda_tensor_wgpu` | [rustdoc](https://docs.rs/ruda-tensor-wgpu/0.21.70/ruda_tensor_wgpu/) | [Cargo-Konfiguration](../../crates/ruda-tensor-wgpu/Cargo.toml) |
| `ruda-tensor-rocm` | `0.21.70` | `ruda_tensor_rocm` | [rustdoc](https://docs.rs/ruda-tensor-rocm/0.21.70/ruda_tensor_rocm/) | [Cargo-Konfiguration](../../crates/ruda-tensor-rocm/Cargo.toml) |
| `ruda-tensor-tch` | `0.21.74` | `ruda_tensor_tch` | [rustdoc](https://docs.rs/ruda-tensor-tch/0.21.74/ruda_tensor_tch/) | [Cargo-Konfiguration](../../crates/ruda-tensor-tch/Cargo.toml) |
| `ruda-autodiff` | `0.21.56` | `ruda_autodiff` | [rustdoc](https://docs.rs/ruda-autodiff/0.21.56/ruda_autodiff/) | [Cargo-Konfiguration](../../crates/ruda-autodiff/Cargo.toml) |
| `ruda-fusion` | `0.21.70` | `ruda_fusion` | [rustdoc](https://docs.rs/ruda-fusion/0.21.70/ruda_fusion/) | [Cargo-Konfiguration](../../crates/ruda-fusion/Cargo.toml) |
| `ruda-tensor-router` | `0.21.63` | `ruda_tensor_router` | [rustdoc](https://docs.rs/ruda-tensor-router/0.21.63/ruda_tensor_router/) | [Cargo-Konfiguration](../../crates/ruda-tensor-router/Cargo.toml) |
| `ruda-tensor-remote` | `0.21.51` | `ruda_tensor_remote` | [rustdoc](https://docs.rs/ruda-tensor-remote/0.21.51/ruda_tensor_remote/) | [Cargo-Konfiguration](../../crates/ruda-tensor-remote/Cargo.toml) |
| `ruda-communication` | `0.21.68` | `ruda_communication` | [rustdoc](https://docs.rs/ruda-communication/0.21.68/ruda_communication/) | [Cargo-Konfiguration](../../crates/ruda-communication/Cargo.toml) |
| `ruda-model` | `0.21.60` | `ruda_model` | [rustdoc](https://docs.rs/ruda-model/0.21.60/ruda_model/) | [Cargo-Konfiguration](../../crates/ruda-model/Cargo.toml) |
| `ruda-model-macros` | `0.21.1` | `ruda_model_macros` | [rustdoc](https://docs.rs/ruda-model-macros/0.21.1/ruda_model_macros/) | [Cargo-Konfiguration](../../crates/ruda-model-macros/Cargo.toml) |
| `ruda-model-codegen` | `0.21.1` | `ruda_model_codegen` | [rustdoc](https://docs.rs/ruda-model-codegen/0.21.1/ruda_model_codegen/) | [Cargo-Konfiguration](../../crates/ruda-model-codegen/Cargo.toml) |
| `ruda-nn` | `0.21.69` | `ruda_nn` | [rustdoc](https://docs.rs/ruda-nn/0.21.69/ruda_nn/) | [Cargo-Konfiguration](../../crates/ruda-nn/Cargo.toml) |
| `ruda-optim` | `0.21.67` | `ruda_optim` | [rustdoc](https://docs.rs/ruda-optim/0.21.67/ruda_optim/) | [Cargo-Konfiguration](../../crates/ruda-optim/Cargo.toml) |
| `ruda-dataset` | `0.21.17` | `ruda_dataset` | [rustdoc](https://docs.rs/ruda-dataset/0.21.17/ruda_dataset/) | [Cargo-Konfiguration](../../crates/ruda-dataset/Cargo.toml) |
| `ruda-io` | `0.21.3` | `ruda_io` | [rustdoc](https://docs.rs/ruda-io/0.21.3/ruda_io/) | [Cargo-Konfiguration](../../crates/ruda-io/Cargo.toml) |
| `ruda-store` | `0.21.59` | `ruda_store` | [rustdoc](https://docs.rs/ruda-store/0.21.59/ruda_store/) | [Cargo-Konfiguration](../../crates/ruda-store/Cargo.toml) |
| `ruda-llm` | `0.1.98` | `rullm` | [rustdoc](https://docs.rs/ruda-llm/0.1.98/rullm/) | [Cargo-Konfiguration](../../crates/ruLLM/Cargo.toml) |
| `ruda-torch-native` | `0.1.0` | `ruda_torch_native` | [Quellcode](../../crates/ruda-torch/src/lib.rs) | [Cargo-Konfiguration](../../crates/ruda-torch/Cargo.toml) |
| `ruda-test-runtime` | `0.1.98` | `ruda_test_runtime` | [rustdoc](https://docs.rs/ruda-test-runtime/0.1.98/ruda_test_runtime/) | [Cargo-Konfiguration](../../crates/ruda-test-runtime/Cargo.toml) |
| `ruda-test-utils` | `0.1.98` | `ruda_test_utils` | [rustdoc](https://docs.rs/ruda-test-utils/0.1.98/ruda_test_utils/) | [Cargo-Konfiguration](../../crates/ruda-test-utils/Cargo.toml) |
| `ruda-facade-consumer` | `0.0.0` | `ruda_facade_consumer` | [Quellcode](../../crates/ruda/tests/consumer/src/lib.rs) | [Cargo-Konfiguration](../../crates/ruda/tests/consumer/Cargo.toml) |
| `ruda-store-pytorch-tests` | `0.21.0` | `ruda_store_pytorch_tests` | [rustdoc](https://docs.rs/ruda-store-pytorch-tests/0.21.0/ruda_store_pytorch_tests/) | [Cargo-Konfiguration](../../crates/ruda-store/pytorch-tests/Cargo.toml) |
| `ruda-store-safetensors-tests` | `0.21.0` | `ruda_store_safetensors_tests` | [rustdoc](https://docs.rs/ruda-store-safetensors-tests/0.21.0/ruda_store_safetensors_tests/) | [Cargo-Konfiguration](../../crates/ruda-store/safetensors-tests/Cargo.toml) |

```sh
cargo doc --locked --no-deps -p ruda --features kernel
cargo doc --locked --no-deps -p ruda-tensor --features api
```
