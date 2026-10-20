# Справочник Rust API

[Документация](README.md) · [Python API](python-api-signatures.md)

Выбирайте crate и версию, используемые приложением. Cargo features определяют доступные модули; опубликованный rustdoc не гарантирует поддержку всех features и бэкендов.

Версии в таблице соответствуют манифестам документируемого снимка `main`. Версионный rustdoc описывает опубликованный пакет, а не последующие API только из `main`; см. [индекс исходников/API](../en/api-reference.md). `ruda-triton`, как и native-компонент PyTorch, собирается из исходников.

| Пакет | Версия | Импорт Rust | API | Конфигурация Cargo |
| --- | --- | --- | --- | --- |
| `ruda` | `0.1.46` | `ruda` | [rustdoc](https://docs.rs/ruda/0.1.46/ruda/) | [Конфигурация Cargo](../../crates/ruda/Cargo.toml) |
| `ruda-core` | `0.1.10` | `ruda_core` | [rustdoc](https://docs.rs/ruda-core/0.1.10/ruda_core/) | [Конфигурация Cargo](../../crates/ruda-core/Cargo.toml) |
| `ruda-compiler` | `0.1.17` | `ruda_compiler` | [rustdoc](https://docs.rs/ruda-compiler/0.1.17/ruda_compiler/) | [Конфигурация Cargo](../../crates/ruda-compiler/Cargo.toml) |
| `ruda-triton` | `0.1.0` | `ruda_triton` | [Source](../../crates/ruda-triton/src/lib.rs) | [Manifest](../../crates/ruda-triton/Cargo.toml) |
| `ruda-kernel` | `0.1.45` | `ruda_kernel` | [rustdoc](https://docs.rs/ruda-kernel/0.1.45/ruda_kernel/) | [Конфигурация Cargo](../../crates/ruda-kernel/Cargo.toml) |
| `ruda-runtime` | `0.1.30` | `ruda_runtime` | [rustdoc](https://docs.rs/ruda-runtime/0.1.30/ruda_runtime/) | [Конфигурация Cargo](../../crates/ruda-runtime/Cargo.toml) |
| `ruda-kernel-macros` | `0.1.11` | `ruda_kernel_macros` | [rustdoc](https://docs.rs/ruda-kernel-macros/0.1.11/ruda_kernel_macros/) | [Конфигурация Cargo](../../crates/ruda-kernel-macros/Cargo.toml) |
| `ruda-ir-macros` | `0.1.2` | `ruda_ir_macros` | [rustdoc](https://docs.rs/ruda-ir-macros/0.1.2/ruda_ir_macros/) | [Конфигурация Cargo](../../crates/ruda-ir-macros/Cargo.toml) |
| `ruda-driver-cuda` | `0.1.98` | `ruda_driver_cuda` | [rustdoc](https://docs.rs/ruda-driver-cuda/0.1.98/ruda_driver_cuda/) | [Конфигурация Cargo](../../crates/ruda-driver-cuda/Cargo.toml) |
| `ruda-driver-hip` | `0.1.98` | `ruda_driver_hip` | [rustdoc](https://docs.rs/ruda-driver-hip/0.1.98/ruda_driver_hip/) | [Конфигурация Cargo](../../crates/ruda-driver-hip/Cargo.toml) |
| `ruda-hip-sys` | `7.14.6085000` | `ruda_hip_sys` | [rustdoc](https://docs.rs/ruda-hip-sys/7.14.6085000/ruda_hip_sys/) | [Конфигурация Cargo](../../crates/ruda-hip-sys/Cargo.toml) |
| `ruda-driver-wgpu` | `0.1.98` | `ruda_driver_wgpu` | [rustdoc](https://docs.rs/ruda-driver-wgpu/0.1.98/ruda_driver_wgpu/) | [Конфигурация Cargo](../../crates/ruda-driver-wgpu/Cargo.toml) |
| `ruda-driver-cpu` | `0.1.98` | `ruda_driver_cpu` | [rustdoc](https://docs.rs/ruda-driver-cpu/0.1.98/ruda_driver_cpu/) | [Конфигурация Cargo](../../crates/ruda-driver-cpu/Cargo.toml) |
| `ruda-driver-cann` | `0.1.98` | `ruda_driver_cann` | [rustdoc](https://docs.rs/ruda-driver-cann/0.1.98/ruda_driver_cann/) | [Конфигурация Cargo](../../crates/ruda-driver-cann/Cargo.toml) |
| `ruda-ascend-kernels` | `0.1.0` | `ruda_ascend_kernels` | [rustdoc](https://docs.rs/ruda-ascend-kernels/0.1.0/ruda_ascend_kernels/) | [Конфигурация Cargo](../../crates/ruda-ascend-kernels/Cargo.toml) |
| `rublas` | `0.1.98` | `rublas` | [rustdoc](https://docs.rs/rublas/0.1.98/rublas/) | [Конфигурация Cargo](../../crates/ruBLAS/Cargo.toml) |
| `ruDNN` | `0.1.98` | `rudnn` | [rustdoc](https://docs.rs/ruDNN/0.1.98/rudnn/) | [Конфигурация Cargo](../../crates/ruDNN/Cargo.toml) |
| `ruPRIM` | `0.1.98` | `ruprim` | [rustdoc](https://docs.rs/ruPRIM/0.1.98/ruprim/) | [Конфигурация Cargo](../../crates/ruPRIM/Cargo.toml) |
| `ruda-fft` | `0.1.98` | `rufft` | [rustdoc](https://docs.rs/ruda-fft/0.1.98/rufft/) | [Конфигурация Cargo](../../crates/ruFFT/Cargo.toml) |
| `ruRAND` | `0.1.98` | `rurand` | [rustdoc](https://docs.rs/ruRAND/0.1.98/rurand/) | [Конфигурация Cargo](../../crates/ruRAND/Cargo.toml) |
| `ruSPARSE` | `0.1.98` | `rusparse` | [rustdoc](https://docs.rs/ruSPARSE/0.1.98/rusparse/) | [Конфигурация Cargo](../../crates/ruSPARSE/Cargo.toml) |
| `ruTENSOR` | `0.1.98` | `rutensor` | [rustdoc](https://docs.rs/ruTENSOR/0.1.98/rutensor/) | [Конфигурация Cargo](../../crates/ruTENSOR/Cargo.toml) |
| `ruCCL` | `0.21.78` | `ruccl` | [rustdoc](https://docs.rs/ruCCL/0.21.78/ruccl/) | [Конфигурация Cargo](../../crates/ruCCL/Cargo.toml) |
| `ruda-solver` | `0.1.98` | `rusolver` | [rustdoc](https://docs.rs/ruda-solver/0.1.98/rusolver/) | [Конфигурация Cargo](../../crates/ruSOLVER/Cargo.toml) |
| `ruintegrate` | `0.1.98` | `ruintegrate` | [rustdoc](https://docs.rs/ruintegrate/0.1.98/ruintegrate/) | [Конфигурация Cargo](../../crates/ruINTEGRATE/Cargo.toml) |
| `rublas-host` | `0.1.98` | `rublas_host` | [rustdoc](https://docs.rs/rublas-host/0.1.98/rublas_host/) | [Конфигурация Cargo](../../crates/ruBLAS/host/Cargo.toml) |
| `ruDNN-host` | `0.1.98` | `rudnn_host` | [rustdoc](https://docs.rs/ruDNN-host/0.1.98/rudnn_host/) | [Конфигурация Cargo](../../crates/ruDNN/host/Cargo.toml) |
| `ruPRIM-host` | `0.1.98` | `ruprim_host` | [rustdoc](https://docs.rs/ruPRIM-host/0.1.98/ruprim_host/) | [Конфигурация Cargo](../../crates/ruPRIM/host/Cargo.toml) |
| `ruFFT-host` | `0.1.98` | `rufft_host` | [rustdoc](https://docs.rs/ruFFT-host/0.1.98/rufft_host/) | [Конфигурация Cargo](../../crates/ruFFT/host/Cargo.toml) |
| `ruRAND-host` | `0.1.98` | `rurand_host` | [rustdoc](https://docs.rs/ruRAND-host/0.1.98/rurand_host/) | [Конфигурация Cargo](../../crates/ruRAND/host/Cargo.toml) |
| `ruda-tensor` | `0.21.71` | `ruda_tensor` | [rustdoc](https://docs.rs/ruda-tensor/0.21.71/ruda_tensor/) | [Конфигурация Cargo](../../crates/ruda-tensor/Cargo.toml) |
| `ruda-tensor-config` | `0.21.8` | `ruda_tensor_config` | [rustdoc](https://docs.rs/ruda-tensor-config/0.21.8/ruda_tensor_config/) | [Конфигурация Cargo](../../crates/ruda-tensor-config/Cargo.toml) |
| `ruda-tensor-device` | `0.21.71` | `ruda_tensor_device` | [rustdoc](https://docs.rs/ruda-tensor-device/0.21.71/ruda_tensor_device/) | [Конфигурация Cargo](../../crates/ruda-tensor-device/Cargo.toml) |
| `ruda-tensor-host` | `0.21.68` | `ruda_tensor_host` | [rustdoc](https://docs.rs/ruda-tensor-host/0.21.68/ruda_tensor_host/) | [Конфигурация Cargo](../../crates/ruda-tensor-host/Cargo.toml) |
| `ruda-tensor-wgpu` | `0.21.70` | `ruda_tensor_wgpu` | [rustdoc](https://docs.rs/ruda-tensor-wgpu/0.21.70/ruda_tensor_wgpu/) | [Конфигурация Cargo](../../crates/ruda-tensor-wgpu/Cargo.toml) |
| `ruda-tensor-rocm` | `0.21.70` | `ruda_tensor_rocm` | [rustdoc](https://docs.rs/ruda-tensor-rocm/0.21.70/ruda_tensor_rocm/) | [Конфигурация Cargo](../../crates/ruda-tensor-rocm/Cargo.toml) |
| `ruda-tensor-tch` | `0.21.74` | `ruda_tensor_tch` | [rustdoc](https://docs.rs/ruda-tensor-tch/0.21.74/ruda_tensor_tch/) | [Конфигурация Cargo](../../crates/ruda-tensor-tch/Cargo.toml) |
| `ruda-autodiff` | `0.21.56` | `ruda_autodiff` | [rustdoc](https://docs.rs/ruda-autodiff/0.21.56/ruda_autodiff/) | [Конфигурация Cargo](../../crates/ruda-autodiff/Cargo.toml) |
| `ruda-fusion` | `0.21.70` | `ruda_fusion` | [rustdoc](https://docs.rs/ruda-fusion/0.21.70/ruda_fusion/) | [Конфигурация Cargo](../../crates/ruda-fusion/Cargo.toml) |
| `ruda-tensor-router` | `0.21.63` | `ruda_tensor_router` | [rustdoc](https://docs.rs/ruda-tensor-router/0.21.63/ruda_tensor_router/) | [Конфигурация Cargo](../../crates/ruda-tensor-router/Cargo.toml) |
| `ruda-tensor-remote` | `0.21.51` | `ruda_tensor_remote` | [rustdoc](https://docs.rs/ruda-tensor-remote/0.21.51/ruda_tensor_remote/) | [Конфигурация Cargo](../../crates/ruda-tensor-remote/Cargo.toml) |
| `ruda-communication` | `0.21.68` | `ruda_communication` | [rustdoc](https://docs.rs/ruda-communication/0.21.68/ruda_communication/) | [Конфигурация Cargo](../../crates/ruda-communication/Cargo.toml) |
| `ruda-model` | `0.21.60` | `ruda_model` | [rustdoc](https://docs.rs/ruda-model/0.21.60/ruda_model/) | [Конфигурация Cargo](../../crates/ruda-model/Cargo.toml) |
| `ruda-model-macros` | `0.21.1` | `ruda_model_macros` | [rustdoc](https://docs.rs/ruda-model-macros/0.21.1/ruda_model_macros/) | [Конфигурация Cargo](../../crates/ruda-model-macros/Cargo.toml) |
| `ruda-model-codegen` | `0.21.1` | `ruda_model_codegen` | [rustdoc](https://docs.rs/ruda-model-codegen/0.21.1/ruda_model_codegen/) | [Конфигурация Cargo](../../crates/ruda-model-codegen/Cargo.toml) |
| `ruda-nn` | `0.21.69` | `ruda_nn` | [rustdoc](https://docs.rs/ruda-nn/0.21.69/ruda_nn/) | [Конфигурация Cargo](../../crates/ruda-nn/Cargo.toml) |
| `ruda-optim` | `0.21.67` | `ruda_optim` | [rustdoc](https://docs.rs/ruda-optim/0.21.67/ruda_optim/) | [Конфигурация Cargo](../../crates/ruda-optim/Cargo.toml) |
| `ruda-dataset` | `0.21.17` | `ruda_dataset` | [rustdoc](https://docs.rs/ruda-dataset/0.21.17/ruda_dataset/) | [Конфигурация Cargo](../../crates/ruda-dataset/Cargo.toml) |
| `ruda-io` | `0.21.3` | `ruda_io` | [rustdoc](https://docs.rs/ruda-io/0.21.3/ruda_io/) | [Конфигурация Cargo](../../crates/ruda-io/Cargo.toml) |
| `ruda-store` | `0.21.59` | `ruda_store` | [rustdoc](https://docs.rs/ruda-store/0.21.59/ruda_store/) | [Конфигурация Cargo](../../crates/ruda-store/Cargo.toml) |
| `ruda-llm` | `0.1.98` | `rullm` | [rustdoc](https://docs.rs/ruda-llm/0.1.98/rullm/) | [Конфигурация Cargo](../../crates/ruLLM/Cargo.toml) |
| `ruda-torch-native` | `0.1.0` | `ruda_torch_native` | [Исходный код](../../crates/ruda-torch/src/lib.rs) | [Конфигурация Cargo](../../crates/ruda-torch/Cargo.toml) |
| `ruda-test-runtime` | `0.1.98` | `ruda_test_runtime` | [rustdoc](https://docs.rs/ruda-test-runtime/0.1.98/ruda_test_runtime/) | [Конфигурация Cargo](../../crates/ruda-test-runtime/Cargo.toml) |
| `ruda-test-utils` | `0.1.98` | `ruda_test_utils` | [rustdoc](https://docs.rs/ruda-test-utils/0.1.98/ruda_test_utils/) | [Конфигурация Cargo](../../crates/ruda-test-utils/Cargo.toml) |
| `ruda-facade-consumer` | `0.0.0` | `ruda_facade_consumer` | [Исходный код](../../crates/ruda/tests/consumer/src/lib.rs) | [Конфигурация Cargo](../../crates/ruda/tests/consumer/Cargo.toml) |
| `ruda-store-pytorch-tests` | `0.21.0` | `ruda_store_pytorch_tests` | [rustdoc](https://docs.rs/ruda-store-pytorch-tests/0.21.0/ruda_store_pytorch_tests/) | [Конфигурация Cargo](../../crates/ruda-store/pytorch-tests/Cargo.toml) |
| `ruda-store-safetensors-tests` | `0.21.0` | `ruda_store_safetensors_tests` | [rustdoc](https://docs.rs/ruda-store-safetensors-tests/0.21.0/ruda_store_safetensors_tests/) | [Конфигурация Cargo](../../crates/ruda-store/safetensors-tests/Cargo.toml) |

```sh
cargo doc --locked --no-deps -p ruda --features kernel
cargo doc --locked --no-deps -p ruda-tensor --features api
```
