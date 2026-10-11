# Rust API 参考

[文档首页](README.md) · [Python API](python-api-signatures.md)

选择应用实际使用的 crate 与版本。Cargo feature 决定可用模块；已发布的 rustdoc 不表示所有后端和 feature 都可用。

| 包名 | 版本 | Rust 导入名 | API | Cargo 配置 |
| --- | --- | --- | --- | --- |
| `ruda` | `0.1.34` | `ruda` | [rustdoc](https://docs.rs/ruda/0.1.34/ruda/) | [Cargo 配置](../../ruda/Cargo.toml) |
| `ruda-core` | `0.1.6` | `ruda_core` | [rustdoc](https://docs.rs/ruda-core/0.1.6/ruda_core/) | [Cargo 配置](../../ruda-core/Cargo.toml) |
| `ruda-compiler` | `0.1.10` | `ruda_compiler` | [rustdoc](https://docs.rs/ruda-compiler/0.1.10/ruda_compiler/) | [Cargo 配置](../../ruda-compiler/Cargo.toml) |
| `ruda-kernel` | `0.1.33` | `ruda_kernel` | [rustdoc](https://docs.rs/ruda-kernel/0.1.33/ruda_kernel/) | [Cargo 配置](../../ruda-kernel/Cargo.toml) |
| `ruda-runtime` | `0.1.23` | `ruda_runtime` | [rustdoc](https://docs.rs/ruda-runtime/0.1.23/ruda_runtime/) | [Cargo 配置](../../ruda-runtime/Cargo.toml) |
| `ruda-kernel-macros` | `0.1.7` | `ruda_kernel_macros` | [rustdoc](https://docs.rs/ruda-kernel-macros/0.1.7/ruda_kernel_macros/) | [Cargo 配置](../../ruda-kernel-macros/Cargo.toml) |
| `ruda-ir-macros` | `0.1.2` | `ruda_ir_macros` | [rustdoc](https://docs.rs/ruda-ir-macros/0.1.2/ruda_ir_macros/) | [Cargo 配置](../../ruda-ir-macros/Cargo.toml) |
| `ruda-driver-cuda` | `0.1.82` | `ruda_driver_cuda` | [rustdoc](https://docs.rs/ruda-driver-cuda/0.1.82/ruda_driver_cuda/) | [Cargo 配置](../../ruda-driver-cuda/Cargo.toml) |
| `ruda-driver-hip` | `0.1.82` | `ruda_driver_hip` | [rustdoc](https://docs.rs/ruda-driver-hip/0.1.82/ruda_driver_hip/) | [Cargo 配置](../../ruda-driver-hip/Cargo.toml) |
| `ruda-hip-sys` | `7.14.6085000` | `ruda_hip_sys` | [rustdoc](https://docs.rs/ruda-hip-sys/7.14.6085000/ruda_hip_sys/) | [Cargo 配置](../../ruda-hip-sys/Cargo.toml) |
| `ruda-driver-wgpu` | `0.1.82` | `ruda_driver_wgpu` | [rustdoc](https://docs.rs/ruda-driver-wgpu/0.1.82/ruda_driver_wgpu/) | [Cargo 配置](../../ruda-driver-wgpu/Cargo.toml) |
| `ruda-driver-cpu` | `0.1.82` | `ruda_driver_cpu` | [rustdoc](https://docs.rs/ruda-driver-cpu/0.1.82/ruda_driver_cpu/) | [Cargo 配置](../../ruda-driver-cpu/Cargo.toml) |
| `ruda-driver-cann` | `0.1.82` | `ruda_driver_cann` | [rustdoc](https://docs.rs/ruda-driver-cann/0.1.82/ruda_driver_cann/) | [Cargo 配置](../../ruda-driver-cann/Cargo.toml) |
| `ruda-ascend-kernels` | `0.1.0` | `ruda_ascend_kernels` | [rustdoc](https://docs.rs/ruda-ascend-kernels/0.1.0/ruda_ascend_kernels/) | [Cargo 配置](../../ruda-ascend-kernels/Cargo.toml) |
| `rublas` | `0.1.82` | `rublas` | [rustdoc](https://docs.rs/rublas/0.1.82/rublas/) | [Cargo 配置](../../ruBLAS/Cargo.toml) |
| `ruDNN` | `0.1.82` | `rudnn` | [rustdoc](https://docs.rs/ruDNN/0.1.82/rudnn/) | [Cargo 配置](../../ruDNN/Cargo.toml) |
| `ruPRIM` | `0.1.82` | `ruprim` | [rustdoc](https://docs.rs/ruPRIM/0.1.82/ruprim/) | [Cargo 配置](../../ruPRIM/Cargo.toml) |
| `ruda-fft` | `0.1.82` | `rufft` | [rustdoc](https://docs.rs/ruda-fft/0.1.82/rufft/) | [Cargo 配置](../../ruFFT/Cargo.toml) |
| `ruRAND` | `0.1.82` | `rurand` | [rustdoc](https://docs.rs/ruRAND/0.1.82/rurand/) | [Cargo 配置](../../ruRAND/Cargo.toml) |
| `ruSPARSE` | `0.1.82` | `rusparse` | [rustdoc](https://docs.rs/ruSPARSE/0.1.82/rusparse/) | [Cargo 配置](../../ruSPARSE/Cargo.toml) |
| `ruTENSOR` | `0.1.82` | `rutensor` | [rustdoc](https://docs.rs/ruTENSOR/0.1.82/rutensor/) | [Cargo 配置](../../ruTENSOR/Cargo.toml) |
| `ruCCL` | `0.21.62` | `ruccl` | [rustdoc](https://docs.rs/ruCCL/0.21.62/ruccl/) | [Cargo 配置](../../ruCCL/Cargo.toml) |
| `ruda-solver` | `0.1.82` | `rusolver` | [rustdoc](https://docs.rs/ruda-solver/0.1.82/rusolver/) | [Cargo 配置](../../ruSOLVER/Cargo.toml) |
| `ruintegrate` | `0.1.82` | `ruintegrate` | [rustdoc](https://docs.rs/ruintegrate/0.1.82/ruintegrate/) | [Cargo 配置](../../ruINTEGRATE/Cargo.toml) |
| `rublas-host` | `0.1.82` | `rublas_host` | [rustdoc](https://docs.rs/rublas-host/0.1.82/rublas_host/) | [Cargo 配置](../../ruBLAS/host/Cargo.toml) |
| `ruDNN-host` | `0.1.82` | `rudnn_host` | [rustdoc](https://docs.rs/ruDNN-host/0.1.82/rudnn_host/) | [Cargo 配置](../../ruDNN/host/Cargo.toml) |
| `ruPRIM-host` | `0.1.82` | `ruprim_host` | [rustdoc](https://docs.rs/ruPRIM-host/0.1.82/ruprim_host/) | [Cargo 配置](../../ruPRIM/host/Cargo.toml) |
| `ruFFT-host` | `0.1.82` | `rufft_host` | [rustdoc](https://docs.rs/ruFFT-host/0.1.82/rufft_host/) | [Cargo 配置](../../ruFFT/host/Cargo.toml) |
| `ruRAND-host` | `0.1.82` | `rurand_host` | [rustdoc](https://docs.rs/ruRAND-host/0.1.82/rurand_host/) | [Cargo 配置](../../ruRAND/host/Cargo.toml) |
| `ruda-tensor` | `0.21.57` | `ruda_tensor` | [rustdoc](https://docs.rs/ruda-tensor/0.21.57/ruda_tensor/) | [Cargo 配置](../../ruda-tensor/Cargo.toml) |
| `ruda-tensor-config` | `0.21.4` | `ruda_tensor_config` | [rustdoc](https://docs.rs/ruda-tensor-config/0.21.4/ruda_tensor_config/) | [Cargo 配置](../../ruda-tensor-config/Cargo.toml) |
| `ruda-tensor-device` | `0.21.57` | `ruda_tensor_device` | [rustdoc](https://docs.rs/ruda-tensor-device/0.21.57/ruda_tensor_device/) | [Cargo 配置](../../ruda-tensor-device/Cargo.toml) |
| `ruda-tensor-host` | `0.21.54` | `ruda_tensor_host` | [rustdoc](https://docs.rs/ruda-tensor-host/0.21.54/ruda_tensor_host/) | [Cargo 配置](../../ruda-tensor-host/Cargo.toml) |
| `ruda-tensor-wgpu` | `0.21.56` | `ruda_tensor_wgpu` | [rustdoc](https://docs.rs/ruda-tensor-wgpu/0.21.56/ruda_tensor_wgpu/) | [Cargo 配置](../../ruda-tensor-wgpu/Cargo.toml) |
| `ruda-tensor-rocm` | `0.21.56` | `ruda_tensor_rocm` | [rustdoc](https://docs.rs/ruda-tensor-rocm/0.21.56/ruda_tensor_rocm/) | [Cargo 配置](../../ruda-tensor-rocm/Cargo.toml) |
| `ruda-tensor-tch` | `0.21.70` | `ruda_tensor_tch` | [rustdoc](https://docs.rs/ruda-tensor-tch/0.21.70/ruda_tensor_tch/) | [Cargo 配置](../../ruda-tensor-tch/Cargo.toml) |
| `ruda-autodiff` | `0.21.51` | `ruda_autodiff` | [rustdoc](https://docs.rs/ruda-autodiff/0.21.51/ruda_autodiff/) | [Cargo 配置](../../ruda-autodiff/Cargo.toml) |
| `ruda-fusion` | `0.21.56` | `ruda_fusion` | [rustdoc](https://docs.rs/ruda-fusion/0.21.56/ruda_fusion/) | [Cargo 配置](../../ruda-fusion/Cargo.toml) |
| `ruda-tensor-router` | `0.21.47` | `ruda_tensor_router` | [rustdoc](https://docs.rs/ruda-tensor-router/0.21.47/ruda_tensor_router/) | [Cargo 配置](../../ruda-tensor-router/Cargo.toml) |
| `ruda-tensor-remote` | `0.21.46` | `ruda_tensor_remote` | [rustdoc](https://docs.rs/ruda-tensor-remote/0.21.46/ruda_tensor_remote/) | [Cargo 配置](../../ruda-tensor-remote/Cargo.toml) |
| `ruda-communication` | `0.21.54` | `ruda_communication` | [rustdoc](https://docs.rs/ruda-communication/0.21.54/ruda_communication/) | [Cargo 配置](../../ruda-communication/Cargo.toml) |
| `ruda-model` | `0.21.55` | `ruda_model` | [rustdoc](https://docs.rs/ruda-model/0.21.55/ruda_model/) | [Cargo 配置](../../ruda-model/Cargo.toml) |
| `ruda-model-macros` | `0.21.1` | `ruda_model_macros` | [rustdoc](https://docs.rs/ruda-model-macros/0.21.1/ruda_model_macros/) | [Cargo 配置](../../ruda-model-macros/Cargo.toml) |
| `ruda-model-codegen` | `0.21.1` | `ruda_model_codegen` | [rustdoc](https://docs.rs/ruda-model-codegen/0.21.1/ruda_model_codegen/) | [Cargo 配置](../../ruda-model-codegen/Cargo.toml) |
| `ruda-nn` | `0.21.64` | `ruda_nn` | [rustdoc](https://docs.rs/ruda-nn/0.21.64/ruda_nn/) | [Cargo 配置](../../ruda-nn/Cargo.toml) |
| `ruda-optim` | `0.21.62` | `ruda_optim` | [rustdoc](https://docs.rs/ruda-optim/0.21.62/ruda_optim/) | [Cargo 配置](../../ruda-optim/Cargo.toml) |
| `ruda-dataset` | `0.21.13` | `ruda_dataset` | [rustdoc](https://docs.rs/ruda-dataset/0.21.13/ruda_dataset/) | [Cargo 配置](../../ruda-dataset/Cargo.toml) |
| `ruda-io` | `0.21.3` | `ruda_io` | [rustdoc](https://docs.rs/ruda-io/0.21.3/ruda_io/) | [Cargo 配置](../../ruda-io/Cargo.toml) |
| `ruda-store` | `0.21.54` | `ruda_store` | [rustdoc](https://docs.rs/ruda-store/0.21.54/ruda_store/) | [Cargo 配置](../../ruda-store/Cargo.toml) |
| `ruda-llm` | `0.1.82` | `rullm` | [rustdoc](https://docs.rs/ruda-llm/0.1.82/rullm/) | [Cargo 配置](../../ruLLM/Cargo.toml) |
| `ruda-torch-native` | `0.1.0` | `ruda_torch_native` | [源码](../../ruda-torch/src/lib.rs) | [Cargo 配置](../../ruda-torch/Cargo.toml) |
| `ruda-test-runtime` | `0.1.82` | `ruda_test_runtime` | [rustdoc](https://docs.rs/ruda-test-runtime/0.1.82/ruda_test_runtime/) | [Cargo 配置](../../ruda-test-runtime/Cargo.toml) |
| `ruda-test-utils` | `0.1.82` | `ruda_test_utils` | [rustdoc](https://docs.rs/ruda-test-utils/0.1.82/ruda_test_utils/) | [Cargo 配置](../../ruda-test-utils/Cargo.toml) |
| `ruda-facade-consumer` | `0.0.0` | `ruda_facade_consumer` | [源码](../../ruda/tests/consumer/src/lib.rs) | [Cargo 配置](../../ruda/tests/consumer/Cargo.toml) |
| `ruda-store-pytorch-tests` | `0.21.0` | `ruda_store_pytorch_tests` | [rustdoc](https://docs.rs/ruda-store-pytorch-tests/0.21.0/ruda_store_pytorch_tests/) | [Cargo 配置](../../ruda-store/pytorch-tests/Cargo.toml) |
| `ruda-store-safetensors-tests` | `0.21.0` | `ruda_store_safetensors_tests` | [rustdoc](https://docs.rs/ruda-store-safetensors-tests/0.21.0/ruda_store_safetensors_tests/) | [Cargo 配置](../../ruda-store/safetensors-tests/Cargo.toml) |

```sh
cargo doc --locked --no-deps -p ruda --features kernel
cargo doc --locked --no-deps -p ruda-tensor --features api
```
