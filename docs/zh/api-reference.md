# 全栈 Crate 与 API 索引

[文档首页](README.md) · [English](../en/api-reference.md)

[Rust 方法与类型](rust-api.md) · [Python 签名](python-api-signatures.md) · [Python 并行训练](parallel-and-data.md)

## 按任务选择入口层

| 任务 | 首选 API | 指南 |
| --- | --- | --- |
| 编写 GPU Kernel | `ruda`、Kernel DSL、运行时客户端 | [编程指南](programming-guide.md) |
| 在 NVIDIA 上使用 Python／PyTorch | `ruda_torch`、`torch.device("ruda:0")` | [PyTorch 快速上手](native-pytorch-quickstart.md)与 [API 参考](native-pytorch-api.md) |
| 使用 typed tensor 与梯度 | `ruda_tensor::api::Tensor`、`Autodiff<B>` | [张量实用示例](tensor-recipes.md) |
| 选择张量执行目标 | `Host`、`Cuda`、`Rocm`、`Wgpu`、Router、Remote | [后端组合](backend-composition.md) |
| 调用领域算子 | ruBLAS、ruDNN、ruPRIM 等领域库 | [计算库](libraries/README.md) |
| 构建和训练模型 | `Module`、`ruda-nn`、`ruda-optim` | [训练指南](training.md) |
| 半精度存储与 FP32 更新 | `Module::to_dtype`、`Fp32MasterOptimizer`、`GradientsAccumulator::accumulate_with_dtype` | [混合精度训练](training.md#fp32-主参数与混合参数存储) |
| 保留混合存储与待累积梯度 | `TrainingRecord::capture_with_dtypes`、`restore_with_dtypes` | [训练状态](training.md#保存和恢复训练状态) |
| 同步副本或对集合通信求导 | `DataParallel`、`ruda_autodiff::collective` | [分布式训练](training.md#副本训练与可微分张量集合通信) |
| 加载样本或权重 | `Dataset`、`DataLoaderBuilder`、`ModuleSnapshot` | [数据与存储](data-and-storage.md) |
| 运行本地模型推理 | `rullm` | [模型推理](model-inference.md) |

## 包名与源码入口

Cargo 包名、仓库目录名和 Rust 导入名可能不同。例如 ruFFT 位于 `crates/ruFFT/`，安装包名为 `ruda-fft`，导入名为 `rufft`；ruSOLVER 的安装包名为 `ruda-solver`，导入名为 `rusolver`。

下表覆盖解析后的 workspace，包括通过路径依赖加入的 CANN 驱动和测试支持包。包名链接指向 Cargo manifest，源码链接指向 crate 入口。Feature 与版本以 manifest 为准，不能根据目录名推断。测试项目与原生 PyTorch 扩展不是普通的 crates.io 安装目标。

## Kernel、编译器与运行时

| Cargo 包名 | Rust 导入名／源码 | 职责 |
| --- | --- | --- |
| [`ruda`](../../crates/ruda/Cargo.toml) | [`ruda`](../../crates/ruda/src/lib.rs) | GPU Kernel 门面与运行时重导出；编写 GPU 内核的入口 |
| [`ruda-core`](../../crates/ruda-core/Cargo.toml) | [`ruda_core`](../../crates/ruda-core/src/lib.rs) | 共享 IR、张量数据、dtype、设备与内存契约 |
| [`ruda-compiler`](../../crates/ruda-compiler/Cargo.toml) | [`ruda_compiler`](../../crates/ruda-compiler/src/lib.rs) | 内核前端与目标代码生成 |
| [`ruda-triton`](../../crates/ruda-triton/Cargo.toml) | [`ruda_triton`](../../crates/ruda-triton/src/lib.rs) | Triton SIMT 描述 → Ruda IR → PTX；源码构建接入，不是 crates.io 安装包 |
| [`ruda-kernel`](../../crates/ruda-kernel/Cargo.toml) | [`ruda_kernel`](../../crates/ruda-kernel/src/lib.rs) | Rust Kernel DSL、启动接口和设备张量工具 |
| [`ruda-runtime`](../../crates/ruda-runtime/Cargo.toml) | [`ruda_runtime`](../../crates/ruda-runtime/src/lib.rs) | 计算客户端／服务端、存储、流与调度 |
| [`ruda-kernel-macros`](../../crates/ruda-kernel-macros/Cargo.toml) | [`ruda_kernel_macros`](../../crates/ruda-kernel-macros/src/lib.rs) | 将 Kernel Rust 转为 IR 的过程宏 |
| [`ruda-ir-macros`](../../crates/ruda-ir-macros/Cargo.toml) | [`ruda_ir_macros`](../../crates/ruda-ir-macros/src/lib.rs) | IR 实现的派生宏 |

## 驱动与昇腾设备程序

| Cargo 包名 | Rust 导入名／源码 | 职责 |
| --- | --- | --- |
| [`ruda-driver-cuda`](../../crates/ruda-driver-cuda/Cargo.toml) | [`ruda_driver_cuda`](../../crates/ruda-driver-cuda/src/lib.rs) | CUDA 设备、分配、编译与启动 |
| [`ruda-driver-hip`](../../crates/ruda-driver-hip/Cargo.toml) | [`ruda_driver_hip`](../../crates/ruda-driver-hip/src/lib.rs) | AMD HIP 运行时适配 |
| [`ruda-hip-sys`](../../crates/ruda-hip-sys/Cargo.toml) | [`ruda_hip_sys`](../../crates/ruda-hip-sys/src/lib.rs) | 底层 HIP 运行时绑定 |
| [`ruda-driver-wgpu`](../../crates/ruda-driver-wgpu/Cargo.toml) | [`ruda_driver_wgpu`](../../crates/ruda-driver-wgpu/src/lib.rs) | WGPU 初始化、存储和计算执行 |
| [`ruda-driver-cpu`](../../crates/ruda-driver-cpu/Cargo.toml) | [`ruda_driver_cpu`](../../crates/ruda-driver-cpu/src/lib.rs) | CPU Kernel 运行时驱动；区别于 Host 张量后端 |
| [`ruda-driver-cann`](../../crates/ruda-driver-cann/Cargo.toml) | [`ruda_driver_cann`](../../crates/ruda-driver-cann/src/lib.rs) | 动态加载的 CANN AscendCL 接口 |
| [`ruda-ascend-kernels`](../../crates/ruda-ascend-kernels/Cargo.toml) | [`ruda_ascend_kernels`](../../crates/ruda-ascend-kernels/src/lib.rs) | Rust 昇腾设备程序与指令降级 |

## 领域计算库

| Cargo 包名 | Rust 导入名／源码 | 职责 |
| --- | --- | --- |
| [`rublas`](../../crates/ruBLAS/Cargo.toml) | [`rublas`](../../crates/ruBLAS/src/lib.rs) | 矩阵／向量算子与后端分发 |
| [`ruDNN`](../../crates/ruDNN/Cargo.toml) | [`rudnn`](../../crates/ruDNN/src/lib.rs) | 注意力、卷积、池化、归一化与 MoE |
| [`ruPRIM`](../../crates/ruPRIM/Cargo.toml) | [`ruprim`](../../crates/ruPRIM/src/lib.rs) | 归约、扫描、索引与逐元素内核 |
| [`ruda-fft`](../../crates/ruFFT/Cargo.toml) | [`rufft`](../../crates/ruFFT/src/lib.rs) | 傅里叶变换 |
| [`ruRAND`](../../crates/ruRAND/Cargo.toml) | [`rurand`](../../crates/ruRAND/src/lib.rs) | 随机采样与分布 |
| [`ruSPARSE`](../../crates/ruSPARSE/Cargo.toml) | [`rusparse`](../../crates/ruSPARSE/src/lib.rs) | 稀疏格式与算子 |
| [`ruTENSOR`](../../crates/ruTENSOR/Cargo.toml) | [`rutensor`](../../crates/ruTENSOR/src/lib.rs) | 张量收缩、置换与归约 |
| [`ruCCL`](../../crates/ruCCL/Cargo.toml) | [`ruccl`](../../crates/ruCCL/src/lib.rs) | 集合通信算法 |
| [`ruda-solver`](../../crates/ruSOLVER/Cargo.toml) | [`rusolver`](../../crates/ruSOLVER/src/lib.rs) | 主机科学求解器与可选设备求解内核 |
| [`ruintegrate`](../../crates/ruINTEGRATE/Cargo.toml) | [`ruintegrate`](../../crates/ruINTEGRATE/src/lib.rs) | 主机数值积分、ODE 积分与事件定位 |
| [`rublas-host`](../../crates/ruBLAS/host/Cargo.toml) | [`rublas_host`](../../crates/ruBLAS/host/src/lib.rs) | CPU 带步长和批量矩阵乘 |
| [`ruDNN-host`](../../crates/ruDNN/host/Cargo.toml) | [`rudnn_host`](../../crates/ruDNN/host/src/lib.rs) | CPU 神经网络算子实现 |
| [`ruPRIM-host`](../../crates/ruPRIM/host/Cargo.toml) | [`ruprim_host`](../../crates/ruPRIM/host/src/lib.rs) | CPU 张量原语与索引 |
| [`ruFFT-host`](../../crates/ruFFT/host/Cargo.toml) | [`rufft_host`](../../crates/ruFFT/host/src/lib.rs) | CPU 实数傅里叶变换 |
| [`ruRAND-host`](../../crates/ruRAND/host/Cargo.toml) | [`rurand_host`](../../crates/ruRAND/host/src/lib.rs) | 主机随机数生成 |

## 张量、微分与执行组合

| Cargo 包名 | Rust 导入名／源码 | 职责 |
| --- | --- | --- |
| [`ruda-tensor`](../../crates/ruda-tensor/Cargo.toml) | [`ruda_tensor`](../../crates/ruda-tensor/src/lib.rs) | 后端契约与 api feature 下的 api::Tensor |
| [`ruda-tensor-config`](../../crates/ruda-tensor-config/Cargo.toml) | [`ruda_tensor_config`](../../crates/ruda-tensor-config/src/lib.rs) | 自动微分／融合共享配置 |
| [`ruda-tensor-device`](../../crates/ruda-tensor-device/Cargo.toml) | [`ruda_tensor_device`](../../crates/ruda-tensor-device/src/lib.rs) | DeviceBackend、CUDA 适配器与领域库分发 |
| [`ruda-tensor-host`](../../crates/ruda-tensor-host/Cargo.toml) | [`ruda_tensor_host`](../../crates/ruda-tensor-host/src/lib.rs) | Host CPU 张量后端与步长布局 |
| [`ruda-tensor-wgpu`](../../crates/ruda-tensor-wgpu/Cargo.toml) | [`ruda_tensor_wgpu`](../../crates/ruda-tensor-wgpu/src/lib.rs) | WGPU 张量后端与设备初始化 |
| [`ruda-tensor-rocm`](../../crates/ruda-tensor-rocm/Cargo.toml) | [`ruda_tensor_rocm`](../../crates/ruda-tensor-rocm/src/lib.rs) | ROCm 张量适配器 |
| [`ruda-tensor-tch`](../../crates/ruda-tensor-tch/Cargo.toml) | [`ruda_tensor_tch`](../../crates/ruda-tensor-tch/src/lib.rs) | LibTorch 张量适配器 |
| [`ruda-autodiff`](../../crates/ruda-autodiff/Cargo.toml) | [`ruda_autodiff`](../../crates/ruda-autodiff/src/lib.rs) | 梯度图、反向传播与重计算 |
| [`ruda-fusion`](../../crates/ruda-fusion/Cargo.toml) | [`ruda_fusion`](../../crates/ruda-fusion/src/lib.rs) | 张量算子融合规划与执行 |
| [`ruda-tensor-router`](../../crates/ruda-tensor-router/Cargo.toml) | [`ruda_tensor_router`](../../crates/ruda-tensor-router/src/lib.rs) | 本地多后端路由与字节桥接 |
| [`ruda-tensor-remote`](../../crates/ruda-tensor-remote/Cargo.toml) | [`ruda_tensor_remote`](../../crates/ruda-tensor-remote/src/lib.rs) | 远程张量客户端／服务端 |
| [`ruda-communication`](../../crates/ruda-communication/Cargo.toml) | [`ruda_communication`](../../crates/ruda-communication/src/lib.rs) | 传输协议、WebSocket 与张量数据服务 |

## 模型、训练、存储与集成

| Cargo 包名 | Rust 导入名／源码 | 职责 |
| --- | --- | --- |
| [`ruda-model`](../../crates/ruda-model/Cargo.toml) | [`ruda_model`](../../crates/ruda-model/src/lib.rs) | 模块参数、配置、Record 与数据加载器 |
| [`ruda-model-macros`](../../crates/ruda-model-macros/Cargo.toml) | [`ruda_model_macros`](../../crates/ruda-model-macros/src/lib.rs) | Config、Module、Record 派生宏 |
| [`ruda-model-codegen`](../../crates/ruda-model-codegen/Cargo.toml) | [`ruda_model_codegen`](../../crates/ruda-model-codegen/src/lib.rs) | 模型派生宏使用的代码生成器 |
| [`ruda-nn`](../../crates/ruda-nn/Cargo.toml) | [`ruda_nn`](../../crates/ruda-nn/src/lib.rs) | 神经网络层、激活与损失 |
| [`ruda-optim`](../../crates/ruda-optim/Cargo.toml) | [`ruda_optim`](../../crates/ruda-optim/src/lib.rs) | 优化器、梯度累积／裁剪与调度 |
| [`ruda-dataset`](../../crates/ruda-dataset/Cargo.toml) | [`ruda_dataset`](../../crates/ruda-dataset/src/lib.rs) | 索引数据集、数据源与变换 |
| [`ruda-io`](../../crates/ruda-io/Cargo.toml) | [`ruda_io`](../../crates/ruda-io/src/lib.rs) | 主机 I/O 与可选网络下载 |
| [`ruda-store`](../../crates/ruda-store/Cargo.toml) | [`ruda_store`](../../crates/ruda-store/src/lib.rs) | 模型快照、Rudapack、safetensors 与 PyTorch 导入 |
| [`ruda-llm`](../../crates/ruLLM/Cargo.toml) | [`rullm`](../../crates/ruLLM/src/lib.rs) | 模型加载与自回归推理 |
| [`ruda-torch-native`](../../crates/ruda-torch/Cargo.toml) | [`ruda_torch_native`](../../crates/ruda-torch/src/lib.rs) | Python ruda_torch 包的原生 cdylib；源码构建组件 |

## 测试与消费者支持

| Cargo 包名 | Rust 导入名／源码 | 职责 |
| --- | --- | --- |
| [`ruda-test-runtime`](../../crates/ruda-test-runtime/Cargo.toml) | [`ruda_test_runtime`](../../crates/ruda-test-runtime/src/lib.rs) | 运行时／内核测试基础设施 |
| [`ruda-test-utils`](../../crates/ruda-test-utils/Cargo.toml) | [`ruda_test_utils`](../../crates/ruda-test-utils/src/lib.rs) | 内核测试辅助 |
| [`ruda-facade-consumer`](../../crates/ruda/tests/consumer/Cargo.toml) | [`ruda_facade_consumer`](../../crates/ruda/tests/consumer/src/lib.rs) | 门面 feature 连接的外部消费者测试项目 |
| [`ruda-store-pytorch-tests`](../../crates/ruda-store/pytorch-tests/Cargo.toml) | [`ruda_store_pytorch_tests`](../../crates/ruda-store/pytorch-tests/src/lib.rs) | PyTorch 格式互操作测试 |
| [`ruda-store-safetensors-tests`](../../crates/ruda-store/safetensors-tests/Cargo.toml) | [`ruda_store_safetensors_tests`](../../crates/ruda-store/safetensors-tests/src/lib.rs) | Safetensors 格式互操作测试 |

## 查找具体方法

原生 Python 方法见 [PyTorch API 参考](native-pytorch-api.md)、[模型编译](model-compiler.md)、[静态图](static-pytorch-graphs.md)和 [LoRA／NF4 微调](finetuning.md)。算子／应用选择共享[全栈自动调优策略](stack-autotuning.md)。

[架构指南](architecture-training.md)介绍 mHC、压缩注意力／缓存和 Python Muon；[分布式训练指南](distributed-training.md)介绍显式设备、rendezvous、副本初始化、加权归约和逐 rank 恢复。

Typed tensor 方法位于 [crates/ruda-tensor/src/api](../../crates/ruda-tensor/src/api)，后端 trait 位于 [crates/ruda-tensor/src/backend](../../crates/ruda-tensor/src/backend)，领域接口可从各计算库手册进入。使用符号前，先启用暴露该模块的 feature。

设备内存、提交和同步见 [Runtime API](runtime-api.md)，后端初始化和启动契约见 [Driver API](driver-api.md)。模型参数与 Record 类型从 [ruda-model 导出入口](../../crates/ruda-model/src/lib.rs) 查找，不要与编译器中名称相近的 IR 类型混用。
