# 昇腾编译器与 CANN 执行

[文档首页](README.md) · [编译器指南](compiler-guide.md) · [English](../en/ascend-guide.md)

## 选择执行接口

| 组件 | 职责 |
| --- | --- |
| `ruda-compiler/ascend` | 将公共 `KernelDefinition` IR 降为 CCE 设备源码 |
| `ruda-driver-cann/common-ir` | 通过 `CannProgram` 加载匹配的原生构建物 |
| `CannSession`、`CannTensor` | 持有或借用 CANN 执行上下文，管理设备张量 |
| `ruda-ascend-kernels` | 原生 BF16 矩阵内核的结构化 Rust 设备程序 |
| `rublas::cann` | 使用 CANN 张量层的矩阵库入口 |

本指南描述 RUDA 仓库内的昇腾实现。公共 IR 路径以 `Ascend950DT`／`dav-c310` 为目标，编译过程如下：

```text
KernelDefinition → AscendCompiler → kernel.asc
                 → CANN Bisheng → LLD → kernel.o + kernel.ruda
                 → CannProgram → device execution
```

Bisheng 编译生成的 CCE 源码，而不是 Rust 直接生成昇腾 ISA。`CannSession` 是显式执行接口，不是通用 `ruda_tensor::Backend`，也不会自动接入 PyTorch/autograd。它的 ACLNN 张量方法、原生 BF16 矩阵程序和公共 IR 向量程序属于不同执行路径。

## 生成 Map 内核

在用于编译的机器上，从 RUDA 源码根目录执行：

```bash
cargo run --locked -p ruda-compiler --no-default-features --features ascend --example ascend_ir -- --op mul --elements 7 --out ./ascend-mul-ir
```

输出目录必须尚不存在，命令生成：

| 文件 | 内容 |
| --- | --- |
| `kernel.asc` | 生成的 CCE 源码 |
| `kernel.contract` | 目标、入口、绑定字节数与启动配置 |
| `kernel.common-ir.txt` | 公共 IR 表示 |

这些是编译器输出，不是可加载的原生目标文件。逻辑元素数量已经特化到编译契约中，七个元素与八个元素需要不同契约。

所有 Map 绑定都是含 `elements` 个元素的连续 FP32 数组。输入输出顺序如下：

| 操作名 | 输入 | 输出 |
| --- | --- | --- |
| `copy` | X | Y |
| `add`, `mul` | X, Z | Y |
| `silu` | X | Y |
| `silu_mul` | X, Z | Y |
| `silu_backward` | X, dY | dX |
| `silu_mul_backward` | X, Z, dY | dX, dZ |

## 构建原生构建物

在安装了匹配 CANN 工具链的 Linux 编译机上，从 RUDA 根目录执行：

```bash
python3 tools/ascend/build_common_ir.py --toolkit "$ASCEND_HOME_PATH" --op mul --elements 7 --out ./ascend-mul-7
```

将 `ASCEND_HOME_PATH` 设置为已安装的工具链根目录，或把该值替换为明确路径。构建器要求存在 `bin/bisheng`、`bin/ld.lld` 和 SDK 的 `kernel_operator.h`。它调用 Rust emitter，编译、链接后生成 `kernel.o` 和 `kernel.ruda`。输出目录需为新目录，构建器拒绝覆盖已有构建物。

`--emit-only` 跳过 Bisheng／LLD，但仍会编译并运行 Rust emitter，不生成可加载的 `kernel.ruda`。

## 在应用中加载和执行

将应用目录放在 `RUDA` checkout 旁，依赖配置为：

```toml
[dependencies]
ruda-compiler = { path = "../RUDA/ruda-compiler", default-features = false, features = ["ascend"] }
ruda-core = { path = "../RUDA/ruda-core", default-features = false, features = ["compiler-std"] }
ruda-driver-cann = { path = "../RUDA/ruda-driver-cann", features = ["common-ir"] }
```

使用下面的 `src/main.rs`：

```rust
use ruda_compiler::ascend::{
    programs::{definition, MapProgram}, AscendCompiler, AscendOptions, AscendTarget,
};
use ruda_core::{compiler::Compiler, ir::UIntKind, launch::ExecutionMode};
use ruda_driver_cann::{
    tensor::{common_ir::CannProgram, CannSession}, CannDevice,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let artifact = std::env::args().nth(1).ok_or("pass the mul-7 artifact directory")?;
    let compiled = AscendCompiler.compile(
        definition(MapProgram::Mul),
        &AscendOptions {
            target: Some(AscendTarget::Ascend950DT),
            elements: 7,
            ..Default::default()
        },
        ExecutionMode::Checked,
        UIntKind::U64.into(),
    )?;
    let acl = std::env::var("RUDA_CANN_LIBRARY")
        .unwrap_or_else(|_| "libascendcl.so".into());
    let opapi = std::env::var("RUDA_CANN_OPAPI")
        .unwrap_or_else(|_| "libopapi.so".into());
    // Standalone process: exclusive ACL ownership, trusted SDK and artifact.
    let session = unsafe { CannSession::open_exclusive(CannDevice::new(0)?, acl, opapi)? };
    let program = unsafe { CannProgram::load(&session, compiled, artifact)? };
    let x = session.from_f32(&[7], &[1., 2., 3., 4., 5., 6., 7.])?;
    let y = session.from_f32(&[7], &[2., 2., 2., 2., 2., 2., 2.])?;
    let outputs = program.run(&[&x, &y])?;
    assert_eq!(outputs[0].to_f32()?, vec![2., 4., 6., 8., 10., 12., 14.]);
    Ok(())
}
```

在昇腾机器上执行 `cargo run -- ../RUDA/ascend-mul-7`。CANN 动态库必须可被解析；可通过 `RUDA_CANN_LIBRARY` 和 `RUDA_CANN_OPAPI` 指定库路径。

`open_exclusive` 初始化并持有 ACL 设备、上下文及流的生命周期，只适用于没有其他 ACL 所有者的独立进程，包括不能与 `torch_npu` 同时持有这套生命周期。调用者已有资源时，用 `CannSession::attach` 借用已经初始化的上下文和流，且其生命周期必须长于 session 及其张量。两个构造器均为 unsafe，因为资源所有权与动态库要求由调用方保证。

## 行程序与绑定形状

行模式用一个逻辑 32-lane plane 处理一行，设置 `elements = R * W`、`row_width = Some(W)`。宽度范围为 32..4096，且必须是 32 的倍数。行构造器接收有限正数的编译期 epsilon，RMSNorm 使用它进行归一化。

三行、每行 96 个元素的构建命令：

```bash
python3 tools/ascend/build_common_ir.py --toolkit "$ASCEND_HOME_PATH" --op rms_norm --elements 288 --row-width 96 --epsilon 0.00001 --out ./ascend-rms-3x96
```

加载该构建物前，使用 `row_programs::definition(RowProgram::RmsNorm, 96, 1e-5)` 和一致的编译选项重建对应 IR。

| 操作名 | 输入绑定顺序 | 输出绑定顺序 |
| --- | --- | --- |
| `row_sum`, `row_mean`, `row_max` | X[R,W] | Y[R] |
| `softmax`, `log_softmax` | X[R,W] | Y[R,W] |
| `rms_norm` | X[R,W], weight[W] | Y[R,W], rstd[R] |
| `softmax_backward` | Y[R,W], dY[R,W] | dX[R,W] |
| `log_softmax_backward` | logY[R,W], dY[R,W] | dX[R,W] |
| `rms_norm_input_backward` | X[R,W], dY[R,W], weight[W], rstd[R] | dX[R,W] |

表中所有元素均为 FP32。反向输入使用保存的前向输出／统计量，不能随意换成原始输入。RMSNorm 输入反向只返回 dX，不计算仿射权重梯度。

## 分配、复用与完成语义

- `CannProgram::run` 按各输出的编译绑定字节数分配，返回展平张量；RMSNorm 统计量只有 R 个元素，不是 R*W。
- `run_into` 复用调用者提供的输出。所有张量必须属于同一 session，dtype 与字节数必须匹配绑定。
- 只读输入可以互相别名，任何可写缓冲区都不能与其他绑定重叠；没有隐式广播或原地 Map 更新。
- 每次非空调用在返回前完成启动和同步，不是异步提交队列。
- 零元素调用在绑定检查后作为显式空操作处理。`ProgramStats` 提供启动、同步与空调用计数。
- 启动／完成失败后，该程序拒绝继续调用；需按返回错误处理并同步／重建，不能继续复用该程序。

加载器将元数据与编译契约匹配，比较生成源码和目标文件的摘要，并检查所记录的编译器摘要格式；加载时不会重新计算已安装编译器的摘要。这些检查用于发现过期或损坏的构建物，不代表任意原生代码都可以安全加载。

## 目标与降级约束

`AscendCompiler` 要求显式目标、`ExecutionMode::Checked` 和 64 位地址。Map tile 为 8..4096 个 FP32 元素，且是八的倍数；向量核数量配置范围为 1..32。分配规划必须落在配置的 UB 预算内，上限为 128 KiB。

公共 IR 实现接受所支持的 FP32 Map 与行模式，不接受任意 CUDA 内核。这条路径不提供通用步长布局、FP16／BF16 行降级、任意控制流或 RMSNorm 权重反向。原生 BF16 GEMM 属于单独的矩阵路径，见 [ruBLAS](libraries/rublas.md)。

## API 入口

- [编译器与选项](../../ruda-compiler/src/ascend/mod.rs)、[Map 定义](../../ruda-compiler/src/ascend/programs.rs)、[行定义](../../ruda-compiler/src/ascend/row_programs.rs)。
- [构建器](../../tools/ascend/build_common_ir.py)和[编译器 CLI](../../ruda-compiler/examples/ascend_ir.rs)。
- [CannProgram](../../ruda-driver-cann/src/tensor/common_ir/mod.rs)、[session 生命周期](../../ruda-driver-cann/src/tensor/owner.rs)、[借用 session](../../ruda-driver-cann/src/tensor/mod.rs)。
- [原生矩阵接口](../../ruda-driver-cann/src/tensor/deepgemm/mod.rs)与 [ruBLAS 集成](../../ruBLAS/src/cann.rs)。
