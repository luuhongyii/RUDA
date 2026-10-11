# 集合通信与训练接入

[文档首页](README.md) · [ruCCL 参考](libraries/ruccl.md) · [English](../en/distributed-guide.md)

## 三种接口，不同的所有权方式

| 接口 | 会话所有权 | 常用入口 |
| --- | --- | --- |
| 注册式张量集合通信 | 向集合通信服务注册 peer | `register<B>`、`all_reduce<B>`、`finish_collective<B>` |
| 显式 rank 通信器 | 应用持有 rank 和传输会话 | `rank::communicator::RankCommunicator<D>` |
| 进程内集合通信 | 一个通信器持有一组执行上下文 | `in_process::Communicator<C>` |

这些接口复用部分底层组件，但句柄不能互换。注册服务使用的 `PeerId` 不是 `RankCommunicator`，进程内通信器也不会创建多进程网络会话。

逻辑 rank 表示参与方，不必对应一张物理 GPU。仓库示例明确将多个逻辑 rank 放在同一设备上。

## 完整的双 rank CPU 示例

创建二进制项目并添加：

```toml
[dependencies]
ruccl = { package = "ruCCL", version = "0.21" }
ruda-tensor = { version = "0.21", default-features = false, features = ["api-std"] }
ruda-tensor-host = { version = "0.21", default-features = false, features = ["std"] }
```

把以下程序放入 `src/main.rs`，执行 `cargo run`：

```rust
use ruccl::{
    all_reduce, finish_collective, register, AllReduceStrategy, CollectiveConfig,
    ReduceOperation,
};
use ruda_tensor::{
    api::{Tensor, TensorPrimitive},
    DType,
};
use ruda_tensor_host::{Host, HostDevice};

fn main() {
    let config = CollectiveConfig::default()
        .with_num_devices(2)
        .with_local_all_reduce_strategy(AllReduceStrategy::Ring);
    let workers: Vec<_> = (0..2usize).map(|rank| {
        let config = config.clone();
        std::thread::spawn(move || {
            let device = HostDevice;
            let peer = rank.into();
            register::<Host>(peer, device, config).unwrap();
            let input = Tensor::<Host, 1>::from_data(
                [rank as f32, rank as f32 + 2.0], (&device, DType::F32),
            );
            let output = all_reduce::<Host>(
                peer, input.into_primitive().tensor(), ReduceOperation::Sum,
            ).unwrap();
            let result = Tensor::<Host, 1>::from_primitive(TensorPrimitive::Float(output));
            assert_eq!(result.into_data().as_slice::<f32>().unwrap(), &[1.0, 5.0]);
            finish_collective::<Host>(peer).unwrap();
        })
    }).collect();
    for worker in workers {
        worker.join().unwrap();
    }
}
```

每个 rank 提供两个 FP32 值，Sum 向两个 rank 都返回 `[1, 5]`。集合操作需要所有参与方，因此调用必须并发；不能在同一阻塞线程中顺序执行两个 rank。

仓库中的四 rank CUDA 示例：

```bash
cargo run --locked -p ruCCL --features cuda --example all_reduce
```

该示例的四个逻辑 rank 都选择 CUDA 设备 0，不是使用四张 GPU。

## 注册式集合通信与调用顺序

1. 创建匹配的 `CollectiveConfig`，为每个参与方选择唯一 peer ID。
2. 注册具体执行后端及其设备。
3. 所有 rank 按相同顺序进入集合操作。
4. 保持 shape、dtype、归约操作及 root／sender 约定一致。
5. 每个参与方使用 `finish_collective` 结束会话。

`broadcast` 只有发送方传 `Some(tensor)`，接收方传 `None`。`reduce` 只有指定 root 返回 `Some(result)`，`all_reduce` 向每个参与方都返回结果。参数使用后端浮点张量 primitive，而不是直接传 API 张量包装器。

`reset_collective` 丢弃服务状态，不负责完成待处理传输，也不恢复训练检查点。

### 多节点配置

注册服务的全局配置需要同时提供节点数量、全局协调器地址、本节点地址和张量数据服务端口。对应方法为 `with_num_nodes`、`with_global_address`、`with_node_address`、`with_data_service_port`。`is_valid()` 检查这组配置是否完整，不测试网络是否连通。

Global node 实现包含 AllReduce 算法，但其全局 `reduce` 和 `broadcast` 方法未实现。不能从本地 API 名称或策略 setter 推断这两种操作支持多节点。此边界针对注册式／global-node 路径，不代表独立 rank API 的所有方法都具有同样限制。

## 显式 rank 与进程内张量适配

`TensorDevice<B>` 将后端设备适配为集合通信的存储与归约接口。已有后端 primitive 用 `import_float`／`import_int` 导入；集合缓冲区用 `float_tensor()`／`int_tensor()` 导出，再接入后续张量运算。

`RankCommunicator::connect` 接收初始化函数、会合地址、`UniqueId`、rank、world size、超时和队列名。参与方需使用同一会话标识与兼容的拓扑。`from_session` 接受已有的 `TcpRankSession`，`from_transport_with_tuning` 接受显式传输层与调优配置。`transport()` 返回所选传输类型；枚举中存在某个变体，并不代表设备就支持该传输。

从源码根目录运行进程内张量适配示例：

```bash
cargo run --locked -p ruCCL --example tensor_collectives
cargo run --locked -p ruCCL --features cuda --example tensor_collectives
```

第一条命令使用 Host，第二条使用 CUDA。示例导入三个张量，执行 ring sum，导出回张量计算，并检查原始张量句柄保持不变。三个上下文均使用默认设备。进程内通信器报告 `HostStaged`；导入设备张量不等于 GPU 之间零拷贝传输。

## 在优化器更新前同步梯度

启用 `ruda-optim/collective`，区分执行后端与 `Autodiff<Inner>`：

```text
forward → loss.backward() → GradientsParams::from_grads(...)
        → gradient all-reduce → optimizer.step(...)
```

| 会话方式 | 梯度方法 |
| --- | --- |
| 注册 peer | `grads.all_reduce::<Inner>(peer, ReduceOperation::Mean)` |
| 显式 rank 通信器 | `grads.all_reduce_with::<Inner>(&communicator, ReduceOperation::Mean)` |

传入内层后端，不是 `Autodiff<Inner>`。同步处理的是已经提取的参数梯度，集合调用本身不是自动微分规则。

各 rank 的参数 ID 必须对应同一组参数，梯度 shape 和 dtype 必须匹配。显式 rank 实现在发送梯度前按参数 ID 排序，但排序不能修复参数集合缺失或不同，缺失梯度也不会被自动补零。这条路径拒绝量化梯度。

根据训练损失和批次划分选择 Sum 或 Mean。Mean 按 rank 数量平均；各 rank 样本数不同时，它不会自动变成按样本数加权的平均。

## 保存与恢复双 rank 训练示例

从 RUDA 根目录执行：

```bash
cargo run --locked -p ruda-optim --features collective --example collective_training -- run ./collective-state
cargo run --locked -p ruda-optim --features collective --example collective_training -- resume ./collective-state
```

该示例使用注册 peer API，默认后端为 Host；feature 列表追加 `cuda` 可切换到 CUDA 模式。`run` 要求目录尚不存在，第一次更新后保存两个 rank 的状态，再继续第二次更新。`resume` 恢复模型／优化器记录并重新执行第二次更新。

示例的两个逻辑 rank 使用同一默认设备，演示的是状态恢复，而不是任意多节点作业的自动故障恢复。调度器状态和待累积梯度见[训练与状态保存](training.md)，权重格式见[数据与存储](data-and-storage.md)。

## 与其他执行接口的关系

Router 在本地张量后端之间选择执行位置，Remote 将张量操作发送到服务器；二者都不是集合通信器。Router 的 `OperationIr::Distributed` 执行分支未实现，因此仅启用 feature 不会自动把 Router／Remote 接入所有分布式操作，见[后端组合](backend-composition.md)。

## 源码入口

- [注册式 API](../../ruCCL/src/api.rs)、[配置](../../ruCCL/src/config.rs)、[全局节点](../../ruCCL/src/global/node/base.rs)。
- [Rank 通信器](../../ruCCL/src/rank/communicator/mod.rs)与[连接方法](../../ruCCL/src/rank/communicator/connect.rs)。
- [张量适配](../../ruCCL/src/tensor_device/mod.rs)、[进程内所有权](../../ruCCL/src/in_process/owner.rs)、[张量示例](../../ruCCL/examples/tensor_collectives.rs)。
- [显式 rank 梯度归约](../../ruda-optim/src/optim/grads/collective.rs)与[训练示例](../../ruda-optim/examples/collective_training.rs)。
