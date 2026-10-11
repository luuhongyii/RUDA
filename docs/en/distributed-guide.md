# Collective communication and training integration

[Documentation](README.md) · [ruCCL reference](libraries/ruccl.md) · [中文](../zh/distributed-guide.md)

## Three interfaces, different ownership

| Interface | Session ownership | Typical entry |
| --- | --- | --- |
| Registered tensor collectives | Register a peer in the collective service | `register<B>`, `all_reduce<B>`, `finish_collective<B>` |
| Explicit rank communicator | Application owns its rank and transport session | `rank::communicator::RankCommunicator<D>` |
| In-process collective | One communicator owns a list of execution contexts | `in_process::Communicator<C>` |

These interfaces share lower-level components but are not interchangeable handles. A `PeerId` for the registered service is not a `RankCommunicator`, and the in-process communicator does not create a multi-process network session.

A logical rank is a participant, not necessarily a physical GPU. The repository examples deliberately place several logical ranks on the same device.

## Complete two-rank CPU example

Create a binary project and add:

```toml
[dependencies]
ruccl = { package = "ruCCL", version = "0.21" }
ruda-tensor = { version = "0.21", default-features = false, features = ["api-std"] }
ruda-tensor-host = { version = "0.21", default-features = false, features = ["std"] }
```

Put this in `src/main.rs`, then run `cargo run`:

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

Each rank contributes two FP32 values; Sum returns `[1, 5]` to both ranks. The calls execute concurrently because a collective needs all participants. Do not call both ranks sequentially on the same blocking thread.

To use the repository's four-rank CUDA example:

```bash
cargo run --locked -p ruCCL --features cuda --example all_reduce
```

It selects CUDA device 0 for all four logical ranks, not four GPUs.

## Registered collectives and call ordering

1. Construct matching `CollectiveConfig` values and choose a unique peer ID for each participant.
2. Register the concrete execution backend and device.
3. Enter the same sequence of operations on all ranks.
4. Match shape, dtype, reduction operation, and root/sender conventions.
5. Finish each participant's session with `finish_collective`.

`broadcast` takes `Some(tensor)` only on the sender and `None` on receivers. `reduce` returns `Some(result)` only at the chosen root. `all_reduce` returns a result to every participant. Match the backend's floating tensor primitive rather than passing an API tensor wrapper directly.

`reset_collective` discards service state; it does not finish pending transfers or restore a training checkpoint.

### Multi-node configuration

The registered service's global configuration requires all four fields together: node count, global orchestrator address, this node's address, and tensor data-service port. The corresponding methods are `with_num_nodes`, `with_global_address`, `with_node_address`, and `with_data_service_port`. `is_valid()` checks whether this set is complete; it is not a network connectivity test.

The global-node implementation contains AllReduce algorithms. Its global `reduce` and `broadcast` methods are not implemented. Do not infer multi-node support for those operations from the local API names or strategy setters. This boundary refers to the registered/global-node path, not every method of the separate rank APIs.

## Explicit rank and in-process tensor adapters

`TensorDevice<B>` adapts a backend device to collective storage and reduction operations. Use `import_float` / `import_int` for existing backend primitives, and `float_tensor()` / `int_tensor()` to export a collective buffer into subsequent tensor operations.

`RankCommunicator::connect` takes an initializer, rendezvous address, `UniqueId`, rank, world size, timeout, and queue name. All participants must use the same session identity and compatible topology. `from_session` accepts an existing `TcpRankSession`; `from_transport_with_tuning` accepts an explicit transport and tuning configuration. `transport()` reports the selected transport; an enum variant alone does not establish that a device supports it.

The in-process adapter can be exercised directly from the source tree:

```bash
cargo run --locked -p ruCCL --example tensor_collectives
cargo run --locked -p ruCCL --features cuda --example tensor_collectives
```

The first command uses Host; the second uses CUDA. The example imports three tensors, performs a ring sum, exports them back to tensor computation, and checks that the original tensor handle remains unchanged. All three contexts use the default device. The in-process communicator reports `HostStaged`; importing a device tensor does not imply zero-copy GPU peer transfers.

## Synchronize gradients before the optimizer step

Enable `ruda-optim/collective` and keep the execution backend separate from `Autodiff<Inner>`:

```text
forward → loss.backward() → GradientsParams::from_grads(...)
        → gradient all-reduce → optimizer.step(...)
```

| Session style | Gradient method |
| --- | --- |
| Registered peer | `grads.all_reduce::<Inner>(peer, ReduceOperation::Mean)` |
| Explicit rank communicator | `grads.all_reduce_with::<Inner>(&communicator, ReduceOperation::Mean)` |

Pass the inner backend, not `Autodiff<Inner>`. Synchronization works on extracted parameter gradients; the collective call is not itself an automatic differentiation rule.

Parameter IDs must refer to the same parameters on every rank, with matching gradient shapes and dtypes. The explicit-rank implementation sorts parameter IDs before sending gradients. Sorting does not repair missing/different parameter sets, and missing gradients are not filled with zeros. Quantized gradients are rejected by this path.

Choose Sum or Mean according to the training loss and batch partition. Mean divides by rank count; it is not automatically a sample-count-weighted average when ranks process different numbers of samples.

## Save and resume the two-rank training example

From the RUDA root:

```bash
cargo run --locked -p ruda-optim --features collective --example collective_training -- run ./collective-state
cargo run --locked -p ruda-optim --features collective --example collective_training -- resume ./collective-state
```

The example uses the registered-peer API and Host by default. Add `cuda` to the feature list for its CUDA mode. `run` requires a new directory, saves both ranks after the first update, and continues with the second update. `resume` restores those model/optimizer records and repeats the second update.

This example's two logical ranks share the default device. It illustrates state restoration, not automatic recovery of arbitrary multi-node jobs. For scheduler state and pending accumulated gradients, see [Training and saving state](training.md); for weight formats, see [Data and storage](data-and-storage.md).

## Related execution interfaces

Router selects among local tensor backends; Remote sends tensor operations to a server. Neither is a collective communicator. Router's `OperationIr::Distributed` execution branch is not implemented, so enabling the feature alone does not connect Router/Remote to every distributed operation. See [Backend composition](backend-composition.md).

## Source entry points

- [Registered API](../../ruCCL/src/api.rs), [configuration](../../ruCCL/src/config.rs), [global node](../../ruCCL/src/global/node/base.rs).
- [Rank communicator](../../ruCCL/src/rank/communicator/mod.rs) and [connection methods](../../ruCCL/src/rank/communicator/connect.rs).
- [Tensor adapters](../../ruCCL/src/tensor_device/mod.rs), [in-process ownership](../../ruCCL/src/in_process/owner.rs), [tensor example](../../ruCCL/examples/tensor_collectives.rs).
- [Explicit-rank gradient reduction](../../ruda-optim/src/optim/grads/collective.rs) and [training example](../../ruda-optim/examples/collective_training.rs).
