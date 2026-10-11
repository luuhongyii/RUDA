# Ascend compiler and CANN execution

[Documentation](README.md) · [Compiler guide](compiler-guide.md) · [中文](../zh/ascend-guide.md)

## Choose the execution interface

| Component | Responsibility |
| --- | --- |
| `ruda-compiler/ascend` | Lower common `KernelDefinition` IR to CCE device source |
| `ruda-driver-cann/common-ir` | Load the matching native artifact with `CannProgram` |
| `CannSession`, `CannTensor` | Own or borrow the CANN execution context and manage device tensors |
| `ruda-ascend-kernels` | Structured Rust device programs for native BF16 matrix kernels |
| `rublas::cann` | Matrix-library entry points using the CANN tensor layer |

This guide describes the Ascend implementation in the RUDA repository. Its common-IR path targets `Ascend950DT` / `dav-c310`. Compilation follows:

```text
KernelDefinition → AscendCompiler → kernel.asc
                 → CANN Bisheng → LLD → kernel.o + kernel.ruda
                 → CannProgram → device execution
```

Bisheng compiles generated CCE source; this is not direct Rust-to-Ascend ISA generation. `CannSession` is an explicit execution interface, not a generic `ruda_tensor::Backend` or an automatic PyTorch/autograd integration. Its ACLNN tensor methods, native BF16 matrix programs, and common-IR vector programs are separate execution paths.

## Emit a map kernel

From the RUDA source root, on the machine used for compilation:

```bash
cargo run --locked -p ruda-compiler --no-default-features --features ascend --example ascend_ir -- --op mul --elements 7 --out ./ascend-mul-ir
```

The output directory must not already exist. The command writes:

| File | Contents |
| --- | --- |
| `kernel.asc` | Generated CCE source |
| `kernel.contract` | Target, entry point, binding byte lengths, and launch configuration |
| `kernel.common-ir.txt` | Common IR representation |

These are compiler outputs, not a loadable native object. The logical element count is specialized into the compiled contract; seven elements and eight elements require different contracts.

All map bindings are contiguous FP32 arrays of `elements` entries. Input/output order is:

| Operation name | Inputs | Outputs |
| --- | --- | --- |
| `copy` | X | Y |
| `add`, `mul` | X, Z | Y |
| `silu` | X | Y |
| `silu_mul` | X, Z | Y |
| `silu_backward` | X, dY | dX |
| `silu_mul_backward` | X, Z, dY | dX, dZ |

## Build the native artifact

On a Linux compilation host with the matching CANN toolkit installed, run from the RUDA root:

```bash
python3 tools/ascend/build_common_ir.py --toolkit "$ASCEND_HOME_PATH" --op mul --elements 7 --out ./ascend-mul-7
```

Set `ASCEND_HOME_PATH` to the installed toolkit root, or replace its value with an explicit path. The builder requires `bin/bisheng`, `bin/ld.lld`, and the SDK's `kernel_operator.h`. It invokes the Rust emitter, compiles, links, and produces `kernel.o` plus `kernel.ruda`. Use a fresh output directory; the builder rejects overwriting an existing artifact.

`--emit-only` skips Bisheng/LLD, but still compiles and runs the Rust emitter. It does not create a loadable `kernel.ruda`.

## Load and execute in an application

Place an application directory alongside the `RUDA` checkout. Its dependencies are:

```toml
[dependencies]
ruda-compiler = { path = "../RUDA/ruda-compiler", default-features = false, features = ["ascend"] }
ruda-core = { path = "../RUDA/ruda-core", default-features = false, features = ["compiler-std"] }
ruda-driver-cann = { path = "../RUDA/ruda-driver-cann", features = ["common-ir"] }
```

Use this `src/main.rs`:

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

On the Ascend machine, run `cargo run -- ../RUDA/ascend-mul-7`. CANN shared libraries must be resolvable; `RUDA_CANN_LIBRARY` and `RUDA_CANN_OPAPI` can select their paths.

`open_exclusive` initializes and owns the ACL device/context/stream lifecycle. Use it only in a standalone process without another ACL owner, including `torch_npu`. For caller-owned resources, `CannSession::attach` borrows an already initialized context and stream, which must outlive the session and its tensors. Both constructors are unsafe because these ownership and library requirements are caller obligations.

## Row programs and binding shapes

Row mode uses one logical 32-lane plane per row. Set `elements = R * W` and `row_width = Some(W)`. Width must be 32..4096 and divisible by 32. The row builder accepts a finite positive compile-time epsilon; RMSNorm uses it in normalization.

For three rows of width 96:

```bash
python3 tools/ascend/build_common_ir.py --toolkit "$ASCEND_HOME_PATH" --op rms_norm --elements 288 --row-width 96 --epsilon 0.00001 --out ./ascend-rms-3x96
```

Recreate the matching IR with `row_programs::definition(RowProgram::RmsNorm, 96, 1e-5)` and identical compiler options before loading this artifact.

| Operation name | Input binding order | Output binding order |
| --- | --- | --- |
| `row_sum`, `row_mean`, `row_max` | X[R,W] | Y[R] |
| `softmax`, `log_softmax` | X[R,W] | Y[R,W] |
| `rms_norm` | X[R,W], weight[W] | Y[R,W], rstd[R] |
| `softmax_backward` | Y[R,W], dY[R,W] | dX[R,W] |
| `log_softmax_backward` | logY[R,W], dY[R,W] | dX[R,W] |
| `rms_norm_input_backward` | X[R,W], dY[R,W], weight[W], rstd[R] | dX[R,W] |

All entries in this table are FP32. The backward inputs use saved forward outputs/statistics, not interchangeable raw inputs. RMSNorm input backward returns dX only; it does not compute the affine weight gradient.

## Allocation, reuse, and completion

- `CannProgram::run` allocates each output using its compiled binding byte length and returns flattened tensors. An RMSNorm statistic has R elements, not R*W.
- `run_into` reuses caller-provided outputs. All tensors must belong to the same session and match the binding dtype and byte length.
- Read-only inputs may alias each other; writable buffers must not overlap any other binding. There is no implicit broadcasting or in-place map update.
- Each nonempty invocation launches and synchronizes before returning. The interface is not an asynchronous submission queue.
- Zero-element invocations are explicit no-ops after binding checks. `ProgramStats` exposes launches, synchronizations, and empty calls.
- After a launch/completion failure, the program rejects further calls; follow the reported error and synchronize/recreate rather than reusing that program.

The loader matches metadata against the compiled contract, compares the generated-source and object digests, and checks the recorded compiler digest's format. It does not re-hash an installed compiler when loading. These checks detect stale or corrupted artifacts; they do not make arbitrary native code safe to load.

## Target and lowering constraints

`AscendCompiler` requires an explicit target, `ExecutionMode::Checked`, and 64-bit addresses. Map tiles are 8..4096 FP32 elements in multiples of eight; the configurable vector-core count is 1..32. Allocation planning must fit the configured UB budget, up to 128 KiB.

The common-IR implementation accepts its supported FP32 map and row patterns, not arbitrary CUDA kernels. Generic strided layouts, FP16/BF16 row lowering, arbitrary control flow, and RMSNorm weight backward are not provided by this path. Native BF16 GEMM belongs to the separate matrix path; see [ruBLAS](libraries/rublas.md).

## API entry points

- [Compiler/options](../../ruda-compiler/src/ascend/mod.rs), [map definitions](../../ruda-compiler/src/ascend/programs.rs), [row definitions](../../ruda-compiler/src/ascend/row_programs.rs).
- [Artifact builder](../../tools/ascend/build_common_ir.py) and [compiler CLI](../../ruda-compiler/examples/ascend_ir.rs).
- [CannProgram](../../ruda-driver-cann/src/tensor/common_ir/mod.rs), [session lifecycle](../../ruda-driver-cann/src/tensor/owner.rs), [borrowed sessions](../../ruda-driver-cann/src/tensor/mod.rs).
- [Native matrix interface](../../ruda-driver-cann/src/tensor/deepgemm/mod.rs) and [ruBLAS integration](../../ruBLAS/src/cann.rs).
