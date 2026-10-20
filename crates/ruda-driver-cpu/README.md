# ruda-driver-cpu

CPU execution backend for Ruda kernels using the MLIR compilation path. This is a compiled-kernel runtime, distinct from the host tensor operators in `ruda-tensor-host`.

## Interfaces

- `CpuDevice` and `CpuRuntime` provide the device and runtime entry points.
- `compilation`, `execution`, and `memory` implement CPU kernel preparation and execution.
- The backend depends on upstream `melior` and the compiler's `mlir` feature; LLVM/MLIR 20 development libraries are required for builds. Set `MLIR_SYS_200_PREFIX` and `TABLEGEN_200_PREFIX` to the LLVM installation prefix when it is not on `PATH`.

## Usage

Cargo package: `ruda-driver-cpu`. Rust import: `ruda_driver_cpu`.

```toml
[dependencies]
ruda-driver-cpu = "0.1"
```

## Features

Default features: `std`, `ruda/runtime-default`, `ruda-core/std`, `ruda-kernel/frontend-default`.

| Feature | Purpose |
| --- | --- |
| `mlir-dump` | Enable compiler MLIR dumps. |
| `std` | Enable standard-library runtime integration. |
| `tracing` | Enable runtime and optimizer tracing. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/crates/ruda-driver-cpu/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/crates/ruda-driver-cpu/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/driver-api.md)
