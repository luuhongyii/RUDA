# ruda-kernel

Rust kernel DSL, reusable kernel building blocks, and runtime-generic device tensor operations. This is the kernel-authoring layer, not the high-level `ruda-tensor::api::Tensor` interface.

## Interfaces

- `dsl`: kernel types, IR expansion, and launch interfaces.
- `library`: reusable device-kernel building blocks.
- `tensor`: device tensor allocation, transfer, readback, and operations.
- `template`: host source templates integrated with compiled kernel tasks.

## Usage

Cargo package: `ruda-kernel`. Rust import: `ruda_kernel`.

```toml
[dependencies]
ruda-kernel = "0.1"
```

## Typed pointers

The additive DSL interface is `Ptr<E, S, A>` and `TypedSlice<E, S, A>`, where `S` is
`GlobalSpace`, `Workgroup`, or `Local`, and `A` is `ReadOnly` (the default) or `ReadWrite`.
Convert an existing slice with `.typed::<GlobalSpace>()`, `.typed::<Workgroup>()`, or
`.typed::<Local>()`; its storage origin is checked during kernel expansion.
`.ptr(index)` produces an element pointer. Pointers support `.offset(elements)`,
`.load()`, `.read_only()`, and `.select(condition, otherwise)`; `.store(value)` is
available only for `ReadWrite`. Selected pointers must have the same element,
address-space, and permission types. Typed slices support `.len()` and `.slice(start, end)`.

These are logical storage views, not arbitrary native addresses. Address-space and
permission labels remain in IR until backend lowering; writable global views also
require a writable kernel binding. Existing slices and native pointer operations
remain available. Typed load/store currently supports non-atomic elements; this
interface does not establish GPU alias, lifetime, or race-freedom proofs.

Calling `.value()` on a `Ptr` produces `ValuePtr<E, S, A>`, represented by a
distinct pointer type in IR. These values support assignment, branch results,
loop-carried offsets, helper-function parameters and returns, and the same
load/store/selection/permission-downgrade operations. CPU and GPU paths preserve pointer
types and phi nodes through SSA and scalar optimization, then lower them to storage
selectors and element offsets at the backend boundary. CUDA/HIP/Metal C++, direct PTX,
WGSL and SPIR-V export optimized pointer SSA as structured scopes before this lowering.
Non-atomic accesses retain the address-space and
permission descriptors. Low-level optimizer users can opt in with
`OptimizerBuilder::with_pointer_values()` and call `Optimizer::lower_pointer_values()`
at the CPU/MLIR emission boundary, or use `lower_optimized_pointer_values()` for
structured GPU emission. The pre-SSA `Scope::lower_pointer_values()` compatibility
path remains available.
`ValuePtr<E, GlobalSpace, A>` can also be a kernel argument, using `ArrayArg` and
the existing buffer-binding ABI, including allocation offsets and inplace aliases.

For `ValuePtr<Atomic<E>, S, A>`, `.atomic_load()` returns `E`; `ReadWrite` also
provides `.atomic_store()`, `.swap()`, `.fetch_add()`, `.fetch_sub()`,
`.fetch_min()`, and `.fetch_max()`. Integer atomics additionally support
`.fetch_and()`, `.fetch_or()`, `.fetch_xor()`, and `.compare_exchange_weak()`.
These retain the existing backend atomic semantics and capability restrictions.

`.native_address()` exports a byte address, and
`ValuePtr<E, S, A>::from_native(address)` imports one on the existing
native-address backends (CUDA/HIP/PTX). Imported addresses must be valid, aligned,
live for the entire kernel, and have the declared address space and access permissions; they carry
no bounds metadata. Native imports are not supported by WGSL, logical SPIR-V,
Metal, or the CPU memref backend. Imported native atomics use the existing
U64-to-atomic-reference support of the direct PTX backend; CUDA/HIP C++ lowering
does not support that conversion.
The original `Ptr` storage-view path, slices, and native operations are unchanged.

## Features

Default features: `frontend-default`.

| Feature | Purpose |
| --- | --- |
| `frontend` | Enable the kernel DSL. |
| `library` | Enable reusable kernels. |
| `device-tensor` | Enable device tensor operations. |
| `source-template` | Enable source-template kernels. |
| `lowering-cpp` | Enable C++ lowering. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-kernel/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-kernel/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/programming-guide.md)
