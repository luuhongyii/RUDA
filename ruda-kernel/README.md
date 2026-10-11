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
