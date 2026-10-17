# PTX Backend Reference

[Documentation](README.md) · [Compiler guide](compiler-guide.md) · [Examples](samples.md) · [中文](../zh/ptx.md) | [日本語](../ja/ptx.md) | [Deutsch](../de/ptx.md) | [Русский](../ru/ptx.md)

## 1. Scope

`ruda_compiler::ptx` generates PTX text from Ruda Kernel IR. It is neither an interpreter for arbitrary PTX programs nor a generator of final NVIDIA machine code.

Enable `ruda-compiler/ptx` for standalone compilation. To select this path through the CUDA runtime, enable `ruda-driver-cuda/direct-ptx`.

## 2. Target types

| Type/field | Meaning |
| --- | --- |
| `PtxTarget::version: (u32, u32)` | PTX major/minor version |
| `PtxTarget::sm: u32` | SM target |
| `PtxCompilationOptions::target: Option<PtxTarget>` | Explicit target configuration |
| `PtxCompiler` | Direct backend implementing the public Compiler trait |

Compilation returns a validation error when `target` is absent. The standalone compiler does not infer GPU architecture from its host machine.

Environment variable parsing accepts `major.minor` with a major version of at least 6 and a minor version no greater than 9. Syntactic validity alone does not establish driver or generator support for that version.

## 3. Compilation output

`PtxKernel` contains:

- `source`: PTX text.
- `entrypoint`: entry point name.
- `ruda_dim`: the original kernel's workgroup dimensions.
- `shared_memory_bytes`: required shared memory.
- `dynamic_metadata_index`: the dynamic metadata pointer argument position, when needed.

Preserve the argument layout, entry point, and shared memory requirements when calling the execution layer. The text alone does not contain the complete launch contract.

## 4. Error handling

Unsupported IR returns `CompilationError::UnsupportedInstruction`. Invalid configuration or structure returns `CompilationError::Validation`. Diagnostics include the `Direct PTX:` prefix. The backend does not automatically fall back to NVRTC.

Definitions: [PTX module](../../ruda-compiler/src/ptx/mod.rs), [compiler tests](../../ruda-compiler/src/ptx/tests.rs), and [runtime backend selection](../../ruda-driver-cuda/src/compiler_backend.rs).

## 5. Packed FP4 transport and NVFP4 scaled MMA

`StorageType::Packed(ElemType::Float(FloatKind::E2M1), 2)` stores two FP4 values per byte. Direct PTX supports bit-preserving copy/index/select/reinterpret and memory transport for this storage. It does not treat a packed byte as a scalar FP4 arithmetic value; unsupported arithmetic or conversion returns an error. Transport alone does not require an `sm_120a` target.

`CoopMma::ExecuteScaled` currently accepts only SM 120 with PTX 8.7 or newer, a `16×8×64` tile, packed E2M1x2 A/B, F32 C/D, E4M3 scales, and `scales_factor = 4`. It emits the `mxf4nvf4.block_scale.scale_vec::4X` instruction and changes the generated target to `sm_120a`. Other architectures, tile shapes, scale formats or factors are rejected; sufficient fragment register capacity is also required.

See [packed operation handling](../../ruda-compiler/src/ptx/packed.rs), [scaled MMA lowering](../../ruda-compiler/src/ptx/matrix/scaled.rs), and the [packed transport/layout example](../../ruda-driver-cuda/examples/ptx_packed.rs). That example consumes exported PTX files and checks byte transport and fragment coordinates; it does not execute NVFP4 MMA arithmetic.
