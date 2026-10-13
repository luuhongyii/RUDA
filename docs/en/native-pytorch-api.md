# Native PyTorch API reference

[Documentation](README.md) · [Package guide](../../ruda-torch/README.md) · [中文](../zh/native-pytorch-api.md)

[All public signatures](python-api-signatures.md) · [Parallel training, data and recovery](parallel-and-data.md)

## Random generators and reproducible state

`Generator(seed=None)` uses Philox4x32-10 with RUDA-owned seed/counter state. `manual_seed(seed)` resets the default generator and returns it; `manual_seed_all(seed)` seeds that same single-device generator. Seeds fit signed/unsigned 64-bit integers. `Generator.manual_seed`, `initial_seed`, `get_state`, and `set_state` manage an explicit generator. State is a 24-byte CPU uint8 tensor with RUDA's version marker, not a CUDA RNG state.

`uniform_(tensor, from_=0., to=1., generator=None)`, `normal_(tensor, mean=0., std=1., generator=None)`, and `bernoulli_(tensor, p=.5, generator=None)` fill FP32/FP16/BF16 RUDA storage in place using native random API 1. Parameters are finite, uniform bounds ordered, standard deviation nonnegative and probability in `[0,1]`. A noncontiguous target uses a contiguous device temporary and copies back. `generator` is a RUDA Generator, not a CPU/CUDA generator; omission uses `default_generator`. Counter reservation precedes dispatch and a failed operation does not reuse its reserved samples.

`get_rng_state(device=None)` / `set_rng_state(state, device=None)` access default state; the `_all` variants use a one-item state list. `get_amp_supported_dtype()` returns FP16/BF16 accepted by the integration, not a GPU capability check. `is_initialized()` reports bridge initialization. `register_random_ops()` is called during import to register ATen random/dropout operations; application code need not register them again.

```python
import torch
import ruda_torch as r

generator = r.Generator(7)
x = torch.empty(8, device="ruda:0", dtype=torch.float32)
saved = generator.get_state()
r.uniform_(x, generator=generator)
first = x.cpu()
generator.set_state(saved)
r.uniform_(x, generator=generator)
torch.testing.assert_close(x.cpu(), first, rtol=0, atol=0)
```

Source: [random generator and registration](../../ruda-torch/python/ruda_torch/random.py), [device/RNG wrappers](../../ruda-torch/python/ruda_torch/__init__.py).

## Device and component contracts

Import `ruda_torch` before creating tensors on `ruda:0`. Another PrivateUse1 backend cannot already be registered in the process. `is_available()` reports the initialized integration, not a complete inventory of GPU instruction support. `device_count()` is 1 and `current_device()` is 0. `synchronize(device=None)` accepts the native device only and waits for completion. `execution_stats()` returns cumulative native launch, transfer and selected-library counters; it is not a timer or an autotuning result.

Rust/C++ components require base ABI 10. Optional features negotiate their own versions: graph 3, training 4, router 1, sequence 1, NF4 decode 1, NF4 matmul 1, scientific 1, BLAS 1, linalg 1, FFT 1, packed quantization 1 and quantized paged attention 1. Paged backward accepts native API 1 or 2 with a compatible bridge; ordered backward requires 2. A base-ABI match alone does not provide every optional feature. Use a matching [source build, release wheel or precompiled bundle](../../ruda-torch/README.md#build-and-install).

The eager native backend does not install a generic CPU fallback. Unsupported operators, unavailable capabilities and device errors propagate. CUDA tensors and RUDA tensors are different allocations and are not interchangeable merely because they use the same GPU.

## Scientific operations

These operations use native RUDA GPU kernels, not a CPU numerical fallback or NVIDIA library ABI wrappers. FP64 and complex storage are provided by scientific API 1. Dtype support is per operation: it does not extend all random, attention, training or graph operators. Basic FP64/complex arithmetic, reductions, matrix multiplication, indexing and real/complex view conversion are available; FP64 `erf` requires `RUDA_CUDA_COMPILER=nvrtc` and is not supported by direct PTX.

### BLAS

| Public function | Input and operation contract |
| --- | --- |
| `ruda_torch.gemm(a, b, c=None, *, alpha=1, beta=0, transpose_a=False, transpose_b=False, conjugate_a=False, conjugate_b=False, f32_math="strict", strategy="auto")` | FP32/FP64/complex64/complex128, matching dtype/device; leading matrix batches broadcast. Computes `alpha*op(a)@op(b)+beta*c`. Transpose and conjugate flags are independent. C has exact output shape; nonzero beta requires C. `f32_math` accepts `strict`/`tf32`; strategies are `auto`/`scalar`. Complex execution uses scalar GPU accumulation. |
| `ruda_torch.fused_gemm(a, b, bias, *, activation="none", strategy="auto", out_dtype=None)` | Matching FP16/BF16 `[M,K]`, `[K,N]` inputs and bias `[N]`. Bias/output may use input dtype or FP32. Activation is `none`, `relu` or tanh-formula `gelu`; one output rounding follows the fused epilogue. Strategies: `auto`, `scalar`, `tensor_core`. |
| `ruda_torch.grouped_gemm(x, weights, row_experts, *, strategy="auto")` | Matching FP32/FP16/BF16 `[M,K]` and `[E,N,K]`, int32/int64 IDs `[M]` in arbitrary row order. Output `[M,N]`; negative/out-of-range IDs are padding with zero output and gradient contribution. Strategies: `auto`, `scalar`, `tensor_core`. |
| `ruda_torch.triangular_solve(a, b, *, left=True, upper=True, transpose=False, unit_diagonal=False, alpha=1.0)` | Real FP32/FP64, square triangular A and matrix B, broadcast batches. Solves `op(A)X=alpha*B` or `X op(A)=alpha*B`. The opposite triangle and a unit diagonal are not read; zero nonunit pivots follow IEEE division semantics. Returns X, not the legacy PyTorch tuple. |

These four public BLAS functions provide first-order autograd and reject higher-order backward. Explicit `tensor_core` requires supported hardware and layouts; unsupported setup raises rather than silently switching strategy. Native operations retain their U32 element/stride indexing limits. Standard `mm`, `bmm` and `addmm` also dispatch FP64/complex tensors through BLAS.

### Linear algebra

Import installs the backend kernels used by the following standard PyTorch APIs on real FP32/FP64 RUDA tensors. Public composite operators retain PyTorch's own autograd formulas.

| Family | Public PyTorch entry points |
| --- | --- |
| LU and reusable solves | `torch.linalg.solve`, `solve_ex`, `lu_factor`, `lu_factor_ex`, `lu_solve`, `torch.lu_solve`, `torch.lu_unpack` |
| Cholesky | `torch.linalg.cholesky`, `cholesky_ex`, `torch.cholesky_solve`, `torch.cholesky_inverse` |
| Triangular systems | `torch.linalg.solve_triangular`, legacy `torch.triangular_solve` |
| Householder QR | `torch.linalg.qr`, `torch.geqrf`, `torch.ormqr` |
| Symmetric eigenproblem | `torch.linalg.eigh`, `eigvalsh` |
| Singular-value decomposition | `torch.linalg.svd`, `svdvals` |
| Inverse and determinants | `torch.linalg.inv`, `inv_ex`, `det`, `slogdet` |
| Full-rank least squares | `torch.linalg.lstsq` |

Leading batches broadcast where the operation allows it. Solve vector RHS follows PyTorch's vector-shape rules; LU pivots exposed to PyTorch are one-based int32. `_ex` variants return int32 `info`, with `check_errors` controlling explicit status checking. Status checking may synchronize and read the small status tensor. Standalone LU/Cholesky factorization retains partial factors on failure; do not treat nonzero info as a usable successful factorization.

QR accepts `reduced`, `complete` and `r`; complete QR and `svd(full_matrices=True)` return full orthogonal bases. `geqrf` and QR mode `r` retain PyTorch's unsupported-backward behavior. SVD uses the native Jacobi implementation with `driver=None`; named NVIDIA drivers such as `gesvd`/`gesvda` are rejected. Least squares uses Householder QR for tall systems or QR of the transpose for wide minimum-norm systems, requires full rank, and accepts only `driver=None`/`gels`; it does not substitute normal equations or a rank-revealing driver. Complex solver inputs are not supported.

### FFT and explicit plans

Standard `torch.fft` APIs dispatch to RUDA without replacing the global Python functions. The same functions are available in `ruda_torch.fft`:

| Dimensionality | Functions |
| --- | --- |
| 1D | `fft`, `ifft`, `rfft`, `irfft`, `hfft`, `ihfft` |
| 2D | `fft2`, `ifft2`, `rfft2`, `irfft2`, `hfft2`, `ihfft2` |
| nD | `fftn`, `ifftn`, `rfftn`, `irfftn`, `hfftn`, `ihfftn` |
| Layout/frequencies | `fftshift`, `ifftshift`, `fftfreq`, `rfftfreq` |

Transforms use FP32/FP64 and complex64/complex128 storage; integer input converts to the selected FP32/FP64 default dtype. FP16/BF16/complex32 are rejected, not silently promoted. Frequencies use FP32/FP64. `n`/`s` crop or zero-pad the transformed dimensions; the final transform axis is the compressed Hermitian axis. Supply the original odd length when expanding compressed input with the `irfft`/`hfft` families. `norm=None`/`backward` normalizes the inverse, `forward` normalizes the forward, and `ortho` uses square-root normalization. Exact non-power-of-two lengths use Bluestein convolution rather than rounding the transform length or executing a CPU DFT.

Transform `out=` variants resize/copy the result but reject autograd when a participating input/output requires gradients. Non-`out` transforms provide native adjoint backward, including higher-order linear gradients. Plans are explicit, queue-bound owners, not an unbounded global cache:

```python
import torch
import ruda_torch as r

a = torch.tensor([[4., 1.], [1., 3.]], dtype=torch.float64).to('ruda:0')
b = torch.ones((2, 1), dtype=a.dtype, device=a.device)
x = torch.linalg.solve(a, b)
product = r.gemm(a, a)
signal = torch.arange(7, dtype=torch.float64, device='ruda:0')
with r.fft.FFTPlan(signal, kind='rfft', s=(7,), dim=(0,), norm='ortho') as plan:
    spectrum = plan.execute(signal)
    retained = plan.retained_bytes
    plan.clear_workspace()
restored = torch.fft.irfft(spectrum, n=7, norm='ortho')
```

`FFTPlan(input, *, kind='fft', s=None, dim=None, norm=None)` accepts `fft`, `ifft`, `rfft`, `irfft`, `hfft` and `ihfft` kinds. Input shape/dtype/device and transform axes/lengths are fixed; signal values may change. `execute(input, *, out=None)` uses the retained plan, `retained_bytes` reports owned storage, `clear_workspace()` releases lazy scratch, and `close()` releases the plan. Context-manager exit calls `close()`.

Sources: [BLAS](../../ruda-torch/python/ruda_torch/blas.py), [linalg registration](../../ruda-torch/python/ruda_torch/linalg.py), [FFT](../../ruda-torch/python/ruda_torch/fft.py), [scientific storage/operations](../../ruda-torch/src/scientific.rs). Native Rust contracts: [ruBLAS](libraries/rublas.md), [ruSOLVER](libraries/rusolver.md), [ruFFT](libraries/rufft.md).

## Normalization and gated activation

The following fused functions require contiguous, dense FP32/FP16/BF16 `ruda:0` tensors without unresolved conjugate/negative views. Outputs retain the activation dtype; saved normalization statistics are FP32. Only first-order derivatives are supported.

| API | Parameters, shape and result |
| --- | --- |
| `rms_norm(x, weight=None, eps=None)` | Normalize the positive last dimension D of `[...,D]`. Optional weight `[D]` on the same device, input dtype or FP32. `eps=None` uses `torch.finfo(x.dtype).eps`; an explicit epsilon must remain finite and positive in FP32. |
| `RMSNorm(width, eps=1e-5, elementwise_affine=True, device=None, dtype=None)` | Positive width; optional affine parameter defaults to FP32. CPU construction is allowed, but move the module to RUDA before forward. `forward(x)` requires last-axis width equality. |
| `layer_norm(x, weight=None, bias=None, eps=1e-5)` | Last-axis normalization. Each supplied affine vector is `[D]`, input dtype or FP32, on the input device; positive finite FP32 epsilon. |
| `LayerNorm(width, eps=1e-5, elementwise_affine=True, bias=True, device=None, dtype=None)` | Positive width; affine storage defaults to FP32. `elementwise_affine=False` disables both parameters; `bias=False` disables bias alone. |
| `silu_mul(gate, up)` | Same shape/dtype/device, no broadcasting. Returns `SiLU(gate) * up` with the low-precision storage-rounding boundary retained. |

Do not mutate saved inputs/weights before backward. `create_graph=True` is rejected by these fused backward interfaces. Their optional FP32 affine parameters are a different contract from explicit StaticGraph RMSNorm, which requires same-dtype weights. Source: [training.py](../../ruda-torch/python/ruda_torch/training.py).

## Optimizer and loss scaling

`AdamW(params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-2, fused_step=False, max_grad_norm=None, hierarchical_stats=False)` requires nonoverlapping, normal leaf RUDA parameters with the same contiguous floating contract above. Sparse gradients and `amsgrad`, `maximize`, `differentiable`, `capturable` are not supported. LR/decay are nonnegative, epsilon positive and both betas in `[0,1)` after FP32 conversion.

- `step(closure=None, loss_scale=1.0)` updates active parameters using FP32 masters/moments. Clear gradients before the next accumulation window. The default path unscales gradients in place and reads a four-byte finite flag before updates.
- `fused_step=True` keeps gradients unchanged, reads a 12-byte statistics report and still uses one update kernel per active parameter. `max_grad_norm` is an optional nonnegative global unscaled L2 limit; it requires this mode.
- `hierarchical_stats=True` also requires fused mode and adds bounded reduction stages for large statistics workspaces.
- Nonfinite gradients skip the whole update. Inspect `last_step_skipped`, `last_step_had_grad`, and, for the fused path, `last_grad_norm`/`last_clip_coef`. This host-readback path is not optimizer graph capture.
- `state_dict()` / `load_state_dict()` preserve optimizer state and supported step options. Restore against matching parameter shapes and dtype contracts.

`GradScaler(init_scale=65536., growth_factor=2., backoff_factor=0.5, growth_interval=2000, min_scale=2**-24, max_scale=2**24)` is RUDA's scaler, not `torch.amp.GradScaler`. Its scale bounds are positive, growth factor greater than one, backoff in `(0,1)` and interval a positive integer. Use `scale(loss).backward()`, `step(optimizer)`, `update()` on one optimizer cycle; the loss must contain one FP32 element on RUDA. `step` accepts RUDA AdamW, Muon or MuonAdamW without a closure or extra arguments. There is no `unscale_()` or multi-optimizer cycle. `get_scale()` reads the current scale; snapshot scaler state only between completed cycles.

For Python Muon parameter grouping, Newton–Schulz options and checkpoint rules, see the [architecture and Python Muon guide](architecture-training.md#python-muon-and-explicit-adamw-exclusions) and [optimizer definitions](../../ruda-torch/python/ruda_torch/optim.py). Do not interpret a local shard update as full-matrix Muon.

## Streams and events

| API | Execution/lifetime contract |
| --- | --- |
| `Stream(priority=0)` | Signed int32 priority; nonzero requests negotiate native support and use the driver-supported range. `get_stream_priority_range()` queries that range and `Stream.priority` reads the effective priority. Reuse the bounded stream pool. |
| `current_stream(device=None)`, `default_stream(device=None)` | Native device only; the default stream has ID zero. |
| `stream(s)` | Context manager that restores the previous stream on exit. |
| `Stream.wait_stream(other)`, `wait_event(event)` | Insert device-side ordering; do not replace them with host readiness queries. |
| `Stream.record_event(event=None)` | Record and return an event on that stream. |
| `Event(enable_timing=False)` | `record(stream=None)` defaults to the current stream; `wait(stream=None)` inserts an event dependency. |
| `Stream.query()`, `Event.query()` | Report readiness without a host completion wait. |
| `Stream.synchronize()`, `Event.synchronize()` | Wait on the host. |
| `start.elapsed_time(end)` | Milliseconds; both events must have timing enabled and their work must be complete. |
| `record_stream(tensor, s)` | Retain allocation for already-submitted work; does not establish a producer/consumer dependency. |
| `Event.close()` | Release the native event explicitly; reuse requires recording again. |

Set `RUDA_TORCH_ASYNC=1` before first native submission to select asynchronous dispatch; it is cached for the process. Scalar extraction/readback and explicit synchronization still wait. See the [stream example](../../ruda-torch/README.md#streams-events-and-asynchronous-dispatch).

## Paged attention and fixed router weights

`PagedAttentionPlan(page_size=..., num_pages=..., block_tables=..., kv_lengths=..., sequence_ids=..., positions=..., splits=1, backward_strategy='atomic')` owns immutable scheduling metadata. Use matching contiguous same-dtype FP32/FP16/BF16 native tensors and execution queue; recreate the plan when its schedule changes.

| Method | Tensor contract |
| --- | --- |
| `attention(q, k, v, scale=..., causal=True)` | Q `[queries, query_heads, features]`; K/V `[physical_pages,page_size,kv_heads,features]` with their respective key/value feature widths. Query heads divisible by KV heads; finite positive scale, finite values, feature widths at most 1024. |
| `mla(absorbed_query, position_query, latent_cache, position_cache, scale=..., causal=True)` | Queries `[queries,heads,rank]` / `[queries,heads,position_dim]`; caches `[pages,page_size,1,rank]` / `[pages,page_size,1,position_dim]`; positional width at most 256. Returns `[queries,heads,rank]` context, before value/output projection. |
| `workspace_bytes(query_heads, value_dim)` | Planned forward split scratch only, not backward memory or total peak VRAM. |

`splits=1` is unsplit; values 2–32 use partial attention and FP32 merging with a 64 MiB per-workspace limit. Caller owns KV content/updates and positional encoding. Use the model's original MLA QK scale. Floating caches support first-order gradients, but not arbitrary external masks or higher-order derivatives. Packed caches use the separate inference-only [quantized path](#quantized-kv-attention). Atomic backward is the default; `backward_strategy='ordered'` explicitly chooses atomic-free history reduction, not autotuning. Detailed ownership/compaction options: [package attention guide](../../ruda-torch/README.md#paged-gqa-and-mla).

`selected_router_weights(logits, indices, scoring='softmax', renormalize=False, scale=1.)` takes contiguous native logits `[tokens,experts]` in FP32/FP16/BF16 and indices `[tokens,top_k]` in int32/int64, with `1 <= top_k <= min(experts,64)`. It returns FP32 selected weights and first-order logits gradients, not expert selection. Softmax covers all experts before gather; sigmoid is pointwise. Optional renormalization covers gathered slots and scale is applied last. Duplicate indices accumulate gradients. Invalid index values yield NaN for the affected forward/backward row with bounds-safe reads rather than a host index synchronization. See [router implementation](../../ruda-torch/python/ruda_torch/_router.py).

## Sequence training and learned fake quantization

`solve_triangular(a, b, upper=False, left=True, unitriangular=False)` accepts same-device, same-floating-dtype matrices. A is `[...,N,N]`; B is `[...,N,K]` for a left solve or `[...,K,N]` for a right solve, with broadcastable batch dimensions. Flags must be Python bools. RUDA uses sequence API 1; CPU/CUDA are explicitly selected references. First-order autograd is provided.

`gated_delta_rule(query, key, value, beta, log_decay, initial_state=None, query_scale=None, normalize_qk=False, norm_eps=1e-6, chunk_size=64, checkpoint_chunks=True, native_forward=True)` requires Q/K `[B,H,T,Dk]`, V `[B,H,T,Dv]`, beta/decay `[B,H,T]`, optional initial state `[B,H,Dk,Dv]` on one floating device. Q/K/V storage dtypes match. `chunk_size` is positive. The result is `(output, final_state)`; output is `[B,H,T,Dv]`. State/intermediates use FP32, or FP64 for explicit CPU FP64 input. The default RUDA forward reuses ruDNN chunk execution; backward recomputes same-device chunk operations and is not a standalone fused DeltaNet backward. Both output and final-state gradients propagate. Exact scale/normalization and empty-sequence behavior: [sequence implementation](../../ruda-torch/python/ruda_torch/sequence_training.py).

`learned_fake_quantize(x, scales, bits=4, block_shape=None, symmetric=True)` accepts nonempty FP32/FP16/BF16 input and same-device FP32 scales. Bits are 2, 4 or 8. No block shape means exactly one scale; otherwise each axis has a positive block size and scales number `product(ceil(shape/block_shape))`, including edge blocks. Scales are floored at `1e-8`; their gradients below the floor are zero. It returns a **floating** tensor with first-order straight-through derivatives, not packed inference weights. `LearnedFakeQuantize(initial_scales, ...)` stores a trainable, checkpointable FP32 scale parameter; keep it FP32 when moving the module. See [quantization.py](../../ruda-torch/python/ruda_torch/quantization.py) and [NF4 fine-tuning](finetuning.md) for the separate packed-base format.

## Packed quantization and checkpoint import

Packed quantization API 1 provides `int2`, `int3`, `int4`, `int5`, `int6`, `int8`, `fp4_e2m1`, `fp8_e4m3` and `fp8_e5m2`. `PackedQuantizedTensor(codes, scales, offsets, shape, format, group_size, group_index=None)` stores row-byte-aligned uint8 codes and FP32 `[rows, ceil(width/group_size)]` scale/offset metadata. Integer codes are unsigned, with signed/asymmetric values represented by offsets. Optional int32 `group_index[width]` maps columns to groups for GPTQ act-order. Storage and metadata are contiguous, frozen and on one device; native execution requires RUDA.

| API | Contract |
| --- | --- |
| `quantize_packed(x, *, format='int4', group_size=128, symmetric=True, calibration='minmax', percentile=99.9, mse_steps=80, scales=None, zero_points=None, offsets=None)` | FP32/FP16/BF16 RUDA input; groups run along the last axis. Returns packed storage. Explicit scales have the metadata shape above; supply zero points or offsets, not both. |
| `PackedLinear(weight, bias=None)` / `PackedLinear.from_linear(linear, **kwargs)` | Weight `[out,in]`; input `[...,in]` and output `[...,out]` use FP32/FP16/BF16. Frozen weights/bias; input gradients are available and LoRA may wrap the layer. |
| `PackedEmbedding(weight, *, padding_idx=None, output_dtype=torch.float16)` | Weight `[vocabulary,width]`; RUDA int32/int64 indices. Decodes selected rows only; no weight training. |
| `weight.dequantize(dtype=torch.float32)` / `weight.to(device)` | Explicit full floating expansion or transfer of the packed representation. |
| `scaled_float8_mm(a, b, scale_a, scale_b, *, out_dtype=torch.float16, bias=None)` | RUDA E4M3FN/E5M2 matrices `[M,K]` and `[K,N]`; FP32 scales are scalar or `[M,1]` / `[1,N]`. Output is FP16/BF16/FP32. |

Packed Linear currently decodes weights inside a scalar dot-product kernel with FP32 accumulation; it is not a Tensor Core GEMM implementation. FP8 storage/casting and scaled matmul do not imply all ATen FP8 operations are supported. `torch._scaled_mm` requires an explicit FP16/BF16/FP32 `out_dtype`; FP8 output and `scale_result` are not supported.

```python
import torch
import ruda_torch as r

weight = torch.ones((64, 128), device="ruda:0", dtype=torch.float16)
layer = r.PackedLinear(r.quantize_packed(weight, format="int4", group_size=32))
x = torch.ones((2, 128), device="ruda:0", dtype=torch.float16)
y = layer(x)
```

### Calibration and affine QAT

`calibrate_quantization(x, *, format='int4', group_size=128, symmetric=True, method='minmax', percentile=99.9, mse_steps=80)` returns FP32 scales and additive offsets. Methods are `minmax`, `percentile` and `mse`; native calibration does not copy samples to CPU. `QuantizationObserver(*, channel_axis=None, averaging_constant=None)` collects per-tensor/per-channel min/max, or EMA with an explicit averaging constant; `calculate_qparams(*, format='int8', symmetric=False)` returns scales and zero points. `affine_fake_quantize(x, scales, zero_points, *, bits=8)` returns floating values with straight-through input/scale gradients and frozen zero points; it does not create packed weights. The existing learned fake quantization and NF4 paths remain separate.

### GPTQ and GGUF

The safetensors loader requires `python -m pip install safetensors`; it is not bundled in the wheel. GGUF parsing uses the package's own reader.

- `gptq_linear(state, prefix, *, in_features, out_features, bits, group_size, checkpoint_format='gptq', desc_act=False, zero_point_wrap=False, device='ruda:0', dtype=torch.float16, chunk_rows=128)` imports standard GPTQ 2/3/4/8-bit weights, scales and `g_idx`. `checkpoint_format` accepts `gptq` (v1 zero-minus-one encoding) or `gptq_v2` (direct zeros); `group_size=-1` means full input width. `zero_point_wrap=True` explicitly selects legacy modulo wrapping at v1 zero-point boundaries. Marlin is not supported.
- `load_gptq_safetensors(model, directory, *, target_modules, config=None, device='ruda:0', dtype=torch.float16, zero_point_wrap=False, chunk_rows=128)` loads local safetensors into an explicitly constructed model with selected Linear modules. No model architecture, key renaming, download or remote code is inferred.
- `GGUFWeights(paths)` reads local GGUF v2/v3 files, including multiple shards. `info(name)` returns `GGUFTensorInfo`; `packed(name, *, device='ruda:0', chunk_rows=64, row_permutation=None)` imports Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1 and Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K. IQ/TQ and other quantizers are not supported.
- `load_gguf(model, paths, *, name_map, device='ruda:0', dtype=torch.float16, row_permutations=None, chunk_rows=64)` requires explicit model-parameter-to-GGUF-name mappings and Q/K RoPE row permutations where layouts differ. Quantized Linear/Embedding weights remain packed; supported dense tensors are loaded separately.

Both importers preserve integer codes and quantization parameters through repacking, without requantizing or expanding full floating weights. GGUF block-scale metadata may expand in RUDA's representation, so packed memory size need not equal the file's byte size. They are weight importers, not automatic model/tokenizer adapters.

### Quantized KV attention

`QuantizedKVCache(*, num_pages, page_size, kv_heads, key_dim, value_dim=None, format='int8', value_format=None, group_size=32, symmetric=True, device='ruda:0')` stores packed physical K/V pages and FP32 metadata. The default value width/format matches the key width/format. `update_(slots, key, value)` packs only new `[tokens,Hkv,D]` values; slots are distinct physical `page*page_size+offset` indices on the cache device. `copy_pages_(source_pages, destination_pages)` copies codes and metadata for explicit page duplication. The caller manages page ownership and must not expose unwritten history in the plan.

`cache.attention(plan, query, *, scale, causal=True)` and `plan.attention(query, cache.key, cache.value, scale=..., causal=True)` dispatch quantized paged MHA/GQA. `plan.mla(absorbed_query, position_query, latent_cache, position_cache, scale=..., causal=True)` accepts packed latent/position caches for MLA. The lower-level `quantized_paged_attention(plan, q, k, v, *, scale, causal=True, position_query=None, position_cache=None)` exposes both forms. Feature groups are contiguous; GPTQ-style `group_index` is not accepted for caches.

Quantized paged API 1 decodes packed history into registers rather than expanding a dense historical cache, retaining chunked prefill/decode and the plan's split/merge scheduling. This path is **inference-only**; use floating caches for backward. Query tensors remain FP32/FP16/BF16, and the existing plan geometry, scale and device constraints apply.

Sources: [packed tensors](../../ruda-torch/python/ruda_torch/packed_quantization.py), [calibration/QAT](../../ruda-torch/python/ruda_torch/quantization_calibration.py), [GPTQ](../../ruda-torch/python/ruda_torch/gptq.py), [GGUF](../../ruda-torch/python/ruda_torch/gguf.py), [quantized cache](../../ruda-torch/python/ruda_torch/quantized_cache.py).

## Compilation, graphs and models

- [Replicated multi-process training](distributed-training.md#python-replicated-training-with-nccl): `ReplicaGroup`, explicit CUDA ordinal, NCCL storage interop, token-weighted synchronization and SFT integration.

- [General model compiler](model-compiler.md): `compile`, `make_backend`, `CompiledModel`, `CompiledFunction`, exact native overloads, shape/stream cache and error policies.
- [Fixed-address static graphs](static-pytorch-graphs.md): `StaticGraph`, `GraphOp`, replay/output lifetime and explicit first-order training.
- [Fine-tuning API](finetuning.md): all public packing, LoRA/NF4 loading, causal supervision, adapter and training-checkpoint entry points.
- [Architecture training guide](architecture-training.md): mHC, DSA/CSA/HCA, functional compressed KV caches, hybrid model construction and Python Muon groups.
- [Hybrid model source](../../ruda-torch/python/ruda_torch/hybrid_model.py): `MHCTransformerBlock`, `HybridAttentionLanguageModel`, `next_token_loss`; model composition rather than a pretrained-checkpoint loader.
- [mHC source](../../ruda-torch/python/ruda_torch/mhc.py): `MHC`, `MHCResidual`, `MHCSequential`, `MHCCoefficients`, `sinkhorn`.
- [Compressed attention source](../../ruda-torch/python/ruda_torch/sparse_attention.py): `LightningIndexer`/`DSAIndexer`, `indexer_kl_loss`, `LearnedKVCompressor`, `CompressedSparseAttention`/`CSA`, `HeavilyCompressedAttention`/`HCA`, `RotaryEmbedding`, `AttentionOutput`, `IndexerOutput`, `CompressedAttentionCache`, `CompressionState`.

Architecture components compose same-device tensor operations; their names do not imply compatibility with every pretrained DeepSeek architecture. API metadata errors normally raise `ValueError`/`TypeError`; native execution errors propagate, and asynchronous errors can surface during synchronization or readback.
