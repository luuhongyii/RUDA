# Python public API signatures

[Documentation](README.md) · [PyTorch API](native-pytorch-api.md) · [Rust API](rust-api.md)

Constructors and methods defined by RUDA are listed below. Standard inherited PyTorch methods follow their base-class contracts.

## Scientific BLAS and FFT

Contracts and standard `torch.linalg` dispatch: [scientific API](native-pytorch-api.md#scientific-operations).

| Public name / call | Source |
| --- | --- |
| `ruda_torch.gemm(a, b, c=None, *, alpha=1, beta=0, transpose_a=False, transpose_b=False, conjugate_a=False, conjugate_b=False, f32_math='strict', strategy='auto')` | [Source](../../ruda-torch/python/ruda_torch/blas.py) |
| `ruda_torch.fused_gemm(a, b, bias, *, activation='none', strategy='auto', out_dtype=None)` | [Source](../../ruda-torch/python/ruda_torch/blas.py) |
| `ruda_torch.grouped_gemm(x, weights, row_experts, *, strategy='auto')` | [Source](../../ruda-torch/python/ruda_torch/blas.py) |
| `ruda_torch.triangular_solve(a, b, *, left=True, upper=True, transpose=False, unit_diagonal=False, alpha=1.0)` | [Source](../../ruda-torch/python/ruda_torch/blas.py) |
| `ruda_torch.fft.fft(input, n=None, dim=-1, norm=None, *, out=None)` | [Source](../../ruda-torch/python/ruda_torch/fft.py) |
| `ruda_torch.fft.ifft(input, n=None, dim=-1, norm=None, *, out=None)` | [Source](../../ruda-torch/python/ruda_torch/fft.py) |
| `ruda_torch.fft.rfft(input, n=None, dim=-1, norm=None, *, out=None)` | [Source](../../ruda-torch/python/ruda_torch/fft.py) |
| `ruda_torch.fft.irfft(input, n=None, dim=-1, norm=None, *, out=None)` | [Source](../../ruda-torch/python/ruda_torch/fft.py) |
| `ruda_torch.fft.hfft(input, n=None, dim=-1, norm=None, *, out=None)` | [Source](../../ruda-torch/python/ruda_torch/fft.py) |
| `ruda_torch.fft.ihfft(input, n=None, dim=-1, norm=None, *, out=None)` | [Source](../../ruda-torch/python/ruda_torch/fft.py) |
| `ruda_torch.fft.fft2(input, s=None, dim=(-2, -1), norm=None, *, out=None)` | [Source](../../ruda-torch/python/ruda_torch/fft.py) |
| `ruda_torch.fft.ifft2(input, s=None, dim=(-2, -1), norm=None, *, out=None)` | [Source](../../ruda-torch/python/ruda_torch/fft.py) |
| `ruda_torch.fft.rfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None)` | [Source](../../ruda-torch/python/ruda_torch/fft.py) |
| `ruda_torch.fft.irfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None)` | [Source](../../ruda-torch/python/ruda_torch/fft.py) |
| `ruda_torch.fft.hfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None)` | [Source](../../ruda-torch/python/ruda_torch/fft.py) |
| `ruda_torch.fft.ihfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None)` | [Source](../../ruda-torch/python/ruda_torch/fft.py) |
| `ruda_torch.fft.fftn(input, s=None, dim=None, norm=None, *, out=None)` | [Source](../../ruda-torch/python/ruda_torch/fft.py) |
| `ruda_torch.fft.ifftn(input, s=None, dim=None, norm=None, *, out=None)` | [Source](../../ruda-torch/python/ruda_torch/fft.py) |
| `ruda_torch.fft.rfftn(input, s=None, dim=None, norm=None, *, out=None)` | [Source](../../ruda-torch/python/ruda_torch/fft.py) |
| `ruda_torch.fft.irfftn(input, s=None, dim=None, norm=None, *, out=None)` | [Source](../../ruda-torch/python/ruda_torch/fft.py) |
| `ruda_torch.fft.hfftn(input, s=None, dim=None, norm=None, *, out=None)` | [Source](../../ruda-torch/python/ruda_torch/fft.py) |
| `ruda_torch.fft.ihfftn(input, s=None, dim=None, norm=None, *, out=None)` | [Source](../../ruda-torch/python/ruda_torch/fft.py) |
| `ruda_torch.fft.fftshift(input, dim=None)` | [Source](../../ruda-torch/python/ruda_torch/fft.py) |
| `ruda_torch.fft.ifftshift(input, dim=None)` | [Source](../../ruda-torch/python/ruda_torch/fft.py) |
| `ruda_torch.fft.fftfreq(n, d=1.0, *, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False, out=None)` | [Source](../../ruda-torch/python/ruda_torch/fft.py) |
| `ruda_torch.fft.rfftfreq(n, d=1.0, *, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False, out=None)` | [Source](../../ruda-torch/python/ruda_torch/fft.py) |
| `ruda_torch.fft.FFTPlan(input, *, kind='fft', s=None, dim=None, norm=None)` | [Source](../../ruda-torch/python/ruda_torch/fft.py) |
| `ruda_torch.fft.FFTPlan.execute(input, *, out=None)` | [Source](../../ruda-torch/python/ruda_torch/fft.py) |
| `ruda_torch.fft.FFTPlan.clear_workspace()` | [Source](../../ruda-torch/python/ruda_torch/fft.py) |
| `ruda_torch.fft.FFTPlan.close()` | [Source](../../ruda-torch/python/ruda_torch/fft.py) |

`ruda_torch.fft.FFTPlan.retained_bytes` is a read-only property, not a method.

## `__init__`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.is_available()` | [Source](../../ruda-torch/python/ruda_torch/__init__.py#L37) |
| `ruda_torch.is_initialized()` | [Source](../../ruda-torch/python/ruda_torch/__init__.py#L40) |
| `ruda_torch.device_count()` | [Source](../../ruda-torch/python/ruda_torch/__init__.py#L48) |
| `ruda_torch.current_device()` | [Source](../../ruda-torch/python/ruda_torch/__init__.py#L51) |
| `ruda_torch.get_amp_supported_dtype()` | [Source](../../ruda-torch/python/ruda_torch/__init__.py#L57) |
| `ruda_torch.synchronize(device=None)` | [Source](../../ruda-torch/python/ruda_torch/__init__.py#L61) |
| `ruda_torch.manual_seed_all(seed)` | [Source](../../ruda-torch/python/ruda_torch/__init__.py#L85) |
| `ruda_torch.manual_seed(seed)` | [Source](../../ruda-torch/python/ruda_torch/__init__.py#L88) |
| `ruda_torch.get_rng_state(device=None)` | [Source](../../ruda-torch/python/ruda_torch/__init__.py#L91) |
| `ruda_torch.set_rng_state(state, device=None)` | [Source](../../ruda-torch/python/ruda_torch/__init__.py#L94) |
| `ruda_torch.get_rng_state_all()` | [Source](../../ruda-torch/python/ruda_torch/__init__.py#L97) |
| `ruda_torch.set_rng_state_all(states)` | [Source](../../ruda-torch/python/ruda_torch/__init__.py#L100) |
| `ruda_torch.device(index=None)` | [Source](../../ruda-torch/python/ruda_torch/__init__.py#L106) |
| `ruda_torch.execution_stats()` | [Source](../../ruda-torch/python/ruda_torch/__init__.py#L114) |

## `random`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.Generator` | [Source](../../ruda-torch/python/ruda_torch/random.py#L11) |
| `ruda_torch.default_generator` | [Source](../../ruda-torch/python/ruda_torch/random.py#L69) |
| `ruda_torch.uniform_(tensor, from_=0.0, to=1.0, *, generator=None)` | [Source](../../ruda-torch/python/ruda_torch/random.py#L78) |
| `ruda_torch.normal_(tensor, mean=0.0, std=1.0, *, generator=None)` | [Source](../../ruda-torch/python/ruda_torch/random.py#L84) |
| `ruda_torch.bernoulli_(tensor, p=0.5, *, generator=None)` | [Source](../../ruda-torch/python/ruda_torch/random.py#L90) |
| `ruda_torch.register_random_ops()` | [Source](../../ruda-torch/python/ruda_torch/random.py#L111) |
| `ruda_torch.random.native_dropout(input, p, train=True)` | [Source](../../ruda-torch/python/ruda_torch/random.py#L96) |

### `Generator`

| Public name / call | Source |
| --- | --- |
| `Generator(seed=None)` | [Source](../../ruda-torch/python/ruda_torch/random.py#L17) |
| `Generator.manual_seed(seed)` | [Source](../../ruda-torch/python/ruda_torch/random.py#L21) |
| `Generator.initial_seed()` | [Source](../../ruda-torch/python/ruda_torch/random.py#L28) |
| `Generator.get_state()` | [Source](../../ruda-torch/python/ruda_torch/random.py#L31) |
| `Generator.set_state(state)` | [Source](../../ruda-torch/python/ruda_torch/random.py#L35) |
| `Generator.fill_(tensor, distribution, first, second)` | [Source](../../ruda-torch/python/ruda_torch/random.py#L45) |

## `_streams`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.Stream` | [Source](../../ruda-torch/python/ruda_torch/_streams.py#L10) |
| `ruda_torch.Event` | [Source](../../ruda-torch/python/ruda_torch/_streams.py#L27) |
| `ruda_torch.stream(value)` | [Source](../../ruda-torch/python/ruda_torch/_streams.py#L66) |
| `ruda_torch.current_stream(device=None)` | [Source](../../ruda-torch/python/ruda_torch/_streams.py#L56) |
| `ruda_torch.default_stream(device=None)` | [Source](../../ruda-torch/python/ruda_torch/_streams.py#L59) |
| `ruda_torch.record_stream(tensor, value)` | [Source](../../ruda-torch/python/ruda_torch/_streams.py#L74) |

### `Stream`

| Public name / call | Source |
| --- | --- |
| `Stream(*, priority=0, _id=None)` | [Source](../../ruda-torch/python/ruda_torch/_streams.py#L11) |
| `Stream.query()` | [Source](../../ruda-torch/python/ruda_torch/_streams.py#L15) |
| `Stream.synchronize()` | [Source](../../ruda-torch/python/ruda_torch/_streams.py#L17) |
| `Stream.wait_event(event)` | [Source](../../ruda-torch/python/ruda_torch/_streams.py#L19) |
| `Stream.wait_stream(other)` | [Source](../../ruda-torch/python/ruda_torch/_streams.py#L21) |
| `Stream.record_event(event=None)` | [Source](../../ruda-torch/python/ruda_torch/_streams.py#L23) |

### `Event`

| Public name / call | Source |
| --- | --- |
| `Event(*, enable_timing=False)` | [Source](../../ruda-torch/python/ruda_torch/_streams.py#L28) |
| `Event.record(value=None)` | [Source](../../ruda-torch/python/ruda_torch/_streams.py#L30) |
| `Event.wait(value=None)` | [Source](../../ruda-torch/python/ruda_torch/_streams.py#L34) |
| `Event.query()` | [Source](../../ruda-torch/python/ruda_torch/_streams.py#L36) |
| `Event.synchronize()` | [Source](../../ruda-torch/python/ruda_torch/_streams.py#L38) |
| `Event.elapsed_time(other)` | [Source](../../ruda-torch/python/ruda_torch/_streams.py#L40) |
| `Event.close()` | [Source](../../ruda-torch/python/ruda_torch/_streams.py#L45) |
| `Event.__del__()` | [Source](../../ruda-torch/python/ruda_torch/_streams.py#L48) |

## `_paged`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.PagedAttentionPlan` | [Source](../../ruda-torch/python/ruda_torch/_paged.py#L52) |

### `PagedAttentionPlan`

| Public name / call | Source |
| --- | --- |
| `PagedAttentionPlan(*, page_size, num_pages, block_tables, kv_lengths, sequence_ids, positions, splits=1, backward_strategy='atomic')` | [Source](../../ruda-torch/python/ruda_torch/_paged.py#L53) |
| `PagedAttentionPlan.backward_strategy` | [Source](../../ruda-torch/python/ruda_torch/_paged.py#L84) |
| `PagedAttentionPlan.splits` | [Source](../../ruda-torch/python/ruda_torch/_paged.py#L88) |
| `PagedAttentionPlan.workspace_bytes(query_heads, value_dim)` | [Source](../../ruda-torch/python/ruda_torch/_paged.py#L91) |
| `PagedAttentionPlan.attention(q, k, v, *, scale, causal=True)` | [Source](../../ruda-torch/python/ruda_torch/_paged.py#L113) |
| `PagedAttentionPlan.mla(absorbed_query, position_query, latent_cache, position_cache, *, scale, causal=True)` | [Source](../../ruda-torch/python/ruda_torch/_paged.py#L132) |

## `_graph`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.StaticGraph` | [Source](../../ruda-torch/python/ruda_torch/_graph.py#L10) |
| `ruda_torch.GraphOp` | [Source](../../ruda-torch/python/ruda_torch/_graph_spec.py#L17) |

### `StaticGraph`

| Public name / call | Source |
| --- | --- |
| `StaticGraph.from_model(model, **options)` | [Source](../../ruda-torch/python/ruda_torch/_graph.py#L37) |
| `StaticGraph(inputs, nodes, *, outputs=None, infer_dependencies=False, track_completion=False, optimize=False, reuse_workspace=False, training=False)` | [Source](../../ruda-torch/python/ruda_torch/_graph.py#L47) |
| `StaticGraph.replay()` | [Source](../../ruda-torch/python/ruda_torch/_graph.py#L125) |
| `StaticGraph.run_eager()` | [Source](../../ruda-torch/python/ruda_torch/_graph.py#L129) |
| `StaticGraph.synchronize()` | [Source](../../ruda-torch/python/ruda_torch/_graph.py#L137) |
| `StaticGraph.query()` | [Source](../../ruda-torch/python/ruda_torch/_graph.py#L138) |
| `StaticGraph.query_completion()` | [Source](../../ruda-torch/python/ruda_torch/_graph.py#L139) |
| `StaticGraph.wait_completion()` | [Source](../../ruda-torch/python/ruda_torch/_graph.py#L140) |
| `StaticGraph.info` | [Source](../../ruda-torch/python/ruda_torch/_graph.py#L143) |
| `StaticGraph.close()` | [Source](../../ruda-torch/python/ruda_torch/_graph.py#L158) |
| `StaticGraph.__enter__()` | [Source](../../ruda-torch/python/ruda_torch/_graph.py#L164) |
| `StaticGraph.__exit__(*_)` | [Source](../../ruda-torch/python/ruda_torch/_graph.py#L165) |

### `GraphOp`

Fields: `kind: str`; `output: str`; `left: str`; <code>right: str &#124; None = None</code>; `scalar: float = 0.0`; <code>shape: tuple[int, ...] &#124; None = None</code>; <code>dtype: str &#124; None = None</code>.

| Public name / call | Source |
| --- | --- |
| `GraphOp.add(output, left, right, *, alpha=1.0)` | [Source](../../ruda-torch/python/ruda_torch/_graph_spec.py#L34) |
| `GraphOp.mul(output, left, right)` | [Source](../../ruda-torch/python/ruda_torch/_graph_spec.py#L36) |
| `GraphOp.silu(output, value)` | [Source](../../ruda-torch/python/ruda_torch/_graph_spec.py#L38) |
| `GraphOp.silu_mul(output, gate, up)` | [Source](../../ruda-torch/python/ruda_torch/_graph_spec.py#L40) |
| `GraphOp.copy(output, value)` | [Source](../../ruda-torch/python/ruda_torch/_graph_spec.py#L42) |
| `GraphOp.rms_norm(output, value, weight=None, *, eps=1e-05)` | [Source](../../ruda-torch/python/ruda_torch/_graph_spec.py#L44) |

## `training`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.LayerNorm` | [Source](../../ruda-torch/python/ruda_torch/training.py#L141) |
| <code>ruda_torch.layer_norm(x: torch.Tensor, weight: torch.Tensor &#124; None=None, bias: torch.Tensor &#124; None=None, *, eps: float=1e-05)</code> | [Source](../../ruda-torch/python/ruda_torch/training.py#L127) |
| `ruda_torch.RMSNorm` | [Source](../../ruda-torch/python/ruda_torch/training.py#L82) |
| <code>ruda_torch.rms_norm(x: torch.Tensor, weight: torch.Tensor &#124; None=None, *, eps: float &#124; None=None)</code> | [Source](../../ruda-torch/python/ruda_torch/training.py#L65) |
| `ruda_torch.silu_mul(gate: torch.Tensor, up: torch.Tensor)` | [Source](../../ruda-torch/python/ruda_torch/training.py#L183) |
| `ruda_torch.AdamW` | [Source](../../ruda-torch/python/ruda_torch/training.py#L207) |
| `ruda_torch.GradScaler` | [Source](../../ruda-torch/python/ruda_torch/training.py#L521) |

### `LayerNorm`

| Public name / call | Source |
| --- | --- |
| `LayerNorm(width: int, *, eps=1e-05, elementwise_affine=True, bias=True, device=None, dtype=None)` | [Source](../../ruda-torch/python/ruda_torch/training.py#L143) |
| `LayerNorm.forward(x)` | [Source](../../ruda-torch/python/ruda_torch/training.py#L160) |

### `RMSNorm`

| Public name / call | Source |
| --- | --- |
| `RMSNorm(width: int, *, eps=1e-05, elementwise_affine=True, device=None, dtype=None)` | [Source](../../ruda-torch/python/ruda_torch/training.py#L84) |
| `RMSNorm.forward(x)` | [Source](../../ruda-torch/python/ruda_torch/training.py#L95) |

### `AdamW`

| Public name / call | Source |
| --- | --- |
| `AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, *, fused_step=False, max_grad_norm=None, hierarchical_stats=False)` | [Source](../../ruda-torch/python/ruda_torch/training.py#L224) |
| `AdamW.step(closure=None, *, loss_scale=1.0, global_grad_norm=None)` | [Source](../../ruda-torch/python/ruda_torch/training.py#L277) |
| `AdamW.state_dict()` | [Source](../../ruda-torch/python/ruda_torch/training.py#L455) |
| `AdamW.load_state_dict(state_dict)` | [Source](../../ruda-torch/python/ruda_torch/training.py#L466) |

### `GradScaler`

| Public name / call | Source |
| --- | --- |
| `GradScaler(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, min_scale=2.0 ** (-24), max_scale=2.0 ** 24)` | [Source](../../ruda-torch/python/ruda_torch/training.py#L530) |
| `GradScaler.scale(loss: torch.Tensor)` | [Source](../../ruda-torch/python/ruda_torch/training.py#L547) |
| `GradScaler.begin_backward()` | [Source](../../ruda-torch/python/ruda_torch/training.py#L555) |
| `GradScaler.record_step(skipped)` | [Source](../../ruda-torch/python/ruda_torch/training.py#L561) |
| `GradScaler.step(optimizer: AdamW, *args, **kwargs)` | [Source](../../ruda-torch/python/ruda_torch/training.py#L568) |
| `GradScaler.update()` | [Source](../../ruda-torch/python/ruda_torch/training.py#L594) |
| `GradScaler.get_scale()` | [Source](../../ruda-torch/python/ruda_torch/training.py#L607) |
| `GradScaler.state_dict()` | [Source](../../ruda-torch/python/ruda_torch/training.py#L610) |
| `GradScaler.load_state_dict(state)` | [Source](../../ruda-torch/python/ruda_torch/training.py#L617) |

## `_router`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.selected_router_weights(logits: torch.Tensor, indices: torch.Tensor, *, scoring: str='softmax', renormalize: bool=False, scale: float=1.0) -> torch.Tensor` | [Source](../../ruda-torch/python/ruda_torch/_router.py#L49) |

## `compiler`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.compile(model=None, *, capture='aot', native='auto', device_type='ruda', fullgraph=False, dynamic=None, min_native_ops=2, cache_size=4, decompositions=None)` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L365) |
| `ruda_torch.make_backend(*, native='auto', device_type='ruda', min_native_ops=2, cache_size=4, decompositions=None)` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L232) |
| `ruda_torch.CompiledModel` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L242) |
| `ruda_torch.CompiledFunction` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L316) |
| `ruda_torch.NativeCoverageError` | [Source](../../ruda-torch/python/ruda_torch/_compile_native.py#L19) |
| `ruda_torch.GraphExecutionError` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L23) |
| `ruda_torch.compiler.CompileOptions` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L28) |
| `ruda_torch.compiler.RudaBackend` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L155) |

### `CompiledModel`

| Public name / call | Source |
| --- | --- |
| `CompiledModel(model, backend)` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L250) |
| `CompiledModel.original` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L264) |
| `CompiledModel.info` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L268) |
| `CompiledModel.forward(*args, **kwargs)` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L271) |
| `CompiledModel.state_dict(*args, **kwargs)` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L280) |
| `CompiledModel.load_state_dict(state_dict, strict=True, assign=False)` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L289) |
| `CompiledModel.named_parameters(prefix='', recurse=True, remove_duplicate=True)` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L292) |
| `CompiledModel.named_buffers(prefix='', recurse=True, remove_duplicate=True)` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L295) |
| `CompiledModel.__getattr__(name)` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L298) |
| `CompiledModel.close()` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L305) |
| `CompiledModel.__enter__()` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L308) |
| `CompiledModel.__exit__(*_)` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L312) |

### `CompiledFunction`

| Public name / call | Source |
| --- | --- |
| `CompiledFunction(function, backend)` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L318) |
| `CompiledFunction.__call__(*args, **kwargs)` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L324) |
| `CompiledFunction.info` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L333) |
| `CompiledFunction.close()` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L336) |
| `CompiledFunction.__enter__()` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L339) |
| `CompiledFunction.__exit__(*_)` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L343) |

### `NativeCoverageError`

### `GraphExecutionError`

### `CompileOptions`

Fields: `capture: str = 'aot'`; `native: str = 'auto'`; `device_type: str = 'ruda'`; `fullgraph: bool = False`; <code>dynamic: bool &#124; None = None</code>; `min_native_ops: int = 2`; `cache_size: int = 4`.

| Public name / call | Source |
| --- | --- |
| `CompileOptions.__post_init__()` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L37) |

### `RudaBackend`

| Public name / call | Source |
| --- | --- |
| `RudaBackend(options=None, *, decompositions=None)` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L162) |
| `RudaBackend.info` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L172) |
| `RudaBackend.__call__(gm, example_inputs)` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L202) |
| `RudaBackend.close()` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L221) |
| `RudaBackend.__enter__()` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L224) |
| `RudaBackend.__exit__(*_)` | [Source](../../ruda-torch/python/ruda_torch/compiler.py#L228) |

## `quantization`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.LearnedFakeQuantize` | [Source](../../ruda-torch/python/ruda_torch/quantization.py#L84) |
| <code>ruda_torch.learned_fake_quantize(x: Tensor, scales: Tensor, *, bits: int=4, block_shape: Sequence[int] &#124; None=None, symmetric: bool=True) -&gt; Tensor</code> | [Source](../../ruda-torch/python/ruda_torch/quantization.py#L38) |

### `LearnedFakeQuantize`

| Public name / call | Source |
| --- | --- |
| <code>LearnedFakeQuantize(initial_scales: Tensor, *, bits: int=4, block_shape: Sequence[int] &#124; None=None, symmetric: bool=True)</code> | [Source](../../ruda-torch/python/ruda_torch/quantization.py#L90) |
| `LearnedFakeQuantize.forward(x: Tensor) -> Tensor` | [Source](../../ruda-torch/python/ruda_torch/quantization.py#L104) |

## `mhc`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.MHC` | [Source](../../ruda-torch/python/ruda_torch/mhc.py#L44) |
| `ruda_torch.MHCResidual` | [Source](../../ruda-torch/python/ruda_torch/mhc.py#L129) |
| `ruda_torch.MHCSequential` | [Source](../../ruda-torch/python/ruda_torch/mhc.py#L149) |
| `ruda_torch.MHCCoefficients` | [Source](../../ruda-torch/python/ruda_torch/mhc.py#L19) |
| `ruda_torch.sinkhorn(logits: torch.Tensor, iterations: int=20) -> torch.Tensor` | [Source](../../ruda-torch/python/ruda_torch/mhc.py#L25) |

### `MHC`

| Public name / call | Source |
| --- | --- |
| `MHC(width: int, streams: int=4, *, sinkhorn_iterations: int=20, eps: float=1e-06, gate_init: float=0.01, device=None, dtype=None)` | [Source](../../ruda-torch/python/ruda_torch/mhc.py#L51) |
| `MHC.expand(x: torch.Tensor) -> torch.Tensor` | [Source](../../ruda-torch/python/ruda_torch/mhc.py#L75) |
| `MHC.reduce(x: torch.Tensor) -> torch.Tensor` | [Source](../../ruda-torch/python/ruda_torch/mhc.py#L80) |
| `MHC.coefficients(x: torch.Tensor) -> MHCCoefficients` | [Source](../../ruda-torch/python/ruda_torch/mhc.py#L91) |
| `MHC.pre(x: torch.Tensor) -> tuple[torch.Tensor, MHCCoefficients]` | [Source](../../ruda-torch/python/ruda_torch/mhc.py#L104) |
| `MHC.post(x: torch.Tensor, branch_output: torch.Tensor, coefficients: MHCCoefficients) -> torch.Tensor` | [Source](../../ruda-torch/python/ruda_torch/mhc.py#L110) |
| `MHC.forward(x: torch.Tensor, branch: nn.Module, *args, **kwargs) -> torch.Tensor` | [Source](../../ruda-torch/python/ruda_torch/mhc.py#L124) |

### `MHCResidual`

| Public name / call | Source |
| --- | --- |
| `MHCResidual(branch: nn.Module, width: int, streams: int=4, *, checkpoint_branch: bool=False, **kwargs)` | [Source](../../ruda-torch/python/ruda_torch/mhc.py#L131) |
| `MHCResidual.forward(x: torch.Tensor, *args, **kwargs) -> torch.Tensor` | [Source](../../ruda-torch/python/ruda_torch/mhc.py#L140) |

### `MHCSequential`

| Public name / call | Source |
| --- | --- |
| `MHCSequential(width: int, branches, streams: int=4, **kwargs)` | [Source](../../ruda-torch/python/ruda_torch/mhc.py#L151) |
| `MHCSequential.forward(x: torch.Tensor) -> torch.Tensor` | [Source](../../ruda-torch/python/ruda_torch/mhc.py#L159) |

### `MHCCoefficients`

Fields: `pre: torch.Tensor`; `post: torch.Tensor`; `residual: torch.Tensor`.

## `sparse_attention`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.LightningIndexer` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L114) |
| `ruda_torch.DSAIndexer` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L114) |
| <code>ruda_torch.indexer_kl_loss(scores: torch.Tensor, teacher: torch.Tensor, valid: torch.Tensor &#124; None=None) -&gt; torch.Tensor</code> | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L35) |
| `ruda_torch.LearnedKVCompressor` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L294) |
| `ruda_torch.CompressedSparseAttention` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L627) |
| `ruda_torch.HeavilyCompressedAttention` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L633) |
| `ruda_torch.CSA` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L627) |
| `ruda_torch.HCA` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L633) |
| `ruda_torch.AttentionOutput` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L24) |
| `ruda_torch.IndexerOutput` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L29) |
| `ruda_torch.CompressedAttentionCache` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L397) |
| `ruda_torch.CompressionState` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L283) |
| `ruda_torch.RotaryEmbedding` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L69) |

### `LightningIndexer`

| Public name / call | Source |
| --- | --- |
| <code>LightningIndexer(width: int, num_heads: int=4, head_dim: int=16, *, query_dim: int &#124; None=None, topk: int=32, query_chunk_size: int=32, key_chunk_size: int=128, external_keys: bool=False, detach_inputs: bool=True, rope_dim: int=0, eps: float=1e-06, device=None, dtype=None)</code> | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L124) |
| <code>LightningIndexer.project_keys(x: torch.Tensor, pos: torch.Tensor &#124; None=None) -&gt; torch.Tensor</code> | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L149) |
| `LightningIndexer.scores(x, keys, *, query_latent=None, query_positions=None)` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L185) |
| `LightningIndexer.selected_scores(x, keys, indices, *, query_latent=None, query_positions=None)` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L191) |
| `LightningIndexer.select(x, keys, *, allowed=None, query_latent=None, query_positions=None, key_valid=None, query_valid=None, key_end_positions=None)` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L198) |
| `LightningIndexer.forward(x, key_states=None, *, prepared_keys=None, allowed=None, query_latent=None, query_positions=None, causal=True, key_end_positions=None, key_valid=None, query_valid=None)` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L248) |
| `LightningIndexer.distillation_loss(x, prepared_keys, teacher, *, allowed=None, indices=None, query_latent=None, query_positions=None)` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L269) |

### `DSAIndexer`

| Public name / call | Source |
| --- | --- |
| <code>DSAIndexer(width: int, num_heads: int=4, head_dim: int=16, *, query_dim: int &#124; None=None, topk: int=32, query_chunk_size: int=32, key_chunk_size: int=128, external_keys: bool=False, detach_inputs: bool=True, rope_dim: int=0, eps: float=1e-06, device=None, dtype=None)</code> | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L124) |
| <code>DSAIndexer.project_keys(x: torch.Tensor, pos: torch.Tensor &#124; None=None) -&gt; torch.Tensor</code> | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L149) |
| `DSAIndexer.scores(x, keys, *, query_latent=None, query_positions=None)` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L185) |
| `DSAIndexer.selected_scores(x, keys, indices, *, query_latent=None, query_positions=None)` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L191) |
| `DSAIndexer.select(x, keys, *, allowed=None, query_latent=None, query_positions=None, key_valid=None, query_valid=None, key_end_positions=None)` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L198) |
| `DSAIndexer.forward(x, key_states=None, *, prepared_keys=None, allowed=None, query_latent=None, query_positions=None, causal=True, key_end_positions=None, key_valid=None, query_valid=None)` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L248) |
| `DSAIndexer.distillation_loss(x, prepared_keys, teacher, *, allowed=None, indices=None, query_latent=None, query_positions=None)` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L269) |

### `LearnedKVCompressor`

| Public name / call | Source |
| --- | --- |
| `LearnedKVCompressor(width: int, head_dim: int, ratio: int, *, overlap=False, eps=1e-06, device=None, dtype=None)` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L301) |
| `LearnedKVCompressor.forward(x, valid_mask=None)` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L351) |
| <code>LearnedKVCompressor.append(x, valid_mask=None, state: CompressionState &#124; None=None)</code> | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L359) |

### `CompressedSparseAttention`

| Public name / call | Source |
| --- | --- |
| `CompressedSparseAttention(width: int, num_heads: int, **kwargs)` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L629) |
| `CompressedSparseAttention.forward(x, *, valid_mask=None, return_aux=False, indexer_warmup=False)` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L560) |
| <code>CompressedSparseAttention.forward_cached(x, cache: CompressedAttentionCache &#124; None=None, *, valid_mask=None)</code> | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L583) |

### `HeavilyCompressedAttention`

| Public name / call | Source |
| --- | --- |
| `HeavilyCompressedAttention(width: int, num_heads: int, *, compress_ratio=128, **kwargs)` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L635) |
| `HeavilyCompressedAttention.forward(x, *, valid_mask=None, return_aux=False, indexer_warmup=False)` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L560) |
| <code>HeavilyCompressedAttention.forward_cached(x, cache: CompressedAttentionCache &#124; None=None, *, valid_mask=None)</code> | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L583) |

### `CSA`

| Public name / call | Source |
| --- | --- |
| `CSA(width: int, num_heads: int, **kwargs)` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L629) |
| `CSA.forward(x, *, valid_mask=None, return_aux=False, indexer_warmup=False)` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L560) |
| <code>CSA.forward_cached(x, cache: CompressedAttentionCache &#124; None=None, *, valid_mask=None)</code> | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L583) |

### `HCA`

| Public name / call | Source |
| --- | --- |
| `HCA(width: int, num_heads: int, *, compress_ratio=128, **kwargs)` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L635) |
| `HCA.forward(x, *, valid_mask=None, return_aux=False, indexer_warmup=False)` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L560) |
| <code>HCA.forward_cached(x, cache: CompressedAttentionCache &#124; None=None, *, valid_mask=None)</code> | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L583) |

### `AttentionOutput`

Fields: `output: torch.Tensor`; `indexer_loss: torch.Tensor`.

### `IndexerOutput`

Fields: `indices: torch.Tensor`; `scores: torch.Tensor`; `valid: torch.Tensor`.

### `CompressedAttentionCache`

Fields: `owner: int`; `parameter_versions: tuple`; `seen: int`; `compressed: torch.Tensor`; `compressed_valid: torch.Tensor`; <code>index_keys: torch.Tensor &#124; None</code>; `local: torch.Tensor`; `local_valid: torch.Tensor`; `compression: CompressionState`; <code>index_compression: CompressionState &#124; None</code>.

| Public name / call | Source |
| --- | --- |
| `CompressedAttentionCache.reorder(batch_indices: torch.Tensor)` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L409) |
| `CompressedAttentionCache.tensor_bytes` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L420) |

### `CompressionState`

Fields: `tail: torch.Tensor`; `tail_valid: torch.Tensor`; `previous: torch.Tensor`; `previous_valid: torch.Tensor`.

| Public name / call | Source |
| --- | --- |
| `CompressionState.reorder(batch_indices: torch.Tensor)` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L289) |

### `RotaryEmbedding`

| Public name / call | Source |
| --- | --- |
| `RotaryEmbedding(rope_dim: int, base: float=10000.0, *, device=None)` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L75) |
| `RotaryEmbedding.forward(x: torch.Tensor, pos: torch.Tensor, *, inverse: bool=False) -> torch.Tensor` | [Source](../../ruda-torch/python/ruda_torch/sparse_attention.py#L93) |

## `optim`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.Muon` | [Source](../../ruda-torch/python/ruda_torch/optim.py#L144) |
| `ruda_torch.MuonAdamW` | [Source](../../ruda-torch/python/ruda_torch/optim.py#L340) |
| `ruda_torch.muon_orthogonalize(gradient: torch.Tensor, *, ns_steps=5, coefficients=(3.4445, -4.775, 2.0315), eps=1e-07, stable_normalization=True) -> torch.Tensor` | [Source](../../ruda-torch/python/ruda_torch/optim.py#L37) |

### `Muon`

| Public name / call | Source |
| --- | --- |
| `Muon(params, lr=0.02, *, momentum=0.95, weight_decay=0.0, nesterov=True, momentum_mode='sgd', dampening=0.0, ns_steps=5, ns_coefficients=(3.4445, -4.775, 2.0315), eps=1e-07, adjust_lr='original', matrix_layout='as_stored', stable_normalization=True, flatten=False, max_grad_norm=None)` | [Source](../../ruda-torch/python/ruda_torch/optim.py#L155) |
| `Muon.add_param_group(param_group)` | [Source](../../ruda-torch/python/ruda_torch/optim.py#L172) |
| `Muon.step(closure=None, *, loss_scale=1.0)` | [Source](../../ruda-torch/python/ruda_torch/optim.py#L212) |
| `Muon.state_dict()` | [Source](../../ruda-torch/python/ruda_torch/optim.py#L302) |
| `Muon.load_state_dict(state_dict)` | [Source](../../ruda-torch/python/ruda_torch/optim.py#L307) |

### `MuonAdamW`

| Public name / call | Source |
| --- | --- |
| `MuonAdamW(param_groups, **kwargs)` | [Source](../../ruda-torch/python/ruda_torch/optim.py#L342) |
| `MuonAdamW.add_param_group(param_group)` | [Source](../../ruda-torch/python/ruda_torch/optim.py#L172) |
| `MuonAdamW.step(closure=None, *, loss_scale=1.0)` | [Source](../../ruda-torch/python/ruda_torch/optim.py#L212) |
| `MuonAdamW.state_dict()` | [Source](../../ruda-torch/python/ruda_torch/optim.py#L302) |
| `MuonAdamW.load_state_dict(state_dict)` | [Source](../../ruda-torch/python/ruda_torch/optim.py#L307) |
| `MuonAdamW.from_model(model: nn.Module, *, muon_modules, adamw_modules=(), lr=0.02, adamw_lr=0.001, weight_decay=0.0, betas=(0.9, 0.999), **kwargs)` | [Source](../../ruda-torch/python/ruda_torch/optim.py#L349) |

## `hybrid_model`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.MHCTransformerBlock` | [Source](../../ruda-torch/python/ruda_torch/hybrid_model.py#L66) |
| `ruda_torch.HybridAttentionLanguageModel` | [Source](../../ruda-torch/python/ruda_torch/hybrid_model.py#L121) |
| `ruda_torch.HybridAttentionBackbone` | [Source](../../ruda-torch/python/ruda_torch/hybrid_model.py#L204) |
| `ruda_torch.HybridAttentionCache` | [Source](../../ruda-torch/python/ruda_torch/hybrid_model.py#L28) |
| `ruda_torch.next_token_loss(logits: torch.Tensor, tokens: torch.Tensor, *, valid_mask=None)` | [Source](../../ruda-torch/python/ruda_torch/hybrid_model.py#L251) |

### `MHCTransformerBlock`

| Public name / call | Source |
| --- | --- |
| `MHCTransformerBlock(width: int, num_heads: int, *, streams=4, attention_kind='csa', feedforward_width=None, sinkhorn_iterations=20, eps=1e-06, **attention_kwargs)` | [Source](../../ruda-torch/python/ruda_torch/hybrid_model.py#L72) |
| `MHCTransformerBlock.forward(state, *, valid_mask=None, return_aux=False, indexer_warmup=False)` | [Source](../../ruda-torch/python/ruda_torch/hybrid_model.py#L91) |
| `MHCTransformerBlock.forward_cached(state, cache=None, *, valid_mask=None)` | [Source](../../ruda-torch/python/ruda_torch/hybrid_model.py#L105) |

### `HybridAttentionLanguageModel`

| Public name / call | Source |
| --- | --- |
| `HybridAttentionLanguageModel(vocab_size: int, width: int, num_heads: int, num_layers: int, *, streams=4, csa_ratio=4, hca_ratio=128, tie_embeddings=False, sinkhorn_iterations=20, eps=1e-06, **attention_kwargs)` | [Source](../../ruda-torch/python/ruda_torch/hybrid_model.py#L129) |
| `HybridAttentionLanguageModel.forward(tokens, *, valid_mask=None, return_aux=False, indexer_warmup=False)` | [Source](../../ruda-torch/python/ruda_torch/hybrid_model.py#L151) |
| `HybridAttentionLanguageModel.forward_hidden(tokens, *, valid_mask=None, return_aux=False, indexer_warmup=False)` | [Source](../../ruda-torch/python/ruda_torch/hybrid_model.py#L155) |
| `HybridAttentionLanguageModel.hidden_backbone()` | [Source](../../ruda-torch/python/ruda_torch/hybrid_model.py#L159) |
| `HybridAttentionLanguageModel.forward_hidden_cached(tokens, cache=None, *, valid_mask=None)` | [Source](../../ruda-torch/python/ruda_torch/hybrid_model.py#L163) |
| `HybridAttentionLanguageModel.forward_cached(tokens, cache=None, *, valid_mask=None)` | [Source](../../ruda-torch/python/ruda_torch/hybrid_model.py#L194) |
| `HybridAttentionLanguageModel.forward_cached_last(tokens, cache=None, *, valid_mask=None)` | [Source](../../ruda-torch/python/ruda_torch/hybrid_model.py#L198) |

### `HybridAttentionBackbone`

| Public name / call | Source |
| --- | --- |
| `HybridAttentionBackbone(model)` | [Source](../../ruda-torch/python/ruda_torch/hybrid_model.py#L212) |
| `HybridAttentionBackbone.forward(input_ids, attention_mask=None, *, position_ids=None, cu_seqlens=None, max_seqlen=None)` | [Source](../../ruda-torch/python/ruda_torch/hybrid_model.py#L220) |

### `HybridAttentionCache`

Fields: `owner: int`; `signature: tuple`; `seen: int`; `layers: tuple[CompressedAttentionCache, ...]`.

| Public name / call | Source |
| --- | --- |
| `HybridAttentionCache.reorder(batch_indices)` | [Source](../../ruda-torch/python/ruda_torch/hybrid_model.py#L35) |
| `HybridAttentionCache.tensor_bytes` | [Source](../../ruda-torch/python/ruda_torch/hybrid_model.py#L43) |

## `sequence_training`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.gated_delta_rule(query, key, value, beta, log_decay, *, initial_state=None, query_scale=None, normalize_qk=False, norm_eps=1e-06, chunk_size=64, checkpoint_chunks=True, native_forward=True)` | [Source](../../ruda-torch/python/ruda_torch/sequence_training.py#L137) |
| `ruda_torch.solve_triangular(a, b, *, upper=False, left=True, unitriangular=False)` | [Source](../../ruda-torch/python/ruda_torch/sequence_training.py#L52) |
| `ruda_torch.sequence_training.triangular_mask(x, diagonal=0, *, upper=False)` | [Source](../../ruda-torch/python/ruda_torch/sequence_training.py#L11) |

## `finetuning`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.LoRALinear` | [Source](../../ruda-torch/python/ruda_torch/finetuning.py#L210) |
| `ruda_torch.NF4Linear` | [Source](../../ruda-torch/python/ruda_torch/finetuning.py#L138) |
| `ruda_torch.inject_lora(model, *, target_modules, rank=16, alpha=16.0, adapter_dtype=torch.float32, dropout=0.0, use_rslora=False)` | [Source](../../ruda-torch/python/ruda_torch/finetuning.py#L289) |
| `ruda_torch.quantize_nf4(model, *, target_modules, block_size=64, tile_rows=128)` | [Source](../../ruda-torch/python/ruda_torch/finetuning.py#L322) |
| `ruda_torch.adapter_state_dict(model)` | [Source](../../ruda-torch/python/ruda_torch/finetuning.py#L358) |
| `ruda_torch.load_adapter_state_dict(model, state)` | [Source](../../ruda-torch/python/ruda_torch/finetuning.py#L388) |
| `ruda_torch.merge_lora(model)` | [Source](../../ruda-torch/python/ruda_torch/finetuning.py#L429) |
| `ruda_torch.load_nf4_safetensors(model, directory, *, target_modules, device, dtype=torch.bfloat16, block_size=64, tile_rows=128, parameter_dtypes=None, buffer_dtypes=None)` | [Source](../../ruda-torch/python/ruda_torch/finetuning.py#L471) |
| `ruda_torch.finetune_state_dict(model, optimizer, *, base_id, step, data_state, scaler=None)` | [Source](../../ruda-torch/python/ruda_torch/finetuning.py#L620) |
| `ruda_torch.load_finetune_state_dict(model, optimizer, state, *, base_id, scaler=None)` | [Source](../../ruda-torch/python/ruda_torch/finetuning.py#L646) |
| `ruda_torch.finetuning.pack_nf4(weight, *, block_size=64, chunk_blocks=1024)` | [Source](../../ruda-torch/python/ruda_torch/finetuning.py#L40) |

### `LoRALinear`

| Public name / call | Source |
| --- | --- |
| `LoRALinear(base, *, rank=16, alpha=16.0, adapter_dtype=torch.float32, dropout=0.0, use_rslora=False)` | [Source](../../ruda-torch/python/ruda_torch/finetuning.py#L212) |
| `LoRALinear.get_extra_state()` | [Source](../../ruda-torch/python/ruda_torch/finetuning.py#L238) |
| `LoRALinear.scaling` | [Source](../../ruda-torch/python/ruda_torch/finetuning.py#L245) |
| `LoRALinear.set_extra_state(state)` | [Source](../../ruda-torch/python/ruda_torch/finetuning.py#L248) |
| `LoRALinear.forward(x)` | [Source](../../ruda-torch/python/ruda_torch/finetuning.py#L252) |

### `NF4Linear`

| Public name / call | Source |
| --- | --- |
| `NF4Linear(in_features, out_features, packed, scales, *, block_size=64, tile_rows=128, bias=None)` | [Source](../../ruda-torch/python/ruda_torch/finetuning.py#L146) |
| `NF4Linear.from_linear(linear, *, block_size=64, tile_rows=128)` | [Source](../../ruda-torch/python/ruda_torch/finetuning.py#L172) |
| `NF4Linear.get_extra_state()` | [Source](../../ruda-torch/python/ruda_torch/finetuning.py#L189) |
| `NF4Linear.set_extra_state(state)` | [Source](../../ruda-torch/python/ruda_torch/finetuning.py#L193) |
| `NF4Linear.forward(x)` | [Source](../../ruda-torch/python/ruda_torch/finetuning.py#L197) |

## `causal_finetuning`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.chunked_lm_cross_entropy(hidden, head, labels, *, token_chunk_size=32, ignore_index=-100, shift=True, reduction='mean', recompute=True, label_smoothing=0.0)` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L25) |
| `ruda_torch.SFTCollator` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L85) |
| `ruda_torch.CausalLMFinetuner` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L181) |
| `ruda_torch.SFTTrainer` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L303) |
| `ruda_torch.load_hf_nf4_model(directory, *, target_modules, device, dtype=torch.bfloat16, auto_class=None, rank=16, alpha=16.0, block_size=64, tile_rows=128, adapter_dtype=torch.float32, parameter_dtypes=None, config_kwargs=None, model_kwargs=None)` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L261) |
| `ruda_torch.activation_checkpoint_modules(model, target_modules, *, preserve_rng_state=True)` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L150) |
| `ruda_torch.PackedCausalLMFinetuner` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L232) |
| `ruda_torch.CausalLMTrainer` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L669) |

### `SFTCollator`

| Public name / call | Source |
| --- | --- |
| `SFTCollator(tokenizer=None, *, max_length, pad_token_id=None, train_on_prompt=False, truncate=False, template_kwargs=None)` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L92) |
| `SFTCollator.encode(sample)` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L105) |
| `SFTCollator.__call__(samples)` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L135) |

### `CausalLMFinetuner`

| Public name / call | Source |
| --- | --- |
| `CausalLMFinetuner(backbone, head, *, token_chunk_size=32, activation_checkpointing=True, checkpoint_modules=None, preserve_rng_state=True, label_smoothing=0.0)` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L183) |
| `CausalLMFinetuner.hidden(input_ids, attention_mask)` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L208) |
| `CausalLMFinetuner.forward(input_ids, attention_mask, labels, *, reduction='mean', label_smoothing=None)` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L225) |

### `SFTTrainer`

| Public name / call | Source |
| --- | --- |
| `SFTTrainer(model, optimizer, *, base_id, run_config, scaler=None, scheduler=None, replica_group=None, gradient_overlap=False, bucket_bytes=25 * 1024 * 1024, sampler=None)` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L310) |
| `SFTTrainer.train_step(microbatches)` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L374) |
| `SFTTrainer.train_step_autotuned(microbatches, *, autotuner=None)` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L464) |
| `SFTTrainer.evaluate(microbatches)` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L475) |
| `SFTTrainer.save(directory)` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L562) |
| `SFTTrainer.resume(path)` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L590) |
| `SFTTrainer.save_distributed(directory)` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L617) |
| `SFTTrainer.resume_distributed(directory, *, generation=None)` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L631) |
| `SFTTrainer.write_progress(directory, metrics, *, total_steps=None)` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L651) |

### `PackedCausalLMFinetuner`

| Public name / call | Source |
| --- | --- |
| `PackedCausalLMFinetuner(backbone, head, *, token_chunk_size=32, activation_checkpointing=True, checkpoint_modules=None, preserve_rng_state=True, label_smoothing=0.0)` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L183) |
| `PackedCausalLMFinetuner.hidden(input_ids, attention_mask)` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L208) |
| `PackedCausalLMFinetuner.forward(input_ids, attention_mask, labels, position_ids, cu_seqlens, max_seqlen, *, reduction='mean', label_smoothing=None)` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L242) |

### `CausalLMTrainer`

| Public name / call | Source |
| --- | --- |
| `CausalLMTrainer(model, optimizer, *, base_id, run_config, scaler=None, scheduler=None, replica_group=None, gradient_overlap=False, bucket_bytes=25 * 1024 * 1024, sampler=None)` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L310) |
| `CausalLMTrainer.train_step(microbatches)` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L374) |
| `CausalLMTrainer.train_step_autotuned(microbatches, *, autotuner=None)` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L464) |
| `CausalLMTrainer.evaluate(microbatches)` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L475) |
| `CausalLMTrainer.save(directory)` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L562) |
| `CausalLMTrainer.resume(path)` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L590) |
| `CausalLMTrainer.save_distributed(directory)` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L617) |
| `CausalLMTrainer.resume_distributed(directory, *, generation=None)` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L631) |
| `CausalLMTrainer.write_progress(directory, metrics, *, total_steps=None)` | [Source](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L651) |

## `distributed_training`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.ReplicaGroup` | [Source](../../ruda-torch/python/ruda_torch/distributed_training.py#L13) |

### `ReplicaGroup`

| Public name / call | Source |
| --- | --- |
| `ReplicaGroup(process_group=None, *, device_type='ruda')` | [Source](../../ruda-torch/python/ruda_torch/distributed_training.py#L20) |
| `ReplicaGroup.gather_metadata(value)` | [Source](../../ruda-torch/python/ruda_torch/distributed_training.py#L52) |
| `ReplicaGroup.sum_(tensor)` | [Source](../../ruda-torch/python/ruda_torch/distributed_training.py#L74) |
| `ReplicaGroup.max_(tensor)` | [Source](../../ruda-torch/python/ruda_torch/distributed_training.py#L81) |
| `ReplicaGroup.min_(tensor)` | [Source](../../ruda-torch/python/ruda_torch/distributed_training.py#L87) |
| `ReplicaGroup.broadcast_(tensor, root=0)` | [Source](../../ruda-torch/python/ruda_torch/distributed_training.py#L93) |
| `ReplicaGroup.all_gather(tensor, *, axis=0)` | [Source](../../ruda-torch/python/ruda_torch/distributed_training.py#L103) |
| `ReplicaGroup.reduce_scatter(tensor, *, axis=0)` | [Source](../../ruda-torch/python/ruda_torch/distributed_training.py#L116) |
| `ReplicaGroup.begin_gradient_overlap(*, global_weight, normalized=False, bucket_bytes=25 * 1024 * 1024)` | [Source](../../ruda-torch/python/ruda_torch/distributed_training.py#L131) |
| `ReplicaGroup.initialize(model, *, root=0, broadcast_buffers=True)` | [Source](../../ruda-torch/python/ruda_torch/distributed_training.py#L141) |
| `ReplicaGroup.validate_model(model)` | [Source](../../ruda-torch/python/ruda_torch/distributed_training.py#L188) |
| `ReplicaGroup.total_weight(local_weight)` | [Source](../../ruda-torch/python/ruda_torch/distributed_training.py#L199) |
| `ReplicaGroup.synchronize_gradients(*, local_weight, normalized=False, missing='error')` | [Source](../../ruda-torch/python/ruda_torch/distributed_training.py#L209) |
| `ReplicaGroup.validate_training_options(value)` | [Source](../../ruda-torch/python/ruda_torch/distributed_training.py#L265) |

## `parallel_training`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.ColumnParallelLinear` | [Source](../../ruda-torch/python/ruda_torch/parallel_training.py#L57) |
| `ruda_torch.RowParallelLinear` | [Source](../../ruda-torch/python/ruda_torch/parallel_training.py#L81) |
| `ruda_torch.VocabParallelEmbedding` | [Source](../../ruda-torch/python/ruda_torch/parallel_training.py#L104) |
| `ruda_torch.TensorParallelMLP` | [Source](../../ruda-torch/python/ruda_torch/parallel_training.py#L131) |
| `ruda_torch.TensorParallelAttention` | [Source](../../ruda-torch/python/ruda_torch/parallel_training.py#L147) |
| `ruda_torch.tensor_parallelize(model, group, plan)` | [Source](../../ruda-torch/python/ruda_torch/parallel_training.py#L205) |
| `ruda_torch.tensor_parallel_state_dict(model, group)` | [Source](../../ruda-torch/python/ruda_torch/parallel_training.py#L273) |
| `ruda_torch.load_tensor_parallel_state_dict(model, state, group, *, strict=True)` | [Source](../../ruda-torch/python/ruda_torch/parallel_training.py#L296) |
| `ruda_torch.copy_to_tensor_parallel(value, group)` | [Source](../../ruda-torch/python/ruda_torch/parallel_training.py#L41) |
| `ruda_torch.reduce_from_tensor_parallel(value, group)` | [Source](../../ruda-torch/python/ruda_torch/parallel_training.py#L45) |
| `ruda_torch.gather_from_tensor_parallel(value, group, *, axis=-1)` | [Source](../../ruda-torch/python/ruda_torch/parallel_training.py#L49) |
| `ruda_torch.scatter_to_tensor_parallel(value, group, *, axis=-1)` | [Source](../../ruda-torch/python/ruda_torch/parallel_training.py#L53) |
| `ruda_torch.parallel_training.GradientOverlap` | [Source](../../ruda-torch/python/ruda_torch/parallel_training.py#L324) |

### `ColumnParallelLinear`

| Public name / call | Source |
| --- | --- |
| `ColumnParallelLinear(linear, group, *, gather_output=True)` | [Source](../../ruda-torch/python/ruda_torch/parallel_training.py#L59) |
| `ColumnParallelLinear.forward(value)` | [Source](../../ruda-torch/python/ruda_torch/parallel_training.py#L75) |

### `RowParallelLinear`

| Public name / call | Source |
| --- | --- |
| `RowParallelLinear(linear, group, *, input_is_parallel=False)` | [Source](../../ruda-torch/python/ruda_torch/parallel_training.py#L83) |
| `RowParallelLinear.forward(value)` | [Source](../../ruda-torch/python/ruda_torch/parallel_training.py#L97) |

### `VocabParallelEmbedding`

| Public name / call | Source |
| --- | --- |
| `VocabParallelEmbedding(embedding, group)` | [Source](../../ruda-torch/python/ruda_torch/parallel_training.py#L106) |
| `VocabParallelEmbedding.forward(tokens)` | [Source](../../ruda-torch/python/ruda_torch/parallel_training.py#L121) |

### `TensorParallelMLP`

| Public name / call | Source |
| --- | --- |
| `TensorParallelMLP(gate, up, down, group, *, activation=None)` | [Source](../../ruda-torch/python/ruda_torch/parallel_training.py#L133) |
| `TensorParallelMLP.forward(value)` | [Source](../../ruda-torch/python/ruda_torch/parallel_training.py#L143) |

### `TensorParallelAttention`

| Public name / call | Source |
| --- | --- |
| `TensorParallelAttention(query, key, value, output, group, *, num_heads, num_kv_heads=None, head_dim=None, scale=None, dropout_p=0.0, attention_impl=None)` | [Source](../../ruda-torch/python/ruda_torch/parallel_training.py#L153) |
| `TensorParallelAttention.forward(hidden, *, attention_mask=None, is_causal=False, position_transform=None)` | [Source](../../ruda-torch/python/ruda_torch/parallel_training.py#L179) |

### `GradientOverlap`

| Public name / call | Source |
| --- | --- |
| `GradientOverlap(group, *, global_weight, normalized, bucket_bytes)` | [Source](../../ruda-torch/python/ruda_torch/parallel_training.py#L326) |
| `GradientOverlap.finish(*, local_weight, missing='zero')` | [Source](../../ruda-torch/python/ruda_torch/parallel_training.py#L370) |

## `sharded_training`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.FullyShardedModule` | [Source](../../ruda-torch/python/ruda_torch/sharded_training.py#L91) |
| `ruda_torch.fully_shard(model, group, *, unit_paths=None, device=None, root=0, recompute=True, shard_buffers=None, ensure_backward=False)` | [Source](../../ruda-torch/python/ruda_torch/sharded_training.py#L305) |
| `ruda_torch.Zero2Optimizer` | [Source](../../ruda-torch/python/ruda_torch/sharded_training.py#L346) |
| `ruda_torch.ShardedReplicaGroup` | [Source](../../ruda-torch/python/ruda_torch/sharded_training.py#L474) |

### `FullyShardedModule`

| Public name / call | Source |
| --- | --- |
| `FullyShardedModule(module, group, *, device=None, root=0, recompute=True, shard_buffers=None, ensure_backward=False)` | [Source](../../ruda-torch/python/ruda_torch/sharded_training.py#L101) |
| `FullyShardedModule.forward(*args, **kwargs)` | [Source](../../ruda-torch/python/ruda_torch/sharded_training.py#L218) |
| `FullyShardedModule.train(mode=True)` | [Source](../../ruda-torch/python/ruda_torch/sharded_training.py#L226) |
| `FullyShardedModule.extra_repr()` | [Source](../../ruda-torch/python/ruda_torch/sharded_training.py#L233) |
| `FullyShardedModule.get_extra_state()` | [Source](../../ruda-torch/python/ruda_torch/sharded_training.py#L236) |
| `FullyShardedModule.set_extra_state(state)` | [Source](../../ruda-torch/python/ruda_torch/sharded_training.py#L243) |
| `FullyShardedModule.full_state_dict()` | [Source](../../ruda-torch/python/ruda_torch/sharded_training.py#L249) |
| `FullyShardedModule.load_full_state_dict(state, *, strict=True)` | [Source](../../ruda-torch/python/ruda_torch/sharded_training.py#L266) |

### `Zero2Optimizer`

| Public name / call | Source |
| --- | --- |
| `Zero2Optimizer(parameter_groups, group, optimizer_factory)` | [Source](../../ruda-torch/python/ruda_torch/sharded_training.py#L353) |
| `Zero2Optimizer.synchronize_gradients(*, local_weight, normalized=False, missing='zero')` | [Source](../../ruda-torch/python/ruda_torch/sharded_training.py#L387) |
| `Zero2Optimizer.step(closure=None, **kwargs)` | [Source](../../ruda-torch/python/ruda_torch/sharded_training.py#L419) |
| `Zero2Optimizer.zero_grad(set_to_none=True)` | [Source](../../ruda-torch/python/ruda_torch/sharded_training.py#L447) |
| `Zero2Optimizer.state_dict()` | [Source](../../ruda-torch/python/ruda_torch/sharded_training.py#L455) |
| `Zero2Optimizer.load_state_dict(state)` | [Source](../../ruda-torch/python/ruda_torch/sharded_training.py#L459) |

### `ShardedReplicaGroup`

| Public name / call | Source |
| --- | --- |
| `ShardedReplicaGroup(model, group)` | [Source](../../ruda-torch/python/ruda_torch/sharded_training.py#L481) |
| `ShardedReplicaGroup.gather_metadata(value)` | [Source](../../ruda-torch/python/ruda_torch/sharded_training.py#L494) |
| `ShardedReplicaGroup.validate_training_options(value)` | [Source](../../ruda-torch/python/ruda_torch/sharded_training.py#L497) |
| `ShardedReplicaGroup.total_weight(value)` | [Source](../../ruda-torch/python/ruda_torch/sharded_training.py#L500) |
| `ShardedReplicaGroup.sum_(tensor)` | [Source](../../ruda-torch/python/ruda_torch/sharded_training.py#L503) |
| `ShardedReplicaGroup.validate_model(model)` | [Source](../../ruda-torch/python/ruda_torch/sharded_training.py#L506) |
| `ShardedReplicaGroup.validate_microbatches(batches)` | [Source](../../ruda-torch/python/ruda_torch/sharded_training.py#L511) |
| `ShardedReplicaGroup.synchronize_gradients(*, local_weight, normalized=False, missing='zero')` | [Source](../../ruda-torch/python/ruda_torch/sharded_training.py#L518) |
| `ShardedReplicaGroup.prepare_optimizer_step(optimizer, *, loss_scale=1.0)` | [Source](../../ruda-torch/python/ruda_torch/sharded_training.py#L535) |

## `pipeline_training`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.PipelineStage` | [Source](../../ruda-torch/python/ruda_torch/pipeline_training.py#L31) |
| `ruda_torch.PipelineTensorSpec` | [Source](../../ruda-torch/python/ruda_torch/pipeline_training.py#L12) |

### `PipelineStage`

| Public name / call | Source |
| --- | --- |
| `PipelineStage(module, group, *, input_spec, output_spec)` | [Source](../../ruda-torch/python/ruda_torch/pipeline_training.py#L41) |
| `PipelineStage.invocation_contract` | [Source](../../ruda-torch/python/ruda_torch/pipeline_training.py#L92) |
| `PipelineStage.validate_interfaces(count, microbatch_specs=None)` | [Source](../../ruda-torch/python/ruda_torch/pipeline_training.py#L94) |
| `PipelineStage.run(inputs=None, targets=None, *, loss_sum, global_weight, loss_scale=1.0, microbatch_specs=None, schedule='1f1b')` | [Source](../../ruda-torch/python/ruda_torch/pipeline_training.py#L149) |

### `PipelineTensorSpec`

Fields: `shape: tuple[int, ...]`; `dtype: torch.dtype = torch.float32`.

| Public name / call | Source |
| --- | --- |
| `PipelineTensorSpec.validate()` | [Source](../../ruda-torch/python/ruda_torch/pipeline_training.py#L16) |
| `PipelineTensorSpec.allocate(device)` | [Source](../../ruda-torch/python/ruda_torch/pipeline_training.py#L22) |
| `PipelineTensorSpec.has_floating` | [Source](../../ruda-torch/python/ruda_torch/pipeline_training.py#L27) |

## `structured_pipeline`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.StructuredPipelineStage` | [Source](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L149) |
| `ruda_torch.PipelineTreeSpec` | [Source](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L58) |
| `ruda_torch.PipelineLeafSpec` | [Source](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L33) |
| `ruda_torch.PipelineConstant` | [Source](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L21) |

### `StructuredPipelineStage`

| Public name / call | Source |
| --- | --- |
| `StructuredPipelineStage(module, group, *, input_spec, output_spec, input_mode='single')` | [Source](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L165) |
| `StructuredPipelineStage.invocation_contract` | [Source](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L227) |
| `StructuredPipelineStage.validate_interfaces(count, microbatch_specs=None)` | [Source](../../ruda-torch/python/ruda_torch/pipeline_training.py#L94) |
| `StructuredPipelineStage.run(inputs=None, targets=None, *, loss_sum, global_weight, loss_scale=1.0, microbatch_specs=None, schedule='1f1b')` | [Source](../../ruda-torch/python/ruda_torch/pipeline_training.py#L149) |

### `PipelineTreeSpec`

Fields: `leaves: tuple`; `tree: object`.

| Public name / call | Source |
| --- | --- |
| `PipelineTreeSpec(structure)` | [Source](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L71) |
| `PipelineTreeSpec.from_flat(tree, leaves)` | [Source](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L79) |
| `PipelineTreeSpec.validate()` | [Source](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L94) |
| `PipelineTreeSpec.has_floating` | [Source](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L101) |
| `PipelineTreeSpec.allocate(device)` | [Source](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L104) |
| `PipelineTreeSpec.matches(value)` | [Source](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L109) |

### `PipelineLeafSpec`

| Public name / call | Source |
| --- | --- |
| `PipelineLeafSpec.validate()` | [Source](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L45) |
| `PipelineLeafSpec.allocate(device)` | [Source](../../ruda-torch/python/ruda_torch/pipeline_training.py#L22) |
| `PipelineLeafSpec.has_floating` | [Source](../../ruda-torch/python/ruda_torch/pipeline_training.py#L27) |
| `PipelineLeafSpec(shape, dtype=torch.float32)` | [Source](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L40) |
| `PipelineLeafSpec.__eq__(other)` | [Source](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L51) |
| `PipelineLeafSpec.__hash__()` | [Source](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L54) |

### `PipelineConstant`

Fields: `value: object`.

| Public name / call | Source |
| --- | --- |
| `PipelineConstant.__post_init__()` | [Source](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L25) |
| `PipelineConstant.__eq__(other)` | [Source](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L27) |
| `PipelineConstant.__hash__()` | [Source](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L30) |

## `distributed_checkpoint`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.DistributedCheckpoint` | [Source](../../ruda-torch/python/ruda_torch/distributed_checkpoint.py#L129) |
| `ruda_torch.model_shard_layout(model)` | [Source](../../ruda-torch/python/ruda_torch/distributed_checkpoint.py#L47) |

### `DistributedCheckpoint`

| Public name / call | Source |
| --- | --- |
| `DistributedCheckpoint(group, directory)` | [Source](../../ruda-torch/python/ruda_torch/distributed_checkpoint.py#L137) |
| `DistributedCheckpoint.save(model, optimizer, *, step, application_state=None, scheduler=None, scaler=None)` | [Source](../../ruda-torch/python/ruda_torch/distributed_checkpoint.py#L140) |
| `DistributedCheckpoint.load(model, optimizer, *, generation=None, scheduler=None, scaler=None, validate_application=None)` | [Source](../../ruda-torch/python/ruda_torch/distributed_checkpoint.py#L196) |
| `DistributedCheckpoint.consolidate(*, generation=None)` | [Source](../../ruda-torch/python/ruda_torch/distributed_checkpoint.py#L256) |
| `DistributedCheckpoint.load_consolidated(model, optimizer, state, *, reshard_application, scheduler=None, scaler=None)` | [Source](../../ruda-torch/python/ruda_torch/distributed_checkpoint.py#L307) |

## `adapter_interop`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.load_peft_adapter(model, directory, *, module_map=None, prefix='base_model.model.', adapter_dtype=torch.float32)` | [Source](../../ruda-torch/python/ruda_torch/adapter_interop.py#L19) |
| `ruda_torch.save_peft_adapter(model, directory, *, base_model_name_or_path, prefix='base_model.model.', task_type=None, module_map=None, replica_group=None)` | [Source](../../ruda-torch/python/ruda_torch/adapter_interop.py#L119) |

## `attention`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.scaled_dot_product_attention(query, key, value, *, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False)` | [Source](../../ruda-torch/python/ruda_torch/attention.py#L9) |

## `block_attention`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.block_scaled_dot_product_attention(query, key, value, *, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, query_block_size=128, key_block_size=256, generator=None, causal_offset=0, window_size=None)` | [Source](../../ruda-torch/python/ruda_torch/block_attention.py#L162) |

## `parallel_loss`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.vocab_parallel_cross_entropy(logits, targets, group, *, vocab_start=None, global_vocab_size=None, ignore_index=-100, label_smoothing=0.0, reduction='mean')` | [Source](../../ruda-torch/python/ruda_torch/parallel_loss.py#L56) |

## `transformers_attention`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.register_transformers_attention(name='ruda_block', *, query_block_size=128, key_block_size=256)` | [Source](../../ruda-torch/python/ruda_torch/transformers_attention.py#L7) |

## `parallel_mesh`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.ParallelMesh` | [Source](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L11) |
| `ruda_torch.TensorParallelGroup` | [Source](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L94) |

### `ParallelMesh`

| Public name / call | Source |
| --- | --- |
| `ParallelMesh(*, data_parallel=1, tensor_parallel=1, pipeline_parallel=1, device_type='ruda', timeout_seconds=1800)` | [Source](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L18) |
| `ParallelMesh.data` | [Source](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L63) |
| `ParallelMesh.tensor` | [Source](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L66) |
| `ParallelMesh.pipeline` | [Source](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L69) |
| `ParallelMesh.stage` | [Source](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L72) |
| `ParallelMesh.stage_mesh()` | [Source](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L74) |
| `ParallelMesh.close()` | [Source](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L78) |

### `TensorParallelGroup`

| Public name / call | Source |
| --- | --- |
| `TensorParallelGroup(model, group)` | [Source](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L101) |
| `TensorParallelGroup.gather_metadata(value)` | [Source](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L108) |
| `TensorParallelGroup.validate_training_options(value)` | [Source](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L110) |
| `TensorParallelGroup.validate_model(model)` | [Source](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L112) |
| `TensorParallelGroup.total_weight(local_weight)` | [Source](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L117) |
| `TensorParallelGroup.validate_microbatches(batches)` | [Source](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L123) |
| `TensorParallelGroup.sum_(tensor)` | [Source](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L129) |
| `TensorParallelGroup.synchronize_gradients(*, local_weight, normalized=False, missing='zero')` | [Source](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L133) |
| `TensorParallelGroup.prepare_optimizer_step(optimizer, *, loss_scale=1.0)` | [Source](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L147) |

## `sharded_optim`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.ShardedMuon` | [Source](../../ruda-torch/python/ruda_torch/sharded_optim.py#L49) |
| `ruda_torch.distributed_grad_norm(parameters, group, *, loss_scale=1.0, replicated_ids=(), replication_factors=None)` | [Source](../../ruda-torch/python/ruda_torch/sharded_optim.py#L11) |

### `ShardedMuon`

| Public name / call | Source |
| --- | --- |
| `ShardedMuon(params, group, lr=0.02, *, momentum=0.95, weight_decay=0.0, nesterov=True, momentum_mode='sgd', dampening=0.0, ns_steps=5, ns_coefficients=(3.4445, -4.775, 2.0315), eps=1e-07, adjust_lr='original', matrix_layout='as_stored', stable_normalization=True, flatten=False, max_grad_norm=None, owners=None)` | [Source](../../ruda-torch/python/ruda_torch/sharded_optim.py#L60) |
| `ShardedMuon.step(closure=None, *, loss_scale=1.0)` | [Source](../../ruda-torch/python/ruda_torch/sharded_optim.py#L94) |
| `ShardedMuon.state_dict()` | [Source](../../ruda-torch/python/ruda_torch/sharded_optim.py#L173) |
| `ShardedMuon.load_state_dict(record)` | [Source](../../ruda-torch/python/ruda_torch/sharded_optim.py#L179) |

## `quantization_interop`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.bnb_nf4_linear(state, weight_key, *, device='ruda:0', dtype=torch.float16, bias=None, tile_rows=128)` | [Source](../../ruda-torch/python/ruda_torch/quantization_interop.py#L12) |
| `ruda_torch.load_bnb_nf4_safetensors(model, directory, *, target_modules, device='ruda:0', dtype=torch.float16, tile_rows=128)` | [Source](../../ruda-torch/python/ruda_torch/quantization_interop.py#L98) |

## `parallel_adapters`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.ColumnParallelLoRALinear` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L169) |
| `ruda_torch.RowParallelLoRALinear` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L191) |
| `ruda_torch.ColumnParallelNF4Linear` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L92) |
| `ruda_torch.RowParallelNF4Linear` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L106) |
| `ruda_torch.shard_nf4_linear(layer, group, *, axis)` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L35) |
| `ruda_torch.parallel_adapters.lora_from_parallel_base(base, *, rank=16, alpha=16.0, adapter_dtype=torch.float32, dropout=0.0, use_rslora=False)` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L213) |
| `ruda_torch.parallel_adapters.merge_parallel_lora(layer)` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L266) |
| `ruda_torch.parallel_adapters.parallel_linear(source, group, *, axis, **options)` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L283) |
| `ruda_torch.parallel_adapters.parameter_shards(source, replacement, axis)` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L294) |
| `ruda_torch.parallel_adapters.adapter_base_kind(layer)` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L307) |
| `ruda_torch.parallel_adapters.full_adapter_matrix(layer, name)` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L311) |
| `ruda_torch.parallel_adapters.local_adapter_matrix(layer, name, value)` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L317) |
| `ruda_torch.parallel_adapters.nf4_layout(module, prefix)` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L325) |
| `ruda_torch.parallel_adapters.join_nf4_field(values, spec)` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L339) |
| `ruda_torch.parallel_adapters.slice_nf4_field(value, spec, rank, world)` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L371) |
| `ruda_torch.parallel_adapters.full_nf4_state(module)` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L401) |
| `ruda_torch.parallel_adapters.local_nf4_state(module, state, prefix)` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L420) |

### `ColumnParallelLoRALinear`

| Public name / call | Source |
| --- | --- |
| `ColumnParallelLoRALinear.scaling` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L158) |
| `ColumnParallelLoRALinear.get_extra_state()` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L160) |
| `ColumnParallelLoRALinear.set_extra_state(state)` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L165) |
| `ColumnParallelLoRALinear(layer, group, *, gather_output=True)` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L172) |
| `ColumnParallelLoRALinear.forward(value)` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L181) |

### `RowParallelLoRALinear`

| Public name / call | Source |
| --- | --- |
| `RowParallelLoRALinear.scaling` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L158) |
| `RowParallelLoRALinear.get_extra_state()` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L160) |
| `RowParallelLoRALinear.set_extra_state(state)` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L165) |
| `RowParallelLoRALinear(layer, group, *, input_is_parallel=False)` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L194) |
| `RowParallelLoRALinear.forward(value)` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L203) |

### `ColumnParallelNF4Linear`

| Public name / call | Source |
| --- | --- |
| `ColumnParallelNF4Linear.get_extra_state()` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L84) |
| `ColumnParallelNF4Linear.set_extra_state(state)` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L88) |
| `ColumnParallelNF4Linear(layer, group, *, gather_output=True)` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L95) |
| `ColumnParallelNF4Linear.forward(value)` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L101) |

### `RowParallelNF4Linear`

| Public name / call | Source |
| --- | --- |
| `RowParallelNF4Linear.get_extra_state()` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L84) |
| `RowParallelNF4Linear.set_extra_state(state)` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L88) |
| `RowParallelNF4Linear(layer, group, *, input_is_parallel=False)` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L109) |
| `RowParallelNF4Linear.forward(value)` | [Source](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L116) |

## `parallel_checkpoint_io`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.load_tensor_parallel_safetensors(model, directory, group, *, device, dtype=torch.bfloat16, nf4_targets=(), block_size=64, tile_rows=128, parameter_dtypes=None, buffer_dtypes=None, nf4_format='dense')` | [Source](../../ruda-torch/python/ruda_torch/parallel_checkpoint_io.py#L14) |
| `ruda_torch.load_tensor_parallel_bnb_nf4_safetensors(model, directory, group, *, target_modules, device='ruda:0', dtype=torch.float16, tile_rows=128, parameter_dtypes=None, buffer_dtypes=None)` | [Source](../../ruda-torch/python/ruda_torch/parallel_checkpoint_io.py#L175) |

## `hybrid_parallel`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.DataTensorParallelGroup` | [Source](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L10) |
| `ruda_torch.hybrid_parallel.join_mesh_field(values, spec)` | [Source](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L142) |
| `ruda_torch.hybrid_parallel.slice_mesh_field(value, spec, rank, world)` | [Source](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L173) |

### `DataTensorParallelGroup`

| Public name / call | Source |
| --- | --- |
| `DataTensorParallelGroup(model, mesh, *, data_root=0, tensor_root=0, broadcast_buffers=True)` | [Source](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L18) |
| `DataTensorParallelGroup.gather_metadata(value)` | [Source](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L41) |
| `DataTensorParallelGroup.validate_training_options(value)` | [Source](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L43) |
| `DataTensorParallelGroup.validate_model(model)` | [Source](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L45) |
| `DataTensorParallelGroup.total_weight(local_weight)` | [Source](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L49) |
| `DataTensorParallelGroup.validate_microbatches(batches)` | [Source](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L60) |
| `DataTensorParallelGroup.sum_(value)` | [Source](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L67) |
| `DataTensorParallelGroup.synchronize_gradients(*, local_weight, normalized=False, missing='zero')` | [Source](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L71) |
| `DataTensorParallelGroup.begin_gradient_overlap(*, global_weight, normalized=False, bucket_bytes=25 * 1024 * 1024)` | [Source](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L90) |
| `DataTensorParallelGroup.replication_factors(parameters)` | [Source](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L93) |
| `DataTensorParallelGroup.prepare_optimizer_step(optimizer, *, loss_scale=1.0)` | [Source](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L97) |
| `DataTensorParallelGroup.checkpoint_layout(layout)` | [Source](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L132) |

## `tensor_parallel_optim`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.TensorParallelMuon` | [Source](../../ruda-torch/python/ruda_torch/tensor_parallel_optim.py#L13) |
| `ruda_torch.MeshShardedMuon` | [Source](../../ruda-torch/python/ruda_torch/tensor_parallel_optim.py#L220) |

### `TensorParallelMuon`

| Public name / call | Source |
| --- | --- |
| `TensorParallelMuon(params, model, group, lr=0.02, *, momentum=0.95, weight_decay=0.0, nesterov=True, momentum_mode='sgd', dampening=0.0, ns_steps=5, ns_coefficients=(3.4445, -4.775, 2.0315), eps=1e-07, adjust_lr='original', matrix_layout='as_stored', stable_normalization=True, flatten=False, max_grad_norm=None, owners=None, consensus_group=None)` | [Source](../../ruda-torch/python/ruda_torch/tensor_parallel_optim.py#L25) |
| `TensorParallelMuon.step(closure=None, *, loss_scale=1.0)` | [Source](../../ruda-torch/python/ruda_torch/tensor_parallel_optim.py#L89) |
| `TensorParallelMuon.state_dict()` | [Source](../../ruda-torch/python/ruda_torch/tensor_parallel_optim.py#L182) |
| `TensorParallelMuon.load_state_dict(record)` | [Source](../../ruda-torch/python/ruda_torch/tensor_parallel_optim.py#L188) |

### `MeshShardedMuon`

| Public name / call | Source |
| --- | --- |
| `MeshShardedMuon(params, model, mesh, lr=0.02, *, momentum=0.95, weight_decay=0.0, nesterov=True, momentum_mode='sgd', dampening=0.0, ns_steps=5, ns_coefficients=(3.4445, -4.775, 2.0315), eps=1e-07, adjust_lr='original', matrix_layout='as_stored', stable_normalization=True, flatten=False, max_grad_norm=None, owners=None)` | [Source](../../ruda-torch/python/ruda_torch/tensor_parallel_optim.py#L232) |
| `MeshShardedMuon.step(closure=None, *, loss_scale=1.0)` | [Source](../../ruda-torch/python/ruda_torch/tensor_parallel_optim.py#L89) |
| `MeshShardedMuon.state_dict()` | [Source](../../ruda-torch/python/ruda_torch/tensor_parallel_optim.py#L182) |
| `MeshShardedMuon.load_state_dict(record)` | [Source](../../ruda-torch/python/ruda_torch/tensor_parallel_optim.py#L188) |

## `sharded_mesh`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.ShardedDataTensorParallelGroup` | [Source](../../ruda-torch/python/ruda_torch/sharded_mesh.py#L118) |
| `ruda_torch.sharded_mesh.sharded_mesh_layout(model, mesh)` | [Source](../../ruda-torch/python/ruda_torch/sharded_mesh.py#L22) |
| `ruda_torch.sharded_mesh.join_fsdp_extra(values, spec)` | [Source](../../ruda-torch/python/ruda_torch/sharded_mesh.py#L84) |
| `ruda_torch.sharded_mesh.slice_fsdp_extra(value, spec, rank, world)` | [Source](../../ruda-torch/python/ruda_torch/sharded_mesh.py#L103) |

### `ShardedDataTensorParallelGroup`

| Public name / call | Source |
| --- | --- |
| `ShardedDataTensorParallelGroup(model, mesh, *, data_root=0, tensor_root=0, broadcast_buffers=True)` | [Source](../../ruda-torch/python/ruda_torch/sharded_mesh.py#L128) |
| `ShardedDataTensorParallelGroup.gather_metadata(value)` | [Source](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L41) |
| `ShardedDataTensorParallelGroup.validate_training_options(value)` | [Source](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L43) |
| `ShardedDataTensorParallelGroup.validate_model(model)` | [Source](../../ruda-torch/python/ruda_torch/sharded_mesh.py#L171) |
| `ShardedDataTensorParallelGroup.total_weight(local_weight)` | [Source](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L49) |
| `ShardedDataTensorParallelGroup.validate_microbatches(batches)` | [Source](../../ruda-torch/python/ruda_torch/sharded_mesh.py#L175) |
| `ShardedDataTensorParallelGroup.sum_(value)` | [Source](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L67) |
| `ShardedDataTensorParallelGroup.synchronize_gradients(*, local_weight, normalized=False, missing='zero')` | [Source](../../ruda-torch/python/ruda_torch/sharded_mesh.py#L186) |
| `ShardedDataTensorParallelGroup.begin_gradient_overlap(**options)` | [Source](../../ruda-torch/python/ruda_torch/sharded_mesh.py#L183) |
| `ShardedDataTensorParallelGroup.replication_factors(parameters)` | [Source](../../ruda-torch/python/ruda_torch/sharded_mesh.py#L193) |
| `ShardedDataTensorParallelGroup.prepare_optimizer_step(optimizer, *, loss_scale=1.0)` | [Source](../../ruda-torch/python/ruda_torch/sharded_mesh.py#L198) |
| `ShardedDataTensorParallelGroup.checkpoint_layout(layout)` | [Source](../../ruda-torch/python/ruda_torch/sharded_mesh.py#L219) |

## `pipeline_trainer`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.PipelineTrainer` | [Source](../../ruda-torch/python/ruda_torch/pipeline_trainer.py#L15) |

### `PipelineTrainer`

| Public name / call | Source |
| --- | --- |
| `PipelineTrainer(stage, optimizer, mesh, *, stage_name, base_id, run_config, scaler=None, scheduler=None, input_is_parallel=False, tied_parameters=None)` | [Source](../../ruda-torch/python/ruda_torch/pipeline_trainer.py#L24) |
| `PipelineTrainer.train_step(inputs=None, targets=None, *, local_weight, loss_sum, microbatch_specs=None, schedule='1f1b')` | [Source](../../ruda-torch/python/ruda_torch/pipeline_trainer.py#L62) |
| `PipelineTrainer.train_step_autotuned(inputs=None, targets=None, *, local_weight, loss_sum, microbatch_specs=None, autotuner=None)` | [Source](../../ruda-torch/python/ruda_torch/pipeline_trainer.py#L161) |
| `PipelineTrainer.save_distributed(directory, *, application_state=None)` | [Source](../../ruda-torch/python/ruda_torch/pipeline_trainer.py#L193) |
| `PipelineTrainer.resume_distributed(directory, *, generation=None)` | [Source](../../ruda-torch/python/ruda_torch/pipeline_trainer.py#L240) |
| `PipelineTrainer.consolidate(directory, *, generation=None)` | [Source](../../ruda-torch/python/ruda_torch/pipeline_trainer.py#L265) |
| `PipelineTrainer.resume_consolidated(state, *, reshard_application, stage_transform=None)` | [Source](../../ruda-torch/python/ruda_torch/pipeline_trainer.py#L272) |

## `pipeline_ties`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.PipelineTiedParameters` | [Source](../../ruda-torch/python/ruda_torch/pipeline_ties.py#L11) |

### `PipelineTiedParameters`

| Public name / call | Source |
| --- | --- |
| `PipelineTiedParameters(model, mesh, bindings)` | [Source](../../ruda-torch/python/ruda_torch/pipeline_ties.py#L20) |
| `PipelineTiedParameters.validate_optimizer(optimizer)` | [Source](../../ruda-torch/python/ruda_torch/pipeline_ties.py#L73) |
| `PipelineTiedParameters.synchronize_gradients()` | [Source](../../ruda-torch/python/ruda_torch/pipeline_ties.py#L83) |
| `PipelineTiedParameters.replication_count(parameter)` | [Source](../../ruda-torch/python/ruda_torch/pipeline_ties.py#L92) |
| `PipelineTiedParameters.synchronize_state(optimizer)` | [Source](../../ruda-torch/python/ruda_torch/pipeline_ties.py#L97) |
| `PipelineTiedParameters.close()` | [Source](../../ruda-torch/python/ruda_torch/pipeline_ties.py#L120) |

## `training_data`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.StatefulShardSampler` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L9) |
| `ruda_torch.TokenBudgetBatchSampler` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L148) |
| `ruda_torch.collate_varlen_causal_lm(samples, *, ignore_index=-100)` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L287) |

### `StatefulShardSampler`

| Public name / call | Source |
| --- | --- |
| `StatefulShardSampler(length, *, source_id, rank=0, replicas=1, shuffle=False, seed=0, tail='uneven')` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L18) |
| `StatefulShardSampler.from_mesh(length, mesh, **options)` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L33) |
| `StatefulShardSampler.__len__()` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L47) |
| `StatefulShardSampler.remaining` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L52) |
| `StatefulShardSampler.__iter__()` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L54) |
| `StatefulShardSampler.commit(count)` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L61) |
| `StatefulShardSampler.rewind_uncommitted()` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L66) |
| `StatefulShardSampler.set_epoch(epoch)` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L70) |
| `StatefulShardSampler.state_dict()` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L78) |
| `StatefulShardSampler.load_state_dict(state)` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L86) |
| `StatefulShardSampler.validate_state_dict(state)` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L110) |
| `StatefulShardSampler.consolidate(states)` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L116) |
| `StatefulShardSampler.from_consolidated(state, *, rank, replicas)` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L137) |

### `TokenBudgetBatchSampler`

| Public name / call | Source |
| --- | --- |
| `TokenBudgetBatchSampler(sampler, lengths, *, maximum_tokens, layout, maximum_examples=None)` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L162) |
| `TokenBudgetBatchSampler.committed` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L178) |
| `TokenBudgetBatchSampler.issued` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L182) |
| `TokenBudgetBatchSampler.remaining_examples` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L186) |
| `TokenBudgetBatchSampler.peek()` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L210) |
| `TokenBudgetBatchSampler.__iter__()` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L214) |
| `TokenBudgetBatchSampler.__len__()` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L219) |
| `TokenBudgetBatchSampler.commit(examples)` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L227) |
| `TokenBudgetBatchSampler.rewind_uncommitted()` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L230) |
| `TokenBudgetBatchSampler.set_epoch(epoch)` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L233) |
| `TokenBudgetBatchSampler.state_dict()` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L236) |
| `TokenBudgetBatchSampler.validate_state_dict(state)` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L240) |
| `TokenBudgetBatchSampler.load_state_dict(state)` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L253) |
| `TokenBudgetBatchSampler.consolidate(states)` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L258) |
| `TokenBudgetBatchSampler.from_consolidated(state, *, rank, replicas)` | [Source](../../ruda-torch/python/ruda_torch/training_data.py#L279) |

## `varlen_attention`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.varlen_scaled_dot_product_attention(query, key, value, cu_seqlens_q, cu_seqlens_k=None, *, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, query_block_size=128, key_block_size=256, generator=None, window_size=None, causal_alignment='upper_left')` | [Source](../../ruda-torch/python/ruda_torch/varlen_attention.py#L17) |

## `training_checkpoint`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.training_state_dict(model, optimizer, *, base_id, step, data_state, scaler=None)` | [Source](../../ruda-torch/python/ruda_torch/training_checkpoint.py#L42) |
| `ruda_torch.validate_training_state_dict(model, optimizer, state, *, base_id, scaler=None)` | [Source](../../ruda-torch/python/ruda_torch/training_checkpoint.py#L76) |
| `ruda_torch.load_training_state_dict(model, optimizer, state, *, base_id, scaler=None)` | [Source](../../ruda-torch/python/ruda_torch/training_checkpoint.py#L105) |

## `training_autotune`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.TrainingAutotuner` | [Source](../../ruda-torch/python/ruda_torch/training_autotune.py#L278) |
| `ruda_torch.TrainingPlan` | [Source](../../ruda-torch/python/ruda_torch/training_autotune.py#L50) |
| `ruda_torch.PipelineAutotuner` | [Source](../../ruda-torch/python/ruda_torch/training_autotune.py#L515) |

### `TrainingAutotuner`

| Public name / call | Source |
| --- | --- |
| `TrainingAutotuner(*, repeats=3, max_candidates=12, max_seconds=300.0, rtol=0.0, atol=0.0, max_snapshot_bytes=None, progress=None)` | [Source](../../ruda-torch/python/ruda_torch/training_autotune.py#L290) |
| `TrainingAutotuner.tune(trainer, microbatches)` | [Source](../../ruda-torch/python/ruda_torch/training_autotune.py#L388) |
| `TrainingAutotuner.train_step(trainer, microbatches)` | [Source](../../ruda-torch/python/ruda_torch/training_autotune.py#L493) |

### `TrainingPlan`

Fields: `examples_per_microbatch: int`; <code>token_chunk_size: int &#124; None</code>; `packed: bool`; `geometry: tuple`.

| Public name / call | Source |
| --- | --- |
| `TrainingPlan.apply(trainer, batches)` | [Source](../../ruda-torch/python/ruda_torch/training_autotune.py#L57) |

### `PipelineAutotuner`

| Public name / call | Source |
| --- | --- |
| `PipelineAutotuner(*, repeats=3, max_candidates=12, max_seconds=300.0, rtol=0.0, atol=0.0, max_snapshot_bytes=None, progress=None)` | [Source](../../ruda-torch/python/ruda_torch/training_autotune.py#L290) |
| `PipelineAutotuner.tune(trainer, inputs=None, targets=None, *, local_weight, loss_sum, microbatch_specs=None)` | [Source](../../ruda-torch/python/ruda_torch/training_autotune.py#L528) |
| `PipelineAutotuner.train_step(trainer, inputs=None, targets=None, *, local_weight, loss_sum, microbatch_specs=None)` | [Source](../../ruda-torch/python/ruda_torch/training_autotune.py#L607) |

## `sharded_adapter_interop`

| Public name / call | Source |
| --- | --- |
| `ruda_torch.sharded_adapter_interop.has_sharded_units(model)` | [Source](../../ruda-torch/python/ruda_torch/sharded_adapter_interop.py#L11) |
| `ruda_torch.sharded_adapter_interop.adapter_layers(model, *, remove_duplicate=True)` | [Source](../../ruda-torch/python/ruda_torch/sharded_adapter_interop.py#L15) |
| `ruda_torch.sharded_adapter_interop.sharded_adapter_state_dict(model)` | [Source](../../ruda-torch/python/ruda_torch/sharded_adapter_interop.py#L54) |
| `ruda_torch.sharded_adapter_interop.load_sharded_adapter_state_dict(model, state)` | [Source](../../ruda-torch/python/ruda_torch/sharded_adapter_interop.py#L83) |
| `ruda_torch.sharded_adapter_interop.load_sharded_peft_adapter(model, pairs, config, *, module_map=None)` | [Source](../../ruda-torch/python/ruda_torch/sharded_adapter_interop.py#L122) |
