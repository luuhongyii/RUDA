# Python 公開 API シグネチャ

[ドキュメント](README.md) · [PyTorch API](native-pytorch-api.md) · [Rust API](rust-api.md)

RUDA が定義するコンストラクタとメソッドを示します。PyTorch から継承する標準メソッドには基底クラスの契約が適用されます。

Packed 量子化、校正／QAT、GPTQ／GGUF、量子化 KV キャッシュのシグネチャは [packed API 参照](../en/native-pytorch-api.md#packed-quantization-and-checkpoint-import)を参照してください。

## 数学バックエンドの選択

| API | Source |
| --- | --- |
| `ruda_torch.get_math_backend()` | [Source](../../ruda-torch/python/ruda_torch/math_backend.py) |
| `ruda_torch.set_math_backend(name)` | [Source](../../ruda-torch/python/ruda_torch/math_backend.py) |
| `ruda_torch.math_backend(name)` | [Source](../../ruda-torch/python/ruda_torch/math_backend.py) |

## `__init__`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.is_available()` | [ソース](../../ruda-torch/python/ruda_torch/__init__.py#L37) |
| `ruda_torch.is_initialized()` | [ソース](../../ruda-torch/python/ruda_torch/__init__.py#L40) |
| `ruda_torch.device_count()` | [ソース](../../ruda-torch/python/ruda_torch/__init__.py#L48) |
| `ruda_torch.current_device()` | [ソース](../../ruda-torch/python/ruda_torch/__init__.py#L51) |
| `ruda_torch.get_amp_supported_dtype()` | [ソース](../../ruda-torch/python/ruda_torch/__init__.py#L57) |
| `ruda_torch.synchronize(device=None)` | [ソース](../../ruda-torch/python/ruda_torch/__init__.py#L61) |
| `ruda_torch.manual_seed_all(seed)` | [ソース](../../ruda-torch/python/ruda_torch/__init__.py#L85) |
| `ruda_torch.manual_seed(seed)` | [ソース](../../ruda-torch/python/ruda_torch/__init__.py#L88) |
| `ruda_torch.get_rng_state(device=None)` | [ソース](../../ruda-torch/python/ruda_torch/__init__.py#L91) |
| `ruda_torch.set_rng_state(state, device=None)` | [ソース](../../ruda-torch/python/ruda_torch/__init__.py#L94) |
| `ruda_torch.get_rng_state_all()` | [ソース](../../ruda-torch/python/ruda_torch/__init__.py#L97) |
| `ruda_torch.set_rng_state_all(states)` | [ソース](../../ruda-torch/python/ruda_torch/__init__.py#L100) |
| `ruda_torch.device(index=None)` | [ソース](../../ruda-torch/python/ruda_torch/__init__.py#L106) |
| `ruda_torch.execution_stats()` | [ソース](../../ruda-torch/python/ruda_torch/__init__.py#L114) |

## `random`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.Generator` | [ソース](../../ruda-torch/python/ruda_torch/random.py#L11) |
| `ruda_torch.default_generator` | [ソース](../../ruda-torch/python/ruda_torch/random.py#L69) |
| `ruda_torch.uniform_(tensor, from_=0.0, to=1.0, *, generator=None)` | [ソース](../../ruda-torch/python/ruda_torch/random.py#L78) |
| `ruda_torch.normal_(tensor, mean=0.0, std=1.0, *, generator=None)` | [ソース](../../ruda-torch/python/ruda_torch/random.py#L84) |
| `ruda_torch.bernoulli_(tensor, p=0.5, *, generator=None)` | [ソース](../../ruda-torch/python/ruda_torch/random.py#L90) |
| `ruda_torch.register_random_ops()` | [ソース](../../ruda-torch/python/ruda_torch/random.py#L111) |
| `ruda_torch.random.native_dropout(input, p, train=True)` | [ソース](../../ruda-torch/python/ruda_torch/random.py#L96) |

### `Generator`

| 公開名／呼び出し | ソース |
| --- | --- |
| `Generator(seed=None)` | [ソース](../../ruda-torch/python/ruda_torch/random.py#L17) |
| `Generator.manual_seed(seed)` | [ソース](../../ruda-torch/python/ruda_torch/random.py#L21) |
| `Generator.initial_seed()` | [ソース](../../ruda-torch/python/ruda_torch/random.py#L28) |
| `Generator.get_state()` | [ソース](../../ruda-torch/python/ruda_torch/random.py#L31) |
| `Generator.set_state(state)` | [ソース](../../ruda-torch/python/ruda_torch/random.py#L35) |
| `Generator.fill_(tensor, distribution, first, second)` | [ソース](../../ruda-torch/python/ruda_torch/random.py#L45) |

## `_streams`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.Stream` | [ソース](../../ruda-torch/python/ruda_torch/_streams.py#L10) |
| `ruda_torch.Event` | [ソース](../../ruda-torch/python/ruda_torch/_streams.py#L27) |
| `ruda_torch.stream(value)` | [ソース](../../ruda-torch/python/ruda_torch/_streams.py#L66) |
| `ruda_torch.current_stream(device=None)` | [ソース](../../ruda-torch/python/ruda_torch/_streams.py#L56) |
| `ruda_torch.default_stream(device=None)` | [ソース](../../ruda-torch/python/ruda_torch/_streams.py#L59) |
| `ruda_torch.record_stream(tensor, value)` | [ソース](../../ruda-torch/python/ruda_torch/_streams.py#L74) |

### `Stream`

| 公開名／呼び出し | ソース |
| --- | --- |
| `Stream(*, priority=0, _id=None)` | [ソース](../../ruda-torch/python/ruda_torch/_streams.py#L11) |
| `Stream.query()` | [ソース](../../ruda-torch/python/ruda_torch/_streams.py#L15) |
| `Stream.synchronize()` | [ソース](../../ruda-torch/python/ruda_torch/_streams.py#L17) |
| `Stream.wait_event(event)` | [ソース](../../ruda-torch/python/ruda_torch/_streams.py#L19) |
| `Stream.wait_stream(other)` | [ソース](../../ruda-torch/python/ruda_torch/_streams.py#L21) |
| `Stream.record_event(event=None)` | [ソース](../../ruda-torch/python/ruda_torch/_streams.py#L23) |

### `Event`

| 公開名／呼び出し | ソース |
| --- | --- |
| `Event(*, enable_timing=False)` | [ソース](../../ruda-torch/python/ruda_torch/_streams.py#L28) |
| `Event.record(value=None)` | [ソース](../../ruda-torch/python/ruda_torch/_streams.py#L30) |
| `Event.wait(value=None)` | [ソース](../../ruda-torch/python/ruda_torch/_streams.py#L34) |
| `Event.query()` | [ソース](../../ruda-torch/python/ruda_torch/_streams.py#L36) |
| `Event.synchronize()` | [ソース](../../ruda-torch/python/ruda_torch/_streams.py#L38) |
| `Event.elapsed_time(other)` | [ソース](../../ruda-torch/python/ruda_torch/_streams.py#L40) |
| `Event.close()` | [ソース](../../ruda-torch/python/ruda_torch/_streams.py#L45) |
| `Event.__del__()` | [ソース](../../ruda-torch/python/ruda_torch/_streams.py#L48) |

## `_paged`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.PagedAttentionPlan` | [ソース](../../ruda-torch/python/ruda_torch/_paged.py#L52) |

### `PagedAttentionPlan`

| 公開名／呼び出し | ソース |
| --- | --- |
| `PagedAttentionPlan(*, page_size, num_pages, block_tables, kv_lengths, sequence_ids, positions, splits=1, backward_strategy='atomic')` | [ソース](../../ruda-torch/python/ruda_torch/_paged.py#L53) |
| `PagedAttentionPlan.backward_strategy` | [ソース](../../ruda-torch/python/ruda_torch/_paged.py#L84) |
| `PagedAttentionPlan.splits` | [ソース](../../ruda-torch/python/ruda_torch/_paged.py#L88) |
| `PagedAttentionPlan.workspace_bytes(query_heads, value_dim)` | [ソース](../../ruda-torch/python/ruda_torch/_paged.py#L91) |
| `PagedAttentionPlan.attention(q, k, v, *, scale, causal=True)` | [ソース](../../ruda-torch/python/ruda_torch/_paged.py#L113) |
| `PagedAttentionPlan.mla(absorbed_query, position_query, latent_cache, position_cache, *, scale, causal=True)` | [ソース](../../ruda-torch/python/ruda_torch/_paged.py#L132) |

## `_graph`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.StaticGraph` | [ソース](../../ruda-torch/python/ruda_torch/_graph.py#L10) |
| `ruda_torch.GraphOp` | [ソース](../../ruda-torch/python/ruda_torch/_graph_spec.py#L17) |

### `StaticGraph`

| 公開名／呼び出し | ソース |
| --- | --- |
| `StaticGraph.from_model(model, **options)` | [ソース](../../ruda-torch/python/ruda_torch/_graph.py#L37) |
| `StaticGraph(inputs, nodes, *, outputs=None, infer_dependencies=False, track_completion=False, optimize=False, reuse_workspace=False, training=False)` | [ソース](../../ruda-torch/python/ruda_torch/_graph.py#L47) |
| `StaticGraph.replay()` | [ソース](../../ruda-torch/python/ruda_torch/_graph.py#L125) |
| `StaticGraph.run_eager()` | [ソース](../../ruda-torch/python/ruda_torch/_graph.py#L129) |
| `StaticGraph.synchronize()` | [ソース](../../ruda-torch/python/ruda_torch/_graph.py#L137) |
| `StaticGraph.query()` | [ソース](../../ruda-torch/python/ruda_torch/_graph.py#L138) |
| `StaticGraph.query_completion()` | [ソース](../../ruda-torch/python/ruda_torch/_graph.py#L139) |
| `StaticGraph.wait_completion()` | [ソース](../../ruda-torch/python/ruda_torch/_graph.py#L140) |
| `StaticGraph.info` | [ソース](../../ruda-torch/python/ruda_torch/_graph.py#L143) |
| `StaticGraph.close()` | [ソース](../../ruda-torch/python/ruda_torch/_graph.py#L158) |
| `StaticGraph.__enter__()` | [ソース](../../ruda-torch/python/ruda_torch/_graph.py#L164) |
| `StaticGraph.__exit__(*_)` | [ソース](../../ruda-torch/python/ruda_torch/_graph.py#L165) |

### `GraphOp`

フィールド: `kind: str`; `output: str`; `left: str`; <code>right: str &#124; None = None</code>; `scalar: float = 0.0`; <code>shape: tuple[int, ...] &#124; None = None</code>; <code>dtype: str &#124; None = None</code>.

| 公開名／呼び出し | ソース |
| --- | --- |
| `GraphOp.add(output, left, right, *, alpha=1.0)` | [ソース](../../ruda-torch/python/ruda_torch/_graph_spec.py#L34) |
| `GraphOp.mul(output, left, right)` | [ソース](../../ruda-torch/python/ruda_torch/_graph_spec.py#L36) |
| `GraphOp.silu(output, value)` | [ソース](../../ruda-torch/python/ruda_torch/_graph_spec.py#L38) |
| `GraphOp.silu_mul(output, gate, up)` | [ソース](../../ruda-torch/python/ruda_torch/_graph_spec.py#L40) |
| `GraphOp.copy(output, value)` | [ソース](../../ruda-torch/python/ruda_torch/_graph_spec.py#L42) |
| `GraphOp.rms_norm(output, value, weight=None, *, eps=1e-05)` | [ソース](../../ruda-torch/python/ruda_torch/_graph_spec.py#L44) |

## `training`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.LayerNorm` | [ソース](../../ruda-torch/python/ruda_torch/training.py#L141) |
| <code>ruda_torch.layer_norm(x: torch.Tensor, weight: torch.Tensor &#124; None=None, bias: torch.Tensor &#124; None=None, *, eps: float=1e-05)</code> | [ソース](../../ruda-torch/python/ruda_torch/training.py#L127) |
| `ruda_torch.RMSNorm` | [ソース](../../ruda-torch/python/ruda_torch/training.py#L82) |
| <code>ruda_torch.rms_norm(x: torch.Tensor, weight: torch.Tensor &#124; None=None, *, eps: float &#124; None=None)</code> | [ソース](../../ruda-torch/python/ruda_torch/training.py#L65) |
| `ruda_torch.silu_mul(gate: torch.Tensor, up: torch.Tensor)` | [ソース](../../ruda-torch/python/ruda_torch/training.py#L183) |
| `ruda_torch.AdamW` | [ソース](../../ruda-torch/python/ruda_torch/training.py#L207) |
| `ruda_torch.GradScaler` | [ソース](../../ruda-torch/python/ruda_torch/training.py#L521) |

### `LayerNorm`

| 公開名／呼び出し | ソース |
| --- | --- |
| `LayerNorm(width: int, *, eps=1e-05, elementwise_affine=True, bias=True, device=None, dtype=None)` | [ソース](../../ruda-torch/python/ruda_torch/training.py#L143) |
| `LayerNorm.forward(x)` | [ソース](../../ruda-torch/python/ruda_torch/training.py#L160) |

### `RMSNorm`

| 公開名／呼び出し | ソース |
| --- | --- |
| `RMSNorm(width: int, *, eps=1e-05, elementwise_affine=True, device=None, dtype=None)` | [ソース](../../ruda-torch/python/ruda_torch/training.py#L84) |
| `RMSNorm.forward(x)` | [ソース](../../ruda-torch/python/ruda_torch/training.py#L95) |

### `AdamW`

| 公開名／呼び出し | ソース |
| --- | --- |
| `AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, *, fused_step=False, max_grad_norm=None, hierarchical_stats=False)` | [ソース](../../ruda-torch/python/ruda_torch/training.py#L224) |
| `AdamW.step(closure=None, *, loss_scale=1.0, global_grad_norm=None)` | [ソース](../../ruda-torch/python/ruda_torch/training.py#L277) |
| `AdamW.state_dict()` | [ソース](../../ruda-torch/python/ruda_torch/training.py#L455) |
| `AdamW.load_state_dict(state_dict)` | [ソース](../../ruda-torch/python/ruda_torch/training.py#L466) |

### `GradScaler`

| 公開名／呼び出し | ソース |
| --- | --- |
| `GradScaler(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, min_scale=2.0 ** (-24), max_scale=2.0 ** 24)` | [ソース](../../ruda-torch/python/ruda_torch/training.py#L530) |
| `GradScaler.scale(loss: torch.Tensor)` | [ソース](../../ruda-torch/python/ruda_torch/training.py#L547) |
| `GradScaler.begin_backward()` | [ソース](../../ruda-torch/python/ruda_torch/training.py#L555) |
| `GradScaler.record_step(skipped)` | [ソース](../../ruda-torch/python/ruda_torch/training.py#L561) |
| `GradScaler.step(optimizer: AdamW, *args, **kwargs)` | [ソース](../../ruda-torch/python/ruda_torch/training.py#L568) |
| `GradScaler.update()` | [ソース](../../ruda-torch/python/ruda_torch/training.py#L594) |
| `GradScaler.get_scale()` | [ソース](../../ruda-torch/python/ruda_torch/training.py#L607) |
| `GradScaler.state_dict()` | [ソース](../../ruda-torch/python/ruda_torch/training.py#L610) |
| `GradScaler.load_state_dict(state)` | [ソース](../../ruda-torch/python/ruda_torch/training.py#L617) |

## `_router`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.selected_router_weights(logits: torch.Tensor, indices: torch.Tensor, *, scoring: str='softmax', renormalize: bool=False, scale: float=1.0) -> torch.Tensor` | [ソース](../../ruda-torch/python/ruda_torch/_router.py#L49) |

## `compiler`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.compile(model=None, *, capture='aot', native='auto', device_type='ruda', fullgraph=False, dynamic=None, min_native_ops=2, cache_size=4, decompositions=None)` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L365) |
| `ruda_torch.make_backend(*, native='auto', device_type='ruda', min_native_ops=2, cache_size=4, decompositions=None)` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L232) |
| `ruda_torch.CompiledModel` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L242) |
| `ruda_torch.CompiledFunction` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L316) |
| `ruda_torch.NativeCoverageError` | [ソース](../../ruda-torch/python/ruda_torch/_compile_native.py#L19) |
| `ruda_torch.GraphExecutionError` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L23) |
| `ruda_torch.compiler.CompileOptions` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L28) |
| `ruda_torch.compiler.RudaBackend` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L155) |

### `CompiledModel`

| 公開名／呼び出し | ソース |
| --- | --- |
| `CompiledModel(model, backend)` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L250) |
| `CompiledModel.original` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L264) |
| `CompiledModel.info` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L268) |
| `CompiledModel.forward(*args, **kwargs)` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L271) |
| `CompiledModel.state_dict(*args, **kwargs)` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L280) |
| `CompiledModel.load_state_dict(state_dict, strict=True, assign=False)` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L289) |
| `CompiledModel.named_parameters(prefix='', recurse=True, remove_duplicate=True)` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L292) |
| `CompiledModel.named_buffers(prefix='', recurse=True, remove_duplicate=True)` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L295) |
| `CompiledModel.__getattr__(name)` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L298) |
| `CompiledModel.close()` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L305) |
| `CompiledModel.__enter__()` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L308) |
| `CompiledModel.__exit__(*_)` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L312) |

### `CompiledFunction`

| 公開名／呼び出し | ソース |
| --- | --- |
| `CompiledFunction(function, backend)` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L318) |
| `CompiledFunction.__call__(*args, **kwargs)` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L324) |
| `CompiledFunction.info` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L333) |
| `CompiledFunction.close()` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L336) |
| `CompiledFunction.__enter__()` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L339) |
| `CompiledFunction.__exit__(*_)` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L343) |

### `NativeCoverageError`

### `GraphExecutionError`

### `CompileOptions`

フィールド: `capture: str = 'aot'`; `native: str = 'auto'`; `device_type: str = 'ruda'`; `fullgraph: bool = False`; <code>dynamic: bool &#124; None = None</code>; `min_native_ops: int = 2`; `cache_size: int = 4`.

| 公開名／呼び出し | ソース |
| --- | --- |
| `CompileOptions.__post_init__()` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L37) |

### `RudaBackend`

| 公開名／呼び出し | ソース |
| --- | --- |
| `RudaBackend(options=None, *, decompositions=None)` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L162) |
| `RudaBackend.info` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L172) |
| `RudaBackend.__call__(gm, example_inputs)` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L202) |
| `RudaBackend.close()` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L221) |
| `RudaBackend.__enter__()` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L224) |
| `RudaBackend.__exit__(*_)` | [ソース](../../ruda-torch/python/ruda_torch/compiler.py#L228) |

## `quantization`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.LearnedFakeQuantize` | [ソース](../../ruda-torch/python/ruda_torch/quantization.py#L84) |
| <code>ruda_torch.learned_fake_quantize(x: Tensor, scales: Tensor, *, bits: int=4, block_shape: Sequence[int] &#124; None=None, symmetric: bool=True) -&gt; Tensor</code> | [ソース](../../ruda-torch/python/ruda_torch/quantization.py#L38) |

### `LearnedFakeQuantize`

| 公開名／呼び出し | ソース |
| --- | --- |
| <code>LearnedFakeQuantize(initial_scales: Tensor, *, bits: int=4, block_shape: Sequence[int] &#124; None=None, symmetric: bool=True)</code> | [ソース](../../ruda-torch/python/ruda_torch/quantization.py#L90) |
| `LearnedFakeQuantize.forward(x: Tensor) -> Tensor` | [ソース](../../ruda-torch/python/ruda_torch/quantization.py#L104) |

## `mhc`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.MHC` | [ソース](../../ruda-torch/python/ruda_torch/mhc.py#L44) |
| `ruda_torch.MHCResidual` | [ソース](../../ruda-torch/python/ruda_torch/mhc.py#L129) |
| `ruda_torch.MHCSequential` | [ソース](../../ruda-torch/python/ruda_torch/mhc.py#L149) |
| `ruda_torch.MHCCoefficients` | [ソース](../../ruda-torch/python/ruda_torch/mhc.py#L19) |
| `ruda_torch.sinkhorn(logits: torch.Tensor, iterations: int=20) -> torch.Tensor` | [ソース](../../ruda-torch/python/ruda_torch/mhc.py#L25) |

### `MHC`

| 公開名／呼び出し | ソース |
| --- | --- |
| `MHC(width: int, streams: int=4, *, sinkhorn_iterations: int=20, eps: float=1e-06, gate_init: float=0.01, device=None, dtype=None)` | [ソース](../../ruda-torch/python/ruda_torch/mhc.py#L51) |
| `MHC.expand(x: torch.Tensor) -> torch.Tensor` | [ソース](../../ruda-torch/python/ruda_torch/mhc.py#L75) |
| `MHC.reduce(x: torch.Tensor) -> torch.Tensor` | [ソース](../../ruda-torch/python/ruda_torch/mhc.py#L80) |
| `MHC.coefficients(x: torch.Tensor) -> MHCCoefficients` | [ソース](../../ruda-torch/python/ruda_torch/mhc.py#L91) |
| `MHC.pre(x: torch.Tensor) -> tuple[torch.Tensor, MHCCoefficients]` | [ソース](../../ruda-torch/python/ruda_torch/mhc.py#L104) |
| `MHC.post(x: torch.Tensor, branch_output: torch.Tensor, coefficients: MHCCoefficients) -> torch.Tensor` | [ソース](../../ruda-torch/python/ruda_torch/mhc.py#L110) |
| `MHC.forward(x: torch.Tensor, branch: nn.Module, *args, **kwargs) -> torch.Tensor` | [ソース](../../ruda-torch/python/ruda_torch/mhc.py#L124) |

### `MHCResidual`

| 公開名／呼び出し | ソース |
| --- | --- |
| `MHCResidual(branch: nn.Module, width: int, streams: int=4, *, checkpoint_branch: bool=False, **kwargs)` | [ソース](../../ruda-torch/python/ruda_torch/mhc.py#L131) |
| `MHCResidual.forward(x: torch.Tensor, *args, **kwargs) -> torch.Tensor` | [ソース](../../ruda-torch/python/ruda_torch/mhc.py#L140) |

### `MHCSequential`

| 公開名／呼び出し | ソース |
| --- | --- |
| `MHCSequential(width: int, branches, streams: int=4, **kwargs)` | [ソース](../../ruda-torch/python/ruda_torch/mhc.py#L151) |
| `MHCSequential.forward(x: torch.Tensor) -> torch.Tensor` | [ソース](../../ruda-torch/python/ruda_torch/mhc.py#L159) |

### `MHCCoefficients`

フィールド: `pre: torch.Tensor`; `post: torch.Tensor`; `residual: torch.Tensor`.

## `sparse_attention`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.LightningIndexer` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L114) |
| `ruda_torch.DSAIndexer` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L114) |
| <code>ruda_torch.indexer_kl_loss(scores: torch.Tensor, teacher: torch.Tensor, valid: torch.Tensor &#124; None=None) -&gt; torch.Tensor</code> | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L35) |
| `ruda_torch.LearnedKVCompressor` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L294) |
| `ruda_torch.CompressedSparseAttention` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L627) |
| `ruda_torch.HeavilyCompressedAttention` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L633) |
| `ruda_torch.CSA` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L627) |
| `ruda_torch.HCA` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L633) |
| `ruda_torch.AttentionOutput` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L24) |
| `ruda_torch.IndexerOutput` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L29) |
| `ruda_torch.CompressedAttentionCache` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L397) |
| `ruda_torch.CompressionState` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L283) |
| `ruda_torch.RotaryEmbedding` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L69) |

### `LightningIndexer`

| 公開名／呼び出し | ソース |
| --- | --- |
| <code>LightningIndexer(width: int, num_heads: int=4, head_dim: int=16, *, query_dim: int &#124; None=None, topk: int=32, query_chunk_size: int=32, key_chunk_size: int=128, external_keys: bool=False, detach_inputs: bool=True, rope_dim: int=0, eps: float=1e-06, device=None, dtype=None)</code> | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L124) |
| <code>LightningIndexer.project_keys(x: torch.Tensor, pos: torch.Tensor &#124; None=None) -&gt; torch.Tensor</code> | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L149) |
| `LightningIndexer.scores(x, keys, *, query_latent=None, query_positions=None)` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L185) |
| `LightningIndexer.selected_scores(x, keys, indices, *, query_latent=None, query_positions=None)` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L191) |
| `LightningIndexer.select(x, keys, *, allowed=None, query_latent=None, query_positions=None, key_valid=None, query_valid=None, key_end_positions=None)` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L198) |
| `LightningIndexer.forward(x, key_states=None, *, prepared_keys=None, allowed=None, query_latent=None, query_positions=None, causal=True, key_end_positions=None, key_valid=None, query_valid=None)` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L248) |
| `LightningIndexer.distillation_loss(x, prepared_keys, teacher, *, allowed=None, indices=None, query_latent=None, query_positions=None)` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L269) |

### `DSAIndexer`

| 公開名／呼び出し | ソース |
| --- | --- |
| <code>DSAIndexer(width: int, num_heads: int=4, head_dim: int=16, *, query_dim: int &#124; None=None, topk: int=32, query_chunk_size: int=32, key_chunk_size: int=128, external_keys: bool=False, detach_inputs: bool=True, rope_dim: int=0, eps: float=1e-06, device=None, dtype=None)</code> | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L124) |
| <code>DSAIndexer.project_keys(x: torch.Tensor, pos: torch.Tensor &#124; None=None) -&gt; torch.Tensor</code> | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L149) |
| `DSAIndexer.scores(x, keys, *, query_latent=None, query_positions=None)` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L185) |
| `DSAIndexer.selected_scores(x, keys, indices, *, query_latent=None, query_positions=None)` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L191) |
| `DSAIndexer.select(x, keys, *, allowed=None, query_latent=None, query_positions=None, key_valid=None, query_valid=None, key_end_positions=None)` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L198) |
| `DSAIndexer.forward(x, key_states=None, *, prepared_keys=None, allowed=None, query_latent=None, query_positions=None, causal=True, key_end_positions=None, key_valid=None, query_valid=None)` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L248) |
| `DSAIndexer.distillation_loss(x, prepared_keys, teacher, *, allowed=None, indices=None, query_latent=None, query_positions=None)` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L269) |

### `LearnedKVCompressor`

| 公開名／呼び出し | ソース |
| --- | --- |
| `LearnedKVCompressor(width: int, head_dim: int, ratio: int, *, overlap=False, eps=1e-06, device=None, dtype=None)` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L301) |
| `LearnedKVCompressor.forward(x, valid_mask=None)` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L351) |
| <code>LearnedKVCompressor.append(x, valid_mask=None, state: CompressionState &#124; None=None)</code> | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L359) |

### `CompressedSparseAttention`

| 公開名／呼び出し | ソース |
| --- | --- |
| `CompressedSparseAttention(width: int, num_heads: int, **kwargs)` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L629) |
| `CompressedSparseAttention.forward(x, *, valid_mask=None, return_aux=False, indexer_warmup=False)` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L560) |
| <code>CompressedSparseAttention.forward_cached(x, cache: CompressedAttentionCache &#124; None=None, *, valid_mask=None)</code> | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L583) |

### `HeavilyCompressedAttention`

| 公開名／呼び出し | ソース |
| --- | --- |
| `HeavilyCompressedAttention(width: int, num_heads: int, *, compress_ratio=128, **kwargs)` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L635) |
| `HeavilyCompressedAttention.forward(x, *, valid_mask=None, return_aux=False, indexer_warmup=False)` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L560) |
| <code>HeavilyCompressedAttention.forward_cached(x, cache: CompressedAttentionCache &#124; None=None, *, valid_mask=None)</code> | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L583) |

### `CSA`

| 公開名／呼び出し | ソース |
| --- | --- |
| `CSA(width: int, num_heads: int, **kwargs)` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L629) |
| `CSA.forward(x, *, valid_mask=None, return_aux=False, indexer_warmup=False)` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L560) |
| <code>CSA.forward_cached(x, cache: CompressedAttentionCache &#124; None=None, *, valid_mask=None)</code> | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L583) |

### `HCA`

| 公開名／呼び出し | ソース |
| --- | --- |
| `HCA(width: int, num_heads: int, *, compress_ratio=128, **kwargs)` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L635) |
| `HCA.forward(x, *, valid_mask=None, return_aux=False, indexer_warmup=False)` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L560) |
| <code>HCA.forward_cached(x, cache: CompressedAttentionCache &#124; None=None, *, valid_mask=None)</code> | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L583) |

### `AttentionOutput`

フィールド: `output: torch.Tensor`; `indexer_loss: torch.Tensor`.

### `IndexerOutput`

フィールド: `indices: torch.Tensor`; `scores: torch.Tensor`; `valid: torch.Tensor`.

### `CompressedAttentionCache`

フィールド: `owner: int`; `parameter_versions: tuple`; `seen: int`; `compressed: torch.Tensor`; `compressed_valid: torch.Tensor`; <code>index_keys: torch.Tensor &#124; None</code>; `local: torch.Tensor`; `local_valid: torch.Tensor`; `compression: CompressionState`; <code>index_compression: CompressionState &#124; None</code>.

| 公開名／呼び出し | ソース |
| --- | --- |
| `CompressedAttentionCache.reorder(batch_indices: torch.Tensor)` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L409) |
| `CompressedAttentionCache.tensor_bytes` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L420) |

### `CompressionState`

フィールド: `tail: torch.Tensor`; `tail_valid: torch.Tensor`; `previous: torch.Tensor`; `previous_valid: torch.Tensor`.

| 公開名／呼び出し | ソース |
| --- | --- |
| `CompressionState.reorder(batch_indices: torch.Tensor)` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L289) |

### `RotaryEmbedding`

| 公開名／呼び出し | ソース |
| --- | --- |
| `RotaryEmbedding(rope_dim: int, base: float=10000.0, *, device=None)` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L75) |
| `RotaryEmbedding.forward(x: torch.Tensor, pos: torch.Tensor, *, inverse: bool=False) -> torch.Tensor` | [ソース](../../ruda-torch/python/ruda_torch/sparse_attention.py#L93) |

## `optim`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.Muon` | [ソース](../../ruda-torch/python/ruda_torch/optim.py#L144) |
| `ruda_torch.MuonAdamW` | [ソース](../../ruda-torch/python/ruda_torch/optim.py#L340) |
| `ruda_torch.muon_orthogonalize(gradient: torch.Tensor, *, ns_steps=5, coefficients=(3.4445, -4.775, 2.0315), eps=1e-07, stable_normalization=True) -> torch.Tensor` | [ソース](../../ruda-torch/python/ruda_torch/optim.py#L37) |

### `Muon`

| 公開名／呼び出し | ソース |
| --- | --- |
| `Muon(params, lr=0.02, *, momentum=0.95, weight_decay=0.0, nesterov=True, momentum_mode='sgd', dampening=0.0, ns_steps=5, ns_coefficients=(3.4445, -4.775, 2.0315), eps=1e-07, adjust_lr='original', matrix_layout='as_stored', stable_normalization=True, flatten=False, max_grad_norm=None)` | [ソース](../../ruda-torch/python/ruda_torch/optim.py#L155) |
| `Muon.add_param_group(param_group)` | [ソース](../../ruda-torch/python/ruda_torch/optim.py#L172) |
| `Muon.step(closure=None, *, loss_scale=1.0)` | [ソース](../../ruda-torch/python/ruda_torch/optim.py#L212) |
| `Muon.state_dict()` | [ソース](../../ruda-torch/python/ruda_torch/optim.py#L302) |
| `Muon.load_state_dict(state_dict)` | [ソース](../../ruda-torch/python/ruda_torch/optim.py#L307) |

### `MuonAdamW`

| 公開名／呼び出し | ソース |
| --- | --- |
| `MuonAdamW(param_groups, **kwargs)` | [ソース](../../ruda-torch/python/ruda_torch/optim.py#L342) |
| `MuonAdamW.add_param_group(param_group)` | [ソース](../../ruda-torch/python/ruda_torch/optim.py#L172) |
| `MuonAdamW.step(closure=None, *, loss_scale=1.0)` | [ソース](../../ruda-torch/python/ruda_torch/optim.py#L212) |
| `MuonAdamW.state_dict()` | [ソース](../../ruda-torch/python/ruda_torch/optim.py#L302) |
| `MuonAdamW.load_state_dict(state_dict)` | [ソース](../../ruda-torch/python/ruda_torch/optim.py#L307) |
| `MuonAdamW.from_model(model: nn.Module, *, muon_modules, adamw_modules=(), lr=0.02, adamw_lr=0.001, weight_decay=0.0, betas=(0.9, 0.999), **kwargs)` | [ソース](../../ruda-torch/python/ruda_torch/optim.py#L349) |

## `hybrid_model`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.MHCTransformerBlock` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_model.py#L66) |
| `ruda_torch.HybridAttentionLanguageModel` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_model.py#L121) |
| `ruda_torch.HybridAttentionBackbone` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_model.py#L204) |
| `ruda_torch.HybridAttentionCache` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_model.py#L28) |
| `ruda_torch.next_token_loss(logits: torch.Tensor, tokens: torch.Tensor, *, valid_mask=None)` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_model.py#L251) |

### `MHCTransformerBlock`

| 公開名／呼び出し | ソース |
| --- | --- |
| `MHCTransformerBlock(width: int, num_heads: int, *, streams=4, attention_kind='csa', feedforward_width=None, sinkhorn_iterations=20, eps=1e-06, **attention_kwargs)` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_model.py#L72) |
| `MHCTransformerBlock.forward(state, *, valid_mask=None, return_aux=False, indexer_warmup=False)` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_model.py#L91) |
| `MHCTransformerBlock.forward_cached(state, cache=None, *, valid_mask=None)` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_model.py#L105) |

### `HybridAttentionLanguageModel`

| 公開名／呼び出し | ソース |
| --- | --- |
| `HybridAttentionLanguageModel(vocab_size: int, width: int, num_heads: int, num_layers: int, *, streams=4, csa_ratio=4, hca_ratio=128, tie_embeddings=False, sinkhorn_iterations=20, eps=1e-06, **attention_kwargs)` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_model.py#L129) |
| `HybridAttentionLanguageModel.forward(tokens, *, valid_mask=None, return_aux=False, indexer_warmup=False)` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_model.py#L151) |
| `HybridAttentionLanguageModel.forward_hidden(tokens, *, valid_mask=None, return_aux=False, indexer_warmup=False)` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_model.py#L155) |
| `HybridAttentionLanguageModel.hidden_backbone()` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_model.py#L159) |
| `HybridAttentionLanguageModel.forward_hidden_cached(tokens, cache=None, *, valid_mask=None)` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_model.py#L163) |
| `HybridAttentionLanguageModel.forward_cached(tokens, cache=None, *, valid_mask=None)` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_model.py#L194) |
| `HybridAttentionLanguageModel.forward_cached_last(tokens, cache=None, *, valid_mask=None)` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_model.py#L198) |

### `HybridAttentionBackbone`

| 公開名／呼び出し | ソース |
| --- | --- |
| `HybridAttentionBackbone(model)` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_model.py#L212) |
| `HybridAttentionBackbone.forward(input_ids, attention_mask=None, *, position_ids=None, cu_seqlens=None, max_seqlen=None)` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_model.py#L220) |

### `HybridAttentionCache`

フィールド: `owner: int`; `signature: tuple`; `seen: int`; `layers: tuple[CompressedAttentionCache, ...]`.

| 公開名／呼び出し | ソース |
| --- | --- |
| `HybridAttentionCache.reorder(batch_indices)` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_model.py#L35) |
| `HybridAttentionCache.tensor_bytes` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_model.py#L43) |

## `sequence_training`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.gated_delta_rule(query, key, value, beta, log_decay, *, initial_state=None, query_scale=None, normalize_qk=False, norm_eps=1e-06, chunk_size=64, checkpoint_chunks=True, native_forward=True)` | [ソース](../../ruda-torch/python/ruda_torch/sequence_training.py#L137) |
| `ruda_torch.solve_triangular(a, b, *, upper=False, left=True, unitriangular=False)` | [ソース](../../ruda-torch/python/ruda_torch/sequence_training.py#L52) |
| `ruda_torch.sequence_training.triangular_mask(x, diagonal=0, *, upper=False)` | [ソース](../../ruda-torch/python/ruda_torch/sequence_training.py#L11) |

## `finetuning`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.LoRALinear` | [ソース](../../ruda-torch/python/ruda_torch/finetuning.py#L210) |
| `ruda_torch.NF4Linear` | [ソース](../../ruda-torch/python/ruda_torch/finetuning.py#L138) |
| `ruda_torch.inject_lora(model, *, target_modules, rank=16, alpha=16.0, adapter_dtype=torch.float32, dropout=0.0, use_rslora=False)` | [ソース](../../ruda-torch/python/ruda_torch/finetuning.py#L289) |
| `ruda_torch.quantize_nf4(model, *, target_modules, block_size=64, tile_rows=128)` | [ソース](../../ruda-torch/python/ruda_torch/finetuning.py#L322) |
| `ruda_torch.adapter_state_dict(model)` | [ソース](../../ruda-torch/python/ruda_torch/finetuning.py#L358) |
| `ruda_torch.load_adapter_state_dict(model, state)` | [ソース](../../ruda-torch/python/ruda_torch/finetuning.py#L388) |
| `ruda_torch.merge_lora(model)` | [ソース](../../ruda-torch/python/ruda_torch/finetuning.py#L429) |
| `ruda_torch.load_nf4_safetensors(model, directory, *, target_modules, device, dtype=torch.bfloat16, block_size=64, tile_rows=128, parameter_dtypes=None, buffer_dtypes=None)` | [ソース](../../ruda-torch/python/ruda_torch/finetuning.py#L471) |
| `ruda_torch.finetune_state_dict(model, optimizer, *, base_id, step, data_state, scaler=None)` | [ソース](../../ruda-torch/python/ruda_torch/finetuning.py#L620) |
| `ruda_torch.load_finetune_state_dict(model, optimizer, state, *, base_id, scaler=None)` | [ソース](../../ruda-torch/python/ruda_torch/finetuning.py#L646) |
| `ruda_torch.finetuning.pack_nf4(weight, *, block_size=64, chunk_blocks=1024)` | [ソース](../../ruda-torch/python/ruda_torch/finetuning.py#L40) |

### `LoRALinear`

| 公開名／呼び出し | ソース |
| --- | --- |
| `LoRALinear(base, *, rank=16, alpha=16.0, adapter_dtype=torch.float32, dropout=0.0, use_rslora=False)` | [ソース](../../ruda-torch/python/ruda_torch/finetuning.py#L212) |
| `LoRALinear.get_extra_state()` | [ソース](../../ruda-torch/python/ruda_torch/finetuning.py#L238) |
| `LoRALinear.scaling` | [ソース](../../ruda-torch/python/ruda_torch/finetuning.py#L245) |
| `LoRALinear.set_extra_state(state)` | [ソース](../../ruda-torch/python/ruda_torch/finetuning.py#L248) |
| `LoRALinear.forward(x)` | [ソース](../../ruda-torch/python/ruda_torch/finetuning.py#L252) |

### `NF4Linear`

| 公開名／呼び出し | ソース |
| --- | --- |
| `NF4Linear(in_features, out_features, packed, scales, *, block_size=64, tile_rows=128, bias=None)` | [ソース](../../ruda-torch/python/ruda_torch/finetuning.py#L146) |
| `NF4Linear.from_linear(linear, *, block_size=64, tile_rows=128)` | [ソース](../../ruda-torch/python/ruda_torch/finetuning.py#L172) |
| `NF4Linear.get_extra_state()` | [ソース](../../ruda-torch/python/ruda_torch/finetuning.py#L189) |
| `NF4Linear.set_extra_state(state)` | [ソース](../../ruda-torch/python/ruda_torch/finetuning.py#L193) |
| `NF4Linear.forward(x)` | [ソース](../../ruda-torch/python/ruda_torch/finetuning.py#L197) |

## `causal_finetuning`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.chunked_lm_cross_entropy(hidden, head, labels, *, token_chunk_size=32, ignore_index=-100, shift=True, reduction='mean', recompute=True, label_smoothing=0.0)` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L25) |
| `ruda_torch.SFTCollator` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L85) |
| `ruda_torch.CausalLMFinetuner` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L181) |
| `ruda_torch.SFTTrainer` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L303) |
| `ruda_torch.load_hf_nf4_model(directory, *, target_modules, device, dtype=torch.bfloat16, auto_class=None, rank=16, alpha=16.0, block_size=64, tile_rows=128, adapter_dtype=torch.float32, parameter_dtypes=None, config_kwargs=None, model_kwargs=None)` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L261) |
| `ruda_torch.activation_checkpoint_modules(model, target_modules, *, preserve_rng_state=True)` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L150) |
| `ruda_torch.PackedCausalLMFinetuner` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L232) |
| `ruda_torch.CausalLMTrainer` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L669) |

### `SFTCollator`

| 公開名／呼び出し | ソース |
| --- | --- |
| `SFTCollator(tokenizer=None, *, max_length, pad_token_id=None, train_on_prompt=False, truncate=False, template_kwargs=None)` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L92) |
| `SFTCollator.encode(sample)` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L105) |
| `SFTCollator.__call__(samples)` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L135) |

### `CausalLMFinetuner`

| 公開名／呼び出し | ソース |
| --- | --- |
| `CausalLMFinetuner(backbone, head, *, token_chunk_size=32, activation_checkpointing=True, checkpoint_modules=None, preserve_rng_state=True, label_smoothing=0.0)` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L183) |
| `CausalLMFinetuner.hidden(input_ids, attention_mask)` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L208) |
| `CausalLMFinetuner.forward(input_ids, attention_mask, labels, *, reduction='mean', label_smoothing=None)` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L225) |

### `SFTTrainer`

| 公開名／呼び出し | ソース |
| --- | --- |
| `SFTTrainer(model, optimizer, *, base_id, run_config, scaler=None, scheduler=None, replica_group=None, gradient_overlap=False, bucket_bytes=25 * 1024 * 1024, sampler=None)` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L310) |
| `SFTTrainer.train_step(microbatches)` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L374) |
| `SFTTrainer.train_step_autotuned(microbatches, *, autotuner=None)` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L464) |
| `SFTTrainer.evaluate(microbatches)` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L475) |
| `SFTTrainer.save(directory)` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L562) |
| `SFTTrainer.resume(path)` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L590) |
| `SFTTrainer.save_distributed(directory)` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L617) |
| `SFTTrainer.resume_distributed(directory, *, generation=None)` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L631) |
| `SFTTrainer.write_progress(directory, metrics, *, total_steps=None)` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L651) |

### `PackedCausalLMFinetuner`

| 公開名／呼び出し | ソース |
| --- | --- |
| `PackedCausalLMFinetuner(backbone, head, *, token_chunk_size=32, activation_checkpointing=True, checkpoint_modules=None, preserve_rng_state=True, label_smoothing=0.0)` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L183) |
| `PackedCausalLMFinetuner.hidden(input_ids, attention_mask)` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L208) |
| `PackedCausalLMFinetuner.forward(input_ids, attention_mask, labels, position_ids, cu_seqlens, max_seqlen, *, reduction='mean', label_smoothing=None)` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L242) |

### `CausalLMTrainer`

| 公開名／呼び出し | ソース |
| --- | --- |
| `CausalLMTrainer(model, optimizer, *, base_id, run_config, scaler=None, scheduler=None, replica_group=None, gradient_overlap=False, bucket_bytes=25 * 1024 * 1024, sampler=None)` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L310) |
| `CausalLMTrainer.train_step(microbatches)` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L374) |
| `CausalLMTrainer.train_step_autotuned(microbatches, *, autotuner=None)` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L464) |
| `CausalLMTrainer.evaluate(microbatches)` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L475) |
| `CausalLMTrainer.save(directory)` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L562) |
| `CausalLMTrainer.resume(path)` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L590) |
| `CausalLMTrainer.save_distributed(directory)` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L617) |
| `CausalLMTrainer.resume_distributed(directory, *, generation=None)` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L631) |
| `CausalLMTrainer.write_progress(directory, metrics, *, total_steps=None)` | [ソース](../../ruda-torch/python/ruda_torch/causal_finetuning.py#L651) |

## `distributed_training`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.ReplicaGroup` | [ソース](../../ruda-torch/python/ruda_torch/distributed_training.py#L13) |

### `ReplicaGroup`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ReplicaGroup(process_group=None, *, device_type='ruda')` | [ソース](../../ruda-torch/python/ruda_torch/distributed_training.py#L20) |
| `ReplicaGroup.gather_metadata(value)` | [ソース](../../ruda-torch/python/ruda_torch/distributed_training.py#L52) |
| `ReplicaGroup.sum_(tensor)` | [ソース](../../ruda-torch/python/ruda_torch/distributed_training.py#L74) |
| `ReplicaGroup.max_(tensor)` | [ソース](../../ruda-torch/python/ruda_torch/distributed_training.py#L81) |
| `ReplicaGroup.min_(tensor)` | [ソース](../../ruda-torch/python/ruda_torch/distributed_training.py#L87) |
| `ReplicaGroup.broadcast_(tensor, root=0)` | [ソース](../../ruda-torch/python/ruda_torch/distributed_training.py#L93) |
| `ReplicaGroup.all_gather(tensor, *, axis=0)` | [ソース](../../ruda-torch/python/ruda_torch/distributed_training.py#L103) |
| `ReplicaGroup.reduce_scatter(tensor, *, axis=0)` | [ソース](../../ruda-torch/python/ruda_torch/distributed_training.py#L116) |
| `ReplicaGroup.begin_gradient_overlap(*, global_weight, normalized=False, bucket_bytes=25 * 1024 * 1024)` | [ソース](../../ruda-torch/python/ruda_torch/distributed_training.py#L131) |
| `ReplicaGroup.initialize(model, *, root=0, broadcast_buffers=True)` | [ソース](../../ruda-torch/python/ruda_torch/distributed_training.py#L141) |
| `ReplicaGroup.validate_model(model)` | [ソース](../../ruda-torch/python/ruda_torch/distributed_training.py#L188) |
| `ReplicaGroup.total_weight(local_weight)` | [ソース](../../ruda-torch/python/ruda_torch/distributed_training.py#L199) |
| `ReplicaGroup.synchronize_gradients(*, local_weight, normalized=False, missing='error')` | [ソース](../../ruda-torch/python/ruda_torch/distributed_training.py#L209) |
| `ReplicaGroup.validate_training_options(value)` | [ソース](../../ruda-torch/python/ruda_torch/distributed_training.py#L265) |

## `parallel_training`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.ColumnParallelLinear` | [ソース](../../ruda-torch/python/ruda_torch/parallel_training.py#L57) |
| `ruda_torch.RowParallelLinear` | [ソース](../../ruda-torch/python/ruda_torch/parallel_training.py#L81) |
| `ruda_torch.VocabParallelEmbedding` | [ソース](../../ruda-torch/python/ruda_torch/parallel_training.py#L104) |
| `ruda_torch.TensorParallelMLP` | [ソース](../../ruda-torch/python/ruda_torch/parallel_training.py#L131) |
| `ruda_torch.TensorParallelAttention` | [ソース](../../ruda-torch/python/ruda_torch/parallel_training.py#L147) |
| `ruda_torch.tensor_parallelize(model, group, plan)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_training.py#L205) |
| `ruda_torch.tensor_parallel_state_dict(model, group)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_training.py#L273) |
| `ruda_torch.load_tensor_parallel_state_dict(model, state, group, *, strict=True)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_training.py#L296) |
| `ruda_torch.copy_to_tensor_parallel(value, group)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_training.py#L41) |
| `ruda_torch.reduce_from_tensor_parallel(value, group)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_training.py#L45) |
| `ruda_torch.gather_from_tensor_parallel(value, group, *, axis=-1)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_training.py#L49) |
| `ruda_torch.scatter_to_tensor_parallel(value, group, *, axis=-1)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_training.py#L53) |
| `ruda_torch.parallel_training.GradientOverlap` | [ソース](../../ruda-torch/python/ruda_torch/parallel_training.py#L324) |

### `ColumnParallelLinear`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ColumnParallelLinear(linear, group, *, gather_output=True)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_training.py#L59) |
| `ColumnParallelLinear.forward(value)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_training.py#L75) |

### `RowParallelLinear`

| 公開名／呼び出し | ソース |
| --- | --- |
| `RowParallelLinear(linear, group, *, input_is_parallel=False)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_training.py#L83) |
| `RowParallelLinear.forward(value)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_training.py#L97) |

### `VocabParallelEmbedding`

| 公開名／呼び出し | ソース |
| --- | --- |
| `VocabParallelEmbedding(embedding, group)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_training.py#L106) |
| `VocabParallelEmbedding.forward(tokens)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_training.py#L121) |

### `TensorParallelMLP`

| 公開名／呼び出し | ソース |
| --- | --- |
| `TensorParallelMLP(gate, up, down, group, *, activation=None)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_training.py#L133) |
| `TensorParallelMLP.forward(value)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_training.py#L143) |

### `TensorParallelAttention`

| 公開名／呼び出し | ソース |
| --- | --- |
| `TensorParallelAttention(query, key, value, output, group, *, num_heads, num_kv_heads=None, head_dim=None, scale=None, dropout_p=0.0, attention_impl=None)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_training.py#L153) |
| `TensorParallelAttention.forward(hidden, *, attention_mask=None, is_causal=False, position_transform=None)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_training.py#L179) |

### `GradientOverlap`

| 公開名／呼び出し | ソース |
| --- | --- |
| `GradientOverlap(group, *, global_weight, normalized, bucket_bytes)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_training.py#L326) |
| `GradientOverlap.finish(*, local_weight, missing='zero')` | [ソース](../../ruda-torch/python/ruda_torch/parallel_training.py#L370) |

## `sharded_training`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.FullyShardedModule` | [ソース](../../ruda-torch/python/ruda_torch/sharded_training.py#L91) |
| `ruda_torch.fully_shard(model, group, *, unit_paths=None, device=None, root=0, recompute=True, shard_buffers=None, ensure_backward=False)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_training.py#L305) |
| `ruda_torch.Zero2Optimizer` | [ソース](../../ruda-torch/python/ruda_torch/sharded_training.py#L346) |
| `ruda_torch.ShardedReplicaGroup` | [ソース](../../ruda-torch/python/ruda_torch/sharded_training.py#L474) |

### `FullyShardedModule`

| 公開名／呼び出し | ソース |
| --- | --- |
| `FullyShardedModule(module, group, *, device=None, root=0, recompute=True, shard_buffers=None, ensure_backward=False)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_training.py#L101) |
| `FullyShardedModule.forward(*args, **kwargs)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_training.py#L218) |
| `FullyShardedModule.train(mode=True)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_training.py#L226) |
| `FullyShardedModule.extra_repr()` | [ソース](../../ruda-torch/python/ruda_torch/sharded_training.py#L233) |
| `FullyShardedModule.get_extra_state()` | [ソース](../../ruda-torch/python/ruda_torch/sharded_training.py#L236) |
| `FullyShardedModule.set_extra_state(state)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_training.py#L243) |
| `FullyShardedModule.full_state_dict()` | [ソース](../../ruda-torch/python/ruda_torch/sharded_training.py#L249) |
| `FullyShardedModule.load_full_state_dict(state, *, strict=True)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_training.py#L266) |

### `Zero2Optimizer`

| 公開名／呼び出し | ソース |
| --- | --- |
| `Zero2Optimizer(parameter_groups, group, optimizer_factory)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_training.py#L353) |
| `Zero2Optimizer.synchronize_gradients(*, local_weight, normalized=False, missing='zero')` | [ソース](../../ruda-torch/python/ruda_torch/sharded_training.py#L387) |
| `Zero2Optimizer.step(closure=None, **kwargs)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_training.py#L419) |
| `Zero2Optimizer.zero_grad(set_to_none=True)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_training.py#L447) |
| `Zero2Optimizer.state_dict()` | [ソース](../../ruda-torch/python/ruda_torch/sharded_training.py#L455) |
| `Zero2Optimizer.load_state_dict(state)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_training.py#L459) |

### `ShardedReplicaGroup`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ShardedReplicaGroup(model, group)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_training.py#L481) |
| `ShardedReplicaGroup.gather_metadata(value)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_training.py#L494) |
| `ShardedReplicaGroup.validate_training_options(value)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_training.py#L497) |
| `ShardedReplicaGroup.total_weight(value)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_training.py#L500) |
| `ShardedReplicaGroup.sum_(tensor)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_training.py#L503) |
| `ShardedReplicaGroup.validate_model(model)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_training.py#L506) |
| `ShardedReplicaGroup.validate_microbatches(batches)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_training.py#L511) |
| `ShardedReplicaGroup.synchronize_gradients(*, local_weight, normalized=False, missing='zero')` | [ソース](../../ruda-torch/python/ruda_torch/sharded_training.py#L518) |
| `ShardedReplicaGroup.prepare_optimizer_step(optimizer, *, loss_scale=1.0)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_training.py#L535) |

## `pipeline_training`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.PipelineStage` | [ソース](../../ruda-torch/python/ruda_torch/pipeline_training.py#L31) |
| `ruda_torch.PipelineTensorSpec` | [ソース](../../ruda-torch/python/ruda_torch/pipeline_training.py#L12) |

### `PipelineStage`

| 公開名／呼び出し | ソース |
| --- | --- |
| `PipelineStage(module, group, *, input_spec, output_spec)` | [ソース](../../ruda-torch/python/ruda_torch/pipeline_training.py#L41) |
| `PipelineStage.invocation_contract` | [ソース](../../ruda-torch/python/ruda_torch/pipeline_training.py#L92) |
| `PipelineStage.validate_interfaces(count, microbatch_specs=None)` | [ソース](../../ruda-torch/python/ruda_torch/pipeline_training.py#L94) |
| `PipelineStage.run(inputs=None, targets=None, *, loss_sum, global_weight, loss_scale=1.0, microbatch_specs=None, schedule='1f1b')` | [ソース](../../ruda-torch/python/ruda_torch/pipeline_training.py#L149) |

### `PipelineTensorSpec`

フィールド: `shape: tuple[int, ...]`; `dtype: torch.dtype = torch.float32`.

| 公開名／呼び出し | ソース |
| --- | --- |
| `PipelineTensorSpec.validate()` | [ソース](../../ruda-torch/python/ruda_torch/pipeline_training.py#L16) |
| `PipelineTensorSpec.allocate(device)` | [ソース](../../ruda-torch/python/ruda_torch/pipeline_training.py#L22) |
| `PipelineTensorSpec.has_floating` | [ソース](../../ruda-torch/python/ruda_torch/pipeline_training.py#L27) |

## `structured_pipeline`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.StructuredPipelineStage` | [ソース](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L149) |
| `ruda_torch.PipelineTreeSpec` | [ソース](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L58) |
| `ruda_torch.PipelineLeafSpec` | [ソース](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L33) |
| `ruda_torch.PipelineConstant` | [ソース](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L21) |

### `StructuredPipelineStage`

| 公開名／呼び出し | ソース |
| --- | --- |
| `StructuredPipelineStage(module, group, *, input_spec, output_spec, input_mode='single')` | [ソース](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L165) |
| `StructuredPipelineStage.invocation_contract` | [ソース](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L227) |
| `StructuredPipelineStage.validate_interfaces(count, microbatch_specs=None)` | [ソース](../../ruda-torch/python/ruda_torch/pipeline_training.py#L94) |
| `StructuredPipelineStage.run(inputs=None, targets=None, *, loss_sum, global_weight, loss_scale=1.0, microbatch_specs=None, schedule='1f1b')` | [ソース](../../ruda-torch/python/ruda_torch/pipeline_training.py#L149) |

### `PipelineTreeSpec`

フィールド: `leaves: tuple`; `tree: object`.

| 公開名／呼び出し | ソース |
| --- | --- |
| `PipelineTreeSpec(structure)` | [ソース](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L71) |
| `PipelineTreeSpec.from_flat(tree, leaves)` | [ソース](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L79) |
| `PipelineTreeSpec.validate()` | [ソース](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L94) |
| `PipelineTreeSpec.has_floating` | [ソース](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L101) |
| `PipelineTreeSpec.allocate(device)` | [ソース](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L104) |
| `PipelineTreeSpec.matches(value)` | [ソース](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L109) |

### `PipelineLeafSpec`

| 公開名／呼び出し | ソース |
| --- | --- |
| `PipelineLeafSpec.validate()` | [ソース](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L45) |
| `PipelineLeafSpec.allocate(device)` | [ソース](../../ruda-torch/python/ruda_torch/pipeline_training.py#L22) |
| `PipelineLeafSpec.has_floating` | [ソース](../../ruda-torch/python/ruda_torch/pipeline_training.py#L27) |
| `PipelineLeafSpec(shape, dtype=torch.float32)` | [ソース](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L40) |
| `PipelineLeafSpec.__eq__(other)` | [ソース](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L51) |
| `PipelineLeafSpec.__hash__()` | [ソース](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L54) |

### `PipelineConstant`

フィールド: `value: object`.

| 公開名／呼び出し | ソース |
| --- | --- |
| `PipelineConstant.__post_init__()` | [ソース](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L25) |
| `PipelineConstant.__eq__(other)` | [ソース](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L27) |
| `PipelineConstant.__hash__()` | [ソース](../../ruda-torch/python/ruda_torch/structured_pipeline.py#L30) |

## `distributed_checkpoint`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.DistributedCheckpoint` | [ソース](../../ruda-torch/python/ruda_torch/distributed_checkpoint.py#L129) |
| `ruda_torch.model_shard_layout(model)` | [ソース](../../ruda-torch/python/ruda_torch/distributed_checkpoint.py#L47) |

### `DistributedCheckpoint`

| 公開名／呼び出し | ソース |
| --- | --- |
| `DistributedCheckpoint(group, directory)` | [ソース](../../ruda-torch/python/ruda_torch/distributed_checkpoint.py#L137) |
| `DistributedCheckpoint.save(model, optimizer, *, step, application_state=None, scheduler=None, scaler=None)` | [ソース](../../ruda-torch/python/ruda_torch/distributed_checkpoint.py#L140) |
| `DistributedCheckpoint.load(model, optimizer, *, generation=None, scheduler=None, scaler=None, validate_application=None)` | [ソース](../../ruda-torch/python/ruda_torch/distributed_checkpoint.py#L196) |
| `DistributedCheckpoint.consolidate(*, generation=None)` | [ソース](../../ruda-torch/python/ruda_torch/distributed_checkpoint.py#L256) |
| `DistributedCheckpoint.load_consolidated(model, optimizer, state, *, reshard_application, scheduler=None, scaler=None)` | [ソース](../../ruda-torch/python/ruda_torch/distributed_checkpoint.py#L307) |

## `adapter_interop`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.load_peft_adapter(model, directory, *, module_map=None, prefix='base_model.model.', adapter_dtype=torch.float32)` | [ソース](../../ruda-torch/python/ruda_torch/adapter_interop.py#L19) |
| `ruda_torch.save_peft_adapter(model, directory, *, base_model_name_or_path, prefix='base_model.model.', task_type=None, module_map=None, replica_group=None)` | [ソース](../../ruda-torch/python/ruda_torch/adapter_interop.py#L119) |

## `attention`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.scaled_dot_product_attention(query, key, value, *, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False)` | [ソース](../../ruda-torch/python/ruda_torch/attention.py#L9) |

## `block_attention`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.block_scaled_dot_product_attention(query, key, value, *, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, query_block_size=128, key_block_size=256, generator=None, causal_offset=0, window_size=None)` | [ソース](../../ruda-torch/python/ruda_torch/block_attention.py#L162) |

## `parallel_loss`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.vocab_parallel_cross_entropy(logits, targets, group, *, vocab_start=None, global_vocab_size=None, ignore_index=-100, label_smoothing=0.0, reduction='mean')` | [ソース](../../ruda-torch/python/ruda_torch/parallel_loss.py#L56) |

## `transformers_attention`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.register_transformers_attention(name='ruda_block', *, query_block_size=128, key_block_size=256)` | [ソース](../../ruda-torch/python/ruda_torch/transformers_attention.py#L7) |

## `parallel_mesh`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.ParallelMesh` | [ソース](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L11) |
| `ruda_torch.TensorParallelGroup` | [ソース](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L94) |

### `ParallelMesh`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ParallelMesh(*, data_parallel=1, tensor_parallel=1, pipeline_parallel=1, device_type='ruda', timeout_seconds=1800)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L18) |
| `ParallelMesh.data` | [ソース](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L63) |
| `ParallelMesh.tensor` | [ソース](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L66) |
| `ParallelMesh.pipeline` | [ソース](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L69) |
| `ParallelMesh.stage` | [ソース](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L72) |
| `ParallelMesh.stage_mesh()` | [ソース](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L74) |
| `ParallelMesh.close()` | [ソース](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L78) |

### `TensorParallelGroup`

| 公開名／呼び出し | ソース |
| --- | --- |
| `TensorParallelGroup(model, group)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L101) |
| `TensorParallelGroup.gather_metadata(value)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L108) |
| `TensorParallelGroup.validate_training_options(value)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L110) |
| `TensorParallelGroup.validate_model(model)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L112) |
| `TensorParallelGroup.total_weight(local_weight)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L117) |
| `TensorParallelGroup.validate_microbatches(batches)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L123) |
| `TensorParallelGroup.sum_(tensor)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L129) |
| `TensorParallelGroup.synchronize_gradients(*, local_weight, normalized=False, missing='zero')` | [ソース](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L133) |
| `TensorParallelGroup.prepare_optimizer_step(optimizer, *, loss_scale=1.0)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_mesh.py#L147) |

## `sharded_optim`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.ShardedMuon` | [ソース](../../ruda-torch/python/ruda_torch/sharded_optim.py#L49) |
| `ruda_torch.distributed_grad_norm(parameters, group, *, loss_scale=1.0, replicated_ids=(), replication_factors=None)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_optim.py#L11) |

### `ShardedMuon`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ShardedMuon(params, group, lr=0.02, *, momentum=0.95, weight_decay=0.0, nesterov=True, momentum_mode='sgd', dampening=0.0, ns_steps=5, ns_coefficients=(3.4445, -4.775, 2.0315), eps=1e-07, adjust_lr='original', matrix_layout='as_stored', stable_normalization=True, flatten=False, max_grad_norm=None, owners=None)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_optim.py#L60) |
| `ShardedMuon.step(closure=None, *, loss_scale=1.0)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_optim.py#L94) |
| `ShardedMuon.state_dict()` | [ソース](../../ruda-torch/python/ruda_torch/sharded_optim.py#L173) |
| `ShardedMuon.load_state_dict(record)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_optim.py#L179) |

## `quantization_interop`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.bnb_nf4_linear(state, weight_key, *, device='ruda:0', dtype=torch.float16, bias=None, tile_rows=128)` | [ソース](../../ruda-torch/python/ruda_torch/quantization_interop.py#L12) |
| `ruda_torch.load_bnb_nf4_safetensors(model, directory, *, target_modules, device='ruda:0', dtype=torch.float16, tile_rows=128)` | [ソース](../../ruda-torch/python/ruda_torch/quantization_interop.py#L98) |

## `parallel_adapters`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.ColumnParallelLoRALinear` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L169) |
| `ruda_torch.RowParallelLoRALinear` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L191) |
| `ruda_torch.ColumnParallelNF4Linear` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L92) |
| `ruda_torch.RowParallelNF4Linear` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L106) |
| `ruda_torch.shard_nf4_linear(layer, group, *, axis)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L35) |
| `ruda_torch.parallel_adapters.lora_from_parallel_base(base, *, rank=16, alpha=16.0, adapter_dtype=torch.float32, dropout=0.0, use_rslora=False)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L213) |
| `ruda_torch.parallel_adapters.merge_parallel_lora(layer)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L266) |
| `ruda_torch.parallel_adapters.parallel_linear(source, group, *, axis, **options)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L283) |
| `ruda_torch.parallel_adapters.parameter_shards(source, replacement, axis)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L294) |
| `ruda_torch.parallel_adapters.adapter_base_kind(layer)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L307) |
| `ruda_torch.parallel_adapters.full_adapter_matrix(layer, name)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L311) |
| `ruda_torch.parallel_adapters.local_adapter_matrix(layer, name, value)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L317) |
| `ruda_torch.parallel_adapters.nf4_layout(module, prefix)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L325) |
| `ruda_torch.parallel_adapters.join_nf4_field(values, spec)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L339) |
| `ruda_torch.parallel_adapters.slice_nf4_field(value, spec, rank, world)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L371) |
| `ruda_torch.parallel_adapters.full_nf4_state(module)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L401) |
| `ruda_torch.parallel_adapters.local_nf4_state(module, state, prefix)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L420) |

### `ColumnParallelLoRALinear`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ColumnParallelLoRALinear.scaling` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L158) |
| `ColumnParallelLoRALinear.get_extra_state()` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L160) |
| `ColumnParallelLoRALinear.set_extra_state(state)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L165) |
| `ColumnParallelLoRALinear(layer, group, *, gather_output=True)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L172) |
| `ColumnParallelLoRALinear.forward(value)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L181) |

### `RowParallelLoRALinear`

| 公開名／呼び出し | ソース |
| --- | --- |
| `RowParallelLoRALinear.scaling` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L158) |
| `RowParallelLoRALinear.get_extra_state()` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L160) |
| `RowParallelLoRALinear.set_extra_state(state)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L165) |
| `RowParallelLoRALinear(layer, group, *, input_is_parallel=False)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L194) |
| `RowParallelLoRALinear.forward(value)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L203) |

### `ColumnParallelNF4Linear`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ColumnParallelNF4Linear.get_extra_state()` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L84) |
| `ColumnParallelNF4Linear.set_extra_state(state)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L88) |
| `ColumnParallelNF4Linear(layer, group, *, gather_output=True)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L95) |
| `ColumnParallelNF4Linear.forward(value)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L101) |

### `RowParallelNF4Linear`

| 公開名／呼び出し | ソース |
| --- | --- |
| `RowParallelNF4Linear.get_extra_state()` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L84) |
| `RowParallelNF4Linear.set_extra_state(state)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L88) |
| `RowParallelNF4Linear(layer, group, *, input_is_parallel=False)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L109) |
| `RowParallelNF4Linear.forward(value)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_adapters.py#L116) |

## `parallel_checkpoint_io`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.load_tensor_parallel_safetensors(model, directory, group, *, device, dtype=torch.bfloat16, nf4_targets=(), block_size=64, tile_rows=128, parameter_dtypes=None, buffer_dtypes=None, nf4_format='dense')` | [ソース](../../ruda-torch/python/ruda_torch/parallel_checkpoint_io.py#L14) |
| `ruda_torch.load_tensor_parallel_bnb_nf4_safetensors(model, directory, group, *, target_modules, device='ruda:0', dtype=torch.float16, tile_rows=128, parameter_dtypes=None, buffer_dtypes=None)` | [ソース](../../ruda-torch/python/ruda_torch/parallel_checkpoint_io.py#L175) |

## `hybrid_parallel`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.DataTensorParallelGroup` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L10) |
| `ruda_torch.hybrid_parallel.join_mesh_field(values, spec)` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L142) |
| `ruda_torch.hybrid_parallel.slice_mesh_field(value, spec, rank, world)` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L173) |

### `DataTensorParallelGroup`

| 公開名／呼び出し | ソース |
| --- | --- |
| `DataTensorParallelGroup(model, mesh, *, data_root=0, tensor_root=0, broadcast_buffers=True)` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L18) |
| `DataTensorParallelGroup.gather_metadata(value)` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L41) |
| `DataTensorParallelGroup.validate_training_options(value)` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L43) |
| `DataTensorParallelGroup.validate_model(model)` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L45) |
| `DataTensorParallelGroup.total_weight(local_weight)` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L49) |
| `DataTensorParallelGroup.validate_microbatches(batches)` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L60) |
| `DataTensorParallelGroup.sum_(value)` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L67) |
| `DataTensorParallelGroup.synchronize_gradients(*, local_weight, normalized=False, missing='zero')` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L71) |
| `DataTensorParallelGroup.begin_gradient_overlap(*, global_weight, normalized=False, bucket_bytes=25 * 1024 * 1024)` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L90) |
| `DataTensorParallelGroup.replication_factors(parameters)` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L93) |
| `DataTensorParallelGroup.prepare_optimizer_step(optimizer, *, loss_scale=1.0)` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L97) |
| `DataTensorParallelGroup.checkpoint_layout(layout)` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L132) |

## `tensor_parallel_optim`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.TensorParallelMuon` | [ソース](../../ruda-torch/python/ruda_torch/tensor_parallel_optim.py#L13) |
| `ruda_torch.MeshShardedMuon` | [ソース](../../ruda-torch/python/ruda_torch/tensor_parallel_optim.py#L220) |

### `TensorParallelMuon`

| 公開名／呼び出し | ソース |
| --- | --- |
| `TensorParallelMuon(params, model, group, lr=0.02, *, momentum=0.95, weight_decay=0.0, nesterov=True, momentum_mode='sgd', dampening=0.0, ns_steps=5, ns_coefficients=(3.4445, -4.775, 2.0315), eps=1e-07, adjust_lr='original', matrix_layout='as_stored', stable_normalization=True, flatten=False, max_grad_norm=None, owners=None, consensus_group=None)` | [ソース](../../ruda-torch/python/ruda_torch/tensor_parallel_optim.py#L25) |
| `TensorParallelMuon.step(closure=None, *, loss_scale=1.0)` | [ソース](../../ruda-torch/python/ruda_torch/tensor_parallel_optim.py#L89) |
| `TensorParallelMuon.state_dict()` | [ソース](../../ruda-torch/python/ruda_torch/tensor_parallel_optim.py#L182) |
| `TensorParallelMuon.load_state_dict(record)` | [ソース](../../ruda-torch/python/ruda_torch/tensor_parallel_optim.py#L188) |

### `MeshShardedMuon`

| 公開名／呼び出し | ソース |
| --- | --- |
| `MeshShardedMuon(params, model, mesh, lr=0.02, *, momentum=0.95, weight_decay=0.0, nesterov=True, momentum_mode='sgd', dampening=0.0, ns_steps=5, ns_coefficients=(3.4445, -4.775, 2.0315), eps=1e-07, adjust_lr='original', matrix_layout='as_stored', stable_normalization=True, flatten=False, max_grad_norm=None, owners=None)` | [ソース](../../ruda-torch/python/ruda_torch/tensor_parallel_optim.py#L232) |
| `MeshShardedMuon.step(closure=None, *, loss_scale=1.0)` | [ソース](../../ruda-torch/python/ruda_torch/tensor_parallel_optim.py#L89) |
| `MeshShardedMuon.state_dict()` | [ソース](../../ruda-torch/python/ruda_torch/tensor_parallel_optim.py#L182) |
| `MeshShardedMuon.load_state_dict(record)` | [ソース](../../ruda-torch/python/ruda_torch/tensor_parallel_optim.py#L188) |

## `sharded_mesh`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.ShardedDataTensorParallelGroup` | [ソース](../../ruda-torch/python/ruda_torch/sharded_mesh.py#L118) |
| `ruda_torch.sharded_mesh.sharded_mesh_layout(model, mesh)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_mesh.py#L22) |
| `ruda_torch.sharded_mesh.join_fsdp_extra(values, spec)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_mesh.py#L84) |
| `ruda_torch.sharded_mesh.slice_fsdp_extra(value, spec, rank, world)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_mesh.py#L103) |

### `ShardedDataTensorParallelGroup`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ShardedDataTensorParallelGroup(model, mesh, *, data_root=0, tensor_root=0, broadcast_buffers=True)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_mesh.py#L128) |
| `ShardedDataTensorParallelGroup.gather_metadata(value)` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L41) |
| `ShardedDataTensorParallelGroup.validate_training_options(value)` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L43) |
| `ShardedDataTensorParallelGroup.validate_model(model)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_mesh.py#L171) |
| `ShardedDataTensorParallelGroup.total_weight(local_weight)` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L49) |
| `ShardedDataTensorParallelGroup.validate_microbatches(batches)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_mesh.py#L175) |
| `ShardedDataTensorParallelGroup.sum_(value)` | [ソース](../../ruda-torch/python/ruda_torch/hybrid_parallel.py#L67) |
| `ShardedDataTensorParallelGroup.synchronize_gradients(*, local_weight, normalized=False, missing='zero')` | [ソース](../../ruda-torch/python/ruda_torch/sharded_mesh.py#L186) |
| `ShardedDataTensorParallelGroup.begin_gradient_overlap(**options)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_mesh.py#L183) |
| `ShardedDataTensorParallelGroup.replication_factors(parameters)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_mesh.py#L193) |
| `ShardedDataTensorParallelGroup.prepare_optimizer_step(optimizer, *, loss_scale=1.0)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_mesh.py#L198) |
| `ShardedDataTensorParallelGroup.checkpoint_layout(layout)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_mesh.py#L219) |

## `pipeline_trainer`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.PipelineTrainer` | [ソース](../../ruda-torch/python/ruda_torch/pipeline_trainer.py#L15) |

### `PipelineTrainer`

| 公開名／呼び出し | ソース |
| --- | --- |
| `PipelineTrainer(stage, optimizer, mesh, *, stage_name, base_id, run_config, scaler=None, scheduler=None, input_is_parallel=False, tied_parameters=None)` | [ソース](../../ruda-torch/python/ruda_torch/pipeline_trainer.py#L24) |
| `PipelineTrainer.train_step(inputs=None, targets=None, *, local_weight, loss_sum, microbatch_specs=None, schedule='1f1b')` | [ソース](../../ruda-torch/python/ruda_torch/pipeline_trainer.py#L62) |
| `PipelineTrainer.train_step_autotuned(inputs=None, targets=None, *, local_weight, loss_sum, microbatch_specs=None, autotuner=None)` | [ソース](../../ruda-torch/python/ruda_torch/pipeline_trainer.py#L161) |
| `PipelineTrainer.save_distributed(directory, *, application_state=None)` | [ソース](../../ruda-torch/python/ruda_torch/pipeline_trainer.py#L193) |
| `PipelineTrainer.resume_distributed(directory, *, generation=None)` | [ソース](../../ruda-torch/python/ruda_torch/pipeline_trainer.py#L240) |
| `PipelineTrainer.consolidate(directory, *, generation=None)` | [ソース](../../ruda-torch/python/ruda_torch/pipeline_trainer.py#L265) |
| `PipelineTrainer.resume_consolidated(state, *, reshard_application, stage_transform=None)` | [ソース](../../ruda-torch/python/ruda_torch/pipeline_trainer.py#L272) |

## `pipeline_ties`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.PipelineTiedParameters` | [ソース](../../ruda-torch/python/ruda_torch/pipeline_ties.py#L11) |

### `PipelineTiedParameters`

| 公開名／呼び出し | ソース |
| --- | --- |
| `PipelineTiedParameters(model, mesh, bindings)` | [ソース](../../ruda-torch/python/ruda_torch/pipeline_ties.py#L20) |
| `PipelineTiedParameters.validate_optimizer(optimizer)` | [ソース](../../ruda-torch/python/ruda_torch/pipeline_ties.py#L73) |
| `PipelineTiedParameters.synchronize_gradients()` | [ソース](../../ruda-torch/python/ruda_torch/pipeline_ties.py#L83) |
| `PipelineTiedParameters.replication_count(parameter)` | [ソース](../../ruda-torch/python/ruda_torch/pipeline_ties.py#L92) |
| `PipelineTiedParameters.synchronize_state(optimizer)` | [ソース](../../ruda-torch/python/ruda_torch/pipeline_ties.py#L97) |
| `PipelineTiedParameters.close()` | [ソース](../../ruda-torch/python/ruda_torch/pipeline_ties.py#L120) |

## `training_data`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.StatefulShardSampler` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L9) |
| `ruda_torch.TokenBudgetBatchSampler` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L148) |
| `ruda_torch.collate_varlen_causal_lm(samples, *, ignore_index=-100)` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L287) |

### `StatefulShardSampler`

| 公開名／呼び出し | ソース |
| --- | --- |
| `StatefulShardSampler(length, *, source_id, rank=0, replicas=1, shuffle=False, seed=0, tail='uneven')` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L18) |
| `StatefulShardSampler.from_mesh(length, mesh, **options)` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L33) |
| `StatefulShardSampler.__len__()` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L47) |
| `StatefulShardSampler.remaining` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L52) |
| `StatefulShardSampler.__iter__()` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L54) |
| `StatefulShardSampler.commit(count)` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L61) |
| `StatefulShardSampler.rewind_uncommitted()` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L66) |
| `StatefulShardSampler.set_epoch(epoch)` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L70) |
| `StatefulShardSampler.state_dict()` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L78) |
| `StatefulShardSampler.load_state_dict(state)` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L86) |
| `StatefulShardSampler.validate_state_dict(state)` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L110) |
| `StatefulShardSampler.consolidate(states)` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L116) |
| `StatefulShardSampler.from_consolidated(state, *, rank, replicas)` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L137) |

### `TokenBudgetBatchSampler`

| 公開名／呼び出し | ソース |
| --- | --- |
| `TokenBudgetBatchSampler(sampler, lengths, *, maximum_tokens, layout, maximum_examples=None)` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L162) |
| `TokenBudgetBatchSampler.committed` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L178) |
| `TokenBudgetBatchSampler.issued` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L182) |
| `TokenBudgetBatchSampler.remaining_examples` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L186) |
| `TokenBudgetBatchSampler.peek()` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L210) |
| `TokenBudgetBatchSampler.__iter__()` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L214) |
| `TokenBudgetBatchSampler.__len__()` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L219) |
| `TokenBudgetBatchSampler.commit(examples)` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L227) |
| `TokenBudgetBatchSampler.rewind_uncommitted()` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L230) |
| `TokenBudgetBatchSampler.set_epoch(epoch)` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L233) |
| `TokenBudgetBatchSampler.state_dict()` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L236) |
| `TokenBudgetBatchSampler.validate_state_dict(state)` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L240) |
| `TokenBudgetBatchSampler.load_state_dict(state)` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L253) |
| `TokenBudgetBatchSampler.consolidate(states)` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L258) |
| `TokenBudgetBatchSampler.from_consolidated(state, *, rank, replicas)` | [ソース](../../ruda-torch/python/ruda_torch/training_data.py#L279) |

## `varlen_attention`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.varlen_scaled_dot_product_attention(query, key, value, cu_seqlens_q, cu_seqlens_k=None, *, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, query_block_size=128, key_block_size=256, generator=None, window_size=None, causal_alignment='upper_left')` | [ソース](../../ruda-torch/python/ruda_torch/varlen_attention.py#L17) |

## `training_checkpoint`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.training_state_dict(model, optimizer, *, base_id, step, data_state, scaler=None)` | [ソース](../../ruda-torch/python/ruda_torch/training_checkpoint.py#L42) |
| `ruda_torch.validate_training_state_dict(model, optimizer, state, *, base_id, scaler=None)` | [ソース](../../ruda-torch/python/ruda_torch/training_checkpoint.py#L76) |
| `ruda_torch.load_training_state_dict(model, optimizer, state, *, base_id, scaler=None)` | [ソース](../../ruda-torch/python/ruda_torch/training_checkpoint.py#L105) |

## `training_autotune`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.TrainingAutotuner` | [ソース](../../ruda-torch/python/ruda_torch/training_autotune.py#L278) |
| `ruda_torch.TrainingPlan` | [ソース](../../ruda-torch/python/ruda_torch/training_autotune.py#L50) |
| `ruda_torch.PipelineAutotuner` | [ソース](../../ruda-torch/python/ruda_torch/training_autotune.py#L515) |

### `TrainingAutotuner`

| 公開名／呼び出し | ソース |
| --- | --- |
| `TrainingAutotuner(*, repeats=3, max_candidates=12, max_seconds=300.0, rtol=0.0, atol=0.0, max_snapshot_bytes=None, progress=None)` | [ソース](../../ruda-torch/python/ruda_torch/training_autotune.py#L290) |
| `TrainingAutotuner.tune(trainer, microbatches)` | [ソース](../../ruda-torch/python/ruda_torch/training_autotune.py#L388) |
| `TrainingAutotuner.train_step(trainer, microbatches)` | [ソース](../../ruda-torch/python/ruda_torch/training_autotune.py#L493) |

### `TrainingPlan`

フィールド: `examples_per_microbatch: int`; <code>token_chunk_size: int &#124; None</code>; `packed: bool`; `geometry: tuple`.

| 公開名／呼び出し | ソース |
| --- | --- |
| `TrainingPlan.apply(trainer, batches)` | [ソース](../../ruda-torch/python/ruda_torch/training_autotune.py#L57) |

### `PipelineAutotuner`

| 公開名／呼び出し | ソース |
| --- | --- |
| `PipelineAutotuner(*, repeats=3, max_candidates=12, max_seconds=300.0, rtol=0.0, atol=0.0, max_snapshot_bytes=None, progress=None)` | [ソース](../../ruda-torch/python/ruda_torch/training_autotune.py#L290) |
| `PipelineAutotuner.tune(trainer, inputs=None, targets=None, *, local_weight, loss_sum, microbatch_specs=None)` | [ソース](../../ruda-torch/python/ruda_torch/training_autotune.py#L528) |
| `PipelineAutotuner.train_step(trainer, inputs=None, targets=None, *, local_weight, loss_sum, microbatch_specs=None)` | [ソース](../../ruda-torch/python/ruda_torch/training_autotune.py#L607) |

## `sharded_adapter_interop`

| 公開名／呼び出し | ソース |
| --- | --- |
| `ruda_torch.sharded_adapter_interop.has_sharded_units(model)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_adapter_interop.py#L11) |
| `ruda_torch.sharded_adapter_interop.adapter_layers(model, *, remove_duplicate=True)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_adapter_interop.py#L15) |
| `ruda_torch.sharded_adapter_interop.sharded_adapter_state_dict(model)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_adapter_interop.py#L54) |
| `ruda_torch.sharded_adapter_interop.load_sharded_adapter_state_dict(model, state)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_adapter_interop.py#L83) |
| `ruda_torch.sharded_adapter_interop.load_sharded_peft_adapter(model, pairs, config, *, module_map=None)` | [ソース](../../ruda-torch/python/ruda_torch/sharded_adapter_interop.py#L122) |
