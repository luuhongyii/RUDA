# Python parallel training, data, and recovery

[Documentation](README.md) · [Exact API signatures](python-api-signatures.md) · [Distributed training](distributed-training.md) · [中文](../zh/parallel-and-data.md)

## Device and rank ownership

The application launches processes, chooses each physical CUDA ordinal before native work, and initializes `torch.distributed`. RUDA still exposes `ruda:0` inside each worker. Native communication uses NCCL aliases of RUDA-owned CUDA storage; `device_type='cpu'` explicitly selects Gloo reference execution, not an automatic fallback.

`ParallelMesh(data_parallel=..., tensor_parallel=..., pipeline_parallel=...)` covers the entire initialized process group. Its shape is `(DP, PP, TP)`, its coordinates are `(data_rank, pipeline_rank, tensor_rank)`, and global rank is `(data_rank * PP + pipeline_rank) * TP + tensor_rank`. Every rank supplies identical configuration and creates groups in the same order. `mesh.data`, `mesh.pipeline`, `mesh.tensor`, and `mesh.stage` are different communication scopes; `close()` releases owned groups. The mesh neither launches workers nor infers stage partitions.

Sources: [ParallelMesh and TensorParallelGroup](../../ruda-torch/python/ruda_torch/parallel_mesh.py), [ReplicaGroup](../../ruda-torch/python/ruda_torch/distributed_training.py).

## Tensor-parallel model conversion

| API | Partition and data contract |
| --- | --- |
| `ColumnParallelLinear(linear, group, gather_output=True)` | Equal output-feature shards of an actual `nn.Linear`; input gradients sum across TP ranks. Disable gather to retain local features. |
| `RowParallelLinear(linear, group, input_is_parallel=False)` | Equal input-feature shards; scatter a full input unless already local, sum partial outputs, then add replicated bias once. |
| `VocabParallelEmbedding(embedding, group)` | Equal vocabulary shards; replicated token input and summed embedding output. `max_norm`, frequency-scaled gradients, and sparse embedding gradients are rejected. |
| `TensorParallelMLP(gate, up, down, group, activation=None)` | Local gate/up intermediate features feed a row-parallel down projection. Default activation is SiLU. |
| `TensorParallelAttention(query, key, value, output, group, num_heads=..., ...)` | Explicit MHA/GQA projections; query and KV head counts both divide TP size. Input is `[batch, sequence, hidden]`. Supply the model's positional transform and mask explicitly. |
| `tensor_parallelize(model, group, plan)` | In-place conversion using exact nonoverlapping module paths and explicit `column`, `row`, or `embedding` kinds. Include all consumers of tied weights with compatible partitions. Create the optimizer after conversion. |

The model-parallel derivative pairs are: `copy_to_tensor_parallel` copies forward / sums backward; `reduce_from_tensor_parallel` sums forward / passes backward; `gather_from_tensor_parallel` gathers forward / slices backward; `scatter_to_tensor_parallel` slices forward / gathers backward. These are not ordinary data-parallel gradient averaging.

`tensor_parallel_state_dict` collectively reconstructs full state on every rank; `load_tensor_parallel_state_dict` loads an ordinary full dictionary into the prepared partition. Full reconstruction can consume full-model host memory. For large local safetensors checkpoints, `load_tensor_parallel_safetensors` fills an already partitioned meta model directly; caller-provided dtype maps and exact NF4 targets remain explicit.

Source: [parallel projections, conversion and state](../../ruda-torch/python/ruda_torch/parallel_training.py), [streamed checkpoint loading](../../ruda-torch/python/ruda_torch/parallel_checkpoint_io.py).

## NF4 and LoRA partitions and interoperability

`ColumnParallelNF4Linear` and `RowParallelNF4Linear` partition the frozen packed base, while preserving its original codes/scales. `shard_nf4_linear` copies original scales into smaller blocks rather than requantizing floating weights. Packed bytes are not ordinary matrix-axis shards.

`ColumnParallelLoRALinear` uses local base/B output rows with replicated A and summed A gradients. `RowParallelLoRALinear` uses local base/A input columns and reduces rank activations before replicated B. Conversion follows the actual dense/NF4/LoRA layer type, not a model naming convention.

- `bnb_nf4_linear` and `load_bnb_nf4_safetensors` consume explicit bitsandbytes-style quantization metadata and local files; unsupported format/geometry is rejected. This is checkpoint interoperability, not a dependency on bitsandbytes device execution.
- `load_tensor_parallel_bnb_nf4_safetensors` streams prequantized bases into rank-local packed TP projections.
- `load_peft_adapter` loads explicit PEFT A/B matrices with caller-specified paths; `save_peft_adapter` exports full matrices to a new directory. It does not infer model-family target paths.

Sources: [parallel adapters](../../ruda-torch/python/ruda_torch/parallel_adapters.py), [NF4 checkpoint formats](../../ruda-torch/python/ruda_torch/quantization_interop.py), [PEFT adapter files](../../ruda-torch/python/ruda_torch/adapter_interop.py).

## FSDP, ZeRO, and exact Muon updates

`FullyShardedModule(module, group, ...)` stores rank-local parameter, gradient, and optimizer-state slices. `fully_shard(model, group, unit_paths=...)` wraps the root or explicit nonoverlapping units. Non-reentrant recomputation releases gathered full weights after forward and gathers them for backward. Every rank must execute the same collective-bearing unit order, including recomputation. Tied weights belong to one unit; keep their consumers together or use a root unit.

`Zero2Optimizer(parameter_groups, group, optimizer_factory)` leaves model parameters replicated, reduce-scatters gradients, and holds rank-local optimizer state. Its gradient synchronization takes explicit effective weight and missing-gradient policy. It is distinct from FSDP parameter sharding.

| Coordinator / optimizer | Responsibility |
| --- | --- |
| `ShardedReplicaGroup` | Data-parallel coordination for a model containing FSDP units. |
| `TensorParallelGroup` | One logical minibatch shared by TP ranks. |
| `DataTensorParallelGroup` | Token-weighted data replicas of TP models. |
| `ShardedDataTensorParallelGroup` | Data-reduce-scattered FSDP slices of actual TP layers, without a second DP sum. |
| `ShardedMuon`, `TensorParallelMuon`, `MeshShardedMuon` | Orthogonalize complete logical matrices, not local fragments; use the matching storage/topology contract. |
| `distributed_grad_norm` | Global L2 norm accounting for replicated tensors once and actual shards once each. |

Do not independently average TP ranks that share one loss, or apply full-matrix Muon to a fragment as though it were the whole parameter. Exact constructor/group/optimizer options are in the [signature reference](python-api-signatures.md).

Sources: [FSDP/ZeRO](../../ruda-torch/python/ruda_torch/sharded_training.py), [hybrid coordination](../../ruda-torch/python/ruda_torch/hybrid_parallel.py), [nested FSDP/TP](../../ruda-torch/python/ruda_torch/sharded_mesh.py), [sharded Muon](../../ruda-torch/python/ruda_torch/sharded_optim.py), [TP/mesh Muon](../../ruda-torch/python/ruda_torch/tensor_parallel_optim.py).

## Pipeline boundaries and optimizer steps

`PipelineStage(module, group, input_spec=..., output_spec=...)` executes one caller-constructed stage per pipeline rank. `PipelineTensorSpec` declares actual shape and dtype; adjacent output/input specs must match. `StructuredPipelineStage`, `PipelineTreeSpec`, `PipelineLeafSpec`, and `PipelineConstant` extend boundaries to explicit trees, dense tensor leaves, and immutable constants. Stage partitioning and invocation are not inferred from a pretrained architecture.

`PipelineTrainer(stage, optimizer, mesh, stage_name=..., base_id=..., run_config=...)` coordinates DP/TP/PP optimizer boundaries. Stage names are unique filesystem namespaces. First-stage inputs and last-stage targets are CPU microbatches; the caller supplies `loss_sum(output, target)` and actual `local_weight`. The default schedule is `1f1b`; every DP replica agrees on its logical weight across PP/TP ranks. FSDP pipelines require matching collective-bearing microbatch counts across replicas.

`PipelineTiedParameters` binds the same logical weight through explicit symbols on owner stages and validates its optimizer ownership. Save/resume uses the trainer's all-stage recovery methods; retrying a partially updated stage without restore is not recovery.

Sources: [tensor pipeline](../../ruda-torch/python/ruda_torch/pipeline_training.py), [tree boundaries](../../ruda-torch/python/ruda_torch/structured_pipeline.py), [trainer](../../ruda-torch/python/ruda_torch/pipeline_trainer.py), [stage ties](../../ruda-torch/python/ruda_torch/pipeline_ties.py).

## Committed data cursors and packed supervision

`StatefulShardSampler(length, source_id=..., rank=..., replicas=..., shuffle=False, seed=0, tail='uneven')` records consumed examples independently of DataLoader prefetch. Call `commit(n)` after consuming `n` examples; merely yielding indices does not advance saved progress. `tail='uneven'` keeps nonduplicated uneven shards; `tail='drop'` explicitly omits the final incomplete DP round. `from_mesh` uses the data coordinate, so TP/PP partners share the same sample order. Preserve the immutable dataset identity and explicit epoch/shuffle policy.

`TokenBudgetBatchSampler` batches using actual caller-provided token costs while retaining sampler order. `collate_varlen_causal_lm` packs explicit labels without cross-document next-token targets or implicit truncation. `PackedCausalLMFinetuner` expects a flat-token backbone/head contract; `CausalLMTrainer` provides full-parameter training instead of SFTTrainer's adapter-only snapshots.

Sources: [sampler and token budgets](../../ruda-torch/python/ruda_torch/training_data.py), [causal trainers](../../ruda-torch/python/ruda_torch/causal_finetuning.py).

## Full-model and all-rank checkpoints

`training_state_dict(model, optimizer, base_id=..., step=..., data_state=..., scaler=None)` captures full persistent model state, storage dtypes/ties, actual optimizer groups, optional scaler, CPU/device RNG and caller data state. Capture at a completed optimizer boundary with all gradients cleared. It does not write files or synchronize ranks. Nonpersistent buffers must be reconstructed by the same architecture.

`validate_training_state_dict` checks identity, geometry/dtypes, aliases, optimizer/scaler classes/layouts and RNG device sets before load. `load_training_state_dict` restores into existing parameters and returns the completed step and data state. Serialize with `torch.save`; load tensor/basic-container checkpoints using `torch.load(..., map_location='cpu', weights_only=True)`.

`DistributedCheckpoint(group, directory)` writes and validates rank-local files before publishing a generation manifest. `save`, `load`, `consolidate`, and `load_consolidated` have distinct storage/topology contracts; `model_shard_layout` describes actual FSDP/TP storage and logical ties. Application/sampler positions belong in application state. Topology-changing restoration requires explicit compatible resharding; a file from one rank is not an all-rank checkpoint.

PipelineTrainer's `save_distributed`/`resume_distributed` cover all stages; consolidated restoration takes explicit application resharding and optional stage transformation. Keep checkpoint files outside the repository.

Sources: [full snapshot](../../ruda-torch/python/ruda_torch/training_checkpoint.py), [all-rank generations](../../ruda-torch/python/ruda_torch/distributed_checkpoint.py), [all-stage recovery](../../ruda-torch/python/ruda_torch/pipeline_trainer.py).

## Whole-update autotuning

`TrainingAutotuner` measures actual training windows and searches microbatch and LM-head chunk layouts without consuming a lasting optimizer update. Trials snapshot and restore parameters, optimizer/scaler/scheduler, mutable state, RNG, counters, and gradients. The snapshot itself consumes memory; use actual intended geometry, not a smaller surrogate. `TrainingPlan.apply` checks batch geometry and preserves supervised token count.

`PipelineAutotuner` compares native 1F1B and fill/drain schedules on the unchanged stage mesh. This is separate from [operator-level stack autotuning](stack-autotuning.md). Signature options, candidate budgets and policy controls are in [training_autotune.py](../../ruda-torch/python/ruda_torch/training_autotune.py).

## Attention selection

| API | Storage and masking contract |
| --- | --- |
| `scaled_dot_product_attention` | Same-device floating MHA/GQA composition that materializes scores. Bool masks mean True=allowed; floating masks add to logits. Explicit causal masking and an external mask are mutually exclusive. |
| `block_scaled_dot_product_attention` | Same-device online-softmax query/key tiles with explicit dropout generator, offsets and window options; not a fused attention kernel. |
| `varlen_scaled_dot_product_attention` | `[total_tokens, heads, features]` payloads with explicit CPU integer `cu_seqlens`; documents are isolated. Unequal Q/K lengths use explicit `upper_left` or `lower_right` causal alignment. |
| `vocab_parallel_cross_entropy` | Local contiguous vocabulary intervals, identical logical targets/order across TP ranks, optional padded vocabulary; returns one logical loss, not a DP rank sum. |
| `register_transformers_attention` | Opt-in distinct `ruda_` name for Transformers AttentionInterface plus its standard mask adapter. It does not patch a model or override built-ins. |

Block score storage is bounded by tiles, but a Transformers-provided dense mask can still consume quadratic memory. Output attention weights, head masks and soft-capped logits are not supplied by this registration path. Sources: [dense attention](../../ruda-torch/python/ruda_torch/attention.py), [block attention](../../ruda-torch/python/ruda_torch/block_attention.py), [varlen attention](../../ruda-torch/python/ruda_torch/varlen_attention.py), [vocabulary loss](../../ruda-torch/python/ruda_torch/parallel_loss.py), [Transformers registration](../../ruda-torch/python/ruda_torch/transformers_attention.py).
