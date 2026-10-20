# Model-independent LoRA and NF4 fine-tuning

[Documentation](README.md) · [Training](training.md) · [中文](../zh/finetuning.md)

## Choose the integration layer

| Model interface | Entry point |
| --- | --- |
| Rust `ruda-model` model | `ruda_nn::LoRALinearConfig`, with the existing tensor backend and optimizer |
| Rust packed Llama/Qwen2 | `rullm::SftTrainer`, full-parameter SFT or `LlamaLora` adapters; see below |
| Ordinary PyTorch model | `ruda_torch.inject_lora`, optionally preceded by `quantize_nf4` |
| Local Hugging Face safetensors checkpoint | `load_hf_nf4_model` or the architecture-independent `load_nf4_safetensors` |
| Causal language-model training | `CausalLMFinetuner`, `SFTCollator`, `SFTTrainer` |

The model-independent APIs do not select a model family, infer adapter targets, download weights or install missing device operators. The model's actual forward and backward operations must be supported by its execution backend. Native PyTorch uses `ruda:0`, not `torch.cuda` storage. Prepare the matching [native components](../../crates/ruda-torch/README.md#build-and-install), including the NF4 extensions when using packed weights.

For Python safetensors streaming, install `safetensors`. The Python Hugging Face helper and command-line example also require `transformers` and `accelerate`. The selected architecture must exist in the installed Transformers version without remote code. FP16 is the low-precision option for T4; do not select BF16 on hardware without the required support.

## Rust ruLLM SFT and LoRA

`rullm::training` trains floating-point `PackedLlamaForCausalLm` models loaded from local Llama/Qwen2 checkpoints. It is separate from the Python `SFTTrainer`, PEFT, and NF4 APIs below; the Rust example does not need Python Transformers. Its backend must implement `AutodiffBackend + TrainingBackend`; the NVIDIA example uses `Autodiff<Cuda<bf16, i32>>`, or `Autodiff<Cuda<f32, i32>>` with `--f32`. This is not AWQ/NF4 quantized-base training or a Qwen3.5 trainer.

### Data and command line

Prepare a local [model directory](model-inference.md#prepare-a-local-model) and a JSONL dataset. Each nonblank line can be one of:

```jsonl
{"messages":[{"role":"user","content":"What is KV caching?"},{"role":"assistant","content":"It reuses keys and values from previous tokens."}]}
{"prompt":"Question: 1 + 1?\nAnswer:","completion":" 2"}
{"text":"A passage for continued causal-language-model training."}
```

The example uses `TextFormat::QwenChat`: `messages` receive Qwen2 ChatML formatting and only assistant spans are supervised. Prompt/completion records are tokenized as their literal strings, with only the completion supervised; text records supervise the whole text. Labels are unshifted in `SftExample`; `SftBatch::collate` shifts them once and right-pads with ignored targets (`IGNORE_INDEX = -100`). `from_jsonl` truncates to its `max_tokens` argument and returns the number of samples skipped for being too short or having no targets. The CLI passes `--max-length + 1`, producing at most `--max-length` input positions after shifting.

Run from the repository root; the output directory below is outside the checkout:

```powershell
$env:RUDA_CUDA_COMPILER = 'nvrtc'
cargo run --release --locked -p ruda-llm --features nvidia --example qwen2_sft -- ./models/qwen2 ../train.jsonl ../qwen2-sft-run --steps 200 --micro-batch 4 --accumulation 4 --max-length 1024 --checkpoint-every 50
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--steps` / `--lr` | `200` / `2e-4` | Total optimizer-step target and peak learning rate |
| `--micro-batch` / `--accumulation` | `4` / `4` | Examples per microbatch and microbatches per optimizer step |
| `--max-length` | `1024` | Maximum shifted input length; must fit the model's RoPE capacity |
| `--rank` / `--alpha` | `16` / `32` | LoRA rank and scaling numerator |
| `--full` | Not set | Train all model parameters instead of only LoRA adapters |
| `--checkpoint-every` | `50` | Optimizer steps between saves; 0 saves only at the end |
| `--f32` | Not set | Use FP32 rather than the default BF16 backend |

BF16 WMMA fragments are advertised only for CUDA sm_80 and newer; the capability check does not make those fragments available on older devices. Select a dtype supported by the model's actual operators; `--f32` is the example's alternative to BF16.

### Training APIs and memory controls

| API | Contract |
| --- | --- |
| `model.into_packed_training()` | Pack projections while retaining trainable parameters |
| `SftTrainer::full(model, config, device)` | Train the full packed model with AdamW |
| `LlamaLora::new(model.lora_shapes(), lora_config, &device)` | Build FP32 adapters; `SftTrainer::lora` freezes the base |
| `LoraTargets::ALL` / `ATTENTION` | Select packed qkv/output/gate-up/down projections, or only qkv/output; individual flags are also available |
| `SftLoader::new(dataset, micro_batch, pad_token_id, seed)` | Repeat seeded shuffled epochs; default length-sorting window is 64 |
| `trainer.step(&micro_batches)` | Perform one token-normalized accumulated step; caller owns loader-position bookkeeping |
| `trainer.train(&mut loader, callback)` | Drive to `schedule.total_steps`, save loader position/checkpoints, and invoke a nonblocking progress callback |

LoRA A/B matrices use `[in, rank]` / `[rank, out]`, with a zero-initialized B and update scale `alpha / rank`. They remain FP32 and are cast at the projection boundary. The CLI selects `LoraTargets::ALL` and seed 0. These are ruLLM packed-projection adapters, not a PEFT adapter checkpoint format.

Attention uses tiled causal/windowed forward and recomputed backward probabilities. The vocabulary loss processes full-vocabulary row chunks instead of retaining `[batch, sequence, vocabulary]` logits. `TrainConfig::loss_chunk_bytes` defaults to 128 MiB for one FP32 logits chunk; it is not a total GPU-memory limit. Accumulated losses are divided by the window's total valid-target count, not averaged microbatch means. AdamW defaults are betas 0.9/0.95, epsilon 1e-8, weight decay 0, global-norm clipping at 1.0, and linear warmup followed by cosine decay to 10% of peak. Nonfinite loss/norm skips the update and increments `skipped_steps`.

`TrainProgress` reports step/total, loss, learning rate, gradient norm, tokens, elapsed time, recent token rate, ETA, epoch, and latest-checkpoint age. The CLI emits JSON lines without interactive prompts. The callback does not measure GPU utilization or memory. For explicitly registered ruCCL peers, feature `distributed` adds `with_data_parallel(peer)`; shard the loader with `with_shard(rank, world_size)`. The CLI does not launch ranks or initialize collectives.

### Checkpoints, resume, and export

The CLI writes `<output-dir>/checkpoints/step-NNNNNNNN`, containing `trainable.safetensors`, the optimizer record, and `state.json`; `latest` is a text file naming the newest checkpoint. The API's `keep_checkpoints` defaults to 1, and older step directories are pruned. Model and optimizer files are staged in a temporary directory before promotion. Keep space for the current checkpoint, its replacement, and the final exported weights.

Re-running the same command/output directory calls `trainer.resume()` automatically. Reconstruct the same base model, training mode, adapter layout, dataset/order, microbatch size, and training configuration. LoRA checkpoints contain adapters and optimizer state, **not the frozen base**; they do not replace the original model directory. `train` restores the loader's epoch/cursor/seed, then runs until the configured total step target, not for that many additional steps. `load_checkpoint(path)` selects a specific checkpoint. Adam/AdamW restored moments are converted to the current gradient dtype so FP32 adapters retain compatible optimizer state on a BF16 base.

At the end, the example merges LoRA into the base when applicable and writes `<output-dir>/model.safetensors`. Copy the source `config.json`, tokenizer files, and applicable generation configuration alongside it to form a loadable model directory; do not reuse a sharded-weight index describing the original weights. `export_huggingface_safetensors(&model, path, tie_word_embeddings)` writes HF Llama/Qwen2 tensor names and omits `lm_head.weight` when embeddings are tied. This export is deployment weights, not an optimizer/loader checkpoint for resuming training.

## Select projections explicitly

Inspect `model.named_modules()` before modifying the model. `target_modules` is either the string `'all-linear'` or a nonempty sequence of exact qualified module names, not suffix matching or a regular expression. A root `Linear` must first be placed in a container. Selecting one alias of a shared module also replaces its other aliases.

For dense LoRA, call `inject_lora(model, target_modules=targets, rank=16, alpha=16., adapter_dtype=torch.float32)`. It modifies and returns the model, freezes all original parameters, clears their gradients and leaves only adapter parameters trainable. Construct the optimizer **after** injection and pass only parameters whose `requires_grad` is true. Repeated injection is rejected.

Each `LoRALinear` computes `base(x) + (alpha / rank) * B(A(x))`. The base is `torch.nn.Linear` or `NF4Linear`; A has shape `[rank, in_features]` and B `[out_features, rank]`. B starts at zero. The Python adapter has no dropout parameter. `rank` is a positive integer, `alpha` is finite and positive, and `adapter_dtype` is FP32, FP16 or BF16. The update is converted to the base output dtype before addition.

### Rust LoRA

`LoRALinearConfig::new(rank, alpha).with_dropout(p).init(base)` consumes an existing Rust `Linear<B>` and freezes its base without changing base parameter IDs. A maps input width to rank, B maps rank to output width and starts at zero. Both use the base weight dtype. `forward` preserves supported leading dimensions.

Rust requires `rank > 0`, finite `alpha`, and `0 <= dropout < 1`; invalid configuration asserts. Unlike the Python API, Rust does not require positive alpha. `merge(self)` consumes the adapter and produces a frozen dense projection without adapter dropout. It is not an optimizer-state conversion or a checkpoint for resuming LoRA training. See [Rust implementation](../../crates/ruda-nn/src/modules/lora.rs).

## Pack a frozen NF4 base

For a model already loaded on CPU, call `quantize_nf4(model, target_modules=targets, block_size=64, tile_rows=128)` **before** injecting LoRA and before moving the model to RUDA. Selected weights must be contiguous CPU FP32/FP16/BF16 matrices. Keep embedding/output ties outside the quantization targets: shared module aliases are preserved, but a weight tied to another module is rejected. Conversion commits one layer at a time; failure leaves completed layers converted.

`NF4Linear` keeps packed weight codes, FP32 block scales and an FP32 codebook, without a retained dense weight shadow. It supports first-order input gradients, not gradients for the frozen packed base. Moving or changing the module's floating dtype keeps scales and the codebook in FP32.

The RUDA NF4 format is row-major, two codes per byte, first value in the high nibble. Each flat block has one FP32 absolute maximum. Partial final blocks and odd element counts are supported. This is not a bitsandbytes or PEFT checkpoint format, and NF4 is not interchangeable with AWQ INT4 or learned fake quantization.

On RUDA, FP16/BF16 uses fused tile dequantization and GEMM when NF4 matmul API 1 is present. FP32, or absence of that optional matmul capability, uses bounded tiled decoding; decoding still requires NF4 API 1. A failed fused call is not retried through the tiled path. `tile_rows` controls the decoded output-row tile on that path, not the training batch size. CPU/CUDA execution is explicitly selected same-device reference execution, not a recovery path from a failed RUDA call.

## Stream a local checkpoint

`load_nf4_safetensors(model, directory, target_modules=targets, device='ruda:0', dtype=torch.float16)` expects an already constructed model with **all parameters on meta**. It reads `model.safetensors`, or `model.safetensors.index.json` and its listed shards, one tensor at a time. Selected dense weights are quantized on CPU and uploaded packed; remaining tensors are loaded separately. Tensor names and shapes must match the architecture exactly. There is no key-renaming rule or tokenizer handling.

- `parameter_dtypes` and `buffer_dtypes` are dictionaries of exact floating tensor names to FP32/FP16/BF16. Conflicting overrides on tied tensors are rejected.
- A dtype-preserved weight cannot also be an NF4 target. Non-target floating parameters use `dtype` unless overridden; buffers preserve their original dtype unless overridden.
- Nonpersistent buffers omitted from the checkpoint must already be materialized by the constructor, not left on meta.
- Shards must be existing files inside the checkpoint directory. Missing tensors, mismatched shapes/kinds, invalid ties or unsupported target types raise errors.
- Loading is not transactional: after a failure, retry with a freshly constructed meta model.

`load_hf_nf4_model(directory, target_modules=targets, device='ruda:0', dtype=torch.float16, rank=16, alpha=16.)` creates a local HF architecture on meta, reapplies its declared weight ties, streams the base and injects adapters. `auto_class` defaults to `AutoModelForCausalLM`; callers can supply another compatible AutoModel class. `config_kwargs` and `model_kwargs` are passed to the relevant constructors; local-file and remote-code controls cannot be overridden. `_keep_in_fp32_modules` declarations are respected, and `parameter_dtypes` can supply explicit overrides. The helper does not choose the text backbone, vocabulary head or multimodal submodel for the caller.

Both loaders default to `dtype=torch.bfloat16`; explicitly pass FP16 for T4. Their packing defaults are `block_size=64, tile_rows=128`. The HF helper additionally defaults to `rank=16, alpha=16., adapter_dtype=torch.float32`; `parameter_dtypes`, `config_kwargs` and `model_kwargs` default to None. The lower-level loader also accepts `buffer_dtypes=None`.

## Supervision, loss and accumulation

`SFTCollator(tokenizer=None, max_length=..., pad_token_id=..., train_on_prompt=False, truncate=False)` right-pads batches on CPU and returns exactly `input_ids`, `attention_mask` and `labels`, all shaped `[B,T]`. IDs/labels are int64; the mask is Bool. Padding labels are `-100`.

Supply one of these input forms:

- Pretokenized records with nonempty, equally sized `input_ids` and `labels`. Labels are vocabulary IDs or `-100`. IDs are nonnegative integers. Do not pre-shift labels.
- Chat records with `messages` and a tokenizer. By default, its chat template must declare assistant spans through `assistant_masks` or `assistant_tokens_mask`. Missing/empty spans are errors. `train_on_prompt=True` explicitly supervises all template tokens instead.

`pad_token_id` must be supplied or declared by the tokenizer; it is not guessed. Samples exceeding `max_length` fail unless `truncate=True`, which explicitly keeps the prefix. `template_kwargs` cannot override encoding/supervision controls. The attention mask and label mask have separate purposes.

`CausalLMFinetuner(backbone, head, token_chunk_size=32, activation_checkpointing=True, checkpoint_modules=None, preserve_rng_state=True)` requires an explicit backbone/head split. The backbone accepts `input_ids` and `attention_mask` and returns `[B,T,D]` hidden states directly or through `last_hidden_state`; vocabulary logits are not a backbone result. A backbone with `config` is called with `use_cache=False, return_dict=True`. The head is a dense, NF4 or LoRA linear layer with input width D.

`chunked_lm_cross_entropy(hidden, head, labels, token_chunk_size=32, ignore_index=-100, shift=True, reduction='mean', recompute=True)` projects each token chunk against the **complete vocabulary**. It never constructs the full `[B,T,V]` logits tensor. `shift=True` pairs hidden positions `[:-1]` with labels `[1:]`. `mean` divides by nonignored target count; `sum` returns their loss sum. No valid targets yields a differentiable zero. Labels must be int32/int64 on the hidden-state device; nonignored IDs must be valid vocabulary indices. Backward recomputes chunk logits when requested, retaining head and hidden-state gradients.

Activation checkpointing uses non-reentrant recomputation. Explicit `checkpoint_modules` are nonempty, distinct, non-overlapping paths relative to the backbone. Otherwise the wrapper calls its checkpointing interface, or checkpoints the whole backbone. `activation_checkpoint_modules(model, paths, preserve_rng_state=True)` is also available directly and preserves state-dict paths. Disable RNG preservation only deliberately for deterministic forwards; it is not equivalent to saving RNG for a later process restart.

| Method | Arguments and return |
| --- | --- |
| `SFTCollator.encode(sample)` | One chat/pretokenized record → two Python lists `(input_ids, labels)`, before padding. |
| `SFTCollator(samples)` | Nonempty batch → the three CPU tensors described above. `template_kwargs=None` is the default. |
| `CausalLMFinetuner.hidden(input_ids, attention_mask)` | Backbone inputs → `[B,T,D]` hidden states without the vocabulary projection. |
| `CausalLMFinetuner.forward(input_ids, attention_mask, labels, reduction='mean')` | Scalar shifted full-vocabulary loss; reduction is mean or sum. |
| `activation_checkpoint_modules(model, target_modules, preserve_rng_state=True)` | Modifies and returns the model; already checkpointed modules and overlapping paths are errors. |

`SFTTrainer(model, optimizer, base_id=..., run_config=..., scaler=None, scheduler=None, replica_group=None)` accepts that wrapper or its RUDA compiled wrapper. `train_step(microbatches)` requires collated CPU batches and at least one supervised shifted target across the window. It transfers each batch to the model device, backpropagates each loss sum divided by the **total token count across the window**, then performs one optimizer step and clears gradients. It does not average unequal microbatch means. An optional scaler requires an optimizer exposing `last_step_skipped`; the scheduler advances only when the update is not skipped.

With an explicitly initialized [ReplicaGroup](distributed-training.md#python-replicated-training-with-nccl), normalization/loss reporting use global tokens and accumulated gradients synchronize before the update. A local empty window is allowed when the group has supervision. Initialize the final model before creating the optimizer; save separate rank-local checkpoints and restore matching rank/world size. The default remains single-process.

Metrics include attempted step/window number, microbatch cursor, supervised token counts, mean loss, elapsed time, rate and whether the optimizer update was skipped. Step/cursor counters still advance after a skipped update. `write_progress(directory, metrics, total_steps=...)` writes local `progress.json`; its ETA uses recent comparable step durations. RUDA GPU peak memory is reported as unknown rather than inferred from CUDA allocations.

## Command-line training

The existing [example](../../crates/ruda-torch/python/examples/finetune_causal_lm.py) uses local model files and JSONL input. Set `MODEL_DIR`, `DATA_JSONL`, `RUN_DIR`, `BASE_ID`, `BACKBONE`, `HEAD`, `PAD_TOKEN_ID`, and the Bash array `TARGETS` to your actual checkpoint, dataset, exact module paths and tokenizer padding ID. `RUN_DIR` must be outside the repository; `BASE_ID` identifies the exact frozen weights and quantization configuration. CLI `--targets` takes exact paths, not the API's `'all-linear'` string.

Start with a bounded trial on the intended batch/sequence shapes:

```bash
python crates/ruda-torch/python/examples/finetune_causal_lm.py \
  --model "$MODEL_DIR" --data "$DATA_JSONL" --output "$RUN_DIR" \
  --base-id "$BASE_ID" --backbone "$BACKBONE" --head "$HEAD" \
  --targets "${TARGETS[@]}" --dtype fp16 --pad-token-id "$PAD_TOKEN_ID" \
  --max-length 512 --batch-size 1 --accumulation 2 --steps 2 \
  --rank 16 --alpha 16 --token-chunk-size 32 --checkpoint-every 1
```

Add `--chat` for messages, `--train-on-prompt` or `--truncate` only when those supervision policies are intended. `--checkpoint-modules` takes explicit backbone-relative paths. `--compile` enables the general AOT compiler; it does not imply all operations are native. `--cpu-reference` is a deliberate CPU reference mode. The file is traversed once without implicit repetition or shuffling and must contain enough full microbatches for the requested steps.

Keep the trial's checkpoints/progress outside Git. Assess actual step time and memory on comparable inputs before increasing the workload; a small-input trial is not a capacity estimate for a larger model or sequence. To resume, repeat the same configuration with `--resume "$RUN_DIR/checkpoints/latest.pt"`; `--steps` is the new total target, not an additional step count.

## Save adapters or resume training

| API | Contents / contract |
| --- | --- |
| `adapter_state_dict(model)` | CPU copies of A/B plus versioned target names, dimensions, rank, alpha and dense/NF4 base kind; no base weights |
| `load_adapter_state_dict(model, state)` | Requires an already prepared model with identical adapter layout; validates all entries before copying tensors |
| `finetune_state_dict(model, optimizer, base_id=..., step=..., data_state=..., scaler=None)` | Adapter, optimizer type/state/group ordering, optional scaler, CPU RNG, explicitly used CUDA RNG, completed-window counter and caller data state |
| `load_finetune_state_dict(model, optimizer, state, base_id=..., scaler=None)` | Restores those values, clears gradients and returns `(step, data_state)` |
| `SFTTrainer.save(directory)` | Also records run configuration, scheduler and cursor/token totals; writes and checks `next.pt`, rotates `latest.pt` to `previous.pt`, then promotes the new checkpoint |
| `SFTTrainer.resume(path)` | Requires exact `run_config`, base identity, optimizer layout/type and scheduler/scaler configuration; restores trainer counters |

Snapshot only at optimizer-step boundaries **after** `zero_grad(set_to_none=True)` and scaler `update()`. Partial accumulated gradients are not saved. The optimizer must contain every trainable A/B parameter exactly once and no frozen/non-adapter parameter. The frozen base is not duplicated: recreate the same base, quantization and adapters, then the same optimizer groups, before restoring. Restore the data source to the saved microbatch cursor; custom sampler state belongs in `data_state`.

For direct APIs, serialize the returned mapping with `torch.save` and load with `torch.load(..., map_location='cpu', weights_only=True)` using compatible tensor/basic-container state. `base_id` and `run_config` are supplied by the caller; the CLI additionally captures input/source/dependency/native-library identities. The low-level snapshot does not automatically capture Python/NumPy RNG, arbitrary external sampler state or an independent RUDA device RNG state.

`merge_lora(model)` is only for non-root, eval-mode **dense** LoRA layers whose base weights are not shared with another module. It permanently replaces them with frozen dense layers. NF4 adapters are not expanded/requantized by this API; keep the adapter representation for NF4 deployment. Save an adapter checkpoint before merging if later adapter training is needed.

## Constructor and packing reference

| API | Parameters and limits |
| --- | --- |
| `finetuning.pack_nf4(weight, block_size=64, chunk_blocks=1024)` | Nonempty contiguous CPU `[out,in]` FP32/FP16/BF16 finite weight; positive even block size and positive chunk count; returns uint8 bytes and FP32 scales |
| `NF4Linear(in_features, out_features, packed, scales, block_size=64, tile_rows=128, bias=None)` | Positive geometry/tile values; at most uint32 weight elements and block size; contiguous same-device packed/scales with lengths `ceil(out*in/2)` and `ceil(out*in/block_size)`; scales frozen; optional floating bias `[out]` |
| `NF4Linear.from_linear(linear, block_size=64, tile_rows=128)` | CPU dense layer preprocessing; retains its training mode and frozen bias |
| `NF4Linear.forward(x)` | FP32/FP16/BF16 `[...,in]` on the weight device → `[...,out]`; autocast uses the device's selected activation dtype |
| `LoRALinear(base, rank=16, alpha=16., adapter_dtype=torch.float32)` | Dense/NF4 base; freezes base parameters and creates trainable A/B on its device |
| `quantize_nf4`, `inject_lora` | In-place model conversion; exact target names; quantize before injection; optimizer created afterward |

NF4Linear and LoRALinear `get_extra_state()` / `set_extra_state(state)` store and strictly check their versioned geometry/configuration during ordinary module state-dict loading. They are separate from adapter-only export and do not load a missing frozen base.

Invalid metadata/configuration raises `ValueError`/`TypeError`; missing native capabilities raise errors rather than CPU fallback. Read [packing and checkpoint implementation](../../crates/ruda-torch/python/ruda_torch/finetuning.py), [causal training implementation](../../crates/ruda-torch/python/ruda_torch/causal_finetuning.py), and the [native API reference](native-pytorch-api.md) for adjacent training interfaces.

## DPO, PPO and GRPO

Python exports `DPOModel`/`DPOTrainer`, `PPOModel`/`PPOTrainer`, `GRPOModel`/`GRPOTrainer`, `CausalValueModel`, `PackedCausalValueModel`, `PreferenceCollator`, `select_response_batch`, and `chunked_lm_log_probs`. Policies and frozen references use the explicit padded `CausalLMFinetuner` or flat-token `PackedCausalLMFinetuner` backbone/head contract (dense, NF4 or LoRA); construct independent references with the same layout and supply `reference_id`. Prompt/padding labels are `-100`; completion labels equal input IDs. Full-vocabulary log probabilities use chunked projection and backward recomputation, including vocabulary-parallel heads. Dropout is disabled for rollout and update forwards without disabling parameter gradients.

```python
objective = DPOModel(policy, reference, reference_id=reference_id, beta=0.1)
optimizer = torch.optim.AdamW([p for p in objective.parameters() if p.requires_grad], lr=1e-6)
trainer = DPOTrainer(objective, optimizer, base_id=base_id, run_config=run_config)
collator = PreferenceCollator(max_length=max_length, pad_token_id=pad_token_id)
metrics = trainer.train_step([collator(preference_pairs)])
```

Each preference pair contains explicitly labeled `chosen` and `rejected` samples with identical prompts. DPO uses completion-summed policy/reference log-ratios and the mean negative log-sigmoid over pairs. `PreferenceCollator` arranges chosen rows first, then rejected rows.

For PPO, build `PPOModel(policy, reference, critic, reference_id=...)`; `critic` is an independent `CausalValueModel(backbone, scalar_head, ...)`. Create the optimizer **after** this wrapper so it owns both actor and critic trainable parameters; separate parameter groups may select different learning rates. `PPOTrainer.prepare_rollout(batch, rewards, terminal)` accepts a CPU response batch, actual per-token rewards `[B,T-1]`, and boolean terminal flags `[B]`. It snapshots old-policy log probabilities and values, subtracts `kl_coefficient * (old_logp-reference_logp)` from rewards, and computes GAE (`gamma`, `gae_lambda`). EOS trajectories bootstrap zero; truncated trajectories bootstrap the final critic value. Keep the returned batch fixed across PPO epochs and pass it to `train_step`. The objective is token-mean clipped policy loss plus `0.5 * value_coefficient` times clipped value error; `value_clip=None` disables value clipping. No entropy bonus or reward function is implicit.

For GRPO, use `GRPOModel(policy, reference, reference_id=...)` and `GRPOTrainer.prepare_groups(batch, rewards, group_ids)`. Supply outcome rewards and complete same-prompt groups with at least two responses. Advantages use population standard deviation plus `reward_epsilon`; equal rewards give zero advantage. Fixed old/reference log probabilities feed the clipped ratio objective and differentiable `exp(reference_logp-logp) - (reference_logp-logp) - 1` KL estimator. Reduction is the mean of per-response token means, including unequal response lengths. Snapshot complete groups before splitting them into update microbatches.

PPO/GRPO trainers provide `sample(prompt, max_new_tokens=..., eos_token_ids=...)`: unfiltered temperature-one full-vocabulary categorical sampling, including EOS, with an explicit terminal/truncated result. The caller supplies prompts, grouping and reward computation; sampling never invents these. TP sampling broadcasts selected tokens within the vocabulary group. Models must have sufficient context capacity for the requested rollout.

For backbones whose logits or values depend on total forward length (for example dynamic/long RoPE), construct the Python objective with `prefix_scoring=True`. It evaluates each action/value on its actual unpadded autoregressive prefix rather than the final response length, at the cost of one backbone forward per prefix. Native trainers select this path automatically for runtime RoPE. With a replicated vocabulary head over a TP backbone, supply the actual TP `rollout_group` to the Python trainer so sampling broadcasts tokens there as well.

All three trainers reuse `CausalLMTrainer` accumulation, scheduler/scaler, coordinator and atomic local/distributed checkpoint paths. Normalization is global pairs (DPO), completion tokens (PPO), or responses (GRPO), not averages of unequal microbatch means. Metrics report `normalization_units`, not SFT token counts. `evaluate` reports objectives rather than perplexity. `save`/`resume` include actor, reference, critic where present, optimizer slots, algorithm/reference contract, RNG and `trainer.data_state`. Put source position, fixed pending rollouts, PPO/GRPO epoch/minibatch cursors and any explicit generator state in `data_state` before saving; no partially accumulated gradients are saved. Use `save_distributed`/`resume_distributed` with the actual FSDP/ZeRO/DP/TP coordinator for all-rank commits.

For packed Python post-training, use `PreferenceCollator(packed=True)` for DPO/reward pairs and `collate_varlen_causal_lm(samples)` for PPO/GRPO responses. Supply explicitly labeled samples with a nonempty prompt and contiguous nonempty completion. Packed batches have flat `input_ids`, `attention_mask`, `labels`, `position_ids` `[N]`, CPU integer `cu_seqlens` `[D+1]`, and integer `max_seqlen`; there is no padding or implicit truncation. The backbone must consume this metadata and implement actual document-isolated varlen attention. PPO uses an independent `PackedCausalValueModel(backbone, scalar_head)` with the same layout. Context sharding uses the separate `ContextParallelFinetuner` interface below.

Packed PPO rewards, log probabilities, old action values, advantages and returns are target-aligned `[N]`: action at token `j` predicts that token from its preceding state within its own document. Document starts and prompt targets are masked. The standalone `chunked_lm_log_probs` utility returns `[N-1]` for shifted flat inputs; objective wrappers prepend zero to align `[N]`. GAE restarts per document; true terminal responses bootstrap zero and truncated responses bootstrap their own final critic state. GRPO outcome rewards/group IDs/advantages and reward-model scores remain `[D]`. DPO/reward pair means, PPO token means and GRPO response means retain the padded weighting rules. Packed `prefix_scoring=True` uses each document's actual prefix and corresponding position IDs; `sample` returns a one-document packed batch.

Use `select_response_batch(fixed_rollout, indices)` to split already snapshotted PPO/GRPO responses into microbatches. It selects whole documents, rebuilds packed boundaries/maximum length, preserves explicit positions, and copies fixed statistics without rescoring the old policy. It also supports padded batches. For DPO/reward selection, keep all selected chosen responses first, followed by their matched rejected responses; for example, `[0, 2]` and `[1, 3]` split a two-pair chosen-first batch. Packed checkpoints record the layout and cannot be resumed through a padded objective.

For context-parallel DPO/PPO/GRPO, construct the policy and independent reference with `ContextParallelFinetuner(backbone, head, context_group, group_key=...)`. PPO uses `ContextParallelValueModel(critic_backbone, scalar_head, context_group, group_key=...)`; reward learning wraps the same independent scalar model in `RewardModel`. All paths must use the actual same CP process group. Keep the existing HF/native ring-attention adapter and supply the model-mesh coordinator containing that CP group (or its `ReplicaGroup` for a CP-only job). Hidden states, attention KV and trainable weights are not gathered to compute post-training losses.

Every CP peer supplies the same complete detached CPU source batch, padded or packed. `train_step`/`evaluate` shard complete batches automatically; `shard_post_training_batch(batch, context_group)` explicitly returns local token/position/document tensors, pre-shifted `context_targets`, and CPU `response_metadata`. DPO/reward pair losses and GRPO response sums/counts are reduced across CP before the nonlinear objective; each pair/response is counted once, not once per rank. PPO sums disjoint completion-action losses. The coordinator then sums the corresponding CP parameter gradients exactly once.

`prepare_rollout` and `prepare_groups` still accept complete CPU source responses and caller rewards. They gather only detached per-token scalar log-probability/value snapshots, compute document-local GAE or complete-group advantages, and return fixed context-sharded updates. EOS/truncation bootstraps retain the same rules across shard boundaries. `select_response_batch` also repacks these context updates without rescoring. `score` returns the complete reward vector on each CP peer; `token_rewards` returns sparse rewards in the complete source layout. `prefix_scoring=True` repartitions each actual response prefix, transfers its selected scalar and adjoint between prefix/response owners, and never substitutes final-length RoPE. CP `sample` executes the sharded backbone through the trainer wrapper and broadcasts selected tokens across CP and the supplied TP rollout group. All peers must call these collective-bearing APIs in the same order. Put pending context updates and source/epoch cursors in `data_state` before the existing local/all-rank checkpoint calls; context checkpoints have a distinct layout contract.

Native Rust exports the same algorithms as `rullm::training::{DpoTrainer, PpoTrainer, GrpoTrainer}` over `SftTrainer` actors (full parameters or LoRA). Construct `ReferencePolicy::new(model, adapters, identity)`, then `DpoTrainer::new(actor, reference, beta, seed)`, `PpoTrainer::new(actor, ValueCritic::new(critic_backbone, device), critic_train_config, reference, ppo_config, seed)`, or `GrpoTrainer::new(actor, reference, grpo_config, seed)`. `Response` separates prompt/completion and terminal status; `PreferencePair` contains chosen/rejected responses. PPO `prepare_rollout(response, token_rewards)` and GRPO `prepare_group(responses, outcome_rewards)` produce serializable fixed rollout statistics. `step` performs a real optimizer update over the supplied effective batch; PPO updates the independent critic optimizer as well. PPO/GRPO `sample` uses the persisted categorical RNG. Attach ruCCL to the actor using `with_data_parallel` before wrapping; local batches must be nonempty and replicas call updates in the same order. Normalization weights and both actor/critic gradients are reduced through that peer. `save_checkpoint`/`load_checkpoint` atomically retain actor/optimizer, critic/optimizer, algorithm/reference metadata and `state` under the actor's checkpoint-retention policy. Put pending rollout/update/source cursors in `state.data`; reload the same immutable reference (and the frozen base for LoRA) before resuming. These APIs do not start GPU jobs automatically.

### Native packed batches and paged rollouts

Rust `DpoTrainer::step`, `PpoTrainer::step`, `GrpoTrainer::step` and `RewardTrainer::step` pack supplied responses into flat, document-isolated training forwards, including full/LoRA projections and memory-efficient attention backward. Pair, completion-token and response normalization remain unchanged. `PpoTrainer::prepare_rollouts(responses, &token_rewards)` snapshots a ragged batch and restarts GAE per document; `RewardTrainer::score_batch` and `token_rewards_batch` provide matching batch rewards. Split the caller's effective workload into explicit updates when limiting activation memory.

Native PPO/GRPO and `SftTrainer` expose `sample_paged` over a `DeviceBackend` actor. Pass prompt token vectors, the positive per-request token budget, EOS IDs, `ContinuousBatchConfig` and `PagedKvCacheConfig`; the SFT method also takes `&mut PostTrainingState`. Each call captures current full weights or merges the current LoRA adapters into an inference snapshot, allocates the actual paged KV engine and continuously batches requests. It returns responses in prompt order, includes sampled EOS and distinguishes termination from truncation. Sampling uses the full vocabulary at temperature one without filtering or penalties. The inference snapshot and KV pools are released after the batch; subsequent calls capture subsequent actor updates.

`sample_paged` returns `PagedResponse { response, rotary_context, behavior_log_probs }`, leaving the original three-field `Response` contract unchanged. Paged generation records actual append ends, including prefill chunks and replayed prefixes, and selected-token log probabilities directly from engine logits. Use PPO `prepare_rollouts` or GRPO `prepare_paged_group`, then `step_paged`; reward batch APIs accept either response type. `PreferencePair`, `PpoRollout` and `GrpoRollout` have a default `Response` type parameter, preserving existing callers, and retain `PagedResponse` metadata for paged updates. Packed actor/reference/critic/reward forwards replay those rotary contexts; PPO/GRPO use captured behavior probabilities rather than rescoring rounded or updated weights. Ordinary runtime-RoPE responses retain actual full-prefix scoring. The request-seed cursor commits only after the whole batch succeeds: checkpoint before rollout to retry its request-local RNG streams, and store successful response/fixed-update buffers in `state.data` before checkpointing. In-flight KV is not checkpointed.

## Pairwise reward models

`RewardModel(CausalValueModel(backbone, scalar_head, ...), prefix_scoring=False)` and `RewardTrainer` train an independent reward model from the same `PreferenceCollator` pairs. The score is the scalar at the last non-padding response token, including EOS when present. The Bradley-Terry objective is the pair-mean `-log_sigmoid(score(chosen)-score(rejected))`; there is no LM loss, frozen reference, reward centering, clipping or normalization. Construct the optimizer after wrapping; full-parameter, NF4 and LoRA backbones follow the existing explicit finetuner contract. Dropout-free forwards retain activation recomputation. Set `prefix_scoring=True` for length-dependent backbones so each response is evaluated at its actual unpadded length.

```python
reward_model = RewardModel(CausalValueModel(reward_backbone, scalar_head))
reward_optimizer = torch.optim.AdamW([p for p in reward_model.parameters() if p.requires_grad], lr=1e-6)
reward_trainer = RewardTrainer(reward_model, reward_optimizer, base_id=reward_base_id, run_config=run_config)
reward_trainer.train_step([collator(preference_pairs)])
ppo_rollout = ppo_trainer.prepare_rollout(response_batch, reward_trainer.token_rewards(response_batch), terminal)
grpo_rollout = grpo_trainer.prepare_groups(response_batch, reward_trainer.score(response_batch), group_ids)
```

`score` returns detached CPU outcome rewards `[B]`; `token_rewards` places each score only on its final completion action in `[B,T-1]`, with zero prompt/padding/other-action entries. For packed responses, wrap `PackedCausalValueModel` instead: scores are `[D]` and sparse token rewards are target-aligned `[N]`, with each score at its document's final token. Reward scoring is explicitly requested and does not modify the reward model or automatically update the policy. `RewardTrainer` reuses pair-weighted accumulation, distributed reduction, objective evaluation, `data_state` cursors and atomic local/all-rank checkpoints, including backbone, scalar head, optimizer, scheduler/scaler and RNG.

Rust exports `RewardTrainer::new(actor, head_train_config)` over a full/LoRA `SftTrainer`. It creates a bias-free scalar head and trains both paths from the same backward pass; the backbone/adapter and scalar-head AdamW schedules and norm-clipping limits are independently configured. `step(&[PreferencePair])`, `score(&Response)` and `token_rewards(&Response)` use the same loss, pooling and explicit outcome placement. Pass the latter to PPO `prepare_rollout`, or the former scores to GRPO `prepare_group`. ruCCL normalization and head gradients use the actor's existing peer. `save_checkpoint`/`load_checkpoint` commit backbone/adapters, head, both optimizers, contracts and `state.data` together under the actor's retention policy; a frozen LoRA base must be reloaded from the same source.
