# Model Loading and Inference

[Documentation](README.md) · [Training](training.md) · [中文](../zh/model-inference.md) | [日本語](../ja/model-inference.md) | [Deutsch](../de/model-inference.md) | [Русский](../ru/model-inference.md)

ruLLM provides model loading, tokenization, cached autoregressive generation, and request scheduling. Its Cargo package is `ruda-llm`; its Rust import name is `rullm`.

## Prepare a local model

Keep the model files in one directory:

| File | Purpose |
| --- | --- |
| `config.json` | Model architecture and parameters |
| `model.safetensors` | Single-file weights |
| `model.safetensors.index.json` and every listed shard | Sharded weights, instead of a single weight file |
| `tokenizer.json` | Text tokenization and decoding |
| `tokenizer_config.json` | ChatML template required by Qwen2/Qwen2.5 pipelines |
| `generation_config.json` | Optional Qwen2/Qwen2.5 generation configuration supplying EOS settings |
| `preprocessor_config.json` | Qwen3.5 image preprocessing configuration, required for image input |

Loaders accept local directories and do not download models. Select the loader for your architecture:

| Model/input | Loader |
| --- | --- |
| Unquantized Llama text | `load_huggingface_llama_pipeline::<B>` |
| Unquantized Qwen2/Qwen2.5 text | `load_huggingface_qwen2_pipeline::<B>` |
| AWQ GEMM INT4 Qwen2/Qwen2.5 | `load_huggingface_awq_qwen2_pipeline::<R>` |
| Unquantized Qwen3.5 text | `load_huggingface_qwen35_text::<B>` |
| Qwen3.5 vision encoder | `load_huggingface_qwen35_vision::<B>` |
| Qwen3.5 images and text | `load_huggingface_qwen35_multimodal::<B>` |

`B` is a tensor Backend; `R` is a device Runtime. The Qwen3.5 text loader uses a `qwen3_5_text` configuration with per-layer `layer_types` and rejects quantization configurations. Do not pass AWQ files to a floating-point loader.

## Run the text generation examples

After [setting up the environment](getting-started.md), run these commands from the source root. The relative paths point to two local model directories:

```powershell
$env:RUDA_CUDA_COMPILER = 'nvrtc'
cargo run --locked -p ruda-llm --features nvidia --example qwen2_generate -- ./models/qwen2 "Hello" 32 1
```

The Qwen3.5 example defaults to Ruda IR → PTX when `RUDA_CUDA_COMPILER` is unset. Use `--release` for optimized execution. Unsupported direct PTX operations return errors rather than falling back to NVRTC. The PTX version must match the target GPU and driver; see the [PTX backend reference](ptx.md).

```powershell
Remove-Item Env:RUDA_CUDA_COMPILER -ErrorAction SilentlyContinue
$env:RUDA_PTX_VERSION = '8.0'
cargo run --release --locked -p ruda-llm --features nvidia-ptx --example qwen35_generate -- ./models/qwen35 "Hello" 8 1
```

The CUDA C++ / NVRTC path remains available with an explicit selection:

```powershell
$env:RUDA_CUDA_COMPILER = 'nvrtc'
cargo run --release --locked -p ruda-llm --features nvidia-ptx --example qwen35_generate -- ./models/qwen35 "Hello" 8 1
```

Arguments are the model directory, raw text prompt, maximum new tokens, and run count. The last two default to `32` and `1` for Qwen2, or `8` and `1` for Qwen3.5. Run count must be positive. Examples print JSON lines for loading and generation. Generation lines include `text`, `generated_token_ids`, and `stopped_on_eos`.

These examples encode the prompt directly without adding a chat template. For conversational models, prepare the prompt with `render_chat` as below.

## Generate a chat reply in an application

For an application directory alongside the `RUDA` source directory, use these dependencies. 

```toml
[dependencies]
rullm = { package = "ruda-llm", path = "../RUDA/ruLLM", features = ["nvidia"] }
ruda-tensor-device = { path = "../RUDA/ruda-tensor-device", default-features = false, features = ["std", "cuda"] }
ruda-driver-cuda = { path = "../RUDA/ruda-driver-cuda", default-features = false, features = ["std"] }
half = "=2.7.1"

```

This complete `src/main.rs` accepts a model directory and user message as command-line arguments. Later function examples reuse its `B` alias, imports, and dependencies:

```rust
use half::bf16;
use ruda_tensor_device::cuda::{Cuda, CudaDevice};
use rullm::{
    GreedyGenerationConfig, Qwen2ChatMessage, Qwen2ChatRole,
    load_huggingface_qwen2_pipeline,
};
use std::{error::Error, io};

type B = Cuda<bf16, i32>;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let directory = args.next().ok_or_else(|| io::Error::other("model directory required"))?;
    let prompt = args.next().ok_or_else(|| io::Error::other("prompt required"))?;
    let device = CudaDevice::default();
    let pipeline = load_huggingface_qwen2_pipeline::<B>(&directory, &device)?;
    let messages = [Qwen2ChatMessage::new(Qwen2ChatRole::User, prompt)];
    let prompt = pipeline.render_chat(&messages, true)?;
    let output = pipeline.generate_text(
        &prompt,
        GreedyGenerationConfig {
            max_new_tokens: 32,
            eos_token_ids: pipeline.loaded.default_eos_token_ids.clone(),
        },
        false,
        true,
        &device,
    )?;
    println!("{}", output.generated_text);
    Ok(())
}
```

`render_chat` handles plain-text Qwen2/Qwen2.5 ChatML messages. Passing `true` adds the assistant generation prefix. Roles also include `System` and `Assistant`; supply messages in conversation order. This API does not accept tool-call messages.

The two boolean arguments to `generate_text` are `add_special_tokens` and `skip_special_tokens`. Do not add special tokens again after rendering ChatML; decoding can skip them. `generated_text` contains only the new reply, while `text` includes both prompt and reply.

## Temperature, top-k, and top-p sampling

Pass the rendered prompt to `generate_text_sampled`:

```rust
use rullm::{
    GeneratedText, HuggingFaceLoadError, HuggingFaceQwen2Pipeline,
    SamplingConfig, SamplingGenerationConfig,
};

fn sample_text(
    pipeline: &HuggingFaceQwen2Pipeline<B>,
    prompt: &str,
    device: &CudaDevice,
) -> Result<GeneratedText, HuggingFaceLoadError> {
    pipeline.generate_text_sampled(
        prompt,
        SamplingGenerationConfig {
            max_new_tokens: 32,
            eos_token_ids: pipeline.loaded.default_eos_token_ids.clone(),
            sampling: SamplingConfig {
                temperature: 0.8,
                top_k: 40,
                top_p: 0.9,
                seed: Some(42),
            },
        },
        false,
        true,
        device,
    )
}
```

| Parameter | Values and effect |
| --- | --- |
| `temperature` | Finite and positive; scales logits first |
| `top_k` | Keeps high-scoring tokens next; 0 disables filtering, ties at the boundary are retained |
| `top_p` | Finally filters by cumulative probability; range (0, 1], with 1 disabling filtering |
| `seed` | Request-local sampling seed; `None` uses system entropy |
| `max_new_tokens` | Limits new tokens, excluding the prompt |
| `eos_token_ids` | Stops after generating any listed token |

Use `generate_text` for maximum-logit decoding rather than setting temperature to zero. A fixed seed fixes the sampling random sequence, not numerical consistency across models or backends.

### Optional min-p and history penalties

`SamplingConfig` keeps its existing four fields. Pass a separate `SamplingOptions` to `TokenSampler::new_with_options`, `ContinuousBatchScheduler::submit_sampled_with_options`, or `ContinuousBatchRunner::submit_sampled_with_options`. The model-independent streaming entry point is `generate_causal_sampled_stream_with_options`; the existing entry points use default options.

| Option | Default | Meaning |
| --- | --- | --- |
| `min_p` | `0.0` | Range [0, 1]; after top-k/top-p, discard tokens below this fraction of the maximum token probability |
| `repetition_penalty` | `1.0` | Finite and positive; divide positive logits and multiply negative logits for tokens seen in the prompt or generated output |
| `presence_penalty` | `0.0` | Range [-2, 2]; subtract once for a token present in generated output |
| `frequency_penalty` | `0.0` | Range [-2, 2]; subtract per occurrence in generated output |

Penalties are applied before temperature and filtering. When using `TokenSampler` directly, call `sample_with_history(logits, prompt_ids, generated_ids)` to supply the history; `sample(logits)` has empty history. `to_state` / `from_state` preserve these options and the exact sampler RNG position; older records without options use their defaults.

## Load AWQ weights

The AWQ loader takes `CudaRuntime`, not `Cuda<bf16, i32>`, as its generic parameter:

```rust
use ruda_driver_cuda::CudaRuntime;
use rullm::{HuggingFaceAwqQwen2Pipeline, load_huggingface_awq_qwen2_pipeline};

fn load_awq(
    directory: &str,
    device: &CudaDevice,
) -> Result<HuggingFaceAwqQwen2Pipeline<CudaRuntime>, HuggingFaceLoadError> {
    load_huggingface_awq_qwen2_pipeline::<CudaRuntime>(directory, device)
}
```

The returned pipeline also provides `render_chat`, `encode`, `decode`, `generate_text`, and `generate_text_sampled`. AWQ configuration requires `quant_method = "awq"`, `bits = 4`, `version = "gemm"`, and `zero_point = true`. `group_size` must be positive or `-1`. Linear projections use F16 activations. See [ruBLAS INT4](libraries/rublas.md) for packed weight layouts.

## Qwen3.5 image input

`generate_image_files_greedy` takes token IDs for one prompt, JPEG/PNG paths in prompt order, an image processor, and generation options:

```rust
use rullm::{
    Qwen35ImageProcessor, TokenGenerationOutput,
    load_huggingface_qwen35_multimodal,
};

fn generate_from_images(
    directory: &str,
    prompt_ids: &[i32],
    paths: &[&str],
    device: &CudaDevice,
) -> Result<TokenGenerationOutput, Box<dyn Error>> {
    let model = load_huggingface_qwen35_multimodal::<B>(directory, device)?;
    let processor = Qwen35ImageProcessor::from_huggingface(directory)?;
    let output = model.generate_image_files_greedy(
        prompt_ids,
        paths,
        &processor,
        &GreedyGenerationConfig {
            max_new_tokens: 32,
            eos_token_ids: vec![model.text.config.eos_token_id],
        },
    )?;
    Ok(output)
}
```

Encode `prompt_ids` with the model's `tokenizer.json`. Follow the model's prompt template, keeping one placeholder token per image using `image_token_id` from `config.json`. Image paths must match placeholder count and order. This entry point expands placeholders internally; do not expand them beforehand. It handles still images and rejects video tokens.

For existing RGB8 data, use `Qwen35RgbImage { pixels, height, width }` with `generate_rgb_greedy`. Pixels are interleaved RGB bytes without row padding. For separate preprocessing, `preprocess_files` and `preprocess_rgb` return FP32 `patches`, a two-dimensional `shape`, and per-image `grids`.

When serving multiple requests, load the model and processor outside the request loop. Decode the returned `generated_token_ids` with the same tokenizer.

## Caches and continuous batching

Single-request generation functions create and maintain a cache for that generation. To drive Qwen3.5 manually, call `model.new_cache()`, pass the prompt tensor to `forward_cached_last(tokens, &mut cache)`, then pass only new tokens. Do not resubmit the entire cached prefix.

To integrate your own batch executor, use `ContinuousBatchScheduler`:

1. Create it with `ContinuousBatchScheduler::new(config, kv_config)`. `ContinuousBatchConfig` sets `max_active_sequences` and `max_batch_tokens`. `PagedKvCacheConfig` sets `block_size`, `num_pages`, and `max_sequence_length`. All must be positive.
2. Submit requests with `submit(prompt_token_ids, generation)` and retain each returned `RequestId`.
3. Call `schedule()` for a `ScheduledBatch`. Its `kind` distinguishes Prefill/Decode, or Verify when speculative decoding is enabled; `sequences` supplies each row's tokens, start position, context length, and block table.
4. For Prefill/Decode, run the model in row order, write the corresponding KV pages, and select each row's next token. On success, call `complete_batch(batch.id, &generated_token_ids)` with one result token per row. On failure, call `fail_batch(batch.id)` to cancel that batch's page reservations. Verify batches use `complete_verify_batch`, as described below.
5. Retrieve completed requests and generated tokens with `pop_finished()`, then continue scheduling.

The scheduler manages requests and page-allocation metadata. The caller's executor owns device KV storage and model execution. Use `token_matrix()`, `context_lengths()`, and `flattened_block_table()` to construct batch inputs. Complete or cancel an outstanding batch before calling `schedule()` again.

API reference: [ruLLM exports](../../ruLLM/src/lib.rs), [Generation options](../../ruLLM/src/generation.rs), [Scheduler](../../ruLLM/src/continuous_batch.rs).

## Paged Llama/Qwen2 execution

`PagedLlamaEngine` owns per-layer physical K/V pools; `ContinuousBatchRunner` combines that engine with request scheduling and token selection. Load a floating-point Llama/Qwen2 model and call `into_packed_inference()` before constructing the runner. This path accepts F32/F16/BF16 models with **static, full-head RoPE**; dynamic or partial RoPE is rejected. It is not the AWQ loader or the Qwen3.5 batch-cache path below.

Run the NVIDIA BF16 example from the repository root:

```powershell
$env:RUDA_CUDA_COMPILER = 'nvrtc'
cargo run --release --locked -p ruda-llm --features nvidia --example qwen2_continuous_batch -- ./models/qwen2 32 "Hello" "Explain KV caching"
```

Arguments are the local model directory, positive maximum new-token count, and one or more raw-text prompts. The example also reruns each request through dense single-sequence generation and reports token comparisons; its total runtime is not a batching-only throughput measurement. BF16 batch and single-sequence kernels can differ near tied logits, so successful execution does not promise identical token IDs.

| API/configuration | Behavior |
| --- | --- |
| `ContinuousBatchRunner::new(&model, batching, kv, &device)` | Allocates the pools and enables chunked prefill and recompute preemption |
| `ContinuousBatchConfig` | `max_active_sequences` bounds active requests; `max_batch_tokens` bounds a device batch and the prefill chunk |
| `PagedKvCacheConfig` | `block_size` is tokens per page; `num_pages` is the shared pool capacity; `max_sequence_length` is the per-request limit |
| `submit` / `submit_sampled_with_options` | Submit greedy or sampled token-ID requests; retain the returned `RequestId` |
| `step(&model, &device)` / `pop_finished()` | Execute one batch and retrieve finished requests; `run_until_idle` drives the loop |
| `set_prefix_caching(true)` | Reuse matching full-block prefixes; disabled by default on a standalone runner |
| `set_speculative_decoding(Some(SpeculativeConfig { draft_tokens: 3, ngram: 3 }))` | Enable prompt-lookup drafts for greedy requests; sampled requests continue normal decoding; `None` disables it |

Pools are allocated for the configured capacity, not only the tokens currently in use. `runner.engine().pool_bytes()` reports their payload bytes; model weights and execution workspaces are additional. Recompute preemption releases a newer request's KV pages and later prefills its prompt plus generated history again; it trades extra computation for pool availability, not host offload.

For a custom executor, `ContinuousBatchScheduler::new` retains the non-preempting defaults. Opt in with `new_paged`; inspect `samples_token(&row)` **before completing the batch** to distinguish a final prefill chunk from a chunk whose output token is ignored. Verify batches need logits at every draft position and `complete_verify_batch`, not `complete_batch`. `snapshot()` retains the existing snapshot fields; `paged_snapshot()` / `runner.snapshot()` return `PagedBatchSnapshot` with preemption, prefix-token and draft-acceptance counters. `PagedKvCacheManager::cached_page_count()` reports unreferenced reusable prefix pages.

The dense packed path also uses tiled attention for long prefills and windowed attention. Floating-point loaders read Mistral `sliding_window` from layer zero; Qwen2 enables it only with `use_sliding_window`, starting at `max_window_layers` (defaulting to the layer count). This is configuration-driven, not a server flag.

## OpenAI-compatible HTTP server

The [NVIDIA example](../../ruLLM/examples/qwen2_openai_server.rs) uses `serve_openai` and the paged runner. It needs the same local floating-point Qwen2/Qwen2.5 files as generation, uses BF16, and does not require `--features serving`:

```powershell
$env:RUDA_CUDA_COMPILER = 'nvrtc'
cargo run --release --locked -p ruda-llm --features nvidia --example qwen2_openai_server -- ./models/qwen2 --address 127.0.0.1:8000 --kv-gib 1 --max-context 2048
```

| CLI option | Default | Purpose |
| --- | --- | --- |
| `--address` | `127.0.0.1:8000` | Bind address |
| `--api-key` | Unset | Require an `Authorization: Bearer` token |
| `--kv-gib` / `--block-size` | `1.0` / `16` | KV payload budget in GiB and tokens per page; excludes weights/workspaces |
| `--max-context` | Model limit, capped by pool capacity | Per-request context limit |
| `--max-active` / `--max-batch-tokens` | `16` / `512` | Active-request and batch/prefill-chunk limits |
| `--no-prefix-cache` | Not set | Disable the server's default prefix caching |
| `--speculative` | `0` | Draft-token count for greedy prompt-lookup verification; n-gram length is 3 |

Endpoints are `GET /health`, `GET /v1/models`, `POST /v1/completions`, and `POST /v1/chat/completions`. A Bash/curl streaming request is:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"temperature":0.8,"top_p":0.9,"min_p":0.05,"stream":true}'
```

`temperature: 0` selects greedy decoding; omitted temperature defaults to 1. Sampled requests accept `top_k`, `top_p`, `seed`, and the four `SamplingOptions` fields. `max_tokens` defaults to 256; chat also accepts `max_completion_tokens`, which takes precedence. `stop` accepts a string or up to four strings. `stream: true` returns SSE ending with `[DONE]`; false returns a JSON response with usage counts.

This is a text-only HTTP/1.1 subset, not the full OpenAI API: one completion (`n = 1`), text `system`/`user`/`assistant` messages, no tool execution or image inference. Networking uses the standard library, has no TLS, and is unauthenticated unless an API key is configured. `OpenAiServerConfig` exposes the default-token limit and request-body limit (1 MiB by default) for library callers. `model_name` controls the single model advertised by the server.

## Qwen3.5 fused paged attention

`Qwen35BatchCache` selects its attention path when `new_batch_cache` creates it. Set `RUDA_PAGED_ATTENTION=fused` before cache creation to use ruDNN's fused paged attention in Qwen3.5 continuous batching. Unset or `legacy` retains the existing path; other values are rejected. Changing the variable does not switch an existing cache, and it is not a global switch for every generation entry point.

The fused path keeps a physical KV arena per full-attention layer and uses the scheduler's page tables for prefill/decode without gathering historical KV into padded tensors. Arena dimensions use `num_pages`, `block_size`, KV heads and head dimension; reserve GPU memory for that capacity. Inputs follow ruDNN's contiguous F32/F16/BF16 contract. Shared-prefix writes require scheduler-level copy-on-write. After a failed device append, rebuild the cache rather than continuing from its previous position.

## Reusable attention and MoE components

`rullm::gpu_inference` exposes `TensorOps`, `gqa_scores`, `gqa_value_product`, `mla_absorb_query`, `mla_scores` and `mla_value_product`. These compose caller-supplied device tensor operations without repeating cached GQA heads or expanding historical MLA latent values. The caller supplies masking, softmax, positional encoding and model-specific scaling.

`rullm::device_inference` re-exports ruDNN's `HostPlan`, `DevicePlan`, `PagedAttentionError`, `SwiGluExperts`, `RoutingOptions`, `GroupRoutingOptions`, `route_sigmoid_grouped` and `GroupedStrategy`. These are building blocks, not a complete MLA/MoE model loader; adapters still provide projections, routing configuration and cache ownership.
