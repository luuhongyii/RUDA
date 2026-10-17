# 模型加载与推理

[文档首页](README.md) · [训练](training.md) · [English](../en/model-inference.md) | [日本語](../ja/model-inference.md) | [Deutsch](../de/model-inference.md) | [Русский](../ru/model-inference.md)

ruLLM 提供模型加载、分词、带缓存的自回归生成和请求调度。Cargo package 名为 `ruda-llm`，Rust 导入名为 `rullm`。

## 准备本地模型

将模型文件保存在同一目录：

| 文件 | 用途 |
| --- | --- |
| `config.json` | 模型结构和参数 |
| `model.safetensors` | 单文件权重 |
| `model.safetensors.index.json` 及其中列出的全部分片 | 分片权重，替代单文件权重 |
| `tokenizer.json` | 文本分词与解码 |
| `tokenizer_config.json` | Qwen2／Qwen2.5 pipeline 要求的 ChatML 模板 |
| `generation_config.json` | Qwen2／Qwen2.5 可选生成配置，读取 EOS 设置 |
| `preprocessor_config.json` | Qwen3.5 图片预处理配置，图片输入时需要 |

加载函数接收本地目录，不自动下载模型。按模型结构选择入口：

| 模型／输入 | 加载入口 |
| --- | --- |
| 非量化 Llama 文本 | `load_huggingface_llama_pipeline::<B>` |
| 非量化 Qwen2／Qwen2.5 文本 | `load_huggingface_qwen2_pipeline::<B>` |
| AWQ GEMM INT4 Qwen2／Qwen2.5 | `load_huggingface_awq_qwen2_pipeline::<R>` |
| 非量化 Qwen3.5 文本 | `load_huggingface_qwen35_text::<B>` |
| Qwen3.5 视觉编码器 | `load_huggingface_qwen35_vision::<B>` |
| Qwen3.5 图片与文本 | `load_huggingface_qwen35_multimodal::<B>` |

`B` 是张量 Backend，`R` 是设备 Runtime。Qwen3.5 文本入口使用 `qwen3_5_text` 配置及逐层 `layer_types`，不接收量化配置；不要把 AWQ 文件传给浮点加载入口。

## 运行文本生成示例

完成[环境配置](getting-started.md)后，在源码根目录运行。下面的相对路径分别指向两份本地模型目录：

```powershell
$env:RUDA_CUDA_COMPILER = 'nvrtc'
cargo run --locked -p ruda-llm --features nvidia --example qwen2_generate -- ./models/qwen2 "Hello" 32 1
```

Qwen3.5 示例在未设置 `RUDA_CUDA_COMPILER` 时默认使用 Ruda IR → PTX。使用 `--release` 启用优化构建。直接 PTX 不支持的操作会报错，不回退到 NVRTC。PTX 版本需与目标 GPU 和驱动匹配，见 [PTX 后端参考](ptx.md)。

```powershell
Remove-Item Env:RUDA_CUDA_COMPILER -ErrorAction SilentlyContinue
$env:RUDA_PTX_VERSION = '8.0'
cargo run --release --locked -p ruda-llm --features nvidia-ptx --example qwen35_generate -- ./models/qwen35 "Hello" 8 1
```

保留 CUDA C++ / NVRTC 路径，可显式选择：

```powershell
$env:RUDA_CUDA_COMPILER = 'nvrtc'
cargo run --release --locked -p ruda-llm --features nvidia-ptx --example qwen35_generate -- ./models/qwen35 "Hello" 8 1
```

参数依次为模型目录、原始文本提示、最大新增 token 数、运行次数。Qwen2 示例后两个参数默认为 `32`、`1`；Qwen3.5 默认为 `8`、`1`，运行次数必须大于零。示例输出加载信息和生成结果的 JSON 行，生成行包含 `text`、`generated_token_ids`、`stopped_on_eos`。

这两个示例直接编码提示文本，不自动添加聊天模板。对话模型使用下面的 `render_chat` 调用准备提示。

## 在应用中生成对话回复

应用目录与 `RUDA` 源码目录同级时，依赖配置如下。

```toml
[dependencies]
rullm = { package = "ruda-llm", path = "../RUDA/ruLLM", features = ["nvidia"] }
ruda-tensor-device = { path = "../RUDA/ruda-tensor-device", default-features = false, features = ["std", "cuda"] }
ruda-driver-cuda = { path = "../RUDA/ruda-driver-cuda", default-features = false, features = ["std"] }
half = "=2.7.1"

```

下面是完整的 `src/main.rs`，命令行接收模型目录和用户消息。后续函数示例复用这里的 `B`、导入及依赖：

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

`render_chat` 处理 Qwen2／Qwen2.5 的纯文本 ChatML 消息，第二个参数 `true` 添加 assistant 生成前缀。角色还可以是 `System`、`Assistant`；消息按会话顺序传入。该入口不接收工具调用消息。

`generate_text` 的两个布尔参数依次是 `add_special_tokens` 和 `skip_special_tokens`。已经渲染 ChatML 时不重复添加特殊 token；解码时可跳过特殊 token。`generated_text` 只含新增回复，`text` 包含提示及回复。

## 温度、top-k 与 top-p 采样

将渲染后的提示交给 `generate_text_sampled`：

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

| 参数 | 取值与效果 |
| --- | --- |
| `temperature` | 有限且大于 0；先缩放 logits |
| `top_k` | 再保留高分 token；0 关闭，边界同分 token 一并保留 |
| `top_p` | 最后按累计概率筛选；范围为 (0, 1]，1 关闭 |
| `seed` | 当前请求的采样随机种子；`None` 使用系统熵 |
| `max_new_tokens` | 只限制新增 token 数，不包含提示 |
| `eos_token_ids` | 生成任一指定 token 后停止 |

需要最大 logit 解码时使用 `generate_text`，而不是把温度设为零。固定 seed 固定的是采样随机序列，不代替模型及后端的数值一致性。

### 可选 min-p 与历史惩罚

`SamplingConfig` 保持原有四个字段。额外的 `SamplingOptions` 传给 `TokenSampler::new_with_options`、`ContinuousBatchScheduler::submit_sampled_with_options` 或 `ContinuousBatchRunner::submit_sampled_with_options`。模型无关的流式入口为 `generate_causal_sampled_stream_with_options`；原有入口使用默认选项。

| 选项 | 默认值 | 含义 |
| --- | --- | --- |
| `min_p` | `0.0` | 范围 [0, 1]；top-k／top-p 后，剔除概率低于最大 token 概率这一比例的 token |
| `repetition_penalty` | `1.0` | 有限且为正；对提示或已生成内容中出现过的 token，正 logits 除以该值，负 logits 乘以该值 |
| `presence_penalty` | `0.0` | 范围 [-2, 2]；对已生成内容中出现过的 token 减去一次该值 |
| `frequency_penalty` | `0.0` | 范围 [-2, 2]；按 token 在已生成内容中的出现次数减去该值 |

惩罚先于温度和筛选应用。直接使用 `TokenSampler` 时，通过 `sample_with_history(logits, prompt_ids, generated_ids)` 提供历史；`sample(logits)` 的历史为空。`to_state`／`from_state` 保存选项及采样 RNG 的准确位置；没有选项字段的旧记录使用默认值。

## 加载 AWQ 权重

AWQ 入口的泛型是 `CudaRuntime`，不是 `Cuda<bf16, i32>`：

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

返回的 pipeline 同样提供 `render_chat`、`encode`、`decode`、`generate_text` 和 `generate_text_sampled`。AWQ 配置要求 `quant_method = "awq"`、`bits = 4`、`version = "gemm"`、`zero_point = true`；`group_size` 为正数或 `-1`。线性投影使用 F16 激活，打包权重布局见 [ruBLAS INT4](libraries/rublas.md)。

## Qwen3.5 图片输入

`generate_image_files_greedy` 接收一条提示的 token ID、按提示顺序排列的 JPEG／PNG 路径、图片处理器和生成配置：

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

使用该模型的 `tokenizer.json` 编码 `prompt_ids`，并按模型提示模板为每张图片保留一个 `config.json` 中的 `image_token_id` 占位 token。图片路径数量和顺序必须与占位 token 一致。此入口内部完成占位展开，不要事先重复展开；它处理静态图片，不接收视频 token。

已经持有 RGB8 数据时，用 `Qwen35RgbImage { pixels, height, width }` 和 `generate_rgb_greedy`。像素为无行填充、交错存储的 RGB 字节。需要单独预处理时，`preprocess_files`／`preprocess_rgb` 返回 FP32 `patches`、二维 `shape` 和各图片的 `grids`。

同一模型处理多个请求时，将模型和处理器的加载移到请求循环外；生成函数的输出 `generated_token_ids` 可以交回相同 tokenizer 解码。

## 缓存与连续批处理

单请求生成入口自动创建并维护本次生成的缓存。手动驱动 Qwen3.5 时，调用 `model.new_cache()`，将提示张量传给 `forward_cached_last(tokens, &mut cache)`，之后只传新增 token；不要把已经缓存的整个前缀再次提交。

自行接入批量执行器时，使用 `ContinuousBatchScheduler`：

1. 用 `ContinuousBatchScheduler::new(config, kv_config)` 创建调度器。`ContinuousBatchConfig` 设置 `max_active_sequences` 和 `max_batch_tokens`；`PagedKvCacheConfig` 设置 `block_size`、`num_pages` 和 `max_sequence_length`，这些值都必须大于零。
2. 用 `submit(prompt_token_ids, generation)` 提交请求并保留返回的 `RequestId`。
3. 调用 `schedule()` 获取可执行的 `ScheduledBatch`。`kind` 区分 Prefill／Decode，启用推测解码时还会出现 Verify；`sequences` 给出每行的 token、起始位置、上下文长度和页表。
4. 对 Prefill／Decode 批次，执行器按行顺序运行模型、写入对应 KV 页，并选出每行的下一个 token。成功后调用 `complete_batch(batch.id, &generated_token_ids)`，每行必须对应一个结果 token。失败时调用 `fail_batch(batch.id)` 撤销该批的页预留。Verify 批次使用下文的 `complete_verify_batch`。
5. 用 `pop_finished()` 取回完成请求及生成 token；然后继续调度。

调度器管理请求和页分配元数据；设备 KV 存储与模型执行由调用方的执行器负责。`token_matrix()`、`context_lengths()`、`flattened_block_table()` 可用于构造批量输入。存在未完成批次时，先完成或撤销该批次，再调用 `schedule()`。

接口参考：[ruLLM 导出](../../ruLLM/src/lib.rs)、[生成配置](../../ruLLM/src/generation.rs)、[调度器](../../ruLLM/src/continuous_batch.rs)。

## Llama／Qwen2 分页执行

`PagedLlamaEngine` 持有逐层物理 K／V 池，`ContinuousBatchRunner` 将其与请求调度、token 选择连接起来。加载浮点 Llama／Qwen2 后，先调用 `into_packed_inference()`，再构造 runner。此路径支持 F32／F16／BF16、完整或部分旋转维度，以及动态 NTK 和 long-RoPE；逐 token 旋转上下文随共享前缀页、抢占重算和投机接受的输入保留。直接使用 engine 时，调度器提交后须调用 `commit_batch(batch, accepted_input_counts)`，失败后调用 `fail_batch(batch.id)`；runner 自动完成这些操作。它不是 AWQ 加载器，也不同于下文的 Qwen3.5 batch-cache 路径。

在仓库根目录运行 NVIDIA BF16 示例：

```powershell
$env:RUDA_CUDA_COMPILER = 'nvrtc'
cargo run --release --locked -p ruda-llm --features nvidia --example qwen2_continuous_batch -- ./models/qwen2 32 "Hello" "Explain KV caching"
```

参数依次为本地模型目录、正数的最大新增 token 数、一个或多个原始文本提示。示例还会逐请求重跑稠密单序列生成并报告 token 比较，因此总耗时不等于纯批处理吞吐测试。BF16 批量与单序列内核可能在 logits 近似同分时产生不同 token，命令成功不代表 token ID 完全一致。

| 接口／配置 | 行为 |
| --- | --- |
| `ContinuousBatchRunner::new(&model, batching, kv, &device)` | 分配 KV 池，启用分块预填充和重算抢占 |
| `ContinuousBatchConfig` | `max_active_sequences` 限制活跃请求；`max_batch_tokens` 限制设备批次及预填充分块 |
| `PagedKvCacheConfig` | `block_size` 为每页 token 数；`num_pages` 为共享池页数；`max_sequence_length` 为单请求长度上限 |
| `submit`／`submit_sampled_with_options` | 提交贪心或采样 token-ID 请求，保留返回的 `RequestId` |
| `step(&model, &device)`／`pop_finished()` | 执行一批并取回完成请求；`run_until_idle` 自动驱动循环 |
| `set_prefix_caching(true)` | 复用匹配的完整块前缀；独立 runner 默认关闭 |
| `set_speculative_decoding(Some(SpeculativeConfig { draft_tokens: 3, ngram: 3 }))` | 对贪心请求启用提示查找草稿；采样请求仍正常解码；传 `None` 关闭 |

KV 池按配置容量分配，不是仅为当前 token 分配。`runner.engine().pool_bytes()` 返回池的有效载荷字节数，模型权重和执行工作区另计。重算抢占释放较新请求的 KV 页，之后重新预填充其提示和已生成历史；它以额外计算换取池空间，不是向主机卸载。

自行接入执行器时，`ContinuousBatchScheduler::new` 保留不抢占的默认行为；用 `new_paged` 显式启用新行为。在**完成批次之前**查询 `samples_token(&row)`，区分最终预填充分块与忽略输出 token 的中间分块。Verify 批次需要每个草稿位置的 logits，并通过 `complete_verify_batch` 提交，而非 `complete_batch`。`snapshot()` 保留旧字段；`paged_snapshot()`／`runner.snapshot()` 返回带抢占、前缀 token、草稿接受计数的 `PagedBatchSnapshot`。`PagedKvCacheManager::cached_page_count()` 返回无活跃引用、可复用的前缀页数。

稠密 packed 路径也为长提示预填充和滑动窗口采用分块注意力。浮点加载器（包括 TP rank 加载器）对 Mistral 从第 0 层应用 `sliding_window`；Qwen2 仅在 `use_sliding_window` 启用时，从 `max_window_layers` 指定层开始应用，未指定时该值为总层数。这由模型配置控制，不是服务启动参数。自行执行分页批次时，请求完成或取消后调用 `engine.forget_request(request_id)` 释放其历史；已提交的共享前缀页元数据仍可复用。

## OpenAI 风格 HTTP 服务

[NVIDIA 示例](../../ruLLM/examples/qwen2_openai_server.rs) 使用 `serve_openai` 和分页 runner。准备与生成相同的本地浮点 Qwen2／Qwen2.5 文件，示例使用 BF16，不要求 `--features serving`：

```powershell
$env:RUDA_CUDA_COMPILER = 'nvrtc'
cargo run --release --locked -p ruda-llm --features nvidia --example qwen2_openai_server -- ./models/qwen2 --address 127.0.0.1:8000 --kv-gib 1 --max-context 2048
```

| 命令行选项 | 默认值 | 用途 |
| --- | --- | --- |
| `--address` | `127.0.0.1:8000` | 监听地址 |
| `--api-key` | 不设置 | 要求 `Authorization: Bearer` token |
| `--kv-gib`／`--block-size` | `1.0`／`16` | KV 有效载荷预算（GiB）和每页 token 数，不含权重／工作区 |
| `--max-context` | 模型上限，并受池容量约束 | 单请求上下文上限 |
| `--max-active`／`--max-batch-tokens` | `16`／`512` | 活跃请求和批次／预填充分块限制 |
| `--no-prefix-cache` | 不设置 | 关闭服务默认启用的前缀缓存 |
| `--speculative` | `0` | 贪心提示查找验证的草稿 token 数，n-gram 长度固定为 3 |

接口为 `GET /health`、`GET /v1/models`、`POST /v1/completions`、`POST /v1/chat/completions`。Bash／curl 流式请求示例：

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"temperature":0.8,"top_p":0.9,"min_p":0.05,"stream":true}'
```

`temperature: 0` 选择贪心解码，省略时默认 1。采样请求接受 `top_k`、`top_p`、`seed` 及四个 `SamplingOptions` 字段。`max_tokens` 默认 256；对话接口还接受优先级更高的 `max_completion_tokens`。`stop` 为一个字符串或最多四个字符串；`stream: true` 返回以 `[DONE]` 结束的 SSE，否则返回包含 usage 计数的 JSON。

这是纯文本 HTTP/1.1 子集，不是完整 OpenAI API：单个补全（`n = 1`），文本 `system`／`user`／`assistant` 消息，不执行工具或图片推理。网络层使用标准库，不提供 TLS；未配置 API key 时不认证。库调用方可通过 `OpenAiServerConfig` 设置默认 token 上限及请求体上限，后者默认 1 MiB；`model_name` 控制服务展示的单个模型名称。

## Qwen3.5 融合分页注意力

`Qwen35BatchCache` 在 `new_batch_cache` 创建时选择注意力路径。创建缓存前设置 `RUDA_PAGED_ATTENTION=fused`，即可在 Qwen3.5 连续批处理中使用 ruDNN 融合分页注意力。未设置或设为 `legacy` 时保留原路径；其他值会被拒绝。修改变量不会切换已有缓存，也不是所有生成入口的全局开关。

融合路径为每个全注意力层保存物理 KV arena，预填充与解码使用调度器页表，不将历史 KV 收集为带 padding 的张量。arena 按 `num_pages`、`block_size`、KV 头数和头维度分配，应为该容量预留显存。输入遵循 ruDNN 的连续 F32/F16/BF16 契约。共享前缀写入需要调度器执行写时复制。设备追加失败后，应重建缓存，不能继续沿用先前位置。

## 可复用的注意力与 MoE 组件

`rullm::gpu_inference` 提供 `TensorOps`、`gqa_scores`、`gqa_value_product`、`mla_absorb_query`、`mla_scores` 和 `mla_value_product`。这些接口组合调用方提供的设备张量操作，不重复缓存的 GQA 头，也不展开历史 MLA 潜在值。掩码、softmax、位置编码与模型特定缩放由调用方提供。

`rullm::device_inference` 重导出 ruDNN 的 `HostPlan`、`DevicePlan`、`PagedAttentionError`、`SwiGluExperts`、`RoutingOptions`、`GroupRoutingOptions`、`route_sigmoid_grouped` 和 `GroupedStrategy`。这些是计算组件，不是完整的 MLA/MoE 模型加载器；适配器仍需提供投影、路由配置与缓存所有权。
