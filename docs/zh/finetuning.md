# 通用 LoRA 与 NF4 微调

[文档目录](README.md) · [训练指南](training.md) · [English](../en/finetuning.md)

## 选择接入层

| 模型接口 | 入口 |
| --- | --- |
| Rust `ruda-model` 模型 | `ruda_nn::LoRALinearConfig`，沿用张量后端与优化器 |
| Rust packed Llama／Qwen2 | `rullm::SftTrainer`，全参数 SFT 或 `LlamaLora` 适配器，见下文 |
| 普通 PyTorch 模型 | `ruda_torch.inject_lora`，可先执行 `quantize_nf4` |
| 本地 Hugging Face safetensors 权重 | `load_hf_nf4_model` 或与架构无关的 `load_nf4_safetensors` |
| 因果语言模型训练 | `CausalLMFinetuner`、`SFTCollator`、`SFTTrainer` |

通用接口不自动选择模型家族、猜测适配器目标、下载权重或补齐缺失设备算子。模型实际调用的前向／反向操作仍须由执行后端支持。原生 PyTorch 使用 `ruda:0`，不是 `torch.cuda` 存储。先准备匹配的[原生组件](../../ruda-torch/README.md#build-and-install)，使用打包权重时还须具备对应 NF4 扩展。

Python safetensors 流式加载需要 `safetensors`；Python HF 辅助入口和命令行示例还需要 `transformers`、`accelerate`。所选架构须能由已安装的 Transformers 在不执行远程代码的情况下构造。T4 的低精度路径选择 FP16，不要在缺少相应硬件支持时选择 BF16。

## Rust ruLLM SFT 与 LoRA

`rullm::training` 训练从本地 Llama／Qwen2 权重加载的浮点 `PackedLlamaForCausalLm`。它与下文 Python `SFTTrainer`、PEFT、NF4 接口相互独立，Rust 示例不需要 Python Transformers。后端须实现 `AutodiffBackend + TrainingBackend`；NVIDIA 示例使用 `Autodiff<Cuda<bf16, i32>>`，加 `--f32` 则使用 `Autodiff<Cuda<f32, i32>>`。此入口不是 AWQ／NF4 量化 base 训练，也不是 Qwen3.5 训练器。

### 数据与命令行

准备本地[模型目录](model-inference.md#准备本地模型)和 JSONL 数据集。每个非空行可采用以下格式之一：

```jsonl
{"messages":[{"role":"user","content":"什么是 KV 缓存？"},{"role":"assistant","content":"它复用先前 token 的 key 和 value。"}]}
{"prompt":"问题：1 + 1？\n答案：","completion":"2"}
{"text":"用于继续训练因果语言模型的一段文本。"}
```

示例选择 `TextFormat::QwenChat`：`messages` 按 Qwen2 ChatML 格式化，只监督 assistant 区间；prompt／completion 记录按原字符串分词，仅监督 completion；text 记录监督全文。`SftExample` 中的 labels 不提前错位，由 `SftBatch::collate` 错位一次，并在右侧用忽略标签补齐（`IGNORE_INDEX = -100`）。`from_jsonl` 按 `max_tokens` 截断，返回过短或无监督目标的跳过样本数。CLI 传入 `--max-length + 1`，错位后输入长度最多为 `--max-length`。

在仓库根目录运行，下面的输出目录位于仓库外：

```powershell
$env:RUDA_CUDA_COMPILER = 'nvrtc'
cargo run --release --locked -p ruda-llm --features nvidia --example qwen2_sft -- ./models/qwen2 ../train.jsonl ../qwen2-sft-run --steps 200 --micro-batch 4 --accumulation 4 --max-length 1024 --checkpoint-every 50
```

| 选项 | 默认值 | 含义 |
| --- | --- | --- |
| `--steps`／`--lr` | `200`／`2e-4` | 优化器总步数目标与峰值学习率 |
| `--micro-batch`／`--accumulation` | `4`／`4` | 每个 microbatch 的样本数、每个优化器 step 累积的 microbatch 数 |
| `--max-length` | `1024` | 错位后输入长度上限，须在模型 RoPE 容量内 |
| `--rank`／`--alpha` | `16`／`32` | LoRA 秩与缩放分子 |
| `--full` | 不设置 | 训练全部模型参数，而非仅训练 LoRA 适配器 |
| `--checkpoint-every` | `50` | 存档间隔的优化器步数；0 表示仅结束时存档 |
| `--f32` | 不设置 | 改用 FP32，默认后端为 BF16 |

BF16 WMMA fragment 仅在 CUDA sm_80 及以上宣告支持；能力判断不会让较老设备获得这些 fragment。类型仍须由模型实际算子支持，`--f32` 是本示例提供的 BF16 替代选项。

### 训练接口与内存控制

| 接口 | 契约 |
| --- | --- |
| `model.into_packed_training()` | 合并投影并保留参数可训练性 |
| `SftTrainer::full(model, config, device)` | 用 AdamW 训练整个 packed 模型 |
| `LlamaLora::new(model.lora_shapes(), lora_config, &device)` | 构造 FP32 适配器；`SftTrainer::lora` 冻结 base |
| `LoraTargets::ALL`／`ATTENTION` | 选择 packed qkv／output／gate-up／down，或仅 qkv／output；也可逐标志选择 |
| `SftLoader::new(dataset, micro_batch, pad_token_id, seed)` | 按种子洗牌并循环 epoch，默认长度排序窗口为 64 |
| `trainer.step(&micro_batches)` | 执行一次按 token 归一化的累积 step；调用方维护 loader 位置 |
| `trainer.train(&mut loader, callback)` | 驱动到 `schedule.total_steps`，保存 loader 位置／checkpoint，并调用非阻塞进度回调 |

LoRA A／B 形状为 `[in, rank]`／`[rank, out]`，B 初始化为零，更新按 `alpha / rank` 缩放。适配器保留 FP32，在投影边界转换类型。CLI 选择 `LoraTargets::ALL`、seed 0；这是 ruLLM packed 投影适配器，不是 PEFT 适配器存档格式。

注意力采用分块因果／窗口前向，并在反向重算概率。词表损失逐行块处理完整词表，不保留 `[batch, sequence, vocabulary]` 的全部 logits。`TrainConfig::loss_chunk_bytes` 默认 128 MiB，控制单个 FP32 logits 块，不是整个任务的显存上限。累计损失除以整个窗口有效目标 token 总数，不是对 microbatch 均值再平均。AdamW 默认 betas 0.9／0.95、epsilon 1e-8、weight decay 0、全局范数裁剪 1.0；学习率线性预热后余弦衰减到峰值的 10%。非有限 loss／范数跳过更新，并增加 `skipped_steps`。

`TrainProgress` 包含当前／总步数、loss、学习率、梯度范数、token 数、耗时、近期速率、ETA、epoch 及最近存档年龄。CLI 输出 JSON 行，不要求交互输入；回调不测量 GPU 利用率或内存。对于调用方已注册的 ruCCL peers，`distributed` feature 提供 `with_data_parallel(peer)`，loader 对应调用 `with_shard(rank, world_size)`；CLI 不负责启动 ranks 或初始化集合通信。

### 存档、恢复与导出

CLI 在 `<output-dir>/checkpoints/step-NNNNNNNN` 保存 `trainable.safetensors`、优化器记录和 `state.json`；`latest` 是记录最新存档目录名的文本文件。API 的 `keep_checkpoints` 默认 1，会清理较旧的 step 目录。模型与优化器文件先写入临时目录再提升为正式存档；磁盘应容纳当前存档、替换存档及最终导出权重。

同一输出目录重跑命令会自动调用 `trainer.resume()`。恢复时重建相同的 base、训练模式、适配器布局、数据集／顺序、microbatch 大小和训练配置。LoRA 存档包含适配器与优化器状态，**不包含冻结 base**，不能替代原模型目录。`train` 恢复 loader 的 epoch／cursor／seed，再运行到配置的总步数目标，而非额外运行这么多步。`load_checkpoint(path)` 可指定存档。Adam／AdamW 恢复的 moments 会转为当前梯度类型，以便 BF16 base 上的 FP32 适配器继续使用相容的优化器状态。

结束时，示例按需将 LoRA 合并到 base，并写入 `<output-dir>/model.safetensors`。将源模型的 `config.json`、分词器文件及适用的生成配置放在旁边，组成可加载目录；不要沿用描述原权重分片的索引文件。`export_huggingface_safetensors(&model, path, tie_word_embeddings)` 使用 HF Llama／Qwen2 张量名，绑定 embedding 时省略 `lm_head.weight`。这是部署权重，不是用于续训的优化器／loader 存档。

## 明确选择投影层

修改前检查 `model.named_modules()`。`target_modules` 可以是字符串 `'all-linear'`，或非空的**完整限定模块名列表**，不是后缀匹配或正则表达式。根节点本身是 `Linear` 时，先放入容器。选中共享模块的一个别名，也会替换它的其他别名。

稠密 LoRA 调用 `inject_lora(model, target_modules=targets, rank=16, alpha=16., adapter_dtype=torch.float32)`。它原地修改并返回模型，冻结全部原参数、清除其梯度，仅保留适配器参数可训练。**注入之后**再创建优化器，只传入 `requires_grad=True` 的参数；重复注入会报错。

`LoRALinear` 计算 `base(x) + (alpha / rank) * B(A(x))`。base 为 `torch.nn.Linear` 或 `NF4Linear`；A 的形状为 `[rank,in_features]`，B 为 `[out_features,rank]`，B 初始化为零。Python 适配器没有 dropout 参数。`rank` 须为正整数，`alpha` 须有限且为正，`adapter_dtype` 可为 FP32／FP16／BF16；更新量转为 base 输出类型后相加。

### Rust LoRA

`LoRALinearConfig::new(rank, alpha).with_dropout(p).init(base)` 接收已有 Rust `Linear<B>`，冻结 base 而不改变其参数 ID。A 从输入宽度投影到 rank，B 从 rank 投影到输出宽度并初始化为零，两者使用 base 权重类型；`forward` 保留受支持的前导维度。

Rust 要求 `rank > 0`、`alpha` 有限、`0 <= dropout < 1`，无效配置触发断言；与 Python 不同，不要求 alpha 为正。`merge(self)` 消耗适配器，生成不含适配器 dropout 的冻结稠密投影，不转换优化器状态，也不能用于恢复 LoRA 训练。见 [Rust 实现](../../ruda-nn/src/modules/lora.rs)。

## 打包冻结的 NF4 base

对于已加载到 CPU 的模型，在注入 LoRA、搬到 RUDA **之前**调用 `quantize_nf4(model, target_modules=targets, block_size=64, tile_rows=128)`。所选权重须为连续 CPU FP32／FP16／BF16 矩阵。量化目标应排除 embedding／输出头的共享权重：共享模块别名会保留，但与其他模块绑定的权重会被拒绝。转换逐层提交，失败时已完成层仍保持转换后的状态。

`NF4Linear` 保存打包编码、FP32 分块尺度和 FP32 码本，不保留整份稠密权重副本。支持一阶输入梯度，冻结打包 base 不参与求导。移动模块或改变其浮点类型时，尺度与码本仍保持 FP32。

RUDA NF4 格式为行优先，每字节两个编码，第一个值放在高半字节；每个展平分块对应一个 FP32 绝对值最大值。支持末尾不足一块及奇数元素数。它不是 bitsandbytes／PEFT checkpoint 格式，也不等同于 AWQ INT4 或可学习尺度伪量化。

RUDA 上具备 NF4 matmul API 1 时，FP16／BF16 使用融合分块反量化 GEMM。FP32 或缺少该可选 matmul 能力时使用有界分块解码，解码仍要求 NF4 API 1。融合调用失败不会改道重试。`tile_rows` 控制该解码路径的输出行分块，不是训练批量大小。CPU／CUDA 为明确选定的同设备参考执行，不是 RUDA 失败后的恢复路径。

## 流式加载本地权重

`load_nf4_safetensors(model, directory, target_modules=targets, device='ruda:0', dtype=torch.float16)` 要求先构造模型，**全部参数位于 meta**。它逐张量读取 `model.safetensors`，或 `model.safetensors.index.json` 列出的分片。选中的稠密权重在 CPU 上量化后上传打包存储，其他张量分别加载；名称与形状须与模型架构准确匹配，不自动改键名或处理 tokenizer。

- `parameter_dtypes`、`buffer_dtypes` 是完整浮点张量名到 FP32／FP16／BF16 的字典；共享张量上的冲突覆盖会报错。
- 指定保留类型的权重不能同时作为 NF4 目标。非目标浮点参数默认使用 `dtype`；buffer 默认保留原类型，除非显式覆盖。
- 权重文件省略的非持久 buffer 必须由构造器实际创建，不能仍停留在 meta。
- 分片须为权重目录内已有文件。缺少张量、形状／种类不匹配、无效共享关系或目标类型不支持均报错。
- 加载不是事务操作；失败后重新构造 meta 模型再重试。

`load_hf_nf4_model(directory, target_modules=targets, device='ruda:0', dtype=torch.float16, rank=16, alpha=16.)` 在 meta 上构造本地 HF 架构，重新应用其权重绑定声明，流式加载 base，再注入适配器。`auto_class` 默认 `AutoModelForCausalLM`，可显式提供其他兼容 AutoModel 类。`config_kwargs`、`model_kwargs` 传给对应构造器，但不允许覆盖本地文件／远程代码控制。它保留 `_keep_in_fp32_modules` 声明，也接受 `parameter_dtypes` 覆盖；不会替调用者选择文本 backbone、词表头或多模态子模型。

两个加载器默认 `dtype=torch.bfloat16`，T4 必须显式传 FP16；打包默认 `block_size=64, tile_rows=128`。HF 辅助入口还默认 `rank=16, alpha=16., adapter_dtype=torch.float32`，`parameter_dtypes`、`config_kwargs`、`model_kwargs` 默认 None；低层加载器另有 `buffer_dtypes=None`。

## 监督、损失与累积

`SFTCollator(tokenizer=None, max_length=..., pad_token_id=..., train_on_prompt=False, truncate=False)` 在 CPU 上右侧补齐，返回且仅返回 `[B,T]` 的 `input_ids`、`attention_mask`、`labels`。IDs／labels 为 int64，mask 为 Bool，padding 标签为 `-100`。

输入有两种形式：

- 预分词记录：非空、等长的 `input_ids` 与 `labels`。标签为词表 ID 或 `-100`；输入 ID 为非负整数。标签不要提前错位。
- 对话记录：`messages` 与 tokenizer。默认要求模板通过 `assistant_masks` 或 `assistant_tokens_mask` 声明 assistant 区间；缺失或全空会报错。显式 `train_on_prompt=True` 则监督模板的全部 token。

`pad_token_id` 须显式提供或由 tokenizer 声明，不自动猜测。超过 `max_length` 默认报错；`truncate=True` 显式保留前缀。`template_kwargs` 不得覆盖编码／监督控制。attention mask 和标签 mask 用途不同。

`CausalLMFinetuner(backbone, head, token_chunk_size=32, activation_checkpointing=True, checkpoint_modules=None, preserve_rng_state=True)` 要求明确拆分 backbone 与词表头。backbone 接收 `input_ids`、`attention_mask`，直接返回 `[B,T,D]` 隐藏状态或带 `last_hidden_state` 的对象，不能返回词表 logits。有 `config` 的 backbone 还会收到 `use_cache=False, return_dict=True`。head 为输入宽度 D 的稠密、NF4 或 LoRA 线性层。

`chunked_lm_cross_entropy(hidden, head, labels, token_chunk_size=32, ignore_index=-100, shift=True, reduction='mean', recompute=True)` 对每个 token 分块投影到**完整词表**，不构造完整 `[B,T,V]` logits。`shift=True` 将隐藏位置 `[:-1]` 配对到标签 `[1:]`；`mean` 按非忽略标签数归一化，`sum` 返回损失和。没有有效标签时返回可求导的零。标签须为同设备 int32／int64，非忽略 ID 须在词表范围内。按选项在反向重算各块 logits，保留隐藏状态与 head 参数梯度。

激活 checkpoint 使用非重入式重算。显式 `checkpoint_modules` 为相对 backbone 的非空、不同且互不嵌套路径；未指定时调用 backbone 自带 checkpoint 接口，否则重算整个 backbone。也可直接调用 `activation_checkpoint_modules(model, paths, preserve_rng_state=True)`，不改变 state-dict 路径。只有明确允许且前向确定性成立时才关闭 RNG 保留；重算中的 RNG 保留不等于进程重启后的 RNG 存档。

| 方法 | 参数与返回 |
| --- | --- |
| `SFTCollator.encode(sample)` | 一条对话／预分词记录 → padding 前的两个 Python 列表 `(input_ids, labels)`。 |
| `SFTCollator(samples)` | 非空批次 → 上述三个 CPU 张量，`template_kwargs` 默认 None。 |
| `CausalLMFinetuner.hidden(input_ids, attention_mask)` | backbone 输入 → 未做词表投影的 `[B,T,D]` 隐藏状态。 |
| `CausalLMFinetuner.forward(input_ids, attention_mask, labels, reduction='mean')` | 标量错位完整词表损失，reduction 为 mean 或 sum。 |
| `activation_checkpoint_modules(model, target_modules, preserve_rng_state=True)` | 原地修改并返回模型，重复 checkpoint 或路径嵌套均报错。 |

`SFTTrainer(model, optimizer, base_id=..., run_config=..., scaler=None, scheduler=None, replica_group=None)` 接收该 wrapper 或其 RUDA 编译 wrapper。`train_step(microbatches)` 要求 CPU 整理后的批次，整个窗口至少有一个有效错位标签。它逐批搬到模型设备，每批损失和除以**整个窗口的有效 token 总数**后反传，再执行一次优化器 step 并清空梯度；不是对不同长度 microbatch 的均值再平均。scaler 要求优化器暴露 `last_step_skipped`；scheduler 仅在更新未跳过时推进。

传入明确初始化的 [ReplicaGroup](distributed-training.md#python-多进程副本训练与-nccl) 后，按全局 token 归一化／报告损失，并在更新前同步累计梯度；其他 rank 有监督时允许本地空窗口。先初始化最终模型再创建优化器，各 rank 分开存档，恢复须匹配 rank／world size。默认仍为单进程。

指标包括尝试的 step／窗口编号、microbatch 游标、监督 token 数、平均损失、耗时、速率与是否跳过更新。跳过更新仍推进 step／cursor。`write_progress(directory, metrics, total_steps=...)` 在本地写 `progress.json`，ETA 来自近期可比较 step 耗时；RUDA GPU 峰值内存明确为未知，不从 CUDA 分配统计推断。

## 命令行训练

已有[示例](../../ruda-torch/python/examples/finetune_causal_lm.py) 使用本地模型与 JSONL 输入。先将 `MODEL_DIR`、`DATA_JSONL`、`RUN_DIR`、`BASE_ID`、`BACKBONE`、`HEAD`、`PAD_TOKEN_ID` 和 Bash 数组 `TARGETS` 设置为实际权重、数据、准确模块路径及 tokenizer 的 padding ID。`RUN_DIR` 必须位于仓库外，`BASE_ID` 标识准确的冻结权重及量化配置。CLI `--targets` 接收完整路径，不接收 API 的 `'all-linear'` 特殊字符串。

先在预期批量／序列形状上运行有界短任务：

```bash
python ruda-torch/python/examples/finetune_causal_lm.py \
  --model "$MODEL_DIR" --data "$DATA_JSONL" --output "$RUN_DIR" \
  --base-id "$BASE_ID" --backbone "$BACKBONE" --head "$HEAD" \
  --targets "${TARGETS[@]}" --dtype fp16 --pad-token-id "$PAD_TOKEN_ID" \
  --max-length 512 --batch-size 1 --accumulation 2 --steps 2 \
  --rank 16 --alpha 16 --token-chunk-size 32 --checkpoint-every 1
```

对话输入加 `--chat`，只有明确需要相应监督策略时才加 `--train-on-prompt`／`--truncate`。`--checkpoint-modules` 接收相对 backbone 的路径；`--compile` 启用 AOT，但不表示全部操作原生；`--cpu-reference` 明确选择 CPU 参考模式。数据文件只遍历一次，不隐式重复或洗牌，须有足够的完整 microbatch。

checkpoint／进度文件留在仓库外。扩大工作量前，先评估相同尺寸的实际 step 耗时与内存；小输入短跑不代表更大模型或序列能容纳。恢复时重复同一配置并加 `--resume "$RUN_DIR/checkpoints/latest.pt"`；`--steps` 为新的总目标，不是额外步数。

## 保存适配器与恢复训练

| API | 内容／契约 |
| --- | --- |
| `adapter_state_dict(model)` | A／B 的 CPU 副本及版本化的目标名、尺寸、rank、alpha、稠密／NF4 base 类型，不含 base 权重 |
| `load_adapter_state_dict(model, state)` | 目标模型须已具有相同适配器布局，验证全部条目后再复制 |
| `finetune_state_dict(model, optimizer, base_id=..., step=..., data_state=..., scaler=None)` | 适配器、优化器类型／状态／分组顺序、可选 scaler、CPU RNG、实际使用的 CUDA RNG、窗口计数和调用者数据状态 |
| `load_finetune_state_dict(model, optimizer, state, base_id=..., scaler=None)` | 恢复上述内容、清空梯度，返回 `(step, data_state)` |
| `SFTTrainer.save(directory)` | 额外保存 run 配置、scheduler、游标／token 总数；写入并检查 `next.pt`，将 `latest.pt` 轮换为 `previous.pt`，再提升新存档 |
| `SFTTrainer.resume(path)` | 要求准确匹配 `run_config`、base 身份、优化器布局／类型及 scheduler／scaler 配置，恢复训练器计数 |

仅在优化器 step 边界、`zero_grad(set_to_none=True)` 和 scaler `update()` **之后**存档；不保存半个累积窗口。优化器须包含每个可训练 A／B 参数且恰好一次，不能包含冻结或非适配器参数。冻结 base 不重复存入 checkpoint：先重建相同 base、量化、适配器和优化器分组，再恢复。数据源推进到存档 microbatch 游标；自定义 sampler 状态由调用者放入 `data_state`。

直接 API 返回的映射用 `torch.save` 序列化，再用 `torch.load(..., map_location='cpu', weights_only=True)` 加载兼容的张量／基本容器状态。`base_id` 与 `run_config` 由调用者提供；CLI 额外记录输入、源码、依赖及原生库身份。低层存档不会自动捕获 Python／NumPy RNG、外部 sampler 状态或独立 RUDA 设备 RNG 状态。

`merge_lora(model)` 仅支持非根节点、eval 模式且 base 权重不与其他模块共享的**稠密** LoRA，永久替换为冻结稠密层。它不展开／重新量化 NF4 base；NF4 部署保留适配器形式。之后还要恢复适配器训练时，应在 merge 前保存适配器存档。

## 构造与打包 API 参考

| API | 参数与限制 |
| --- | --- |
| `finetuning.pack_nf4(weight, block_size=64, chunk_blocks=1024)` | 非空、连续、有限的 CPU `[out,in]` FP32／FP16／BF16 权重；正偶数分块大小、正 chunk 数；返回 uint8 字节和 FP32 尺度 |
| `NF4Linear(in_features, out_features, packed, scales, block_size=64, tile_rows=128, bias=None)` | 正尺寸／分块值，权重元素数及 block size 不超过 uint32；同设备连续 packed／scales 长度分别为 `ceil(out*in/2)`、`ceil(out*in/block_size)`；scales 冻结；可选同设备浮点 bias `[out]` |
| `NF4Linear.from_linear(linear, block_size=64, tile_rows=128)` | CPU 稠密层预处理，保留训练模式与冻结 bias |
| `NF4Linear.forward(x)` | 权重设备上的 FP32／FP16／BF16 `[...,in]` → `[...,out]`；autocast 使用该设备选定的激活类型 |
| `LoRALinear(base, rank=16, alpha=16., adapter_dtype=torch.float32)` | 稠密／NF4 base；冻结 base 参数，在同设备创建可训练 A／B |
| `quantize_nf4`、`inject_lora` | 原地转换，完整目标名；先量化再注入，之后再创建优化器 |

NF4Linear、LoRALinear 的 `get_extra_state()`／`set_extra_state(state)` 在普通模块 state-dict 加载时保存并严格检查版本化尺寸／配置，与仅导出适配器的入口不同，不负责加载缺失的冻结 base。

无效元数据／配置抛出 `ValueError`／`TypeError`；缺失原生能力会报错，不改走 CPU。相关实现：[打包及存档](../../ruda-torch/python/ruda_torch/finetuning.py)、[因果训练](../../ruda-torch/python/ruda_torch/causal_finetuning.py)，其他训练接口见[原生 API 参考](native-pytorch-api.md)。
