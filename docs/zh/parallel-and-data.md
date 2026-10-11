# Python 并行训练、数据与恢复

[文档首页](README.md) · [准确 API 签名](python-api-signatures.md) · [分布式训练](distributed-training.md) · [English](../en/parallel-and-data.md)

## 设备与 rank 所有权

应用负责启动进程、在原生工作开始前选择物理 CUDA ordinal，以及初始化 `torch.distributed`。每个进程内仍使用 `ruda:0`。原生通信使用 RUDA 自有 CUDA 存储的 NCCL 别名；`device_type='cpu'` 是显式 Gloo 参考执行，不是自动 fallback。

`ParallelMesh(data_parallel=..., tensor_parallel=..., pipeline_parallel=...)` 必须覆盖整个已初始化的进程组。shape 为 `(DP, PP, TP)`，坐标为 `(data_rank, pipeline_rank, tensor_rank)`，全局 rank 为 `(data_rank * PP + pipeline_rank) * TP + tensor_rank`。所有 rank 使用相同配置，并按相同顺序创建通信组。`mesh.data`、`mesh.pipeline`、`mesh.tensor` 和 `mesh.stage` 是不同通信范围；`close()` 释放其拥有的组。mesh 不负责启动进程或推断模型的 stage 划分。

源码：[ParallelMesh 与 TensorParallelGroup](../../ruda-torch/python/ruda_torch/parallel_mesh.py)、[ReplicaGroup](../../ruda-torch/python/ruda_torch/distributed_training.py)。

## 张量并行模型转换

| API | 分片与数据契约 |
| --- | --- |
| `ColumnParallelLinear(linear, group, gather_output=True)` | 按实际 `nn.Linear` 的输出特征等分；输入梯度在 TP rank 间求和。禁用 gather 可保留本地特征。 |
| `RowParallelLinear(linear, group, input_is_parallel=False)` | 按输入特征等分；未声明本地输入时拆分完整输入，部分输出求和后只加一次复制的 bias。 |
| `VocabParallelEmbedding(embedding, group)` | 词表等分，token 输入复制、embedding 输出求和。不支持 `max_norm`、按频率缩放或稀疏 embedding 梯度。 |
| `TensorParallelMLP(gate, up, down, group, activation=None)` | 本地 gate/up 中间特征接行并行 down 投影；默认激活为 SiLU。 |
| `TensorParallelAttention(query, key, value, output, group, num_heads=..., ...)` | 显式 MHA/GQA 投影，Q 与 KV 头数均能被 TP 大小整除；输入为 `[batch, sequence, hidden]`。显式提供模型位置变换和 mask。 |
| `tensor_parallelize(model, group, plan)` | 按准确、非重叠模块路径和显式 `column`／`row`／`embedding` 类型原地转换。共享权重的全部消费者必须使用兼容分片；转换后再创建优化器。 |

模型并行的微分对应关系为：`copy_to_tensor_parallel` 前向复制、反向求和；`reduce_from_tensor_parallel` 前向求和、反向原样传递；`gather_from_tensor_parallel` 前向拼接、反向取本地片段；`scatter_to_tensor_parallel` 前向取片段、反向拼接。它们不是普通 DP 梯度平均。

`tensor_parallel_state_dict` 通过集合通信在每个 rank 重建完整状态；`load_tensor_parallel_state_dict` 将普通完整字典加载到已准备的分片。完整重建可能占用完整模型的主机内存。大型本地 safetensors 可使用 `load_tensor_parallel_safetensors` 直接填充已分片的 meta 模型；dtype 映射与准确 NF4 目标由调用者提供。

源码：[投影、转换与状态](../../ruda-torch/python/ruda_torch/parallel_training.py)、[流式 checkpoint 加载](../../ruda-torch/python/ruda_torch/parallel_checkpoint_io.py)。

## NF4／LoRA 分片与互操作

`ColumnParallelNF4Linear`、`RowParallelNF4Linear` 对冻结的 packed base 分片，保留原始编码与 scales。`shard_nf4_linear` 将原始 scales 复制到更小的 block，不会对浮点权重重新量化。packed 字节不能当作普通矩阵轴分片。

`ColumnParallelLoRALinear` 使用本地 base/B 输出行、复制的 A 和求和后的 A 梯度。`RowParallelLoRALinear` 使用本地 base/A 输入列，在复制的 B 前归约 rank 激活。转换依据实际 dense／NF4／LoRA 层类型，而不是模型命名惯例。

- `bnb_nf4_linear`、`load_bnb_nf4_safetensors` 读取显式 bitsandbytes 风格元数据与本地文件，拒绝不支持的格式／尺寸；这是 checkpoint 互操作，不依赖 bitsandbytes 的设备执行。
- `load_tensor_parallel_bnb_nf4_safetensors` 将预量化 base 流式加载到本地 packed TP 投影。
- `load_peft_adapter` 按调用者指定的路径加载 PEFT A/B 矩阵；`save_peft_adapter` 将完整矩阵导出到新目录，不推断模型家族的目标路径。

源码：[并行适配器](../../ruda-torch/python/ruda_torch/parallel_adapters.py)、[NF4 文件格式](../../ruda-torch/python/ruda_torch/quantization_interop.py)、[PEFT 文件](../../ruda-torch/python/ruda_torch/adapter_interop.py)。

## FSDP、ZeRO 与准确 Muon 更新

`FullyShardedModule(module, group, ...)` 保存 rank 本地的参数、梯度和优化器状态片段。`fully_shard(model, group, unit_paths=...)` 包装根模块或显式非重叠单元。非重入重计算在前向后释放已聚合的完整权重，反向时再次聚合。所有 rank 必须按相同顺序执行包含集合通信的单元，包括重计算。共享权重属于同一个单元；将其消费者放在一起，或使用根单元。

`Zero2Optimizer(parameter_groups, group, optimizer_factory)` 保留复制的模型参数，对梯度 reduce-scatter，并维护本地优化器状态。梯度同步使用显式有效权重和缺失梯度策略；它与 FSDP 参数分片不同。

| 协调器／优化器 | 职责 |
| --- | --- |
| `ShardedReplicaGroup` | 含 FSDP 单元模型的数据并行协调。 |
| `TensorParallelGroup` | TP rank 共享同一逻辑 minibatch。 |
| `DataTensorParallelGroup` | TP 模型的数据副本，按 token 加权。 |
| `ShardedDataTensorParallelGroup` | 实际 TP 层的 FSDP 数据片段，使用 DP reduce-scatter，不做第二次 DP 求和。 |
| `ShardedMuon`、`TensorParallelMuon`、`MeshShardedMuon` | 正交化完整逻辑矩阵，而非本地片段；使用匹配的存储和拓扑契约。 |
| `distributed_grad_norm` | 全局 L2 范数：复制张量计一次，各实际分片各计一次。 |

不能对共享同一 loss 的 TP rank 再独立平均，也不能将片段视为完整参数实施全矩阵 Muon。准确构造器、group 和优化器选项见[签名参考](python-api-signatures.md)。

源码：[FSDP／ZeRO](../../ruda-torch/python/ruda_torch/sharded_training.py)、[混合协调](../../ruda-torch/python/ruda_torch/hybrid_parallel.py)、[嵌套 FSDP／TP](../../ruda-torch/python/ruda_torch/sharded_mesh.py)、[分片 Muon](../../ruda-torch/python/ruda_torch/sharded_optim.py)、[TP／mesh Muon](../../ruda-torch/python/ruda_torch/tensor_parallel_optim.py)。

## 流水线边界与优化器步

`PipelineStage(module, group, input_spec=..., output_spec=...)` 在每个流水线 rank 执行一个调用者构造的 stage。`PipelineTensorSpec` 声明实际 shape/dtype，相邻输出／输入必须匹配。`StructuredPipelineStage`、`PipelineTreeSpec`、`PipelineLeafSpec`、`PipelineConstant` 将边界扩展到显式树、dense tensor 叶子和不可变常量；不会从预训练架构推断 stage 或调用方式。

`PipelineTrainer(stage, optimizer, mesh, stage_name=..., base_id=..., run_config=...)` 协调 DP/TP/PP 优化器边界。stage 名是唯一文件系统命名空间。第一 stage 输入和最后 stage 目标是 CPU microbatch；调用者提供 `loss_sum(output, target)` 和实际 `local_weight`。默认 schedule 为 `1f1b`，每个 DP 副本内的 PP/TP rank 必须同意逻辑权重；FSDP 流水线各副本的含集合通信 microbatch 数量必须匹配。

`PipelineTiedParameters` 按显式符号绑定拥有者 stage 上的同一逻辑权重，并校验优化器所有权。使用 trainer 的全 stage 恢复方法；直接重试部分更新的 stage 不是恢复。

源码：[单 tensor 流水线](../../ruda-torch/python/ruda_torch/pipeline_training.py)、[树边界](../../ruda-torch/python/ruda_torch/structured_pipeline.py)、[trainer](../../ruda-torch/python/ruda_torch/pipeline_trainer.py)、[stage 共享权重](../../ruda-torch/python/ruda_torch/pipeline_ties.py)。

## 已提交数据游标与 packed 监督

`StatefulShardSampler(length, source_id=..., rank=..., replicas=..., shuffle=False, seed=0, tail='uneven')` 独立于 DataLoader 预取记录已消耗样本。真正消费 `n` 个样本后调用 `commit(n)`；仅产出索引不会推进存档进度。`tail='uneven'` 保留不重复的非均匀分片；`tail='drop'` 显式丢弃最后不完整的 DP 轮次。`from_mesh` 使用数据坐标，使 TP/PP 伙伴共享样本顺序。保留准确不可变数据集身份、epoch 与 shuffle 策略。

`TokenBudgetBatchSampler` 按调用者提供的实际 token 成本组批，保留 sampler 顺序。`collate_varlen_causal_lm` 打包显式 labels，不产生跨文档 next-token 目标，也不隐式截断。`PackedCausalLMFinetuner` 使用 flat-token backbone/head 契约；`CausalLMTrainer` 提供完整参数训练，而非 SFTTrainer 的适配器专用快照。

源码：[sampler 与 token 预算](../../ruda-torch/python/ruda_torch/training_data.py)、[因果训练器](../../ruda-torch/python/ruda_torch/causal_finetuning.py)。

## 完整模型与全 rank checkpoint

`training_state_dict(model, optimizer, base_id=..., step=..., data_state=..., scaler=None)` 保存完整持久模型状态、存储 dtype/共享关系、实际优化器组、可选 scaler、CPU/设备 RNG 和调用者数据状态。在优化器步完成、全部梯度清空后捕获；它不写文件、不同步 rank。相同架构负责重建非持久 buffer。

`validate_training_state_dict` 在加载前检查身份、尺寸/dtype、共享关系、优化器/scaler 类与布局、RNG 设备集合。`load_training_state_dict` 恢复到已有参数并返回已完成步数和数据状态。使用 `torch.save` 序列化；tensor／基本容器状态可通过 `torch.load(..., map_location='cpu', weights_only=True)` 加载。

`DistributedCheckpoint(group, directory)` 先写入并校验各 rank 文件，再发布 generation manifest。`save`、`load`、`consolidate`、`load_consolidated` 有不同存储／拓扑契约；`model_shard_layout` 描述实际 FSDP/TP 存储与逻辑共享权重。应用／sampler 游标放在 application state；改变拓扑需要显式兼容的 resharding，一个 rank 的文件不是全 rank checkpoint。

PipelineTrainer 的 `save_distributed`／`resume_distributed` 覆盖全部 stage；合并状态恢复接受显式应用 resharding 和可选 stage 变换。checkpoint 文件保存在仓库外。

源码：[完整快照](../../ruda-torch/python/ruda_torch/training_checkpoint.py)、[全 rank generation](../../ruda-torch/python/ruda_torch/distributed_checkpoint.py)、[全 stage 恢复](../../ruda-torch/python/ruda_torch/pipeline_trainer.py)。

## 完整更新自动调优

`TrainingAutotuner` 测量实际训练窗口，搜索 microbatch 与 LM-head chunk 布局，不永久消耗一次优化器更新。试运行保存并恢复参数、优化器/scaler/scheduler、可变状态、RNG、计数器和梯度。快照自身占用内存；必须使用实际目标尺寸，而非缩小的替代输入。`TrainingPlan.apply` 检查 batch 尺寸并保持监督 token 数量。

`PipelineAutotuner` 在不改变 stage mesh 的情况下比较原生 1F1B 与 fill/drain 调度。它与[算子层全栈自动调优](stack-autotuning.md)不同。签名选项、候选预算与策略见 [training_autotune.py](../../ruda-torch/python/ruda_torch/training_autotune.py)。

## 注意力选择

| API | 存储与 mask 契约 |
| --- | --- |
| `scaled_dot_product_attention` | 同设备浮点 MHA/GQA 组合，物化 score；Bool mask 的 True 表示允许，浮点 mask 加到 logits。显式 causal 与外部 mask 互斥。 |
| `block_scaled_dot_product_attention` | 同设备 online-softmax query/key tile，显式 dropout generator、offset 与 window；不是 fused 注意力 kernel。 |
| `varlen_scaled_dot_product_attention` | `[total_tokens, heads, features]` 数据与显式 CPU 整数 `cu_seqlens`，文档互不跨越；Q/K 长度不同时显式选择 `upper_left`／`lower_right` 因果对齐。 |
| `vocab_parallel_cross_entropy` | 本地连续词表区间，TP rank 的逻辑 targets／顺序相同，可含词表 padding；返回一个逻辑 loss，不是 DP rank 总和。 |
| `register_transformers_attention` | 显式注册独立 `ruda_` 名称及 Transformers AttentionInterface 的标准 mask 适配器，不修改模型或覆盖内置实现。 |

block score 存储由 tile 限定，但 Transformers 提供的 dense mask 仍可能占用二次方内存。此注册路径不提供输出注意力权重、head mask 或 soft-capped logits。源码：[dense 注意力](../../ruda-torch/python/ruda_torch/attention.py)、[block 注意力](../../ruda-torch/python/ruda_torch/block_attention.py)、[varlen 注意力](../../ruda-torch/python/ruda_torch/varlen_attention.py)、[词表 loss](../../ruda-torch/python/ruda_torch/parallel_loss.py)、[Transformers 注册](../../ruda-torch/python/ruda_torch/transformers_attention.py)。
