# Python: параллельное обучение, данные и восстановление

[Документация](README.md) · [Точные сигнатуры](python-api-signatures.md) · [English](../en/parallel-and-data.md)

## Процессы и mesh

Приложение запускает workers, выбирает CUDA ordinal перед native-работой и инициализирует `torch.distributed`. Внутри каждого worker устройство — `ruda:0`. Native-связь использует NCCL aliases памяти RUDA CUDA. `device_type='cpu'` явно выбирает Gloo reference, не автоматический fallback.

`ParallelMesh` покрывает всю инициализированную global group. Shape — `(DP,PP,TP)`, координаты — `(data_rank,pipeline_rank,tensor_rank)`, rank — `(data_rank*PP+pipeline_rank)*TP+tensor_rank`. Все ranks используют одинаковые настройки и порядок создания групп. `data`, `pipeline`, `tensor`, `stage` — разные области связи; `close()` освобождает собственные группы. Mesh не запускает процессы и не угадывает разделение модели. [Код](../../crates/ruda-torch/python/ruda_torch/parallel_mesh.py).

## TP-проекции и веса

`ColumnParallelLinear` равномерно делит выходные признаки и суммирует входные градиенты. `gather_output=False` сохраняет локальный выход. `RowParallelLinear` делит входные признаки, разбивает полный вход при необходимости, суммирует частичные выходы и один раз добавляет реплицированный bias. `VocabParallelEmbedding` делит словарь с общими tokens; max_norm, частотное масштабирование и sparse gradients не поддерживаются.

`TensorParallelMLP` соединяет локальные gate/up с row-parallel down; активация по умолчанию — SiLU. `TensorParallelAttention` использует явные Q/K/V/output-проекции. Число Q- и KV-heads делится на размер TP. Вход — `[batch,sequence,hidden]`; позиционное преобразование и mask задаёт приложение.

`tensor_parallelize` изменяет модель по точным непересекающимся путям и plan `column`/`row`/`embedding`. Все потребители общих весов включаются с совместимым разбиением; optimizer создаётся после преобразования. Производные: copy→sum, sum→identity, gather→slice, scatter→gather, а не DP averaging.

`tensor_parallel_state_dict` восстанавливает полный state на каждом rank; `load_tensor_parallel_state_dict` загружает в готовое разбиение. Полная реконструкция требует соответствующей host memory. `load_tensor_parallel_safetensors` непосредственно заполняет уже разделённую meta-модель. [Проекции](../../crates/ruda-torch/python/ruda_torch/parallel_training.py) · [Загрузка](../../crates/ruda-torch/python/ruda_torch/parallel_checkpoint_io.py).

## LoRA/NF4 и внешние форматы

`ColumnParallelNF4Linear`, `RowParallelNF4Linear`, `shard_nf4_linear` сохраняют исходные packed codes и scales; packed bytes — не обычная ось матрицы. Column-LoRA использует локальные base/B и общий A; Row-LoRA — локальные base/A и общий B, с соответствующим суммированием градиентов/активаций.

`bnb_nf4_linear`, `load_bnb_nf4_safetensors`, `load_tensor_parallel_bnb_nf4_safetensors` читают явные метаданные квантования и локальные файлы. Это checkpoint interoperability, не GPU-выполнение bitsandbytes. `load_peft_adapter`/`save_peft_adapter` работают с явными A/B и путями без догадок о семействе модели. Экспорт идёт в новый каталог. [NF4](../../crates/ruda-torch/python/ruda_torch/quantization_interop.py) · [PEFT](../../crates/ruda-torch/python/ruda_torch/adapter_interop.py).

## FSDP, ZeRO и Muon

`FullyShardedModule`/`fully_shard` хранят локальные slices параметров, градиентов и optimizer state. Используйте root или явные непересекающиеся units; общий параметр и его потребители принадлежат одной unit. Non-reentrant recomputation освобождает полные веса после forward и собирает их для backward. Порядок выполнения units с collectives одинаков на всех ranks.

`Zero2Optimizer` сохраняет реплицированные параметры, reduce-scatter gradients и разделённый optimizer state. `ShardedReplicaGroup` координирует FSDP-DP; `TensorParallelGroup` — общий логический minibatch; `DataTensorParallelGroup` — token-weighted DP/TP; `ShardedDataTensorParallelGroup` — FSDP/TP без второго DP sum.

`ShardedMuon`, `TensorParallelMuon`, `MeshShardedMuon` ортогонализуют полные логические матрицы, не считают фрагмент целым параметром. `distributed_grad_norm` учитывает реплицированные значения один раз и каждый shard один раз. [FSDP/ZeRO](../../crates/ruda-torch/python/ruda_torch/sharded_training.py) · [Muon](../../crates/ruda-torch/python/ruda_torch/tensor_parallel_optim.py).

## Pipeline

`PipelineStage` выполняет построенную приложением stage на каждом rank. `PipelineTensorSpec` задаёт фактические shape/dtype, соседние границы совпадают. `StructuredPipelineStage`, `PipelineTreeSpec`, `PipelineLeafSpec`, `PipelineConstant` описывают явные trees, tensor leaves и константы.

`PipelineTrainer` требует уникальные stage names как файловые namespaces. Входы первой stage и targets последней — CPU microbatches; `loss_sum` и фактический `local_weight` явные. По умолчанию schedule — `1f1b`. TP/PP внутри DP replica согласуют логический weight; FSDP pipelines требуют одинаковое число microbatches с collectives между replicas. `PipelineTiedParameters` связывает общие веса stages по явным symbols. Независимый retry частично обновлённой stage — не восстановление. [Trainer](../../crates/ruda-torch/python/ruda_torch/pipeline_trainer.py).

## Данные и checkpoints

`StatefulShardSampler` отделяет потреблённые samples от prefetch. Вызывайте `commit(n)` после реального потребления n примеров. `tail='uneven'` не дублирует данные; `tail='drop'` явно отбрасывает неполный DP round. `from_mesh` использует data coordinate. Сохраняйте неизменяемый `source_id`, epoch, shuffle и seed. `TokenBudgetBatchSampler` сохраняет порядок с фактическими token costs.

`collate_varlen_causal_lm` не создаёт next-token targets между документами и не обрезает неявно. `PackedCausalLMFinetuner` требует flat-token backbone/head; `CausalLMTrainer` обучает полные параметры. [Данные](../../crates/ruda-torch/python/ruda_torch/training_data.py).

`training_state_dict` сохраняет полный persistent model, dtype/ties, optimizer, scaler, CPU/device RNG и data state на optimizer boundary с очищенными градиентами. Записи файлов и синхронизации ranks нет. `validate_training_state_dict` проверяет контракты; `load_training_state_dict` восстанавливает существующие параметры и возвращает step/data state. Используйте `torch.save` и `torch.load(...,map_location='cpu',weights_only=True)`.

`DistributedCheckpoint` проверяет все rank files перед публикацией generation manifest. `save`/`load` и `consolidate`/`load_consolidated` имеют разные topology contracts. Application/sampler state и resharding при смене topology задаются явно. PipelineTrainer сохраняет и восстанавливает все stages. Храните checkpoints вне репозитория. [Восстановление](../../crates/ruda-torch/python/ruda_torch/distributed_checkpoint.py).

## Autotuning и attention

`TrainingAutotuner` измеряет реальное полное forward/backward/update и восстанавливает параметры, optimizer/scaler/scheduler, buffers, RNG и cursor. Snapshot тоже требует памяти. `TrainingPlan.apply` сохраняет фактическую geometry и число supervised tokens. `PipelineAutotuner` сравнивает 1F1B и fill/drain на том же mesh. [Autotuning](../../crates/ruda-torch/python/ruda_torch/training_autotune.py).

`scaled_dot_product_attention` материализует scores; Bool True разрешает, float mask добавляется, внешний mask и causal взаимоисключающие. `block_scaled_dot_product_attention` использует same-device online-softmax tiles; `varlen_scaled_dot_product_attention` изолирует документы с `[tokens,heads,features]` и CPU integer boundaries. Causal alignment при разных Q/K lengths явное. Это не fused kernels.

`vocab_parallel_cross_entropy` использует непрерывный local vocabulary и общие TP targets для одного логического loss. `register_transformers_attention` регистрирует отдельное имя `ruda_` и стандартный mask adapter. Dense masks всё ещё могут потреблять квадратичную память; output attention weights, head masks и softcap не поддерживаются. [Attention](../../crates/ruda-torch/python/ruda_torch/block_attention.py).
