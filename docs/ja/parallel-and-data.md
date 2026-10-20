# Python 並列学習、データと復旧

[ドキュメント](README.md) · [正確なシグネチャ](python-api-signatures.md) · [English](../en/parallel-and-data.md)

## プロセスと mesh

アプリケーションがプロセスを起動し、native 処理前に CUDA ordinal を選び、`torch.distributed` を初期化します。各 worker 内の RUDA デバイスは `ruda:0` です。native 通信は RUDA 所有 CUDA storage の NCCL alias を使います。`device_type='cpu'` は明示的 Gloo 参照で、自動 fallback ではありません。

`ParallelMesh` は初期化済み global group 全体を覆います。shape は `(DP, PP, TP)`、座標は `(data_rank, pipeline_rank, tensor_rank)`、rank は `(data_rank*PP+pipeline_rank)*TP+tensor_rank`。全 rank が同じ設定と group 作成順序を使用します。`data`、`pipeline`、`tensor`、`stage` は別の通信範囲です。`close()` は所有 group を解放します。mesh は worker 起動や stage 分割を推測しません。[ソース](../../crates/ruda-torch/python/ruda_torch/parallel_mesh.py)。

## TP 投影と重み

`ColumnParallelLinear` は出力特徴を等分し、入力勾配を合計します。`gather_output=False` は出力をローカルに保ちます。`RowParallelLinear` は入力特徴を等分し、必要なとき入力を分割し、部分出力の合計後に複製 bias を一度だけ加えます。`VocabParallelEmbedding` は語彙を等分し、token 入力を複製します。max_norm、頻度スケーリング、sparse 勾配は未対応です。

`TensorParallelMLP` はローカル gate/up を row-parallel down に接続し、既定活性化は SiLU。`TensorParallelAttention` は明示的 Q/K/V/output 投影を使い、Q と KV head 数がともに TP サイズで割り切れる必要があります。入力は `[batch,sequence,hidden]`、位置変換と mask は呼び出し側が指定します。

`tensor_parallelize` は重ならない正確な module path と `column`/`row`/`embedding` plan でモデルを変更します。共有重みの全利用箇所を互換分割に含め、変換後に optimizer を作ります。微分は copy→sum、sum→identity、gather→slice、scatter→gather で、DP 平均とは異なります。

`tensor_parallel_state_dict` は全 rank に完全状態を再構築し、`load_tensor_parallel_state_dict` は準備済み分割へロードします。完全再構築には完全モデル相当のホストメモリが必要です。`load_tensor_parallel_safetensors` は分割済み meta モデルへローカル checkpoint を直接ロードします。[投影](../../crates/ruda-torch/python/ruda_torch/parallel_training.py)・[ロード](../../crates/ruda-torch/python/ruda_torch/parallel_checkpoint_io.py)。

## LoRA/NF4 と外部形式

`ColumnParallelNF4Linear`、`RowParallelNF4Linear`、`shard_nf4_linear` は元の packed code と scale を保持します。packed byte は普通の行列軸ではありません。column LoRA はローカル base/B と複製 A、row LoRA はローカル base/A と複製 B を使い、対応する合計勾配・活性化を実行します。

`bnb_nf4_linear`、`load_bnb_nf4_safetensors`、`load_tensor_parallel_bnb_nf4_safetensors` は明示的量子化メタデータを読む checkpoint 互換機能です。bitsandbytes の GPU 実行ではありません。`load_peft_adapter`/`save_peft_adapter` は明示的 A/B と指定 path を扱い、モデル系列を推測しません。出力先は新規ディレクトリです。[NF4](../../crates/ruda-torch/python/ruda_torch/quantization_interop.py)・[PEFT](../../crates/ruda-torch/python/ruda_torch/adapter_interop.py)。

## FSDP、ZeRO と Muon

`FullyShardedModule`/`fully_shard` は parameter、gradient、optimizer state をローカル slice として保持します。root または明示的非重複 unit を使用し、共有 parameter の利用箇所は同じ unit に置きます。非再入 recomputation は forward 後に完全重みを解放し、backward 時に再 gather します。全 rank の collective を含む実行順序は同一です。

`Zero2Optimizer` は parameter を複製したまま gradient を reduce-scatter し、optimizer state を分割します。`ShardedReplicaGroup` は FSDP DP、`TensorParallelGroup` は一つの論理 minibatch、`DataTensorParallelGroup` は token 加重 DP/TP、`ShardedDataTensorParallelGroup` は二重 DP sum を避ける FSDP/TP を調整します。

`ShardedMuon`、`TensorParallelMuon`、`MeshShardedMuon` は完全論理行列を正交化し、ローカル断片を完全行列と見なしません。`distributed_grad_norm` は複製値を一度、各 shard を一度数えます。[FSDP/ZeRO](../../crates/ruda-torch/python/ruda_torch/sharded_training.py)・[Muon](../../crates/ruda-torch/python/ruda_torch/tensor_parallel_optim.py)。

## Pipeline

`PipelineStage` は呼び出し側が作った stage を各 rank で実行します。`PipelineTensorSpec` に実際の shape/dtype を指定し、隣接境界を一致させます。`StructuredPipelineStage`、`PipelineTreeSpec`、`PipelineLeafSpec`、`PipelineConstant` は明示的 tree、tensor leaf、定数を扱います。

`PipelineTrainer` の stage 名は固有の filesystem namespace。先頭 input と末尾 target は CPU microbatch、`loss_sum` と実際の `local_weight` は明示的です。既定 schedule は `1f1b`。DP 副本内の TP/PP は同じ論理 weight を使用し、FSDP pipeline の collective-bearing microbatch 数は副本間で一致させます。`PipelineTiedParameters` は明示的 symbol で stage 間共有重みを結びます。部分更新 stage の単独再試行は復旧ではありません。[trainer](../../crates/ruda-torch/python/ruda_torch/pipeline_trainer.py)。

## データと checkpoint

`StatefulShardSampler` は消費した sample と prefetch を分離します。実際に n 件消費してから `commit(n)` を呼びます。`tail='uneven'` は重複なし、`tail='drop'` は未完了 DP round を明示的に除外します。`from_mesh` は data 座標を使い、dataset の `source_id`、epoch、shuffle/seed を保持します。`TokenBudgetBatchSampler` は実際の token cost と順序を保ちます。

`collate_varlen_causal_lm` は文書間 next-token target や暗黙 truncation を作りません。`PackedCausalLMFinetuner` は flat-token backbone/head、`CausalLMTrainer` は完全 parameter 学習です。[データ](../../crates/ruda-torch/python/ruda_torch/training_data.py)。

`training_state_dict` は gradient 消去済み optimizer 境界で完全 persistent model、dtype/ties、optimizer、scaler、CPU/device RNG と data state を取得します。ファイル書込や rank 同期は行いません。`validate_training_state_dict` は契約を確認し、`load_training_state_dict` は既存 parameter に復元して step/data state を返します。`torch.save` と `torch.load(...,map_location='cpu',weights_only=True)` を使います。

`DistributedCheckpoint` は各 rank file を検証してから generation manifest を公開します。`save`/`load` と `consolidate`/`load_consolidated` は異なる topology 契約です。sampler/application state と topology 変更時の resharding は明示的です。PipelineTrainer の全 stage 保存・再開を使用し、checkpoint は repository 外に置きます。[復旧](../../crates/ruda-torch/python/ruda_torch/distributed_checkpoint.py)。

## 調整と attention

`TrainingAutotuner` は実際の完全 forward/backward/update を測り、parameter、optimizer/scaler/scheduler、buffer、RNG、cursor を復元します。snapshot 自体のメモリも必要です。`TrainingPlan.apply` は実際の geometry と監督 token 数を維持します。`PipelineAutotuner` は同じ mesh の 1F1B と fill/drain を比較します。[調整](../../crates/ruda-torch/python/ruda_torch/training_autotune.py)。

`scaled_dot_product_attention` は score を物化し、Bool True は許可、float mask は加算、外部 mask と causal は排他的です。`block_scaled_dot_product_attention` は同じ device の online-softmax tile、`varlen_scaled_dot_product_attention` は `[tokens,heads,features]` と CPU 整数境界で文書を分離します。Q/K 長さが違う causal alignment は明示的です。fused kernel ではありません。

`vocab_parallel_cross_entropy` は連続 local 語彙と TP 共通 target を使い、一つの論理 loss を返します。`register_transformers_attention` は固有 `ruda_` 名と標準 mask adapter を登録します。dense mask は二次メモリを使う場合があり、attention weight 出力、head mask、softcap は未対応です。[attention](../../crates/ruda-torch/python/ruda_torch/block_attention.py)。
