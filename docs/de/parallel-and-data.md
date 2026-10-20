# Python: paralleles Training, Daten und Wiederherstellung

[Dokumentation](README.md) · [Genaue Signaturen](python-api-signatures.md) · [English](../en/parallel-and-data.md)

## Prozesse und Mesh

Die Anwendung startet Worker, wählt vor nativer Arbeit deren CUDA-Ordinal und initialisiert `torch.distributed`. Jeder Worker nutzt intern `ruda:0`. Native Kommunikation verwendet NCCL-Aliase des RUDA-CUDA-Speichers. `device_type='cpu'` ist ausdrücklich Gloo-Referenzausführung, kein automatischer Fallback.

`ParallelMesh` umfasst die gesamte initialisierte globale Gruppe. Shape ist `(DP, PP, TP)`, Koordinaten sind `(data_rank,pipeline_rank,tensor_rank)`, Rank ist `(data_rank*PP+pipeline_rank)*TP+tensor_rank`. Alle Ranks verwenden dieselbe Konfiguration und Gruppenerstellungsreihenfolge. `data`, `pipeline`, `tensor`, `stage` sind unterschiedliche Kommunikationsbereiche; `close()` gibt eigene Gruppen frei. Das Mesh startet keine Prozesse und errät keine Stufen. [Quelle](../../crates/ruda-torch/python/ruda_torch/parallel_mesh.py).

## TP-Projektionen und Gewichte

`ColumnParallelLinear` teilt Ausgabefeatures gleichmäßig und summiert Eingabegradienten. `gather_output=False` behält lokale Ausgaben. `RowParallelLinear` teilt Eingabefeatures, zerlegt vollständige Eingaben bei Bedarf, summiert Teilausgaben und addiert den replizierten Bias einmal. `VocabParallelEmbedding` teilt das Vokabular; Tokens sind repliziert. max_norm, frequenzskalierte und sparse Gradienten werden abgelehnt.

`TensorParallelMLP` verbindet lokale gate/up-Features mit row-parallelem down; Standardaktivierung ist SiLU. `TensorParallelAttention` verwendet ausdrücklich Q/K/V/output-Projektionen. Q- und KV-Kopfzahlen müssen beide durch die TP-Größe teilbar sein. Eingaben sind `[batch,sequence,hidden]`; Positionsänderung und Masken legt der Aufrufer fest.

`tensor_parallelize` verändert das Modell anhand genauer, nicht überlappender Pfade und `column`/`row`/`embedding`-Pläne. Alle Verbraucher geteilter Gewichte benötigen kompatible Partitionen; danach wird der Optimizer erstellt. Ableitungspaare sind copy→sum, sum→identity, gather→slice, scatter→gather, nicht DP-Mittelung.

`tensor_parallel_state_dict` rekonstruiert vollständigen Zustand auf jedem Rank; `load_tensor_parallel_state_dict` lädt in die vorbereitete Partition. Vollständige Rekonstruktion benötigt entsprechenden Hostspeicher. `load_tensor_parallel_safetensors` lädt direkt in das bereits partitionierte Meta-Modell. [Projektionen](../../crates/ruda-torch/python/ruda_torch/parallel_training.py) · [Laden](../../crates/ruda-torch/python/ruda_torch/parallel_checkpoint_io.py).

## LoRA/NF4 und externe Formate

`ColumnParallelNF4Linear`, `RowParallelNF4Linear`, `shard_nf4_linear` erhalten originale gepackte Codes und Skalen. Gepackte Bytes sind keine gewöhnlichen Matrixachsen. Column-LoRA nutzt lokale base/B und repliziertes A; Row-LoRA lokale base/A und repliziertes B, mit den zugehörigen Summen von Gradienten beziehungsweise Aktivierungen.

`bnb_nf4_linear`, `load_bnb_nf4_safetensors`, `load_tensor_parallel_bnb_nf4_safetensors` lesen ausdrückliche Quantisierungsmetadaten und lokale Dateien; dies ist Checkpoint-Interoperabilität, keine bitsandbytes-GPU-Ausführung. `load_peft_adapter`/`save_peft_adapter` behandeln explizite A/B-Matrizen und Zielpfade, ohne Modellfamilien zu erraten. Export verwendet ein neues Verzeichnis. [NF4](../../crates/ruda-torch/python/ruda_torch/quantization_interop.py) · [PEFT](../../crates/ruda-torch/python/ruda_torch/adapter_interop.py).

## FSDP, ZeRO und Muon

`FullyShardedModule`/`fully_shard` speichern lokale Parameter-, Gradienten- und Optimizer-Slices. Verwenden Sie die Wurzel oder ausdrückliche nicht überlappende Einheiten; geteilte Parameter gehören mit ihren Verbrauchern in eine Einheit. Nicht-reentrante Neuberechnung gibt vollständige Gewichte nach Forward frei und sammelt sie für Backward erneut. Alle Ranks benötigen dieselbe Collective-Ausführungsreihenfolge.

`Zero2Optimizer` repliziert Parameter, reduce-scattert Gradienten und partitioniert Optimizer-Zustand. `ShardedReplicaGroup` koordiniert FSDP-DP, `TensorParallelGroup` einen gemeinsamen logischen Minibatch, `DataTensorParallelGroup` tokengewichtetes DP/TP und `ShardedDataTensorParallelGroup` FSDP/TP ohne zweite DP-Summe.

`ShardedMuon`, `TensorParallelMuon`, `MeshShardedMuon` orthogonalisieren vollständige logische Matrizen, niemals lokale Fragmente als vollständige Parameter. `distributed_grad_norm` zählt replizierte Werte einmal und jeden tatsächlichen Shard einmal. [FSDP/ZeRO](../../crates/ruda-torch/python/ruda_torch/sharded_training.py) · [Muon](../../crates/ruda-torch/python/ruda_torch/tensor_parallel_optim.py).

## Pipeline

`PipelineStage` führt eine vom Aufrufer gebaute Stufe pro Rank aus. `PipelineTensorSpec` enthält tatsächliche Shape/dtype; benachbarte Grenzen stimmen überein. `StructuredPipelineStage`, `PipelineTreeSpec`, `PipelineLeafSpec`, `PipelineConstant` beschreiben ausdrückliche Bäume, Tensorblätter und Konstanten.

`PipelineTrainer` verlangt eindeutige Stufennamen als Dateisystem-Namensräume. Erste Eingaben und letzte Targets sind CPU-Microbatches; `loss_sum` und tatsächliches `local_weight` sind explizit. Standard ist `1f1b`. TP/PP einer DP-Replik stimmen über logisches Gewicht überein; FSDP-Pipelines benötigen gleiche Collective-Microbatchzahlen zwischen Repliken. `PipelineTiedParameters` bindet gemeinsame Stufengewichte über ausdrückliche Symbole. Einzelnes Wiederholen einer teilweise aktualisierten Stufe ist keine Wiederherstellung. [Trainer](../../crates/ruda-torch/python/ruda_torch/pipeline_trainer.py).

## Daten und Checkpoints

`StatefulShardSampler` trennt verbrauchte Samples von Prefetch. Rufen Sie `commit(n)` erst nach tatsächlichem Verbrauch auf. `tail='uneven'` vermeidet Duplikate; `tail='drop'` entfernt ausdrücklich die unvollständige DP-Runde. `from_mesh` nutzt die Datenkoordinate. Bewahren Sie unveränderliche `source_id`, Epoche und Shuffle/Seed-Politik. `TokenBudgetBatchSampler` erhält Reihenfolge und tatsächliche Tokenkosten.

`collate_varlen_causal_lm` erzeugt keine dokumentübergreifenden Next-Token-Ziele und schneidet nicht implizit ab. `PackedCausalLMFinetuner` erwartet einen Flat-Token-Backbone/Head; `CausalLMTrainer` trainiert vollständige Parameter. [Daten](../../crates/ruda-torch/python/ruda_torch/training_data.py).

`training_state_dict` erfasst vollständigen persistenten Modellzustand, dtype/Ties, Optimizer, Scaler, CPU/Geräte-RNG und Datenzustand an einer Optimizer-Grenze mit gelöschten Gradienten. Es schreibt keine Dateien und synchronisiert keine Ranks. `validate_training_state_dict` prüft Verträge; `load_training_state_dict` restauriert vorhandene Parameter und liefert Schritt/Datenzustand. Verwenden Sie `torch.save` und `torch.load(...,map_location='cpu',weights_only=True)`.

`DistributedCheckpoint` prüft alle Rank-Dateien vor Veröffentlichung eines Generation-Manifests. `save`/`load` und `consolidate`/`load_consolidated` besitzen unterschiedliche Topologieverträge. Sampler-/Anwendungszustand und Resharding bei Topologieänderung sind ausdrücklich. PipelineTrainer speichert und restauriert alle Stufen. Checkpoints liegen außerhalb des Repositorys. [Wiederherstellung](../../crates/ruda-torch/python/ruda_torch/distributed_checkpoint.py).

## Tuning und Attention

`TrainingAutotuner` misst vollständige tatsächliche Forward/Backward/Update-Fenster und restauriert Parameter, Optimizer/Scaler/Scheduler, Buffer, RNG und Cursor. Auch der Snapshot braucht Speicher. `TrainingPlan.apply` erhält tatsächliche Geometrie und überwachte Tokenzahl. `PipelineAutotuner` vergleicht 1F1B und Fill/Drain auf demselben Mesh. [Tuning](../../crates/ruda-torch/python/ruda_torch/training_autotune.py).

`scaled_dot_product_attention` materialisiert Scores; Bool True erlaubt Einträge, Floatmasken werden addiert, externe Masken und causal sind exklusiv. `block_scaled_dot_product_attention` verwendet gerätelokale Online-Softmax-Kacheln; `varlen_scaled_dot_product_attention` isoliert Dokumente mit `[tokens,heads,features]` und CPU-Ganzzahlgrenzen. Bei ungleichen Q/K-Längen ist causal alignment ausdrücklich. Dies sind keine fused Kernel.

`vocab_parallel_cross_entropy` nutzt zusammenhängendes lokales Vokabular und gemeinsame TP-Targets für einen logischen Loss. `register_transformers_attention` registriert einen eigenen `ruda_`-Namen samt Standardmaskenadapter. Dense Masken können weiterhin quadratischen Speicher benötigen; Attention-Gewichte, Headmasken und Softcap werden nicht angeboten. [Attention](../../crates/ruda-torch/python/ruda_torch/block_attention.py).
