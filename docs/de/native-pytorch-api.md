# Native PyTorch-API-Verträge

[Inhalt](README.md) · [Ausführliche Methodenreferenz](../en/native-pytorch-api.md)

## Packed-Quantisierung und quantisierte KV-Caches

Packed-API 1 ergänzt INT2/3/4/5/6/8, FP4 E2M1 und FP8 E4M3/E5M2 mit `PackedQuantizedTensor`, `PackedLinear` und `PackedEmbedding`. Linear unterstützt Eingabegradienten bei eingefrorenen Gewichten; die Berechnung nutzt skalare FP32-Akkumulation, nicht Tensor Core GEMM. `scaled_float8_mm` liefert FP16/BF16/FP32. MinMax/Percentile/MSE, `QuantizationObserver` und `affine_fake_quantize` ergänzen Kalibrierung/QAT.

Die Importer unterstützen Standard-GPTQ 2/3/4/8-bit mit `g_idx` und v1/v2 sowie GGUF v2/v3 Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1 und Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K; kein Marlin oder IQ/TQ. Modell, Zielmodule und GGUF-Namens-/Zeilenabbildungen werden explizit vorgegeben. `QuantizedKVCache` und Paged-API 1 lesen gepackte MHA/GQA/MLA-Caches direkt, nur für Inferenz; Backward benötigt Fließkomma-Caches. [Signaturen, Layouts und Beispiele](../en/native-pytorch-api.md#packed-quantization-and-checkpoint-import).

[Öffentliche Signaturen](python-api-signatures.md) · [Paralleles Training, Daten und Wiederherstellung](parallel-and-data.md)

## Zufall und Zustand

`Generator(seed=None)` verwendet Philox4x32-10. `manual_seed`/`manual_seed_all` initialisieren den Standardgenerator für das einzelne Gerät; Seeds sind vorzeichenbehaftete oder vorzeichenlose 64-Bit-Ganzzahlen. `get_rng_state`/`set_rng_state` und Generator `get_state`/`set_state` verwenden einen RUDA-spezifischen CPU-uint8-Zustand mit 24 Bytes, keinen CUDA-RNG-Zustand. Die `_all`-Varianten verwenden eine Liste mit einem Zustand.

`uniform_`, `normal_` und `bernoulli_` füllen FP32/FP16/BF16-RUDA-Tensoren über native random API 1. Parameter müssen endlich, Grenzen geordnet, Standardabweichung nichtnegativ und Wahrscheinlichkeit in `[0,1]` sein. `generator` ist ein RUDA Generator; ohne Angabe gilt `default_generator`. Reservierte Counter eines fehlgeschlagenen Dispatch werden nicht wiederverwendet. `get_amp_supported_dtype` liefert FP16/BF16, `is_initialized` den Initialisierungszustand. `register_random_ops` läuft beim Import; keine erneute Registrierung ist nötig. [Quellcode](../../ruda-torch/python/ruda_torch/random.py).

Import ruda_torch registriert PrivateUse1; kein anderer Backend im selben Prozess. ruda:0 und torch.cuda verwenden verschiedene Allokationen. device_count1/current_device0. is_available zeigt Initialisierung, nicht vollständige Hardwarefähigkeit. execution_stats sind kumulative Zähler, keine Timer.

Rust/C++: Basis-ABI10, Graph3, Training4. Router/Sequence/NF4-Decode/NF4-Matmul jeweils1, Paged-Backward kompatibel1/2 und Ordered2. Basis-ABI allein stellt keine optionalen APIs bereit. Nichtunterstützte Operatoren schlagen fehl; kein generischer CPU-Fallback.

## Training

Fused-Normalisierung benötigt kontiguierliche Dense-ruda:0 FP32/FP16/BF16 ohne unresolved Views. Statistik FP32, Ausgabe Eingabe-Dtype, nur erste Ableitungen.

| API | Vertrag |
| --- | --- |
| rms_norm(x,weight=None,eps=None) | `[...,D]`, positive letzte Dimension. Weight `[D]` auf demselben Gerät, Eingabe-Dtype/FP32. eps=None nutzt finfo.eps. |
| RMSNorm(width,eps=1e-5,elementwise_affine=True) | Positive Breite, affine Parameter standardmäßig FP32; CPU-Konstruktion vor explizitem RUDA-Move. |
| layer_norm / LayerNorm | Nur letzte Achse; optionale Weight/Bias `[D]` im Eingabetyp/FP32, endliches positives eps. |
| silu_mul(gate,up) | Gleiche Shape/Dtype/Device, kein Broadcasting, erhält Speicherrundungsgrenze. |

AdamW: lr.001, betas(.9,.999), eps1e-8, decay.01, fused_step=False, hierarchical_stats=False, max_grad_norm=None. Nichtüberlappende normale Leaf-Parameter, kontiguierliche passende Gradienten; kein capturable/amsgrad/maximize/differentiable. FP32-Master/Momente. Standard verändert Gradienten beim Unscale und liest4 Byte; fused erhält sie und liest12 Byte, weiterhin ein Kernel pro aktivem Parameter. Clipping/hierarchical benötigen fused; nichtendliche Gradienten skippen den ganzen Step.

GradScaler unterstützt ein RUDA AdamW/Muon/MuonAdamW pro Zyklus mit FP32-Skalar-Loss: scale/backward → step → update. Kein unscale_, keine Closure oder Multi-Optimizer-Zyklen. Defaults65536/2/.5/2000 und Grenzen `2**-24..2**24`; Zustand nur zwischen abgeschlossenen Zyklen speichern.

## Streams und Attention

Stream-Priorität nur0. current_stream/default_stream/stream-Kontext; wait_event/wait_stream etablieren Gerätereihenfolge, query wartet nicht, synchronize wartet hostseitig. elapsed_time benötigt abgeschlossene timingfähige Events und liefert ms. record_stream hält Speicher am Leben, erzeugt keine Abhängigkeit. RUDA_TORCH_ASYNC vor erster Submission setzen; prozessweit gecacht.

PagedAttentionPlan hält unveränderlichen Schedule. Q `[queries,q_heads,key_dim]`, K/V `[pages,page_size,kv_heads,dim]`, kontiguierlicher gleicher Floating-Typ; q_heads durch kv_heads teilbar, dim<=1024, Scale endlich positiv. MLA-Queries `[queries,heads,rank]` plus Positionsanteil, Caches `[pages,page_size,1,rank]`, position_dim<=256. Caller hält Cache-Updates/RoPE und originale QK-Skalierung. Splits1 oder2–32, workspace_bytes nur Forward-Scratch. Erste Ableitungen, atomic Standard, ordered explizit und kein Autotuning.

selected_router_weights: logits `[tokens,experts]` Floating, indices `[tokens,top_k]` int32/int64, top_k<=min(experts,64). FP32-Gewichte/dlogits, keine Auswahl. Softmax über alle Experten, Sigmoid pointwise, renormalize über gewählte Slots. Doppelte Indizes akkumulieren; ungültiger Index erzeugt NaN der betroffenen Forward-/Backward-Zeile.

## Sequenzen und Quantisierung

solve_triangular: gleiches Floating-Dtype/Device, A `[...,N,N]`, links B `[...,N,K]` oder rechts `[...,K,N]`, broadcastbare Batchdimensionen; bool-Flags, erste Ableitungen, RUDA-Sequence-API1.

gated_delta_rule: Q/K `[B,H,T,Dk]`, V `[B,H,T,Dv]`, beta/decay `[B,H,T]`, optional State `[B,H,Dk,Dv]`. Q/K/V-Dtype gleich, chunk_size64, Ergebnis output/final_state mit beiden Gradienten. Half-State FP32. Native Chunk-Forward und gleichgerätige Backward-Neuberechnung, kein eigenständiger fusionierter Backward.

learned_fake_quantize: bits2/4/8, Floating-Eingabe und FP32-Scales auf demselben Gerät. Ohne block_shape ein Scale, sonst positive Blöcke pro Achse und `product(ceil(shape/block))` Scales. Unter floor1e-8 kein Scale-Gradient. Floating-Ergebnis erster Ordnung, kein gepacktes NF4.

Weitere Referenzen: [Compiler](model-compiler.md), [Graph](static-pytorch-graphs.md), [Fine-tuning](finetuning.md), [mHC/CSA/HCA/Muon](architecture-training.md), [Verteiltes Training](distributed-training.md). Asynchrone Fehler können bei Sync/Readback sichtbar werden.
