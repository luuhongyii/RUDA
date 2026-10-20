# ネイティブ PyTorch API 契約

[目次](README.md) · [詳細なメソッド参照](../en/native-pytorch-api.md)

## 数学バックエンドの選択

以下は現在の `main` ソースビルドの API で、古い `v0.1.96` wheel には含まれません。既定は native RUDA です。import 前の `RUDA_TORCH_MATH_BACKEND=ruda` または `nvidia` で初期値を指定できます。`get_math_backend()` は現在の context の選択を返し、`set_math_backend(name)` は変更前の名前を返します。`math_backend(name)` は終了時や例外時に元の選択を復元する context manager です。

NVIDIA 経路は RUDA の allocation と device service 上で cuBLAS／cuBLASLt、cuSOLVER、cuFFT を呼びます。`torch.cuda` への変換でも、すべての演算のリダイレクトでもありません。ライブラリ不足や未対応の明示的 NVIDIA 経路はエラーになります。

`RUDA_TORCH_CUBLAS_LIBRARY`／`CUBLASLT_LIBRARY`／`CUSOLVER_LIBRARY`／`CUFFT_LIBRARY` はそれぞれ完全な `RUDA_TORCH_` 接頭辞で指定します。`RUDA_TORCH_DLL_DIR` は検索ディレクトリ、`RUDA_TORCH_CUDA_LIBRARY` は driver の指定です。wheel はビルド時の Python minor と正確な PyTorch バージョンに結び付き、環境変更時には Rust と C++ を同時に再ビルドします。

複素 linalg、LDL／一般固有値／行列指数、FP16／complex32 FFT の条件は[英語 API](../en/native-pytorch-api.md#scientific-operations)を参照してください。[選択実装](../../ruda-torch/python/ruda_torch/math_backend.py)・[ライブラリ探索](../../ruda-torch/python/ruda_torch/_vendor_runtime.py)・[ビルド設定](../../ruda-torch/python/setup.py)。

## Packed 量子化と量子化 KV キャッシュ

Packed API 1 は `PackedQuantizedTensor`、`PackedLinear`、`PackedEmbedding` による INT2/3/4/5/6/8、FP4 E2M1、FP8 E4M3/E5M2 を提供します。Linear は凍結重みへの入力勾配を扱い、スカラー FP32 累積を使います。Tensor Core GEMM ではありません。`scaled_float8_mm` の出力は FP16/BF16/FP32 です。MinMax／Percentile／MSE、`QuantizationObserver`、`affine_fake_quantize` が校正／QAT を提供します。

標準 GPTQ 2/3/4/8-bit（`g_idx`、v1/v2）と GGUF v2/v3 の Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1、Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K を読み込みます。Marlin、IQ/TQ は非対応です。モデル、対象モジュール、GGUF キー／行配置の対応は明示指定します。`QuantizedKVCache` と paged API 1 は圧縮 MHA/GQA/MLA キャッシュを直接読む推論専用経路で、逆伝播には浮動小数点キャッシュを使います。[シグネチャ、配置、使用例](../en/native-pytorch-api.md#packed-quantization-and-checkpoint-import)。

[公開シグネチャ](python-api-signatures.md) · [並列学習、データと復旧](parallel-and-data.md)

## 乱数と状態

`Generator(seed=None)` は Philox4x32-10 を使います。`manual_seed`／`manual_seed_all` は単一デバイスの既定生成器を初期化し、seed は符号付き／符号なし 64 bit 整数です。`get_rng_state`／`set_rng_state` と Generator の `get_state`／`set_state` は RUDA 固有の 24 byte CPU uint8 状態を扱います。CUDA RNG 状態ではありません。`_all` 版は状態 1 個のリストを扱います。

`uniform_`、`normal_`、`bernoulli_` は native random API 1 で FP32/FP16/BF16 RUDA tensor を埋めます。有限パラメータ、順序付き一様分布境界、非負の標準偏差、`[0,1]` の確率が必要です。`generator` は RUDA Generator、省略時は `default_generator`。失敗した dispatch の予約 counter は再利用されません。`get_amp_supported_dtype` は FP16/BF16、`is_initialized` は初期化状態を返します。`register_random_ops` は import 時に呼ばれるため再登録は不要です。[ソース](../../ruda-torch/python/ruda_torch/random.py)。

import ruda_torch は PrivateUse1 を登録し、別 backend と同じプロセスでは使えません。ruda:0 と torch.cuda の storage は別です。device_count=1、current_device=0。is_available は統合初期化を示し、すべての GPU 命令への対応を保証しません。execution_stats は累積 counter で timer ではありません。

Rust/C++ は base ABI 10、graph 3、training 4。router/sequence/NF4 decode/NF4 matmul は各 API 1、paged backward は互換 bridge の API 1/2、ordered は2。base ABI 一致だけでは任意拡張を提供しません。未対応演算はエラーで、汎用 CPU fallback はありません。

## 学習と dtype

融合 normalization は連続 dense ruda:0 FP32/FP16/BF16、未解決 conjugate/negative view 不可。統計は FP32、出力は input dtype、一階微分のみ。

| API | 条件 |
| --- | --- |
| rms_norm(x,weight=None,eps=None) | `[...,D]`、非空末尾。weight `[D]` 同デバイスで input dtype/FP32。eps None は finfo.eps。 |
| RMSNorm(width,eps=1e-5,elementwise_affine=True) | width 正、affine 既定 FP32。CPU constructor 後 RUDA へ移す。 |
| layer_norm / LayerNorm | 末軸のみ。任意 weight/bias `[D]` は input dtype/FP32、eps 有限正値。 |
| silu_mul(gate,up) | 同 shape/dtype/device、broadcast なし、保存丸め境界を保持。 |

AdamW 既定 lr .001、betas(.9,.999)、eps1e-8、decay .01、fused_step=False、hierarchical_stats=False、max_grad_norm=None。非重複連続 leaf parameter、dense matching gradient を使い、capturable/amsgrad/maximize/differentiable は対象外。FP32 master/moment。既定は gradient を unscale して4 byte readback、fused は gradient を保持し12 byte readback。一 active parameter 一 kernel。clipping/hierarchical は fused を要求し、非有限値は step 全体を skip。

GradScaler は RUDA AdamW/Muon/MuonAdamW 一 optimizer の FP32 scalar loss に使用。scale/backward → step → update の順。unscale_、closure、多 optimizer cycle はありません。init_scale65536、growth2、backoff.5、interval2000、範囲 `2**-24..2**24`。state は完了 cycle 間で保存。

## 実行と attention

Stream priority は0。current_stream/default_stream、stream context を使います。wait_event/wait_stream は device 順序、query は非待機、synchronize は host wait。timing-enabled Event の elapsed_time は完了後の ms。record_stream は寿命保持のみで依存を作りません。RUDA_TORCH_ASYNC は初回提出前に設定し process に cache されます。

PagedAttentionPlan は immutable schedule。Q `[queries,q_heads,key_dim]`、K/V `[pages,page_size,kv_heads,dim]`、同連続 floating storage、q_heads は kv_heads の倍数、dim<=1024、scale は有限正。MLA query `[queries,heads,rank]` と位置幅、cache `[pages,page_size,1,rank]`、position_dim<=256。caller が cache 更新/position encoding を所有し、元の QK scale を使います。splits=1 または2–32、workspace_bytes は forward scratch のみ。一階微分、atomic 既定、ordered は明示選択で autotune ではありません。

selected_router_weights は logits `[tokens,experts]` floating と indices `[tokens,top_k]` int32/int64、top_k<=min(experts,64)。FP32 weight と dlogits を返し、選択はしません。softmax は全 expert、sigmoid は pointwise、renormalize は選択 slot。重複 index は勾配を加算し、無効 index の行は forward/backward とも NaN。

## シーケンスと量子化

solve_triangular は同 floating dtype/device の A `[...,N,N]`、左 B `[...,N,K]` / 右 B `[...,K,N]`、batch broadcast と一階微分。flag は bool、RUDA sequence API 1。

gated_delta_rule は Q/K `[B,H,T,Dk]`、V `[B,H,T,Dv]`、beta/decay `[B,H,T]`、任意状態 `[B,H,Dk,Dv]`。Q/K/V dtype 一致、chunk_size64 が既定、output と final_state の両勾配。低精度 state は FP32。native chunk forward と同デバイス再計算 backward で、独立融合 backward ではありません。

learned_fake_quantize は bits2/4/8、floating input と同デバイス FP32 scale。block_shape None は scale 一つ、指定時は各軸の正 block と `product(ceil(shape/block))` scales。floor1e-8 以下は scale 勾配ゼロ。一階 floating output で packed NF4 ではありません。

関連：[モデル compiler](model-compiler.md)、[固定 graph](static-pytorch-graphs.md)、[LoRA/NF4](finetuning.md)、[mHC/CSA/HCA/Muon](architecture-training.md)、[分散学習](distributed-training.md)。非同期エラーは sync/readback で現れる場合があります。
