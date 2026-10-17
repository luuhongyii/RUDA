# PTX バックエンドリファレンス

[English](../en/ptx.md) | [简体中文](../zh/ptx.md) | **日本語** | [Deutsch](../de/ptx.md) | [Русский](../ru/ptx.md)

[ドキュメント](README.md) · [コンパイラ ガイド](compiler-guide.md) · [例](samples.md) · [中文](../zh/ptx.md)

## 1. 範囲

`ruda_compiler::ptx` は、Ruda カーネル IR から PTX テキストを生成します。これは、任意の PTX プログラムのインタープリターでも、最終的な NVIDIA マシン コードのジェネレーターでもありません。

スタンドアロン コンパイルに対して `ruda-compiler/ptx` を有効にします。 CUDA ランタイムを介してこのパスを選択するには、`ruda-driver-cuda/direct-ptx` を有効にします。

## 2. ターゲットの種類

|タイプ/フィールド|意味|
| --- | --- |
|`PtxTarget::version: (u32, u32)`|PTX メジャー/マイナー バージョン|
|`PtxTarget::sm: u32`|SM ターゲット|
|`PtxCompilationOptions::target: Option<PtxTarget>`|明示的なターゲット構成|
|`PtxCompiler`|パブリック コンパイラ特性を実装するダイレクト バックエンド|

`target` が存在しない場合、コンパイルは検証エラーを返します。スタンドアロン コンパイラーは、ホスト マシンから GPU アーキテクチャを推論しません。

環境変数の解析では、メジャー バージョンが 6 以上、マイナー バージョンが 9 以下の `major.minor` を受け入れます。構文の妥当性だけでは、そのバージョンのドライバーまたはジェネレーターのサポートは確立されません。

## 3. コンパイル出力

`PtxKernel` には次の内容が含まれます。

- `source`: PTX テキスト。
- `entrypoint`: エントリ ポイント名。
- `ruda_dim`: 元のカーネルのワークグループ ディメンション。
- `shared_memory_bytes`: 共有メモリが必要です。
- `dynamic_metadata_index`: 必要な場合の動的メタデータ ポインター引数の位置。

実行層を呼び出すときに、引数のレイアウト、エントリ ポイント、および共有メモリの要件を保持します。このテキストだけでは完全な立ち上げ契約は含まれていません。

## 4. エラー処理

サポートされていない IR は `CompilationError::UnsupportedInstruction` を返します。無効な構成または構造は、`CompilationError::Validation` を返します。診断には、`Direct PTX:` プレフィックスが含まれます。バックエンドは自動的に NVRTC にフォールバックしません。

定義: [PTX モジュール](../../ruda-compiler/src/ptx/mod.rs)、[コンパイラー テスト](../../ruda-compiler/src/ptx/tests.rs)、および [ランタイム バックエンドの選択](../../ruda-driver-cuda/src/compiler_backend.rs)。

## 5. Packed FP4 転送と NVFP4 scaled MMA

`StorageType::Packed(ElemType::Float(FloatKind::E2M1), 2)` は 1 byte に FP4 値を 2 個格納します。直接 PTX はビットを保持する copy／index／select／reinterpret とメモリ転送に対応します。packed byte を単一 FP4 算術値として扱わず、未対応の演算や変換はエラーになります。転送のみでは `sm_120a` は不要です。

`CoopMma::ExecuteScaled` は SM 120、PTX 8.7 以上、`16×8×64` tile、packed E2M1x2 A／B、F32 C／D、E4M3 scale、`scales_factor = 4` のみ受け付けます。`mxf4nvf4.block_scale.scale_vec::4X` を生成し、出力 target は `sm_120a` になります。他のアーキテクチャ、tile、scale 型や係数は拒否され、fragment のレジスタ容量も必要です。

[Packed 処理](../../ruda-compiler/src/ptx/packed.rs)・[scaled MMA](../../ruda-compiler/src/ptx/matrix/scaled.rs)・[転送／配置サンプル](../../ruda-driver-cuda/examples/ptx_packed.rs)。サンプルは出力済み PTX を読み、byte 転送と fragment 座標を調べるもので、NVFP4 MMA 算術は実行しません。
