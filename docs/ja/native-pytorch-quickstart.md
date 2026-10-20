# ネイティブ PyTorch クイックスタート

[ドキュメント](README.md) · [API](native-pytorch-api.md) · [English](../en/native-pytorch-quickstart.md)

`ruda-torch` は単一の NVIDIA GPU 上で PyTorch 演算を実行します。RUDA tensor の作成前に `ruda_torch` をインポートしてください。`ruda:0` と `torch.cuda` は別のバックエンドで、両者の tensor は交換できません。

## 1. ビルドとインストール

[NVIDIA 環境](getting-started.md)、Rust/Cargo、PyTorch と setuptools をインストールした Python、C++20 コンパイラが必要です。Windows では x64 MSVC 開発者シェルを使用します。リポジトリのルートで実行してください。

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --locked -p ruda-torch-native
python -m pip install --no-build-isolation --no-deps -e ./ruda-torch/python
```

Python の起動前にコンパイラを選択します。直接 PTX を使用する PowerShell 設定：

```powershell
$env:RUDA_CUDA_COMPILER = 'ptx'
$env:RUDA_PTX_VERSION = '8.0'
```

Bash の設定：

```sh
export RUDA_CUDA_COMPILER=ptx
export RUDA_PTX_VERSION=8.0
```

GPU とドライバが対応する [PTX バージョン](ptx.md)を選択します。`RUDA_CUDA_COMPILER=nvrtc` は CUDA C++/NVRTC 経路を選択します。ローダーはパッケージ内のライブラリ、次にリポジトリの debug ライブラリを探します。release ビルドなど別の場所では、インポート前に `RUDA_TORCH_LIBRARY` を設定してください。Rust と C++ の両コンポーネントに ABI 10 が必要です。

[v0.1.96 wheel](../../crates/ruda-torch/README.md#release-wheels) を使えば Rust/C++ のローカルビルドは不要です。CPython 3.13、Linux x86_64 は PyTorch `2.10.0+cu126`（Ubuntu 24.04 でビルド）、Windows x86_64 は `2.13.0+cu130` が必要です。ネイティブライブラリを同梱するため、独自の `RUDA_TORCH_LIBRARY` 指定は不要です。別配布の Linux/Colab バンドルは従来の[インストール手順](../../crates/ruda-torch/README.md#linuxcolab-precompiled-bundle)に従います。

## 2. データ転送と結果の読み出し

```python
import torch
import ruda_torch

device = torch.device("ruda:0")
host = torch.arange(4, dtype=torch.float32)
x = host.to(device)
y = x + x
ruda_torch.synchronize()
result = y.cpu()
torch.testing.assert_close(result, host + host)
print(result)
print(ruda_torch.execution_stats())
```

結果は `[0., 2., 4., 6.]` です。`.to(device)` は入力をアップロードし、`.cpu()` は出力を読み出します。`synchronize()` は実行完了を待ちます。`execution_stats()` は累積演算／転送カウンタで、所要時間ではありません。`is_available()` は統合の初期化を示し、すべての演算や GPU 命令の対応を保証しません。

## 3. FP32 学習を 1 ステップ実行

モデルとデータをホストで初期化し、明示的に RUDA へ移動します。以下は一階自動微分と RUDA AdamW を使用します。

```python
import torch
import ruda_torch

torch.manual_seed(7)
device = torch.device("ruda:0")
model = torch.nn.Sequential(
    torch.nn.Linear(8, 16),
    torch.nn.SiLU(),
    torch.nn.Linear(16, 4),
).to(device)
x = torch.ones(3, 8, dtype=torch.float32).to(device)
target = torch.zeros(3, 4, dtype=torch.float32).to(device)
optimizer = ruda_torch.AdamW(model.parameters(), lr=1e-3)

optimizer.zero_grad(set_to_none=True)
loss = (model(x) - target).square().mean()
loss.backward()
optimizer.step()
ruda_torch.synchronize()
print("loss:", loss.detach().cpu().item())
```

次の勾配蓄積ウィンドウ前に勾配を消去します。FP16/BF16、FP32 master、loss scaling、clipping、optimizer 状態には独自の契約があります。変更や checkpoint 復元前に [API リファレンス](native-pytorch-api.md)を確認してください。

## 4. 次の手順

- [モデルコンパイラ](model-compiler.md)：`ruda_torch.compile`、native ポリシー、未完了の backward が終了した後のキャッシュ解放。
- [静的サブグラフ](static-pytorch-graphs.md)：固定アドレスの `GraphOp` と replay 出力の寿命。
- [LoRA/NF4](finetuning.md)：ローカル重み、正確なターゲット、token 加重勾配、保存と再開。
- [アーキテクチャ](architecture-training.md)：attention、圧縮 KV、mHC、Python Muon。
- [公開 API シグネチャ](python-api-signatures.md)：コンストラクタ、メソッド、ソース位置。

## 5. エラーの切り分け

ライブラリのロード失敗時は `RUDA_TORCH_LIBRARY` と依存ライブラリを確認します。Windows の `RUDA_TORCH_DLL_DIR` はセミコロン区切りの依存ディレクトリです。ABI や optional API の不一致は同じ checkout から Rust/C++ を再ビルドします。別の PrivateUse1 バックエンドが登録済みのプロセスは使用できません。

未対応の演算は shape/dtype/device 契約を確認してください。eager バックエンドは汎用 CPU fallback を提供しません。コンパイラの `native='auto'` も元のデバイスでの dispatch であり、CPU への移動ではありません。デバイスエラーは伝播します。[デバッグ](debugging.md)も参照してください。
