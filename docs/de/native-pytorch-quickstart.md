# Native PyTorch: Schnellstart

[Dokumentation](README.md) · [API](native-pytorch-api.md) · [English](../en/native-pytorch-quickstart.md)

`ruda-torch` führt PyTorch-Operationen auf einer NVIDIA-GPU aus. Importieren Sie `ruda_torch`, bevor Sie RUDA-Tensoren anlegen. `ruda:0` ist ein anderes Backend als `torch.cuda`; deren Tensoren sind nicht austauschbar.

## 1. Bauen und installieren

Benötigt werden die [NVIDIA-Umgebung](getting-started.md), Rust/Cargo, Python mit PyTorch und setuptools sowie ein C++20-Compiler. Unter Windows verwenden Sie eine x64-MSVC-Entwicklershell. Im Repository-Stamm:

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --locked -p ruda-torch-native
python -m pip install --no-build-isolation --no-deps -e ./ruda-torch/python
```

Wählen Sie den Compiler vor dem Python-Start. Direktes PTX in PowerShell:

```powershell
$env:RUDA_CUDA_COMPILER = 'ptx'
$env:RUDA_PTX_VERSION = '8.0'
```

In Bash:

```sh
export RUDA_CUDA_COMPILER=ptx
export RUDA_PTX_VERSION=8.0
```

Die [PTX-Version](ptx.md) muss zur GPU und zum Treiber passen. `RUDA_CUDA_COMPILER=nvrtc` wählt CUDA C++/NVRTC. Der Loader sucht zuerst die mitgelieferte native Bibliothek, danach den debug-Build des Repositorys. Setzen Sie vor dem Import `RUDA_TORCH_LIBRARY` für einen release-Build oder einen anderen Pfad. Rust-Bibliothek und C++-Erweiterung benötigen beide ABI 10.

Für Windows-Wheels und Linux/Colab-Bundles gelten die in der [Installationsanleitung](../../ruda-torch/README.md#build-and-install) genannten Python-, PyTorch- und Plattformversionen.

## 2. Daten übertragen und Ergebnis lesen

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

Das Ergebnis ist `[0., 2., 4., 6.]`. `.to(device)` lädt Eingaben hoch, `.cpu()` liest Ausgaben zurück. `synchronize()` wartet auf die Ausführung. `execution_stats()` liefert kumulative Operations-/Transferzähler, keine Laufzeit. `is_available()` meldet die initialisierte Integration, nicht die Unterstützung sämtlicher Operatoren oder GPU-Instruktionen.

## 3. Ein FP32-Trainingsschritt

Initialisieren Sie Modell und Daten auf dem Host und verschieben Sie beide ausdrücklich zu RUDA. Das Beispiel verwendet Autograd erster Ordnung und RUDA AdamW.

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

Löschen Sie Gradienten vor dem nächsten Akkumulationsfenster. FP16/BF16, FP32-Masterparameter, Loss Scaling, Clipping und Optimizer-Zustände haben eigene Verträge. Lesen Sie die [API-Referenz](native-pytorch-api.md), bevor Sie Datentypen ändern oder Checkpoints wiederherstellen.

## 4. Weitere Schritte

- [Modellcompiler](model-compiler.md): `ruda_torch.compile`, native Richtlinien und Cache-Freigabe nach abgeschlossenem Backward.
- [Statische Teilgraphen](static-pytorch-graphs.md): feste `GraphOp`-Adressen und Lebensdauer von Replay-Ausgaben.
- [LoRA/NF4](finetuning.md): lokale Gewichte, genaue Ziele, tokengewichtete Gradienten und Wiederaufnahme.
- [Architekturkomponenten](architecture-training.md): Attention, komprimierter KV-Cache, mHC und Python Muon.
- [Öffentliche API-Signaturen](python-api-signatures.md): Konstruktoren, Methoden und Quellcodepositionen.

## 5. Fehler eingrenzen

Prüfen Sie bei Ladefehlern `RUDA_TORCH_LIBRARY` und abhängige Bibliotheken. Unter Windows nimmt `RUDA_TORCH_DLL_DIR` semikolongetrennte Abhängigkeitsverzeichnisse an. Bei ABI-/optional-API-Konflikten bauen Sie Rust und C++ aus demselben Checkout neu. Ein Prozess mit bereits registriertem anderem PrivateUse1-Backend kann nicht verwendet werden.

Bei nicht unterstützten Operatoren prüfen Sie shape/dtype/device-Verträge. Das eager-Backend hat keinen generischen CPU-Fallback. Auch `native='auto'` im Compiler behält den ursprünglichen Gerätedispatch bei und verschiebt nicht auf die CPU. Gerätefehler werden weitergegeben; siehe [Debugging](debugging.md).
