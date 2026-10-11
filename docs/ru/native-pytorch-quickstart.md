# Быстрый старт: нативный PyTorch

[Документация](README.md) · [API](native-pytorch-api.md) · [English](../en/native-pytorch-quickstart.md)

`ruda-torch` выполняет операции PyTorch на одной NVIDIA GPU. Импортируйте `ruda_torch` до создания RUDA-тензоров. `ruda:0` и `torch.cuda` — разные бэкенды; их тензоры не взаимозаменяемы.

## 1. Сборка и установка

Нужны [NVIDIA-окружение](getting-started.md), Rust/Cargo, Python с PyTorch и setuptools и компилятор C++20. В Windows используйте x64 MSVC developer shell. Из корня репозитория:

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --locked -p ruda-torch-native
python -m pip install --no-build-isolation --no-deps -e ./ruda-torch/python
```

Выберите компилятор до запуска Python. Прямой PTX в PowerShell:

```powershell
$env:RUDA_CUDA_COMPILER = 'ptx'
$env:RUDA_PTX_VERSION = '8.0'
```

В Bash:

```sh
export RUDA_CUDA_COMPILER=ptx
export RUDA_PTX_VERSION=8.0
```

Выберите [версию PTX](ptx.md), поддерживаемую GPU и драйвером. `RUDA_CUDA_COMPILER=nvrtc` выбирает CUDA C++/NVRTC. Загрузчик сначала ищет библиотеку внутри пакета, затем debug-библиотеку репозитория. Для release-сборки или другого пути задайте `RUDA_TORCH_LIBRARY` перед импортом. Rust-библиотека и C++-расширение должны использовать ABI 10.

Windows wheel и Linux/Colab bundle требуют сочетания версий Python, PyTorch и платформы, указанного в [инструкции установки](../../ruda-torch/README.md#build-and-install).

## 2. Передача данных и чтение результата

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

Результат — `[0., 2., 4., 6.]`. `.to(device)` загружает вход, `.cpu()` читает выход на хост. `synchronize()` ждёт завершения выполнения. `execution_stats()` возвращает накопленные счётчики операций/передач, а не время. `is_available()` сообщает об инициализации интеграции, но не о поддержке всех операторов или GPU-инструкций.

## 3. Один шаг обучения FP32

Инициализируйте модель и данные на хосте, затем явно перенесите их в RUDA. Пример использует autograd первого порядка и RUDA AdamW.

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

Очищайте градиенты перед следующим окном накопления. FP16/BF16, FP32 master-параметры, loss scaling, clipping и состояние оптимизатора имеют отдельные контракты. Перед сменой dtype или восстановлением checkpoint прочитайте [справочник API](native-pytorch-api.md).

## 4. Дальнейшие шаги

- [Компилятор модели](model-compiler.md): `ruda_torch.compile`, политика native и освобождение кешей после завершения backward.
- [Статические подграфы](static-pytorch-graphs.md): фиксированные адреса `GraphOp` и срок жизни результатов replay.
- [LoRA/NF4](finetuning.md): локальные веса, точные цели, взвешивание градиентов по токенам и восстановление.
- [Архитектурные компоненты](architecture-training.md): attention, сжатый KV-кеш, mHC и Python Muon.
- [Публичные сигнатуры API](python-api-signatures.md): конструкторы, методы и исходный код.

## 5. Диагностика ошибок

При ошибках загрузки проверьте `RUDA_TORCH_LIBRARY` и библиотеки-зависимости. В Windows `RUDA_TORCH_DLL_DIR` принимает каталоги зависимостей через точку с запятой. При несовпадении ABI/optional API пересоберите Rust и C++ из одного checkout. Нельзя использовать процесс, в котором другой PrivateUse1-бэкенд уже зарегистрирован.

Для неподдерживаемых операторов проверьте shape/dtype/device. У eager-бэкенда нет общего CPU-fallback. Политика компилятора `native='auto'` также сохраняет исходное устройство, а не переносит выполнение на CPU. Ошибки устройства передаются вызывающему коду; см. [отладку](debugging.md).
