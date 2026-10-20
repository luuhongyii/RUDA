# Общий stack autotuning

[Содержание](README.md) · [Runtime](runtime-api.md) · [Политики и примеры](../en/stack-autotuning.md)

[v0.1.95](https://github.com/shuqi2077/RUDA/releases/tag/v0.1.95) содержит `ruda-v0.1.95-a100-sxm4-40gb-autotune.tar.gz` и `SHA256SUMS`; те же данные находятся в [`data/autotune/a100-sxm4-40gb/20261013/`](../../data/autotune/a100-sxm4-40gb/20261013). Распакуйте `stack-autotune-v1` в каталог кеша, переданный `enable_stack_autotune`; `table.csv` служит для просмотра. Сохраняются точные ключи устройства, драйвера, сборки, workload и контекста, а также TTL. Записи benchmark не становятся автоматически совместимыми с PyTorch wheels; импорт Python их не загружает. [Использование и ограничения](../en/stack-autotuning.md#distributed-a100-tables).

Controller проверяет семантически одинаковые варианты, измеряет завершённую работу и кеширует для workload/device. Подключены matmul, attention, forward convolution, fused matmul, packed greedy generation. FFT/sparse/solver/communication, multi-GPU placement и глобальный поиск всей модели автоматически не включены.

enable_stack_autotune(policy,cache_directory) один раз до model/worker/первого вычисления. None — memory-only. Нужны device stack-autotune, fusion device-stack-autotune либо LLM stack-autotune. Native std, не no-std/WASM.

| Mode | Поведение |
| --- | --- |
| Explore | Годный cache, иначе проверка/измерение. |
| CacheOnly | Нет нового timing search, miss использует reference, cold disk hit проверяется. |
| Disabled | Reference без этого cache/trials, не возврат старого LocalTuner route. |

Без установки сохраняется старый route. Defaults EndToEnd/обязательная validation,2 warmups/7 pairs,32 candidates/30s soft budget, min_speedup1.05/relative MAD.15. Capacity1024, TTL7 дней, parallel1, regression7 pairs/ratio1.15, workspace_limit=None. Abs1e-4/rel1e-3, совокупный readback64MiB.

Повторно измеряйте reference/candidate с чередованием порядка, выбирается median ratios, не единичный минимум. EndToEnd включает внутренние allocation/layout/submission/completion, подготовка isolated inputs снаружи. Budget проверяется между завершёнными trials и не прерывает kernel.1.05 — правило выбора, не наблюдённое ускорение.

Writable states изолированы, matmul сохраняет strides/offsets, generation использует свои KV caches. NaN/Inf/неправильный output отвергается; нет validator — reference. Require_validation=False явно разрешает unverified выбор. Совпадение token/конца на calibration prompt не гарантирует все prompts.

Keys включают revisions, device/driver/build, точные shape/stride/dtype/precision/options/context. Неизвестный driver ограничивает disk reuse. RUDA_AUTOTUNE_DRIVER_TAG/BUILD_TAG/CONTEXT_TAG задаются до инициализации как immutable deployment-данные; default context не обнаруживает реальную изоляцию.

Disk stack-autotune-v1 сверяет полный key/version/checksum/TTL. Digest не криптографический и не подпись. Memory hit не требует forced sync/readback; cold disk validation/explore повышают latency, калибруйте offline/startup.

Конкурентный/nested miss использует reference, cross-process exclusion нет. Неизвестная completion fault-ит tuning lane без GPU reset/replay текущего запроса. Hard workspace limit отвергает неизвестные estimates и может отвергнуть reference. Record_comparison принимает проверенные caller-пары при одинаковых условиях, не запускает shadow models самостоятельно.

StackTuner new/policy/stats/reports/select/invalidate/lower_level_fingerprint — публичные controller APIs. Stats не GPU utilization, reports bounded local diagnostics. Invalidate влияет на будущий выбор, удержанный GenerationPlan требует явной recalibration. DiskCache — внутренний, не экспортированный runtime-тип.
