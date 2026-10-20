use alloc::format;
use alloc::sync::Arc;
use alloc::vec::Vec;
use ruda_core::profile::{Instant, ProfileDuration};

use core::time::Duration;

use alloc::string::{String, ToString};
use ruda_core::benchmark::{BenchmarkComputations, BenchmarkDurations};

use crate::runtime::config::{Logger, autotune::AutotuneLogLevel};
use crate::runtime::server::LaunchError;
use crate::runtime::tune::{AutotuneResult, TimeBound, TuneCache, tune_benchmark_with_eviction};
use crate::runtime::{client::ComputeClient, backend::Runtime};

use super::{AutotuneKey, AutotuneOutput, TunableSet, TuneCacheResult, TuneInputs, Bounds, AutotuneLogContext, AutotuneLoggerExt};

#[derive(Debug)]
/// Runs autotune benchmarks for a single device and caches the results.
///
/// On wasm, [`tune`](Self::tune) spawns its work on the browser event loop; elsewhere
/// it blocks inline. Either way the benchmarking itself is synchronous; only the
/// per-sample profile resolution is awaited.
pub struct Tuner<K: AutotuneKey> {
    cache: Arc<spin::RwLock<TuneCache<K>>>,
    logger: Arc<spin::Mutex<Logger>>,
}

/// The measured outcome for a given autotune invocation.
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
#[derive(new, Debug, Clone, PartialEq, Eq)]
pub struct AutotuneOutcome {
    /// Candidate name.
    pub name: String,
    /// Candidate index in the tuned set.
    pub index: usize,
    /// Statistics from completed measurements.
    pub computation: BenchmarkComputations,
}

impl core::fmt::Display for AutotuneOutcome {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Autotune[{}] name {} => {:?}",
            self.index, self.name, self.computation
        )
    }
}

/// Error from running autotune.
#[derive(Debug, Clone)]
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
pub enum AutotuneError {
    /// An unknown error happened.
    Unknown {
        /// The name of the tunable.
        name: String,
        /// The unknown error,
        err: String,
    },
    /// All samples are invalid.
    InvalidSamples {
        /// The name of the tunable.
        name: String,
    },
    /// No autotune was flagged as valid for the problem.
    ///
    /// # Warning
    ///
    /// This is an unrecoverable error and will cause a panic.
    NoValidKernelFound {
        /// The formatted context on why no valid kernel was found.
        context: String,
    },
    /// The autotune is skipped manually.
    Skip {
        /// The name of the skipped kernel.
        name: String,
    },

    /// An error happened when launching a kernel.
    Launch(LaunchError),

    /// At least one sample carried no timing measurement.
    NotMeasured {
        /// The name of the tunable.
        name: String,
    },
}

impl From<LaunchError> for AutotuneError {
    fn from(value: LaunchError) -> Self {
        Self::Launch(value)
    }
}

/// A successfully-queued benchmark: the profile futures for each sample, plus its metadata.
struct PendingBench {
    index: usize,
    name: String,
    profiles: Vec<ProfileDuration>,
    launch: Option<Duration>,
}

/// A queued tuning job: all data needed to resolve samples and commit the result.
/// Holds no references so it's trivially `Send + 'static` for the wasm spawn path.
struct TuneRequest<K: AutotuneKey> {
    key: K,
    results: Vec<AutotuneResult>,
    #[cfg(serializable)]
    checksum: String,
    #[cfg(persistence)]
    recording: super::record::TuneRecording<K>,
    log_context: Option<AutotuneLogContext>,
    #[cfg(serializable)]
    bounds: Option<Bounds>,
    #[cfg(serializable)]
    limit: Option<Duration>,
    pending: Vec<PendingBench>,
    decided: Option<usize>,
    unreported_mismatches: Vec<String>,
}

#[allow(clippy::new_without_default)]
impl<K: AutotuneKey> Tuner<K> {
    /// Create a tuner. Its cache is seeded from the persistent on-disk cache when
    /// `std_io` is enabled.
    pub fn new(name: &str, device_id: &str) -> Self {
        Self {
            cache: Arc::new(spin::RwLock::new(TuneCache::new(name, device_id))),
            logger: Arc::new(spin::Mutex::new(Logger::new())),
        }
    }

    /// Fetch the fastest autotune operation index for an autotune key.
    pub fn fastest(&self, key: &K) -> TuneCacheResult {
        self.cache.read().fastest(key)
    }

    /// Only a validated, completed decision; no cache loading or benchmark work.
    pub fn settled(&self, key: &K) -> Option<usize> {
        match self.cache.read().fastest(key) {
            TuneCacheResult::Hit { fastest_index } => Some(fastest_index),
            _ => None,
        }
    }

    /// Check the cache, validate checksums if needed, and kick off a tuning job if the
    /// key is a miss. Returns the resolved cache state.
    pub fn check_tune<'a, R: Runtime, F: TuneInputs, Out: AutotuneOutput, Id>(
        &self,
        key: &K,
        inputs: &F::At<'a>,
        tunables: &TunableSet<K, F, Out, Id>,
        #[cfg_attr(not(serializable), allow(unused))] checksum: impl FnOnce() -> String + Send + Sync,
        client: &ComputeClient<R>,
    ) -> TuneCacheResult
    where
        <F as TuneInputs>::At<'a>: Clone + Send,
    {
        self.check_tune_logged(key, inputs, tunables, checksum, client, self.log_context(), Vec::new())
    }

    pub(crate) fn log_context(&self) -> Option<AutotuneLogContext> {
        AutotuneLogContext::new(&mut self.logger.lock())
    }

    pub(crate) fn check_tune_logged<'a, R: Runtime, F: TuneInputs, Out: AutotuneOutput, Id>(
        &self,
        key: &K,
        inputs: &F::At<'a>,
        tunables: &TunableSet<K, F, Out, Id>,
        #[cfg_attr(not(serializable), allow(unused))] checksum: impl FnOnce() -> String + Send + Sync,
        client: &ComputeClient<R>,
        mut log_context: Option<AutotuneLogContext>,
        unreported_mismatches: Vec<String>,
    ) -> TuneCacheResult
    where
        <F as TuneInputs>::At<'a>: Clone + Send,
    {
        // Failed input kernels must not be consumed as failed tuning candidates.
        // Submit outside the cache lock to preserve the device/cache lock order.
        client.flush().expect("Cannot autotune with pre-existing stream errors");
        {
            let mut cache = self.cache.write();
            cache.sync_environment();
            let cur = cache.fastest(key);

            #[cfg(serializable)]
            let cur = if matches!(cur, TuneCacheResult::Unchecked) {
                let mut log = self.logger.lock();
                let checksum = checksum();
                if let AutotuneLogLevel::Full = log.log_level_autotune() {
                    log.log_autotune(&format!("validate checksum key={key}, checksum={checksum}"));
                }
                cache.validate_checksum(key, &checksum)
            } else {
                cur
            };

            match cur {
                TuneCacheResult::Hit { .. } | TuneCacheResult::Pending => {
                    require_no_unreported_mismatches(&unreported_mismatches);
                    return cur;
                }
                TuneCacheResult::Miss | TuneCacheResult::Unchecked => {
                    cache.mark_pending(key.clone())
                }
            }
            // Scope the guard: the rest of this function re-locks `self.cache` (fast
            // path insert, `process_request`), and the write lock is non-reentrant.
        }

        log::info!("Tuning {key}");

        // Fast path: single tunable, no benchmarking needed.
        if tunables.len() == 1 {
            require_no_unreported_mismatches(&unreported_mismatches);
            self.cache.write().cache_insert(key.clone(), 0);
            return TuneCacheResult::Hit { fastest_index: 0 };
        }

        let mut results: Vec<AutotuneResult> = tunables
            .autotunables()
            .map(|a| {
                AutotuneResult::error(AutotuneError::Skip {
                    name: a.name.to_string(),
                })
            })
            .collect();

        #[cfg(serializable)]
        let checksum = tunables.compute_checksum();
        #[cfg(persistence)]
        let mut recording = super::record::TuneRecording::new(&self.cache.read(), key, &checksum);
        #[cfg(persistence)]
        if recording.is_open() {
            log_context.get_or_insert_with(AutotuneLogContext::default);
        }

        let test_inputs = {
            let _real_run = crate::runtime::dry_run::RealRun::new();
            tunables.generate_inputs(key, inputs)
        };
        client.flush().expect("Autotune input generation failed");
        let bounds = tunables.bounds(key, inputs);
        let limit = bounds.as_ref().and_then(TimeBound::time_limit);
        let mut evictor = tunables.evictor(key, inputs);
        #[cfg(not(target_family = "wasm"))]
        let short_circuit = {
            use crate::runtime::config::{RudaRuntimeConfig, RuntimeConfig};
            limit.is_some() && tunables.is_short_circuit_enabled()
                && !RudaRuntimeConfig::get().autotune.disable_short_circuit
        };
        let mut plan = tunables.plan(key);
        #[cfg(persistence)]
        recording.plan(&plan, &tunables.autotunables().map(|op| op.name.as_str()).collect::<Vec<_>>());
        log_context.set_bounds(bounds.clone());
        log_context.set_limit(limit);

        #[cfg(not(target_family = "wasm"))]
        {
            use crate::runtime::config::{RudaRuntimeConfig, RuntimeConfig};
            let config = &RudaRuntimeConfig::get().autotune.bench;
            if config.adaptive {
                let mut schedule = super::schedule::Schedule {
                    config: config.clone(),
                    limit,
                    short_circuit,
                    track_steps: log_context.is_some(),
                    evictor: evictor.take(),
                };
                let operations: Vec<_> = tunables.autotunables().collect();
                let outcome = schedule.run_plan(
                    key, &mut plan, &operations, &test_inputs, client, &mut results,
                );
                for (name, elapsed) in outcome.steps {
                    log_context.push_tuning_step(name, elapsed);
                }
                if let Some(name) = outcome.short_circuit {
                    log_context.push_short_circuit(name);
                }
                let request = TuneRequest {
                    key: key.clone(),
                    results,
                    #[cfg(serializable)]
                    checksum,
                    #[cfg(persistence)]
                    recording,
                    log_context,
                    #[cfg(serializable)]
                    bounds,
                    #[cfg(serializable)]
                    limit,
                    pending: Vec::new(),
                    decided: outcome.decided,
                    unreported_mismatches,
                };
                return ruda_core::future::block_on(
                    process_request(request, &self.cache, &self.logger),
                );
            }
        }

        // Walk the plan batch by batch, launching each benchmark synchronously. A
        // successful launch queues a `PendingBench` for the async resolver below;
        // launch errors go straight into `results`. Retry the next batch if a whole
        // batch failed to queue anything.
        let mut pending = Vec::<PendingBench>::new();
        #[cfg(not(target_family = "wasm"))]
        let mut batch_success = false;
        loop {
            let tunable_indices = plan.next_batch().indices();
            if tunable_indices.is_empty() {
                panic!(
                    "Can't execute the autotune plan for key: {key:?}\n - plan: {plan:?}\n - results: {results:?}"
                );
            }

            for index in tunable_indices {
                let op = tunables.fastest(index);
                let started = log_context.is_some().then(Instant::now);

                match tune_benchmark_with_eviction(
                    op, test_inputs.clone(), client.clone(), evictor.as_deref_mut(),
                ) {
                    Ok(profiles) => {
                        let bench = PendingBench {
                            index, name: op.name.clone(), profiles,
                            launch: started.as_ref().map(Instant::elapsed),
                        };
                        #[cfg(not(target_family = "wasm"))]
                        if short_circuit {
                            let result = ruda_core::future::block_on(resolve_bench(bench));
                            let close_enough = result.outcome.as_ref()
                                .is_ok_and(|outcome| outcome.computation.median <= limit.unwrap());
                            batch_success |= result.outcome.is_ok();
                            results[index] = result;
                            if let Some(started) = started {
                                log_context.push_tuning_step(op.name.clone(), started.elapsed());
                            }
                            if close_enough {
                                log_context.push_short_circuit(op.name.clone());
                                break;
                            }
                            continue;
                        }
                        pending.push(bench);
                    }
                    Err(err) => {
                        results[index] = AutotuneResult::error(err);
                        if let Some(started) = started {
                            log_context.push_tuning_step(op.name.clone(), started.elapsed());
                        }
                    }
                }
            }

            #[cfg(not(target_family = "wasm"))]
            if !pending.is_empty() || batch_success {
                break;
            }
            #[cfg(target_family = "wasm")]
            if !pending.is_empty() {
                break;
            }
        }

        let request = TuneRequest {
            key: key.clone(),
            results,
            #[cfg(serializable)]
            checksum,
            #[cfg(persistence)]
            recording,
            log_context,
            #[cfg(serializable)]
            bounds,
            #[cfg(serializable)]
            limit,
            pending,
            decided: None,
            unreported_mismatches,
        };

        // Resolve samples and commit the result. On wasm this runs on the browser
        // event loop; elsewhere it blocks inline.
        #[cfg(target_family = "wasm")]
        {
            let cache = self.cache.clone();
            let logger = self.logger.clone();
            wasm_bindgen_futures::spawn_local(async move {
                process_request(request, &cache, &logger).await;
            });

            return TuneCacheResult::Pending;
        }

        #[cfg(not(target_family = "wasm"))]
        ruda_core::future::block_on(process_request(request, &self.cache, &self.logger))
    }
}

async fn resolve_bench(bench: PendingBench) -> AutotuneResult {
    let PendingBench { index, name, profiles, launch: _ } = bench;
    if profiles.is_empty() {
        return AutotuneResult::error(AutotuneError::Unknown {
            name, err: "No profiling available".to_string(),
        });
    }
    let timing_method = profiles[0].timing_method();
    let mut durations = Vec::with_capacity(profiles.len());
    let mut measured = true;
    for profile in profiles {
        match profile.resolve().await {
            Some(ticks) => durations.push(ticks.duration()),
            None => measured = false,
        }
    }
    if !measured {
        return AutotuneResult::error(AutotuneError::NotMeasured { name });
    }
    AutotuneResult::success(AutotuneOutcome::new(
        name, index, BenchmarkComputations::new(&BenchmarkDurations::from_durations(
            timing_method, durations,
        )),
    ))
}

/// Await every profile sample, pick the fastest tunable, commit to the cache.
async fn process_request<K: AutotuneKey>(
    request: TuneRequest<K>,
    cache: &spin::RwLock<TuneCache<K>>,
    logger: &spin::Mutex<Logger>,
) -> TuneCacheResult {
    let TuneRequest {
        key,
        mut results,
        #[cfg(serializable)]
        checksum,
        #[cfg(persistence)]
        recording,
        mut log_context,
        #[cfg(serializable)]
        bounds,
        #[cfg(serializable)]
        limit,
        pending,
        decided,
        unreported_mismatches,
    } = request;

    for bench in pending {
        let index = bench.index;
        let name = bench.name.clone();
        let launch = bench.launch;
        let started = Instant::now();
        results[index] = resolve_bench(bench).await;
        if let Some(launch) = launch {
            log_context.push_tuning_step(name, launch + started.elapsed());
        }
    }

    let unmeasured = decided.is_some_and(|index| results[index].outcome.is_err());
    results.sort_by_cached_key(|result| {
        result
            .outcome
            .as_ref()
            .map(|r| r.computation.score())
            .unwrap_or(u64::MAX)
    });

    let fastest_index = decided.unwrap_or_else(|| {
        results
            .first()
            .expect("At least one kernel needed.")
            .outcome
            .as_ref()
            .expect("At least one kernel has to succeed.")
            .index
    });

    {
        if unmeasured {
            require_no_unreported_mismatches(&unreported_mismatches);
            logger.lock().log_autotune(&format!(
                "Unmeasured decision {fastest_index}-{key}; not persisted. Results: {results:?}",
            ));
        } else {
            super::log::log_selected_result(
                &mut logger.lock(), &key, &results, log_context.as_ref(), Some(fastest_index),
            );
        }
        let mut cache = cache.write();
        cache.cache_insert(key.clone(), fastest_index);
        #[cfg(serializable)]
        let stored = !unmeasured && cache.persistent_cache_insert(
            key, checksum, fastest_index, results, bounds, limit,
        );
        drop(cache);
        #[cfg(all(serializable, not(persistence)))]
        let _ = stored;
        #[cfg(persistence)]
        recording.finish(super::record::Answer { winner: fastest_index, stored }, log_context.as_ref());
    }

    TuneCacheResult::Hit { fastest_index }
}

pub(crate) fn require_no_unreported_mismatches(names: &[String]) {
    assert!(
        names.is_empty(),
        "Autotune output mismatch for {names:?}; this path produces no measured decision to report it.",
    );
}

#[cfg(feature = "runtime-autotune-checks")]
pub(crate) fn check_autotune_outputs<O: AutotuneOutput>(
    mut checks_outputs: Vec<(String, Result<O, AutotuneError>)>,
) -> (Vec<crate::runtime::tune::log::CheckResult>, Vec<String>) {
    let _real_run = crate::runtime::dry_run::RealRun::new();
    if checks_outputs.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let reference_idx = checks_outputs
        .iter()
        .position(|(_, res)| res.is_ok())
        .unwrap_or(checks_outputs.len() - 1);
    let reference = checks_outputs.remove(reference_idx);
    let reference_result = reference.1;
    #[cfg(std_io)]
    let reference_name = reference.0;

    let decisions_enabled = is_decisions_enabled();

    #[cfg(std_io)]
    {
        let reference_passed = reference_result.is_ok();
        let (mut check_results, mismatches) = execute_checks(checks_outputs, reference_result, decisions_enabled);
        check_results.push(crate::runtime::tune::log::CheckResult {
            name: reference_name,
            passed: reference_passed,
        });

        (check_results, mismatches)
    }

    #[cfg(not(std_io))]
    {
        execute_checks(checks_outputs, reference_result, decisions_enabled)
    }
}

/// Whether a mismatch should be collected rather than fatal: it can only be reported if decisions
/// are written somewhere, so without a sink a failed check panics on the spot instead of passing
/// silently.
#[cfg(feature = "runtime-autotune-checks")]
fn is_decisions_enabled() -> bool {
    use crate::runtime::config::RuntimeConfig;
    crate::runtime::config::RudaRuntimeConfig::get()
        .autotune
        .decisions_enabled()
}

#[cfg(feature = "runtime-autotune-checks")]
fn execute_checks<O: AutotuneOutput>(
    checks_outputs: Vec<(String, Result<O, AutotuneError>)>,
    reference_result: Result<O, AutotuneError>,
    decisions_enabled: bool,
) -> (Vec<crate::runtime::tune::log::CheckResult>, Vec<String>) {
    let mut check_results = Vec::new();
    let mut mismatches = Vec::new();

    let Ok(reference) = reference_result else {
        for (name, _) in checks_outputs.into_iter() {
            check_results.push(crate::runtime::tune::log::CheckResult {
                name,
                passed: false,
            });
        }
        return (check_results, mismatches);
    };

    for (name, other_result) in checks_outputs.into_iter() {
        if let Ok(other) = other_result {
            let passed = check_equivalence(&reference, other, decisions_enabled);
            if !passed {
                mismatches.push(name.clone());
            }
            check_results.push(crate::runtime::tune::log::CheckResult { name, passed });
        } else {
            check_results.push(crate::runtime::tune::log::CheckResult {
                name,
                passed: false,
            });
        }
    }

    (check_results, mismatches)
}

#[cfg(feature = "runtime-autotune-checks")]
fn check_equivalence<O: AutotuneOutput>(reference: &O, other: O, decisions_enabled: bool) -> bool {
    // When decisions are written somewhere, we catch the panic so we can collect and report every
    // check failure. Without a sink, we let it panic immediately rather than pass silently.
    if decisions_enabled {
        #[cfg(std_io)]
        {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                reference.check_equivalence(other);
            }))
            .is_ok()
        }
        #[cfg(not(std_io))]
        {
            reference.check_equivalence(other);
            true
        }
    } else {
        reference.check_equivalence(other);
        true
    }
}
