use crate::runtime::{
    config::RudaRuntimeConfig,
    throughput::{ThroughputCache, ThroughputError, ThroughputKey, ThroughputValue},
};
use alloc::boxed::Box;
use alloc::sync::Arc;
use ruda_core::profile::{Duration, Instant};
use ruda_core::config::RuntimeConfig;
use spin::Mutex;
#[allow(unused_imports)]
use num_traits::Float;

type Cache = Arc<Mutex<ThroughputCache>>;

/// Wall clock a warmup may spend growing its iteration count. Generous next to
/// the tens of milliseconds a converging one needs, and the only thing bounding
/// a probe whose timer is too coarse to ever reach the target.
const WARMUP_BUDGET: Duration = Duration::from_secs(2);

/// Wall clock a plateau must hold across before it is accepted, sized against a
/// clock transition, which takes hundreds of milliseconds.
const PLATEAU_FLOOR: Duration = Duration::from_millis(250);

/// Wall clock one shape's peak is sampled over. A sample count would price a
/// probe filling the duration target forty times one whose pass is microseconds.
const SAMPLE_BUDGET: Duration = Duration::from_millis(200);

/// Samples that may go by without improving before the peak is accepted. Also
/// the floor on samples, since the first always improves on infinity and only
/// the ones after it can go stale.
const SAMPLE_PATIENCE: usize = 12;

/// Wall clock one launch of a shape is grown or cut toward, so that a sweep
/// times every shape over the same span and none of them wears a larger share
/// of the fixed cost of a launch than the rest.
const TARGET_DURATION: Duration = Duration::from_millis(20);

/// Samples a ranking pass keeps the fastest of. A count and not a wall clock:
/// under a budget a short pass draws more, and the fastest of more is faster.
const RANK_SAMPLES: usize = 3;

/// Configuration and payload for a benchmarkable compute kernel.
pub struct KernelConfig {
    /// A closure that executes the kernel for the given number of iterations and returns the duration.
    pub sample: Box<dyn Fn(usize) -> Result<Duration, ThroughputError>>,
    /// The number of operations processed in one iteration.
    pub ops_count: usize,
    /// Iterations a launch must carry however quickly they turn out to run,
    /// which a duration target cannot express.
    pub min_iterations: usize,
}

/// What a ranking pass found: how fast the shape answered, and the iteration
/// count it settled on for the next shape to start from.
pub struct Ranked {
    /// The shape's rate, measured briefly enough to order it and not to report
    /// it.
    pub value: ThroughputValue,
    /// What one launch of this shape was timed over.
    pub iterations: usize,
}

/// A marker for measuring throughput of compute kernels.
pub struct ThroughputBenchmarker {
    cache: Cache,
    cache_enabled: bool,
}

impl ThroughputBenchmarker {
    /// Creates a new `ThroughputBenchmarker` with the given cache.
    pub fn new(cache: Cache) -> Self {
        let cache_enabled = !RudaRuntimeConfig::get().throughput.disable_cache;
        Self {
            cache,
            cache_enabled,
        }
    }

    /// The value already measured for `key`, where the cache is on and holds one.
    pub fn cached(&self, key: ThroughputKey) -> Option<ThroughputValue> {
        if !self.cache_enabled {
            return None;
        }

        self.cache.lock().get_current(&key).copied()
    }

    /// The value for `key`, measured by `probe` unless the cache holds one.
    ///
    /// # Errors
    ///
    /// Whatever `probe` reports. Only a measurement is cached.
    pub fn measure(
        &mut self,
        key: ThroughputKey,
        probe: impl FnOnce() -> Result<ThroughputValue, ThroughputError>,
    ) -> Result<ThroughputValue, ThroughputError> {
        if self.cache_enabled
            && let Some(cached_value) = self.cache.lock().get_current(&key)
        {
            return Ok(*cached_value);
        }

        let _real_run = crate::runtime::dry_run::RealRun::new();
        let value = probe()?;

        if self.cache_enabled {
            self.cache.lock().insert(key, value);
        }

        Ok(value)
    }

    /// Warm one shape of a kernel up to its plateau, then keep its fastest sample.
    pub fn sample(kernel_config: KernelConfig) -> Result<ThroughputValue, ThroughputError> {
        let _real_run = crate::runtime::dry_run::RealRun::new();
        let sample = kernel_config.sample;

        let iterations = Self::warmup(kernel_config.min_iterations, WARMUP_BUDGET, &sample)?;
        let duration =
            Self::sample_peak_duration(iterations, &sample, SAMPLE_BUDGET, SAMPLE_PATIENCE)?;

        Ok(ThroughputValue {
            ops_count: kernel_config.ops_count,
            duration,
        })
    }

    /// Warms the device on one shape and reports the count a sample should carry.
    /// It is the device that is warmed, not the shape, so a sweep pays once.
    pub fn warm(kernel_config: &KernelConfig) -> Result<usize, ThroughputError> {
        let _real_run = crate::runtime::dry_run::RealRun::new();
        Self::warmup(
            kernel_config.min_iterations,
            WARMUP_BUDGET,
            &kernel_config.sample,
        )
    }

    /// Keeps the fastest sample of a shape the device is already warm on, at the
    /// count [`warm`](Self::warm) settled.
    pub fn sample_at(kernel_config: &KernelConfig, iterations: usize) -> Result<ThroughputValue, ThroughputError> {
        let _real_run = crate::runtime::dry_run::RealRun::new();
        let iterations = iterations.max(kernel_config.min_iterations).max(1);
        let duration = Self::sample_peak_duration(
            iterations,
            &kernel_config.sample,
            SAMPLE_BUDGET,
            SAMPLE_PATIENCE,
        )?;

        Ok(ThroughputValue {
            ops_count: kernel_config.ops_count,
            duration,
        })
    }

    /// Times one shape briefly, to order it against the others rather than to
    /// report its peak, and reports the count the next shape starts from.
    pub fn rank(kernel_config: &KernelConfig, iterations: usize) -> Result<Ranked, ThroughputError> {
        let _real_run = crate::runtime::dry_run::RealRun::new();
        let settling = iterations.max(kernel_config.min_iterations).max(1);
        // A first launch is what compiling and faulting in cost, so it is spent
        // and the count settled from a second.
        let _ = (kernel_config.sample)(settling)?;
        let took = (kernel_config.sample)(settling)?;

        let iterations = Self::retarget(settling, took)
            .max(kernel_config.min_iterations)
            .max(1);

        let mut fastest = Duration::MAX;
        for _ in 0..RANK_SAMPLES {
            fastest = fastest.min((kernel_config.sample)(iterations)?);
        }

        let duration = fastest / iterations as u32;

        Ok(Ranked {
            value: ThroughputValue {
                ops_count: kernel_config.ops_count,
                duration,
            },
            iterations,
        })
    }

    /// The count that would have taken [`TARGET_DURATION`]. A timer reading zero
    /// says nothing to scale by, so the count stands.
    fn retarget(iterations: usize, took: Duration) -> usize {
        let took = took.as_secs_f64();

        if took <= 0.0 {
            return iterations;
        }

        let scaled = iterations as f64 * (TARGET_DURATION.as_secs_f64() / took);

        if scaled.is_finite() {
            (scaled as usize).max(1)
        } else {
            iterations
        }
    }

    /// Warms up the device by running the kernel multiple times
    /// and estimating the number of iterations needed to reach a stable duration.
    ///
    /// Never returns fewer than `min_iterations`, which the kernel needs to be
    /// measuring what it claims rather than merely to be timed accurately.
    ///
    /// `budget` bounds the growing, not the sampling: a timer too coarse to
    /// ever reach the target reports the same reading at every count, which
    /// asks for a larger one each round without converging.
    fn warmup(
        min_iterations: usize,
        budget: Duration,
        sample: impl Fn(usize) -> Result<Duration, ThroughputError>,
    ) -> Result<usize, ThroughputError> {
        const MAX_WARMUP: usize = 50;
        const MAX_ITERATIONS: usize = 1 << 24;
        // A timer reading zero says nothing about the pass, so doubling against
        // it converges on nothing and stops early. An iteration is a real launch
        // for the probe that measures launches, which pays for every one.
        const MAX_BLIND_ITERATIONS: usize = 1 << 10;
        const PLATEAU_TOL: f64 = 0.03;
        const PATIENCE: usize = 3;
        let target_ms = TARGET_DURATION.as_secs_f64() * 1000.0;

        let mut best = f64::INFINITY;
        let mut stable = 0;
        let mut iterations = min_iterations.max(1);
        let start = Instant::now();
        let mut plateau_start = start;

        for _ in 0..MAX_WARMUP {
            let duration = sample(iterations)?.as_secs_f64() * 1000.0;
            if duration < target_ms {
                let (extra_iters, ceiling) = if duration > 1e-6 {
                    let duration_per_iter = duration / iterations as f64;
                    (
                        ((target_ms - duration) / duration_per_iter).ceil() as usize,
                        MAX_ITERATIONS,
                    )
                } else {
                    (iterations, MAX_BLIND_ITERATIONS)
                };

                let ceiling = ceiling.max(min_iterations);
                if iterations >= ceiling || start.elapsed() >= budget {
                    break;
                }
                iterations = (iterations + extra_iters.max(1)).min(ceiling);
                best = f64::INFINITY;
                stable = 0;
                continue;
            }

            let duration_per_iter = duration / iterations as f64;
            if duration_per_iter < best * (1.0 - PLATEAU_TOL) {
                best = duration_per_iter;
                stable = 0;
                // Growth clears `best`, so the window restarts at the settled count.
                plateau_start = Instant::now();
            } else {
                best = best.min(duration_per_iter);
                stable += 1;
                if stable >= PATIENCE && plateau_start.elapsed() >= PLATEAU_FLOOR {
                    break;
                }
            }
        }

        Ok(iterations)
    }

    /// Sample the peak throughput of the kernel by running it multiple times
    /// and measuring the duration of each iteration.
    fn sample_peak_duration(
        iterations: usize,
        sample_once: impl Fn(usize) -> Result<Duration, ThroughputError>,
        budget: Duration,
        patience: usize,
    ) -> Result<Duration, ThroughputError> {
        debug_assert!(
            iterations > 0,
            "iterations must be positive to avoid division by zero"
        );

        const MAX_SAMPLES: usize = 200;
        const REL_TOL: f64 = 0.01;

        let mut best = f64::INFINITY;
        let mut stale = 0;
        // Wall clock, not the sum of what the samples report: a probe whose
        // timer reads zero would otherwise never spend any of the budget.
        let start = Instant::now();

        for _ in 0..MAX_SAMPLES {
            let s = sample_once(iterations)?.as_secs_f64();
            if s < best * (1.0 - REL_TOL) {
                best = s;
                stale = 0;
            } else {
                best = best.min(s);
                stale += 1;
            }
            if stale >= patience || start.elapsed() >= budget {
                break;
            }
        }

        Ok(Duration::from_secs_f64(best / iterations as f64))
    }
}
