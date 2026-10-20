use alloc::vec::Vec;
use core::time::Duration;
use ruda_core::benchmark::{BenchmarkComputations, BenchmarkDurations};
use ruda_core::profile::TimingMethod;

/// The timings collected for one candidate, plus the small amount of state needed to decide
/// whether it is still worth sampling.
///
/// `BenchmarkComputations::score` is not usable while tuning is in flight: at a single sample the
/// variance is zero and reads as perfect stability. This tracks the sample count explicitly so
/// elimination can require evidence instead of inferring it from a degenerate variance.
#[derive(Debug, Default)]
pub(crate) struct SampleSet {
    durations: Vec<Duration>,
    stalled: u8,
}

impl SampleSet {
    pub(crate) fn push(&mut self, duration: Duration) {
        let before = self.best();
        self.durations.push(duration);

        // This push ages the biased first sample out, so the two bests are taken over different
        // samples and their difference says nothing about progress.
        if self.durations.len() == DISCARD_THRESHOLD {
            self.stalled = 0;
            return;
        }

        let improved = match (before, self.best()) {
            (Some(before), Some(after)) => TimeChange::of(after, before) == TimeChange::Improvement,
            _ => true,
        };

        self.stalled = if improved {
            0
        } else {
            self.stalled.saturating_add(1)
        };
    }

    pub(crate) fn len(&self) -> usize {
        self.durations.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.durations.is_empty()
    }

    /// The first sample follows a single warmup, so it still carries allocation and clock ramp
    /// costs. It is dropped once enough samples remain without it.
    fn reliable(&self) -> &[Duration] {
        if self.durations.len() >= DISCARD_THRESHOLD {
            &self.durations[1..]
        } else {
            &self.durations
        }
    }

    pub(crate) fn best(&self) -> Option<Duration> {
        self.reliable().iter().min().copied()
    }

    pub(crate) fn converged(&self) -> bool {
        self.stalled >= CONVERGENCE_ROUNDS
    }

    /// Whether enough samples independently landed under `limit` to trust a short circuit.
    ///
    /// Unlike [`Self::best`] this counts the warmup-biased first sample, deliberately: the bias
    /// is toward being slower, so clearing the limit despite it is the conservative direction.
    /// Excluding it would also make a short circuit impossible during the first pass, where it
    /// is the only sample there is.
    pub(crate) fn confirmed_under(&self, limit: Duration, required: usize) -> bool {
        self.durations.iter().filter(|d| **d <= limit).count() >= required
    }

    /// Whether any sample at all landed under `limit` — the cheap gate before a candidate is
    /// worth spending confirmation samples on.
    pub(crate) fn any_under(&self, limit: Duration) -> bool {
        self.durations.iter().any(|d| *d <= limit)
    }

    pub(crate) fn computation(&self, method: TimingMethod) -> BenchmarkComputations {
        BenchmarkComputations::new(&BenchmarkDurations::from_durations(
            method,
            self.reliable().to_vec(),
        ))
    }
}

/// How a time compares to a baseline, once noise is set aside.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TimeChange {
    /// Faster than the baseline by more than the [`IMPROVEMENT_THRESHOLD`].
    Improvement,
    /// Within the threshold of the baseline either way: indistinguishable from noise.
    Neutral,
    /// Slower than the baseline by more than the threshold.
    Regression,
}

impl TimeChange {
    /// How `time` compares to `baseline`.
    pub(crate) fn of(time: Duration, baseline: Duration) -> Self {
        if time < baseline.mul_f64(1.0 - IMPROVEMENT_THRESHOLD) {
            Self::Improvement
        } else if time > baseline.mul_f64(1.0 + IMPROVEMENT_THRESHOLD) {
            Self::Regression
        } else {
            Self::Neutral
        }
    }
}

/// Total samples required before the warmup-biased first sample is discarded.
const DISCARD_THRESHOLD: usize = 3;
/// Relative change a time must exceed to count as an improvement or a regression rather than
/// noise.
const IMPROVEMENT_THRESHOLD: f64 = 0.02;
/// Consecutive non-improving samples before a candidate is considered converged.
const CONVERGENCE_ROUNDS: u8 = 2;
