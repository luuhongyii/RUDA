use core::time::Duration;

use alloc::vec::Vec;

use crate::runtime::config::autotune::AutotuneLevel;
use crate::runtime::tune::TuneInputs;

// A bound-builder constructs `AutotuneBound { resource: ResourceBound { .. }, .. }`
// alongside this module's own types, so the neutral record is re-exported
// here too, the same way `Work` is below.
pub use crate::runtime::throughput::ResourceBound;

/// A set of [`AutotuneBound`]s for a given key and reference inputs, with a launch overhead.
#[derive(Debug, Clone, PartialEq)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Bounds {
    /// The bounds for autotuning.
    pub bounds: Vec<AutotuneBound>,
    /// The launch overhead for autotuning.
    pub launch_overhead: Duration,
}

// Sound because [`AutotuneBound`] compares its floats bitwise, so equality stays reflexive.
impl Eq for Bounds {}

/// Produces a set of [`AutotuneBound`]s for a given key and reference inputs.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid bounds generator",
    label = "invalid bounds generator"
)]
pub trait BoundsGenerator<K, I: TuneInputs>: Send + Sync + 'static {
    /// Generate a set of bounds for a given key and reference inputs.
    fn generate<'a>(&self, key: &K, inputs: &I::At<'a>) -> Bounds;
}

/// `Fn(&K, &A) -> Bounds` acts as a [`BoundsGenerator`] when `A` is an owned type. For
/// multi-input kernels, `A` is a tuple that the closure destructures internally.
impl<K, Func, A> BoundsGenerator<K, A> for Func
where
    A: Clone + Send + Sync + 'static,
    K: 'static,
    Func: Send + Sync + 'static + Fn(&K, &A) -> Bounds,
{
    #[inline]
    fn generate<'a>(&self, key: &K, inputs: &<A as TuneInputs>::At<'a>) -> Bounds {
        (self)(key, inputs)
    }
}

/// A calculator that determines the time limit for autotune bounds.
pub trait TimeBound {
    /// Returns the time limit for autotune bounds.
    fn time_limit(&self) -> Option<Duration>;
}

/// A bound for autotuning a throughput kernel: a [`ResourceBound`] plus the
/// threshold over which the kernel is considered accurate.
#[derive(Debug, Clone)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct AutotuneBound {
    /// How much work, against what peak throughput.
    pub resource: ResourceBound,
    /// The threshold for this bound, over which the kernel will be considered accurate.
    pub threshold: f32,
}

/// Bitwise comparison of the measured throughputs, so that equality is reflexive even if a
/// degenerate measurement ever produces a `NaN`, which is what makes the [`Eq`] below sound.
impl PartialEq for AutotuneBound {
    fn eq(&self, other: &Self) -> bool {
        self.resource.peak_per_s.to_bits() == other.resource.peak_per_s.to_bits()
            && self.threshold.to_bits() == other.threshold.to_bits()
            && self.resource.amount == other.resource.amount
    }
}

impl Eq for AutotuneBound {}

// `ruda-core` cannot depend on `ruda-runtime`, so `Work` lives there and
// autotune bounds and benchmark reporting share one definition.
pub use ruda_core::work::Work;

/// Target fractions of modeled peak compute and memory roofline throughput.
#[derive(Debug, Clone, Copy, PartialEq)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Thresholds {
    /// Fraction of peak compute throughput expected.
    pub compute: f32,
    /// Fraction of peak memory bandwidth expected.
    pub memory: f32,
}

impl Thresholds {
    /// The same fraction for both bounds.
    pub const fn uniform(fraction: f32) -> Self {
        Self {
            compute: fraction,
            memory: fraction,
        }
    }

    /// The threshold that sets no limit, so every candidate is measured.
    /// [`AutotuneBound::time_limit`] declines it.
    pub const UNBOUNDED: Self = Self::uniform(0.0);

    /// The fraction of peak an [`AutotuneLevel`] settles for.
    ///
    /// A higher threshold is a tighter time limit, since the limit is the roofline time
    /// divided by the threshold. The fractions are ad-hoc observations rather than a
    /// systematic sweep, so nothing should depend on the exact values.
    pub const fn for_level(level: &AutotuneLevel) -> Self {
        match level {
            AutotuneLevel::Minimal => Self::uniform(0.6),
            AutotuneLevel::Balanced => Self::uniform(0.8),
            AutotuneLevel::Extensive => Self::uniform(0.95),
            AutotuneLevel::Full => Self::UNBOUNDED,
        }
    }
}

impl Default for Thresholds {
    /// The roofline itself: a candidate is expected to reach 100% of the modeled peak, which
    /// is the only fraction that needs no justification.
    fn default() -> Self {
        Self::uniform(1.0)
    }
}

impl TimeBound for AutotuneBound {
    fn time_limit(&self) -> Option<Duration> {
        // The threshold divides the roofline time. A negative one panics `div_f64`, and
        // zero or a subnormal divides the limit away, so neither is a limit to compute.
        if self.threshold <= 0.0 || !self.threshold.is_normal() {
            return None;
        }
        self.resource
            .time_at_peak()
            .map(|limit| limit.div_f64(self.threshold as f64))
    }
}

impl<B: TimeBound> TimeBound for Vec<B> {
    fn time_limit(&self) -> Option<Duration> {
        self.iter().filter_map(|b| b.time_limit()).max()
    }
}

impl TimeBound for Bounds {
    fn time_limit(&self) -> Option<Duration> {
        self.bounds
            .time_limit()
            .map(|limit| limit + self.launch_overhead)
    }
}
