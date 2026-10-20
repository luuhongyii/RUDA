use ruda::runtime::throughput::{
    KernelConfig, ThroughputBenchmarker, ThroughputError, ThroughputValue,
};

/// The shapes a probe can be launched in, ranked against each other so that
/// only the winner is measured in full.
pub(super) struct ShapeSweep<S> {
    shapes: alloc::vec::Vec<S>,
}

impl<S: Copy> ShapeSweep<S> {
    pub(super) fn new(shapes: alloc::vec::Vec<S>) -> Self {
        Self { shapes }
    }

    /// What the device answers fastest, and the shape that answered it.
    ///
    /// Shapes are built one at a time: a memory probe's pool is a large fraction
    /// of what the device will allocate. A rate that is not finite did not run.
    ///
    /// # Errors
    ///
    /// Whatever `build` reports, and [`ThroughputError::NoTiming`] where no
    /// shape ran.
    pub(super) fn fastest(
        &self,
        build: impl Fn(S) -> Result<KernelConfig, ThroughputError>,
    ) -> Result<(ThroughputValue, S), ThroughputError> {
        let (fastest, warmed) = match self.shapes.len() {
            0 => return Err(ThroughputError::NoTiming),
            1 => (self.shapes[0], None),
            _ => {
                let (fastest, iterations) = self.ranked(&build)?;

                (fastest, Some(iterations))
            }
        };

        let config = build(fastest)?;
        let value = match warmed {
            Some(iterations) => ThroughputBenchmarker::sample_at(&config, iterations),
            None => ThroughputBenchmarker::sample(config),
        }?;

        value
            .ops_per_s()
            .is_finite()
            .then_some((value, fastest))
            .ok_or(ThroughputError::NoTiming)
    }

    /// The shape that answers fastest over a ranking pass, and the count the
    /// device was warmed at. Every shape is timed at that same count, so they
    /// are ordered on what they do rather than on which was warmed.
    fn ranked(
        &self,
        build: impl Fn(S) -> Result<KernelConfig, ThroughputError>,
    ) -> Result<(S, usize), ThroughputError> {
        let mut warmed = None;
        let mut fastest: Option<(f64, S)> = None;

        for shape in &self.shapes {
            let config = build(*shape)?;
            let start = match warmed {
                Some(iterations) => iterations,
                None => ThroughputBenchmarker::warm(&config)?,
            };
            let ranked = ThroughputBenchmarker::rank(&config, start)?;
            let rate = ranked.value.ops_per_s();

            // The next shape starts from what this one settled on, so a sweep
            // whose shapes drift in cost keeps its settling launch near target.
            warmed = Some(ranked.iterations);

            if rate.is_finite() && fastest.is_none_or(|(best, _)| rate > best) {
                fastest = Some((rate, *shape));
            }
        }

        fastest
            .zip(warmed)
            .map(|((_, shape), iterations)| (shape, iterations))
            .ok_or(ThroughputError::NoTiming)
    }
}
