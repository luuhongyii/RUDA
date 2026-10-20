use super::{Metric, MetricMetadata, NumericMetricState};
use crate::TrainingError;
use std::time::Instant;

#[derive(Default)]
pub struct LearningRateMetric {
    state: NumericMetricState,
}
impl LearningRateMetric {
    pub fn new() -> Self {
        Self::default()
    }
}
impl Metric for LearningRateMetric {
    numeric_state_metric!(None, false, 2);
    type Input = ();
    fn name(&self) -> &str {
        "Learning Rate"
    }
    fn update(&mut self, _: &()) -> Result<(), TrainingError> {
        self.state.update(0., 1);
        Ok(())
    }
    fn update_with_metadata(
        &mut self,
        _: &(),
        metadata: &MetricMetadata,
    ) -> Result<(), TrainingError> {
        self.state.update(metadata.learning_rate.unwrap_or(0.), 1);
        Ok(())
    }
    fn value(&self) -> Option<f64> {
        self.state.value()
    }
    fn clear(&mut self) {
        self.state.clear();
    }
}

#[derive(Default)]
pub struct IterationSpeedMetric {
    state: NumericMetricState,
    instant: Option<Instant>,
}
impl IterationSpeedMetric {
    pub fn new() -> Self {
        Self::default()
    }
    fn tick(&mut self, iteration: usize) {
        let raw = match self.instant {
            Some(instant) => iteration as f64 / instant.elapsed().as_secs_f64(),
            None => {
                self.instant = Some(Instant::now());
                0.
            }
        };
        self.state.update(raw, 1);
    }
}
impl Metric for IterationSpeedMetric {
    numeric_state_metric!(Some("iter/sec"), true, 2);
    type Input = ();
    fn name(&self) -> &str {
        "Iteration Speed"
    }
    fn update(&mut self, _: &()) -> Result<(), TrainingError> {
        self.tick(0);
        Ok(())
    }
    fn update_with_metadata(
        &mut self,
        _: &(),
        metadata: &MetricMetadata,
    ) -> Result<(), TrainingError> {
        self.tick(metadata.iteration);
        Ok(())
    }
    fn value(&self) -> Option<f64> {
        self.state.value()
    }
    fn clear(&mut self) {
        self.instant = None;
    }
}
