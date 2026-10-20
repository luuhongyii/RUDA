use crate::{Split, TrainingError};
use ruda_model::tensor::{ElementConversion, Int, Tensor, backend::Backend};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    marker::PhantomData,
    sync::{Arc, Mutex},
};

#[macro_use]
mod numeric;
pub use numeric::*;
mod classification;
mod sequence;
mod timing;
#[cfg(feature = "vision")]
pub mod vision;
pub use classification::*;
pub use sequence::*;
pub use timing::*;
#[cfg(feature = "rl")]
mod rl;
#[cfg(feature = "rl")]
pub use rl::*;
#[cfg(feature = "sys-metrics")]
mod system;
#[cfg(feature = "sys-metrics")]
pub use system::*;

/// Converts a model output into the input of a registered metric.
pub trait Adaptor<I> {
    fn adapt(&self) -> I;
}

impl<T> Adaptor<()> for T {
    fn adapt(&self) {}
}

#[derive(Clone, Debug)]
pub struct MetricMetadata {
    pub epoch: usize,
    pub iteration: usize,
    pub progress: ruda_model::data::dataloader::Progress,
    pub learning_rate: Option<f64>,
}

pub trait Metric: Send {
    type Input;
    fn is_numeric(&self) -> bool { true }
    fn name(&self) -> &str;
    fn description(&self) -> Option<String> { None }
    fn attributes(&self) -> MetricAttributes { MetricAttributes::None }
    fn update(&mut self, input: &Self::Input) -> Result<(), TrainingError>;
    fn update_with_metadata(
        &mut self,
        input: &Self::Input,
        _: &MetricMetadata,
    ) -> Result<(), TrainingError> {
        self.update(input)
    }
    /// Finalize metrics that require the complete split, before storing epoch values.
    fn compute(&mut self) -> Result<(), TrainingError> {
        Ok(())
    }
    fn value(&self) -> Option<f64>;
    fn current_value(&self) -> Option<NumericEntry> { None }
    fn running_value(&self) -> Option<NumericEntry> {
        self.value().map(NumericEntry::Value)
    }
    fn final_value(&self) -> NumericEntry {
        NumericEntry::Final(self.value().unwrap_or(f64::NAN))
    }
    fn format_options(&self) -> FormatOptions {
        let format = FormatOptions::new(Arc::new(self.name().into()));
        match self.attributes() {
            MetricAttributes::Numeric(NumericAttributes { unit: Some(unit), .. }) => format.unit(&unit),
            _ => format,
        }
    }
    fn serialized_update(&self) -> SerializedEntry {
        numeric::serialize_values(self.current_value(), self.running_value(), None, self.format_options())
    }
    fn serialized_final(&self) -> SerializedEntry {
        numeric::serialize_values(self.current_value(), self.running_value(), Some(self.final_value()), self.format_options())
    }
    fn definition(&self) -> MetricDefinition {
        MetricDefinition::new(MetricId::new(Arc::new(self.name().into())), self)
    }
    fn snapshot(&self, finalized: bool) -> MetricSnapshot {
        MetricSnapshot {
            definition: self.definition(),
            current: self.current_value(),
            running: self.running_value(),
            final_value: finalized.then(|| self.final_value()),
            serialized: if finalized { self.serialized_final() } else { self.serialized_update() },
        }
    }
    fn clear(&mut self);
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetricEntry {
    pub name: String,
    pub value: Option<f64>,
}

pub struct LossInput<B: Backend> {
    pub tensor: Tensor<B, 1>,
}
pub struct LossMetric<B: Backend> {
    state: NumericMetricState,
    marker: PhantomData<B>,
}
impl<B: Backend> Default for LossMetric<B> {
    fn default() -> Self {
        Self::new()
    }
}
impl<B: Backend> LossMetric<B> {
    pub fn new() -> Self {
        Self {
            state: NumericMetricState::default(),
            marker: PhantomData,
        }
    }
}
impl<B: Backend> Metric for LossMetric<B> {
    numeric_state_metric!(None, false, 2);
    type Input = LossInput<B>;
    fn name(&self) -> &str {
        "Loss"
    }
    fn update(&mut self, input: &Self::Input) -> Result<(), TrainingError> {
        let [weight] = input.tensor.dims();
        if weight > 0 {
            let value = input
                .tensor
                .clone()
                .mean()
                .try_into_scalar()
                .map_err(|error| TrainingError(error.to_string()))?
                .elem::<f64>();
            self.state.update(value, weight);
        }
        Ok(())
    }
    fn value(&self) -> Option<f64> {
        self.state.value()
    }
    fn clear(&mut self) {
        self.state.clear();
    }
}

pub struct AccuracyInput<B: Backend> {
    pub predictions: Tensor<B, 2>,
    pub targets: Tensor<B, 1, Int>,
}
pub struct AccuracyMetric<B: Backend> {
    state: NumericMetricState,
    marker: PhantomData<B>,
    pad_token: Option<usize>,
}
impl<B: Backend> Default for AccuracyMetric<B> {
    fn default() -> Self {
        Self::new()
    }
}
impl<B: Backend> AccuracyMetric<B> {
    pub fn new() -> Self {
        Self {
            state: NumericMetricState::default(),
            marker: PhantomData,
            pad_token: None,
        }
    }
    pub fn with_pad_token(mut self, index: usize) -> Self {
        self.pad_token = Some(index);
        self
    }
}
impl<B: Backend> Metric for AccuracyMetric<B> {
    numeric_state_metric!(Some("%"), true, 2);
    type Input = AccuracyInput<B>;
    fn name(&self) -> &str {
        "Accuracy"
    }
    fn update(&mut self, input: &Self::Input) -> Result<(), TrainingError> {
        let [weight, _] = input.predictions.dims();
        if input.targets.dims() != [weight] {
            return Err(TrainingError(
                "Accuracy predictions and targets have different batch sizes".into(),
            ));
        }
        if weight > 0 {
            let mut matches = input
                .predictions
                .clone()
                .argmax(1)
                .squeeze_dim::<1>(1)
                .equal(input.targets.clone())
                .float();
            let mut valid = weight;
            if let Some(pad) = self.pad_token {
                let mask = input.targets.clone().equal_elem(pad as i64);
                valid -= mask
                    .clone()
                    .int()
                    .sum()
                    .try_into_scalar()
                    .map_err(|error| TrainingError(error.to_string()))?
                    .elem::<i64>() as usize;
                matches = matches.mask_fill(mask, 0.);
            }
            if valid > 0 {
                self.state
                    .update(100. * scalar(matches.sum())? / valid as f64, valid);
            }
        }
        Ok(())
    }
    fn value(&self) -> Option<f64> {
        self.state.value()
    }
    fn clear(&mut self) {
        self.state.clear();
    }
}

/// Shared epoch aggregates. No tensor graphs or per-batch outputs are retained.
#[derive(Clone, Default)]
pub struct EventStore {
    entries: Arc<Mutex<BTreeMap<(usize, Split), Vec<MetricEntry>>>>,
    named_tests: Arc<Mutex<BTreeMap<(usize, String), Vec<MetricEntry>>>>,
}
impl EventStore {
    pub fn test_metrics(&self, epoch: usize, name: &str) -> Vec<MetricEntry> {
        self.named_tests
            .lock()
            .unwrap()
            .get(&(epoch, name.into()))
            .cloned()
            .unwrap_or_default()
    }
    pub fn find_test_metric(&self, metric: &str, epoch: usize, name: &str) -> Option<f64> {
        self.test_metrics(epoch, name)
            .iter()
            .find(|entry| entry.name == metric)
            .and_then(|entry| entry.value)
    }
    pub(crate) fn record_test(&self, epoch: usize, name: &str, metrics: Vec<MetricEntry>) {
        self.named_tests.lock().unwrap().insert((epoch, name.into()), metrics);
    }
    pub(crate) fn snapshot(&self) -> BTreeMap<(usize, Split), Vec<MetricEntry>> {
        self.entries.lock().unwrap().clone()
    }
    pub fn metrics(&self, epoch: usize, split: Split) -> Vec<MetricEntry> {
        self.entries
            .lock()
            .unwrap()
            .get(&(epoch, split))
            .cloned()
            .unwrap_or_default()
    }
    pub fn find_metric(&self, name: &str, epoch: usize, split: Split) -> Option<f64> {
        self.metrics(epoch, split)
            .iter()
            .find(|entry| entry.name == name)
            .and_then(|entry| entry.value)
    }
    pub(crate) fn record(&self, epoch: usize, split: Split, metrics: Vec<MetricEntry>) {
        self.entries.lock().unwrap().insert((epoch, split), metrics);
    }
}

pub trait EarlyStoppingStrategy: Send {
    fn should_stop(&mut self, epoch: usize, store: &EventStore) -> bool;
}
#[derive(Clone, Copy)]
pub enum Direction {
    Lowest,
    Highest,
}
#[derive(Clone)]
pub struct MetricEarlyStoppingStrategy {
    name: String,
    split: Split,
    direction: Direction,
    patience: usize,
    warmup: Option<usize>,
    best_epoch: usize,
    best_value: f64,
}
impl MetricEarlyStoppingStrategy {
    pub fn new(
        name: impl Into<String>,
        split: Split,
        direction: Direction,
        patience: usize,
    ) -> Self {
        Self {
            name: name.into(),
            split,
            direction,
            patience,
            warmup: None,
            best_epoch: 1,
            best_value: match direction {
                Direction::Lowest => f64::MAX,
                Direction::Highest => f64::MIN,
            },
        }
    }
    pub fn with_warmup_epochs(mut self, warmup: Option<usize>) -> Self {
        self.warmup = warmup;
        self
    }
}
impl EarlyStoppingStrategy for MetricEarlyStoppingStrategy {
    fn should_stop(&mut self, epoch: usize, store: &EventStore) -> bool {
        let Some(value) = store.find_metric(&self.name, epoch, self.split) else {
            return false;
        };
        let better = match self.direction {
            Direction::Lowest => value < self.best_value,
            Direction::Highest => value > self.best_value,
        };
        if better {
            self.best_value = value;
            self.best_epoch = epoch;
            return false;
        }
        if self.warmup.is_some_and(|warmup| epoch <= warmup) {
            return false;
        }
        epoch.saturating_sub(self.best_epoch) >= self.patience
    }
}

pub(crate) trait OutputMetric<T>: Send {
    fn is_numeric(&self) -> bool;
    fn definition(&self) -> MetricDefinition;
    fn update(&mut self, output: &T, metadata: &MetricMetadata) -> Result<(), TrainingError>;
    fn compute(&mut self) -> Result<(), TrainingError>;
    fn entry(&self) -> MetricEntry;
    fn snapshot(&self, finalized: bool) -> MetricSnapshot;
    fn clear(&mut self);
}
struct AdaptedMetric<T, M> {
    metric: M,
    marker: PhantomData<fn(T)>,
}
impl<T: Adaptor<M::Input>, M: Metric> OutputMetric<T> for AdaptedMetric<T, M> {
    fn is_numeric(&self) -> bool { self.metric.is_numeric() }
    fn definition(&self) -> MetricDefinition { self.metric.definition() }
    fn snapshot(&self, finalized: bool) -> MetricSnapshot {
        self.metric.snapshot(finalized)
    }
    fn update(&mut self, output: &T, metadata: &MetricMetadata) -> Result<(), TrainingError> {
        self.metric.update_with_metadata(&output.adapt(), metadata)
    }
    fn compute(&mut self) -> Result<(), TrainingError> {
        self.metric.compute()
    }
    fn entry(&self) -> MetricEntry {
        MetricEntry {
            name: self.metric.name().into(),
            value: self.metric.value(),
        }
    }
    fn clear(&mut self) {
        self.metric.clear();
    }
}

fn scalar<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Result<f64, TrainingError> {
    Ok(tensor
        .try_into_scalar()
        .map_err(|error| TrainingError(error.to_string()))?
        .elem::<f64>())
}
pub(crate) fn adapted<T: Adaptor<M::Input> + 'static, M: Metric + 'static>(
    metric: M,
) -> Box<dyn OutputMetric<T>> {
    Box::new(AdaptedMetric {
        metric,
        marker: PhantomData,
    })
}
