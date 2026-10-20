use crate::{TrainOutput, TrainStep};
use ruda_model::{
    module::AutodiffModule,
    record::{PrecisionSettings, Record},
    tensor::backend::{AutodiffBackend, Backend},
};
use ruda_optim::{
    GradientsAccumulator, GradientsParams, LearningRate, Optimizer, lr_scheduler::LrScheduler,
    training::TrainingRecord,
};
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    marker::PhantomData,
    sync::{Arc, Mutex},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrainingError(pub String);

impl fmt::Display for TrainingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for TrainingError {}
impl From<ruda_model::record::RecorderError> for TrainingError {
    fn from(error: ruda_model::record::RecorderError) -> Self {
        Self(error.to_string())
    }
}
impl From<std::io::Error> for TrainingError {
    fn from(error: std::io::Error) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Interruption {
    pub reason: Option<String>,
}

#[derive(Default)]
struct StopState {
    interruption: Option<Interruption>,
    error: Option<TrainingError>,
}

/// A shared stop handle. Errors take precedence over an ordinary stop request.
#[derive(Clone, Default)]
pub struct Interrupter {
    state: Arc<Mutex<StopState>>,
}

impl Interrupter {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn stop(&self, reason: Option<&str>) {
        let mut state = self.state.lock().unwrap();
        if state.interruption.is_none() || reason.is_some() {
            state.interruption = Some(Interruption {
                reason: reason.map(str::to_owned),
            });
        }
    }
    pub fn fail(&self, error: TrainingError) {
        self.state.lock().unwrap().error.get_or_insert(error);
    }
    pub fn should_stop(&self) -> bool {
        let state = self.state.lock().unwrap();
        state.interruption.is_some() || state.error.is_some()
    }
    pub fn interruption(&self) -> Option<Interruption> {
        let state = self.state.lock().unwrap();
        if state.error.is_some() {
            None
        } else {
            state.interruption.clone()
        }
    }
    pub fn error(&self) -> Option<TrainingError> {
        self.state.lock().unwrap().error.clone()
    }
    pub(crate) fn check(&self, result: Result<(), TrainingError>) {
        if let Err(error) = result {
            self.fail(error);
        }
    }
}

/// Counters at a completed epoch boundary, not a batch cursor or RNG snapshot.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EpochState {
    pub completed_epochs: usize,
    pub optimizer_steps: usize,
    pub learning_rate: LearningRate,
}
impl<B: Backend> Record<B> for EpochState {
    type Item<P: PrecisionSettings> = Self;
    fn into_item<P: PrecisionSettings>(self) -> Self {
        self
    }
    fn from_item<P: PrecisionSettings>(item: Self, _: &B::Device) -> Self {
        item
    }
}

pub type LearnerRecord<B, M, O, S> = TrainingRecord<B, M, O, S, EpochState>;

/// Owns the trainable model, optimizer and learning-rate scheduler.
pub struct Learner<B, M, O, S>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: Optimizer<M, B>,
    S: LrScheduler,
{
    pub(crate) model: M,
    pub(crate) optim: O,
    pub(crate) scheduler: S,
    pub(crate) state: EpochState,
    pub(crate) accumulator: GradientsAccumulator<M>,
    marker: PhantomData<B>,
}

impl<B, M, O, S> Learner<B, M, O, S>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: Optimizer<M, B>,
    S: LrScheduler,
{
    pub fn new(model: M, optim: O, scheduler: S) -> Self {
        Self {
            model,
            optim,
            scheduler,
            state: EpochState::default(),
            accumulator: GradientsAccumulator::new(),
            marker: PhantomData,
        }
    }
    pub fn model(&self) -> &M {
        &self.model
    }
    pub fn state(&self) -> &EpochState {
        &self.state
    }
    pub fn fork(self, device: &B::Device) -> Self {
        let accumulator = self.accumulator.to_device::<B>(device, &self.model);
        Self {
            model: self.model.fork(device),
            accumulator,
            ..self
        }
    }
    pub fn into_parts(self) -> (M, O, S) {
        (self.model, self.optim, self.scheduler)
    }
    pub(crate) fn record(&self) -> Result<LearnerRecord<B, M, O, S>, TrainingError> {
        Ok(TrainingRecord::capture(
            &self.model,
            &self.optim,
            &self.scheduler,
            &self.accumulator,
            self.state.clone(),
        )?)
    }
    pub(crate) fn restore(
        &mut self,
        record: LearnerRecord<B, M, O, S>,
        device: &B::Device,
    ) -> Result<(), TrainingError> {
        let restored = record.restore(
            self.model.clone(),
            self.optim.clone(),
            self.scheduler.clone(),
            device,
        )?;
        self.model = restored.model;
        self.optim = restored.optimizer;
        self.scheduler = restored.scheduler;
        self.state = restored.state;
        self.accumulator = restored.accumulator;
        Ok(())
    }

    pub(crate) fn batch_record(
        &self, state: crate::BatchContinuation,
    ) -> Result<crate::BatchLearnerRecord<B, M, O, S>, TrainingError> {
        Ok(crate::BatchLearnerRecord(TrainingRecord::capture_with_dtypes(
            &self.model, &self.optim, &self.scheduler, &self.accumulator, state,
        )?))
    }

    pub(crate) fn restore_batch(
        &mut self, record: crate::BatchLearnerRecord<B, M, O, S>, device: &B::Device,
    ) -> Result<crate::BatchContinuation, TrainingError> {
        let restored = record.0.restore_with_dtypes(
            self.model.clone(), self.optim.clone(), self.scheduler.clone(), device,
        )?;
        self.model = restored.model;
        self.optim = restored.optimizer;
        self.scheduler = restored.scheduler;
        self.accumulator = restored.accumulator;
        self.state = restored.state.counters.clone();
        Ok(restored.state)
    }
}

impl<B, M, O, S> Learner<B, M, O, S>
where
    B: AutodiffBackend,
    M: TrainStep<B>,
    O: Optimizer<M, B>,
    S: LrScheduler,
{
    pub fn train_step(&self, item: M::Input) -> TrainOutput<M::Output> {
        self.model.step(item)
    }
    pub fn optimizer_step(&mut self, grads: GradientsParams) {
        self.state.learning_rate = self.scheduler.step();
        self.model = self
            .model
            .clone()
            .optimize(&mut self.optim, self.state.learning_rate, grads);
        self.state.optimizer_steps += 1;
    }
}
