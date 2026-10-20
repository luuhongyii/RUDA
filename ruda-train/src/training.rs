use crate::{
    Checkpointer, EpochState, EventProcessorTraining, FileCheckpointer, InferenceStep, Interrupter,
    Interruption, KeepLastNCheckpoints, Learner, LearnerEvent, LearnerRecord, MetricsProcessor,
    RunEvent, TrainStep, TrainingError, TrainingItem,
    metric::{EarlyStoppingStrategy, EventStore},
};
use ruda_model::{
    data::dataloader::DataLoader,
    module::{AutodiffModule, Module},
    record::FileRecorder,
    tensor::backend::{AutodiffBackend, Backend},
};
use ruda_optim::{GradientsAccumulator, Optimizer, lr_scheduler::LrScheduler};
use std::{path::Path, sync::Arc};

pub type TrainLoader<B, M> = Arc<dyn DataLoader<B, <M as TrainStep<B>>::Input>>;
pub type ValidInput<B, M> = <<M as AutodiffModule<B>>::InnerModule as InferenceStep<
    <B as AutodiffBackend>::InnerBackend,
>>::Input;
pub type ValidOutput<B, M> = <<M as AutodiffModule<B>>::InnerModule as InferenceStep<
    <B as AutodiffBackend>::InnerBackend,
>>::Output;
pub type ValidLoader<B, M> =
    Arc<dyn DataLoader<<B as AutodiffBackend>::InnerBackend, ValidInput<B, M>>>;

pub struct LearningResult<M> {
    /// Non-autodiff model in validation mode.
    pub model: M,
    pub state: EpochState,
    pub metrics: EventStore,
    pub interrupted: Option<Interruption>,
    pub error: Option<TrainingError>,
    pub stopped_early: bool,
}
impl<M: core::fmt::Debug> LearningResult<M> {
    pub fn summary(&self) -> crate::LearnerSummary {
        crate::LearnerSummary::from_store(&self.metrics, self.state.completed_epochs)
            .with_model(format!("{:?}", self.model))
    }
}

/// Single-device supervised training with summed-gradient accumulation.
///
/// Checkpoints are saved only after a complete train/validation epoch. Restore
/// resumes at the following epoch and restores model/optimizer/scheduler state.
/// Dataloader shuffle state and backend RNG state remain owned by the caller.
pub struct SupervisedTraining<B, M, O, S>
where
    B: AutodiffBackend,
    M: TrainStep<B>,
    M::InnerModule: InferenceStep<B::InnerBackend>,
    O: Optimizer<M, B>,
    S: LrScheduler,
{
    train: TrainLoader<B, M>,
    valid: ValidLoader<B, M>,
    device: B::Device,
    epochs: usize,
    accumulation: usize,
    gradient_checkpointing: Option<ruda_model::tensor::GradientCheckpointingStrategy>,
    interrupter: Interrupter,
    processor: Box<dyn EventProcessorTraining<M::Output, ValidOutput<B, M>>>,
    store: EventStore,
    checkpointer: Option<Box<dyn Checkpointer<LearnerRecord<B, M, O, S>>>>,
    restore_epoch: Option<usize>,
    early_stopping: Option<Box<dyn EarlyStoppingStrategy>>,
    label: Option<String>,
    summary: bool,
}

impl<B, M, O, S> SupervisedTraining<B, M, O, S>
where
    B: AutodiffBackend,
    M: TrainStep<B> + 'static,
    M::InnerModule: InferenceStep<B::InnerBackend>,
    O: Optimizer<M, B> + 'static,
    S: LrScheduler + 'static,
{
    pub fn new(train: TrainLoader<B, M>, valid: ValidLoader<B, M>, device: B::Device) -> Self {
        let processor = MetricsProcessor::new();
        let store = processor.store();
        Self {
            train,
            valid,
            device,
            epochs: 1,
            accumulation: 1,
            gradient_checkpointing: None,
            interrupter: Interrupter::new(),
            processor: Box::new(processor),
            store,
            checkpointer: None,
            restore_epoch: None,
            early_stopping: None,
            label: None,
            summary: false,
        }
    }
    pub fn num_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Recompute checkpointable memory-bound activations during backward.
    pub fn gradient_checkpointing(self) -> Self {
        self.with_gradient_checkpointing_strategy(ruda_model::tensor::GradientCheckpointingStrategy::Balanced)
    }

    /// Select the checkpoint policy for model parameters, batches and training-time factories.
    pub fn with_gradient_checkpointing_strategy(mut self, strategy: ruda_model::tensor::GradientCheckpointingStrategy) -> Self {
        self.gradient_checkpointing = Some(strategy);
        self
    }
    pub fn summary(mut self) -> Self {
        self.summary = true;
        self
    }
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
    /// Gradients are summed, not averaged. A partial final window is also applied.
    pub fn grads_accumulation(mut self, accumulation: usize) -> Self {
        assert!(
            accumulation > 0,
            "Gradient accumulation must be greater than zero"
        );
        self.accumulation = accumulation;
        self
    }
    pub fn interrupter(&self) -> Interrupter {
        self.interrupter.clone()
    }
    pub fn with_interrupter(mut self, interrupter: Interrupter) -> Self {
        self.interrupter = interrupter;
        self
    }
    pub fn with_metrics(
        mut self,
        processor: MetricsProcessor<M::Output, ValidOutput<B, M>>,
    ) -> Self {
        self.store = processor.store();
        self.processor = Box::new(processor);
        self
    }
    pub fn with_async_metrics(
        self,
        processor: MetricsProcessor<M::Output, ValidOutput<B, M>>,
    ) -> Self {
        let store = processor.store();
        self.with_event_processor(crate::AsyncProcessorTraining::new(processor), store)
    }
    pub fn with_event_processor<
        P: EventProcessorTraining<M::Output, ValidOutput<B, M>> + 'static,
    >(
        mut self,
        processor: P,
        store: EventStore,
    ) -> Self {
        self.processor = Box::new(processor);
        self.store = store;
        self
    }

    pub fn with_lazy_async_metrics(
        self,
        processor: MetricsProcessor<
            <M::Output as crate::ItemLazy>::Synced,
            <ValidOutput<B, M> as crate::ItemLazy>::Synced,
        >,
    ) -> Self
    where
        M::Output: crate::ItemLazy,
        ValidOutput<B, M>: crate::ItemLazy,
    {
        let store = processor.store();
        self.with_event_processor(crate::AsyncProcessorTraining::with_lazy(processor), store)
    }
    pub fn with_checkpointer<C: Checkpointer<LearnerRecord<B, M, O, S>> + 'static>(
        mut self,
        checkpointer: C,
    ) -> Self {
        self.checkpointer = Some(Box::new(checkpointer));
        self
    }
    pub fn with_file_checkpointer<F: FileRecorder<B> + 'static>(
        self,
        directory: impl AsRef<Path>,
        recorder: F,
        keep: usize,
    ) -> Self {
        let checkpointer = FileCheckpointer::new(directory, recorder, self.device.clone());
        self.with_checkpointer(KeepLastNCheckpoints::new(checkpointer, keep))
    }
    pub fn checkpoint(mut self, epoch: usize) -> Self {
        self.restore_epoch = Some(epoch);
        self
    }
    pub fn early_stopping<E: EarlyStoppingStrategy + 'static>(mut self, strategy: E) -> Self {
        self.early_stopping = Some(Box::new(strategy));
        self
    }

    pub fn fit(self, learner: Learner<B, M, O, S>) -> LearningResult<M::InnerModule> {
        let Some(policy) = self.gradient_checkpointing else { return self.fit_on_device(learner); };
        let device = self.device.clone();
        let context = ruda_model::tensor::AutodiffContext::Enabled(policy);
        B::with_autodiff_creation_context(&device, context, || self.fit_on_device(learner))
    }

    fn fit_on_device(mut self, learner: Learner<B, M, O, S>) -> LearningResult<M::InnerModule> {
        let mut learner = learner.fork(&self.device);
        let mut train = self.train.to_device(&self.device);
        if let Some(policy) = self.gradient_checkpointing {
            let context = ruda_model::tensor::AutodiffContext::Enabled(policy);
            learner.model = learner.model.with_autodiff_context(context);
            train = train.with_autodiff_context(context);
        }
        let valid = self.valid.to_device(&self.device);
        if let Some(epoch) = self.restore_epoch {
            let result = match &mut self.checkpointer {
                Some(checkpointer) => checkpointer.restore(epoch).and_then(|record| {
                    learner.restore(record, &self.device)?;
                    if learner.state.completed_epochs != epoch {
                        return Err(TrainingError(
                            "Checkpoint epoch and training counters differ".into(),
                        ));
                    }
                    Ok(())
                }),
                None => Err(TrainingError(
                    "Restoring an epoch requires a checkpointer".into(),
                )),
            };
            self.interrupter.check(result);
        }
        self.interrupter
            .check(self.processor.process_run(RunEvent::TrainingStart {
                total_epochs: self.epochs,
                starting_epoch: learner.state.completed_epochs.saturating_add(1),
                total_items: None,
                label: self.label.clone(),
            }));
        let mut stopped_early = false;
        for epoch in learner.state.completed_epochs.saturating_add(1)..=self.epochs {
            if self.interrupter.should_stop() {
                break;
            }
            self.interrupter
                .check(self.processor.process_train(LearnerEvent::StartSplit {
                    epoch,
                    total_items: train.num_items(),
                }));
            let trained = self.train_epoch(&mut learner, &train);
            self.interrupter
                .check(self.processor.process_train(LearnerEvent::EndSplit {
                    epoch,
                    completed: trained,
                }));
            if self.interrupter.should_stop() {
                break;
            }

            self.interrupter
                .check(self.processor.process_valid(LearnerEvent::StartSplit {
                    epoch,
                    total_items: valid.num_items(),
                }));
            let model = learner.model.valid();
            let validated = inference_epoch::<B::InnerBackend, _, M::Output>(
                &model,
                &valid,
                self.processor.as_mut(),
                &self.interrupter,
            );
            self.interrupter
                .check(self.processor.process_valid(LearnerEvent::EndSplit {
                    epoch,
                    completed: validated,
                }));
            if self.interrupter.should_stop() {
                break;
            }
            self.interrupter
                .check(self.processor.process_train(LearnerEvent::EndEpoch(epoch)));
            self.interrupter.check(self.processor.flush());
            if self.interrupter.should_stop() {
                break;
            }

            learner.state.completed_epochs = epoch;
            if let Some(checkpointer) = &mut self.checkpointer {
                self.interrupter.check(
                    learner
                        .record()
                        .and_then(|record| checkpointer.save(epoch, record)),
                );
            }
            if self.interrupter.should_stop() {
                break;
            }
            if self
                .early_stopping
                .as_mut()
                .is_some_and(|strategy| strategy.should_stop(epoch, &self.store))
            {
                stopped_early = true;
                break;
            }
        }
        let end = if self.summary {
            RunEvent::TrainingEndWithSummary {
                epochs: learner.state.completed_epochs,
                model: format!("{:?}", learner.model),
            }
        } else {
            RunEvent::TrainingEnd
        };
        self.interrupter.check(self.processor.process_run(end));
        self.interrupter.check(self.processor.flush());
        LearningResult {
            model: learner.model.valid(),
            state: learner.state,
            metrics: self.store,
            interrupted: self.interrupter.interruption(),
            error: self.interrupter.error(),
            stopped_early,
        }
    }

    fn train_epoch(
        &mut self,
        learner: &mut Learner<B, M, O, S>,
        loader: &TrainLoader<B, M>,
    ) -> bool {
        let mut iterator = loader.iter();
        let mut iteration = 0;
        let mut accumulated = 0;
        let mut accumulator = GradientsAccumulator::new();
        while !self.interrupter.should_stop() {
            let Some(input) = iterator.next() else { break };
            iteration += 1;
            let progress = iterator.progress();
            let output = learner.train_step(input);
            accumulator.accumulate::<B>(&learner.model, output.grads);
            accumulated += 1;
            if accumulated == self.accumulation || progress.items_processed >= progress.items_total
            {
                learner.optimizer_step(accumulator.grads());
                accumulated = 0;
            }
            self.interrupter
                .check(
                    self.processor
                        .process_train(LearnerEvent::ProcessedItem(TrainingItem {
                            item: output.item,
                            progress,
                            iteration,
                            learning_rate: Some(learner.state.learning_rate),
                        })),
                );
        }
        if self.interrupter.should_stop() {
            return false;
        }
        if accumulated > 0 {
            learner.optimizer_step(accumulator.grads());
        }
        true
    }
}

pub(crate) fn inference_epoch<B, M, T>(
    model: &M,
    loader: &Arc<dyn DataLoader<B, M::Input>>,
    processor: &mut dyn EventProcessorTraining<T, M::Output>,
    interrupter: &Interrupter,
) -> bool
where
    B: Backend,
    M: InferenceStep<B>,
{
    let mut iterator = loader.iter();
    let mut iteration = 0;
    while !interrupter.should_stop() {
        let Some(input) = iterator.next() else { break };
        iteration += 1;
        let output = model.step(input);
        interrupter.check(
            processor.process_valid(LearnerEvent::ProcessedItem(TrainingItem {
                item: output,
                progress: iterator.progress(),
                iteration,
                learning_rate: None,
            })),
        );
    }
    !interrupter.should_stop()
}

pub struct EvaluationResult {
    pub interrupted: Option<Interruption>,
    pub error: Option<TrainingError>,
    pub completed: bool,
}

/// Evaluates a non-autodiff model supplied in validation mode.
pub fn evaluate<B, M, P>(
    model: M,
    loader: Arc<dyn DataLoader<B, M::Input>>,
    device: &B::Device,
    processor: &mut P,
    interrupter: &Interrupter,
) -> EvaluationResult
where
    B: Backend,
    M: Module<B> + InferenceStep<B>,
    P: EventProcessorTraining<(), M::Output>,
{
    evaluate_named(
        model,
        loader,
        device,
        processor,
        interrupter,
        crate::renderer::EvaluationName::new("test"),
    )
}

pub fn evaluate_named<B, M, P>(
    model: M,
    loader: Arc<dyn DataLoader<B, M::Input>>,
    device: &B::Device,
    processor: &mut P,
    interrupter: &Interrupter,
    name: crate::renderer::EvaluationName,
) -> EvaluationResult
where
    B: Backend,
    M: Module<B> + InferenceStep<B>,
    P: EventProcessorTraining<(), M::Output>,
{
    evaluate_impl(model, loader, device, processor, interrupter, name, false)
}

pub fn evaluate_named_with_summary<B, M, P>(
    model: M,
    loader: Arc<dyn DataLoader<B, M::Input>>,
    device: &B::Device,
    processor: &mut P,
    interrupter: &Interrupter,
    name: crate::renderer::EvaluationName,
) -> EvaluationResult
where
    B: Backend,
    M: Module<B> + InferenceStep<B>,
    P: EventProcessorTraining<(), M::Output>,
{
    evaluate_impl(model, loader, device, processor, interrupter, name, true)
}

fn evaluate_impl<B, M, P>(
    model: M,
    loader: Arc<dyn DataLoader<B, M::Input>>,
    device: &B::Device,
    processor: &mut P,
    interrupter: &Interrupter,
    name: crate::renderer::EvaluationName,
    summary: bool,
) -> EvaluationResult
where
    B: Backend,
    M: Module<B> + InferenceStep<B>,
    P: EventProcessorTraining<(), M::Output>,
{
    let model = model.fork(device);
    let loader = loader.to_device(device);
    interrupter.check(processor.process_run(RunEvent::EvaluationStart {
        total_tests: 1,
        name,
    }));
    interrupter.check(processor.process_valid(LearnerEvent::StartSplit {
        epoch: 1,
        total_items: loader.num_items(),
    }));
    let completed = inference_epoch::<B, M, ()>(&model, &loader, processor, interrupter);
    interrupter.check(processor.process_valid(LearnerEvent::EndSplit {
        epoch: 1,
        completed,
    }));
    let end = if summary {
        RunEvent::EvaluationEndWithSummary {
            model: format!("{model:?}"),
        }
    } else {
        RunEvent::EvaluationEnd
    };
    interrupter.check(processor.process_run(end));
    interrupter.check(processor.flush());
    EvaluationResult {
        completed: completed && !interrupter.should_stop(),
        interrupted: interrupter.interruption(),
        error: interrupter.error(),
    }
}
