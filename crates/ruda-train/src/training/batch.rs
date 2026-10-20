use super::*;
use crate::{BatchContinuation, ContinuationPhase};

impl<B, M, O, S> SupervisedTraining<B, M, O, S>
where
    B: AutodiffBackend,
    M: TrainStep<B> + 'static,
    M::InnerModule: InferenceStep<B::InnerBackend>,
    O: Optimizer<M, B> + 'static,
    S: LrScheduler + 'static,
{
    pub(super) fn fit_batches(mut self, learner: Learner<B, M, O, S>) -> LearningResult<M::InnerModule> {
        let mut learner = learner.fork(&self.device);
        let mut train = self.train.to_device(&self.device);
        if let Some(policy) = self.gradient_checkpointing {
            let context = ruda_model::tensor::AutodiffContext::Enabled(policy);
            learner.model = learner.model.with_autodiff_context(context);
            train = train.with_autodiff_context(context);
        }
        let valid = self.valid.to_device(&self.device);
        let mut state = BatchContinuation {
            version: 1, epoch: learner.state.completed_epochs.saturating_add(1),
            phase: ContinuationPhase::Train, counters: learner.state.clone(),
            batches: 0, iteration: 0, accumulated: 0, accumulation: self.accumulation,
            train: None, valid: None, train_source: Vec::new(), valid_source: Vec::new(),
            backend_rng: Vec::new(), inner_backend_rng: Vec::new(),
            processor: Vec::new(), event_store: Vec::new(), early_stopping: None,
            stopped_early: false,
        };
        let result = (|| {
            if self.restore_epoch.is_some() {
                return Err(TrainingError("Select either epoch restore or exact batch continuation".into()));
            }
            if self.batch_checkpointer.is_none() {
                return Err(TrainingError("Batch continuation requires a batch checkpointer".into()));
            }
            if let Some(batch) = self.restore_batch {
                let record = self.batch_checkpointer.as_mut().unwrap().restore(batch)?;
                state = learner.restore_batch(record, &self.device)?;
                if state.version != 1 || state.batches != batch
                    || state.accumulation != self.accumulation
                    || state.accumulated >= self.accumulation
                    || state.epoch == 0
                { return Err(TrainingError("Batch continuation configuration or counters differ".into())); }
                train.restore_source(&state.train_source).map_err(TrainingError)?;
                valid.restore_source(&state.valid_source).map_err(TrainingError)?;
                self.processor.restore_checkpoint(&state.processor)?;
                self.store.restore_checkpoint(&state.event_store)?;
                match (self.early_stopping.as_mut(), state.early_stopping.as_deref()) {
                    (Some(strategy), Some(saved)) => strategy.restore_checkpoint(saved)?,
                    (None, None) => {}
                    _ => return Err(TrainingError("Early stopping registration differs from checkpoint".into())),
                }
            }
            self.processor.process_run(RunEvent::TrainingStart {
                total_epochs: self.epochs, starting_epoch: state.epoch,
                total_items: Some(train.num_items()), label: self.label.clone(),
            })?;
            if self.restore_batch.is_some() {
                self.processor.flush()?;
                B::restore_rng_state(&self.device, &state.backend_rng)
                    .map_err(|error| TrainingError(error.to_string()))?;
                B::InnerBackend::restore_rng_state(&self.device, &state.inner_backend_rng)
                    .map_err(|error| TrainingError(error.to_string()))?;
            } else {
                self.save_batch(&learner, &train, &valid, &mut state)?;
            }
            self.run_batches(&mut learner, &train, &valid, &mut state)
        })();
        self.interrupter.check(result);
        let end = if self.summary {
            RunEvent::TrainingEndWithSummary {
                epochs: learner.state.completed_epochs, model: format!("{:?}", learner.model),
            }
        } else { RunEvent::TrainingEnd };
        self.interrupter.check(self.processor.process_run(end));
        self.interrupter.check(self.processor.flush());
        LearningResult {
            model: learner.model.valid(), state: learner.state, metrics: self.store,
            interrupted: self.interrupter.interruption(), error: self.interrupter.error(),
            stopped_early: state.stopped_early,
        }
    }

    fn save_batch(
        &mut self, learner: &Learner<B, M, O, S>,
        train: &TrainLoader<B, M>, valid: &ValidLoader<B, M>, state: &mut BatchContinuation,
    ) -> Result<(), TrainingError> {
        self.processor.flush()?;
        state.processor = self.processor.checkpoint()?;
        state.event_store = self.store.checkpoint()?;
        state.early_stopping = self.early_stopping.as_ref()
            .map(|strategy| strategy.checkpoint()).transpose()?;
        state.train_source = train.checkpoint_source().map_err(TrainingError)?;
        state.valid_source = valid.checkpoint_source().map_err(TrainingError)?;
        state.counters = learner.state.clone();
        let mut record = learner.batch_record(state.clone())?;
        B::sync(&self.device).map_err(|error| TrainingError(error.to_string()))?;
        B::InnerBackend::sync(&self.device).map_err(|error| TrainingError(error.to_string()))?;
        state.backend_rng = B::rng_state(&self.device)
            .map_err(|error| TrainingError(error.to_string()))?;
        state.inner_backend_rng = B::InnerBackend::rng_state(&self.device)
            .map_err(|error| TrainingError(error.to_string()))?;
        record.0.state_mut().1 = state.clone();
        self.batch_checkpointer.as_mut().unwrap().save(state.batches, record)
    }

    fn batch_boundary(
        &mut self, learner: &Learner<B, M, O, S>, train: &TrainLoader<B, M>,
        valid: &ValidLoader<B, M>, state: &mut BatchContinuation,
    ) -> Result<(), TrainingError> {
        if let Some(error) = self.interrupter.error() { return Err(error); }
        if state.batches % self.batch_interval == 0 || self.interrupter.should_stop() {
            self.save_batch(learner, train, valid, state)?;
        }
        Ok(())
    }

    fn run_batches(
        &mut self, learner: &mut Learner<B, M, O, S>, train: &TrainLoader<B, M>,
        valid: &ValidLoader<B, M>, state: &mut BatchContinuation,
    ) -> Result<(), TrainingError> {
        while state.epoch <= self.epochs && state.phase != ContinuationPhase::Finished {
            if self.interrupter.should_stop() {
                if let Some(error) = self.interrupter.error() { return Err(error); }
                self.save_batch(learner, train, valid, state)?;
                return Ok(());
            }
            match state.phase {
                ContinuationPhase::Train => {
                    let mut iterator = train.iter_from_checkpoint(state.train.as_ref()).map_err(TrainingError)?;
                    if state.train.is_none() {
                        state.iteration = 0;
                        self.processor.process_train(LearnerEvent::StartSplit {
                            epoch: state.epoch, total_items: train.num_items(),
                        })?;
                    }
                    state.train = Some(iterator.checkpoint().map_err(TrainingError)?);
                    while !self.interrupter.should_stop() {
                        let Some(input) = iterator.next() else { break };
                        let output = learner.train_step(input);
                        learner.accumulator.accumulate::<B>(&learner.model, output.grads);
                        state.accumulated += 1;
                        let progress = iterator.progress();
                        if state.accumulated == self.accumulation || progress.items_processed >= progress.items_total {
                            let gradients = learner.accumulator.grads();
                            learner.optimizer_step(gradients);
                            state.accumulated = 0;
                        }
                        state.iteration += 1;
                        state.batches += 1;
                        self.processor.process_train(LearnerEvent::ProcessedItem(TrainingItem {
                            item: output.item, progress, iteration: state.iteration,
                            learning_rate: Some(learner.state.learning_rate),
                        }))?;
                        state.train = Some(iterator.checkpoint().map_err(TrainingError)?);
                        self.batch_boundary(learner, train, valid, state)?;
                    }
                    if self.interrupter.should_stop() { continue; }
                    if state.accumulated != 0 {
                        let gradients = learner.accumulator.grads();
                        learner.optimizer_step(gradients);
                        state.accumulated = 0;
                    }
                    self.processor.process_train(LearnerEvent::EndSplit { epoch: state.epoch, completed: true })?;
                    state.phase = ContinuationPhase::Valid;
                    state.iteration = 0;
                }
                ContinuationPhase::Valid => {
                    let model = learner.model.valid();
                    let mut iterator = valid.iter_from_checkpoint(state.valid.as_ref()).map_err(TrainingError)?;
                    if state.valid.is_none() {
                        state.iteration = 0;
                        self.processor.process_valid(LearnerEvent::StartSplit {
                            epoch: state.epoch, total_items: valid.num_items(),
                        })?;
                    }
                    state.valid = Some(iterator.checkpoint().map_err(TrainingError)?);
                    while !self.interrupter.should_stop() {
                        let Some(input) = iterator.next() else { break };
                        let output = model.step(input);
                        state.iteration += 1;
                        state.batches += 1;
                        self.processor.process_valid(LearnerEvent::ProcessedItem(TrainingItem {
                            item: output, progress: iterator.progress(), iteration: state.iteration,
                            learning_rate: None,
                        }))?;
                        state.valid = Some(iterator.checkpoint().map_err(TrainingError)?);
                        self.batch_boundary(learner, train, valid, state)?;
                    }
                    if self.interrupter.should_stop() { continue; }
                    self.processor.process_valid(LearnerEvent::EndSplit { epoch: state.epoch, completed: true })?;
                    state.phase = ContinuationPhase::EpochEnd;
                }
                ContinuationPhase::EpochEnd => {
                    self.processor.process_train(LearnerEvent::EndEpoch(state.epoch))?;
                    self.processor.flush()?;
                    learner.state.completed_epochs = state.epoch;
                    state.stopped_early = self.early_stopping.as_mut()
                        .is_some_and(|strategy| strategy.should_stop(state.epoch, &self.store));
                    if let Some(checkpointer) = &mut self.checkpointer {
                        checkpointer.save(state.epoch, learner.record()?)?;
                    }
                    state.epoch += 1;
                    state.train = None;
                    state.valid = None;
                    state.iteration = 0;
                    state.phase = if state.stopped_early || state.epoch > self.epochs {
                        ContinuationPhase::Finished
                    } else { ContinuationPhase::Train };
                    self.save_batch(learner, train, valid, state)?;
                }
                ContinuationPhase::Finished => break,
            }
        }
        Ok(())
    }
}
