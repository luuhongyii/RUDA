use crate::{
    AsyncProcessorTraining, EventProcessorTraining, InferenceStep, Interrupter, Interruption,
    ItemLazy, LearnerEvent, MetricsProcessor, RunEvent, TrainingError, metric::EventStore,
    renderer::EvaluationName,
};
use ruda_model::{data::dataloader::DataLoader, module::Module, tensor::backend::Backend};
use std::{fmt::Display, sync::Arc};

pub type TestLoader<B, M> = Arc<dyn DataLoader<B, <M as InferenceStep<B>>::Input>>;

pub struct EvaluationsResult {
    pub metrics: EventStore,
    pub total_tests: usize,
    pub completed_tests: Vec<EvaluationName>,
    pub interrupted: Option<Interruption>,
    pub error: Option<TrainingError>,
}
impl EvaluationsResult {
    pub fn completed(&self) -> bool {
        self.completed_tests.len() == self.total_tests
            && self.interrupted.is_none()
            && self.error.is_none()
    }
}

pub struct Evaluator<B, M>
where
    B: Backend,
    M: Module<B> + InferenceStep<B>,
{
    model: M,
    device: B::Device,
    processor: Box<dyn EventProcessorTraining<(), M::Output>>,
    store: EventStore,
    interrupter: Interrupter,
    summary: bool,
}
impl<B, M> Evaluator<B, M>
where
    B: Backend,
    M: Module<B> + InferenceStep<B>,
{
    pub fn new(model: M, device: B::Device) -> Self {
        let processor = MetricsProcessor::new().for_evaluation();
        let store = processor.store();
        Self {
            model,
            device,
            processor: Box::new(processor),
            store,
            interrupter: Interrupter::new(),
            summary: false,
        }
    }
    pub fn with_interrupter(mut self, interrupter: Interrupter) -> Self {
        self.interrupter = interrupter;
        self
    }
    pub fn summary(mut self) -> Self {
        self.summary = true;
        self
    }
    pub fn with_event_processor<P>(mut self, processor: P, store: EventStore) -> Self
    where
        P: EventProcessorTraining<(), M::Output> + 'static,
    {
        self.processor = Box::new(processor);
        self.store = store;
        self
    }
    pub fn with_metrics(self, processor: MetricsProcessor<(), M::Output>) -> Self {
        let store = processor.store();
        self.with_event_processor(processor.for_evaluation(), store)
    }
    pub fn with_async_metrics(self, processor: MetricsProcessor<(), M::Output>) -> Self {
        let store = processor.store();
        self.with_event_processor(
            AsyncProcessorTraining::new(processor.for_evaluation()),
            store,
        )
    }
    pub fn with_lazy_async_metrics(
        self,
        processor: MetricsProcessor<(), <M::Output as ItemLazy>::Synced>,
    ) -> Self
    where
        M::Output: ItemLazy,
    {
        let store = processor.store();
        self.with_event_processor(
            AsyncProcessorTraining::<(), M::Output>::with_lazy(processor.for_evaluation()),
            store,
        )
    }
    pub fn eval<S: Display>(self, name: S, loader: TestLoader<B, M>) -> EvaluationsResult {
        self.eval_all([(name, loader)])
    }
    pub fn eval_all<S: Display>(
        mut self,
        splits: impl IntoIterator<Item = (S, TestLoader<B, M>)>,
    ) -> EvaluationsResult {
        let splits: Vec<_> = splits.into_iter().collect();
        let total_tests = splits.len();
        let mut completed_tests = Vec::new();
        let model = self.model.fork(&self.device);
        self.interrupter.check(
            self.processor
                .process_run(RunEvent::EvaluationsStart { total_tests }),
        );
        for (name, loader) in splits {
            if self.interrupter.should_stop() {
                break;
            }
            let name = EvaluationName::new(name);
            let loader = loader.to_device(&self.device);
            self.interrupter.check(
                self.processor
                    .process_run(RunEvent::EvaluationSelect { name: name.clone() }),
            );
            self.interrupter
                .check(self.processor.process_valid(LearnerEvent::StartSplit {
                    epoch: 1,
                    total_items: loader.num_items(),
                }));
            let completed = crate::training::inference_epoch::<B, M, ()>(
                &model,
                &loader,
                self.processor.as_mut(),
                &self.interrupter,
            );
            self.interrupter
                .check(self.processor.process_valid(LearnerEvent::EndSplit {
                    epoch: 1,
                    completed,
                }));
            self.interrupter.check(self.processor.flush());
            if completed && self.interrupter.error().is_none() {
                completed_tests.push(name);
            }
            if self.interrupter.should_stop() {
                break;
            }
        }
        self.interrupter
            .check(self.processor.process_run(RunEvent::EvaluationsEnd {
                model: self.summary.then(|| format!("{model:?}")),
            }));
        self.interrupter.check(self.processor.flush());
        EvaluationsResult {
            metrics: self.store,
            total_tests,
            completed_tests,
            interrupted: self.interrupter.interruption(),
            error: self.interrupter.error(),
        }
    }
}
