use super::{AgentEvaluationEvent, EpisodeSummary, EvaluationItem, RLEvent, RLEventProcessor};
use crate::{
    MetricLogger, MetricUpdate, Split, TrainingError,
    logger::TrainingProgressLogger,
    metric::{
        Adaptor, EventStore, Metric, MetricDefinition, MetricMetadata, MetricSnapshot,
        OutputMetric, adapted,
    },
    renderer::{MetricState, MetricsRenderer},
};
use std::collections::HashSet;

pub struct RLMetricsProcessor<T, A> {
    train: Vec<Box<dyn OutputMetric<T>>>,
    action: Vec<Box<dyn OutputMetric<A>>>,
    action_valid: Vec<Box<dyn OutputMetric<A>>>,
    episode: Vec<Box<dyn OutputMetric<EpisodeSummary>>>,
    episode_valid: Vec<Box<dyn OutputMetric<EpisodeSummary>>>,
    store: EventStore,
    loggers: Vec<Box<dyn MetricLogger>>,
    renderers: Vec<Box<dyn MetricsRenderer>>,
    progress: Vec<Box<dyn TrainingProgressLogger>>,
}
impl<T, A> Default for RLMetricsProcessor<T, A> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T, A> RLMetricsProcessor<T, A> {
    pub fn new() -> Self {
        Self {
            train: Vec::new(),
            action: Vec::new(),
            action_valid: Vec::new(),
            episode: Vec::new(),
            episode_valid: Vec::new(),
            store: EventStore::default(),
            loggers: Vec::new(),
            renderers: Vec::new(),
            progress: Vec::new(),
        }
    }
    pub fn store(&self) -> EventStore {
        self.store.clone()
    }
    pub fn with_logger<L: MetricLogger + 'static>(mut self, logger: L) -> Self {
        self.loggers.push(Box::new(logger));
        self
    }
    pub fn with_renderer<R: MetricsRenderer + 'static>(mut self, renderer: R) -> Self {
        self.renderers.push(Box::new(renderer));
        self
    }
    pub fn with_progress_logger<L: TrainingProgressLogger + 'static>(mut self, logger: L) -> Self {
        self.progress.push(Box::new(logger));
        self
    }
    pub fn metric_train<M: Metric + 'static>(mut self, metric: M) -> Self
    where
        T: Adaptor<M::Input> + 'static,
    {
        self.train.push(adapted(metric));
        self
    }
    pub fn metric_agent<M: Metric + Clone + 'static>(self, metric: M) -> Self
    where
        A: Adaptor<M::Input> + 'static,
    {
        self.metric_agent_train(metric.clone())
            .metric_agent_valid(metric)
    }
    pub fn metric_agent_train<M: Metric + 'static>(mut self, metric: M) -> Self
    where
        A: Adaptor<M::Input> + 'static,
    {
        self.action.push(adapted(metric));
        self
    }
    pub fn metric_agent_valid<M: Metric + 'static>(mut self, metric: M) -> Self
    where
        A: Adaptor<M::Input> + 'static,
    {
        self.action_valid.push(adapted(metric));
        self
    }
    pub fn metric_episode<M: Metric + Clone + 'static>(self, metric: M) -> Self
    where
        EpisodeSummary: Adaptor<M::Input>,
    {
        self.metric_episode_train(metric.clone())
            .metric_episode_valid(metric)
    }
    pub fn metric_episode_train<M: Metric + 'static>(mut self, metric: M) -> Self
    where
        EpisodeSummary: Adaptor<M::Input>,
    {
        self.episode.push(adapted(metric));
        self
    }
    pub fn metric_episode_valid<M: Metric + 'static>(mut self, metric: M) -> Self
    where
        EpisodeSummary: Adaptor<M::Input>,
    {
        self.episode_valid.push(adapted(metric));
        self
    }
    fn definitions(&self) -> Vec<MetricDefinition> {
        let mut seen = HashSet::new();
        self.train
            .iter()
            .map(|m| m.definition())
            .chain(self.action.iter().map(|m| m.definition()))
            .chain(self.action_valid.iter().map(|m| m.definition()))
            .chain(self.episode.iter().map(|m| m.definition()))
            .chain(self.episode_valid.iter().map(|m| m.definition()))
            .filter(|definition| seen.insert(definition.metric_id.clone()))
            .collect()
    }
    fn deliver(
        &mut self,
        update: MetricUpdate,
        snapshots: Vec<(bool, MetricSnapshot)>,
        mut errors: Vec<String>,
    ) -> Result<(), TrainingError> {
        let mut stored = self.store.metrics(1, update.split);
        for entry in &update.metrics {
            if let Some(previous) = stored
                .iter_mut()
                .find(|previous| previous.name == entry.name)
            {
                *previous = entry.clone();
            } else {
                stored.push(entry.clone());
            }
        }
        self.store.record(1, update.split, stored);
        let serialized: Vec<_> = snapshots
            .iter()
            .map(|(_, snapshot)| snapshot.clone())
            .collect();
        for logger in &mut self.loggers {
            let result = if logger.needs_snapshots() {
                logger.log_snapshots(&update, &serialized)
            } else {
                logger.log(&update)
            };
            if let Err(error) = result {
                errors.push(error.0);
            }
        }
        for renderer in &mut self.renderers {
            for (numeric, snapshot) in &snapshots {
                let state = if *numeric {
                    MetricState::Numeric(snapshot.clone(), snapshot.current.clone())
                } else {
                    MetricState::Generic(snapshot.clone())
                };
                if update.split == Split::Train {
                    renderer.update_train(state);
                } else {
                    renderer.update_valid(state);
                }
            }
        }
        for logger in &mut self.progress {
            if (update.split == Split::Train && update.event == "EnvStep")
                || (update.split == Split::Valid && update.event == "EpisodeEnd")
            {
                logger.update_split(update.items_processed);
            }
            logger.log_event_training(update.event.clone());
        }
        for renderer in &mut self.renderers {
            if (update.split == Split::Train && update.event == "EnvStep")
                || (update.split == Split::Valid && update.event == "EpisodeEnd")
            {
                renderer.update_split(update.items_processed);
            }
            renderer.log_event_training(update.event.clone());
        }
        finish_errors(errors)
    }
}
fn finish_errors(errors: Vec<String>) -> Result<(), TrainingError> {
    if errors.is_empty() {
        Ok(())
    } else {
        Err(TrainingError(errors.join("\n")))
    }
}
fn update_metrics<T>(
    metrics: &mut [Box<dyn OutputMetric<T>>],
    item: EvaluationItem<T>,
    split: Split,
    event: &str,
) -> (MetricUpdate, Vec<(bool, MetricSnapshot)>, Vec<String>) {
    let metadata = MetricMetadata {
        epoch: 1,
        iteration: item.iteration.unwrap_or(item.progress.items_processed),
        progress: item.progress.clone(),
        learning_rate: None,
    };
    let mut errors = Vec::new();
    let mut entries = Vec::new();
    let mut snapshots = Vec::new();
    for metric in metrics {
        match metric.update(&item.item, &metadata) {
            Ok(()) => {
                entries.push(metric.entry());
                snapshots.push((metric.is_numeric(), metric.snapshot(false)));
            }
            Err(error) => errors.push(error.0),
        }
    }
    (
        MetricUpdate {
            split,
            epoch: 1,
            iteration: metadata.iteration,
            items_processed: item.progress.items_processed,
            items_total: item.progress.items_total,
            learning_rate: None,
            metrics: entries,
            event: event.into(),
        },
        snapshots,
        errors,
    )
}
impl<T, A> RLEventProcessor<T, A> for RLMetricsProcessor<T, A> {
    fn process_train(&mut self, event: RLEvent<T, A>) -> Result<(), TrainingError> {
        let (update, snapshots, errors) = match event {
            RLEvent::Start { total_items, label } => {
                let definitions = self.definitions();
                let mut errors = Vec::new();
                for logger in &mut self.loggers {
                    for definition in &definitions {
                        if let Err(error) = logger.log_metric_definition(definition) {
                            errors.push(error.0);
                        }
                    }
                }
                for renderer in &mut self.renderers {
                    for definition in &definitions {
                        renderer.register_metric(definition.clone());
                    }
                    renderer.start(0, 0, Some(total_items), label.as_deref());
                }
                for logger in &mut self.progress {
                    logger.start(0, 0, Some(total_items), label.as_deref());
                }
                return finish_errors(errors);
            }
            RLEvent::TrainStep(item) => {
                update_metrics(&mut self.train, item, Split::Train, "TrainStep")
            }
            RLEvent::EnvStep(item) => {
                update_metrics(&mut self.action, item, Split::Train, "EnvStep")
            }
            RLEvent::EpisodeEnd(item) => {
                update_metrics(&mut self.episode, item, Split::Train, "EpisodeEnd")
            }
            RLEvent::End(summary) => {
                for logger in &mut self.progress {
                    logger.end();
                }
                for renderer in &mut self.renderers {
                    renderer.end();
                    if let Some(summary) = &summary {
                        renderer.on_train_end_with_summary(summary.clone());
                    } else {
                        renderer.on_train_end();
                    }
                }
                return self.flush();
            }
        };
        self.deliver(update, snapshots, errors)
    }
    fn process_valid(&mut self, event: AgentEvaluationEvent<A>) -> Result<(), TrainingError> {
        let (update, snapshots, errors) = match event {
            AgentEvaluationEvent::Start(episodes) => {
                for logger in &mut self.progress {
                    logger.start_split("valid", episodes);
                }
                for renderer in &mut self.renderers {
                    renderer.start_split("valid", episodes);
                }
                return Ok(());
            }
            AgentEvaluationEvent::EnvStep(item) => {
                update_metrics(&mut self.action_valid, item, Split::Valid, "EnvStep")
            }
            AgentEvaluationEvent::EpisodeEnd(item) => {
                update_metrics(&mut self.episode_valid, item, Split::Valid, "EpisodeEnd")
            }
            AgentEvaluationEvent::End => {
                for logger in &mut self.progress {
                    logger.end_split();
                }
                for renderer in &mut self.renderers {
                    renderer.end_split();
                }
                return self.flush();
            }
        };
        self.deliver(update, snapshots, errors)
    }
    fn flush(&mut self) -> Result<(), TrainingError> {
        let mut errors = Vec::new();
        for logger in &mut self.loggers {
            if let Err(error) = logger.flush() {
                errors.push(error.0);
            }
        }
        finish_errors(errors)
    }
}
