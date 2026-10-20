use crate::{
    TrainingError,
    logger::{EvaluationProgressLogger, TrainingProgressLogger},
    metric::{
        Adaptor, EventStore, Metric, MetricDefinition, MetricEntry, MetricId, MetricMetadata,
        MetricSnapshot, NumericEntry, OutputMetric, adapted,
    },
    renderer::{EvaluationName, MetricState, MetricsRenderer},
};
use ruda_model::data::dataloader::Progress;
use ruda_optim::LearningRate;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::Path,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Split {
    Train,
    Valid,
    Test,
}

pub struct TrainingItem<T> {
    pub item: T,
    pub progress: Progress,
    pub iteration: usize,
    pub learning_rate: Option<LearningRate>,
}
pub enum LearnerEvent<T> {
    StartSplit { epoch: usize, total_items: usize },
    ProcessedItem(TrainingItem<T>),
    EndSplit { epoch: usize, completed: bool },
    EndEpoch(usize),
}

#[derive(Clone, Debug)]
pub enum RunEvent {
    TrainingStart {
        total_epochs: usize,
        starting_epoch: usize,
        total_items: Option<usize>,
        label: Option<String>,
    },
    TrainingEnd,
    TrainingEndWithSummary {
        epochs: usize,
        model: String,
    },
    EvaluationStart {
        total_tests: usize,
        name: EvaluationName,
    },
    EvaluationEnd,
    EvaluationEndWithSummary {
        model: String,
    },
    EvaluationsStart {
        total_tests: usize,
    },
    EvaluationSelect {
        name: EvaluationName,
    },
    EvaluationsEnd {
        model: Option<String>,
    },
    TrainingMessage(String),
    EvaluationMessage(String),
}

pub trait EventProcessorTraining<T, V>: Send {
    fn checkpoint(&mut self) -> Result<Vec<u8>, TrainingError> {
        Err(TrainingError("This event processor does not support continuation".into()))
    }
    fn restore_checkpoint(&mut self, _: &[u8]) -> Result<(), TrainingError> {
        Err(TrainingError("This event processor does not support continuation".into()))
    }
    fn process_run(&mut self, _: RunEvent) -> Result<(), TrainingError> {
        Ok(())
    }
    fn process_train(&mut self, event: LearnerEvent<T>) -> Result<(), TrainingError>;
    fn process_valid(&mut self, event: LearnerEvent<V>) -> Result<(), TrainingError>;
    fn flush(&mut self) -> Result<(), TrainingError> {
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetricUpdate {
    pub split: Split,
    pub epoch: usize,
    pub iteration: usize,
    pub items_processed: usize,
    pub items_total: usize,
    pub learning_rate: Option<LearningRate>,
    pub metrics: Vec<MetricEntry>,
    pub event: String,
}

pub trait MetricLogger: Send {
    fn log(&mut self, update: &MetricUpdate) -> Result<(), TrainingError>;
    fn log_named(&mut self, update: &MetricUpdate, _: &str) -> Result<(), TrainingError> {
        self.log(update)
    }
    fn needs_snapshots(&self) -> bool {
        false
    }
    fn log_metric_definition(&mut self, _: &MetricDefinition) -> Result<(), TrainingError> {
        Ok(())
    }
    fn log_snapshots(
        &mut self,
        update: &MetricUpdate,
        _: &[MetricSnapshot],
    ) -> Result<(), TrainingError> {
        self.log(update)
    }
    fn read_numeric(
        &mut self,
        _: &str,
        _: usize,
        _: Split,
    ) -> Result<Vec<NumericEntry>, TrainingError> {
        Err(TrainingError(
            "This logger does not support numeric readback".into(),
        ))
    }
    fn log_snapshots_named(
        &mut self,
        update: &MetricUpdate,
        snapshots: &[MetricSnapshot],
        _: &str,
    ) -> Result<(), TrainingError> {
        self.log_snapshots(update, snapshots)
    }
    fn read_numeric_named(
        &mut self,
        _: &str,
        _: usize,
        _: &str,
    ) -> Result<Vec<NumericEntry>, TrainingError> {
        Err(TrainingError(
            "This logger does not support named numeric readback".into(),
        ))
    }
    fn flush(&mut self) -> Result<(), TrainingError> {
        Ok(())
    }
}

struct RegisteredLogger {
    logger: Box<dyn MetricLogger>,
    definitions: HashSet<MetricId>,
}
struct RegisteredRenderer {
    renderer: Box<dyn MetricsRenderer>,
    definitions: HashSet<MetricId>,
}
pub struct FileMetricLogger {
    writer: BufWriter<File>,
}
impl FileMetricLogger {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, TrainingError> {
        Ok(Self {
            writer: BufWriter::new(OpenOptions::new().create(true).append(true).open(path)?),
        })
    }
}
impl MetricLogger for FileMetricLogger {
    fn log(&mut self, update: &MetricUpdate) -> Result<(), TrainingError> {
        serde_json::to_writer(&mut self.writer, update)
            .map_err(|error| TrainingError(error.to_string()))?;
        self.writer.write_all(b"\n")?;
        Ok(())
    }
    fn log_named(&mut self, update: &MetricUpdate, name: &str) -> Result<(), TrainingError> {
        #[derive(Serialize)]
        struct NamedUpdate<'a> {
            #[serde(flatten)]
            update: &'a MetricUpdate,
            evaluation_name: &'a str,
        }
        serde_json::to_writer(
            &mut self.writer,
            &NamedUpdate {
                update,
                evaluation_name: name,
            },
        )
        .map_err(|error| TrainingError(error.to_string()))?;
        self.writer.write_all(b"\n")?;
        Ok(())
    }
    fn flush(&mut self) -> Result<(), TrainingError> {
        Ok(self.writer.flush()?)
    }
}

struct SplitMetrics<T> {
    metrics: Vec<Box<dyn OutputMetric<T>>>,
    epoch: usize,
    iteration: usize,
    processed: usize,
    total: usize,
    lr: Option<LearningRate>,
    finalized: bool,
}

#[derive(Serialize, Deserialize)]
struct SplitMetricsRecord {
    metrics: Vec<(String, Vec<u8>)>,
    epoch: usize,
    iteration: usize,
    processed: usize,
    total: usize,
    lr: Option<LearningRate>,
    finalized: bool,
}
impl<T> Default for SplitMetrics<T> {
    fn default() -> Self {
        Self {
            metrics: Vec::new(),
            epoch: 0,
            iteration: 0,
            processed: 0,
            total: 0,
            lr: None,
            finalized: false,
        }
    }
}
impl<T> SplitMetrics<T> {
    fn checkpoint(&mut self) -> Result<SplitMetricsRecord, TrainingError> {
        let mut metrics = Vec::with_capacity(self.metrics.len());
        for metric in &mut self.metrics {
            metrics.push((metric.entry().name, metric.checkpoint()?));
        }
        Ok(SplitMetricsRecord {
            metrics, epoch: self.epoch, iteration: self.iteration, processed: self.processed,
            total: self.total, lr: self.lr, finalized: self.finalized,
        })
    }
    fn restore_checkpoint(&mut self, state: SplitMetricsRecord) -> Result<(), TrainingError> {
        if state.metrics.len() != self.metrics.len()
            || state.metrics.iter().zip(&self.metrics).any(|((name, _), metric)| *name != metric.entry().name)
        { return Err(TrainingError("Metric registration differs from checkpoint".into())); }
        for (metric, (_, state)) in self.metrics.iter_mut().zip(state.metrics) {
            metric.restore_checkpoint(&state)?;
        }
        self.epoch = state.epoch;
        self.iteration = state.iteration;
        self.processed = state.processed;
        self.total = state.total;
        self.lr = state.lr;
        self.finalized = state.finalized;
        Ok(())
    }
    fn process(
        &mut self,
        event: LearnerEvent<T>,
        split: Split,
        store: &EventStore,
        test_name: Option<&str>,
    ) -> Result<MetricUpdate, TrainingError> {
        let kind = match event {
            LearnerEvent::StartSplit { epoch, total_items } => {
                self.epoch = epoch;
                self.total = total_items;
                self.processed = 0;
                self.iteration = 0;
                self.lr = None;
                self.finalized = false;
                for metric in &mut self.metrics {
                    metric.clear();
                }
                "start_split"
            }
            LearnerEvent::ProcessedItem(item) => {
                self.finalized = false;
                let metadata = MetricMetadata {
                    epoch: self.epoch,
                    iteration: item.iteration,
                    progress: item.progress.clone(),
                    learning_rate: item.learning_rate,
                };
                for metric in &mut self.metrics {
                    metric.update(&item.item, &metadata)?;
                }
                self.iteration = item.iteration;
                self.processed = item.progress.items_processed;
                self.total = item.progress.items_total;
                self.lr = item.learning_rate;
                "item"
            }
            LearnerEvent::EndSplit { epoch, completed } => {
                if epoch != self.epoch {
                    return Err(TrainingError("Metric split epoch mismatch".into()));
                }
                self.finalized = false;
                if completed {
                    for metric in &mut self.metrics {
                        metric.compute()?;
                    }
                    self.finalized = true;
                    let entries = self.metrics.iter().map(|metric| metric.entry()).collect();
                    if let Some(name) = test_name {
                        store.record_test(epoch, name, entries);
                    } else {
                        store.record(epoch, split, entries);
                    }
                    "end_split"
                } else {
                    "interrupted_split"
                }
            }
            LearnerEvent::EndEpoch(_) => "end_epoch",
        };
        Ok(MetricUpdate {
            split,
            epoch: self.epoch,
            iteration: self.iteration,
            items_processed: self.processed,
            items_total: self.total,
            learning_rate: self.lr,
            metrics: self.metrics.iter().map(|metric| metric.entry()).collect(),
            event: kind.into(),
        })
    }
}

/// Synchronous processing preserves event order and propagates readback/I/O failures.
pub struct MetricsProcessor<T, V> {
    train: SplitMetrics<T>,
    valid: SplitMetrics<V>,
    store: EventStore,
    loggers: Vec<RegisteredLogger>,
    valid_split: Split,
    progress_loggers: Vec<Box<dyn TrainingProgressLogger>>,
    evaluation_loggers: Vec<Box<dyn EvaluationProgressLogger>>,
    renderers: Vec<RegisteredRenderer>,
    evaluation_name: EvaluationName,
    evaluating: bool,
    summary_config: Option<crate::LearnerSummaryConfig>,
    evaluation_tag: Option<String>,
    completed_evaluations: Vec<String>,
}
impl<T, V> Default for MetricsProcessor<T, V> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T, V> MetricsProcessor<T, V> {
    pub fn new() -> Self {
        Self {
            train: SplitMetrics::default(),
            valid: SplitMetrics::default(),
            store: EventStore::default(),
            loggers: Vec::new(),
            valid_split: Split::Valid,
            progress_loggers: Vec::new(),
            evaluation_loggers: Vec::new(),
            renderers: Vec::new(),
            evaluation_name: EvaluationName::new("test"),
            evaluating: false,
            summary_config: None,
            evaluation_tag: None,
            completed_evaluations: Vec::new(),
        }
    }
    pub fn store(&self) -> EventStore {
        self.store.clone()
    }
    pub fn with_summary_config(mut self, config: crate::LearnerSummaryConfig) -> Self {
        self.summary_config = Some(config);
        self
    }
    pub fn with_progress_logger<L: TrainingProgressLogger + 'static>(mut self, logger: L) -> Self {
        self.progress_loggers.push(Box::new(logger));
        self
    }
    pub fn with_evaluation_progress_logger<L: EvaluationProgressLogger + 'static>(
        mut self,
        logger: L,
    ) -> Self {
        self.evaluation_loggers.push(Box::new(logger));
        self
    }
    pub fn with_renderer<R: MetricsRenderer + 'static>(mut self, renderer: R) -> Self {
        self.renderers.push(RegisteredRenderer {
            renderer: Box::new(renderer),
            definitions: HashSet::new(),
        });
        self
    }
    pub fn metric_snapshots(&self, split: Split) -> Vec<MetricSnapshot> {
        if split == Split::Train {
            self.train.metrics.iter().map(|metric| metric.snapshot(self.train.finalized)).collect()
        } else if split == self.valid_split {
            self.valid.metrics.iter().map(|metric| metric.snapshot(self.valid.finalized)).collect()
        } else {
            Vec::new()
        }
    }
    pub fn with_logger<L: MetricLogger + 'static>(mut self, logger: L) -> Self {
        self.loggers.push(RegisteredLogger {
            logger: Box::new(logger),
            definitions: HashSet::new(),
        });
        self
    }
    pub fn metric_train<M: Metric + 'static>(mut self, metric: M) -> Self
    where
        T: Adaptor<M::Input> + 'static,
    {
        self.train.metrics.push(adapted(metric));
        self
    }
    pub fn metric_valid<M: Metric + 'static>(mut self, metric: M) -> Self
    where
        V: Adaptor<M::Input> + 'static,
    {
        self.valid.metrics.push(adapted(metric));
        self
    }
    pub fn for_evaluation(mut self) -> Self {
        self.valid_split = Split::Test;
        self
    }
    fn log(&mut self, update: MetricUpdate) -> Result<(), TrainingError> {
        let snapshots = (!self.renderers.is_empty()
            || self
                .loggers
                .iter()
                .any(|registered| registered.logger.needs_snapshots()))
        .then(|| self.metric_snapshots(update.split));
        for registered in &mut self.loggers {
            if registered.logger.needs_snapshots() {
                let snapshots = snapshots.as_deref().unwrap();
                for snapshot in snapshots {
                    let id = &snapshot.definition.metric_id;
                    if !registered.definitions.contains(id) {
                        registered
                            .logger
                            .log_metric_definition(&snapshot.definition)?;
                        registered.definitions.insert(id.clone());
                    }
                }
                if let Some(name) = self.evaluation_tag.as_deref() {
                    registered
                        .logger
                        .log_snapshots_named(&update, snapshots, name)?;
                } else {
                    registered.logger.log_snapshots(&update, snapshots)?;
                }
            } else {
                if let Some(name) = self.evaluation_tag.as_deref() {
                    registered.logger.log_named(&update, name)?;
                } else {
                    registered.logger.log(&update)?;
                }
            }
        }
        let numeric: Vec<_> = if self.renderers.is_empty() {
            Vec::new()
        } else if update.split == Split::Train {
            self.train
                .metrics
                .iter()
                .map(|metric| metric.is_numeric())
                .collect()
        } else {
            self.valid
                .metrics
                .iter()
                .map(|metric| metric.is_numeric())
                .collect()
        };
        for registered in &mut self.renderers {
            for (snapshot, numeric) in snapshots.as_deref().unwrap().iter().zip(&numeric) {
                if registered
                    .definitions
                    .insert(snapshot.definition.metric_id.clone())
                {
                    registered
                        .renderer
                        .register_metric(snapshot.definition.clone());
                }
                if matches!(update.event.as_str(), "item" | "end_split") {
                    let state = if *numeric {
                        let value = if update.event == "end_split" {
                            snapshot.final_value.clone()
                        } else {
                            snapshot.current.clone()
                        };
                        MetricState::Numeric(snapshot.clone(), value)
                    } else {
                        MetricState::Generic(snapshot.clone())
                    };
                    if update.split == Split::Train {
                        registered.renderer.update_train(state);
                    } else if self.evaluating || update.split == Split::Test {
                        registered
                            .renderer
                            .update_test(self.evaluation_name.clone(), state);
                    } else {
                        registered.renderer.update_valid(state);
                    }
                }
            }
        }
        if self.evaluating || update.split == Split::Test {
            for logger in &mut self.evaluation_loggers {
                evaluation_progress(logger.as_mut(), &update, &self.evaluation_name);
            }
            for registered in &mut self.renderers {
                evaluation_progress(registered.renderer.as_mut(), &update, &self.evaluation_name);
            }
        } else {
            for logger in &mut self.progress_loggers {
                training_progress(logger.as_mut(), &update);
            }
            for registered in &mut self.renderers {
                training_progress(registered.renderer.as_mut(), &update);
            }
        }
        Ok(())
    }
}
impl<T, V> EventProcessorTraining<T, V> for MetricsProcessor<T, V> {
    fn checkpoint(&mut self) -> Result<Vec<u8>, TrainingError> {
        self.flush()?;
        crate::continuation::encode(&(
            1u32, self.train.checkpoint()?, self.valid.checkpoint()?, self.valid_split,
            self.store.checkpoint()?,
        ))
    }
    fn restore_checkpoint(&mut self, bytes: &[u8]) -> Result<(), TrainingError> {
        let (version, train, valid, split, store):
            (u32, SplitMetricsRecord, SplitMetricsRecord, Split, Vec<u8>) =
            crate::continuation::decode(bytes)?;
        if version != 1 || split != self.valid_split {
            return Err(TrainingError("Metric processor configuration differs from checkpoint".into()));
        }
        self.train.restore_checkpoint(train)?;
        self.valid.restore_checkpoint(valid)?;
        self.store.restore_checkpoint(&store)?;
        Ok(())
    }
    fn process_run(&mut self, event: RunEvent) -> Result<(), TrainingError> {
        if !self.renderers.is_empty()
            && matches!(
                event,
                RunEvent::TrainingStart { .. }
                    | RunEvent::EvaluationStart { .. }
                    | RunEvent::EvaluationsStart { .. }
            )
        {
            let definitions: Vec<_> = self
                .train
                .metrics
                .iter()
                .map(|metric| metric.definition())
                .chain(self.valid.metrics.iter().map(|metric| metric.definition()))
                .collect();
            for registered in &mut self.renderers {
                registered.definitions.clear();
                for definition in &definitions {
                    if registered.definitions.insert(definition.metric_id.clone()) {
                        registered.renderer.register_metric(definition.clone());
                    }
                }
            }
        }
        match &event {
            RunEvent::TrainingStart { .. } => {
                self.evaluating = false;
                self.evaluation_tag = None;
            }
            RunEvent::EvaluationStart { name, .. } => {
                self.evaluating = true;
                self.evaluation_name = name.clone();
                self.evaluation_tag = None;
            }
            RunEvent::EvaluationsStart { .. } => {
                self.evaluating = true;
                self.evaluation_tag = None;
                self.completed_evaluations.clear();
            }
            RunEvent::EvaluationSelect { name } => {
                self.evaluation_name = name.clone();
                self.evaluation_tag = Some(name.to_string());
            }
            _ => {}
        }
        for logger in &mut self.progress_loggers {
            training_run(logger.as_mut(), &event);
        }
        for logger in &mut self.evaluation_loggers {
            evaluation_run(logger.as_mut(), &event);
        }
        let summary_result: Result<_, TrainingError> = match &event {
            RunEvent::EvaluationsEnd { model: Some(model) } => {
                let summary = if let Some(config) = self.summary_config.clone() {
                    self.flush().and_then(|_| {
                        config
                            .init_named_eval(&self.completed_evaluations)
                            .map_err(TrainingError)
                    })
                } else {
                    Ok(crate::LearnerSummary::from_named_evaluation_store(
                        &self.store,
                        &self.completed_evaluations,
                    ))
                };
                summary.map(|summary| Some(summary.with_model(model.clone())))
            }
            RunEvent::TrainingEndWithSummary { epochs, model } => {
                let summary = if let Some(config) = self.summary_config.clone() {
                    self.flush()
                        .and_then(|_| config.init().map_err(TrainingError))
                        .map(|summary| summary.through_epoch(*epochs))
                } else {
                    Ok(crate::LearnerSummary::from_store(&self.store, *epochs))
                };
                summary.map(|summary| Some(summary.with_model(model.clone())))
            }
            RunEvent::EvaluationEndWithSummary { model } => {
                let summary = if let Some(config) =
                    self.summary_config.clone().filter(|_| self.valid.finalized)
                {
                    self.flush()
                        .and_then(|_| config.init_eval().map_err(TrainingError))
                        .map(|mut summary| {
                            if let Some(metrics) = summary.metrics.test.remove("") {
                                summary
                                    .metrics
                                    .test
                                    .insert(self.evaluation_name.to_string(), metrics);
                            }
                            summary
                        })
                } else {
                    Ok(crate::LearnerSummary::from_evaluation_store(
                        &self.store,
                        if self.valid.finalized {
                            self.valid.epoch
                        } else {
                            0
                        },
                        self.valid_split,
                        self.evaluation_name.as_str(),
                    ))
                };
                summary.map(|summary| Some(summary.with_model(model.clone())))
            }
            _ => Ok(None),
        };
        let (summary, summary_error) = match summary_result {
            Ok(summary) => (summary, None),
            Err(error) => (None, Some(error)),
        };
        if self.renderers.is_empty() {
            if let Some(summary) = &summary {
                println!("{summary}");
            }
        }
        for registered in &mut self.renderers {
            training_run(registered.renderer.as_mut(), &event);
            evaluation_run(registered.renderer.as_mut(), &event);
            match &event {
                RunEvent::TrainingEnd => registered.renderer.on_train_end(),
                RunEvent::EvaluationEnd => registered.renderer.on_test_end(),
                RunEvent::EvaluationsEnd { .. } => match &summary {
                    Some(summary) => registered
                        .renderer
                        .on_test_end_with_summary(summary.clone()),
                    None => registered.renderer.on_test_end(),
                },
                RunEvent::TrainingEndWithSummary { .. } => match &summary {
                    Some(summary) => registered
                        .renderer
                        .on_train_end_with_summary(summary.clone()),
                    None => registered.renderer.on_train_end(),
                },
                RunEvent::EvaluationEndWithSummary { .. } => match &summary {
                    Some(summary) => registered
                        .renderer
                        .on_test_end_with_summary(summary.clone()),
                    None => registered.renderer.on_test_end(),
                },
                _ => {}
            }
        }
        match summary_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
    fn process_train(&mut self, event: LearnerEvent<T>) -> Result<(), TrainingError> {
        let update = self.train.process(event, Split::Train, &self.store, None)?;
        self.log(update)
    }
    fn process_valid(&mut self, event: LearnerEvent<V>) -> Result<(), TrainingError> {
        let end_named =
            self.evaluation_tag.is_some() && matches!(&event, LearnerEvent::EndSplit { .. });
        let result = self
            .valid
            .process(
                event,
                self.valid_split,
                &self.store,
                self.evaluation_tag.as_deref(),
            )
            .and_then(|update| {
                let completed = update.event == "end_split";
                self.log(update)?;
                if completed {
                    if let Some(name) = &self.evaluation_tag {
                        self.completed_evaluations.push(name.clone());
                    }
                }
                Ok(())
            });
        if end_named && result.is_err() {
            for logger in &mut self.evaluation_loggers {
                logger.end_test();
            }
            for registered in &mut self.renderers {
                registered.renderer.end_test();
            }
        }
        result
    }
    fn flush(&mut self) -> Result<(), TrainingError> {
        for logger in &mut self.loggers {
            logger.logger.flush()?;
        }
        Ok(())
    }
}

fn training_run(logger: &mut dyn TrainingProgressLogger, event: &RunEvent) {
    match event {
        RunEvent::TrainingStart {
            total_epochs,
            starting_epoch,
            total_items,
            label,
        } => logger.start(
            *total_epochs,
            *starting_epoch,
            *total_items,
            label.as_deref(),
        ),
        RunEvent::TrainingEnd | RunEvent::TrainingEndWithSummary { .. } => logger.end(),
        RunEvent::TrainingMessage(message) => logger.log_event_training(message.clone()),
        _ => {}
    }
}
fn evaluation_run(logger: &mut dyn EvaluationProgressLogger, event: &RunEvent) {
    match event {
        RunEvent::EvaluationStart { total_tests, .. }
        | RunEvent::EvaluationsStart { total_tests } => logger.start_global_progress(*total_tests),
        RunEvent::EvaluationEnd
        | RunEvent::EvaluationEndWithSummary { .. }
        | RunEvent::EvaluationsEnd { .. } => logger.end_global_progress(),
        RunEvent::EvaluationMessage(message) => logger.log_event_evaluation(message.clone()),
        _ => {}
    }
}
fn training_progress(logger: &mut dyn TrainingProgressLogger, update: &MetricUpdate) {
    match update.event.as_str() {
        "start_split" => logger.start_split(
            if update.split == Split::Train {
                "train"
            } else {
                "valid"
            },
            update.items_total,
        ),
        "item" => {
            logger.update_split(update.items_processed);
            logger.log_event_training("Iteration".into());
        }
        "end_split" | "interrupted_split" => logger.end_split(),
        "end_epoch" if update.split == Split::Train => logger.update_epoch(update.epoch),
        _ => {}
    }
}
fn evaluation_progress(
    logger: &mut dyn EvaluationProgressLogger,
    update: &MetricUpdate,
    name: &EvaluationName,
) {
    match update.event.as_str() {
        "start_split" => logger.start_test(name.as_str(), update.items_total),
        "item" => {
            logger.update_test_progress(update.items_processed);
            logger.log_event_evaluation("Iteration".into());
        }
        "end_split" | "interrupted_split" => logger.end_test(),
        _ => {}
    }
}
