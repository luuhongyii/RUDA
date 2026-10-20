use super::*;
#[path = "metric_tests.rs"]
mod metric_tests;
#[path = "numeric_metric_tests.rs"]
mod numeric_metric_tests;
#[path = "logger_tests.rs"]
mod logger_tests;
#[path = "progress_tests.rs"]
mod progress_tests;
#[path = "summary_tests.rs"]
mod summary_tests;
#[path = "evaluation_tests.rs"]
mod evaluation_tests;
#[cfg(feature = "rl")]
#[path = "rl_tests.rs"]
mod rl_tests;
#[path = "lazy_tests.rs"]
mod lazy_tests;
#[cfg(feature = "vision")]
#[path = "vision_tests.rs"]
mod vision_tests;
use crate::metric::*;
use ruda_model::{
    data::{
        dataloader::{DataLoader, DataLoaderBuilder, batcher::Batcher},
        dataset::Dataset,
    },
    module::{AutodiffModule, Module, Param},
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{
        ElementConversion, Tensor, TensorData,
        backend::{AutodiffBackend, Backend},
    },
};
use ruda_nn::{Dropout, DropoutConfig};
use ruda_optim::{
    AdamConfig, GradientsParams, Optimizer, SgdConfig,
    lr_scheduler::exponential::ExponentialLrSchedulerConfig,
};
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

#[cfg(not(feature = "test-cuda"))]
pub(crate) type TestBackend = ruda_tensor_host::Host;
#[cfg(all(feature = "test-cuda", not(feature = "test-cuda-native")))]
pub(crate) type TestBackend = ruda_tensor_device::cuda::Cuda;
#[cfg(feature = "test-cuda-native")]
pub(crate) type TestBackend = ruda_tensor_device::cuda::CudaNative;
type AD = ruda_autodiff::Autodiff<TestBackend>;

#[derive(Clone, Debug)]
struct Samples(Vec<f32>);
impl Dataset<f32> for Samples {
    fn get(&self, index: usize) -> Option<f32> {
        self.0.get(index).copied()
    }
    fn len(&self) -> usize {
        self.0.len()
    }
}
struct Batches;
impl<B: Backend> Batcher<B, f32, Tensor<B, 1>> for Batches {
    fn batch(&self, items: Vec<f32>, device: &B::Device) -> Tensor<B, 1> {
        let len = items.len();
        Tensor::from_data(TensorData::new(items, [len]), device)
    }
}
fn loader<B: Backend>(items: &[f32], workers: usize) -> Arc<dyn DataLoader<B, Tensor<B, 1>>> {
    DataLoaderBuilder::new(Batches)
        .batch_size(2)
        .num_workers(workers)
        .build(Samples(items.to_vec()))
}
fn train_loader() -> Arc<dyn DataLoader<AD, Tensor<AD, 1>>> {
    loader(&[1., 2., 3., 4., 5.], 0)
}
fn valid_loader() -> Arc<dyn DataLoader<TestBackend, Tensor<TestBackend, 1>>> {
    loader(&[1., 2., 3.], 0)
}

#[derive(Module, Debug)]
struct Regressor<B: Backend> {
    weight: Param<Tensor<B, 1>>,
    dropout: Dropout,
    step_offset: f64,
}
impl<B: Backend> Regressor<B> {
    fn new() -> Self {
        Self {
            weight: Param::from_tensor(Tensor::zeros([1], &Default::default())),
            dropout: DropoutConfig::new(0.).init(),
            step_offset: 0.,
        }
    }
    fn output(&self, x: Tensor<B, 1>) -> RegressionOutput<B> {
        let [n] = x.dims();
        let targets = x.clone() * 2.;
        let output = self.dropout.forward(x) * self.weight.val();
        let loss = (output.clone() - targets.clone()).square();
        RegressionOutput {
            loss,
            output: output.reshape([n, 1]),
            targets: targets.reshape([n, 1]),
        }
    }
}
impl<B: AutodiffBackend> TrainStep<B> for Regressor<B> {
    type Input = Tensor<B, 1>;
    type Output = RegressionOutput<B::InnerBackend>;
    fn step(&self, x: Self::Input) -> TrainOutput<Self::Output> {
        let output = self.output(x);
        let grads = output.loss.clone().mean().backward();
        TrainOutput::new(
            self,
            grads,
            RegressionOutput {
                loss: output.loss.inner(),
                output: output.output.inner(),
                targets: output.targets.inner(),
            },
        )
    }
    fn optimize<O: Optimizer<Self, B>>(
        self,
        optim: &mut O,
        lr: f64,
        grads: GradientsParams,
    ) -> Self {
        let mut result = optim.step(lr, self, grads);
        if result.step_offset != 0. {
            result.weight = result
                .weight
                .map(|weight| (weight + result.step_offset).detach().require_grad());
        }
        result
    }
}
impl<B: Backend> InferenceStep<B> for Regressor<B> {
    type Input = Tensor<B, 1>;
    type Output = RegressionOutput<B>;
    fn step(&self, x: Self::Input) -> Self::Output {
        self.output(x)
    }
}
fn scalar(model: &Regressor<TestBackend>) -> f64 {
    model.weight.val().into_scalar().elem()
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 2e-5, "{a} differs from {b}");
}
fn metrics() -> MetricsProcessor<RegressionOutput<TestBackend>, RegressionOutput<TestBackend>> {
    MetricsProcessor::new()
        .metric_train(LossMetric::new())
        .metric_valid(LossMetric::new())
}
fn reference(epochs: usize, accumulation: usize, offset: f64) -> (f64, usize) {
    let batches: [&[f64]; 3] = [&[1., 2.], &[3., 4.], &[5.]];
    let mut weight = 0.;
    let mut steps = 0;
    for _ in 0..epochs {
        for window in batches.chunks(accumulation) {
            let gradient: f64 = window
                .iter()
                .map(|batch| {
                    batch
                        .iter()
                        .map(|x| 2. * x * (weight * x - 2. * x))
                        .sum::<f64>()
                        / batch.len() as f64
                })
                .sum();
            weight = weight - 0.01 * gradient + offset;
            steps += 1;
        }
    }
    (weight, steps)
}

#[test]
fn training_matches_independent_sgd_and_validation_metrics() {
    let model = Regressor::<AD>::new();
    let learner = Learner::new(model, SgdConfig::new().init(), 0.01);
    let result = SupervisedTraining::new(train_loader(), valid_loader(), Default::default())
        .num_epochs(3)
        .with_metrics(metrics())
        .fit(learner);
    let expected = reference(3, 1, 0.);
    close(scalar(&result.model), expected.0);
    assert_eq!(result.state.optimizer_steps, expected.1);
    assert_eq!(result.state.completed_epochs, 3);
    assert!(result.error.is_none() && result.interrupted.is_none());
    close(
        result.metrics.find_metric("Loss", 3, Split::Valid).unwrap(),
        (expected.0 - 2.).powi(2) * 14. / 3.,
    );
    assert!(!result.model.dropout.training.val().is_enabled());
    assert!(!result.model.weight.is_require_grad());
}

#[test]
fn accumulation_is_summed_and_flushes_partial_final_window() {
    for accumulation in [2, 4] {
        let learner = Learner::new(Regressor::<AD>::new(), SgdConfig::new().init(), 0.01);
        let result = SupervisedTraining::new(train_loader(), valid_loader(), Default::default())
            .num_epochs(2)
            .grads_accumulation(accumulation)
            .fit(learner);
        let expected = reference(2, accumulation, 0.);
        close(scalar(&result.model), expected.0);
        assert_eq!(result.state.optimizer_steps, expected.1);
        assert!(result.error.is_none());
    }
}

#[test]
fn custom_optimizer_hook_is_called_once_per_update() {
    let mut model = Regressor::<AD>::new();
    model.step_offset = 0.015;
    let learner = Learner::new(model, SgdConfig::new().init(), 0.01);
    let result = SupervisedTraining::new(train_loader(), valid_loader(), Default::default())
        .num_epochs(2)
        .grads_accumulation(2)
        .fit(learner);
    close(scalar(&result.model), reference(2, 2, 0.015).0);
}

#[test]
fn scheduler_advances_only_on_optimizer_updates() {
    let scheduler = ExponentialLrSchedulerConfig::new(0.01, 0.5).init().unwrap();
    let learner = Learner::new(Regressor::<AD>::new(), SgdConfig::new().init(), scheduler);
    let result = SupervisedTraining::new(train_loader(), valid_loader(), Default::default())
        .grads_accumulation(2)
        .num_epochs(2)
        .fit(learner);
    assert_eq!(result.state.optimizer_steps, 4);
    close(result.state.learning_rate, 0.00125);
}

#[test]
fn multithread_dataloaders_train_and_validate() {
    let learner = Learner::new(Regressor::<AD>::new(), SgdConfig::new().init(), 0.01);
    let result = SupervisedTraining::new(
        loader(&[1., 2., 3., 4., 5.], 2),
        loader(&[1., 2., 3.], 2),
        Default::default(),
    )
    .with_metrics(metrics())
    .fit(learner);
    assert!(result.error.is_none());
    assert_eq!(result.state.optimizer_steps, 3);
    assert!(
        result
            .metrics
            .find_metric("Loss", 1, Split::Valid)
            .unwrap()
            .is_finite()
    );
}

#[derive(Clone, Default)]
struct Log(Arc<Mutex<Vec<MetricUpdate>>>);
impl MetricLogger for Log {
    fn log(&mut self, update: &MetricUpdate) -> Result<(), TrainingError> {
        self.0.lock().unwrap().push(update.clone());
        Ok(())
    }
}
struct StopLogger {
    handle: Interrupter,
    split: Split,
    at: usize,
}
impl MetricLogger for StopLogger {
    fn log(&mut self, update: &MetricUpdate) -> Result<(), TrainingError> {
        if update.event == "item" && update.split == self.split && update.iteration == self.at {
            self.handle.stop(Some("requested"));
        }
        Ok(())
    }
}

#[test]
fn event_order_and_progress_are_complete() {
    let log = Log::default();
    let learner = Learner::new(Regressor::<AD>::new(), SgdConfig::new().init(), 0.01);
    let result = SupervisedTraining::new(train_loader(), valid_loader(), Default::default())
        .with_metrics(metrics().with_logger(log.clone()))
        .fit(learner);
    assert!(result.error.is_none());
    let entries = log.0.lock().unwrap();
    let kinds: Vec<_> = entries
        .iter()
        .map(|entry| (entry.split, entry.event.as_str()))
        .collect();
    assert_eq!(
        kinds,
        vec![
            (Split::Train, "start_split"),
            (Split::Train, "item"),
            (Split::Train, "item"),
            (Split::Train, "item"),
            (Split::Train, "end_split"),
            (Split::Valid, "start_split"),
            (Split::Valid, "item"),
            (Split::Valid, "item"),
            (Split::Valid, "end_split"),
            (Split::Train, "end_epoch")
        ]
    );
    assert_eq!(
        (
            entries[3].items_processed,
            entries[3].items_total,
            entries[3].iteration
        ),
        (5, 5, 3)
    );
    assert!(entries[6].learning_rate.is_none());
}

#[test]
fn interrupt_does_not_apply_partial_accumulation_or_validate() {
    let handle = Interrupter::new();
    let learner = Learner::new(Regressor::<AD>::new(), SgdConfig::new().init(), 0.01);
    let result = SupervisedTraining::new(train_loader(), valid_loader(), Default::default())
        .with_interrupter(handle.clone())
        .grads_accumulation(2)
        .with_metrics(metrics().with_logger(StopLogger {
            handle,
            split: Split::Train,
            at: 1,
        }))
        .fit(learner);
    assert_eq!(result.state.optimizer_steps, 0);
    assert_eq!(result.state.completed_epochs, 0);
    close(scalar(&result.model), 0.);
    assert_eq!(
        result.interrupted.unwrap().reason.as_deref(),
        Some("requested")
    );
    assert!(result.metrics.metrics(1, Split::Valid).is_empty());
}

#[test]
fn interrupt_validation_does_not_commit_epoch() {
    let handle = Interrupter::new();
    let learner = Learner::new(Regressor::<AD>::new(), SgdConfig::new().init(), 0.01);
    let result = SupervisedTraining::new(train_loader(), valid_loader(), Default::default())
        .with_interrupter(handle.clone())
        .with_metrics(metrics().with_logger(StopLogger {
            handle,
            split: Split::Valid,
            at: 1,
        }))
        .fit(learner);
    assert_eq!(result.state.optimizer_steps, 3);
    assert_eq!(result.state.completed_epochs, 0);
    assert!(result.metrics.metrics(1, Split::Valid).is_empty());
}

#[test]
fn preexisting_stop_performs_no_training_step() {
    let handle = Interrupter::new();
    handle.stop(None);
    let learner = Learner::new(Regressor::<AD>::new(), SgdConfig::new().init(), 0.01);
    let result = SupervisedTraining::new(train_loader(), valid_loader(), Default::default())
        .with_interrupter(handle)
        .fit(learner);
    assert_eq!(result.state.optimizer_steps, 0);
    close(scalar(&result.model), 0.);
}

#[test]
fn error_takes_precedence_and_first_error_is_retained() {
    let handle = Interrupter::new();
    handle.stop(Some("stop"));
    handle.fail(TrainingError("first".into()));
    handle.fail(TrainingError("second".into()));
    assert!(handle.interruption().is_none());
    assert_eq!(handle.error().unwrap().0, "first");
}

struct FailingLogger;
impl MetricLogger for FailingLogger {
    fn log(&mut self, update: &MetricUpdate) -> Result<(), TrainingError> {
        if update.event == "item" {
            Err(TrainingError("logger failed".into()))
        } else {
            Ok(())
        }
    }
}
#[test]
fn processor_failure_is_reported_without_extra_optimizer_step() {
    let learner = Learner::new(Regressor::<AD>::new(), SgdConfig::new().init(), 0.01);
    let result = SupervisedTraining::new(train_loader(), valid_loader(), Default::default())
        .with_metrics(metrics().with_logger(FailingLogger))
        .fit(learner);
    assert_eq!(result.state.optimizer_steps, 1);
    assert_eq!(result.error.unwrap().0, "logger failed");
}

#[test]
fn empty_loaders_and_zero_epochs_do_not_step_scheduler() {
    for epochs in [0, 2] {
        let learner = Learner::new(Regressor::<AD>::new(), SgdConfig::new().init(), 0.01);
        let result = SupervisedTraining::new(loader(&[], 0), loader(&[], 0), Default::default())
            .num_epochs(epochs)
            .with_metrics(metrics())
            .fit(learner);
        assert_eq!(result.state.optimizer_steps, 0);
        assert_eq!(result.state.completed_epochs, epochs);
        assert!(result.error.is_none());
        assert_eq!(
            result.metrics.find_metric("Loss", epochs, Split::Valid),
            None
        );
    }
}

#[test]
#[should_panic(expected = "greater than zero")]
fn rejects_zero_accumulation() {
    let learner = Learner::new(Regressor::<AD>::new(), SgdConfig::new().init(), 0.01);
    SupervisedTraining::new(train_loader(), valid_loader(), Default::default())
        .grads_accumulation(0)
        .fit(learner);
}

#[test]
fn loss_and_accuracy_use_sample_weighted_aggregates() {
    let device = Default::default();
    let mut loss = LossMetric::<TestBackend>::new();
    loss.update(&LossInput {
        tensor: Tensor::from_floats([1., 3.], &device),
    })
    .unwrap();
    loss.update(&LossInput {
        tensor: Tensor::from_floats([8.], &device),
    })
    .unwrap();
    close(loss.value().unwrap(), 4.);
    let mut accuracy = AccuracyMetric::<TestBackend>::new();
    accuracy
        .update(&AccuracyInput {
            predictions: Tensor::from_floats([[3., 1.], [4., 2.]], &device),
            targets: Tensor::from_ints([0, 1], &device),
        })
        .unwrap();
    accuracy
        .update(&AccuracyInput {
            predictions: Tensor::from_floats([[1., 3.]], &device),
            targets: Tensor::from_ints([1], &device),
        })
        .unwrap();
    close(accuracy.value().unwrap(), 200. / 3.);
    accuracy.clear();
    loss.clear();
    assert!(accuracy.value().is_none() && loss.value().is_none());
}

#[test]
fn early_stopping_uses_completed_validation_and_warmup() {
    let learner = Learner::new(Regressor::<AD>::new(), SgdConfig::new().init(), 0.);
    let result = SupervisedTraining::new(train_loader(), valid_loader(), Default::default())
        .num_epochs(8)
        .with_metrics(metrics())
        .early_stopping(
            MetricEarlyStoppingStrategy::new("Loss", Split::Valid, Direction::Lowest, 2)
                .with_warmup_epochs(Some(3)),
        )
        .fit(learner);
    assert_eq!(result.state.completed_epochs, 4);
    assert!(result.stopped_early && result.interrupted.is_none() && result.error.is_none());
}

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ruda-train-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn epoch_checkpoint_restores_adam_and_scheduler_for_continuation() {
    let directory = Temp::new();
    let run = |epochs, checkpoint| {
        let scheduler = ExponentialLrSchedulerConfig::new(0.01, 0.8).init().unwrap();
        let learner = Learner::new(Regressor::<AD>::new(), AdamConfig::new().init(), scheduler);
        let mut training =
            SupervisedTraining::new(train_loader(), valid_loader(), Default::default())
                .num_epochs(epochs)
                .grads_accumulation(2)
                .with_file_checkpointer(
                    &directory.0,
                    BinFileRecorder::<FullPrecisionSettings>::default(),
                    2,
                );
        if let Some(epoch) = checkpoint {
            training = training.checkpoint(epoch);
        }
        training.fit(learner)
    };
    let first = run(2, None);
    assert!(first.error.is_none());
    assert!(directory.0.join("epoch-1.bin").is_file());
    std::fs::create_dir(directory.0.join("epoch-3.bin")).unwrap();
    let failed_save = run(3, Some(2));
    assert!(failed_save.error.is_some());
    assert!(directory.0.join("epoch-1.bin").is_file());
    assert!(directory.0.join("epoch-2.bin").is_file());
    std::fs::remove_dir(directory.0.join("epoch-3.bin")).unwrap();
    let resumed = run(4, Some(2));
    assert!(resumed.error.is_none());
    assert!(!directory.0.join("epoch-1.bin").exists());
    assert_eq!(std::fs::read_dir(&directory.0).unwrap().count(), 2);
    let uninterrupted = run(4, None);
    assert!(uninterrupted.error.is_none());
    close(scalar(&resumed.model), scalar(&uninterrupted.model));
    assert_eq!(resumed.state, uninterrupted.state);
    assert_eq!(resumed.state.optimizer_steps, 8);
    assert!(!directory.0.join("epoch-1.bin").exists());
    assert!(directory.0.join("epoch-3.bin").exists() && directory.0.join("epoch-4.bin").exists());
}

#[test]
fn missing_checkpoint_is_reported_before_training() {
    let learner = Learner::new(Regressor::<AD>::new(), SgdConfig::new().init(), 0.01);
    let result = SupervisedTraining::new(train_loader(), valid_loader(), Default::default())
        .checkpoint(2)
        .fit(learner);
    assert!(result.error.unwrap().0.contains("checkpointer"));
    assert_eq!(result.state.optimizer_steps, 0);
}

#[test]
fn standalone_evaluation_uses_validation_model_and_flushes_logs() {
    let directory = Temp::new();
    let model = Regressor::<AD>::new().valid();
    let mut processor = MetricsProcessor::<(), RegressionOutput<TestBackend>>::new()
        .metric_valid(LossMetric::new())
        .for_evaluation()
        .with_logger(FileMetricLogger::new(directory.0.join("metrics.jsonl")).unwrap());
    let result = evaluate(
        model,
        valid_loader(),
        &Default::default(),
        &mut processor,
        &Interrupter::new(),
    );
    assert!(result.completed && result.error.is_none());
    close(
        processor
            .store()
            .find_metric("Loss", 1, Split::Test)
            .unwrap(),
        56. / 3.,
    );
    let lines = std::fs::read_to_string(directory.0.join("metrics.jsonl")).unwrap();
    let updates: Vec<MetricUpdate> = lines
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(updates.len(), 4);
    assert_eq!(updates.last().unwrap().event, "end_split");
}
