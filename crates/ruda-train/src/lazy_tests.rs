use super::*;
use ruda_model::tensor::{Int, StreamId};
use std::sync::mpsc;

fn event<T>(item: T) -> LearnerEvent<T> {
    LearnerEvent::ProcessedItem(TrainingItem {
        item,
        progress: ruda_model::data::dataloader::Progress {
            items_processed: 2,
            items_total: 7,
        },
        iteration: 3,
        learning_rate: Some(0.01),
    })
}

#[test]
fn unread_autodiff_outputs_do_not_require_a_user_exit_fence() {
    let device = Default::default();
    let x = Tensor::<AD, 2>::from_floats([[1., 2.], [3., 4.]], &device).require_grad();
    let y = x.clone().matmul(x.transpose()).square() + 1.0;
    drop(y);
}

#[test]
fn backend_identity_matches_native_or_fusion_feature() {
    let name = TestBackend::name(&Default::default());
    #[cfg(not(feature = "test-cuda"))]
    assert_eq!(name, "flex");
    #[cfg(all(feature = "test-cuda", not(feature = "test-cuda-native")))]
    assert!(name.starts_with("fusion<"), "{name}");
    #[cfg(feature = "test-cuda-native")]
    assert!(!name.contains("fusion"), "{name}");
    TestBackend::flush(&Default::default()).unwrap();
}

#[test]
fn backend_and_autodiff_flush_preserve_values_and_gradients() {
    let device = Default::default();
    let x = Tensor::<AD, 1>::from_floats([1., 2., 3.], &device).require_grad();
    let y = x.clone().square() + 2.0;
    AD::flush(&device).unwrap();
    let grads = y.clone().sum().backward();
    x.grad(&grads)
        .unwrap()
        .into_data()
        .assert_eq(&TensorData::from([2., 4., 6.]), false);
    let y = y.inner();
    TestBackend::flush(&device).unwrap();
    let values = std::thread::spawn(move || y.into_data()).join().unwrap();
    values.assert_eq(&TensorData::from([3., 6., 11.]), false);
}

#[test]
fn autodiff_output_wrappers_remove_all_tensor_backend_types() {
    let device = Default::default();
    let loss = Tensor::<AD, 1>::from_floats([2.], &device).require_grad() * 3.0;
    let output = Tensor::<AD, 2>::from_floats([[1., 4.]], &device).require_grad() + 2.0;
    let targets = Tensor::<AD, 1, Int>::from_ints([1], &device);
    let classification: ClassificationOutput<TestBackend> = AutodiffItem(
        ClassificationOutput::new(loss.clone(), output.clone(), targets),
    )
    .sync()
    .unwrap();
    assert!(!TestBackend::ad_enabled(&classification.loss.device()));
    close(classification.loss.into_scalar().elem(), 6.0);
    classification
        .output
        .into_data()
        .assert_eq(&TensorData::from([[3., 6.]]), false);
    let regression: RegressionOutput<TestBackend> = AutodiffItem(RegressionOutput::new(
        loss.clone(),
        output.clone(),
        output.clone(),
    ))
    .sync()
    .unwrap();
    assert!(!regression.output.is_require_grad());
    let multilabel: MultiLabelClassificationOutput<TestBackend> =
        AutodiffItem(MultiLabelClassificationOutput::new(
            loss.clone(),
            output.clone(),
            Tensor::from_ints([[0, 1]], &device),
        ))
        .sync()
        .unwrap();
    multilabel
        .targets
        .into_data()
        .assert_eq(&TensorData::from([[0i32, 1]]), false);
    let sequence: SequenceOutput<TestBackend> = AutodiffItem(SequenceOutput::new(
        loss,
        output.reshape([1, 1, 2]),
        Some(Tensor::from_ints([[1]], &device)),
        Tensor::from_ints([[1]], &device),
    ))
    .sync()
    .unwrap();
    assert!(!sequence.logits.is_require_grad());
    sequence
        .predictions
        .unwrap()
        .into_data()
        .assert_eq(&TensorData::from([[1i32]]), false);
}

struct StreamProbe {
    expected: StreamId,
    thread: std::thread::ThreadId,
    ready: mpsc::Sender<()>,
    release: mpsc::Receiver<()>,
}
impl ItemLazy for StreamProbe {
    type Synced = StreamId;
    fn sync(self) -> Result<StreamId, TrainingError> {
        assert_eq!(StreamId::current(), self.expected);
        assert_ne!(std::thread::current().id(), self.thread);
        self.ready.send(()).unwrap();
        self.release.recv().unwrap();
        Ok(self.expected)
    }
}
struct StreamProcessor(Arc<Mutex<Vec<StreamId>>>);
impl EventProcessorTraining<StreamId, ()> for StreamProcessor {
    fn process_train(&mut self, event: LearnerEvent<StreamId>) -> Result<(), TrainingError> {
        if let LearnerEvent::ProcessedItem(item) = event {
            assert_ne!(StreamId::current(), item.item);
            assert_eq!(item.iteration, 3);
            assert_eq!(item.progress.items_processed, 2);
            assert_eq!(item.progress.items_total, 7);
            assert_eq!(item.learning_rate, Some(0.01));
            self.0.lock().unwrap().push(item.item);
        }
        Ok(())
    }
    fn process_valid(&mut self, _: LearnerEvent<()>) -> Result<(), TrainingError> {
        Ok(())
    }
}
#[test]
fn lazy_preparation_is_off_thread_and_restores_producer_stream() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut processor = AsyncProcessorTraining::<StreamProbe, ()>::with_lazy_capacity(
        StreamProcessor(seen.clone()),
        0,
    );
    let (ready, started) = mpsc::channel();
    let (release, wait) = mpsc::channel();
    let stream = StreamId::current();
    processor
        .process_train(event(StreamProbe {
            expected: stream,
            thread: std::thread::current().id(),
            ready,
            release: wait,
        }))
        .unwrap();
    started.recv().unwrap();
    assert!(seen.lock().unwrap().is_empty());
    release.send(()).unwrap();
    processor.flush().unwrap();
    assert_eq!(*seen.lock().unwrap(), vec![stream]);
    assert_eq!(StreamId::current(), stream);
}

struct Fallible(usize);
impl ItemLazy for Fallible {
    type Synced = usize;
    fn sync(self) -> Result<usize, TrainingError> {
        if self.0 == 0 {
            Err(TrainingError("preparation failure".into()))
        } else {
            Ok(self.0)
        }
    }
}
struct Counts(Arc<Mutex<Vec<usize>>>);
impl EventProcessorTraining<usize, usize> for Counts {
    fn process_train(&mut self, event: LearnerEvent<usize>) -> Result<(), TrainingError> {
        if let LearnerEvent::ProcessedItem(item) = event {
            self.0.lock().unwrap().push(item.item);
        }
        Ok(())
    }
    fn process_valid(&mut self, event: LearnerEvent<usize>) -> Result<(), TrainingError> {
        self.process_train(event)
    }
}
#[test]
fn lazy_preparation_error_is_reported_once_and_queue_continues() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut processor =
        AsyncProcessorTraining::<Fallible, Fallible>::with_lazy(Counts(seen.clone()));
    let mut failures = Vec::new();
    for result in [
        processor.process_train(event(Fallible(0))),
        processor.process_valid(event(Fallible(7))),
        processor.flush(),
    ] {
        if let Err(error) = result {
            failures.push(error.0);
        }
    }
    assert_eq!(failures, vec!["preparation failure"]);
    assert_eq!(*seen.lock().unwrap(), vec![7]);
    processor.flush().unwrap();
}
struct PanicItem;
impl ItemLazy for PanicItem {
    type Synced = usize;
    fn sync(self) -> Result<usize, TrainingError> {
        panic!("prepare panic")
    }
}
#[test]
fn lazy_preparation_panic_reaches_caller() {
    let error = std::panic::catch_unwind(|| {
        let mut processor = AsyncProcessorTraining::<PanicItem, Fallible>::with_lazy(Counts(
            Arc::new(Mutex::new(Vec::new())),
        ));
        let _ = processor.process_train(event(PanicItem));
        let _ = processor.flush();
    })
    .unwrap_err();
    assert_eq!(error.downcast_ref::<&str>(), Some(&"prepare panic"));
}

#[derive(Module, Debug)]
struct LazyRegressor<B: Backend> {
    inner: Regressor<B>,
}
impl<B: AutodiffBackend> TrainStep<B> for LazyRegressor<B> {
    type Input = Tensor<B, 1>;
    type Output = AutodiffItem<RegressionOutput<B>>;
    fn step(&self, input: Self::Input) -> TrainOutput<Self::Output> {
        let output = self.inner.output(input);
        let grads = output.loss.clone().mean().backward();
        TrainOutput::new(self, grads, AutodiffItem(output))
    }
}
impl<B: Backend> InferenceStep<B> for LazyRegressor<B> {
    type Input = Tensor<B, 1>;
    type Output = RegressionOutput<B>;
    fn step(&self, input: Self::Input) -> Self::Output {
        self.inner.output(input)
    }
}
#[test]
fn lazy_autodiff_training_and_native_training_match_sgd_oracle() {
    let learner = Learner::new(
        LazyRegressor::<AD> {
            inner: Regressor::new(),
        },
        SgdConfig::new().init(),
        0.01,
    );
    let result = SupervisedTraining::new(train_loader(), valid_loader(), Default::default())
        .num_epochs(3)
        .with_lazy_async_metrics(metrics())
        .fit(learner);
    assert!(result.error.is_none(), "{:?}", result.error);
    assert_eq!(result.state.completed_epochs, 3);
    close(scalar(&result.model.inner), reference(3, 1, 0.0).0);
    let learner = Learner::new(Regressor::<AD>::new(), SgdConfig::new().init(), 0.01);
    let native = SupervisedTraining::new(train_loader(), valid_loader(), Default::default())
        .num_epochs(3)
        .with_lazy_async_metrics(metrics())
        .fit(learner);
    assert!(native.error.is_none(), "{:?}", native.error);
    close(scalar(&native.model), scalar(&result.model.inner));
    close(
        result.metrics.find_metric("Loss", 3, Split::Valid).unwrap(),
        native.metrics.find_metric("Loss", 3, Split::Valid).unwrap(),
    );
}

struct KeepOutput(Arc<Mutex<Option<RegressionOutput<TestBackend>>>>);
impl EventProcessorTraining<RegressionOutput<TestBackend>, ()> for KeepOutput {
    fn process_train(
        &mut self,
        event: LearnerEvent<RegressionOutput<TestBackend>>,
    ) -> Result<(), TrainingError> {
        if let LearnerEvent::ProcessedItem(item) = event {
            *self.0.lock().unwrap() = Some(item.item);
        }
        Ok(())
    }
    fn process_valid(&mut self, _: LearnerEvent<()>) -> Result<(), TrainingError> {
        Ok(())
    }
}
#[test]
fn producing_thread_can_exit_before_prepared_output_is_read() {
    let output = Arc::new(Mutex::new(None));
    let stored = output.clone();
    let mut processor =
        AsyncProcessorTraining::<RegressionOutput<TestBackend>, ()>::with_lazy(KeepOutput(stored));
    #[cfg(feature = "test-cuda")]
    let mut inspector = None;
    let producer = std::thread::spawn(move || {
        #[cfg(feature = "test-cuda")]
        if !cfg!(feature = "test-cuda-native") {
            inspector = Some(ruda_fusion::inspect::FusionInspector::install(
                StreamId::current(),
            ));
        }
        let device = Default::default();
        let x = Tensor::<TestBackend, 1>::from_floats([1., 2., 3.], &device);
        let value = x.square() + 1.0;
        processor
            .process_train(event(RegressionOutput::new(
                value.clone(),
                value.clone().reshape([3, 1]),
                value.reshape([3, 1]),
            )))
            .unwrap();
        #[cfg(feature = "test-cuda")]
        return (processor, inspector);
        #[cfg(not(feature = "test-cuda"))]
        processor
    });
    #[cfg(feature = "test-cuda")]
    let (mut processor, inspector) = producer.join().unwrap();
    #[cfg(not(feature = "test-cuda"))]
    let mut processor = producer.join().unwrap();
    processor.flush().unwrap();
    #[cfg(feature = "test-cuda")]
    if let Some(inspector) = inspector {
        assert!(
            inspector
                .reports()
                .iter()
                .map(|r| r.total_operations())
                .sum::<usize>()
                >= 2
        );
    }
    let value = output.lock().unwrap().take().unwrap();
    value
        .loss
        .into_data()
        .assert_eq(&TensorData::from([2., 5., 10.]), false);
}

#[test]
fn router_flush_dispatches_each_selected_backend() {
    type R = ruda_tensor_router::Router<(TestBackend, ruda_tensor_host::Host)>;
    for device in [
        ruda_tensor_router::duo::MultiDevice::B1(Default::default()),
        ruda_tensor_router::duo::MultiDevice::B2(Default::default()),
    ] {
        let x = Tensor::<R, 1>::from_floats([2., 3.], &device).square() + 1.0;
        R::flush(&device).unwrap();
        std::thread::spawn(move || x.into_data())
            .join()
            .unwrap()
            .assert_eq(&TensorData::from([5., 10.]), false);
    }
}
