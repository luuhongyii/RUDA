use super::*;
use crate::{
    logger::{EvaluationProgressLogger, TrainingProgressLogger},
    renderer::*,
};

#[cfg(feature = "tui")]
#[test]
fn boxed_default_renderer_integrates_with_training() {
    let processor = metrics().with_renderer(default_renderer(Interrupter::new(), None));
    let result = SupervisedTraining::new(train_loader(), valid_loader(), Default::default())
        .num_epochs(2).with_metrics(processor)
        .fit(Learner::new(Regressor::<AD>::new(), SgdConfig::new().init(), 0.01));
    assert!(result.error.is_none() && result.interrupted.is_none());
    close(scalar(&result.model), reference(2, 1, 0.).0);
}

#[cfg(feature = "tui")]
#[test]
#[ignore = "requires a real PTY; exits without keyboard input"]
fn tui_real_terminal_training_closes_without_input() {
    use std::io::IsTerminal;
    assert!(std::io::stdout().is_terminal() && std::io::stdin().is_terminal());
    for mode in 0..3 {
        let processor = metrics().with_renderer(tui::TuiMetricsRendererWrapper::new(Interrupter::new(), None));
        let training = SupervisedTraining::new(train_loader(), valid_loader(), Default::default())
            .num_epochs(2).label("tui-acceptance");
        let training = match mode {
            0 => training.with_metrics(processor),
            1 => training.with_async_metrics(processor),
            _ => training.with_lazy_async_metrics(processor),
        };
        let result = training.fit(Learner::new(Regressor::<AD>::new(), SgdConfig::new().init(), 0.01));
        assert!(result.error.is_none() && result.interrupted.is_none());
        close(scalar(&result.model), reference(2, 1, 0.).0);
    }
    assert!(!ratatui::crossterm::terminal::is_raw_mode_enabled().unwrap());
}

#[derive(Default)]
struct Observed {
    events: Vec<String>,
    definitions: Vec<MetricDefinition>,
    states: Vec<(String, MetricState)>,
    definitions_at_start: Vec<usize>,
}
#[derive(Clone, Default)]
struct Observer(Arc<Mutex<Observed>>);
impl Observer {
    fn event(&self, event: impl Into<String>) {
        self.0.lock().unwrap().events.push(event.into());
    }
    fn events(&self) -> Vec<String> {
        self.0.lock().unwrap().events.clone()
    }
    fn states(&self) -> Vec<(String, MetricState)> {
        self.0.lock().unwrap().states.clone()
    }
}
impl TrainingProgressLogger for Observer {
    fn start(&mut self, total: usize, starting: usize, items: Option<usize>, label: Option<&str>) {
        self.event(format!("start:{total}:{starting}:{items:?}:{label:?}"));
        let mut state = self.0.lock().unwrap();
        let count = state.definitions.len();
        state.definitions_at_start.push(count);
    }
    fn start_split(&mut self, name: &str, total: usize) {
        self.event(format!("split:{name}:{total}"));
    }
    fn update_split(&mut self, processed: usize) {
        self.event(format!("step:{processed}"));
    }
    fn end_split(&mut self) {
        self.event("end_split");
    }
    fn update_epoch(&mut self, epoch: usize) {
        self.event(format!("epoch:{epoch}"));
    }
    fn end(&mut self) {
        self.event("end");
    }
    fn log_event_training(&mut self, event: String) {
        self.event(format!("event:{event}"));
    }
}
impl EvaluationProgressLogger for Observer {
    fn start_global_progress(&mut self, total: usize) {
        self.event(format!("eval_start:{total}"));
    }
    fn start_test(&mut self, name: &str, total: usize) {
        self.event(format!("test:{name}:{total}"));
    }
    fn update_test_progress(&mut self, processed: usize) {
        self.event(format!("test_step:{processed}"));
    }
    fn end_test(&mut self) {
        self.event("end_test");
    }
    fn end_global_progress(&mut self) {
        self.event("eval_end");
    }
    fn log_event_evaluation(&mut self, event: String) {
        self.event(format!("test_event:{event}"));
    }
}
impl MetricsRendererTraining for Observer {
    fn update_train(&mut self, state: MetricState) {
        self.0.lock().unwrap().states.push(("train".into(), state));
    }
    fn update_valid(&mut self, state: MetricState) {
        self.0.lock().unwrap().states.push(("valid".into(), state));
    }
    fn on_train_end(&mut self) {
        self.event("train_end");
    }
}
impl MetricsRendererEvaluation for Observer {
    fn update_test(&mut self, name: EvaluationName, state: MetricState) {
        self.0
            .lock()
            .unwrap()
            .states
            .push((name.to_string(), state));
    }
    fn on_test_end(&mut self) {
        self.event("test_end");
    }
}
impl MetricsRenderer for Observer {
    fn register_metric(&mut self, definition: MetricDefinition) {
        self.0.lock().unwrap().definitions.push(definition);
    }
}

fn epoch_events(epoch: usize) -> Vec<String> {
    let mut events = vec!["split:train:5".into()];
    for items in [2, 4, 5] {
        events.extend([format!("step:{items}"), "event:Iteration".into()]);
    }
    events.extend(["end_split".into(), "split:valid:3".into()]);
    for items in [2, 3] {
        events.extend([format!("step:{items}"), "event:Iteration".into()]);
    }
    events.extend(["end_split".into(), format!("epoch:{epoch}")]);
    events
}

#[test]
fn real_training_progress_and_rendering_match_across_sync_async_and_lazy() {
    for mode in 0..3 {
        let renderer = Observer::default();
        let progress = Observer::default();
        let processor = metrics()
            .with_renderer(renderer.clone())
            .with_progress_logger(progress.clone());
        let training = SupervisedTraining::new(train_loader(), valid_loader(), Default::default())
            .num_epochs(2)
            .label("fit");
        let training = match mode {
            0 => training.with_metrics(processor),
            1 => training.with_async_metrics(processor),
            _ => training.with_lazy_async_metrics(processor),
        };
        let result = training.fit(Learner::new(
            Regressor::<AD>::new(),
            SgdConfig::new().init(),
            0.01,
        ));
        assert!(result.error.is_none() && result.interrupted.is_none());
        close(scalar(&result.model), reference(2, 1, 0.).0);
        let mut expected = vec!["start:2:1:None:Some(\"fit\")".into()];
        expected.extend(epoch_events(1));
        expected.extend(epoch_events(2));
        expected.push("end".into());
        assert_eq!(progress.events(), expected);
        expected.push("train_end".into());
        assert_eq!(renderer.events(), expected);
        assert_eq!(renderer.0.lock().unwrap().definitions_at_start, [1]);
        let states = renderer.states();
        assert_eq!(states.len(), 14);
        for (position, (split, state)) in states.iter().enumerate() {
            let MetricState::Numeric(snapshot, point) = state else {
                panic!("Loss is numeric")
            };
            assert_eq!(snapshot.definition.name.as_str(), "Loss");
            let final_entry = matches!(position % 7, 3 | 6);
            assert_eq!(matches!(point, Some(NumericEntry::Final(_))), final_entry);
            if final_entry {
                let split = if split == "train" {
                    Split::Train
                } else {
                    Split::Valid
                };
                close(
                    point.as_ref().unwrap().current(),
                    result
                        .metrics
                        .find_metric("Loss", position / 7 + 1, split)
                        .unwrap(),
                );
            } else {
                assert!(matches!(point, Some(NumericEntry::Aggregated { .. })));
                assert_eq!(point, &snapshot.current);
            }
        }
    }
}

#[test]
fn checkpoint_continuation_reports_the_restored_starting_epoch() {
    let directory = Temp::new();
    let run = |epochs, restored, observer: Observer| {
        let mut training =
            SupervisedTraining::new(train_loader(), valid_loader(), Default::default())
                .num_epochs(epochs)
                .with_metrics(metrics().with_renderer(observer))
                .with_file_checkpointer(
                    &directory.0,
                    BinFileRecorder::<FullPrecisionSettings>::default(),
                    2,
                );
        if let Some(epoch) = restored {
            training = training.checkpoint(epoch);
        }
        training.fit(Learner::new(
            Regressor::<AD>::new(),
            SgdConfig::new().init(),
            0.01,
        ))
    };
    assert!(run(2, None, Observer::default()).error.is_none());
    let observer = Observer::default();
    let result = run(4, Some(2), observer.clone());
    assert!(result.error.is_none());
    assert_eq!(observer.events().first().unwrap(), "start:4:3:None:None");
    let mut expected = vec!["start:4:3:None:None".into()];
    expected.extend(epoch_events(3));
    expected.extend(epoch_events(4));
    expected.extend(["end".into(), "train_end".into()]);
    assert_eq!(observer.events(), expected);
    close(scalar(&result.model), reference(4, 1, 0.).0);
}

#[test]
fn progress_ends_for_preinterruption_restore_failure_and_early_stopping() {
    for failure in [false, true] {
        let observer = Observer::default();
        let mut training =
            SupervisedTraining::new(train_loader(), valid_loader(), Default::default())
                .num_epochs(3)
                .with_async_metrics(metrics().with_renderer(observer.clone()));
        if failure {
            training = training.checkpoint(2);
        } else {
            training.interrupter().stop(Some("test interruption"));
        }
        let result = training.fit(Learner::new(
            Regressor::<AD>::new(),
            SgdConfig::new().init(),
            0.01,
        ));
        assert_eq!(result.error.is_some(), failure);
        assert_eq!(result.interrupted.is_some(), !failure);
        assert_eq!(result.state.optimizer_steps, 0);
        assert_eq!(
            observer.events(),
            ["start:3:1:None:None", "end", "train_end"]
        );
        assert!(observer.states().is_empty());
    }
    struct StopAfterOne;
    impl EarlyStoppingStrategy for StopAfterOne {
        fn should_stop(&mut self, _: usize, _: &EventStore) -> bool {
            true
        }
    }
    let observer = Observer::default();
    let result = SupervisedTraining::new(train_loader(), valid_loader(), Default::default())
        .num_epochs(3)
        .with_metrics(metrics().with_renderer(observer.clone()))
        .early_stopping(StopAfterOne)
        .fit(Learner::new(
            Regressor::<AD>::new(),
            SgdConfig::new().init(),
            0.01,
        ));
    assert!(result.stopped_early && result.error.is_none());
    let mut expected = vec!["start:3:1:None:None".into()];
    expected.extend(epoch_events(1));
    expected.extend(["end".into(), "train_end".into()]);
    assert_eq!(observer.events(), expected);
}

#[test]
fn named_evaluation_progress_is_ordered_and_isolated_from_training() {
    for stopped in [false, true] {
        let observer = Observer::default();
        let progress = Observer::default();
        let metrics = MetricsProcessor::<(), RegressionOutput<TestBackend>>::new()
            .metric_valid(LossMetric::new())
            .for_evaluation()
            .with_renderer(observer.clone())
            .with_evaluation_progress_logger(progress.clone());
        let mut processor = AsyncProcessorTraining::new(metrics);
        let interrupter = Interrupter::new();
        if stopped {
            interrupter.stop(Some("pre-interrupted"));
        }
        let result = evaluate_named(
            Regressor::<AD>::new().valid(),
            valid_loader(),
            &Default::default(),
            &mut processor,
            &interrupter,
            EvaluationName::new("holdout A"),
        );
        assert_eq!(result.completed, !stopped);
        assert!(result.error.is_none());
        let mut expected = vec!["eval_start:1".into(), "test:holdout A:3".into()];
        if !stopped {
            expected.extend(
                [
                    "test_step:2",
                    "test_event:Iteration",
                    "test_step:3",
                    "test_event:Iteration",
                ]
                .map(str::to_owned),
            );
        }
        expected.extend(["end_test".into(), "eval_end".into()]);
        assert_eq!(progress.events(), expected);
        expected.push("test_end".into());
        assert_eq!(observer.events(), expected);
        let states = observer.states();
        assert_eq!(states.len(), if stopped { 0 } else { 3 });
        for (name, state) in states {
            assert_eq!(name, "holdout A");
            assert!(matches!(state, MetricState::Numeric(_, _)));
        }
    }
}

#[test]
fn interruption_inside_a_batch_closes_progress_without_publishing_a_final() {
    struct StopOnItem(Interrupter);
    impl MetricLogger for StopOnItem {
        fn log(&mut self, update: &MetricUpdate) -> Result<(), TrainingError> {
            if update.event == "item" {
                self.0.stop(Some("batch boundary"));
            }
            Ok(())
        }
    }
    let observer = Observer::default();
    let interrupter = Interrupter::new();
    let processor = metrics()
        .with_logger(StopOnItem(interrupter.clone()))
        .with_renderer(observer.clone());
    let result = SupervisedTraining::new(train_loader(), valid_loader(), Default::default())
        .num_epochs(3)
        .with_interrupter(interrupter)
        .with_metrics(processor)
        .fit(Learner::new(
            Regressor::<AD>::new(),
            SgdConfig::new().init(),
            0.01,
        ));
    assert!(result.interrupted.is_some() && result.error.is_none());
    assert_eq!(result.state.optimizer_steps, 1);
    assert_eq!(result.state.completed_epochs, 0);
    assert_eq!(
        observer.events(),
        [
            "start:3:1:None:None",
            "split:train:5",
            "step:2",
            "event:Iteration",
            "end_split",
            "end",
            "train_end"
        ]
    );
    let states = observer.states();
    assert_eq!(states.len(), 1);
    assert!(matches!(
        &states[0].1,
        MetricState::Numeric(_, Some(NumericEntry::Aggregated { .. }))
    ));
    assert!(
        result
            .metrics
            .find_metric("Loss", 1, Split::Train)
            .is_none()
    );
}

#[cfg(feature = "sys-metrics")]
#[test]
fn renderer_distinguishes_global_numeric_placeholders_and_actual_cuda_text() {
    let observer = Observer::default();
    let mut processor = MetricsProcessor::<ClassificationOutput<TestBackend>, ()>::new()
        .metric_train(AurocMetric::binary())
        .metric_train(CudaMetric::new())
        .with_renderer(observer.clone());
    processor
        .process_run(RunEvent::TrainingStart {
            total_epochs: 1,
            starting_epoch: 1,
            total_items: None,
            label: None,
        })
        .unwrap();
    processor
        .process_train(LearnerEvent::StartSplit {
            epoch: 1,
            total_items: 4,
        })
        .unwrap();
    let device = Default::default();
    processor
        .process_train(LearnerEvent::ProcessedItem(TrainingItem {
            item: ClassificationOutput::new(
                Tensor::zeros([4], &device),
                Tensor::from_floats([[0.9], [0.8], [0.2], [0.1]], &device),
                Tensor::from_ints([1, 0, 1, 0], &device),
            ),
            progress: ruda_model::data::dataloader::Progress::new(4, 4),
            iteration: 1,
            learning_rate: None,
        }))
        .unwrap();
    processor
        .process_train(LearnerEvent::EndSplit {
            epoch: 1,
            completed: true,
        })
        .unwrap();
    processor.process_run(RunEvent::TrainingEnd).unwrap();
    let states = observer.states();
    assert_eq!(states.len(), 4);
    assert!(
        matches!(&states[0].1, MetricState::Numeric(snapshot, None) if snapshot.serialized.is_not_available())
    );
    assert!(
        matches!(&states[2].1, MetricState::Numeric(_, Some(NumericEntry::Final(value))) if *value == 75.)
    );
    for index in [1, 3] {
        let MetricState::Generic(snapshot) = &states[index].1 else {
            panic!("CUDA telemetry is generic")
        };
        assert!(snapshot.final_value.is_none());
        if std::env::var("RUDA_REQUIRE_GPU").as_deref() == Ok("1") {
            assert!(snapshot.serialized.formatted.contains("GPU #0 - Memory"));
        }
    }
}

#[test]
fn async_run_messages_preserve_order_and_report_processor_errors() {
    struct FailingRun;
    impl EventProcessorTraining<(), ()> for FailingRun {
        fn process_run(&mut self, event: RunEvent) -> Result<(), TrainingError> {
            if let RunEvent::TrainingMessage(message) = event {
                return Err(TrainingError(message));
            }
            Ok(())
        }
        fn process_train(&mut self, _: LearnerEvent<()>) -> Result<(), TrainingError> {
            Ok(())
        }
        fn process_valid(&mut self, _: LearnerEvent<()>) -> Result<(), TrainingError> {
            Ok(())
        }
    }
    let mut processor = AsyncProcessorTraining::new(FailingRun);
    let immediate = processor
        .process_run(RunEvent::TrainingMessage("run error".into()))
        .err();
    let deferred = processor.flush().err();
    let errors: Vec<_> = immediate.into_iter().chain(deferred).collect();
    assert_eq!(errors, [TrainingError("run error".into())]);
    processor.finish().unwrap();
    let observer = Observer::default();
    let mut processor = AsyncProcessorTraining::new(
        MetricsProcessor::<(), ()>::new().with_renderer(observer.clone()),
    );
    processor
        .process_run(RunEvent::TrainingMessage("custom training".into()))
        .unwrap();
    processor
        .process_run(RunEvent::EvaluationMessage("custom eval".into()))
        .unwrap();
    processor.finish().unwrap();
    assert_eq!(
        observer.events(),
        ["event:custom training", "test_event:custom eval"]
    );
}
