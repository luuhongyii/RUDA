use super::*;
use crate::logger::{FileMetricLogger as SerializedFileLogger, InMemoryMetricLogger};
use crate::{
    logger::{EvaluationProgressLogger, TrainingProgressLogger},
    renderer::*,
};

#[derive(Default)]
struct Recorded {
    events: Vec<String>,
    summaries: Vec<LearnerSummary>,
    numeric: Vec<(String, Option<NumericEntry>)>,
}
#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Recorded>>);
impl TrainingProgressLogger for Capture {
    fn start(&mut self, _: usize, _: usize, _: Option<usize>, _: Option<&str>) {}
    fn update_epoch(&mut self, _: usize) {}
    fn start_split(&mut self, _: &str, _: usize) {}
    fn update_split(&mut self, _: usize) {}
    fn end_split(&mut self) {}
    fn end(&mut self) {}
    fn log_event_training(&mut self, _: String) {}
}
impl EvaluationProgressLogger for Capture {
    fn start_global_progress(&mut self, total: usize) {
        self.0.lock().unwrap().events.push(format!("start:{total}"));
    }
    fn start_test(&mut self, name: &str, total: usize) {
        self.0
            .lock()
            .unwrap()
            .events
            .push(format!("test:{name}:{total}"));
    }
    fn update_test_progress(&mut self, n: usize) {
        self.0.lock().unwrap().events.push(format!("items:{n}"));
    }
    fn end_test(&mut self) {
        self.0.lock().unwrap().events.push("end_test".into());
    }
    fn end_global_progress(&mut self) {
        self.0.lock().unwrap().events.push("end".into());
    }
    fn log_event_evaluation(&mut self, _: String) {}
}
impl MetricsRendererTraining for Capture {
    fn update_train(&mut self, _: MetricState) {}
    fn update_valid(&mut self, _: MetricState) {}
}
impl MetricsRendererEvaluation for Capture {
    fn update_test(&mut self, name: EvaluationName, state: MetricState) {
        if let MetricState::Numeric(_, value) = state {
            self.0
                .lock()
                .unwrap()
                .numeric
                .push((name.to_string(), value));
        }
    }
    fn on_test_end(&mut self) {
        self.0.lock().unwrap().events.push("callback".into());
    }
    fn on_test_end_with_summary(&mut self, summary: LearnerSummary) {
        let mut data = self.0.lock().unwrap();
        data.events.push("summary".into());
        data.summaries.push(summary);
    }
}
impl MetricsRenderer for Capture {
    fn register_metric(&mut self, _: MetricDefinition) {}
}
fn eval_metrics() -> MetricsProcessor<(), RegressionOutput<TestBackend>> {
    MetricsProcessor::new().metric_valid(LossMetric::new())
}
fn evaluator() -> Evaluator<TestBackend, Regressor<TestBackend>> {
    Evaluator::new(Regressor::<AD>::new().valid(), Default::default())
}

#[test]
fn multiple_named_evaluations_match_independent_values_files_and_progress() {
    for mode in 0..3 {
        let directory = Temp::new();
        let capture = Capture::default();
        let mut memory = InMemoryMetricLogger::new();
        let processor = eval_metrics()
            .with_logger(SerializedFileLogger::new_eval(&directory.0))
            .with_logger(memory.clone())
            .with_renderer(capture.clone())
            .with_summary_config(LearnerSummaryConfig::new(&directory.0, &["Loss"]));
        let eval = evaluator().summary();
        let eval = match mode {
            0 => eval.with_metrics(processor),
            1 => eval.with_async_metrics(processor),
            _ => eval.with_lazy_async_metrics(processor),
        };
        let result = eval.eval_all([
            (" Hold Out ", loader(&[1., 2., 3.], 0)),
            ("Other", loader(&[4., 5.], 0)),
        ]);
        assert!(result.completed());
        assert_eq!(result.total_tests, 2);
        assert_eq!(
            result
                .completed_tests
                .iter()
                .map(|n| n.as_str())
                .collect::<Vec<_>>(),
            [" Hold Out ", "Other"]
        );
        assert!(result.metrics.metrics(1, Split::Test).is_empty());
        let mut disk = SerializedFileLogger::new_eval(&directory.0);
        for (name, value, file) in [
            (" Hold Out ", 56. / 3., "hold-out"),
            ("Other", 82., "other"),
        ] {
            close(
                result.metrics.find_test_metric("Loss", 1, name).unwrap(),
                value,
            );
            let entries = memory.read_numeric_named("Loss", 1, name).unwrap();
            assert_eq!(entries, disk.read_numeric_named("Loss", 1, name).unwrap());
            assert!(
                directory
                    .0
                    .join("test")
                    .join(file)
                    .join("Loss.log")
                    .is_file()
            );
            match entries.last().unwrap() {
                NumericEntry::Final(actual) => close(*actual, value),
                _ => panic!("Missing final aggregate"),
            }
        }
        assert!(!directory.0.join("Loss.log").exists());
        let data = capture.0.lock().unwrap();
        assert_eq!(
            data.events,
            [
                "start:2",
                "test: Hold Out :3",
                "items:2",
                "items:3",
                "end_test",
                "test:Other:2",
                "items:2",
                "end_test",
                "end",
                "summary"
            ]
        );
        assert_eq!(data.summaries.len(), 1);
        let memory_summary =
            LearnerSummary::from_named_evaluation_store(&result.metrics, &[" Hold Out ", "Other"]);
        assert_eq!(data.summaries[0].metrics, memory_summary.metrics);
        assert_eq!(data.summaries[0].epochs, 1);
        assert!(
            data.summaries[0]
                .model
                .as_deref()
                .unwrap()
                .contains("Regressor")
        );
        assert_eq!(
            data.numeric
                .iter()
                .filter(|(name, _)| name == "Other")
                .count(),
            2
        );
    }
}

#[test]
fn evaluation_empty_list_and_empty_dataset_have_distinct_progress() {
    let capture = Capture::default();
    let result = evaluator()
        .summary()
        .with_metrics(eval_metrics().with_renderer(capture.clone()))
        .eval_all(Vec::<(&str, TestLoader<TestBackend, Regressor<TestBackend>>)>::new());
    assert!(result.completed());
    assert_eq!(result.total_tests, 0);
    let data = capture.0.lock().unwrap();
    assert_eq!(data.events, ["start:0", "end", "summary"]);
    assert_eq!(data.summaries[0].epochs, 0);
    assert_eq!(data.summaries[0].metrics, SummaryMetrics::default());
    drop(data);
    let capture = Capture::default();
    let result = evaluator()
        .summary()
        .with_lazy_async_metrics(eval_metrics().with_renderer(capture.clone()))
        .eval("empty", loader(&[], 0));
    assert!(result.completed());
    let data = capture.0.lock().unwrap();
    assert_eq!(
        data.events,
        ["start:1", "test:empty:0", "end_test", "end", "summary"]
    );
    assert_eq!(data.summaries[0].epochs, 1);
    assert!(data.summaries[0].metrics.test["empty"].is_empty());
}

#[test]
fn repeated_evaluation_name_resets_metrics_and_keeps_last_final() {
    let directory = Temp::new();
    let result = evaluator()
        .with_async_metrics(
            eval_metrics().with_logger(SerializedFileLogger::new_eval(&directory.0)),
        )
        .eval_all([
            ("same", loader(&[1., 2., 3.], 0)),
            ("same", loader(&[4., 5.], 0)),
        ]);
    assert!(result.completed());
    close(
        result.metrics.find_test_metric("Loss", 1, "same").unwrap(),
        82.,
    );
    let mut disk = SerializedFileLogger::new_eval(&directory.0);
    let entries = disk.read_numeric_named("Loss", 1, "same").unwrap();
    assert_eq!(
        entries
            .iter()
            .filter(|n| matches!(n, NumericEntry::Final(_)))
            .count(),
        2
    );
    let summary = LearnerSummary::new_named_eval(&directory.0, &["Loss"], &["same"]).unwrap();
    close(summary.metrics.test["same"][0].entries[0].value, 82.);
}

#[test]
fn evaluation_interruption_closes_progress_and_does_not_start_later_datasets() {
    struct Stop {
        interrupter: Interrupter,
        event: &'static str,
    }
    impl MetricLogger for Stop {
        fn log(&mut self, update: &MetricUpdate) -> Result<(), TrainingError> {
            if update.event == self.event {
                self.interrupter.stop(Some("evaluation boundary"));
            }
            Ok(())
        }
    }
    for when in ["before", "item", "end_split"] {
        let interrupter = Interrupter::new();
        if when == "before" {
            interrupter.stop(Some("before evaluation"));
        }
        let capture = Capture::default();
        let processor = eval_metrics()
            .with_renderer(capture.clone())
            .with_logger(Stop {
                interrupter: interrupter.clone(),
                event: when,
            });
        let eval = evaluator().with_interrupter(interrupter).summary();
        let eval = if when == "end_split" {
            eval.with_async_metrics(processor)
        } else {
            eval.with_metrics(processor)
        };
        let result = eval.eval_all([
            ("first", loader(&[1., 2., 3., 4., 5.], 0)),
            ("later", valid_loader()),
        ]);
        assert!(!result.completed() && result.interrupted.is_some() && result.error.is_none());
        assert_eq!(
            result.completed_tests.len(),
            usize::from(when == "end_split")
        );
        assert!(result.metrics.test_metrics(1, "later").is_empty());
        let data = capture.0.lock().unwrap();
        assert_eq!(&data.events[data.events.len() - 2..], ["end", "summary"]);
        assert!(!data.events.iter().any(|e| e.starts_with("test:later")));
        assert_eq!(
            data.events.iter().filter(|e| *e == "end_test").count(),
            usize::from(when != "before")
        );
        assert_eq!(data.summaries[0].epochs, usize::from(when == "end_split"));
    }
}

#[test]
fn evaluation_metric_io_errors_stop_following_datasets_and_finish_callbacks() {
    for asynchronous in [false, true] {
        let directory = Temp::new();
        std::fs::write(directory.0.join("test"), "not a directory").unwrap();
        let capture = Capture::default();
        let processor = eval_metrics()
            .with_logger(SerializedFileLogger::new_eval(&directory.0))
            .with_renderer(capture.clone());
        let eval = if asynchronous {
            evaluator().with_async_metrics(processor)
        } else {
            evaluator().with_metrics(processor)
        };
        let result = eval.eval_all([("bad", valid_loader()), ("later", valid_loader())]);
        assert!(result.error.is_some() && !result.completed());
        assert!(result.completed_tests.is_empty());
        let data = capture.0.lock().unwrap();
        assert_eq!(&data.events[data.events.len() - 2..], ["end", "callback"]);
        assert_eq!(
            data.events
                .iter()
                .filter(|event| *event == "end_test")
                .count(),
            1
        );
        assert!(!data.events.iter().any(|e| e.starts_with("test:later")));
    }
}

#[test]
fn tagged_epoch_logs_and_readback_use_baseline_tag_format() {
    let directory = Temp::new();
    let mut memory = InMemoryMetricLogger::new();
    let result = evaluator()
        .with_metrics(
            eval_metrics()
                .with_logger(SerializedFileLogger::new(&directory.0))
                .with_logger(memory.clone()),
        )
        .eval_all([
            (" 开发 集 ", valid_loader()),
            ("Upper CASE", loader(&[4., 5.], 0)),
        ]);
    assert!(result.completed());
    let mut disk = SerializedFileLogger::new(&directory.0);
    for (original, tag) in [(" 开发 集 ", "开发-集"), ("Upper CASE", "upper-case")] {
        assert!(
            directory
                .0
                .join("test")
                .join(tag)
                .join("epoch-1/Loss.log")
                .is_file()
        );
        assert_eq!(
            disk.read_numeric_named("Loss", 1, original).unwrap(),
            memory.read_numeric_named("Loss", 1, original).unwrap()
        );
    }
    let summary = LearnerSummary::new(&directory.0, &["Loss"]).unwrap();
    close(
        summary.metrics.test["开发-集"][0].entries[0].value,
        56. / 3.,
    );
    close(summary.metrics.test["upper-case"][0].entries[0].value, 82.);
    assert!(memory.entries("Loss", 1, Split::Test).is_empty());
}

#[test]
fn named_json_logging_adds_identity_without_changing_legacy_updates() {
    let directory = Temp::new();
    let path = directory.0.join("metrics.jsonl");
    let result = evaluator()
        .with_metrics(eval_metrics().with_logger(FileMetricLogger::new(&path).unwrap()))
        .eval("Hold Out", valid_loader());
    assert!(result.completed());
    let mut processor = eval_metrics()
        .for_evaluation()
        .with_logger(FileMetricLogger::new(&path).unwrap());
    let result = evaluate_named(
        Regressor::<AD>::new().valid(),
        valid_loader(),
        &Default::default(),
        &mut processor,
        &Interrupter::new(),
        EvaluationName::new("old-label"),
    );
    assert!(result.completed);
    let rows: Vec<serde_json::Value> = std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 8);
    for row in &rows[..4] {
        assert_eq!(row["evaluation_name"], "Hold Out");
        assert_eq!(row["split"], "Test");
    }
    for row in &rows[4..] {
        assert!(row.get("evaluation_name").is_none());
    }
}

#[test]
fn evaluator_lazy_handoff_supports_distinct_synced_output_types() {
    struct Deferred<B: Backend>(RegressionOutput<B>);
    impl<B: Backend> ItemLazy for Deferred<B> {
        type Synced = RegressionOutput<B>;
        fn sync(self) -> Result<Self::Synced, TrainingError> {
            self.0.sync()
        }
    }
    #[derive(Module, Debug)]
    struct DeferredModel<B: Backend> {
        model: Regressor<B>,
    }
    impl<B: Backend> InferenceStep<B> for DeferredModel<B> {
        type Input = Tensor<B, 1>;
        type Output = Deferred<B>;
        fn step(&self, input: Self::Input) -> Self::Output {
            Deferred(self.model.output(input))
        }
    }
    let result = Evaluator::new(
        DeferredModel {
            model: Regressor::<AD>::new().valid(),
        },
        Default::default(),
    )
    .with_lazy_async_metrics(eval_metrics())
    .eval("converted", valid_loader());
    assert!(result.completed());
    close(
        result
            .metrics
            .find_test_metric("Loss", 1, "converted")
            .unwrap(),
        56. / 3.,
    );
}
