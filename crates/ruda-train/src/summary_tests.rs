use super::*;
use crate::logger::FileMetricLogger as SerializedFileLogger;
use crate::summary::MetricEntry as SummaryEntry;
use crate::{
    logger::{EvaluationProgressLogger, TrainingProgressLogger},
    renderer::*,
};

fn write_log(directory: &std::path::Path, relative: &str, text: &str) {
    let path = directory.join(relative);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}
fn metric<'a>(metrics: &'a [MetricSummary], name: &str) -> &'a MetricSummary {
    metrics.iter().find(|metric| metric.name == name).unwrap()
}

#[test]
fn summary_file_inputs_match_baseline_unweighted_fixture() {
    let directory = Temp::new();
    assert!(LearnerSummary::new(directory.0.join("missing"), &["Loss"]).is_err());
    assert!(LearnerSummary::new(&directory.0, &["Loss"]).is_err());
    std::fs::create_dir_all(directory.0.join("train/epoch-1")).unwrap();
    std::fs::create_dir_all(directory.0.join("valid/epoch-1")).unwrap();
    let empty = LearnerSummary::new(&directory.0, &["Loss"]).unwrap();
    assert_eq!(empty.epochs, 1);
    assert!(empty.metrics.train.is_empty() && empty.metrics.valid.is_empty());
    write_log(&directory.0, "train/epoch-1/Loss.log", "1.0\n2.0");
    write_log(&directory.0, "valid/epoch-1/Loss.log", "1.0");
    let summary = LearnerSummaryConfig::new(&directory.0, &["Loss", "missing"])
        .init()
        .unwrap();
    assert_eq!(summary.epochs, 1);
    assert_eq!(
        summary.metrics.train,
        [MetricSummary {
            name: "Loss".into(),
            entries: vec![SummaryEntry {
                step: 1,
                value: 1.5
            }]
        }]
    );
    assert_eq!(
        summary.metrics.valid,
        [MetricSummary {
            name: "Loss".into(),
            entries: vec![SummaryEntry { step: 1, value: 1. }]
        }]
    );
}

#[test]
fn summary_weights_finals_and_epoch_scopes() {
    let directory = Temp::new();
    write_log(&directory.0, "train/epoch-2/Long_Loss.log", "2,3\n10,1\n");
    write_log(
        &directory.0,
        "valid/epoch-2/Long_Loss.log",
        "NaN,0\n99,100\n8,final\n",
    );
    write_log(&directory.0, "test/epoch-3/Long_Loss.log", "3,2\n7,2\n");
    write_log(&directory.0, "test/holdout/epoch-5/Long_Loss.log", "9\n");
    write_log(&directory.0, "train/epoch-0/Long_Loss.log", "99\n");
    write_log(&directory.0, "train/epoch-invalid/Long_Loss.log", "99\n");
    let summary = LearnerSummary::new(&directory.0, &["Long Loss"]).unwrap();
    assert_eq!(summary.epochs, 5);
    assert_eq!(
        summary.metrics.train[0].entries,
        [SummaryEntry { step: 2, value: 4. }]
    );
    assert_eq!(
        summary.metrics.valid[0].entries,
        [SummaryEntry { step: 2, value: 8. }]
    );
    assert_eq!(
        summary.metrics.test[""][0].entries,
        [SummaryEntry { step: 3, value: 5. }]
    );
    assert_eq!(
        summary.metrics.test["holdout"][0].entries,
        [SummaryEntry { step: 5, value: 9. }]
    );
    let flat = directory.0.join("flat");
    write_log(&flat, "Long_Loss.log", "2,3\n10,1\n4,final\n");
    assert_eq!(
        LearnerSummary::new_eval(flat, &["Long Loss"])
            .unwrap()
            .metrics
            .test[""][0]
            .entries,
        [SummaryEntry { step: 1, value: 4. }]
    );
}

#[test]
fn summary_rejects_corruption_and_preserves_zero_count_nan() {
    let directory = Temp::new();
    for bad in ["not-numeric\n", "1,final\n2,3\n", "N/A\n"] {
        write_log(&directory.0, "train/epoch-1/Loss.log", bad);
        assert!(LearnerSummary::new(&directory.0, &["Loss"]).is_err());
    }
    for text in ["0,0\n", "NaN,0\n1,1\n", "5,1\nNaN,final\n"] {
        write_log(&directory.0, "train/epoch-1/Loss.log", text);
        assert!(
            LearnerSummary::new(&directory.0, &["Loss"])
                .unwrap()
                .metrics
                .train[0]
                .entries[0]
                .value
                .is_nan()
        );
    }
    write_log(&directory.0, "train/epoch-1/Loss.log", "NaN,0\n3,final\n");
    assert_eq!(
        LearnerSummary::new(&directory.0, &["Loss"])
            .unwrap()
            .metrics
            .train[0]
            .entries[0]
            .value,
        3.
    );
    write_log(&directory.0, "train/epoch-1/Loss.log", "\n");
    assert!(
        LearnerSummary::new(&directory.0, &["Loss"])
            .unwrap()
            .metrics
            .train
            .is_empty()
    );
}

#[test]
fn summary_display_and_merge_preserve_nan_ties_and_model_identity() {
    let entries = vec![
        SummaryEntry { step: 1, value: 2. },
        SummaryEntry {
            step: 2,
            value: -0.001,
        },
        SummaryEntry {
            step: 3,
            value: -0.001,
        },
        SummaryEntry {
            step: 4,
            value: f64::NAN,
        },
    ];
    let summary = LearnerSummary {
        epochs: 4,
        model: Some("Linear".into()),
        metrics: SummaryMetrics {
            train: vec![MetricSummary {
                name: "Loss".into(),
                entries,
            }],
            ..Default::default()
        },
    };
    let display = summary.to_string();
    assert!(display.starts_with("======================== Learner Summary ========================\nModel:\nLinear\nTotal Epochs: 4\n"));
    let row = display
        .lines()
        .find(|line| line.starts_with("| Train"))
        .unwrap();
    assert_eq!(
        row.split('|').map(str::trim).collect::<Vec<_>>(),
        ["", "Train", "Loss", "-1.000e-3", "2", "NaN", "4", ""]
    );
    let other = LearnerSummary {
        epochs: 9,
        model: Some("Other".into()),
        metrics: SummaryMetrics {
            train: vec![MetricSummary {
                name: "Loss".into(),
                entries: vec![SummaryEntry { step: 9, value: 7. }],
            }],
            test: std::collections::HashMap::from([(
                "holdout".into(),
                vec![MetricSummary {
                    name: "Accuracy".into(),
                    entries: vec![SummaryEntry {
                        step: 1,
                        value: 90.,
                    }],
                }],
            )]),
            ..Default::default()
        },
    };
    let merged = summary.merge(other);
    assert_eq!(merged.epochs, 4);
    assert!(merged.model.is_none());
    assert_eq!(metric(&merged.metrics.train, "Loss").entries.len(), 5);
    assert!(merged.to_string().contains("Test (holdout)"));
    let same = LearnerSummary::from_store(&EventStore::default(), 0).with_model("Shared".into());
    assert_eq!(same.clone().merge(same).model.as_deref(), Some("Shared"));
}

#[derive(Default)]
struct Captured {
    summaries: Vec<LearnerSummary>,
    events: Vec<&'static str>,
}
#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Captured>>);
impl TrainingProgressLogger for Capture {
    fn start(&mut self, _: usize, _: usize, _: Option<usize>, _: Option<&str>) {}
    fn update_epoch(&mut self, _: usize) {}
    fn start_split(&mut self, _: &str, _: usize) {}
    fn update_split(&mut self, _: usize) {}
    fn end_split(&mut self) {}
    fn end(&mut self) {
        self.0.lock().unwrap().events.push("end");
    }
    fn log_event_training(&mut self, _: String) {}
}
impl EvaluationProgressLogger for Capture {
    fn start_global_progress(&mut self, _: usize) {}
    fn start_test(&mut self, _: &str, _: usize) {}
    fn update_test_progress(&mut self, _: usize) {}
    fn end_test(&mut self) {}
    fn end_global_progress(&mut self) {
        self.0.lock().unwrap().events.push("eval_end");
    }
    fn log_event_evaluation(&mut self, _: String) {}
}
impl MetricsRendererTraining for Capture {
    fn update_train(&mut self, _: MetricState) {}
    fn update_valid(&mut self, _: MetricState) {}
    fn on_train_end_with_summary(&mut self, summary: LearnerSummary) {
        let mut capture = self.0.lock().unwrap();
        capture.events.push("summary");
        capture.summaries.push(summary);
    }
}
impl MetricsRendererEvaluation for Capture {
    fn update_test(&mut self, _: EvaluationName, _: MetricState) {}
    fn on_test_end_with_summary(&mut self, summary: LearnerSummary) {
        let mut capture = self.0.lock().unwrap();
        capture.events.push("eval_summary");
        capture.summaries.push(summary);
    }
}
impl MetricsRenderer for Capture {
    fn register_metric(&mut self, _: MetricDefinition) {}
}

#[test]
fn real_training_summary_matches_memory_file_and_independent_sgd() {
    for mode in 0..3 {
        let directory = Temp::new();
        let capture = Capture::default();
        let processor = metrics()
            .with_logger(SerializedFileLogger::new(&directory.0))
            .with_renderer(capture.clone());
        let training = SupervisedTraining::new(train_loader(), valid_loader(), Default::default())
            .num_epochs(2)
            .summary();
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
        let memory = result.summary();
        let file = LearnerSummary::new(&directory.0, &["Loss"]).unwrap();
        assert_eq!(memory.metrics, file.metrics);
        let capture = capture.0.lock().unwrap();
        assert_eq!(capture.events, ["end", "summary"]);
        assert_eq!(capture.summaries.len(), 1);
        assert_eq!(capture.summaries[0].metrics, file.metrics);
        assert_eq!(capture.summaries[0].epochs, 2);
        assert!(
            capture.summaries[0]
                .model
                .as_deref()
                .unwrap()
                .contains("Regressor")
        );
        let mut weight: f64 = 0.;
        for epoch in 0..2 {
            let mut loss_sum = 0.;
            for batch in [&[1., 2.][..], &[3., 4.][..], &[5.][..]] {
                let sum_squares: f64 = batch.iter().map(|x| x * x).sum();
                loss_sum += (weight - 2.).powi(2) * sum_squares;
                weight -= 0.01 * 2. * (weight - 2.) * sum_squares / batch.len() as f64;
            }
            close(
                metric(&memory.metrics.train, "Loss").entries[epoch].value,
                loss_sum / 5.,
            );
            close(
                metric(&memory.metrics.valid, "Loss").entries[epoch].value,
                (weight - 2.).powi(2) * 14. / 3.,
            );
        }
        close(scalar(&result.model), weight);
    }
}

#[test]
fn named_evaluation_summary_matches_file_and_handles_interruption() {
    for stopped in [false, true] {
        let directory = Temp::new();
        let capture = Capture::default();
        let metrics = MetricsProcessor::<(), RegressionOutput<TestBackend>>::new()
            .metric_valid(LossMetric::new())
            .for_evaluation()
            .with_summary_config(LearnerSummaryConfig::new(&directory.0, &["Loss"]))
            .with_logger(SerializedFileLogger::new_eval(&directory.0))
            .with_renderer(capture.clone());
        let mut processor = AsyncProcessorTraining::with_lazy(metrics);
        let interrupter = Interrupter::new();
        if stopped {
            interrupter.stop(Some("test"));
        }
        let result = evaluate_named_with_summary(
            Regressor::<AD>::new().valid(),
            valid_loader(),
            &Default::default(),
            &mut processor,
            &interrupter,
            EvaluationName::new("holdout"),
        );
        assert_eq!(result.completed, !stopped);
        assert!(result.error.is_none());
        let capture = capture.0.lock().unwrap();
        assert_eq!(capture.events, ["eval_end", "eval_summary"]);
        assert_eq!(capture.summaries.len(), 1);
        let summary = &capture.summaries[0];
        assert_eq!(summary.epochs, usize::from(!stopped));
        assert!(summary.metrics.train.is_empty() && summary.metrics.valid.is_empty());
        if stopped {
            assert!(summary.metrics.test["holdout"].is_empty());
        } else {
            let file = LearnerSummary::new_eval(&directory.0, &["Loss"]).unwrap();
            assert_eq!(summary.metrics.test["holdout"], file.metrics.test[""]);
            close(
                metric(&summary.metrics.test["holdout"], "Loss").entries[0].value,
                56. / 3.,
            );
        }
    }
}

#[test]
fn interrupted_or_failed_training_summary_contains_no_uncompleted_epoch() {
    struct StopOnItem(Interrupter);
    impl MetricLogger for StopOnItem {
        fn log(&mut self, update: &MetricUpdate) -> Result<(), TrainingError> {
            if update.event == "item" {
                self.0.stop(Some("batch boundary"));
            }
            Ok(())
        }
    }
    for fail_restore in [false, true] {
        let interrupter = Interrupter::new();
        let capture = Capture::default();
        let processor = metrics()
            .with_renderer(capture.clone())
            .with_logger(StopOnItem(interrupter.clone()));
        let mut training =
            SupervisedTraining::new(train_loader(), valid_loader(), Default::default())
                .num_epochs(2)
                .summary()
                .with_interrupter(interrupter)
                .with_metrics(processor);
        if fail_restore {
            training = training.checkpoint(2);
        }
        let result = training.fit(Learner::new(
            Regressor::<AD>::new(),
            SgdConfig::new().init(),
            0.01,
        ));
        assert_eq!(result.error.is_some(), fail_restore);
        assert_eq!(result.state.completed_epochs, 0);
        assert_eq!(result.state.optimizer_steps, usize::from(!fail_restore));
        let capture = capture.0.lock().unwrap();
        assert_eq!(capture.events, ["end", "summary"]);
        assert_eq!(capture.summaries[0].epochs, 0);
        assert_eq!(capture.summaries[0].metrics, SummaryMetrics::default());
        assert_eq!(result.summary().metrics, SummaryMetrics::default());
    }
}

#[test]
fn file_summary_recovers_checkpoint_history_and_reports_configuration_errors() {
    struct StopThirdEpoch(Interrupter);
    impl MetricLogger for StopThirdEpoch {
        fn log(&mut self, update: &MetricUpdate) -> Result<(), TrainingError> {
            if update.event == "item" && update.epoch == 3 {
                self.0.stop(Some("partial third epoch"));
            }
            Ok(())
        }
    }
    let directory = Temp::new();
    let run = |epochs, checkpoint, capture: Capture, stop| {
        let interrupter = Interrupter::new();
        let mut processor = metrics()
            .with_logger(SerializedFileLogger::new(&directory.0))
            .with_summary_config(LearnerSummaryConfig::new(&directory.0, &["Loss"]))
            .with_renderer(capture);
        if stop {
            processor = processor.with_logger(StopThirdEpoch(interrupter.clone()));
        }
        let mut training =
            SupervisedTraining::new(train_loader(), valid_loader(), Default::default())
                .num_epochs(epochs)
                .summary()
                .with_async_metrics(processor)
                .with_interrupter(interrupter)
                .with_file_checkpointer(
                    &directory.0,
                    BinFileRecorder::<FullPrecisionSettings>::default(),
                    2,
                );
        if let Some(epoch) = checkpoint {
            training = training.checkpoint(epoch);
        }
        training.fit(Learner::new(
            Regressor::<AD>::new(),
            SgdConfig::new().init(),
            0.01,
        ))
    };
    assert!(run(2, None, Capture::default(), false).error.is_none());
    let partial_capture = Capture::default();
    let partial = run(3, Some(2), partial_capture.clone(), true);
    assert!(partial.error.is_none() && partial.interrupted.is_some());
    assert_eq!(partial.state.completed_epochs, 2);
    {
        let captured = partial_capture.0.lock().unwrap();
        let summary = &captured.summaries[0];
        assert_eq!(summary.epochs, 2);
        assert_eq!(metric(&summary.metrics.train, "Loss").entries.len(), 2);
        assert_eq!(metric(&summary.metrics.valid, "Loss").entries.len(), 2);
    }
    let capture = Capture::default();
    let result = run(4, Some(2), capture.clone(), false);
    assert!(result.error.is_none() && result.interrupted.is_none());
    let capture = capture.0.lock().unwrap();
    assert_eq!(capture.summaries[0].epochs, 4);
    assert_eq!(
        metric(&capture.summaries[0].metrics.valid, "Loss")
            .entries
            .len(),
        4
    );
    close(
        metric(&capture.summaries[0].metrics.valid, "Loss").entries[3].value,
        (reference(4, 1, 0.).0 - 2.).powi(2) * 14. / 3.,
    );
    assert_eq!(
        metric(&result.summary().metrics.valid, "Loss")
            .entries
            .len(),
        2
    );
    let failed_capture = Capture::default();
    let processor = metrics()
        .with_renderer(failed_capture.clone())
        .with_summary_config(LearnerSummaryConfig::new(
            directory.0.join("not-created"),
            &["Loss"],
        ));
    let result = SupervisedTraining::new(train_loader(), valid_loader(), Default::default())
        .num_epochs(1)
        .summary()
        .with_metrics(processor)
        .fit(Learner::new(
            Regressor::<AD>::new(),
            SgdConfig::new().init(),
            0.01,
        ));
    assert!(
        result
            .error
            .as_ref()
            .unwrap()
            .0
            .contains("Artifact directory does not exist")
    );
    assert_eq!(result.state.completed_epochs, 1);
    assert_eq!(failed_capture.0.lock().unwrap().events, ["end"]);
    assert!(failed_capture.0.lock().unwrap().summaries.is_empty());
}
