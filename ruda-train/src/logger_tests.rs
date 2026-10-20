use super::*;
use crate::logger::FileMetricLogger as SerializedFileLogger;
use crate::logger::{AsyncLogger, FileLogger, InMemoryLogger, InMemoryMetricLogger, Logger};
use ruda_model::data::dataloader::Progress;

fn item(
    values: &[f32],
    processed: usize,
    total: usize,
) -> LearnerEvent<RegressionOutput<TestBackend>> {
    let device = Default::default();
    LearnerEvent::ProcessedItem(TrainingItem {
        item: RegressionOutput {
            loss: Tensor::from_data(TensorData::new(values.to_vec(), [values.len()]), &device),
            output: Tensor::zeros([values.len(), 1], &device),
            targets: Tensor::zeros([values.len(), 1], &device),
        },
        progress: Progress {
            items_processed: processed,
            items_total: total,
        },
        iteration: processed,
        learning_rate: Some(0.01),
    })
}

#[test]
fn async_loggers_preserve_order_flush_and_drain_on_drop() {
    let directory = Temp::new();
    let path = directory.0.join("ordered.log");
    let mut file = AsyncLogger::new(FileLogger::new(&path).unwrap());
    for value in 0..100 {
        file.log(value).unwrap();
    }
    file.sync().unwrap();
    let expected: String = (0..100).map(|value| format!("{value}\n")).collect();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), expected);
    file.log(100).unwrap();
    drop(file);
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        format!("{expected}100\n")
    );
    let memory = InMemoryLogger::default();
    let mut logger = AsyncLogger::new(memory.clone());
    for value in ["first", "second", "third"] {
        logger.log(value.to_owned()).unwrap();
    }
    logger.finish().unwrap();
    assert_eq!(memory.values(), ["first", "second", "third"]);
}

#[test]
fn async_logger_returns_final_flush_errors() {
    struct FailingFlush;
    impl Logger<String> for FailingFlush {
        fn log(&mut self, _: String) -> Result<(), TrainingError> {
            Ok(())
        }
        fn flush(&mut self) -> Result<(), TrainingError> {
            Err(TrainingError("flush failed".into()))
        }
    }
    let mut logger = AsyncLogger::new(FailingFlush);
    logger.log("queued".into()).unwrap();
    assert_eq!(logger.sync().unwrap_err().0, "flush failed");
    assert_eq!(logger.finish().unwrap_err().0, "flush failed");
}

#[test]
fn async_logger_propagates_worker_panic_to_finish() {
    struct Panicking;
    impl Logger<()> for Panicking {
        fn log(&mut self, _: ()) -> Result<(), TrainingError> {
            panic!("logger panic payload")
        }
    }
    let failure = std::panic::catch_unwind(|| {
        let mut logger = AsyncLogger::new(Panicking);
        logger.log(()).unwrap();
        logger.finish().unwrap();
    })
    .unwrap_err();
    assert_eq!(
        failure.downcast_ref::<&str>(),
        Some(&"logger panic payload")
    );
}

#[test]
fn metric_logs_keep_batch_weights_finals_splits_and_epochs() {
    let directory = Temp::new();
    let mut memory = InMemoryMetricLogger::new();
    let mut processor = metrics()
        .with_logger(SerializedFileLogger::new(&directory.0))
        .with_logger(memory.clone())
        .with_logger(FileMetricLogger::new(directory.0.join("legacy.jsonl")).unwrap());
    for epoch in [1, 2] {
        processor
            .process_train(LearnerEvent::StartSplit {
                epoch,
                total_items: 4,
            })
            .unwrap();
        processor.process_train(item(&[1., 2., 3.], 3, 4)).unwrap();
        processor.process_train(item(&[10.], 4, 4)).unwrap();
        processor
            .process_train(LearnerEvent::EndSplit {
                epoch,
                completed: true,
            })
            .unwrap();
        processor
            .process_valid(LearnerEvent::StartSplit {
                epoch,
                total_items: 1,
            })
            .unwrap();
        processor.process_valid(item(&[7.], 1, 1)).unwrap();
        processor
            .process_valid(LearnerEvent::EndSplit {
                epoch,
                completed: true,
            })
            .unwrap();
        processor
            .process_train(LearnerEvent::EndEpoch(epoch))
            .unwrap();
        processor
            .process_valid(LearnerEvent::EndEpoch(epoch))
            .unwrap();
    }
    processor.flush().unwrap();
    let mut file = SerializedFileLogger::new(&directory.0);
    for epoch in [1, 2] {
        for (split, expected) in [
            (Split::Train, vec!["2,3", "10,1", "4,final"]),
            (Split::Valid, vec!["7,1", "7,final"]),
        ] {
            assert_eq!(memory.entries("Loss", epoch, split), expected);
            let parsed: Vec<_> = expected
                .iter()
                .map(|value| NumericEntry::deserialize(value).unwrap())
                .collect();
            assert_eq!(file.read_numeric("Loss", epoch, split).unwrap(), parsed);
            assert_eq!(memory.read_numeric("Loss", epoch, split).unwrap(), parsed);
        }
    }
    assert!(
        memory
            .read_numeric("Loss", 3, Split::Train)
            .unwrap()
            .is_empty()
    );
    assert!(
        file.read_numeric("Loss", 1, Split::Test)
            .unwrap()
            .is_empty()
    );
    let definition = LossMetric::<TestBackend>::new().definition();
    assert_eq!(
        memory.definition(&definition.metric_id).unwrap().attributes,
        definition.attributes
    );
    let legacy = std::fs::read_to_string(directory.0.join("legacy.jsonl")).unwrap();
    let updates: Vec<serde_json::Value> = legacy
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(updates.len(), 18);
    assert_eq!(
        updates[2]["metrics"][0],
        serde_json::json!({"name": "Loss", "value": 4.0})
    );
    assert_eq!(updates[0].as_object().unwrap().len(), 8);
    assert_eq!(updates[0]["event"], "start_split");
}

#[test]
fn rank_metric_placeholders_are_not_written_as_numeric_files() {
    let directory = Temp::new();
    let mut memory = InMemoryMetricLogger::new();
    let metric = AurocMetric::<TestBackend>::binary();
    let name = metric.name().to_owned();
    let mut processor = MetricsProcessor::<ClassificationOutput<TestBackend>, ()>::new()
        .metric_train(metric)
        .with_logger(SerializedFileLogger::new(&directory.0))
        .with_logger(memory.clone());
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
            progress: Progress {
                items_processed: 4,
                items_total: 4,
            },
            iteration: 1,
            learning_rate: None,
        }))
        .unwrap();
    processor.flush().unwrap();
    assert!(!directory.0.join("train").exists());
    assert_eq!(memory.entries(&name, 1, Split::Train), ["N/A"]);
    assert!(
        memory
            .read_numeric(&name, 1, Split::Train)
            .unwrap()
            .is_empty()
    );
    processor
        .process_train(LearnerEvent::EndSplit {
            epoch: 1,
            completed: true,
        })
        .unwrap();
    assert_eq!(
        SerializedFileLogger::new(&directory.0)
            .read_numeric(&name, 1, Split::Train)
            .unwrap(),
        [NumericEntry::Final(75.)]
    );
    assert_eq!(
        memory.read_numeric(&name, 1, Split::Train).unwrap(),
        [NumericEntry::Final(75.)]
    );
}

#[test]
fn interrupted_metric_logs_do_not_publish_a_final_value() {
    let directory = Temp::new();
    let mut processor = metrics().with_logger(SerializedFileLogger::new(&directory.0));
    processor
        .process_train(LearnerEvent::StartSplit {
            epoch: 7,
            total_items: 4,
        })
        .unwrap();
    processor.process_train(item(&[1., 2., 3.], 3, 4)).unwrap();
    processor
        .process_train(LearnerEvent::EndSplit {
            epoch: 7,
            completed: false,
        })
        .unwrap();
    processor.flush().unwrap();
    assert_eq!(
        SerializedFileLogger::new(&directory.0)
            .read_numeric("Loss", 7, Split::Train)
            .unwrap(),
        [NumericEntry::Aggregated {
            aggregated_value: 2.,
            count: 3
        }]
    );
    assert!(
        processor
            .store()
            .find_metric("Loss", 7, Split::Train)
            .is_none()
    );
}

#[test]
fn standalone_evaluation_serializes_test_metrics_in_eval_directory() {
    let directory = Temp::new();
    let mut memory = InMemoryMetricLogger::new();
    let mut processor = MetricsProcessor::<(), RegressionOutput<TestBackend>>::new()
        .metric_valid(LossMetric::new())
        .for_evaluation()
        .with_logger(SerializedFileLogger::new_eval(&directory.0))
        .with_logger(memory.clone());
    let result = evaluate(
        Regressor::<AD>::new().valid(),
        valid_loader(),
        &Default::default(),
        &mut processor,
        &Interrupter::new(),
    );
    assert!(result.completed && result.error.is_none());
    assert!(directory.0.join("Loss.log").is_file());
    assert!(!directory.0.join("test").exists());
    let values = SerializedFileLogger::new_eval(&directory.0)
        .read_numeric("Loss", 1, Split::Test)
        .unwrap();
    assert_eq!(values.len(), 3);
    close(values.last().unwrap().current(), 56. / 3.);
    assert_eq!(memory.read_numeric("Loss", 1, Split::Test).unwrap(), values);
    assert!(
        memory
            .read_numeric("Loss", 1, Split::Valid)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn file_metric_logger_reports_io_and_parse_errors() {
    let directory = Temp::new();
    let blocked = directory.0.join("not-a-directory");
    std::fs::write(&blocked, "occupied").unwrap();
    let mut processor = metrics().with_logger(SerializedFileLogger::new(&blocked));
    processor
        .process_train(LearnerEvent::StartSplit {
            epoch: 1,
            total_items: 1,
        })
        .unwrap();
    assert!(processor.process_train(item(&[1.], 1, 1)).is_err());
    let path = directory.0.join("train/epoch-1");
    std::fs::create_dir_all(&path).unwrap();
    std::fs::write(path.join("Loss.log"), "1,2\nnot-numeric\n").unwrap();
    assert!(
        SerializedFileLogger::new(&directory.0)
            .read_numeric("Loss", 1, Split::Train)
            .is_err()
    );
    assert!(FileLogger::new(&directory.0).is_err());
}

#[test]
fn async_training_with_serialized_loggers_matches_independent_sgd() {
    let directory = Temp::new();
    let mut memory = InMemoryMetricLogger::new();
    let processor = metrics()
        .with_logger(SerializedFileLogger::new(&directory.0))
        .with_logger(memory.clone());
    let result = SupervisedTraining::new(train_loader(), valid_loader(), Default::default())
        .num_epochs(3)
        .with_async_metrics(processor)
        .fit(Learner::new(
            Regressor::<AD>::new(),
            SgdConfig::new().init(),
            0.01,
        ));
    assert!(result.error.is_none() && result.interrupted.is_none());
    let expected = reference(3, 1, 0.);
    close(scalar(&result.model), expected.0);
    assert_eq!(result.state.optimizer_steps, expected.1);
    let mut file = SerializedFileLogger::new(&directory.0);
    for epoch in 1..=3 {
        for (split, batches) in [(Split::Train, 3), (Split::Valid, 2)] {
            let values = memory.read_numeric("Loss", epoch, split).unwrap();
            assert_eq!(values.len(), batches + 1);
            assert_eq!(file.read_numeric("Loss", epoch, split).unwrap(), values);
            assert!(matches!(values.last(), Some(NumericEntry::Final(_))));
            close(
                values.last().unwrap().current(),
                result.metrics.find_metric("Loss", epoch, split).unwrap(),
            );
            let (sum, count) = values[..batches].iter().fold((0., 0), |(sum, n), value| {
                let NumericEntry::Aggregated {
                    aggregated_value,
                    count,
                } = value
                else {
                    panic!("Missing batch weight")
                };
                (sum + aggregated_value * *count as f64, n + count)
            });
            close(sum / count as f64, values.last().unwrap().current());
        }
    }
}

#[cfg(feature = "sys-metrics")]
#[test]
fn actual_cuda_text_reaches_metric_loggers_without_invented_numeric_final() {
    let directory = Temp::new();
    let mut memory = InMemoryMetricLogger::new();
    let metric = CudaMetric::new();
    let name = metric.name().to_owned();
    let mut processor = MetricsProcessor::<(), ()>::new()
        .metric_train(metric)
        .with_logger(SerializedFileLogger::new(&directory.0))
        .with_logger(memory.clone());
    processor
        .process_train(LearnerEvent::StartSplit {
            epoch: 1,
            total_items: 1,
        })
        .unwrap();
    processor
        .process_train(LearnerEvent::ProcessedItem(TrainingItem {
            item: (),
            progress: Progress {
                items_processed: 1,
                items_total: 1,
            },
            iteration: 1,
            learning_rate: None,
        }))
        .unwrap();
    let batch_entry = processor
        .metric_snapshots(Split::Train)
        .remove(0)
        .serialized
        .serialized;
    processor
        .process_train(LearnerEvent::EndSplit {
            epoch: 1,
            completed: true,
        })
        .unwrap();
    let snapshot = processor.metric_snapshots(Split::Train).remove(0);
    assert!(snapshot.final_value.is_none());
    let entries = memory.entries(&name, 1, Split::Train);
    assert_eq!(
        entries,
        vec![batch_entry, snapshot.serialized.serialized.clone()]
    );
    if std::env::var("RUDA_REQUIRE_GPU").as_deref() == Ok("1") {
        assert_ne!(entries[0], "Unavailable");
        assert!(snapshot.serialized.formatted.contains("GPU #0 - Memory"));
    }
    let path = directory
        .0
        .join("train/epoch-1")
        .join(format!("{}.log", name.replace(' ', "_")));
    assert_eq!(
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .collect::<Vec<_>>(),
        entries
    );
    assert!(
        memory
            .read_numeric(&name, 1, Split::Train)
            .unwrap()
            .is_empty()
    );
    assert!(
        SerializedFileLogger::new(&directory.0)
            .read_numeric(&name, 1, Split::Train)
            .is_err()
    );
}
