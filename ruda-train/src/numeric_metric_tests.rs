use super::*;
use ruda_model::{data::dataloader::Progress, tensor::Bool};

fn aggregate(entry: Option<NumericEntry>, expected: f64, expected_count: usize) {
    match entry.unwrap() {
        NumericEntry::Aggregated {
            aggregated_value,
            count,
        } => {
            close(aggregated_value, expected);
            assert_eq!(count, expected_count);
        }
        entry => panic!("Expected aggregate, got {entry:?}"),
    }
}

#[test]
fn numeric_state_tracks_current_running_final_and_zero_count() {
    let mut state = NumericMetricState::new();
    assert!(state.value().is_none());
    assert!(state.current_value().current().is_nan());
    assert!(state.running_value().current().is_nan());
    state.update(10., 3);
    state.update(30., 1);
    aggregate(Some(state.current_value()), 30., 1);
    aggregate(Some(state.running_value()), 15., 4);
    assert_eq!(state.final_value(), NumericEntry::Final(15.));
    let format = || {
        FormatOptions::new(Arc::new("Rate".into()))
            .unit("%")
            .precision(2)
    };
    let update = state.compute_update(format());
    assert_eq!(update.formatted, "epoch 15.00 % - batch 30.00 %");
    assert_eq!(update.serialized, "30,1");
    assert_eq!(state.compute_final(format()).serialized, "15,final");
    state.update(f64::NAN, 0);
    assert!(state.current_value().current().is_nan());
    aggregate(Some(state.running_value()), 15., 4);
    assert_eq!(state.compute_update(format()).serialized, "NaN,0");
    state.reset();
    assert!(state.value().is_none());
    state.update(4., 2);
    state.clear();
    assert!(state.final_value().current().is_nan());
}

#[test]
fn numeric_entries_round_trip_and_format_like_fixed_baseline() {
    for entry in [
        NumericEntry::Value(-1.25),
        NumericEntry::Final(f64::INFINITY),
        NumericEntry::Aggregated {
            aggregated_value: 1.5,
            count: 7,
        },
    ] {
        assert_eq!(
            NumericEntry::deserialize(&entry.serialize()).unwrap(),
            entry
        );
    }
    assert!(
        NumericEntry::deserialize("NaN,0")
            .unwrap()
            .current()
            .is_nan()
    );
    for value in ["", "N/A", "1,-1", "x,3", "1,2,3", "1,final,2"] {
        assert!(NumericEntry::deserialize(value).is_err());
    }
    assert_eq!(format_float(0.1, 2), "1.00e-1");
    assert_eq!(format_float(-1.25, 2), "-1.25e0");
    assert_eq!(format_float(1.25, 2), "1.25");
    let low = NumericEntry::Value(1.);
    let high = NumericEntry::Final(2.);
    assert!(low.better_than(&high, false));
    assert!(high.better_than(&low, true));
    assert!(low.better_than(&low, false));
    assert!(!low.better_than(&low, true));
    let placeholder = SerializedEntry::not_available(Some("%"));
    assert_eq!(placeholder.formatted, "epoch N/A % - batch N/A %");
    assert!(placeholder.is_not_available());
    assert!(SerializedEntry::new(String::new(), String::new()).is_not_available());
}

fn attributes<M: Metric>(metric: M, unit: Option<&str>, higher: bool, precision: usize) {
    assert_eq!(
        metric.attributes(),
        NumericAttributes {
            unit: unit.map(str::to_owned),
            higher_is_better: higher,
        }
        .into()
    );
    let definition = metric.definition();
    assert_eq!(
        definition.metric_id,
        MetricId::new(Arc::new(metric.name().into()))
    );
    assert_eq!(definition.name.as_str(), metric.name());
    assert_eq!(definition.attributes, metric.attributes());
    let format = metric.format_options();
    assert_eq!(format.name().as_str(), metric.name());
    assert_eq!(format.unit_value().as_deref(), unit);
    assert_eq!(format.precision_value(), Some(precision));
}

#[test]
fn built_in_metric_definitions_units_and_directions() {
    attributes(LossMetric::<TestBackend>::new(), None, false, 2);
    attributes(AccuracyMetric::<TestBackend>::new(), Some("%"), true, 2);
    attributes(
        TopKAccuracyMetric::<TestBackend>::new(2),
        Some("%"),
        true,
        2,
    );
    attributes(PerplexityMetric::<TestBackend>::new(), None, false, 2);
    attributes(HammingScore::<TestBackend>::new(), Some("%"), true, 2);
    attributes(
        PrecisionMetric::<TestBackend>::default(),
        Some("%"),
        true,
        2,
    );
    attributes(RecallMetric::<TestBackend>::default(), Some("%"), true, 2);
    attributes(
        FBetaScoreMetric::<TestBackend>::default(),
        Some("%"),
        true,
        2,
    );
    attributes(AucPrMetric::<TestBackend>::binary(), Some("%"), true, 2);
    assert_eq!(
        AurocMetric::<TestBackend>::binary().attributes(),
        MetricAttributes::None
    );
    attributes(CharErrorRate::<TestBackend>::new(), Some("%"), false, 2);
    attributes(WordErrorRate::<TestBackend>::new(), Some("%"), false, 2);
    attributes(BleuScore::<TestBackend>::new(), Some("%"), true, 2);
    attributes(RougeLScore::<TestBackend>::new(), Some("%"), true, 2);
    attributes(LearningRateMetric::new(), None, false, 2);
    attributes(IterationSpeedMetric::new(), Some("iter/sec"), true, 2);
    assert_ne!(
        TopKAccuracyMetric::<TestBackend>::new(1)
            .definition()
            .metric_id,
        TopKAccuracyMetric::<TestBackend>::new(2)
            .definition()
            .metric_id
    );
}

#[test]
fn loss_current_batch_does_not_replace_weighted_epoch_value() {
    let device = Default::default();
    let mut metric = LossMetric::<TestBackend>::new();
    metric
        .update(&LossInput {
            tensor: Tensor::from_floats([1., 2., 3.], &device),
        })
        .unwrap();
    metric
        .update(&LossInput {
            tensor: Tensor::from_floats([10.], &device),
        })
        .unwrap();
    aggregate(metric.current_value(), 10., 1);
    aggregate(metric.running_value(), 4., 4);
    close(metric.value().unwrap(), 4.);
    metric.compute().unwrap();
    assert_eq!(metric.final_value(), NumericEntry::Final(4.));
    assert_eq!(metric.serialized_final().serialized, "4,final");
    metric.clear();
    assert!(metric.value().is_none());
    assert!(metric.current_value().unwrap().current().is_nan());
}

#[test]
fn confusion_current_and_running_use_separate_raw_counts() {
    let device = Default::default();
    let first = ConfusionStatsInput::<TestBackend> {
        predictions: Tensor::from_floats([[0.9], [0.8], [0.7], [0.1]], &device),
        targets: Tensor::<TestBackend, 2, Bool>::from_data(
            [[true], [false], [false], [true]],
            &device,
        ),
    };
    let second = ConfusionStatsInput::<TestBackend> {
        predictions: Tensor::from_floats([[0.9], [0.1]], &device),
        targets: Tensor::<TestBackend, 2, Bool>::from_data([[true], [false]], &device),
    };
    let metrics: Vec<(
        Box<dyn Metric<Input = ConfusionStatsInput<TestBackend>>>,
        f64,
    )> = vec![
        (Box::new(PrecisionMetric::binary(0.5)), 50.),
        (Box::new(RecallMetric::binary(0.5)), 200. / 3.),
        (Box::new(FBetaScoreMetric::binary(1., 0.5)), 400. / 7.),
    ];
    for (mut metric, expected) in metrics {
        metric.update(&first).unwrap();
        metric.update(&second).unwrap();
        aggregate(metric.current_value(), 100., 2);
        aggregate(metric.running_value(), expected, 6);
        close(metric.value().unwrap(), expected);
        metric.compute().unwrap();
        close(metric.final_value().current(), expected);
        metric.clear();
        assert!(metric.current_value().is_none() && metric.running_value().is_none());
    }
}

#[test]
fn perplexity_current_and_final_preserve_log_likelihood_aggregation() {
    let device = Default::default();
    let mut metric = PerplexityMetric::<TestBackend>::new().with_pad_token(0);
    metric
        .update(&PerplexityInput {
            outputs: Tensor::from_floats(
                [
                    [0.5f32.ln(), 0.5f32.ln()],
                    [0.75f32.ln(), 0.25f32.ln()],
                    [0., 0.],
                ],
                &device,
            ),
            targets: Tensor::from_ints([1, 1, 0], &device),
        })
        .unwrap();
    aggregate(metric.current_value(), 8f64.sqrt(), 2);
    metric
        .update(&PerplexityInput {
            outputs: Tensor::from_floats([[0.875f32.ln(), 0.125f32.ln()]], &device),
            targets: Tensor::from_ints([1], &device),
        })
        .unwrap();
    aggregate(metric.current_value(), 8., 1);
    aggregate(metric.running_value(), 4., 3);
    close(metric.final_value().current(), 4.);
    metric.clear();
    assert!(metric.current_value().unwrap().current().is_nan());
    assert!(metric.final_value().current().is_infinite());
}

#[test]
fn rank_metrics_have_no_batch_scalar_and_publish_epoch_final() {
    let device = Default::default();
    let first = ConfusionStatsInput::<TestBackend> {
        predictions: Tensor::from_floats([[0.9], [0.8]], &device),
        targets: Tensor::<TestBackend, 2, Bool>::from_data([[true], [false]], &device),
    };
    let second = ConfusionStatsInput::<TestBackend> {
        predictions: Tensor::from_floats([[0.2], [0.1]], &device),
        targets: Tensor::<TestBackend, 2, Bool>::from_data([[true], [false]], &device),
    };
    let metrics: Vec<(
        Box<dyn Metric<Input = ConfusionStatsInput<TestBackend>>>,
        f64,
    )> = vec![
        (Box::new(AurocMetric::binary()), 75.),
        (Box::new(AucPrMetric::binary()), 250. / 3.),
    ];
    for (mut metric, expected) in metrics {
        metric.update(&first).unwrap();
        metric.update(&second).unwrap();
        assert!(metric.current_value().is_none() && metric.running_value().is_none());
        assert!(metric.serialized_update().is_not_available());
        metric.compute().unwrap();
        close(metric.current_value().unwrap().current(), expected);
        close(metric.running_value().unwrap().current(), expected);
        let snapshot = metric.snapshot(true);
        close(snapshot.final_value.unwrap().current(), expected);
        assert!(snapshot.serialized.formatted.ends_with("batch N/A %"));
        assert!(snapshot.serialized.serialized.ends_with(",final"));
        metric.update(&first).unwrap();
        assert!(metric.current_value().is_none());
        metric.clear();
        assert!(metric.value().is_none());
    }
}

#[test]
fn sequence_metrics_publish_current_and_weighted_running_counts() {
    let device = Default::default();
    let mut metric = CharErrorRate::<TestBackend>::new().with_pad_token(0);
    metric
        .update(&CerInput {
            outputs: Tensor::from_ints([[1, 3, 0]], &device),
            targets: Tensor::from_ints([[1, 2, 0]], &device),
        })
        .unwrap();
    metric
        .update(&CerInput {
            outputs: Tensor::from_ints([[1, 2, 3, 4]], &device),
            targets: Tensor::from_ints([[1, 2, 3, 4]], &device),
        })
        .unwrap();
    aggregate(metric.current_value(), 0., 4);
    aggregate(metric.running_value(), 100. / 6., 6);
    close(metric.final_value().current(), 100. / 6.);
}

#[test]
fn learning_rate_current_is_not_the_running_average() {
    let mut metric = LearningRateMetric::new();
    for (iteration, lr) in [(1, 0.1), (2, 0.01)] {
        metric
            .update_with_metadata(
                &(),
                &MetricMetadata {
                    epoch: 1,
                    iteration,
                    progress: Progress {
                        items_processed: iteration,
                        items_total: 2,
                    },
                    learning_rate: Some(lr),
                },
            )
            .unwrap();
    }
    aggregate(metric.current_value(), 0.01, 1);
    aggregate(metric.running_value(), 0.055, 2);
    assert_eq!(metric.serialized_update().serialized, "0.01,1");
}

#[test]
fn processor_snapshots_are_split_scoped_and_final_only_after_compute() {
    let device = Default::default();
    let mut processor = MetricsProcessor::<
        ClassificationOutput<TestBackend>,
        ClassificationOutput<TestBackend>,
    >::new()
    .metric_train(LossMetric::<TestBackend>::new())
    .metric_valid(AurocMetric::<TestBackend>::binary());
    processor
        .process_train(LearnerEvent::StartSplit {
            epoch: 1,
            total_items: 2,
        })
        .unwrap();
    processor
        .process_train(LearnerEvent::ProcessedItem(TrainingItem {
            item: ClassificationOutput::new(
                Tensor::from_floats([2., 4.], &device),
                Tensor::from_floats([[1., 0.], [0., 1.]], &device),
                Tensor::from_ints([0, 1], &device),
            ),
            progress: Progress {
                items_processed: 2,
                items_total: 2,
            },
            iteration: 1,
            learning_rate: Some(0.1),
        }))
        .unwrap();
    let snapshots = processor.metric_snapshots(Split::Train);
    aggregate(snapshots[0].current.clone(), 3., 2);
    assert!(snapshots[0].final_value.is_none());
    assert!(processor.metric_snapshots(Split::Test).is_empty());
    processor
        .process_train(LearnerEvent::EndSplit {
            epoch: 1,
            completed: true,
        })
        .unwrap();
    assert_eq!(
        processor.metric_snapshots(Split::Train)[0].final_value,
        Some(NumericEntry::Final(3.))
    );
    assert!(
        processor.metric_snapshots(Split::Valid)[0]
            .current
            .is_none()
    );
    processor
        .process_train(LearnerEvent::StartSplit {
            epoch: 2,
            total_items: 2,
        })
        .unwrap();
    processor
        .process_train(LearnerEvent::EndSplit {
            epoch: 2,
            completed: false,
        })
        .unwrap();
    assert!(
        processor.metric_snapshots(Split::Train)[0]
            .final_value
            .is_none()
    );
    assert!(processor.store().metrics(2, Split::Train).is_empty());
}

#[cfg(feature = "vision")]
#[test]
fn vision_definitions_and_dice_current_preserve_global_overlap() {
    use crate::metric::vision::*;
    let device = Default::default();
    attributes(
        PsnrMetric::<TestBackend>::new(PsnrMetricConfig::new(1.)),
        Some("dB"),
        true,
        2,
    );
    attributes(
        SsimMetric::<TestBackend>::new(SsimMetricConfig::new(1.)),
        None,
        true,
        4,
    );
    attributes(
        MsSsimMetric::<TestBackend>::new(MsSsimMetricConfig::new(1.), &device),
        None,
        true,
        4,
    );
    attributes(DiceMetric::<TestBackend>::new(), None, true, 4);
    let mut metric = DiceMetric::<TestBackend, 3>::new();
    metric
        .update(&DiceInput::new(
            Tensor::from_ints([[[1, 1, 1, 1]]], &device),
            Tensor::from_ints([[[1, 1, 1, 1]]], &device),
        ))
        .unwrap();
    metric
        .update(&DiceInput::new(
            Tensor::from_ints([[[1, 0, 0, 0]]], &device),
            Tensor::from_ints([[[0, 0, 0, 0]]], &device),
        ))
        .unwrap();
    aggregate(metric.current_value(), 1e-7 / (1. + 1e-7), 1);
    aggregate(metric.running_value(), (8. + 1e-7) / (9. + 1e-7), 2);
    close(metric.final_value().current(), (8. + 1e-7) / (9. + 1e-7));
    metric.clear();
    assert!(metric.value().is_none());
    assert!(metric.current_value().unwrap().current().is_nan());
}
