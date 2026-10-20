use super::*;
use ruda_model::data::dataloader::Progress;
use ruda_model::tensor::Bool;
use std::sync::mpsc;

#[test]
fn padded_accuracy_and_topk_ignore_padding_across_batches() {
    let device = Default::default();
    let outputs = Tensor::<TestBackend, 2>::from_floats(
        [[4., 3., 2.], [2., 4., 3.], [4., 1., 3.], [4., 2., 1.]],
        &device,
    );
    let targets = Tensor::from_ints([0, 1, 2, 1], &device);
    let mut accuracy = AccuracyMetric::new().with_pad_token(0);
    let mut topk = TopKAccuracyMetric::new(2).with_pad_token(0);
    accuracy
        .update(&AccuracyInput {
            predictions: outputs.clone(),
            targets: targets.clone(),
        })
        .unwrap();
    topk.update(&TopKAccuracyInput { outputs, targets })
        .unwrap();
    close(accuracy.value().unwrap(), 100. / 3.);
    close(topk.value().unwrap(), 100.);
    let output = ClassificationOutput::new(
        Tensor::zeros([1], &device),
        Tensor::from_floats([[1., 2., 3.]], &device),
        Tensor::from_ints([2], &device),
    );
    accuracy.update(&output.adapt()).unwrap();
    topk.update(&output.adapt()).unwrap();
    close(accuracy.value().unwrap(), 50.);
    close(topk.value().unwrap(), 100.);
    accuracy.clear();
    topk.clear();
    let output = ClassificationOutput::new(
        Tensor::zeros([1], &device),
        Tensor::from_floats([[1., 2., 3.]], &device),
        Tensor::from_ints([0], &device),
    );
    accuracy.update(&output.adapt()).unwrap();
    topk.update(&output.adapt()).unwrap();
    assert!(accuracy.value().is_none() && topk.value().is_none());
}

#[test]
fn perplexity_accumulates_log_likelihood_not_batch_perplexities() {
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
    metric
        .update(&PerplexityInput {
            outputs: Tensor::from_floats([[0.875f32.ln(), 0.125f32.ln()]], &device),
            targets: Tensor::from_ints([1], &device),
        })
        .unwrap();
    close(metric.value().unwrap(), 4.);
    metric.clear();
    metric
        .update(&PerplexityInput {
            outputs: Tensor::zeros([1, 2], &device),
            targets: Tensor::from_ints([0], &device),
        })
        .unwrap();
    assert_eq!(metric.value(), Some(f64::INFINITY));
}

#[test]
fn hamming_sigmoid_and_strict_threshold() {
    let device = Default::default();
    let mut metric = HammingScore::<TestBackend>::new().with_sigmoid(true);
    metric
        .update(&HammingScoreInput {
            outputs: Tensor::from_floats([[0., 2., -2.], [1., -1., 0.]], &device),
            targets: Tensor::from_ints([[0, 1, 0], [1, 0, 1]], &device),
        })
        .unwrap();
    close(metric.value().unwrap(), 500. / 6.);
    let mut metric = HammingScore::<TestBackend>::new().with_threshold(0.5);
    metric
        .update(&HammingScoreInput {
            outputs: Tensor::from_floats([[0.5, 0.6]], &device),
            targets: Tensor::from_ints([[0, 1]], &device),
        })
        .unwrap();
    close(metric.value().unwrap(), 100.);
}

fn confusion<const N: usize, const C: usize>(
    scores: [[f32; C]; N],
    targets: [[bool; C]; N],
) -> ConfusionStatsInput<TestBackend> {
    ConfusionStatsInput {
        predictions: Tensor::from_floats(scores, &Default::default()),
        targets: Tensor::<TestBackend, 2, Bool>::from_data(targets, &Default::default()),
    }
}

#[test]
fn confusion_metrics_accumulate_counts_and_preserve_beta_default() {
    let first = confusion([[0.9], [0.1]], [[true], [false]]);
    let second = confusion([[0.8]], [[false]]);
    let mut precision = PrecisionMetric::binary(0.5);
    let mut recall = RecallMetric::binary(0.5);
    let mut f1 = FBetaScoreMetric::binary(1., 0.5);
    let mut f2 = FBetaScoreMetric::binary(2., 0.5);
    let mut default = FBetaScoreMetric::default();
    for batch in [&first, &second] {
        precision.update(batch).unwrap();
        recall.update(batch).unwrap();
        f1.update(batch).unwrap();
        f2.update(batch).unwrap();
        default.update(batch).unwrap();
    }
    close(precision.value().unwrap(), 50.);
    close(recall.value().unwrap(), 100.);
    close(f1.value().unwrap(), 200. / 3.);
    close(f2.value().unwrap(), 500. / 6.);
    close(default.value().unwrap(), 50.);
    assert!(default.name().starts_with("FBetaScore (0)"));
    precision.clear();
    recall.clear();
    f1.clear();
    assert!(precision.value().is_none() && recall.value().is_none() && f1.value().is_none());
}

#[test]
fn confusion_macro_micro_and_undefined_fbeta_classes() {
    let data = confusion(
        [
            [0.8, 0.3, 0.1],
            [0.7, 0.6, 0.1],
            [0.2, 0.9, 0.1],
            [0.1, 0.2, 0.1],
        ],
        [
            [true, false, false],
            [false, true, false],
            [false, true, false],
            [true, false, false],
        ],
    );
    for (reduction, p, f) in [
        (ClassReduction::Macro, 50., 75.),
        (ClassReduction::Micro, 75., 75.),
    ] {
        let mut precision = PrecisionMetric::multilabel(0.5, reduction);
        let mut recall = RecallMetric::multilabel(0.5, reduction);
        let mut f1 = FBetaScoreMetric::multilabel(1., 0.5, reduction);
        precision.update(&data).unwrap();
        recall.update(&data).unwrap();
        f1.update(&data).unwrap();
        close(precision.value().unwrap(), p);
        close(recall.value().unwrap(), p);
        close(f1.value().unwrap(), f);
    }
    let mut f1 = FBetaScoreMetric::binary(1., 0.5);
    f1.update(&confusion([[0.1]], [[false]])).unwrap();
    assert!(f1.value().is_none());
}

#[test]
fn multiclass_topk_confusion_and_classification_adaptor() {
    let device = Default::default();
    let output = ClassificationOutput::<TestBackend>::new(
        Tensor::zeros([1], &device),
        Tensor::from_floats([[0.8, 0.5, 0.1], [0.2, 0.3, 0.9]], &device),
        Tensor::from_ints([1, 2], &device),
    );
    let input: ConfusionStatsInput<_> = output.adapt();
    let mut precision = PrecisionMetric::multiclass(2, ClassReduction::Micro);
    let mut recall = RecallMetric::multiclass(2, ClassReduction::Micro);
    let mut f1 = FBetaScoreMetric::multiclass(1., 2, ClassReduction::Micro);
    precision.update(&input).unwrap();
    recall.update(&input).unwrap();
    f1.update(&input).unwrap();
    close(precision.value().unwrap(), 50.);
    close(recall.value().unwrap(), 100.);
    close(f1.value().unwrap(), 200. / 3.);
}

#[test]
fn auc_metrics_are_epoch_global_and_handle_ties() {
    let mut roc = AurocMetric::binary();
    let mut ap = AucPrMetric::binary();
    for input in [
        confusion([[0.8], [0.8]], [[true], [false]]),
        confusion([[0.2], [0.1]], [[true], [false]]),
    ] {
        roc.update(&input).unwrap();
        ap.update(&input).unwrap();
        assert!(roc.value().is_none() && ap.value().is_none());
    }
    roc.compute().unwrap();
    ap.compute().unwrap();
    close(roc.value().unwrap(), 62.5);
    close(ap.value().unwrap(), 100. * (0.5 + 2. / 3.) / 2.);
    roc.clear();
    ap.clear();
    assert!(roc.value().is_none() && ap.value().is_none());
}

fn rank_reference(
    scores: &[f64],
    targets: &[bool],
    n: usize,
    c: usize,
    micro: bool,
    ap: bool,
) -> f64 {
    let groups: Vec<Vec<usize>> = if micro {
        vec![(0..n * c).collect()]
    } else {
        (0..c)
            .map(|col| (0..n).map(|row| row * c + col).collect())
            .collect()
    };
    let mut results = Vec::new();
    for indices in groups {
        let positives: Vec<_> = indices.iter().copied().filter(|&i| targets[i]).collect();
        let negatives: Vec<_> = indices.iter().copied().filter(|&i| !targets[i]).collect();
        if positives.is_empty() || (!ap && negatives.is_empty()) {
            continue;
        }
        let mut sum = 0.;
        for &pos in &positives {
            if ap {
                let above: Vec<_> = indices
                    .iter()
                    .copied()
                    .filter(|&i| scores[i] >= scores[pos])
                    .collect();
                sum += above.iter().filter(|&&i| targets[i]).count() as f64 / above.len() as f64;
            } else {
                for &neg in &negatives {
                    sum += if scores[pos] > scores[neg] {
                        1.
                    } else if scores[pos] == scores[neg] {
                        0.5
                    } else {
                        0.
                    };
                }
            }
        }
        results.push(sum / positives.len() as f64 / if ap { 1. } else { negatives.len() as f64 });
    }
    100. * if results.is_empty() {
        0.5
    } else {
        results.iter().sum::<f64>() / results.len() as f64
    }
}

#[test]
fn auc_macro_micro_match_independent_pair_and_threshold_oracles() {
    let scores = [
        [0.9, 0.3, 0.4],
        [0.2, 0.8, 0.4],
        [0.5, 0.6, 0.4],
        [0.5, 0.1, 0.4],
    ];
    let targets = [
        [true, false, false],
        [false, true, false],
        [true, false, false],
        [false, true, false],
    ];
    let flat_scores: Vec<f64> = scores.iter().flatten().map(|&value| value as f64).collect();
    let flat_targets: Vec<bool> = targets.iter().flatten().copied().collect();
    for reduction in [ClassReduction::Macro, ClassReduction::Micro] {
        let mut roc = AurocMetric::multiclass(reduction);
        let mut ap = AucPrMetric::multilabel(reduction);
        let data = confusion(scores, targets);
        roc.update(&data).unwrap();
        ap.update(&data).unwrap();
        roc.compute().unwrap();
        ap.compute().unwrap();
        let micro = matches!(reduction, ClassReduction::Micro);
        close(
            roc.value().unwrap(),
            rank_reference(&flat_scores, &flat_targets, 4, 3, micro, false),
        );
        close(
            ap.value().unwrap(),
            rank_reference(&flat_scores, &flat_targets, 4, 3, micro, true),
        );
    }
}

#[test]
fn auroc_nan_pairs_and_signed_zero_ties() {
    let data = confusion(
        [[-0.0], [0.0], [0.5], [f32::NAN], [0.5]],
        [[true], [false], [true], [true], [false]],
    );
    let mut roc = AurocMetric::binary();
    roc.update(&data).unwrap();
    roc.compute().unwrap();
    close(roc.value().unwrap(), 100. / 3.);
}

#[test]
fn auc_empty_and_undefined_epochs_preserve_baseline_fallback() {
    let mut roc = AurocMetric::<TestBackend>::default();
    let mut ap = AucPrMetric::<TestBackend>::default();
    roc.compute().unwrap();
    ap.compute().unwrap();
    assert!(roc.value().is_none() && ap.value().is_none());
    let input = confusion([[0.1], [0.2]], [[false], [false]]);
    roc.update(&input).unwrap();
    ap.update(&input).unwrap();
    roc.compute().unwrap();
    ap.compute().unwrap();
    assert_eq!(roc.value(), Some(50.));
    assert_eq!(ap.value(), Some(50.));
}

#[test]
fn cer_and_wer_preserve_distinct_padding_rules() {
    let device = Default::default();
    let outputs =
        Tensor::<TestBackend, 2, ruda_model::tensor::Int>::from_ints([[1, 3, 0, 0]], &device);
    let targets = Tensor::from_ints([[1, 0, 2, 0]], &device);
    let mut cer = CharErrorRate::new().with_pad_token(0);
    let mut wer = WordErrorRate::new().with_pad_token(0);
    cer.update(&CerInput {
        outputs: outputs.clone(),
        targets: targets.clone(),
    })
    .unwrap();
    wer.update(&WerInput { outputs, targets }).unwrap();
    close(cer.value().unwrap(), 50.);
    close(wer.value().unwrap(), 100.);
    let outputs = Tensor::from_ints([[1, 2, 3]], &device);
    let targets = outputs.clone();
    cer.update(&CerInput {
        outputs: outputs.clone(),
        targets: targets.clone(),
    })
    .unwrap();
    wer.update(&WerInput { outputs, targets }).unwrap();
    close(cer.value().unwrap(), 20.);
    close(wer.value().unwrap(), 25.);
}

#[test]
fn bleu_clipping_smoothing_brevity_and_batch_aggregation() {
    let device = Default::default();
    let input = BleuInput::<TestBackend> {
        outputs: Tensor::from_ints([[1, 1, 1, 1]], &device),
        targets: Tensor::from_ints([[1, 2, 3, 4]], &device),
    };
    let mut unigram = BleuScore::with_max_n(1);
    unigram.update(&input).unwrap();
    close(unigram.value().unwrap(), 25.);
    for (smoothing, expected) in [
        (BleuSmoothing::None, 0.),
        (
            BleuSmoothing::AddEpsilon(0.1),
            100. * (0.25f64 * 0.1 / 3.).sqrt(),
        ),
        (BleuSmoothing::Exponential, 100. * (0.25f64 / 6.).sqrt()),
    ] {
        let mut metric = BleuScore::with_max_n(2).with_smoothing(smoothing);
        metric.update(&input).unwrap();
        close(metric.value().unwrap(), expected);
    }
    let shortened = BleuInput {
        outputs: Tensor::from_ints([[1, 2, 0, 0]], &device),
        targets: input.targets.clone(),
    };
    let mut metric = BleuScore::with_max_n(1).with_pad_token(0);
    metric.update(&shortened).unwrap();
    close(metric.value().unwrap(), 100. * (-1f64).exp());
    let mut short = BleuScore::new()
        .with_pad_token(0)
        .with_smoothing(BleuSmoothing::Exponential);
    short.update(&shortened).unwrap();
    close(short.value().unwrap(), 0.);
    let mut metric = BleuScore::with_max_n(2);
    metric.update(&input).unwrap();
    metric
        .update(&BleuInput {
            outputs: input.targets.clone(),
            targets: input.targets.clone(),
        })
        .unwrap();
    close(metric.value().unwrap(), 50.);
}

#[test]
fn rouge_macro_average_and_empty_sequences() {
    let device = Default::default();
    let mut metric = RougeLScore::<TestBackend>::new().with_pad_token(0);
    metric
        .update(&RougeLInput {
            outputs: Tensor::from_ints([[1, 2, 4, 0], [0, 0, 0, 0]], &device),
            targets: Tensor::from_ints([[1, 2, 3, 0], [0, 0, 0, 0]], &device),
        })
        .unwrap();
    close(metric.value().unwrap(), 250. / 3.);
    metric
        .update(&RougeLInput {
            outputs: Tensor::from_ints([[1, 0]], &device),
            targets: Tensor::from_ints([[0, 0]], &device),
        })
        .unwrap();
    close(metric.value().unwrap(), 500. / 9.);
}

#[test]
fn output_adaptors_cover_binary_multilabel_and_sequence_predictions() {
    let device = Default::default();
    let binary = ClassificationOutput::<AD>::new(
        Tensor::zeros([1], &device),
        Tensor::from_floats([[0.8], [0.2]], &device).require_grad(),
        Tensor::from_ints([1, 0], &device),
    )
    .into_inner();
    let data: ConfusionStatsInput<_> = binary.adapt();
    assert_eq!(data.targets.dims(), [2, 1]);
    let multilabel = MultiLabelClassificationOutput::<AD>::new(
        Tensor::zeros([1], &device),
        Tensor::from_floats([[0.8, 0.2]], &device),
        Tensor::from_ints([[1, 0]], &device),
    )
    .into_inner();
    let mut hamming = HammingScore::new();
    hamming.update(&multilabel.adapt()).unwrap();
    close(hamming.value().unwrap(), 100.);
    let output = SequenceOutput::<AD>::new(
        Tensor::zeros([1], &device),
        Tensor::from_floats([[[4., 1.], [1., 4.]]], &device).require_grad(),
        Some(Tensor::from_ints([[1, 1]], &device)),
        Tensor::from_ints([[0, 1]], &device),
    )
    .into_inner();
    let mut accuracy = AccuracyMetric::new();
    accuracy.update(&output.adapt()).unwrap();
    close(accuracy.value().unwrap(), 100.);
    let mut wer = WordErrorRate::new();
    wer.update(&output.adapt()).unwrap();
    close(wer.value().unwrap(), 50.);
    let default = SequenceOutput {
        predictions: None,
        ..output
    };
    wer.clear();
    wer.update(&default.adapt()).unwrap();
    close(wer.value().unwrap(), 0.);
    let regression = RegressionOutput::<AD>::new(
        Tensor::from_floats([2.], &device),
        Tensor::zeros([1, 1], &device),
        Tensor::zeros([1, 1], &device),
    )
    .into_inner();
    let mut loss = LossMetric::new();
    loss.update(&regression.adapt()).unwrap();
    close(loss.value().unwrap(), 2.);
}

#[test]
fn metadata_metrics_and_iteration_origin_reset() {
    let metadata = MetricMetadata {
        epoch: 1,
        iteration: 1,
        progress: Progress::new(2, 4),
        learning_rate: Some(0.1),
    };
    let mut lr = LearningRateMetric::new();
    lr.update_with_metadata(&(), &metadata).unwrap();
    lr.update_with_metadata(
        &(),
        &MetricMetadata {
            learning_rate: Some(0.2),
            ..metadata.clone()
        },
    )
    .unwrap();
    close(lr.value().unwrap(), 0.15);
    let mut speed = IterationSpeedMetric::new();
    speed.update_with_metadata(&(), &metadata).unwrap();
    close(speed.value().unwrap(), 0.);
    speed
        .update_with_metadata(
            &(),
            &MetricMetadata {
                iteration: 2,
                ..metadata
            },
        )
        .unwrap();
    let previous = speed.value();
    assert!(previous.unwrap().is_finite() && previous.unwrap() > 0.);
    speed.clear();
    assert_eq!(speed.value(), previous);
    lr.clear();
    assert!(lr.value().is_none());
}

#[test]
fn epoch_metrics_finalize_before_store_and_async_flush() {
    let processor = MetricsProcessor::<
        ClassificationOutput<TestBackend>,
        ClassificationOutput<TestBackend>,
    >::new()
    .metric_valid(AurocMetric::binary())
    .metric_valid(AucPrMetric::binary());
    let store = processor.store();
    let mut processor = AsyncProcessorTraining::new(processor);
    processor
        .process_valid(LearnerEvent::StartSplit {
            epoch: 1,
            total_items: 2,
        })
        .unwrap();
    for (iteration, (score, target)) in [(0.9, 1), (0.1, 0)].into_iter().enumerate() {
        let device = Default::default();
        let item = ClassificationOutput::new(
            Tensor::zeros([1], &device),
            Tensor::from_floats([[score]], &device),
            Tensor::from_ints([target], &device),
        );
        processor
            .process_valid(LearnerEvent::ProcessedItem(TrainingItem {
                item,
                iteration: iteration + 1,
                progress: Progress::new(iteration + 1, 2),
                learning_rate: None,
            }))
            .unwrap();
    }
    processor
        .process_valid(LearnerEvent::EndSplit {
            epoch: 1,
            completed: true,
        })
        .unwrap();
    processor.flush().unwrap();
    close(
        store.find_metric("AUROC [Macro]", 1, Split::Valid).unwrap(),
        100.,
    );
    close(
        store
            .find_metric("AUC-PR [Macro]", 1, Split::Valid)
            .unwrap(),
        100.,
    );
    processor.finish().unwrap();
}

#[test]
fn async_training_matches_synchronous_training() {
    let learner = Learner::new(Regressor::<AD>::new(), SgdConfig::new().init(), 0.01);
    let result = SupervisedTraining::new(train_loader(), valid_loader(), Default::default())
        .num_epochs(3)
        .with_async_metrics(metrics().metric_train(LearningRateMetric::new()))
        .fit(learner);
    assert!(result.error.is_none());
    close(scalar(&result.model), reference(3, 1, 0.).0);
    close(
        result
            .metrics
            .find_metric("Learning Rate", 3, Split::Train)
            .unwrap(),
        0.01,
    );
    assert!(
        result
            .metrics
            .find_metric("Loss", 3, Split::Valid)
            .unwrap()
            .is_finite()
    );
}

struct OrderedProcessor {
    started: mpsc::SyncSender<std::thread::ThreadId>,
    release: mpsc::Receiver<()>,
    seen: Arc<Mutex<Vec<&'static str>>>,
}
impl EventProcessorTraining<(), ()> for OrderedProcessor {
    fn process_train(&mut self, _: LearnerEvent<()>) -> Result<(), TrainingError> {
        self.started.send(std::thread::current().id()).unwrap();
        self.release.recv().unwrap();
        self.seen.lock().unwrap().push("train");
        Ok(())
    }
    fn process_valid(&mut self, _: LearnerEvent<()>) -> Result<(), TrainingError> {
        self.seen.lock().unwrap().push("valid");
        Ok(())
    }
    fn flush(&mut self) -> Result<(), TrainingError> {
        self.seen.lock().unwrap().push("flush");
        Ok(())
    }
}
#[test]
fn async_worker_is_nonblocking_ordered_and_flush_is_a_barrier() {
    let (started, ready) = mpsc::sync_channel(0);
    let (release, wait) = mpsc::channel();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut processor = AsyncProcessorTraining::new(OrderedProcessor {
        started,
        release: wait,
        seen: seen.clone(),
    });
    processor.process_train(LearnerEvent::EndEpoch(1)).unwrap();
    assert_ne!(ready.recv().unwrap(), std::thread::current().id());
    processor.process_valid(LearnerEvent::EndEpoch(1)).unwrap();
    assert!(seen.lock().unwrap().is_empty());
    release.send(()).unwrap();
    processor.flush().unwrap();
    assert_eq!(*seen.lock().unwrap(), vec!["train", "valid", "flush"]);
}

struct ErrorProcessor;
impl EventProcessorTraining<(), ()> for ErrorProcessor {
    fn process_train(&mut self, _: LearnerEvent<()>) -> Result<(), TrainingError> {
        Err(TrainingError("train".into()))
    }
    fn process_valid(&mut self, _: LearnerEvent<()>) -> Result<(), TrainingError> {
        Err(TrainingError("valid".into()))
    }
    fn flush(&mut self) -> Result<(), TrainingError> {
        Err(TrainingError("flush".into()))
    }
}
#[test]
fn async_errors_are_delivered_once_in_order_including_flush_errors() {
    let mut processor = AsyncProcessorTraining::new(ErrorProcessor);
    let mut errors = Vec::new();
    for result in [
        processor.process_train(LearnerEvent::EndEpoch(1)),
        processor.process_valid(LearnerEvent::EndEpoch(1)),
        processor.flush(),
    ] {
        if let Err(error) = result {
            errors.extend(error.0.lines().map(str::to_owned));
        }
    }
    assert_eq!(errors, vec!["train", "valid", "flush"]);
}

struct PanicProcessor;
impl EventProcessorTraining<(), ()> for PanicProcessor {
    fn process_train(&mut self, _: LearnerEvent<()>) -> Result<(), TrainingError> {
        panic!("metric worker panic")
    }
    fn process_valid(&mut self, _: LearnerEvent<()>) -> Result<(), TrainingError> {
        Ok(())
    }
}
#[test]
fn async_worker_panic_is_rethrown_without_hanging_flush() {
    let result = std::panic::catch_unwind(|| {
        let mut processor = AsyncProcessorTraining::new(PanicProcessor);
        let _ = processor.process_train(LearnerEvent::EndEpoch(1));
        let _ = processor.flush();
    });
    assert!(result.is_err());
}

#[test]
fn dropping_async_processor_drains_accepted_events() {
    let log = Log::default();
    let mut processor = AsyncProcessorTraining::with_capacity(
        MetricsProcessor::<(), ()>::new().with_logger(log.clone()),
        2,
    );
    for epoch in 1..=3 {
        processor
            .process_train(LearnerEvent::StartSplit {
                epoch,
                total_items: 0,
            })
            .unwrap();
    }
    drop(processor);
    assert_eq!(log.0.lock().unwrap().len(), 3);
}

struct FinalizationError;
impl Metric for FinalizationError {
    type Input = ();
    fn name(&self) -> &str {
        "fail"
    }
    fn update(&mut self, _: &()) -> Result<(), TrainingError> {
        Ok(())
    }
    fn compute(&mut self) -> Result<(), TrainingError> {
        Err(TrainingError("finalization failed".into()))
    }
    fn value(&self) -> Option<f64> {
        None
    }
    fn clear(&mut self) {}
}
#[test]
fn finalization_failure_does_not_publish_complete_epoch_metrics() {
    let mut processor = MetricsProcessor::<(), ()>::new().metric_valid(FinalizationError);
    let store = processor.store();
    processor
        .process_valid(LearnerEvent::StartSplit {
            epoch: 1,
            total_items: 0,
        })
        .unwrap();
    assert_eq!(
        processor
            .process_valid(LearnerEvent::EndSplit {
                epoch: 1,
                completed: true
            })
            .unwrap_err()
            .0,
        "finalization failed"
    );
    assert!(store.metrics(1, Split::Valid).is_empty());
}

#[test]
fn metric_configuration_rejections_remain_explicit() {
    assert!(std::panic::catch_unwind(|| TopKAccuracyMetric::<TestBackend>::new(0)).is_err());
    assert!(std::panic::catch_unwind(|| BleuScore::<TestBackend>::with_max_n(0)).is_err());
    assert!(
        std::panic::catch_unwind(|| FBetaScoreMetric::<TestBackend>::multiclass(
            1.,
            0,
            ClassReduction::Macro
        ))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            let mut metric = TopKAccuracyMetric::<TestBackend>::new(3);
            metric
                .update(&TopKAccuracyInput {
                    outputs: Tensor::zeros([1, 2], &Default::default()),
                    targets: Tensor::zeros([1], &Default::default()),
                })
                .unwrap();
        })
        .is_err()
    );
}
