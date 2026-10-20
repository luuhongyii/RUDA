use super::{Metric, NumericEntry, NumericMetricState, scalar};
use crate::TrainingError;
use ruda_model::tensor::{
    Bool, ElementConversion, IndexingUpdateOp, Int, Tensor, Transaction,
    activation::{log_softmax, sigmoid},
    backend::Backend,
};
use std::{marker::PhantomData, num::NonZeroUsize};

#[derive(Clone, Copy, Debug, Default)]
pub enum ClassReduction {
    Micro,
    #[default]
    Macro,
}
#[derive(Clone, Debug)]
pub enum DecisionRule {
    Threshold(f64),
    TopK(NonZeroUsize),
}
impl Default for DecisionRule {
    fn default() -> Self {
        Self::Threshold(0.5)
    }
}

pub struct TopKAccuracyInput<B: Backend> {
    pub outputs: Tensor<B, 2>,
    pub targets: Tensor<B, 1, Int>,
}
pub struct TopKAccuracyMetric<B: Backend> {
    name: String,
    k: usize,
    pad_token: Option<usize>,
    state: NumericMetricState,
    marker: PhantomData<B>,
}
impl<B: Backend> Default for TopKAccuracyMetric<B> {
    fn default() -> Self {
        Self::new(1)
    }
}
impl<B: Backend> TopKAccuracyMetric<B> {
    pub fn new(k: usize) -> Self {
        assert!(
            k > 0,
            "TopKAccuracyMetric requires k to be greater than zero"
        );
        Self {
            name: format!("Top-K Accuracy @ TopK({k})"),
            k,
            pad_token: None,
            state: Default::default(),
            marker: PhantomData,
        }
    }
    pub fn with_pad_token(mut self, index: usize) -> Self {
        self.pad_token = Some(index);
        self
    }
}
impl<B: Backend> Metric for TopKAccuracyMetric<B> {
    numeric_state_metric!(Some("%"), true, 2);
    type Input = TopKAccuracyInput<B>;
    fn name(&self) -> &str {
        &self.name
    }
    fn update(&mut self, input: &Self::Input) -> Result<(), TrainingError> {
        let [n, classes] = input.outputs.dims();
        assert!(
            self.k <= classes,
            "TopKAccuracyMetric k exceeds number of classes"
        );
        assert_eq!(input.targets.dims(), [n]);
        if n == 0 {
            return Ok(());
        }
        let indices = input
            .outputs
            .clone()
            .argsort_descending(1)
            .narrow(1, 0, self.k);
        let mut targets = input.targets.clone();
        let mut valid = n;
        if let Some(pad) = self.pad_token {
            let mask = targets.clone().equal_elem(pad as i64);
            valid -= mask
                .clone()
                .int()
                .sum()
                .try_into_scalar()
                .map_err(|error| TrainingError(error.to_string()))?
                .elem::<i64>() as usize;
            targets = targets.mask_fill(mask, -1);
        }
        if valid > 0 {
            let matches = targets
                .reshape([n, 1])
                .repeat_dim(1, self.k)
                .equal(indices)
                .float()
                .sum();
            self.state
                .update(100. * scalar(matches)? / valid as f64, valid);
        }
        Ok(())
    }
    fn value(&self) -> Option<f64> {
        self.state.value()
    }
    fn clear(&mut self) {
        self.state.clear();
    }
}

pub struct PerplexityInput<B: Backend> {
    pub outputs: Tensor<B, 2>,
    pub targets: Tensor<B, 1, Int>,
}
pub struct PerplexityMetric<B: Backend> {
    nll: f64,
    tokens: usize,
    current: f64,
    current_tokens: usize,
    pad_token: Option<usize>,
    marker: PhantomData<B>,
}
impl<B: Backend> Default for PerplexityMetric<B> {
    fn default() -> Self {
        Self::new()
    }
}
impl<B: Backend> PerplexityMetric<B> {
    pub fn new() -> Self {
        Self {
            nll: 0.,
            tokens: 0,
            current: f64::NAN,
            current_tokens: 0,
            pad_token: None,
            marker: PhantomData,
        }
    }
    pub fn with_pad_token(mut self, index: usize) -> Self {
        self.pad_token = Some(index);
        self
    }
}
impl<B: Backend> Metric for PerplexityMetric<B> {
    numeric_attributes!(None, false, 2);
    fn current_value(&self) -> Option<NumericEntry> {
        Some(NumericEntry::Aggregated { aggregated_value: self.current, count: self.current_tokens })
    }
    fn running_value(&self) -> Option<NumericEntry> {
        self.value().map(|aggregated_value| NumericEntry::Aggregated { aggregated_value, count: self.tokens })
    }
    type Input = PerplexityInput<B>;
    fn name(&self) -> &str {
        "Perplexity"
    }
    fn update(&mut self, input: &Self::Input) -> Result<(), TrainingError> {
        let [n, _] = input.outputs.dims();
        assert_eq!(input.targets.dims(), [n]);
        if n == 0 {
            return Ok(());
        }
        let mut probabilities = log_softmax(input.outputs.clone(), 1)
            .gather(1, input.targets.clone().unsqueeze_dim(1))
            .squeeze_dim::<1>(1);
        let mut valid = n;
        if let Some(pad) = self.pad_token {
            let mask = input.targets.clone().equal_elem(pad as i64);
            valid -= mask
                .clone()
                .int()
                .sum()
                .try_into_scalar()
                .map_err(|error| TrainingError(error.to_string()))?
                .elem::<i64>() as usize;
            probabilities = probabilities.mask_fill(mask, 0.);
        }
        let batch_nll = -scalar(probabilities.sum())?;
        self.nll += batch_nll;
        self.tokens += valid;
        self.current = if valid > 0 { (batch_nll / valid as f64).exp() } else { f64::INFINITY };
        self.current_tokens = valid;
        Ok(())
    }
    fn value(&self) -> Option<f64> {
        Some(if self.tokens > 0 {
            (self.nll / self.tokens as f64).exp()
        } else {
            f64::INFINITY
        })
    }
    fn clear(&mut self) {
        self.nll = 0.;
        self.tokens = 0;
        self.current = f64::NAN;
        self.current_tokens = 0;
    }
}

pub struct HammingScoreInput<B: Backend> {
    pub outputs: Tensor<B, 2>,
    pub targets: Tensor<B, 2, Int>,
}
pub struct HammingScore<B: Backend> {
    name: String,
    threshold: f32,
    sigmoid: bool,
    state: NumericMetricState,
    marker: PhantomData<B>,
}
impl<B: Backend> Default for HammingScore<B> {
    fn default() -> Self {
        Self::new()
    }
}
impl<B: Backend> HammingScore<B> {
    pub fn new() -> Self {
        Self {
            name: "Hamming Score @ Threshold(0.5)".into(),
            threshold: 0.5,
            sigmoid: false,
            state: Default::default(),
            marker: PhantomData,
        }
    }
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self.name = format!("Hamming Score @ Threshold({threshold})");
        self
    }
    pub fn with_sigmoid(mut self, sigmoid: bool) -> Self {
        self.sigmoid = sigmoid;
        self
    }
}
impl<B: Backend> Metric for HammingScore<B> {
    numeric_state_metric!(Some("%"), true, 2);
    type Input = HammingScoreInput<B>;
    fn name(&self) -> &str {
        &self.name
    }
    fn update(&mut self, input: &Self::Input) -> Result<(), TrainingError> {
        let [n, _] = input.outputs.dims();
        assert_eq!(input.outputs.dims(), input.targets.dims());
        if n > 0 {
            let output = if self.sigmoid {
                sigmoid(input.outputs.clone())
            } else {
                input.outputs.clone()
            };
            self.state.update(
                100. * scalar(
                    output
                        .greater_elem(self.threshold)
                        .equal(input.targets.clone().bool())
                        .float()
                        .mean(),
                )?,
                n,
            );
        }
        Ok(())
    }
    fn value(&self) -> Option<f64> {
        self.state.value()
    }
    fn clear(&mut self) {
        self.state.clear();
    }
}

#[derive(Clone, Debug)]
pub struct ConfusionStatsInput<B: Backend> {
    pub predictions: Tensor<B, 2>,
    pub targets: Tensor<B, 2, Bool>,
}
#[derive(Clone, Copy)]
enum Score {
    Precision,
    Recall,
    FBeta(f64),
}
struct ConfusionMetric<B: Backend> {
    name: String,
    decision: DecisionRule,
    reduction: ClassReduction,
    score: Score,
    counts: Option<[Tensor<B, 1>; 3]>,
    value: Option<f64>,
    current: Option<f64>,
    current_count: usize,
    count: usize,
}
impl<B: Backend> ConfusionMetric<B> {
    fn new(score: Score, decision: DecisionRule, reduction: ClassReduction) -> Self {
        let title = match score {
            Score::Precision => "Precision".into(),
            Score::Recall => "Recall".into(),
            Score::FBeta(beta) => format!("FBetaScore ({beta})"),
        };
        Self {
            name: format!("{title} @ {decision:?} [{reduction:?}]"),
            decision,
            reduction,
            score,
            counts: None,
            value: None,
            current: None,
            current_count: 0,
            count: 0,
        }
    }
    fn update(&mut self, input: &ConfusionStatsInput<B>) -> Result<(), TrainingError> {
        let [n, classes] = input.predictions.dims();
        assert_eq!(input.targets.dims(), [n, classes]);
        if n == 0 {
            return Ok(());
        }
        let predicted = match self.decision {
            DecisionRule::Threshold(threshold) => input.predictions.clone().greater_elem(threshold),
            DecisionRule::TopK(k) => {
                assert!(k.get() <= classes, "top_k exceeds number of classes");
                let indices = input
                    .predictions
                    .clone()
                    .argsort_descending(1)
                    .narrow(1, 0, k.get());
                let values = indices.ones_like().float();
                input
                    .predictions
                    .zeros_like()
                    .scatter(1, indices, values, IndexingUpdateOp::Add)
                    .bool()
            }
        };
        let encoded = predicted.int() + input.targets.clone().int() * 2;
        let aggregate = |value| {
            let mask = encoded.clone().equal_elem(value).float();
            match self.reduction {
                ClassReduction::Micro => mask.sum(),
                ClassReduction::Macro => mask.sum_dim(0).squeeze_dim::<1>(0),
            }
        };
        let mut counts = [aggregate(3), aggregate(1), aggregate(2)];
        let current = self.score_value(counts.clone())?;
        if let Some(previous) = self.counts.take() {
            counts = std::array::from_fn(|i| previous[i].clone() + counts[i].clone());
        }
        let value = self.score_value(counts.clone())?;
        self.counts = Some(counts);
        self.value = value;
        self.current = current;
        self.current_count = n;
        self.count += n;
        Ok(())
    }
    fn score_value(&self, [tp, fp, fn_]: [Tensor<B, 1>; 3]) -> Result<Option<f64>, TrainingError> {
        let scores = match self.score {
            Score::Precision | Score::Recall => {
                let denominator = tp.clone()
                    + if matches!(self.score, Score::Precision) {
                        fp
                    } else {
                        fn_
                    };
                let zero = denominator.clone().equal_elem(0.);
                tp / denominator.mask_fill(zero, 1.) * 100.
            }
            Score::FBeta(beta) => {
                let beta2 = beta * beta;
                let numerator = tp * (1. + beta2);
                numerator.clone() / (numerator + fn_ * beta2 + fp) * 100.
            }
        };
        let value = match self.reduction {
            ClassReduction::Micro => scalar(scores)?,
            ClassReduction::Macro => {
                let keep = scores
                    .clone()
                    .is_nan()
                    .bool_not()
                    .argwhere()
                    .squeeze_dim::<1>(1);
                if keep.dims()[0] == 0 {
                    f64::NAN
                } else {
                    scalar(scores.select(0, keep).mean())?
                }
            }
        };
        Ok((!value.is_nan()).then_some(value))
    }
}
macro_rules! confusion_metric {
    ($name:ident) => {
        pub struct $name<B: Backend>(ConfusionMetric<B>);
        impl<B: Backend> Metric for $name<B> {
            numeric_attributes!(Some("%"), true, 2);
            fn current_value(&self) -> Option<NumericEntry> {
                self.0.current.map(|aggregated_value| NumericEntry::Aggregated {
                    aggregated_value, count: self.0.current_count,
                })
            }
            fn running_value(&self) -> Option<NumericEntry> {
                self.0.value.map(|aggregated_value| NumericEntry::Aggregated {
                    aggregated_value, count: self.0.count,
                })
            }
            type Input = ConfusionStatsInput<B>;
            fn name(&self) -> &str {
                &self.0.name
            }
            fn update(&mut self, input: &Self::Input) -> Result<(), TrainingError> {
                self.0.update(input)
            }
            fn value(&self) -> Option<f64> {
                self.0.value
            }
            fn clear(&mut self) {
                self.0.counts = None;
                self.0.value = None;
                self.0.current = None;
                self.0.current_count = 0;
                self.0.count = 0;
            }
        }
    };
}
confusion_metric!(PrecisionMetric);
confusion_metric!(RecallMetric);
confusion_metric!(FBetaScoreMetric);
macro_rules! confusion_constructors {
    ($name:ident, $score:ident) => {
        impl<B: Backend> Default for $name<B> {
            fn default() -> Self {
                Self(ConfusionMetric::new(
                    Score::$score,
                    Default::default(),
                    Default::default(),
                ))
            }
        }
        impl<B: Backend> $name<B> {
            pub fn binary(threshold: f64) -> Self {
                Self(ConfusionMetric::new(
                    Score::$score,
                    DecisionRule::Threshold(threshold),
                    ClassReduction::Macro,
                ))
            }
            pub fn multiclass(top_k: usize, reduction: ClassReduction) -> Self {
                Self(ConfusionMetric::new(
                    Score::$score,
                    DecisionRule::TopK(NonZeroUsize::new(top_k).expect("top_k must be non-zero")),
                    reduction,
                ))
            }
        }
    };
}
confusion_constructors!(PrecisionMetric, Precision);
confusion_constructors!(RecallMetric, Recall);
impl<B: Backend> PrecisionMetric<B> {
    pub fn multilabel(threshold: f64, reduction: ClassReduction) -> Self {
        let mut metric = Self::default();
        metric.0.decision = DecisionRule::Threshold(threshold);
        metric.0.reduction = reduction;
        metric
    }
}
impl<B: Backend> RecallMetric<B> {
    pub fn multilabel(threshold: f64, reduction: ClassReduction) -> Self {
        Self(ConfusionMetric::new(
            Score::Recall,
            DecisionRule::Threshold(threshold),
            reduction,
        ))
    }
}
impl<B: Backend> Default for FBetaScoreMetric<B> {
    fn default() -> Self {
        Self::binary(0., 0.5)
    }
}
impl<B: Backend> FBetaScoreMetric<B> {
    pub fn binary(beta: f64, threshold: f64) -> Self {
        Self::multilabel(beta, threshold, ClassReduction::Macro)
    }
    pub fn multiclass(beta: f64, top_k: usize, reduction: ClassReduction) -> Self {
        Self(ConfusionMetric::new(
            Score::FBeta(beta),
            DecisionRule::TopK(NonZeroUsize::new(top_k).expect("top_k must be non-zero")),
            reduction,
        ))
    }
    pub fn multilabel(beta: f64, threshold: f64, reduction: ClassReduction) -> Self {
        Self(ConfusionMetric::new(
            Score::FBeta(beta),
            DecisionRule::Threshold(threshold),
            reduction,
        ))
    }
}

struct EpochPredictions<B: Backend> {
    scores: Vec<Tensor<B, 2>>,
    targets: Vec<Tensor<B, 2, Bool>>,
    value: Option<f64>,
}
impl<B: Backend> Default for EpochPredictions<B> {
    fn default() -> Self {
        Self {
            scores: Vec::new(),
            targets: Vec::new(),
            value: None,
        }
    }
}
impl<B: Backend> EpochPredictions<B> {
    fn update(&mut self, input: &ConfusionStatsInput<B>) {
        assert_eq!(input.predictions.dims(), input.targets.dims());
        if input.predictions.dims()[0] > 0 {
            self.scores.push(input.predictions.clone());
            self.targets.push(input.targets.clone());
            self.value = None;
        }
    }
    fn tensors(&self) -> (Tensor<B, 2>, Tensor<B, 2, Bool>) {
        (
            Tensor::cat(self.scores.clone(), 0),
            Tensor::cat(self.targets.clone(), 0),
        )
    }
}
macro_rules! auc_metric {
    ($name:ident, $title:literal, $compute:ident, $attributes:expr) => {
        pub struct $name<B: Backend> {
            name: String,
            reduction: ClassReduction,
            state: EpochPredictions<B>,
        }
        impl<B: Backend> Default for $name<B> {
            fn default() -> Self {
                Self::binary()
            }
        }
        impl<B: Backend> $name<B> {
            pub fn binary() -> Self {
                Self::multiclass(ClassReduction::Macro)
            }
            pub fn multiclass(reduction: ClassReduction) -> Self {
                Self {
                    name: format!(concat!($title, " [{:?}]"), reduction),
                    reduction,
                    state: Default::default(),
                }
            }
            pub fn multilabel(reduction: ClassReduction) -> Self {
                Self::multiclass(reduction)
            }
        }
        impl<B: Backend> Metric for $name<B> {
            fn attributes(&self) -> super::MetricAttributes { $attributes }
            fn format_options(&self) -> super::FormatOptions {
                super::FormatOptions::new(std::sync::Arc::new(self.name().into())).unit("%").precision(2)
            }
            fn current_value(&self) -> Option<NumericEntry> {
                self.state.value.map(NumericEntry::Value)
            }
            fn serialized_update(&self) -> super::SerializedEntry {
                super::numeric::serialize_values(None, self.running_value(), None, self.format_options())
            }
            fn serialized_final(&self) -> super::SerializedEntry {
                super::numeric::serialize_values(None, self.running_value(), Some(self.final_value()), self.format_options())
            }
            type Input = ConfusionStatsInput<B>;
            fn name(&self) -> &str {
                &self.name
            }
            fn update(&mut self, input: &Self::Input) -> Result<(), TrainingError> {
                self.state.update(input);
                Ok(())
            }
            fn compute(&mut self) -> Result<(), TrainingError> {
                if !self.state.scores.is_empty() {
                    let (scores, targets) = self.state.tensors();
                    self.state.value = Some(100. * $compute(scores, targets, self.reduction)?);
                }
                Ok(())
            }
            fn value(&self) -> Option<f64> {
                self.state.value
            }
            fn clear(&mut self) {
                self.state = Default::default();
            }
        }
    };
}
auc_metric!(AurocMetric, "AUROC", auroc, super::MetricAttributes::None);
auc_metric!(AucPrMetric, "AUC-PR", average_precision,
    super::NumericAttributes { unit: Some("%".into()), higher_is_better: true }.into());

fn sorted_auc(samples: &mut Vec<(f64, bool)>) -> Option<f64> {
    let positives = samples.iter().filter(|(_, target)| *target).count() as u128;
    let negatives = samples.len() as u128 - positives;
    if positives == 0 || negatives == 0 {
        return None;
    }
    samples.retain(|(score, _)| !score.is_nan());
    samples.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let (mut below, mut twice_wins, mut i) = (0u128, 0u128, 0);
    while i < samples.len() {
        let score = samples[i].0;
        let (mut pos, mut neg) = (0u128, 0u128);
        while i < samples.len() && samples[i].0 == score {
            if samples[i].1 {
                pos += 1;
            } else {
                neg += 1;
            }
            i += 1;
        }
        twice_wins += pos * (2 * below + neg);
        below += neg;
    }
    Some(twice_wins as f64 / (2 * positives * negatives) as f64)
}
fn auroc<B: Backend>(
    scores: Tensor<B, 2>,
    targets: Tensor<B, 2, Bool>,
    reduction: ClassReduction,
) -> Result<f64, TrainingError> {
    let [n, c] = scores.dims();
    let data = Transaction::default()
        .register(scores)
        .register(targets)
        .try_execute()
        .map_err(|error| TrainingError(error.to_string()))?;
    let scores: Vec<f64> = data[0].iter().collect();
    let targets: Vec<bool> = data[1].iter().collect();
    let result = match reduction {
        ClassReduction::Micro => sorted_auc(&mut scores.into_iter().zip(targets).collect()),
        ClassReduction::Macro => {
            let values: Vec<_> = (0..c)
                .filter_map(|class| {
                    sorted_auc(
                        &mut (0..n)
                            .map(|row| (scores[row * c + class], targets[row * c + class]))
                            .collect(),
                    )
                })
                .collect();
            (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
        }
    };
    Ok(result.unwrap_or(0.5))
}
fn average_precision<B: Backend>(
    scores: Tensor<B, 2>,
    targets: Tensor<B, 2, Bool>,
    reduction: ClassReduction,
) -> Result<f64, TrainingError> {
    let [n, c] = scores.dims();
    let (scores, targets) = match reduction {
        ClassReduction::Macro => (scores, targets.float()),
        ClassReduction::Micro => (
            scores.reshape([n * c, 1]),
            targets.float().reshape([n * c, 1]),
        ),
    };
    let [n, c] = scores.dims();
    let device = scores.device();
    let (sorted, order) = scores.sort_descending_with_indices(0);
    let labels = targets.clone().gather(0, order);
    let ranks = Tensor::<B, 1, Int>::arange(1..n as i64 + 1, &device).reshape([n, 1]);
    let mut precision = labels.clone().cumsum(0) / ranks.clone().float();
    if n > 1 {
        let same = sorted
            .clone()
            .narrow(0, 0, n - 1)
            .equal(sorted.narrow(0, 1, n - 1));
        let non_end = Tensor::cat(vec![same, Tensor::zeros([1, c], &device)], 0);
        let end = (ranks - 1)
            .expand([n, c])
            .mask_fill(non_end, n as i64 - 1)
            .flip([0])
            .cummin(0)
            .flip([0]);
        precision = precision.gather(0, end);
    }
    let ap = (precision * (labels / targets.sum_dim(0)))
        .sum_dim(0)
        .squeeze_dim::<1>(0);
    let keep = ap
        .clone()
        .is_nan()
        .bool_not()
        .argwhere()
        .squeeze_dim::<1>(1);
    if keep.dims()[0] == 0 {
        Ok(0.5)
    } else {
        scalar(ap.select(0, keep).mean())
    }
}
