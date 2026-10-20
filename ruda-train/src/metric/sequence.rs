use super::{Metric, NumericMetricState};
use crate::TrainingError;
use ruda_model::tensor::{Int, Tensor, Transaction, backend::Backend};
use std::{collections::HashMap, marker::PhantomData};

macro_rules! sequence_input {
    ($name:ident) => {
        pub struct $name<B: Backend> {
            pub outputs: Tensor<B, 2, Int>,
            pub targets: Tensor<B, 2, Int>,
        }
    };
}
sequence_input!(CerInput);
sequence_input!(WerInput);
sequence_input!(BleuInput);
sequence_input!(RougeLInput);

fn read<B: Backend>(
    outputs: &Tensor<B, 2, Int>,
    targets: &Tensor<B, 2, Int>,
) -> Result<(Vec<i32>, Vec<i32>), TrainingError> {
    assert_eq!(outputs.dims(), targets.dims());
    let data = Transaction::default()
        .register(outputs.clone())
        .register(targets.clone())
        .try_execute()
        .map_err(|error| TrainingError(error.to_string()))?;
    Ok((data[0].iter().collect(), data[1].iter().collect()))
}
fn prefix(tokens: &[i32], pad: Option<usize>) -> &[i32] {
    let len = pad
        .and_then(|pad| tokens.iter().position(|&token| token == pad as i32))
        .unwrap_or(tokens.len());
    &tokens[..len]
}
fn edit_distance(reference: &[i32], prediction: &[i32]) -> usize {
    let mut prev: Vec<_> = (0..=prediction.len()).collect();
    let mut curr = vec![0; prediction.len() + 1];
    for (i, &r) in reference.iter().enumerate() {
        curr[0] = i + 1;
        for (j, &p) in prediction.iter().enumerate() {
            curr[j + 1] = if r == p {
                prev[j]
            } else {
                1 + prev[j].min(prev[j + 1]).min(curr[j])
            };
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[prediction.len()]
}
macro_rules! error_rate {
    ($name:ident, $input:ident, $title:literal, $count_padding:expr) => {
        pub struct $name<B: Backend> {
            state: NumericMetricState,
            pad_token: Option<usize>,
            marker: PhantomData<B>,
        }
        impl<B: Backend> Default for $name<B> {
            fn default() -> Self {
                Self::new()
            }
        }
        impl<B: Backend> $name<B> {
            pub fn new() -> Self {
                Self {
                    state: Default::default(),
                    pad_token: None,
                    marker: PhantomData,
                }
            }
            pub fn with_pad_token(mut self, index: usize) -> Self {
                self.pad_token = Some(index);
                self
            }
        }
        impl<B: Backend> Metric for $name<B> {
            numeric_state_metric!(Some("%"), false, 2);
            type Input = $input<B>;
            fn name(&self) -> &str {
                $title
            }
            fn update(&mut self, input: &Self::Input) -> Result<(), TrainingError> {
                let [n, length] = input.targets.dims();
                let (outputs, targets) = read(&input.outputs, &input.targets)?;
                let (mut edits, mut total) = (0usize, 0usize);
                for row in 0..n {
                    let trim = |tokens: &[i32]| -> usize {
                        if $count_padding {
                            self.pad_token
                                .map(|pad| {
                                    tokens.iter().filter(|&&token| token != pad as i32).count()
                                })
                                .unwrap_or(tokens.len())
                        } else {
                            prefix(tokens, self.pad_token).len()
                        }
                    };
                    let output = &outputs[row * length..(row + 1) * length];
                    let target = &targets[row * length..(row + 1) * length];
                    let target = &target[..trim(target)];
                    edits += edit_distance(target, &output[..trim(output)]);
                    total += target.len();
                }
                if total > 0 {
                    self.state.update(100. * edits as f64 / total as f64, total);
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
    };
}
error_rate!(CharErrorRate, CerInput, "CER", true);
error_rate!(WordErrorRate, WerInput, "WER", false);

#[derive(Clone, Debug, Default)]
pub enum BleuSmoothing {
    #[default]
    None,
    AddEpsilon(f64),
    Exponential,
}
pub struct BleuScore<B: Backend> {
    name: String,
    max_n: usize,
    pad_token: Option<usize>,
    smoothing: BleuSmoothing,
    state: NumericMetricState,
    marker: PhantomData<B>,
}
impl<B: Backend> Default for BleuScore<B> {
    fn default() -> Self {
        Self::with_max_n(4)
    }
}
impl<B: Backend> BleuScore<B> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_max_n(max_n: usize) -> Self {
        assert!(max_n >= 1, "max_n must be at least 1");
        Self {
            name: format!("BLEU-{max_n}"),
            max_n,
            pad_token: None,
            smoothing: BleuSmoothing::None,
            state: Default::default(),
            marker: PhantomData,
        }
    }
    pub fn with_pad_token(mut self, index: usize) -> Self {
        self.pad_token = Some(index);
        self
    }
    pub fn with_smoothing(mut self, smoothing: BleuSmoothing) -> Self {
        self.smoothing = smoothing;
        self
    }
}
fn ngrams(tokens: &[i32], n: usize) -> HashMap<Vec<i32>, usize> {
    let mut counts = HashMap::new();
    for window in tokens.windows(n) {
        *counts.entry(window.to_vec()).or_default() += 1;
    }
    counts
}
fn bleu(
    clipped: &[usize],
    totals: &[usize],
    candidate: usize,
    reference: usize,
    smoothing: &BleuSmoothing,
) -> f64 {
    if candidate == 0 {
        return 0.;
    }
    let bp = if candidate < reference {
        (1. - reference as f64 / candidate as f64).exp()
    } else {
        1.
    };
    let (mut logs, mut multiplier) = (0., 1.);
    for (&matched, &total) in clipped.iter().zip(totals) {
        if total == 0 {
            return 0.;
        }
        let precision = if matched == 0 {
            match smoothing {
                BleuSmoothing::None => return 0.,
                BleuSmoothing::AddEpsilon(epsilon) => epsilon / total as f64,
                BleuSmoothing::Exponential => {
                    multiplier *= 2.;
                    1. / (multiplier * total as f64)
                }
            }
        } else {
            matched as f64 / total as f64
        };
        logs += precision.ln();
    }
    100. * bp * (logs / totals.len() as f64).exp()
}
impl<B: Backend> Metric for BleuScore<B> {
    numeric_state_metric!(Some("%"), true, 2);
    type Input = BleuInput<B>;
    fn name(&self) -> &str {
        &self.name
    }
    fn update(&mut self, input: &Self::Input) -> Result<(), TrainingError> {
        let [n, length] = input.targets.dims();
        let (outputs, targets) = read(&input.outputs, &input.targets)?;
        let (mut clipped, mut totals) = (vec![0; self.max_n], vec![0; self.max_n]);
        let (mut candidate, mut reference) = (0, 0);
        for row in 0..n {
            let output = prefix(&outputs[row * length..(row + 1) * length], self.pad_token);
            let target = prefix(&targets[row * length..(row + 1) * length], self.pad_token);
            candidate += output.len();
            reference += target.len();
            for order in 1..=self.max_n {
                let expected = ngrams(target, order);
                for (ngram, count) in ngrams(output, order) {
                    clipped[order - 1] += count.min(expected.get(&ngram).copied().unwrap_or(0));
                    totals[order - 1] += count;
                }
            }
        }
        self.state.update(
            bleu(&clipped, &totals, candidate, reference, &self.smoothing),
            n,
        );
        Ok(())
    }
    fn value(&self) -> Option<f64> {
        self.state.value()
    }
    fn clear(&mut self) {
        self.state.clear();
    }
}

fn lcs(a: &[i32], b: &[i32]) -> usize {
    let (shorter, longer) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let mut prev = vec![0usize; shorter.len() + 1];
    let mut curr = prev.clone();
    for &x in longer {
        for (j, &y) in shorter.iter().enumerate() {
            curr[j + 1] = if x == y {
                prev[j] + 1
            } else {
                curr[j].max(prev[j + 1])
            };
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[shorter.len()]
}
pub struct RougeLScore<B: Backend> {
    state: NumericMetricState,
    pad_token: Option<usize>,
    marker: PhantomData<B>,
}
impl<B: Backend> Default for RougeLScore<B> {
    fn default() -> Self {
        Self::new()
    }
}
impl<B: Backend> RougeLScore<B> {
    pub fn new() -> Self {
        Self {
            state: Default::default(),
            pad_token: None,
            marker: PhantomData,
        }
    }
    pub fn with_pad_token(mut self, index: usize) -> Self {
        self.pad_token = Some(index);
        self
    }
}
impl<B: Backend> Metric for RougeLScore<B> {
    numeric_state_metric!(Some("%"), true, 2);
    type Input = RougeLInput<B>;
    fn name(&self) -> &str {
        "ROUGE-L"
    }
    fn update(&mut self, input: &Self::Input) -> Result<(), TrainingError> {
        let [n, length] = input.targets.dims();
        let (outputs, targets) = read(&input.outputs, &input.targets)?;
        let mut sum = 0.;
        for row in 0..n {
            let output = prefix(&outputs[row * length..(row + 1) * length], self.pad_token);
            let target = prefix(&targets[row * length..(row + 1) * length], self.pad_token);
            if output.is_empty() && target.is_empty() {
                sum += 100.;
            } else if !output.is_empty() && !target.is_empty() {
                let matched = lcs(target, output) as f64;
                let precision = matched / output.len() as f64;
                let recall = matched / target.len() as f64;
                if precision + recall > 0. {
                    sum += 200. * precision * recall / (precision + recall);
                }
            }
        }
        if n > 0 {
            self.state.update(sum / n as f64, n);
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
