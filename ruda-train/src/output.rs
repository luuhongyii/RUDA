use crate::{ClassificationOutput, RegressionOutput, metric::*};
use ruda_model::tensor::{
    Int, Tensor,
    backend::{AutodiffBackend, Backend},
};

impl<B: Backend> RegressionOutput<B> {
    pub fn new(loss: Tensor<B, 1>, output: Tensor<B, 2>, targets: Tensor<B, 2>) -> Self {
        Self {
            loss,
            output,
            targets,
        }
    }
}
impl<B: AutodiffBackend> RegressionOutput<B> {
    pub fn into_inner(self) -> RegressionOutput<B::InnerBackend> {
        RegressionOutput::new(self.loss.inner(), self.output.inner(), self.targets.inner())
    }
}
impl<B: Backend> ClassificationOutput<B> {
    pub fn new(loss: Tensor<B, 1>, output: Tensor<B, 2>, targets: Tensor<B, 1, Int>) -> Self {
        Self {
            loss,
            output,
            targets,
        }
    }
}
impl<B: AutodiffBackend> ClassificationOutput<B> {
    pub fn into_inner(self) -> ClassificationOutput<B::InnerBackend> {
        ClassificationOutput::new(self.loss.inner(), self.output.inner(), self.targets.inner())
    }
}
impl<B: Backend> Adaptor<TopKAccuracyInput<B>> for ClassificationOutput<B> {
    fn adapt(&self) -> TopKAccuracyInput<B> {
        TopKAccuracyInput {
            outputs: self.output.clone(),
            targets: self.targets.clone(),
        }
    }
}
impl<B: Backend> Adaptor<PerplexityInput<B>> for ClassificationOutput<B> {
    fn adapt(&self) -> PerplexityInput<B> {
        PerplexityInput {
            outputs: self.output.clone(),
            targets: self.targets.clone(),
        }
    }
}
impl<B: Backend> Adaptor<ConfusionStatsInput<B>> for ClassificationOutput<B> {
    fn adapt(&self) -> ConfusionStatsInput<B> {
        let [_, classes] = self.output.dims();
        let targets = if classes > 1 {
            self.targets.clone().one_hot(classes).bool()
        } else {
            self.targets.clone().unsqueeze_dim(1).bool()
        };
        ConfusionStatsInput {
            predictions: self.output.clone(),
            targets,
        }
    }
}

pub struct MultiLabelClassificationOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub output: Tensor<B, 2>,
    pub targets: Tensor<B, 2, Int>,
}
impl<B: Backend> MultiLabelClassificationOutput<B> {
    pub fn new(loss: Tensor<B, 1>, output: Tensor<B, 2>, targets: Tensor<B, 2, Int>) -> Self {
        Self {
            loss,
            output,
            targets,
        }
    }
}
impl<B: AutodiffBackend> MultiLabelClassificationOutput<B> {
    pub fn into_inner(self) -> MultiLabelClassificationOutput<B::InnerBackend> {
        MultiLabelClassificationOutput::new(
            self.loss.inner(),
            self.output.inner(),
            self.targets.inner(),
        )
    }
}
impl<B: Backend> Adaptor<LossInput<B>> for MultiLabelClassificationOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput {
            tensor: self.loss.clone(),
        }
    }
}
impl<B: Backend> Adaptor<HammingScoreInput<B>> for MultiLabelClassificationOutput<B> {
    fn adapt(&self) -> HammingScoreInput<B> {
        HammingScoreInput {
            outputs: self.output.clone(),
            targets: self.targets.clone(),
        }
    }
}
impl<B: Backend> Adaptor<ConfusionStatsInput<B>> for MultiLabelClassificationOutput<B> {
    fn adapt(&self) -> ConfusionStatsInput<B> {
        ConfusionStatsInput {
            predictions: self.output.clone(),
            targets: self.targets.clone().bool(),
        }
    }
}

pub struct SequenceOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub logits: Tensor<B, 3>,
    pub predictions: Option<Tensor<B, 2, Int>>,
    pub targets: Tensor<B, 2, Int>,
}
impl<B: Backend> SequenceOutput<B> {
    pub fn new(
        loss: Tensor<B, 1>,
        logits: Tensor<B, 3>,
        predictions: Option<Tensor<B, 2, Int>>,
        targets: Tensor<B, 2, Int>,
    ) -> Self {
        Self {
            loss,
            logits,
            predictions,
            targets,
        }
    }
    fn predicted_tokens(&self) -> Tensor<B, 2, Int> {
        self.predictions
            .clone()
            .unwrap_or_else(|| self.logits.clone().argmax(2).squeeze_dim::<2>(2))
    }
    fn flat_logits(&self) -> Tensor<B, 2> {
        let [n, length, classes] = self.logits.dims();
        self.logits.clone().reshape([n * length, classes])
    }
    fn flat_targets(&self) -> Tensor<B, 1, Int> {
        let [n, length] = self.targets.dims();
        self.targets.clone().reshape([n * length])
    }
}
impl<B: AutodiffBackend> SequenceOutput<B> {
    pub fn into_inner(self) -> SequenceOutput<B::InnerBackend> {
        SequenceOutput::new(
            self.loss.inner(),
            self.logits.inner(),
            self.predictions.map(Tensor::inner),
            self.targets.inner(),
        )
    }
}
impl<B: Backend> Adaptor<LossInput<B>> for SequenceOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput {
            tensor: self.loss.clone(),
        }
    }
}
impl<B: Backend> Adaptor<AccuracyInput<B>> for SequenceOutput<B> {
    fn adapt(&self) -> AccuracyInput<B> {
        AccuracyInput {
            predictions: self.flat_logits(),
            targets: self.flat_targets(),
        }
    }
}
impl<B: Backend> Adaptor<TopKAccuracyInput<B>> for SequenceOutput<B> {
    fn adapt(&self) -> TopKAccuracyInput<B> {
        TopKAccuracyInput {
            outputs: self.flat_logits(),
            targets: self.flat_targets(),
        }
    }
}
impl<B: Backend> Adaptor<PerplexityInput<B>> for SequenceOutput<B> {
    fn adapt(&self) -> PerplexityInput<B> {
        PerplexityInput {
            outputs: self.flat_logits(),
            targets: self.flat_targets(),
        }
    }
}
impl<B: Backend> Adaptor<CerInput<B>> for SequenceOutput<B> {
    fn adapt(&self) -> CerInput<B> {
        CerInput {
            outputs: self.predicted_tokens(),
            targets: self.targets.clone(),
        }
    }
}
impl<B: Backend> Adaptor<WerInput<B>> for SequenceOutput<B> {
    fn adapt(&self) -> WerInput<B> {
        WerInput {
            outputs: self.predicted_tokens(),
            targets: self.targets.clone(),
        }
    }
}
