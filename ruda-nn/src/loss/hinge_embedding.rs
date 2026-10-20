use super::Reduction;
use ruda_model::{
    config::Config,
    module::{Content, DisplaySettings, Module, ModuleDisplay},
    tensor::{Tensor, backend::Backend},
};

/// Configuration for [`HingeEmbeddingLoss`].
#[derive(Config, Debug)]
pub struct HingeEmbeddingLossConfig {
    /// Margin for negative pairs. Default: `1.0`.
    #[config(default = 1.0)]
    pub margin: f64,
}

impl HingeEmbeddingLossConfig {
    /// Initialize the loss.
    pub fn init(&self) -> HingeEmbeddingLoss {
        HingeEmbeddingLoss {
            margin: self.margin,
        }
    }
}

/// Hinge embedding loss for targets `1` (similar) and `-1` (dissimilar).
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct HingeEmbeddingLoss {
    /// Margin for negative pairs.
    pub margin: f64,
}

impl ModuleDisplay for HingeEmbeddingLoss {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }
    fn custom_content(&self, content: Content) -> Option<Content> {
        content.add("margin", &self.margin).optional()
    }
}

impl HingeEmbeddingLoss {
    /// Reduce element-wise losses to `[1]`. `Auto` uses the mean.
    pub fn forward<B: Backend, const D: usize>(
        &self,
        input: Tensor<B, D>,
        target: Tensor<B, D>,
        reduction: Reduction,
    ) -> Tensor<B, 1> {
        let loss = self.forward_no_reduction(input, target);
        match reduction {
            Reduction::Mean | Reduction::Auto => loss.mean(),
            Reduction::Sum => loss.sum(),
            other => panic!("{other:?} reduction is not supported"),
        }
    }

    /// Return `input` for positive targets and `max(0, margin - input)` otherwise.
    pub fn forward_no_reduction<B: Backend, const D: usize>(
        &self,
        input: Tensor<B, D>,
        target: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let negative = input.clone().neg().add_scalar(self.margin).clamp_min(0.0);
        negative.mask_where(target.equal_elem(1), input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use ruda_model::tensor::TensorData;
    use ruda_model::tensor::Tolerance;
    type FT = f32;

    #[test]
    fn test_hinge_embedding_loss() {
        let device = Default::default();
        let input = Tensor::<TestBackend, 1>::from_data(TensorData::from([0.5, 2.0, 1.5]), &device);
        let target =
            Tensor::<TestBackend, 1>::from_data(TensorData::from([1.0, -1.0, -1.0]), &device);

        let loss = HingeEmbeddingLossConfig::new().init();

        let no_reduction = loss.forward_no_reduction(input.clone(), target.clone());
        let mean = loss.forward(input.clone(), target.clone(), Reduction::Mean);
        let sum = loss.forward(input, target, Reduction::Sum);

        // x if y == 1 else max(0, margin - x); margin defaults to 1.0. Reference from PyTorch.
        let expected = TensorData::from([0.5, 0.0, 0.0]);
        no_reduction
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());

        mean.into_data()
            .assert_approx_eq::<FT>(&TensorData::from([0.166_667]), Tolerance::default());
        sum.into_data()
            .assert_approx_eq::<FT>(&TensorData::from([0.5]), Tolerance::default());
    }

    #[test]
    fn display() {
        let config = HingeEmbeddingLossConfig::new().with_margin(0.5);
        let loss = config.init();

        assert_eq!(alloc::format!("{loss}"), "HingeEmbeddingLoss {margin: 0.5}");
    }
}
