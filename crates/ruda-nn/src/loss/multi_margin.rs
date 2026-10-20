use super::Reduction;
use ruda_model::{
    config::Config,
    module::{Content, DisplaySettings, Module, ModuleDisplay},
    tensor::{Int, Tensor, backend::Backend},
};

/// Configuration for [`MultiMarginLoss`].
#[derive(Config, Debug)]
pub struct MultiMarginLossConfig {
    /// Nonnegative separation margin. Default: `1.0`.
    #[config(default = 1.0)]
    pub margin: f64,
    /// Loss exponent, either `1` or `2`. Default: `1`.
    #[config(default = 1)]
    pub p: i32,
}

impl MultiMarginLossConfig {
    /// Initialize the loss, checking the margin and exponent.
    pub fn init(&self) -> MultiMarginLoss {
        assert!(
            self.margin >= 0.0,
            "Margin for multi margin loss must be non-negative, got {}",
            self.margin
        );
        assert!(
            self.p == 1 || self.p == 2,
            "Multi margin loss only supports p = 1 or p = 2, got {}",
            self.p
        );
        MultiMarginLoss {
            margin: self.margin,
            p: self.p,
        }
    }
}

/// Multi-class hinge loss, averaged over classes and excluding the target class.
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct MultiMarginLoss {
    /// Separation margin between the target score and other scores.
    pub margin: f64,
    /// Loss exponent, either `1` or `2`.
    pub p: i32,
}

impl ModuleDisplay for MultiMarginLoss {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }
    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("margin", &self.margin)
            .add("p", &self.p)
            .optional()
    }
}

impl MultiMarginLoss {
    /// Reduce losses for `[batch, classes]` inputs and `[batch]` class indices to `[1]`.
    pub fn forward<B: Backend>(
        &self,
        input: Tensor<B, 2>,
        target: Tensor<B, 1, Int>,
        reduction: Reduction,
    ) -> Tensor<B, 1> {
        let loss = self.forward_no_reduction(input, target);
        match reduction {
            Reduction::Mean | Reduction::Auto => loss.mean(),
            Reduction::Sum => loss.sum(),
            other => panic!("{other:?} reduction is not supported"),
        }
    }

    /// Compute one loss per sample from `[batch, classes]` scores and `[batch]` indices.
    pub fn forward_no_reduction<B: Backend>(
        &self,
        input: Tensor<B, 2>,
        target: Tensor<B, 1, Int>,
    ) -> Tensor<B, 1> {
        let [batch_size, num_classes] = input.dims();
        let correct = input.clone().gather(1, target.reshape([batch_size, 1]));
        let summed = input
            .sub(correct)
            .add_scalar(self.margin)
            .clamp_min(0.0)
            .powi_scalar(self.p)
            .sum_dim(1);
        let margin_pow_p = if self.p == 2 {
            self.margin * self.margin
        } else {
            self.margin
        };
        let per_sample: Tensor<B, 1> = summed.sub_scalar(margin_pow_p).squeeze_dim(1);
        per_sample.mul_scalar(1.0 / num_classes as f64)
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
    fn test_multi_margin_loss() {
        let device = Default::default();
        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]]),
            &device,
        );
        let target = Tensor::<TestBackend, 1, Int>::from_data(TensorData::from([2, 0]), &device);

        let loss = MultiMarginLossConfig::new().init();

        let no_reduction = loss.forward_no_reduction(input.clone(), target.clone());
        let mean = loss.forward(input.clone(), target.clone(), Reduction::Mean);
        let sum = loss.forward(input, target, Reduction::Sum);

        // (1/C) * sum_{i != y} max(0, margin - x[y] + x[i]); margin=1, p=1. Reference from PyTorch.
        let expected = TensorData::from([0.3, 0.1]);
        no_reduction
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());

        mean.into_data()
            .assert_approx_eq::<FT>(&TensorData::from([0.2]), Tolerance::default());
        sum.into_data()
            .assert_approx_eq::<FT>(&TensorData::from([0.4]), Tolerance::default());
    }

    #[test]
    fn test_multi_margin_loss_p2() {
        let device = Default::default();
        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]]),
            &device,
        );
        let target = Tensor::<TestBackend, 1, Int>::from_data(TensorData::from([2, 0]), &device);

        let loss = MultiMarginLossConfig::new().with_p(2).init();
        let no_reduction = loss.forward_no_reduction(input, target);

        // squared hinge (p = 2); reference from PyTorch.
        let expected = TensorData::from([0.136_667, 0.015]);
        no_reduction
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn display() {
        let config = MultiMarginLossConfig::new().with_margin(0.5);
        let loss = config.init();

        assert_eq!(
            alloc::format!("{loss}"),
            "MultiMarginLoss {margin: 0.5, p: 1}"
        );
    }
}
