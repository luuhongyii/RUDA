use super::Reduction;
use ruda_model::{
    config::Config,
    module::{Content, DisplaySettings, Module, ModuleDisplay},
    tensor::{Tensor, activation::relu, backend::Backend},
};

/// Configuration for [`MarginRankingLoss`].
#[derive(Config, Debug)]
pub struct MarginRankingLossConfig {
    /// Nonnegative separation margin. Default: `0.0`.
    #[config(default = 0.0)]
    pub margin: f64,
}

impl MarginRankingLossConfig {
    /// Initialize the loss, rejecting a negative margin.
    pub fn init(&self) -> MarginRankingLoss {
        assert!(
            self.margin >= 0.0,
            "Margin for margin ranking loss must be a non-negative number."
        );
        MarginRankingLoss {
            margin: self.margin,
        }
    }
}

/// Pairwise ranking loss `max(0, -target * (first - second) + margin)`.
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct MarginRankingLoss {
    /// Separation margin for correctly ranked inputs.
    pub margin: f64,
}

impl ModuleDisplay for MarginRankingLoss {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }
    fn custom_content(&self, content: Content) -> Option<Content> {
        content.add("margin", &self.margin).optional()
    }
}

impl MarginRankingLoss {
    /// Reduce element-wise losses to `[1]`. `Auto` uses the mean.
    pub fn forward<B: Backend, const D: usize>(
        &self,
        first: Tensor<B, D>,
        second: Tensor<B, D>,
        target: Tensor<B, D>,
        reduction: Reduction,
    ) -> Tensor<B, 1> {
        let loss = self.forward_no_reduction(first, second, target);
        match reduction {
            Reduction::Mean | Reduction::Auto => loss.mean(),
            Reduction::Sum => loss.sum(),
            other => panic!("{other:?} reduction is not supported"),
        }
    }

    /// Compute element-wise losses for targets `1` or `-1`.
    pub fn forward_no_reduction<B: Backend, const D: usize>(
        &self,
        first: Tensor<B, D>,
        second: Tensor<B, D>,
        target: Tensor<B, D>,
    ) -> Tensor<B, D> {
        relu(target.mul(first - second).neg().add_scalar(self.margin))
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
    fn test_margin_ranking_loss() {
        let device = Default::default();

        let first = Tensor::<TestBackend, 1>::from_data(TensorData::from([1., 2., 3.]), &device);
        let second = Tensor::<TestBackend, 1>::from_data(TensorData::from([2., 1., 0.5]), &device);
        let target = Tensor::<TestBackend, 1>::from_data(TensorData::from([1., -1., 1.]), &device);

        let loss = MarginRankingLossConfig::new().with_margin(0.5).init();

        let no_reduction = loss.forward_no_reduction(first.clone(), second.clone(), target.clone());
        let mean = loss.forward(
            first.clone(),
            second.clone(),
            target.clone(),
            Reduction::Mean,
        );
        let sum = loss.forward(first, second, target, Reduction::Sum);

        let expected = TensorData::from([1.5, 1.5, 0.0]);
        no_reduction
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());

        let expected = TensorData::from([1.0]);
        mean.into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());

        let expected = TensorData::from([3.0]);
        sum.into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn display() {
        let config = MarginRankingLossConfig::new().with_margin(0.5);
        let loss = config.init();

        assert_eq!(alloc::format!("{loss}"), "MarginRankingLoss {margin: 0.5}");
    }
}
