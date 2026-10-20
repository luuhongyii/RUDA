use super::Reduction;
use ruda_model::{
    config::Config,
    module::{Content, DisplaySettings, Module, ModuleDisplay},
    tensor::{Tensor, backend::Backend, linalg::lp_norm},
};

/// Configuration for [`TripletMarginLoss`].
#[derive(Config, Debug)]
pub struct TripletMarginLossConfig {
    /// Margin between positive and negative distances. Default: `1.0`.
    #[config(default = 1.0)]
    pub margin: f64,
    /// Norm degree for the distances. Default: `2.0`.
    #[config(default = 2.0)]
    pub p: f64,
}

impl TripletMarginLossConfig {
    /// Initialize the loss.
    pub fn init(&self) -> TripletMarginLoss {
        TripletMarginLoss {
            margin: self.margin,
            p: self.p,
        }
    }
}

/// Triplet margin loss using `||anchor - other||_p` distances.
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct TripletMarginLoss {
    /// Margin between positive and negative distances.
    pub margin: f64,
    /// Norm degree for the distances.
    pub p: f64,
}

impl ModuleDisplay for TripletMarginLoss {
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

impl TripletMarginLoss {
    /// Reduce losses for `[batch, embedding]` triplets to `[1]`. `Auto` uses the mean.
    pub fn forward<B: Backend>(
        &self,
        anchor: Tensor<B, 2>,
        positive: Tensor<B, 2>,
        negative: Tensor<B, 2>,
        reduction: Reduction,
    ) -> Tensor<B, 1> {
        let loss = self.forward_no_reduction(anchor, positive, negative);
        match reduction {
            Reduction::Mean | Reduction::Auto => loss.mean(),
            Reduction::Sum => loss.sum(),
            other => panic!("{other:?} reduction is not supported"),
        }
    }

    /// Compute one loss per `[batch, embedding]` triplet, returning `[batch]`.
    pub fn forward_no_reduction<B: Backend>(
        &self,
        anchor: Tensor<B, 2>,
        positive: Tensor<B, 2>,
        negative: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let distance_positive: Tensor<B, 1> =
            lp_norm(anchor.clone() - positive, self.p, 1).squeeze_dim(1);
        let distance_negative: Tensor<B, 1> = lp_norm(anchor - negative, self.p, 1).squeeze_dim(1);
        (distance_positive - distance_negative)
            .add_scalar(self.margin)
            .clamp_min(0.0)
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
    fn test_triplet_margin_loss() {
        let device = Default::default();
        // Sample 0: normal; sample 1: partial; sample 2: clamped to zero.
        let anchor = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]),
            &device,
        );
        let positive = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 1.0], [1.0, 2.0], [0.0, 0.0]]),
            &device,
        );
        let negative = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 0.0], [0.0, 0.0], [5.0, 0.0]]),
            &device,
        );

        let loss = TripletMarginLossConfig::new().init();

        let no_reduction =
            loss.forward_no_reduction(anchor.clone(), positive.clone(), negative.clone());
        let mean = loss.forward(
            anchor.clone(),
            positive.clone(),
            negative.clone(),
            Reduction::Mean,
        );
        let sum = loss.forward(anchor, positive, negative, Reduction::Sum);

        // max(0, ||a-p||_2 - ||a-n||_2 + margin); margin defaults to 1.0. Reference from PyTorch.
        let expected = TensorData::from([1.414214, 0.585786, 0.0]);
        no_reduction
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());

        mean.into_data()
            .assert_approx_eq::<FT>(&TensorData::from([0.666_667]), Tolerance::default());
        sum.into_data()
            .assert_approx_eq::<FT>(&TensorData::from([2.0]), Tolerance::default());
    }

    #[test]
    fn display() {
        let config = TripletMarginLossConfig::new().with_margin(0.5);
        let loss = config.init();

        assert_eq!(
            alloc::format!("{loss}"),
            "TripletMarginLoss {margin: 0.5, p: 2}"
        );
    }
}
