use super::Reduction;
use ruda_model::{
    module::Module,
    tensor::{Tensor, activation::softplus, backend::Backend},
};

/// Binary soft margin loss `log(1 + exp(-target * logit))` for targets `-1` or `1`.
#[derive(Module, Clone, Debug)]
pub struct SoftMarginLoss;

impl Default for SoftMarginLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl SoftMarginLoss {
    /// Create the loss.
    pub fn new() -> Self {
        Self
    }

    /// Reduce element-wise losses to `[1]`. `Auto` uses the mean.
    pub fn forward<B: Backend, const D: usize>(
        &self,
        logits: Tensor<B, D>,
        targets: Tensor<B, D>,
        reduction: Reduction,
    ) -> Tensor<B, 1> {
        let loss = self.forward_no_reduction(logits, targets);
        match reduction {
            Reduction::Mean | Reduction::Auto => loss.mean(),
            Reduction::Sum => loss.sum(),
            other => panic!("{other:?} reduction is not supported"),
        }
    }

    /// Compute element-wise losses without reducing.
    pub fn forward_no_reduction<B: Backend, const D: usize>(
        &self,
        logits: Tensor<B, D>,
        targets: Tensor<B, D>,
    ) -> Tensor<B, D> {
        softplus(targets.mul(logits).neg(), 1.0)
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
    fn test_soft_margin_loss() {
        let device = Default::default();
        let logits = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[0.5, -1.0], [2.0, -0.5]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, -1.0], [-1.0, 1.0]]),
            &device,
        );

        let loss = SoftMarginLoss::new();
        let no_reduction = loss.forward_no_reduction(logits.clone(), targets.clone());
        let mean = loss.forward(logits.clone(), targets.clone(), Reduction::Mean);
        let sum = loss.forward(logits, targets, Reduction::Sum);

        // log(1 + exp(-target * logit)), computed with an independent reference.
        let expected = TensorData::from([[0.474077, 0.313262], [2.126928, 0.974077]]);
        no_reduction
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());

        mean.into_data()
            .assert_approx_eq::<FT>(&TensorData::from([0.972086]), Tolerance::default());
        sum.into_data()
            .assert_approx_eq::<FT>(&TensorData::from([3.888344]), Tolerance::default());
    }

    #[test]
    fn test_soft_margin_loss_confidently_wrong() {
        let device = Default::default();
        // A confidently wrong prediction drives `-target * logit` large and positive, which
        // is exactly where a naive softplus overflows to `inf` and poisons the backward pass.
        let logits =
            Tensor::<TestBackend, 2>::from_data(TensorData::from([[-100.0, -20.0, -5.0]]), &device);
        let targets =
            Tensor::<TestBackend, 2>::from_data(TensorData::from([[1.0, 1.0, 1.0]]), &device);

        let loss = SoftMarginLoss::new();
        let no_reduction = loss.forward_no_reduction(logits, targets);

        // The loss converges to `-target * logit` for confidently wrong predictions.
        let expected = TensorData::from([[100.0, 20.0, 5.006715]]);
        no_reduction
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }
}
