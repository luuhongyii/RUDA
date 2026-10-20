use ruda_model::{
    config::Config,
    module::{Content, DisplaySettings, Module, ModuleDisplay},
    tensor::{Tensor, backend::Backend, linalg::lp_norm},
};

/// Configuration for [`PairwiseDistance`].
#[derive(Config, Debug)]
pub struct PairwiseDistanceConfig {
    /// Norm degree. Default: `2.0`.
    #[config(default = 2.0)]
    pub p: f64,
    /// Constant added to each difference. Default: `1e-6`.
    #[config(default = 1e-6)]
    pub eps: f64,
}

impl PairwiseDistanceConfig {
    /// Initialize the module.
    pub fn init(&self) -> PairwiseDistance {
        PairwiseDistance {
            p: self.p,
            eps: self.eps,
        }
    }
}

/// Row-wise distance `||x1 - x2 + eps||_p`.
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct PairwiseDistance {
    /// Norm degree.
    pub p: f64,
    /// Constant added to each difference.
    pub eps: f64,
}

impl ModuleDisplay for PairwiseDistance {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }
    fn custom_content(&self, content: Content) -> Option<Content> {
        content.add("p", &self.p).add("eps", &self.eps).optional()
    }
}

impl PairwiseDistance {
    /// Reduce `[batch, features]` differences to `[batch]` distances.
    pub fn forward<B: Backend>(&self, x1: Tensor<B, 2>, x2: Tensor<B, 2>) -> Tensor<B, 1> {
        lp_norm((x1 - x2).add_scalar(self.eps), self.p, 1).squeeze_dim(1)
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
    fn pairwise_distance_p2() {
        let device = Default::default();
        let x1 = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0], [3.0, 4.0]]),
            &device,
        );
        let x2 = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 1.0], [0.0, 0.0]]),
            &device,
        );

        let output = PairwiseDistanceConfig::new().init().forward(x1, x2);

        // ||x1 - x2 + eps||_2 per row; reference from PyTorch (eps = 1e-6 is negligible here).
        let expected = TensorData::from([1.0, 5.0]);
        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn pairwise_distance_p1() {
        let device = Default::default();
        let x1 = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0], [3.0, 4.0]]),
            &device,
        );
        let x2 = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 1.0], [0.0, 0.0]]),
            &device,
        );

        let output = PairwiseDistanceConfig::new()
            .with_p(1.0)
            .init()
            .forward(x1, x2);

        // L1 distance per row; reference from PyTorch.
        let expected = TensorData::from([1.0, 7.0]);
        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn display() {
        let layer = PairwiseDistanceConfig::new()
            .with_p(2.0)
            .with_eps(0.5)
            .init();

        assert_eq!(
            alloc::format!("{layer}"),
            "PairwiseDistance {p: 2, eps: 0.5}"
        );
    }
}
