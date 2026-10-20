use ruda_model::{
    config::Config,
    module::{Content, DisplaySettings, Module, ModuleDisplay},
    tensor::{Tensor, backend::Backend, linalg::l2_norm},
};

/// Configuration for [`CosineSimilarity`].
#[derive(Config, Debug)]
pub struct CosineSimilarityConfig {
    /// Dimension to reduce, with negative indices counted from the end. Default: `1`.
    #[config(default = 1)]
    pub dim: isize,
    /// Lower bound for each L2 norm. Default: `1e-8`.
    #[config(default = 1e-8)]
    pub eps: f64,
}

impl CosineSimilarityConfig {
    /// Initialize the module.
    pub fn init(&self) -> CosineSimilarity {
        CosineSimilarity {
            dim: self.dim,
            eps: self.eps,
        }
    }
}

/// Cosine similarity of rank-two inputs, removing the reduced dimension.
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct CosineSimilarity {
    /// Dimension to reduce, with negative indices counted from the end.
    pub dim: isize,
    /// Lower bound for each L2 norm.
    pub eps: f64,
}

impl ModuleDisplay for CosineSimilarity {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }
    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("dim", &self.dim)
            .add("eps", &self.eps)
            .optional()
    }
}

impl CosineSimilarity {
    /// Compute the similarity along `dim` and return a rank-one tensor.
    pub fn forward<B: Backend>(&self, x1: Tensor<B, 2>, x2: Tensor<B, 2>) -> Tensor<B, 1> {
        let dim = if self.dim < 0 { 2 + self.dim } else { self.dim };
        assert!(
            (0..2).contains(&dim),
            "CosineSimilarity dimension is outside [-2, 2)"
        );
        let dim = dim as usize;
        let norm_x1 = l2_norm(x1.clone(), dim).clamp_min(self.eps);
        let norm_x2 = l2_norm(x2.clone(), dim).clamp_min(self.eps);
        ((x1 / norm_x1) * (x2 / norm_x2))
            .sum_dim(dim)
            .squeeze_dim(dim)
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
    fn cosine_similarity_dim1() {
        let device = Default::default();
        let x1 = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 0.0], [1.0, 1.0]]),
            &device,
        );
        let x2 = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 0.0], [0.0, 1.0]]),
            &device,
        );

        let output = CosineSimilarityConfig::new().init().forward(x1, x2);

        // dot(x1, x2) / (||x1|| * ||x2||) per row; reference from PyTorch.
        let expected = TensorData::from([1.0, 0.707107]);
        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn cosine_similarity_negative_dim() {
        let device = Default::default();
        let x1 = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 0.0], [1.0, 1.0]]),
            &device,
        );
        let x2 = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 0.0], [0.0, 1.0]]),
            &device,
        );

        let output = CosineSimilarityConfig::new()
            .with_dim(-1)
            .init()
            .forward(x1, x2);

        let expected = TensorData::from([1.0, 0.707107]);
        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn display() {
        let layer = CosineSimilarityConfig::new()
            .with_dim(-1)
            .with_eps(0.5)
            .init();

        assert_eq!(
            alloc::format!("{layer}"),
            "CosineSimilarity {dim: -1, eps: 0.5}"
        );
    }
}
