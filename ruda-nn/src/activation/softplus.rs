
use ruda_model::config::Config;
use ruda_model::module::Module;
use ruda_model::module::{Content, DisplaySettings, ModuleDisplay};
use ruda_model::tensor::Tensor;
use ruda_model::tensor::activation::softplus_with_threshold;
use ruda_model::tensor::backend::Backend;

/// Softplus layer.
///
/// Applies the softplus function element-wise:
/// `softplus(x) = (1/beta) * log(1 + exp(beta * x))`
///
/// Should be created with [SoftplusConfig](SoftplusConfig).
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct Softplus {
    /// The beta value.
    pub beta: f64,
    /// Above this value of `beta * input`, return the input unchanged.
    pub threshold: f64,
}

/// Configuration to create a [Softplus](Softplus) layer using the [init function](SoftplusConfig::init).
#[derive(Config, Debug)]
pub struct SoftplusConfig {
    /// The beta value. Default is 1.0
    #[config(default = "1.0")]
    pub beta: f64,
    /// Above this value of `beta * input`, return the input unchanged.
    #[config(default = "20.0")]
    pub threshold: f64,
}

impl SoftplusConfig {
    /// Initialize a new [Softplus](Softplus) Layer
    pub fn init(&self) -> Softplus {
        Softplus {
            beta: self.beta,
            threshold: self.threshold,
        }
    }
}

impl ModuleDisplay for Softplus {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("beta", &self.beta)
            .add("threshold", &self.threshold)
            .optional()
    }
}

impl Softplus {
    /// Forward pass for the Softplus layer.
    ///
    /// See [softplus](ruda_tensor::api::activation::softplus) for more information.
    ///
    /// # Shapes
    /// - input: `[..., any]`
    /// - output: `[..., any]`
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        softplus_with_threshold(input, self.beta, self.threshold)
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use ruda_model::tensor::TensorData;
    use ruda_model::tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn test_softplus_forward() {
        let device = Default::default();
        let model: Softplus = SoftplusConfig::new().init();
        let input =
            Tensor::<TestBackend, 2>::from_data(TensorData::from([[0.0, 1.0, -1.0]]), &device);
        let out = model.forward(input);
        // softplus(0) = log(2) ≈ 0.6931
        // softplus(1) = log(1 + e) ≈ 1.3133
        // softplus(-1) = log(1 + e^-1) ≈ 0.3133
        let expected = TensorData::from([[0.6931, 1.3133, 0.3133]]);
        out.to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn test_softplus_with_beta() {
        let device = Default::default();
        let model: Softplus = SoftplusConfig::new().with_beta(2.0).init();
        let input = Tensor::<TestBackend, 2>::from_data(TensorData::from([[0.0, 1.0]]), &device);
        let out = model.forward(input);
        // softplus(0, beta=2) = (1/2) * log(1 + exp(0)) = 0.5 * log(2) ≈ 0.3466
        // softplus(1, beta=2) = (1/2) * log(1 + exp(2)) = 0.5 * log(8.389) ≈ 1.0635
        let expected = TensorData::from([[0.3466, 1.0635]]);
        out.to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn display() {
        let config = SoftplusConfig::new().init();
        assert_eq!(
            alloc::format!("{config}"),
            "Softplus {beta: 1, threshold: 20}"
        );
    }

    #[test]
    fn custom_threshold() {
        let device = Default::default();
        let input = Tensor::<TestBackend, 1>::from_floats([5.], &device);
        SoftplusConfig::new()
            .with_threshold(10.)
            .init()
            .forward(input.clone())
            .into_data()
            .assert_approx_eq::<FT>(&TensorData::from([5.0067153]), Tolerance::default());
        SoftplusConfig::new()
            .with_threshold(1.)
            .init()
            .forward(input)
            .into_data()
            .assert_eq(&TensorData::from([5.0f32]), true);
    }

    #[test]
    fn threshold_boundary_backward() {
        let device = Default::default();
        let input = Tensor::<crate::TestAutodiffBackend, 1>::from_floats([0.5, 1., 2.], &device)
            .require_grad();
        let layer = SoftplusConfig::new()
            .with_beta(2.)
            .with_threshold(2.)
            .init();
        let out = layer.forward(input.clone());
        out.to_data().assert_approx_eq::<FT>(
            &TensorData::from([0.65663084, 1.063464, 2.]),
            Tolerance::default(),
        );
        input
            .grad(&out.sum().backward())
            .unwrap()
            .into_data()
            .assert_approx_eq::<FT>(
                &TensorData::from([0.7310586, 0.8807971, 1.]),
                Tolerance::default(),
            );
    }

    #[test]
    fn wrapper_preserves_threshold() {
        let device = Default::default();
        let config: crate::activation::ActivationConfig =
            SoftplusConfig::new().with_threshold(1.).into();
        config
            .init::<TestBackend>(&device)
            .forward(Tensor::<TestBackend, 1>::from_floats([5., 100., 1000.], &device))
            .into_data()
            .assert_eq(&TensorData::from([5.0f32, 100., 1000.]), true);
    }
}
