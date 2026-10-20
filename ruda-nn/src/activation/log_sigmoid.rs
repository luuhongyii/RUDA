use ruda_model::module::Module;
use ruda_model::tensor::Tensor;
use ruda_model::tensor::backend::Backend;

/// Applies the LogSigmoid activation function element-wise.
///
/// See also [log_sigmoid](ruda_model::tensor::activation::log_sigmoid).
#[derive(Module, Clone, Debug, Default)]
pub struct LogSigmoid;

impl LogSigmoid {
    /// Create the module.
    pub fn new() -> Self {
        Self {}
    }

    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[..., any]`
    /// - output: `[..., any]`
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        ruda_model::tensor::activation::log_sigmoid(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let layer = LogSigmoid::new();

        assert_eq!(alloc::format!("{layer}"), "LogSigmoid");
    }
}
