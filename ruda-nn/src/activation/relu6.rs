use ruda_model::module::Module;
use ruda_model::tensor::Tensor;
use ruda_model::tensor::backend::Backend;

/// Applies the ReLU6 function element-wise, clamping the output to `[0, 6]`.
/// See also [relu6](ruda_model::tensor::activation::relu6)
///
#[derive(Module, Clone, Debug, Default)]
pub struct Relu6;

impl Relu6 {
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
        ruda_model::tensor::activation::relu6(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let layer = Relu6::new();

        assert_eq!(alloc::format!("{layer}"), "Relu6");
    }
}
