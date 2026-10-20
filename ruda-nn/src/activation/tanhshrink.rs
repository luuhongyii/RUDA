use ruda_model::module::Module;
use ruda_model::tensor::Tensor;
use ruda_model::tensor::backend::Backend;

/// Applies the Tanhshrink function element-wise, `x - tanh(x)`.
/// See also [tanhshrink](ruda_model::tensor::activation::tanhshrink)
///
#[derive(Module, Clone, Debug, Default)]
pub struct Tanhshrink;

impl Tanhshrink {
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
        ruda_model::tensor::activation::tanhshrink(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let layer = Tanhshrink::new();

        assert_eq!(alloc::format!("{layer}"), "Tanhshrink");
    }
}
