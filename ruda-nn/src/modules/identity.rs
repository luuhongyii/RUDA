use ruda_model::tensor::Tensor;
use ruda_model::tensor::backend::Backend;
use ruda_model::module::Module;

/// Identity Module.
#[derive(Module, Clone, Default, Debug)]
pub struct Identity;

impl Identity {
    /// Create the module.
    pub fn new() -> Self {
        Self {}
    }

    /// Forward pass, returns the input tensor.
    pub fn forward<B: Backend, const R: usize>(&self, input: Tensor<B, R>) -> Tensor<B, R> {
        input
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let layer = Identity::new();

        assert_eq!(alloc::format!("{layer}"), "Identity");
    }
}
