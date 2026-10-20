pub use crate::tensor::BasicAutodiffOps;

use crate::api::{Tensor, TensorPrimitive, backend::AutodiffBackend};

macro_rules! value_autodiff_context {
    ($kind:ty, $get:ident) => {
        impl<const D: usize, B: crate::Backend> Tensor<B, D, $kind> {
            /// Return the runtime autodiff association of this non-differentiable tensor.
            pub fn autodiff_context(&self) -> crate::AutodiffContext { B::$get(&self.primitive) }

            /// Whether this tensor carries an autodiff association.
            pub fn is_autodiff(&self) -> bool { self.autodiff_context() != crate::AutodiffContext::Disabled }

            /// Enable the runtime association on an autodiff-capable backend.
            pub fn autodiff(self) -> Self {
                if self.is_autodiff() { return self; }
                self.with_autodiff_context(crate::AutodiffContext::Enabled(crate::GradientCheckpointingStrategy::Disabled))
            }

            /// Drop the runtime association without copying tensor storage.
            pub fn without_autodiff(self) -> Self { self.with_autodiff_context(crate::AutodiffContext::Disabled) }

            /// Lift a tensor into an explicitly selected autodiff backend.
            pub fn into_autodiff<A: AutodiffBackend<InnerBackend = B>>(self) -> Tensor<A, D, $kind> {
                Tensor::<A, D, $kind>::from_inner(self)
            }

            /// Return the checkpointing policy used by mixed-kind operations.
            pub fn gradient_checkpointing_strategy(&self) -> Option<crate::GradientCheckpointingStrategy> {
                match self.autodiff_context() {
                    crate::AutodiffContext::Disabled => None,
                    crate::AutodiffContext::Enabled(policy) => Some(policy),
                }
            }

            /// Change the checkpointing policy without enabling autodiff implicitly.
            pub fn with_gradient_checkpointing_strategy(self, strategy: crate::GradientCheckpointingStrategy) -> Self {
                assert!(self.is_autodiff(), "Tensor::with_gradient_checkpointing_strategy requires autodiff; call Tensor::autodiff first");
                self.with_autodiff_context(crate::AutodiffContext::Enabled(strategy))
            }
        }
    };
}

value_autodiff_context!(crate::api::Int, int_autodiff_context);
value_autodiff_context!(crate::api::Bool, bool_autodiff_context);

impl<const D: usize, B: AutodiffBackend> Tensor<B, D> {
    /// Backward pass of the tensor.
    pub fn backward(&self) -> B::Gradients {
        B::backward(self.primitive.clone().tensor())
    }

    /// Get the gradients of a tensor if it exist.
    ///
    /// Returns a new reference to the same tensor. Therefore the same grad tensor can
    /// be accessed multiple times. If you only need to get the gradients one time,
    /// consider using [grad_remove](Tensor::grad_remove) for better performance.
    pub fn grad(&self, grads: &B::Gradients) -> Option<Tensor<B::InnerBackend, D>> {
        match &self.primitive {
            TensorPrimitive::Float(tensor) => B::grad(tensor, grads)
                .map(TensorPrimitive::Float)
                .map(Tensor::new),
            TensorPrimitive::QFloat(tensor) => B::q_grad(tensor, grads)
                .map(TensorPrimitive::Float)
                .map(Tensor::new),
        }
    }

    /// Remove the grad tensor from the [grads](AutodiffBackend::Gradients) struct returning the result.
    pub fn grad_remove(&self, grads: &mut B::Gradients) -> Option<Tensor<B::InnerBackend, D>> {
        match &self.primitive {
            TensorPrimitive::Float(tensor) => B::grad_remove(tensor, grads)
                .map(TensorPrimitive::Float)
                .map(Tensor::new),
            TensorPrimitive::QFloat(tensor) => {
                B::q_grad_remove(tensor, grads)
                    .map(TensorPrimitive::Float)
                    .map(Tensor::new)
            }
        }
    }

    /// Replace the grad tensor from the [grads](AutodiffBackend::Gradients) struct with the provided
    /// gradient.
    pub fn grad_replace(&self, grads: &mut B::Gradients, grad: Tensor<B::InnerBackend, D>) {
        match &self.primitive {
            TensorPrimitive::Float(tensor) => {
                B::grad_replace(tensor, grads, grad.primitive.tensor())
            }
            TensorPrimitive::QFloat(tensor) => B::q_grad_replace(
                tensor,
                grads,
                grad.primitive.tensor(),
            ),
        }
    }
}

impl<const D: usize, B: AutodiffBackend, K: BasicAutodiffOps<B>> Tensor<B, D, K> {
    /// Returns the inner tensor without the autodiff information.
    pub fn inner(self) -> Tensor<B::InnerBackend, D, K::InnerKind> {
        Tensor::new(K::inner(self.primitive))
    }

    /// Convert a tensor to the autodiff backend.
    ///
    /// # Arguments
    ///
    /// * `inner` - The tensor to convert.
    ///
    /// # Returns
    ///
    /// The tensor converted to the autodiff backend.
    pub fn from_inner(inner: Tensor<B::InnerBackend, D, K::InnerKind>) -> Self {
        Self::new(K::from_inner(inner.primitive))
    }
}

// TODO: a lot of the `tensor.inner` / `Tensor::from_inner(...)` are actually scoped to perform some operations
// so it might be cleaner and easier to manage the device etc. if we provide a method to scope the autodiff?
