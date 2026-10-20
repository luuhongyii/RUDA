use crate::{
    checkpoint::builder::CheckpointerBuilder,
    grads::Gradients,
    graph::StepBoxed,
    tensor::{AutodiffTensor, NodeRefCount},
};
use ruda_tensor::Backend;
#[cfg(feature = "distributed")]
use ruda_tensor::distributed::DistributedBackend;

/// Client used to communicate with the autodiff server.
pub trait AutodiffClient: Send + Clone {
    /// Register a new step.
    fn register(&self, node_id: NodeRefCount, step: StepBoxed, actions: CheckpointerBuilder);
    /// Backpropagate a graph with no distributed parameter registrations.
    fn backward_local<B: Backend>(&self, tensor: AutodiffTensor<B>) -> Gradients;
    #[cfg(not(feature = "distributed"))]
    /// Call backpropagation from the given tensor.
    fn backward<B: Backend>(&self, tensor: AutodiffTensor<B>) -> Gradients;
    #[cfg(feature = "distributed")]
    /// Call backpropagation from the given tensor.
    fn backward<B: DistributedBackend>(&self, tensor: AutodiffTensor<B>) -> Gradients;
}

/// Client implementation in used.
pub type AutodiffClientImpl = super::graph::GraphMutexClient;
