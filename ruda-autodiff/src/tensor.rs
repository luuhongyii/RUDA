use crate::{
    checkpoint::{base::Checkpointer, builder::CheckpointerBuilder},
    grads::Gradients,
    graph::{ComputingProperty, Node, NodeId, NodeRef, Parent, Requirement, Step},
    runtime::{AutodiffClient, AutodiffClientImpl},
};
use alloc::{boxed::Box, sync::Arc, vec};
use ruda_tensor::{Backend, TensorMetadata};

#[cfg(feature = "distributed")]
use ruda_tensor::distributed::{DistributedBackend, DistributedParamId, DistributedParams};

#[derive(Debug, Clone)]
pub struct AutodiffTensor<B: Backend> {
    pub primitive: B::FloatTensorPrimitive,
    pub node: NodeRef,
    pub rc: NodeRefCount,
}

impl<B: Backend> TensorMetadata for AutodiffTensor<B> {
    fn dtype(&self) -> ruda_core::tensor::DType {
        self.primitive.dtype()
    }

    fn shape(&self) -> ruda_core::tensor::Shape {
        self.primitive.shape()
    }

    fn rank(&self) -> usize {
        self.primitive.rank()
    }

    fn can_mut(&self) -> bool {
        self.primitive.can_mut()
    }
}

pub type NodeRefCount = Arc<NodeId>;

/// A non-differentiable value carrying the runtime context used by mixed-kind operations.
#[derive(Debug, Clone)]
pub struct AutodiffValue<T: TensorMetadata> {
    pub primitive: T,
    pub context: ruda_tensor::AutodiffContext,
}

impl<T: TensorMetadata> AutodiffValue<T> {
    pub fn new<C: crate::checkpoint::strategy::CheckpointStrategy>(primitive: T) -> Self {
        Self { primitive, context: ruda_tensor::AutodiffContext::Enabled(C::policy()) }
    }

    pub fn with_context(primitive: T, context: ruda_tensor::AutodiffContext) -> Self {
        Self { primitive, context }
    }

    pub(crate) fn set_context<C: crate::checkpoint::strategy::CheckpointStrategy>(mut self, context: ruda_tensor::AutodiffContext) -> Self {
        if let ruda_tensor::AutodiffContext::Enabled(policy @ ruda_tensor::GradientCheckpointingStrategy::Custom(_)) = context {
            assert_eq!(policy, C::policy(), "Checkpointing strategy is not provided by this backend");
        }
        self.context = context;
        self
    }
}

impl<T: TensorMetadata> TensorMetadata for AutodiffValue<T> {
    fn dtype(&self) -> ruda_core::tensor::DType { self.primitive.dtype() }
    fn shape(&self) -> ruda_core::tensor::Shape { self.primitive.shape() }
    fn rank(&self) -> usize { self.primitive.rank() }
    fn can_mut(&self) -> bool { self.primitive.can_mut() }
}

#[derive(new, Debug)]
pub(crate) struct RootStep {
    node: NodeRef,
}

impl Step for RootStep {
    fn step(self: Box<Self>, _grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
        // Nothing to do
    }

    fn node(&self) -> NodeId {
        self.node.id
    }

    fn parents(&self) -> &[Parent] {
        &self.node.parents
    }

    fn depth(&self) -> usize {
        self.node.order
    }

    #[cfg(feature = "distributed")]
    fn distributed_params(&self) -> Option<DistributedParams> {
        self.node.distributed_params.clone()
    }

    #[cfg(feature = "distributed")]
    fn distributed_backend(&self) -> Option<core::any::TypeId> { self.node.native_backend }
}

impl<B: Backend> AutodiffTensor<B> {
    pub(crate) fn new_leaf<C: crate::checkpoint::strategy::CheckpointStrategy>(primitive: B::FloatTensorPrimitive) -> Self {
        let context = crate::context::current::<B, C>(&B::float_device(&primitive));
        Self::new(primitive).with_context::<C>(context)
    }

    pub(crate) fn context<C: crate::checkpoint::strategy::CheckpointStrategy>(&self) -> ruda_tensor::AutodiffContext {
        if self.node.autodiff_enabled {
            ruda_tensor::AutodiffContext::Enabled(self.node.checkpointing.unwrap_or_else(C::policy))
        } else { ruda_tensor::AutodiffContext::Disabled }
    }

    pub(crate) fn without_autodiff(self) -> Self {
        if !self.node.autodiff_enabled { return self; }
        Self::plain(self.primitive)
    }

    fn plain(primitive: B::FloatTensorPrimitive) -> Self {
        let mut tensor = Self::new(primitive);
        Arc::make_mut(&mut tensor.node).autodiff_enabled = false;
        tensor
    }

    pub(crate) fn with_context<C: crate::checkpoint::strategy::CheckpointStrategy>(self, context: ruda_tensor::AutodiffContext) -> Self {
        match context {
            ruda_tensor::AutodiffContext::Disabled => self.without_autodiff(),
            ruda_tensor::AutodiffContext::Enabled(policy) => {
                if matches!(policy, ruda_tensor::GradientCheckpointingStrategy::Custom(_)) {
                    assert_eq!(policy, C::policy(), "Checkpointing strategy is not provided by this backend");
                }
                let tensor = if self.node.autodiff_enabled { self } else { Self::new(self.primitive) };
                tensor.with_checkpointing(Some(policy))
            }
        }
    }

    pub(crate) fn with_checkpointing(mut self, policy: Option<ruda_tensor::GradientCheckpointingStrategy>) -> Self {
        Arc::make_mut(&mut self.node).checkpointing = policy;
        self
    }

    /// Create a new leaf tensor.
    pub fn new(primitive: B::FloatTensorPrimitive) -> Self {
        let id = NodeId::new();
        let node: NodeRef = Node::new(
            vec![],
            0,
            id,
            Requirement::None,
            ComputingProperty::Ambiguous,
            AutodiffClientImpl::new(),
            #[cfg(feature = "distributed")]
            None,
        )
        .with_backend::<B>()
        .into();

        Self {
            rc: Arc::new(node.id),
            primitive,
            node: node.clone(),
        }
    }

    pub fn is_tracked(&self) -> bool {
        !self.node.requirement.is_none()
    }

    /// Mark the tensor as requiring gradients.
    ///
    /// # Panics
    ///
    /// It panics if the tensor is not a leaf.
    pub fn require_grad(mut self) -> Self {
        assert!(self.node.autodiff_enabled, "Tensor::require_grad requires autodiff; call Tensor::autodiff first");
        match self.node.requirement {
            Requirement::Grad => self,
            Requirement::GradInBackward => {
                panic!("Can't convert a non leaf tensor into a tracked tensor")
            }
            Requirement::None => {
                self.node = Node::new(
                    vec![],
                    0,
                    self.node.id,
                    Requirement::Grad,
                    self.node.properties.clone(),
                    self.node.client.clone(),
                    #[cfg(feature = "distributed")]
                    self.node.distributed_params.clone(),
                )
                .with_backend::<B>()
                .with_checkpointing(self.node.checkpointing)
                .into();
                let step = RootStep::new(self.node.clone());

                self.register_step(step, CheckpointerBuilder::default())
            }
        }
    }

    /// Create a tensor from parent infos.
    pub fn from_parents(
        primitive: B::FloatTensorPrimitive,
        parent_nodes: &[NodeRef],
        requirement: Requirement,
        computing_properties: ComputingProperty,
    ) -> Self {
        if !parent_nodes.is_empty() && parent_nodes.iter().all(|node| !node.autodiff_enabled) {
            return Self::plain(primitive);
        }
        let order = parent_nodes
            .iter()
            .map(|node| node.order)
            .reduce(usize::max)
            .unwrap_or(0)
            + 1;

        let client = parent_nodes
            .first()
            .map(|node| node.client.clone())
            .unwrap_or_else(AutodiffClientImpl::new);

        let mut node = Node::new(
            parent_nodes
                .iter()
                .filter_map(|node| node.clone_if_require_grad())
                .map(|node| Parent::new(node.id, node.order == 0))
                .collect(),
            order,
            NodeId::new(),
            requirement,
            computing_properties,
            client,
            #[cfg(feature = "distributed")]
            None,
        )
        .with_backend::<B>()
        .with_checkpointing(parent_nodes.iter().filter(|node| node.autodiff_enabled).find_map(|node| node.checkpointing));
        node.autodiff_enabled = parent_nodes.is_empty() || parent_nodes.iter().any(|node| node.autodiff_enabled);
        let node: NodeRef = node.into();

        Self {
            rc: Arc::new(node.id),
            primitive,
            node,
        }
    }

    /// Register a step into a graph for that tensor.
    ///
    /// # Warning
    ///
    /// This should be called only once per tensor.
    pub fn register_step<S: Step + 'static>(
        self,
        step_that_created_the_tensor: S,
        actions: CheckpointerBuilder,
    ) -> Self {
        if !self.node.autodiff_enabled { return self; }
        self.node.client.register(
            self.rc.clone(),
            Box::new(step_that_created_the_tensor),
            actions,
        );
        self
    }

    pub fn into_primitive(self) -> B::FloatTensorPrimitive {
        self.primitive
    }

    /// Backpropagate without collective registration, including mixed-native-backend graphs.
    /// Distributed parameter graphs must use the distributed backward entry point instead.
    pub fn backward_local(self) -> Gradients {
        assert!(self.is_tracked(), "Backward requires a tracked autodiff tensor");
        let client = self.node.client.clone();
        AutodiffClient::backward_local::<B>(&client, self)
    }

    #[cfg(not(feature = "distributed"))]
    pub fn backward(self) -> Gradients {
        assert!(self.is_tracked(), "Backward requires a tracked autodiff tensor");
        let client = self.node.client.clone();

        AutodiffClient::backward::<B>(&client, self)
    }

    pub fn grad(&self, grads: &Gradients) -> Option<B::FloatTensorPrimitive> {
        if !matches!(self.node.requirement, Requirement::Grad) { return None; }
        grads.get::<B>(self)
    }

    pub fn grad_remove(&self, grads: &mut Gradients) -> Option<B::FloatTensorPrimitive> {
        if !matches!(self.node.requirement, Requirement::Grad) { return None; }
        grads.remove::<B>(self)
    }

    pub fn grad_replace(&self, grads: &mut Gradients, grad: B::FloatTensorPrimitive) {
        grads.remove::<B>(self);
        grads.register::<B>(self.node.id, grad);
    }

    #[cfg(feature = "distributed")]
    /// Mark the tensor as distributed across multiple devices.
    /// Its gradients will be automatically aggregated from those devices after the backward pass.
    ///
    /// # Arguments
    ///
    /// * `param_id` - The module tensor's [`DistributedParamId`].
    pub fn grad_distributed(mut self, param_id: DistributedParamId) -> Self {
        self.node = Node::new(
            vec![],
            0,
            self.node.id,
            self.node.requirement,
            self.node.properties.clone(),
            self.node.client.clone(),
            Some(DistributedParams { param_id }),
        )
        .with_backend::<B>()
        .with_checkpointing(self.node.checkpointing)
        .with_autodiff_enabled(self.node.autodiff_enabled)
        .into();
        let step = RootStep::new(self.node.clone());

        self.register_step(step, CheckpointerBuilder::default())
    }
}

#[cfg(feature = "distributed")]
impl<B: DistributedBackend> AutodiffTensor<B> {
    pub fn backward(self) -> Gradients {
        assert!(self.is_tracked(), "Backward requires a tracked autodiff tensor");
        let device = B::float_device(&self.primitive);
        let client = self.node.client.clone();

        let mut grads = AutodiffClient::backward::<B>(&client, self);
        grads.sync_collective::<B>(&device);
        grads
    }
}


/// Quantized storage with an optional differentiable dequantized surrogate.
///
/// Input gradients use an unclipped identity straight-through estimator (STE).
/// Quantization parameters are constants, not learnable parameters.
#[derive(Debug, Clone)]
pub struct AutodiffQTensor<B: Backend> {
    /// Quantized storage owned by the wrapped backend.
    pub primitive: B::QuantizedTensorPrimitive,
    pub(crate) surrogate: Option<AutodiffTensor<B>>,
    pub(crate) checkpointing: Option<ruda_tensor::GradientCheckpointingStrategy>,
    pub(crate) autodiff_enabled: bool,
}

impl<B: Backend> AutodiffQTensor<B> {
    pub(crate) fn context<C: crate::checkpoint::strategy::CheckpointStrategy>(&self) -> ruda_tensor::AutodiffContext {
        if self.autodiff_enabled {
            ruda_tensor::AutodiffContext::Enabled(self.checkpointing.unwrap_or_else(C::policy))
        } else { ruda_tensor::AutodiffContext::Disabled }
    }

    pub(crate) fn with_context<C: crate::checkpoint::strategy::CheckpointStrategy>(mut self, context: ruda_tensor::AutodiffContext) -> Self {
        match context {
            ruda_tensor::AutodiffContext::Disabled => {
                self.autodiff_enabled = false;
                self.checkpointing = None;
                self.surrogate = None;
                self
            }
            ruda_tensor::AutodiffContext::Enabled(policy) => {
                if matches!(policy, ruda_tensor::GradientCheckpointingStrategy::Custom(_)) {
                    assert_eq!(policy, C::policy(), "Checkpointing strategy is not provided by this backend");
                }
                self.autodiff_enabled = true;
                self.with_checkpointing(Some(policy))
            }
        }
    }

    pub(crate) fn with_checkpointing(mut self, policy: Option<ruda_tensor::GradientCheckpointingStrategy>) -> Self {
        self.checkpointing = policy;
        self.surrogate = self.surrogate.map(|tensor| tensor.with_checkpointing(policy));
        self
    }

    /// Associate native quantized storage without creating a gradient surrogate.
    pub fn untracked(primitive: B::QuantizedTensorPrimitive) -> Self {
        Self { primitive, surrogate: None, checkpointing: None, autodiff_enabled: true }
    }

    /// Query a retained surrogate gradient without dequantizing the stored value.
    pub fn grad(&self, grads: &Gradients) -> Option<B::FloatTensorPrimitive> {
        self.surrogate.as_ref()?.grad(grads)
    }

    /// Remove a retained surrogate gradient.
    pub fn grad_remove(&self, grads: &mut Gradients) -> Option<B::FloatTensorPrimitive> {
        self.surrogate.as_ref()?.grad_remove(grads)
    }

    /// Replace the retained gradient of an existing surrogate.
    pub fn grad_replace(&self, grads: &mut Gradients, grad: B::FloatTensorPrimitive) {
        self.surrogate.as_ref().expect("Quantized tensor has no gradient surrogate").grad_replace(grads, grad);
    }
}

impl<B: Backend> TensorMetadata for AutodiffQTensor<B> {
    fn dtype(&self) -> ruda_core::tensor::DType { self.primitive.dtype() }
    fn shape(&self) -> ruda_core::tensor::Shape { self.primitive.shape() }
    fn rank(&self) -> usize { self.primitive.rank() }
    fn can_mut(&self) -> bool { self.primitive.can_mut() }
}
impl<B: Backend> ruda_tensor::QTensorPrimitive for AutodiffQTensor<B> {
    fn scheme(&self) -> &ruda_core::tensor::QuantScheme {
        ruda_tensor::QTensorPrimitive::scheme(&self.primitive)
    }
}
