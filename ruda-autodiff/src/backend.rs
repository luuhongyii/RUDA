use crate::{
    checkpoint::strategy::{CheckpointStrategy, NoCheckpointing},
    grads::Gradients,
    tensor::{AutodiffTensor, AutodiffQTensor, AutodiffValue},
};
use alloc::{format, string::String};
use core::marker::PhantomData;

use ruda_tensor::{
    backend::{AutodiffBackend, Backend, BackendTypes, ExecutionError},
    tensor::{BoolTensor, IntTensor, QuantizedTensor},
};

#[cfg(feature = "distributed")]
use ruda_tensor::distributed::{DistributedBackend, DistributedParamId, DistributedParams};

/// Enable auto-differentiation on a backend.
///
/// This works as a backend decorator, extending the functionality of any backend with
/// backpropagation.
#[derive(Clone, Copy, Debug, Default)]
pub struct Autodiff<B, C = NoCheckpointing> {
    _b: PhantomData<B>,
    _checkpoint_strategy: PhantomData<C>,
}

impl<B: Backend, C: CheckpointStrategy> BackendTypes for Autodiff<B, C> {
    type Device = B::Device;

    type FloatTensorPrimitive = AutodiffTensor<B>;
    type FloatElem = B::FloatElem;

    type IntTensorPrimitive = AutodiffValue<B::IntTensorPrimitive>;
    type IntElem = B::IntElem;

    type BoolTensorPrimitive = AutodiffValue<B::BoolTensorPrimitive>;
    type BoolElem = B::BoolElem;

    type QuantizedTensorPrimitive = AutodiffQTensor<B>;
}

impl<B: Backend, C: CheckpointStrategy> Backend for Autodiff<B, C> {
    fn ad_enabled(device: &Self::Device) -> bool {
        matches!(Self::autodiff_creation_context(device), ruda_tensor::AutodiffContext::Enabled(_))
    }

    fn autodiff_creation_context(device: &Self::Device) -> ruda_tensor::AutodiffContext {
        crate::context::current::<B, C>(device)
    }

    fn with_autodiff_creation_context<R>(device: &Self::Device, context: ruda_tensor::AutodiffContext, func: impl FnOnce() -> R) -> R {
        crate::context::run::<B, C, R>(device, context, func)
    }

    fn name(device: &Self::Device) -> String {
        format!("autodiff<{}>", B::name(device))
    }

    fn seed(device: &B::Device, seed: u64) {
        B::seed(device, seed)
    }

    fn rng_state(device: &B::Device) -> Result<alloc::vec::Vec<u8>, ExecutionError> {
        B::rng_state(device)
    }

    fn restore_rng_state(device: &B::Device, state: &[u8]) -> Result<(), ExecutionError> {
        B::restore_rng_state(device, state)
    }

    fn sync(device: &B::Device) -> Result<(), ExecutionError> {
        B::sync(device)
    }

    fn profile<O: Send + 'static>(device: &Self::Device, options: ruda_tensor::ProfileOptions,
        func: impl FnOnce() -> O + Send) -> Result<(O, ruda_tensor::ProfileDuration), ExecutionError> {
        B::profile(device, options, func)
    }
    fn profile_start(device: &Self::Device) -> Result<Option<ruda_tensor::ProfileToken>, ExecutionError> {
        B::profile_start(device)
    }
    fn profile_end(device: &Self::Device, token: ruda_tensor::ProfileToken, options: ruda_tensor::ProfileOptions)
        -> Result<ruda_tensor::ProfileDuration, ExecutionError> { B::profile_end(device, token, options) }
    fn profile_abandon(device: &Self::Device, token: ruda_tensor::ProfileToken) { B::profile_abandon(device, token); }

    fn flush(device: &B::Device) -> Result<(), ExecutionError> {
        B::flush(device)
    }

    fn graph_prepare(device: &B::Device) -> Result<(), ExecutionError> { B::graph_prepare(device) }
    fn graph_start_capture(device: &B::Device) -> Result<(), ExecutionError> { B::graph_start_capture(device) }
    fn graph_stop_capture(device: &B::Device) -> Result<ruda_tensor::BackendGraph, ExecutionError> { B::graph_stop_capture(device) }
    fn graph_abort_capture(device: &B::Device) { B::graph_abort_capture(device); }
    unsafe fn graph_replay(device: &B::Device, graph: &ruda_tensor::BackendGraph) -> Result<(), ExecutionError> {
        unsafe { B::graph_replay(device, graph) }
    }

    fn memory_persistent_allocations<
        Output: Send,
        Input: Send,
        Func: Fn(Input) -> Output + Send,
    >(
        device: &Self::Device,
        input: Input,
        func: Func,
    ) -> Output {
        B::memory_persistent_allocations(device, input, func)
    }

    fn memory_cleanup(device: &Self::Device) {
        B::memory_cleanup(device)
    }

    fn staging<'a, Iter>(data: Iter, device: &Self::Device)
    where
        Iter: Iterator<Item = &'a mut ruda_tensor::TensorData>,
    {
        B::staging(data, device);
    }

    fn supports_dtype(device: &Self::Device, dtype: ruda_core::tensor::DType) -> bool {
        B::supports_dtype(device, dtype)
    }

    fn dtype_usage(device: &Self::Device, dtype: ruda_core::tensor::DType) -> ruda_tensor::DTypeUsageSet {
        B::dtype_usage(device, dtype)
    }

    fn device_count(type_id: u16) -> usize {
        B::device_count(type_id)
    }
}

#[cfg(not(feature = "distributed"))]
impl<B: Backend, C: CheckpointStrategy> AutodiffBackend for Autodiff<B, C> {
    type InnerBackend = B;
    type Gradients = Gradients;

    fn q_grad(tensor: &AutodiffQTensor<B>, grads: &Gradients) -> Option<B::FloatTensorPrimitive> {
        tensor.surrogate.as_ref()?.grad(grads)
    }
    fn q_grad_remove(tensor: &AutodiffQTensor<B>, grads: &mut Gradients) -> Option<B::FloatTensorPrimitive> {
        tensor.surrogate.as_ref()?.grad_remove(grads)
    }
    fn q_grad_replace(tensor: &AutodiffQTensor<B>, grads: &mut Gradients, grad: B::FloatTensorPrimitive) {
        tensor.surrogate.as_ref().expect("Quantized tensor has no gradient surrogate").grad_replace(grads, grad);
    }

    fn backward(tensor: AutodiffTensor<B>) -> Gradients {
        tensor.backward()
    }

    fn grad(tensor: &AutodiffTensor<B>, grads: &Gradients) -> Option<B::FloatTensorPrimitive> {
        tensor.grad(grads)
    }

    fn grad_remove(
        tensor: &AutodiffTensor<B>,
        grads: &mut Gradients,
    ) -> Option<B::FloatTensorPrimitive> {
        tensor.grad_remove(grads)
    }
    fn inner(tensor: AutodiffTensor<B>) -> B::FloatTensorPrimitive {
        tensor.primitive
    }

    fn from_inner(tensor: B::FloatTensorPrimitive) -> AutodiffTensor<B> {
        let context = match Self::autodiff_creation_context(&B::float_device(&tensor)) {
            ruda_tensor::AutodiffContext::Disabled => ruda_tensor::AutodiffContext::Enabled(C::policy()),
            context => context,
        };
        AutodiffTensor::new(tensor).with_context::<C>(context)
    }

    fn grad_replace(
        tensor: &AutodiffTensor<B>,
        grads: &mut Self::Gradients,
        grad: B::FloatTensorPrimitive,
    ) {
        tensor.grad_replace(grads, grad);
    }

    fn int_inner(tensor: IntTensor<Self>) -> IntTensor<Self::InnerBackend> {
        tensor.primitive
    }

    fn bool_inner(tensor: BoolTensor<Self>) -> BoolTensor<Self::InnerBackend> {
        tensor.primitive
    }

    fn int_from_inner(tensor: IntTensor<Self::InnerBackend>) -> IntTensor<Self> {
        let context = match Self::autodiff_creation_context(&B::int_device(&tensor)) {
            ruda_tensor::AutodiffContext::Disabled => ruda_tensor::AutodiffContext::Enabled(C::policy()),
            context => context,
        };
        AutodiffValue::with_context(tensor, context)
    }

    fn bool_from_inner(tensor: BoolTensor<Self::InnerBackend>) -> BoolTensor<Self> {
        let context = match Self::autodiff_creation_context(&B::bool_device(&tensor)) {
            ruda_tensor::AutodiffContext::Disabled => ruda_tensor::AutodiffContext::Enabled(C::policy()),
            context => context,
        };
        AutodiffValue::with_context(tensor, context)
    }

    fn q_inner(tensor: QuantizedTensor<Self>) -> QuantizedTensor<Self::InnerBackend> {
        tensor.primitive
    }

    fn q_from_inner(tensor: QuantizedTensor<Self::InnerBackend>) -> QuantizedTensor<Self> {
        let context = match Self::autodiff_creation_context(&B::q_device(&tensor)) {
            ruda_tensor::AutodiffContext::Disabled => ruda_tensor::AutodiffContext::Enabled(C::policy()),
            context => context,
        };
        AutodiffQTensor::untracked(tensor).with_context::<C>(context)
    }
}

#[cfg(feature = "distributed")]
impl<B: DistributedBackend, C: CheckpointStrategy> AutodiffBackend for Autodiff<B, C> {
    type InnerBackend = B;
    type Gradients = Gradients;

    fn q_grad(tensor: &AutodiffQTensor<B>, grads: &Gradients) -> Option<B::FloatTensorPrimitive> {
        tensor.surrogate.as_ref()?.grad(grads)
    }
    fn q_grad_remove(tensor: &AutodiffQTensor<B>, grads: &mut Gradients) -> Option<B::FloatTensorPrimitive> {
        tensor.surrogate.as_ref()?.grad_remove(grads)
    }
    fn q_grad_replace(tensor: &AutodiffQTensor<B>, grads: &mut Gradients, grad: B::FloatTensorPrimitive) {
        tensor.surrogate.as_ref().expect("Quantized tensor has no gradient surrogate").grad_replace(grads, grad);
    }

    fn backward(tensor: AutodiffTensor<B>) -> Gradients {
        tensor.backward()
    }

    fn grad(tensor: &AutodiffTensor<B>, grads: &Gradients) -> Option<B::FloatTensorPrimitive> {
        tensor.grad(grads)
    }

    fn grad_remove(
        tensor: &AutodiffTensor<B>,
        grads: &mut Gradients,
    ) -> Option<B::FloatTensorPrimitive> {
        tensor.grad_remove(grads)
    }
    fn inner(tensor: AutodiffTensor<B>) -> B::FloatTensorPrimitive {
        tensor.primitive
    }

    fn from_inner(tensor: B::FloatTensorPrimitive) -> AutodiffTensor<B> {
        let context = match Self::autodiff_creation_context(&B::float_device(&tensor)) {
            ruda_tensor::AutodiffContext::Disabled => ruda_tensor::AutodiffContext::Enabled(C::policy()),
            context => context,
        };
        AutodiffTensor::new(tensor).with_context::<C>(context)
    }

    fn grad_replace(
        tensor: &AutodiffTensor<B>,
        grads: &mut Self::Gradients,
        grad: B::FloatTensorPrimitive,
    ) {
        tensor.grad_replace(grads, grad);
    }

    fn int_inner(tensor: IntTensor<Self>) -> IntTensor<Self::InnerBackend> {
        tensor.primitive
    }

    fn bool_inner(tensor: BoolTensor<Self>) -> BoolTensor<Self::InnerBackend> {
        tensor.primitive
    }

    fn int_from_inner(tensor: IntTensor<Self::InnerBackend>) -> IntTensor<Self> {
        let context = match Self::autodiff_creation_context(&B::int_device(&tensor)) {
            ruda_tensor::AutodiffContext::Disabled => ruda_tensor::AutodiffContext::Enabled(C::policy()),
            context => context,
        };
        AutodiffValue::with_context(tensor, context)
    }

    fn bool_from_inner(tensor: BoolTensor<Self::InnerBackend>) -> BoolTensor<Self> {
        let context = match Self::autodiff_creation_context(&B::bool_device(&tensor)) {
            ruda_tensor::AutodiffContext::Disabled => ruda_tensor::AutodiffContext::Enabled(C::policy()),
            context => context,
        };
        AutodiffValue::with_context(tensor, context)
    }

    fn q_inner(tensor: QuantizedTensor<Self>) -> QuantizedTensor<Self::InnerBackend> {
        tensor.primitive
    }

    fn q_from_inner(tensor: QuantizedTensor<Self::InnerBackend>) -> QuantizedTensor<Self> {
        let context = match Self::autodiff_creation_context(&B::q_device(&tensor)) {
            ruda_tensor::AutodiffContext::Disabled => ruda_tensor::AutodiffContext::Enabled(C::policy()),
            context => context,
        };
        AutodiffQTensor::untracked(tensor).with_context::<C>(context)
    }

    fn set_distributed_params(
        tensor: AutodiffTensor<B>,
        param_id: DistributedParamId,
    ) -> AutodiffTensor<B> {
        tensor.grad_distributed(param_id)
    }

    fn distributed_params(tensor: &AutodiffTensor<B>) -> Option<DistributedParams> {
        tensor.node.distributed_params.clone()
    }

    fn is_distributed(tensor: &AutodiffTensor<B>) -> bool {
        tensor.node.distributed_params.is_some()
    }
}
