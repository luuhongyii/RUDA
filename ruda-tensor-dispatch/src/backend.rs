use alloc::string::String;
use ruda_tensor::{Backend, BackendTypes, AutodiffBackend, AutodiffContext, GradientCheckpointingStrategy,
    DType, DTypeUsageSet, DeviceId, DeviceOps, ExecutionError, TensorData};
use ruda_tensor::backend::{ProfileOptions, ProfileToken, ProfileDuration, BackendGraph};
use crate::{DispatchDevice, DispatchTensor, BackendTensor, context};

#[derive(Clone, Copy, Debug, Default)]
pub struct Dispatch;

impl BackendTypes for Dispatch {
    type Device = DispatchDevice;
    type FloatTensorPrimitive = DispatchTensor;
    type IntTensorPrimitive = DispatchTensor;
    type BoolTensorPrimitive = DispatchTensor;
    type QuantizedTensorPrimitive = DispatchTensor;
    type FloatElem = f32;
    type IntElem = i32;
    type BoolElem = bool;
}

impl Backend for Dispatch {
    fn ad_enabled(device: &DispatchDevice) -> bool {
        matches!(context::current(device), AutodiffContext::Enabled(_))
    }
    fn autodiff_creation_context(device: &DispatchDevice) -> AutodiffContext { context::current(device) }
    fn with_autodiff_creation_context<R>(device: &DispatchDevice, context: AutodiffContext, func: impl FnOnce() -> R) -> R {
        context::run(device, context, func)
    }
    fn name(device: &DispatchDevice) -> String {
        macro_rules! call {
            ($B:ty, $variant:ident, $device:expr) => { <$B>::name($device) };
        }
        native_dispatch!(device, call)
    }
    fn seed(device: &DispatchDevice, seed: u64) {
        macro_rules! call {
            ($B:ty, $variant:ident, $device:expr) => { <$B>::seed($device, seed) };
        }
        native_dispatch!(device, call)
    }
    fn sync(device: &DispatchDevice) -> Result<(), ExecutionError> {
        macro_rules! call {
            ($B:ty, $variant:ident, $device:expr) => { <$B>::sync($device) };
        }
        native_dispatch!(device, call)
    }
    fn flush(device: &DispatchDevice) -> Result<(), ExecutionError> {
        macro_rules! call {
            ($B:ty, $variant:ident, $device:expr) => { <$B>::flush($device) };
        }
        native_dispatch!(device, call)
    }
    fn memory_cleanup(device: &DispatchDevice) {
        macro_rules! call {
            ($B:ty, $variant:ident, $device:expr) => { <$B>::memory_cleanup($device) };
        }
        native_dispatch!(device, call)
    }
    fn memory_persistent_allocations<Output: Send, Input: Send, Func: Fn(Input) -> Output + Send>(device: &DispatchDevice, input: Input, func: Func) -> Output {
        macro_rules! call {
            ($B:ty, $variant:ident, $device:expr) => { <$B>::memory_persistent_allocations($device, input, func) };
        }
        native_dispatch!(device, call)
    }
    fn profile<O: Send + 'static>(device: &DispatchDevice, options: ProfileOptions, func: impl FnOnce() -> O + Send) -> Result<(O, ProfileDuration), ExecutionError> {
        macro_rules! call {
            ($B:ty, $variant:ident, $device:expr) => { <$B>::profile($device, options, func) };
        }
        native_dispatch!(device, call)
    }
    fn profile_start(device: &DispatchDevice) -> Result<Option<ProfileToken>, ExecutionError> {
        macro_rules! call {
            ($B:ty, $variant:ident, $device:expr) => { <$B>::profile_start($device) };
        }
        native_dispatch!(device, call)
    }
    fn profile_end(device: &DispatchDevice, token: ProfileToken, options: ProfileOptions) -> Result<ProfileDuration, ExecutionError> {
        macro_rules! call {
            ($B:ty, $variant:ident, $device:expr) => { <$B>::profile_end($device, token, options) };
        }
        native_dispatch!(device, call)
    }
    fn profile_abandon(device: &DispatchDevice, token: ProfileToken) {
        macro_rules! call {
            ($B:ty, $variant:ident, $device:expr) => { <$B>::profile_abandon($device, token) };
        }
        native_dispatch!(device, call)
    }
    fn graph_prepare(device: &DispatchDevice) -> Result<(), ExecutionError> {
        macro_rules! call {
            ($B:ty, $variant:ident, $device:expr) => { <$B>::graph_prepare($device) };
        }
        native_dispatch!(device, call)
    }
    fn graph_start_capture(device: &DispatchDevice) -> Result<(), ExecutionError> {
        macro_rules! call {
            ($B:ty, $variant:ident, $device:expr) => { <$B>::graph_start_capture($device) };
        }
        native_dispatch!(device, call)
    }
    fn graph_stop_capture(device: &DispatchDevice) -> Result<BackendGraph, ExecutionError> {
        macro_rules! call {
            ($B:ty, $variant:ident, $device:expr) => { <$B>::graph_stop_capture($device) };
        }
        native_dispatch!(device, call)
    }
    fn graph_abort_capture(device: &DispatchDevice) {
        macro_rules! call {
            ($B:ty, $variant:ident, $device:expr) => { <$B>::graph_abort_capture($device) };
        }
        native_dispatch!(device, call)
    }
    fn dtype_usage(device: &DispatchDevice, dtype: DType) -> DTypeUsageSet {
        macro_rules! call {
            ($B:ty, $variant:ident, $device:expr) => { <$B>::dtype_usage($device, dtype) };
        }
        native_dispatch!(device, call)
    }
    fn staging<'a, Iter: Iterator<Item = &'a mut TensorData>>(data: Iter, device: &DispatchDevice) {
        macro_rules! call {
            ($B:ty, $variant:ident, $device:expr) => { <$B>::staging(data, $device) };
        }
        native_dispatch!(device, call)
    }
    fn device_count(type_id: u16) -> usize {
        let native = type_id & 255;
        match (type_id & 0x7fff) >> 8 {
1 => <crate::backends::Host as Backend>::device_count(native),
#[cfg(feature = "cuda")]
2 => <crate::backends::Cuda as Backend>::device_count(native),
#[cfg(feature = "cpu")]
3 => <crate::backends::Cpu as Backend>::device_count(native),
#[cfg(feature = "wgpu")]
4 => <crate::backends::Wgpu as Backend>::device_count(native),
#[cfg(feature = "rocm")]
5 => <crate::backends::Rocm as Backend>::device_count(native),
#[cfg(feature = "tch")]
6 => <crate::backends::LibTorch as Backend>::device_count(native),
#[cfg(feature = "remote")]
7 => <crate::backends::Remote as Backend>::device_count(native),
#[cfg(feature = "capture")]
8 => 0,
            _ => 0,
        }
    }
}

impl AutodiffBackend for Dispatch {
    type InnerBackend = Self;
    type Gradients = ruda_autodiff::grads::Gradients;

    #[cfg(feature = "distributed")]
    fn set_distributed_params(tensor: DispatchTensor, param_id: ruda_tensor::distributed::DistributedParamId) -> DispatchTensor {
        crate::distributed::mark(tensor, param_id)
    }

    #[cfg(feature = "distributed")]
    fn distributed_params(tensor: &DispatchTensor) -> Option<ruda_tensor::distributed::DistributedParams> {
        crate::distributed::params(tensor)
    }

    #[cfg(feature = "distributed")]
    fn is_distributed(tensor: &DispatchTensor) -> bool { crate::distributed::params(tensor).is_some() }

    fn backward(tensor: DispatchTensor) -> Self::Gradients {
        match tensor {
DispatchTensor::Host(BackendTensor::AdFloat(tensor)) => tensor.backward_local(),
#[cfg(feature = "cuda")]
DispatchTensor::Cuda(BackendTensor::AdFloat(tensor)) => {
    #[cfg(feature = "distributed")]
    { tensor.backward() }
    #[cfg(not(feature = "distributed"))]
    { tensor.backward_local() }
},
#[cfg(feature = "cpu")]
DispatchTensor::Cpu(BackendTensor::AdFloat(tensor)) => tensor.backward_local(),
#[cfg(feature = "wgpu")]
DispatchTensor::Wgpu(BackendTensor::AdFloat(tensor)) => tensor.backward_local(),
#[cfg(feature = "rocm")]
DispatchTensor::Rocm(BackendTensor::AdFloat(tensor)) => tensor.backward_local(),
#[cfg(feature = "tch")]
DispatchTensor::LibTorch(BackendTensor::AdFloat(tensor)) => {
    #[cfg(feature = "distributed")]
    { tensor.backward() }
    #[cfg(not(feature = "distributed"))]
    { tensor.backward_local() }
},
#[cfg(feature = "remote")]
DispatchTensor::Remote(BackendTensor::AdFloat(tensor)) => {
    #[cfg(feature = "distributed")]
    { tensor.backward() }
    #[cfg(not(feature = "distributed"))]
    { tensor.backward_local() }
},
            _ => panic!("Backward requires an autodiff float tensor"),
        }
    }
    fn grad(tensor: &DispatchTensor, grads: &Self::Gradients) -> Option<DispatchTensor> {
        match tensor {
DispatchTensor::Host(BackendTensor::AdFloat(tensor)) => tensor.grad(grads).map(|value| DispatchTensor::Host(BackendTensor::Float(value))),
#[cfg(feature = "cuda")]
DispatchTensor::Cuda(BackendTensor::AdFloat(tensor)) => tensor.grad(grads).map(|value| DispatchTensor::Cuda(BackendTensor::Float(value))),
#[cfg(feature = "cpu")]
DispatchTensor::Cpu(BackendTensor::AdFloat(tensor)) => tensor.grad(grads).map(|value| DispatchTensor::Cpu(BackendTensor::Float(value))),
#[cfg(feature = "wgpu")]
DispatchTensor::Wgpu(BackendTensor::AdFloat(tensor)) => tensor.grad(grads).map(|value| DispatchTensor::Wgpu(BackendTensor::Float(value))),
#[cfg(feature = "rocm")]
DispatchTensor::Rocm(BackendTensor::AdFloat(tensor)) => tensor.grad(grads).map(|value| DispatchTensor::Rocm(BackendTensor::Float(value))),
#[cfg(feature = "tch")]
DispatchTensor::LibTorch(BackendTensor::AdFloat(tensor)) => tensor.grad(grads).map(|value| DispatchTensor::LibTorch(BackendTensor::Float(value))),
#[cfg(feature = "remote")]
DispatchTensor::Remote(BackendTensor::AdFloat(tensor)) => tensor.grad(grads).map(|value| DispatchTensor::Remote(BackendTensor::Float(value))),
            _ => None,
        }
    }
    fn grad_remove(tensor: &DispatchTensor, grads: &mut Self::Gradients) -> Option<DispatchTensor> {
        match tensor {
DispatchTensor::Host(BackendTensor::AdFloat(tensor)) => tensor.grad_remove(grads).map(|value| DispatchTensor::Host(BackendTensor::Float(value))),
#[cfg(feature = "cuda")]
DispatchTensor::Cuda(BackendTensor::AdFloat(tensor)) => tensor.grad_remove(grads).map(|value| DispatchTensor::Cuda(BackendTensor::Float(value))),
#[cfg(feature = "cpu")]
DispatchTensor::Cpu(BackendTensor::AdFloat(tensor)) => tensor.grad_remove(grads).map(|value| DispatchTensor::Cpu(BackendTensor::Float(value))),
#[cfg(feature = "wgpu")]
DispatchTensor::Wgpu(BackendTensor::AdFloat(tensor)) => tensor.grad_remove(grads).map(|value| DispatchTensor::Wgpu(BackendTensor::Float(value))),
#[cfg(feature = "rocm")]
DispatchTensor::Rocm(BackendTensor::AdFloat(tensor)) => tensor.grad_remove(grads).map(|value| DispatchTensor::Rocm(BackendTensor::Float(value))),
#[cfg(feature = "tch")]
DispatchTensor::LibTorch(BackendTensor::AdFloat(tensor)) => tensor.grad_remove(grads).map(|value| DispatchTensor::LibTorch(BackendTensor::Float(value))),
#[cfg(feature = "remote")]
DispatchTensor::Remote(BackendTensor::AdFloat(tensor)) => tensor.grad_remove(grads).map(|value| DispatchTensor::Remote(BackendTensor::Float(value))),
            _ => None,
        }
    }
    fn q_grad(tensor: &DispatchTensor, grads: &Self::Gradients) -> Option<DispatchTensor> {
        match tensor {
DispatchTensor::Host(BackendTensor::AdQuantized(tensor)) => tensor.grad(grads).map(|value| DispatchTensor::Host(BackendTensor::Float(value))),
#[cfg(feature = "cuda")]
DispatchTensor::Cuda(BackendTensor::AdQuantized(tensor)) => tensor.grad(grads).map(|value| DispatchTensor::Cuda(BackendTensor::Float(value))),
#[cfg(feature = "cpu")]
DispatchTensor::Cpu(BackendTensor::AdQuantized(tensor)) => tensor.grad(grads).map(|value| DispatchTensor::Cpu(BackendTensor::Float(value))),
#[cfg(feature = "wgpu")]
DispatchTensor::Wgpu(BackendTensor::AdQuantized(tensor)) => tensor.grad(grads).map(|value| DispatchTensor::Wgpu(BackendTensor::Float(value))),
#[cfg(feature = "rocm")]
DispatchTensor::Rocm(BackendTensor::AdQuantized(tensor)) => tensor.grad(grads).map(|value| DispatchTensor::Rocm(BackendTensor::Float(value))),
#[cfg(feature = "tch")]
DispatchTensor::LibTorch(BackendTensor::AdQuantized(tensor)) => tensor.grad(grads).map(|value| DispatchTensor::LibTorch(BackendTensor::Float(value))),
#[cfg(feature = "remote")]
DispatchTensor::Remote(BackendTensor::AdQuantized(tensor)) => tensor.grad(grads).map(|value| DispatchTensor::Remote(BackendTensor::Float(value))),
            _ => None,
        }
    }
    fn q_grad_remove(tensor: &DispatchTensor, grads: &mut Self::Gradients) -> Option<DispatchTensor> {
        match tensor {
DispatchTensor::Host(BackendTensor::AdQuantized(tensor)) => tensor.grad_remove(grads).map(|value| DispatchTensor::Host(BackendTensor::Float(value))),
#[cfg(feature = "cuda")]
DispatchTensor::Cuda(BackendTensor::AdQuantized(tensor)) => tensor.grad_remove(grads).map(|value| DispatchTensor::Cuda(BackendTensor::Float(value))),
#[cfg(feature = "cpu")]
DispatchTensor::Cpu(BackendTensor::AdQuantized(tensor)) => tensor.grad_remove(grads).map(|value| DispatchTensor::Cpu(BackendTensor::Float(value))),
#[cfg(feature = "wgpu")]
DispatchTensor::Wgpu(BackendTensor::AdQuantized(tensor)) => tensor.grad_remove(grads).map(|value| DispatchTensor::Wgpu(BackendTensor::Float(value))),
#[cfg(feature = "rocm")]
DispatchTensor::Rocm(BackendTensor::AdQuantized(tensor)) => tensor.grad_remove(grads).map(|value| DispatchTensor::Rocm(BackendTensor::Float(value))),
#[cfg(feature = "tch")]
DispatchTensor::LibTorch(BackendTensor::AdQuantized(tensor)) => tensor.grad_remove(grads).map(|value| DispatchTensor::LibTorch(BackendTensor::Float(value))),
#[cfg(feature = "remote")]
DispatchTensor::Remote(BackendTensor::AdQuantized(tensor)) => tensor.grad_remove(grads).map(|value| DispatchTensor::Remote(BackendTensor::Float(value))),
            _ => None,
        }
    }
    fn grad_replace(tensor: &DispatchTensor, grads: &mut Self::Gradients, grad: DispatchTensor) {
        match (tensor, grad) {
(DispatchTensor::Host(BackendTensor::AdFloat(tensor)), DispatchTensor::Host(value)) =>
    tensor.grad_replace(grads, value.into_native_float(AutodiffContext::Disabled)),
#[cfg(feature = "cuda")]
(DispatchTensor::Cuda(BackendTensor::AdFloat(tensor)), DispatchTensor::Cuda(value)) =>
    tensor.grad_replace(grads, value.into_native_float(AutodiffContext::Disabled)),
#[cfg(feature = "cpu")]
(DispatchTensor::Cpu(BackendTensor::AdFloat(tensor)), DispatchTensor::Cpu(value)) =>
    tensor.grad_replace(grads, value.into_native_float(AutodiffContext::Disabled)),
#[cfg(feature = "wgpu")]
(DispatchTensor::Wgpu(BackendTensor::AdFloat(tensor)), DispatchTensor::Wgpu(value)) =>
    tensor.grad_replace(grads, value.into_native_float(AutodiffContext::Disabled)),
#[cfg(feature = "rocm")]
(DispatchTensor::Rocm(BackendTensor::AdFloat(tensor)), DispatchTensor::Rocm(value)) =>
    tensor.grad_replace(grads, value.into_native_float(AutodiffContext::Disabled)),
#[cfg(feature = "tch")]
(DispatchTensor::LibTorch(BackendTensor::AdFloat(tensor)), DispatchTensor::LibTorch(value)) =>
    tensor.grad_replace(grads, value.into_native_float(AutodiffContext::Disabled)),
#[cfg(feature = "remote")]
(DispatchTensor::Remote(BackendTensor::AdFloat(tensor)), DispatchTensor::Remote(value)) =>
    tensor.grad_replace(grads, value.into_native_float(AutodiffContext::Disabled)),
            _ => panic!("Gradient and autodiff tensor must use the same backend"),
        }
    }
    fn q_grad_replace(tensor: &DispatchTensor, grads: &mut Self::Gradients, grad: DispatchTensor) {
        match (tensor, grad) {
(DispatchTensor::Host(BackendTensor::AdQuantized(tensor)), DispatchTensor::Host(value)) =>
    tensor.grad_replace(grads, value.into_native_float(AutodiffContext::Disabled)),
#[cfg(feature = "cuda")]
(DispatchTensor::Cuda(BackendTensor::AdQuantized(tensor)), DispatchTensor::Cuda(value)) =>
    tensor.grad_replace(grads, value.into_native_float(AutodiffContext::Disabled)),
#[cfg(feature = "cpu")]
(DispatchTensor::Cpu(BackendTensor::AdQuantized(tensor)), DispatchTensor::Cpu(value)) =>
    tensor.grad_replace(grads, value.into_native_float(AutodiffContext::Disabled)),
#[cfg(feature = "wgpu")]
(DispatchTensor::Wgpu(BackendTensor::AdQuantized(tensor)), DispatchTensor::Wgpu(value)) =>
    tensor.grad_replace(grads, value.into_native_float(AutodiffContext::Disabled)),
#[cfg(feature = "rocm")]
(DispatchTensor::Rocm(BackendTensor::AdQuantized(tensor)), DispatchTensor::Rocm(value)) =>
    tensor.grad_replace(grads, value.into_native_float(AutodiffContext::Disabled)),
#[cfg(feature = "tch")]
(DispatchTensor::LibTorch(BackendTensor::AdQuantized(tensor)), DispatchTensor::LibTorch(value)) =>
    tensor.grad_replace(grads, value.into_native_float(AutodiffContext::Disabled)),
#[cfg(feature = "remote")]
(DispatchTensor::Remote(BackendTensor::AdQuantized(tensor)), DispatchTensor::Remote(value)) =>
    tensor.grad_replace(grads, value.into_native_float(AutodiffContext::Disabled)),
            _ => panic!("Gradient and autodiff tensor must use the same backend"),
        }
    }
    fn inner(tensor: DispatchTensor) -> DispatchTensor { tensor.with_context(AutodiffContext::Disabled) }
    fn from_inner(tensor: DispatchTensor) -> DispatchTensor {
        let context = match context::current(&tensor.device()) {
            AutodiffContext::Disabled => AutodiffContext::Enabled(GradientCheckpointingStrategy::Disabled),
            context => context,
        };
        tensor.with_context(context)
    }
    fn int_inner(tensor: DispatchTensor) -> DispatchTensor { tensor.with_context(AutodiffContext::Disabled) }
    fn int_from_inner(tensor: DispatchTensor) -> DispatchTensor {
        let context = match context::current(&tensor.device()) {
            AutodiffContext::Disabled => AutodiffContext::Enabled(GradientCheckpointingStrategy::Disabled),
            context => context,
        };
        tensor.with_context(context)
    }
    fn bool_inner(tensor: DispatchTensor) -> DispatchTensor { tensor.with_context(AutodiffContext::Disabled) }
    fn bool_from_inner(tensor: DispatchTensor) -> DispatchTensor {
        let context = match context::current(&tensor.device()) {
            AutodiffContext::Disabled => AutodiffContext::Enabled(GradientCheckpointingStrategy::Disabled),
            context => context,
        };
        tensor.with_context(context)
    }
    fn q_inner(tensor: DispatchTensor) -> DispatchTensor { tensor.with_context(AutodiffContext::Disabled) }
    fn q_from_inner(tensor: DispatchTensor) -> DispatchTensor {
        let context = match context::current(&tensor.device()) {
            AutodiffContext::Disabled => AutodiffContext::Enabled(GradientCheckpointingStrategy::Disabled),
            context => context,
        };
        tensor.with_context(context)
    }
}
