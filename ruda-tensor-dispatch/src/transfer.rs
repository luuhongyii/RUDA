use ruda_tensor::{Backend, DeviceOps, AutodiffContext, GradientCheckpointingStrategy};
use ruda_tensor::ops::*;
use ruda_tensor::tensor::FloatTensor;
use ruda_autodiff::{Autodiff, AutodiffValue, checkpoint::strategy::{CheckpointStrategy, NoCheckpointing, BalancedCheckpointing}};
use crate::{BackendTensor, DispatchTensor, DispatchDevice, context::Selection};

#[derive(Debug)]
struct HostTransfer;

fn float_storage<Src: Backend, Dst: Backend>(tensor: FloatTensor<Src>, device: &Dst::Device) -> FloatTensor<Dst> {
    let data = ruda_core::future::block_on(Src::float_into_data(tensor))
        .expect("Cross-backend float readback failed");
    Dst::float_from_data(data, device)
}

impl<Src: Backend, Dst: Backend> ruda_autodiff::ops::DifferentiableTransfer<Src, Dst> for HostTransfer {
    fn forward(tensor: FloatTensor<Src>, device: &Dst::Device) -> FloatTensor<Dst> { float_storage::<Src, Dst>(tensor, device) }
    fn backward(tensor: FloatTensor<Dst>, device: &Src::Device) -> FloatTensor<Src> { float_storage::<Dst, Src>(tensor, device) }
}

fn quantized_storage<Src: Backend, Dst: Backend>(tensor: Src::QuantizedTensorPrimitive, device: &Dst::Device) -> Dst::QuantizedTensorPrimitive {
    let data = ruda_core::future::block_on(Src::q_into_data(tensor))
        .expect("Cross-backend quantized readback failed");
    Dst::q_from_data(data, device)
}

fn float_ad<Src: Backend, Dst: Backend, C: CheckpointStrategy>(tensor: BackendTensor<Src>, device: &Dst::Device) -> BackendTensor<Dst> {
    let context = tensor.context();
    BackendTensor::AdFloat(Autodiff::<Src, C>::to_backend::<Dst, HostTransfer>(tensor.into_ad_float(context), device))
}

fn quantized_ad<Src: Backend, Dst: Backend, C: CheckpointStrategy>(tensor: BackendTensor<Src>, device: &Dst::Device) -> BackendTensor<Dst> {
    let context = tensor.context();
    BackendTensor::AdQuantized(Autodiff::<Src, C>::quantized_to_backend::<Dst, HostTransfer>(
        tensor.into_ad_quantized(context), device, quantized_storage::<Src, Dst>))
}

fn float_cross<Src: Backend, Dst: Backend>(tensor: BackendTensor<Src>, device: &Dst::Device) -> BackendTensor<Dst> {
    match tensor.context() {
        AutodiffContext::Disabled => BackendTensor::Float(float_storage::<Src, Dst>(tensor.into_native_float(AutodiffContext::Disabled), device)),
        AutodiffContext::Enabled(GradientCheckpointingStrategy::Disabled) => float_ad::<Src, Dst, NoCheckpointing>(tensor, device),
        AutodiffContext::Enabled(GradientCheckpointingStrategy::Balanced) => float_ad::<Src, Dst, BalancedCheckpointing>(tensor, device),
        AutodiffContext::Enabled(GradientCheckpointingStrategy::Custom(_)) => panic!("Dispatch accepts Disabled or Balanced checkpointing"),
    }
}

fn q_cross<Src: Backend, Dst: Backend>(tensor: BackendTensor<Src>, device: &Dst::Device) -> BackendTensor<Dst> {
    match tensor.context() {
        AutodiffContext::Disabled => BackendTensor::Quantized(quantized_storage::<Src, Dst>(tensor.into_native_quantized(AutodiffContext::Disabled), device)),
        AutodiffContext::Enabled(GradientCheckpointingStrategy::Disabled) => quantized_ad::<Src, Dst, NoCheckpointing>(tensor, device),
        AutodiffContext::Enabled(GradientCheckpointingStrategy::Balanced) => quantized_ad::<Src, Dst, BalancedCheckpointing>(tensor, device),
        AutodiffContext::Enabled(GradientCheckpointingStrategy::Custom(_)) => panic!("Dispatch accepts Disabled or Balanced checkpointing"),
    }
}

fn int_cross<Src: Backend, Dst: Backend>(tensor: BackendTensor<Src>, device: &Dst::Device) -> BackendTensor<Dst> {
    let context = tensor.context();
    let primitive = tensor.into_native_int(context);
    let data = ruda_core::future::block_on(Src::int_into_data(primitive))
        .expect("Cross-backend int readback failed");
    let output = Dst::int_from_data(data, device);
    BackendTensor::Int(output).with_context(context)
}

fn bool_cross<Src: Backend, Dst: Backend>(tensor: BackendTensor<Src>, device: &Dst::Device) -> BackendTensor<Dst> {
    let context = tensor.context();
    let primitive = tensor.into_native_bool(context);
    let data = ruda_core::future::block_on(Src::bool_into_data(primitive))
        .expect("Cross-backend bool readback failed");
    let output = Dst::bool_from_data(data, device);
    BackendTensor::Bool(output).with_context(context)
}

pub(crate) fn float_to_device(tensor: DispatchTensor, device: &DispatchDevice) -> DispatchTensor {
    #[cfg(feature = "capture")]
    if matches!(device.inner(), DispatchDevice::Capture(_)) {
        assert_eq!(tensor.context(), AutodiffContext::Disabled, "Disable autodiff before moving a tensor to capture");
        assert_eq!(device.context(), AutodiffContext::Disabled, "Capture does not support autodiff");
    }
    let same_backend = match (&tensor, device.inner()) {
(DispatchTensor::Host(_), DispatchDevice::Host(_)) => true,
#[cfg(feature = "cuda")]
(DispatchTensor::Cuda(_), DispatchDevice::Cuda(_)) => true,
#[cfg(feature = "cpu")]
(DispatchTensor::Cpu(_), DispatchDevice::Cpu(_)) => true,
#[cfg(feature = "wgpu")]
(DispatchTensor::Wgpu(_), DispatchDevice::Wgpu(_)) => true,
#[cfg(feature = "rocm")]
(DispatchTensor::Rocm(_), DispatchDevice::Rocm(_)) => true,
#[cfg(feature = "tch")]
(DispatchTensor::LibTorch(_), DispatchDevice::LibTorch(_)) => true,
#[cfg(feature = "remote")]
(DispatchTensor::Remote(_), DispatchDevice::Remote(_)) => true,
#[cfg(feature = "capture")]
(DispatchTensor::Capture(_), DispatchDevice::Capture(_)) => true,
        _ => false,
    };
    if same_backend {
        let selection = Selection { device: device.inner().clone(), context: tensor.context() };
        macro_rules! same {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let output = <$B>::float_to_device(unpack!($mode, $variant, Float, tensor, $context), $device);
                pack!($mode, $variant, Float, output)
            }};
        }
        return dispatch!(selection, same);
    }
    macro_rules! destination {
        ($Dst:ty, $variant:ident, $device:expr) => {{
            let tensor = match tensor {
DispatchTensor::Host(tensor) => float_cross::<crate::backends::Host, $Dst>(tensor, $device),
#[cfg(feature = "cuda")]
DispatchTensor::Cuda(tensor) => float_cross::<crate::backends::Cuda, $Dst>(tensor, $device),
#[cfg(feature = "cpu")]
DispatchTensor::Cpu(tensor) => float_cross::<crate::backends::Cpu, $Dst>(tensor, $device),
#[cfg(feature = "wgpu")]
DispatchTensor::Wgpu(tensor) => float_cross::<crate::backends::Wgpu, $Dst>(tensor, $device),
#[cfg(feature = "rocm")]
DispatchTensor::Rocm(tensor) => float_cross::<crate::backends::Rocm, $Dst>(tensor, $device),
#[cfg(feature = "tch")]
DispatchTensor::LibTorch(tensor) => float_cross::<crate::backends::LibTorch, $Dst>(tensor, $device),
#[cfg(feature = "remote")]
DispatchTensor::Remote(tensor) => float_cross::<crate::backends::Remote, $Dst>(tensor, $device),
#[cfg(feature = "capture")]
DispatchTensor::Capture(_) => panic!("Cannot move a tensor from capture to an execution backend"),
            };
            DispatchTensor::$variant(tensor)
        }};
    }
    native_dispatch!(device, destination)
}

pub(crate) fn int_to_device(tensor: DispatchTensor, device: &DispatchDevice) -> DispatchTensor {
    #[cfg(feature = "capture")]
    if matches!(device.inner(), DispatchDevice::Capture(_)) {
        assert_eq!(tensor.context(), AutodiffContext::Disabled, "Disable autodiff before moving a tensor to capture");
        assert_eq!(device.context(), AutodiffContext::Disabled, "Capture does not support autodiff");
    }
    let same_backend = match (&tensor, device.inner()) {
(DispatchTensor::Host(_), DispatchDevice::Host(_)) => true,
#[cfg(feature = "cuda")]
(DispatchTensor::Cuda(_), DispatchDevice::Cuda(_)) => true,
#[cfg(feature = "cpu")]
(DispatchTensor::Cpu(_), DispatchDevice::Cpu(_)) => true,
#[cfg(feature = "wgpu")]
(DispatchTensor::Wgpu(_), DispatchDevice::Wgpu(_)) => true,
#[cfg(feature = "rocm")]
(DispatchTensor::Rocm(_), DispatchDevice::Rocm(_)) => true,
#[cfg(feature = "tch")]
(DispatchTensor::LibTorch(_), DispatchDevice::LibTorch(_)) => true,
#[cfg(feature = "remote")]
(DispatchTensor::Remote(_), DispatchDevice::Remote(_)) => true,
#[cfg(feature = "capture")]
(DispatchTensor::Capture(_), DispatchDevice::Capture(_)) => true,
        _ => false,
    };
    if same_backend {
        let selection = Selection { device: device.inner().clone(), context: tensor.context() };
        macro_rules! same {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let output = <$B>::int_to_device(unpack!($mode, $variant, Int, tensor, $context), $device);
                pack!($mode, $variant, Int, output)
            }};
        }
        return dispatch!(selection, same);
    }
    macro_rules! destination {
        ($Dst:ty, $variant:ident, $device:expr) => {{
            let tensor = match tensor {
DispatchTensor::Host(tensor) => int_cross::<crate::backends::Host, $Dst>(tensor, $device),
#[cfg(feature = "cuda")]
DispatchTensor::Cuda(tensor) => int_cross::<crate::backends::Cuda, $Dst>(tensor, $device),
#[cfg(feature = "cpu")]
DispatchTensor::Cpu(tensor) => int_cross::<crate::backends::Cpu, $Dst>(tensor, $device),
#[cfg(feature = "wgpu")]
DispatchTensor::Wgpu(tensor) => int_cross::<crate::backends::Wgpu, $Dst>(tensor, $device),
#[cfg(feature = "rocm")]
DispatchTensor::Rocm(tensor) => int_cross::<crate::backends::Rocm, $Dst>(tensor, $device),
#[cfg(feature = "tch")]
DispatchTensor::LibTorch(tensor) => int_cross::<crate::backends::LibTorch, $Dst>(tensor, $device),
#[cfg(feature = "remote")]
DispatchTensor::Remote(tensor) => int_cross::<crate::backends::Remote, $Dst>(tensor, $device),
#[cfg(feature = "capture")]
DispatchTensor::Capture(_) => panic!("Cannot move a tensor from capture to an execution backend"),
            };
            DispatchTensor::$variant(tensor)
        }};
    }
    native_dispatch!(device, destination)
}

pub(crate) fn bool_to_device(tensor: DispatchTensor, device: &DispatchDevice) -> DispatchTensor {
    #[cfg(feature = "capture")]
    if matches!(device.inner(), DispatchDevice::Capture(_)) {
        assert_eq!(tensor.context(), AutodiffContext::Disabled, "Disable autodiff before moving a tensor to capture");
        assert_eq!(device.context(), AutodiffContext::Disabled, "Capture does not support autodiff");
    }
    let same_backend = match (&tensor, device.inner()) {
(DispatchTensor::Host(_), DispatchDevice::Host(_)) => true,
#[cfg(feature = "cuda")]
(DispatchTensor::Cuda(_), DispatchDevice::Cuda(_)) => true,
#[cfg(feature = "cpu")]
(DispatchTensor::Cpu(_), DispatchDevice::Cpu(_)) => true,
#[cfg(feature = "wgpu")]
(DispatchTensor::Wgpu(_), DispatchDevice::Wgpu(_)) => true,
#[cfg(feature = "rocm")]
(DispatchTensor::Rocm(_), DispatchDevice::Rocm(_)) => true,
#[cfg(feature = "tch")]
(DispatchTensor::LibTorch(_), DispatchDevice::LibTorch(_)) => true,
#[cfg(feature = "remote")]
(DispatchTensor::Remote(_), DispatchDevice::Remote(_)) => true,
#[cfg(feature = "capture")]
(DispatchTensor::Capture(_), DispatchDevice::Capture(_)) => true,
        _ => false,
    };
    if same_backend {
        let selection = Selection { device: device.inner().clone(), context: tensor.context() };
        macro_rules! same {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let output = <$B>::bool_to_device(unpack!($mode, $variant, Bool, tensor, $context), $device);
                pack!($mode, $variant, Bool, output)
            }};
        }
        return dispatch!(selection, same);
    }
    macro_rules! destination {
        ($Dst:ty, $variant:ident, $device:expr) => {{
            let tensor = match tensor {
DispatchTensor::Host(tensor) => bool_cross::<crate::backends::Host, $Dst>(tensor, $device),
#[cfg(feature = "cuda")]
DispatchTensor::Cuda(tensor) => bool_cross::<crate::backends::Cuda, $Dst>(tensor, $device),
#[cfg(feature = "cpu")]
DispatchTensor::Cpu(tensor) => bool_cross::<crate::backends::Cpu, $Dst>(tensor, $device),
#[cfg(feature = "wgpu")]
DispatchTensor::Wgpu(tensor) => bool_cross::<crate::backends::Wgpu, $Dst>(tensor, $device),
#[cfg(feature = "rocm")]
DispatchTensor::Rocm(tensor) => bool_cross::<crate::backends::Rocm, $Dst>(tensor, $device),
#[cfg(feature = "tch")]
DispatchTensor::LibTorch(tensor) => bool_cross::<crate::backends::LibTorch, $Dst>(tensor, $device),
#[cfg(feature = "remote")]
DispatchTensor::Remote(tensor) => bool_cross::<crate::backends::Remote, $Dst>(tensor, $device),
#[cfg(feature = "capture")]
DispatchTensor::Capture(_) => panic!("Cannot move a tensor from capture to an execution backend"),
            };
            DispatchTensor::$variant(tensor)
        }};
    }
    native_dispatch!(device, destination)
}

pub(crate) fn q_to_device(tensor: DispatchTensor, device: &DispatchDevice) -> DispatchTensor {
    #[cfg(feature = "capture")]
    if matches!(device.inner(), DispatchDevice::Capture(_)) {
        assert_eq!(tensor.context(), AutodiffContext::Disabled, "Disable autodiff before moving a tensor to capture");
        assert_eq!(device.context(), AutodiffContext::Disabled, "Capture does not support autodiff");
    }
    let same_backend = match (&tensor, device.inner()) {
(DispatchTensor::Host(_), DispatchDevice::Host(_)) => true,
#[cfg(feature = "cuda")]
(DispatchTensor::Cuda(_), DispatchDevice::Cuda(_)) => true,
#[cfg(feature = "cpu")]
(DispatchTensor::Cpu(_), DispatchDevice::Cpu(_)) => true,
#[cfg(feature = "wgpu")]
(DispatchTensor::Wgpu(_), DispatchDevice::Wgpu(_)) => true,
#[cfg(feature = "rocm")]
(DispatchTensor::Rocm(_), DispatchDevice::Rocm(_)) => true,
#[cfg(feature = "tch")]
(DispatchTensor::LibTorch(_), DispatchDevice::LibTorch(_)) => true,
#[cfg(feature = "remote")]
(DispatchTensor::Remote(_), DispatchDevice::Remote(_)) => true,
#[cfg(feature = "capture")]
(DispatchTensor::Capture(_), DispatchDevice::Capture(_)) => true,
        _ => false,
    };
    if same_backend {
        let selection = Selection { device: device.inner().clone(), context: tensor.context() };
        macro_rules! same {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let output = <$B>::q_to_device(unpack!($mode, $variant, Quantized, tensor, $context), $device);
                pack!($mode, $variant, Quantized, output)
            }};
        }
        return dispatch!(selection, same);
    }
    macro_rules! destination {
        ($Dst:ty, $variant:ident, $device:expr) => {{
            let tensor = match tensor {
DispatchTensor::Host(tensor) => q_cross::<crate::backends::Host, $Dst>(tensor, $device),
#[cfg(feature = "cuda")]
DispatchTensor::Cuda(tensor) => q_cross::<crate::backends::Cuda, $Dst>(tensor, $device),
#[cfg(feature = "cpu")]
DispatchTensor::Cpu(tensor) => q_cross::<crate::backends::Cpu, $Dst>(tensor, $device),
#[cfg(feature = "wgpu")]
DispatchTensor::Wgpu(tensor) => q_cross::<crate::backends::Wgpu, $Dst>(tensor, $device),
#[cfg(feature = "rocm")]
DispatchTensor::Rocm(tensor) => q_cross::<crate::backends::Rocm, $Dst>(tensor, $device),
#[cfg(feature = "tch")]
DispatchTensor::LibTorch(tensor) => q_cross::<crate::backends::LibTorch, $Dst>(tensor, $device),
#[cfg(feature = "remote")]
DispatchTensor::Remote(tensor) => q_cross::<crate::backends::Remote, $Dst>(tensor, $device),
#[cfg(feature = "capture")]
DispatchTensor::Capture(_) => panic!("Cannot move a tensor from capture to an execution backend"),
            };
            DispatchTensor::$variant(tensor)
        }};
    }
    native_dispatch!(device, destination)
}
