use ruda_autodiff::{Autodiff, AutodiffValue};
use ruda_tensor::{Backend, AutodiffContext, TensorMetadata, QTensorPrimitive, DeviceOps, DType, Shape};
use ruda_tensor::ops::*;
use ruda_tensor::tensor::{FloatTensor, IntTensor, BoolTensor, QuantizedTensor};
use ruda_core::tensor::{QuantScheme, QuantAcc, QuantPropagation};
use crate::{DispatchDevice, backends};

#[derive(Clone, Debug)]
pub enum BackendTensor<B: Backend> {
Float(FloatTensor<B>),
AdFloat(FloatTensor<Autodiff<B>>),
Int(IntTensor<B>),
AdInt(IntTensor<Autodiff<B>>),
Bool(BoolTensor<B>),
AdBool(BoolTensor<Autodiff<B>>),
Quantized(QuantizedTensor<B>),
AdQuantized(QuantizedTensor<Autodiff<B>>),
}

impl<B: Backend> BackendTensor<B> {
pub(crate) fn into_native_float(self, _: AutodiffContext) -> FloatTensor<B> {
    match self {
        Self::Float(tensor) => tensor,
        Self::AdFloat(tensor) => tensor.primitive,
        _ => panic!("Expected a float tensor"),
    }
}
pub(crate) fn into_ad_float(self, context: AutodiffContext) -> FloatTensor<Autodiff<B>> {
    let tensor = match self {
        Self::AdFloat(tensor) => tensor,
        Self::Float(tensor) => <FloatTensor<Autodiff<B>>>::new(tensor),
        _ => panic!("Expected a float tensor"),
    };
    Autodiff::<B>::float_with_autodiff_context(tensor, context)
}
pub(crate) fn into_native_int(self, _: AutodiffContext) -> IntTensor<B> {
    match self {
        Self::Int(tensor) => tensor,
        Self::AdInt(tensor) => tensor.primitive,
        _ => panic!("Expected a int tensor"),
    }
}
pub(crate) fn into_ad_int(self, context: AutodiffContext) -> IntTensor<Autodiff<B>> {
    let tensor = match self {
        Self::AdInt(tensor) => tensor,
        Self::Int(tensor) => AutodiffValue::with_context(tensor, context),
        _ => panic!("Expected a int tensor"),
    };
    Autodiff::<B>::int_with_autodiff_context(tensor, context)
}
pub(crate) fn into_native_bool(self, _: AutodiffContext) -> BoolTensor<B> {
    match self {
        Self::Bool(tensor) => tensor,
        Self::AdBool(tensor) => tensor.primitive,
        _ => panic!("Expected a bool tensor"),
    }
}
pub(crate) fn into_ad_bool(self, context: AutodiffContext) -> BoolTensor<Autodiff<B>> {
    let tensor = match self {
        Self::AdBool(tensor) => tensor,
        Self::Bool(tensor) => AutodiffValue::with_context(tensor, context),
        _ => panic!("Expected a bool tensor"),
    };
    Autodiff::<B>::bool_with_autodiff_context(tensor, context)
}
pub(crate) fn into_native_quantized(self, _: AutodiffContext) -> QuantizedTensor<B> {
    match self {
        Self::Quantized(tensor) => tensor,
        Self::AdQuantized(tensor) => tensor.primitive,
        _ => panic!("Expected a quantized tensor"),
    }
}
pub(crate) fn into_ad_quantized(self, context: AutodiffContext) -> QuantizedTensor<Autodiff<B>> {
    let tensor = match self {
        Self::AdQuantized(tensor) => tensor,
        Self::Quantized(tensor) => ruda_autodiff::AutodiffQTensor::untracked(tensor),
        _ => panic!("Expected a quantized tensor"),
    };
    Autodiff::<B>::q_with_autodiff_context(tensor, context)
}

    pub fn context(&self) -> AutodiffContext {
        match self {
Self::AdFloat(tensor) => Autodiff::<B>::float_autodiff_context(tensor),
Self::AdInt(tensor) => Autodiff::<B>::int_autodiff_context(tensor),
Self::AdBool(tensor) => Autodiff::<B>::bool_autodiff_context(tensor),
Self::AdQuantized(tensor) => Autodiff::<B>::q_autodiff_context(tensor),
            _ => AutodiffContext::Disabled,
        }
    }

    pub fn with_context(self, context: AutodiffContext) -> Self {
        match context {
            AutodiffContext::Disabled => match self {
value @ (Self::Float(_) | Self::AdFloat(_)) => Self::Float(value.into_native_float(context)),
value @ (Self::Int(_) | Self::AdInt(_)) => Self::Int(value.into_native_int(context)),
value @ (Self::Bool(_) | Self::AdBool(_)) => Self::Bool(value.into_native_bool(context)),
value @ (Self::Quantized(_) | Self::AdQuantized(_)) => Self::Quantized(value.into_native_quantized(context)),
            },
            AutodiffContext::Enabled(_) => match self {
value @ (Self::Float(_) | Self::AdFloat(_)) => Self::AdFloat(value.into_ad_float(context)),
value @ (Self::Int(_) | Self::AdInt(_)) => Self::AdInt(value.into_ad_int(context)),
value @ (Self::Bool(_) | Self::AdBool(_)) => Self::AdBool(value.into_ad_bool(context)),
value @ (Self::Quantized(_) | Self::AdQuantized(_)) => Self::AdQuantized(value.into_ad_quantized(context)),
            },
        }
    }

    pub fn device(&self) -> B::Device {
        match self {
Self::Float(tensor) => B::float_device(tensor),
Self::AdFloat(tensor) => B::float_device(&tensor.primitive),
Self::Int(tensor) => B::int_device(tensor),
Self::AdInt(tensor) => B::int_device(&tensor.primitive),
Self::Bool(tensor) => B::bool_device(tensor),
Self::AdBool(tensor) => B::bool_device(&tensor.primitive),
Self::Quantized(tensor) => B::q_device(tensor),
Self::AdQuantized(tensor) => B::q_device(&tensor.primitive),
        }
    }
}

impl<B: Backend> TensorMetadata for BackendTensor<B> {
fn dtype(&self) -> DType {
    match self {
Self::Float(tensor) => tensor.dtype(),
Self::AdFloat(tensor) => tensor.dtype(),
Self::Int(tensor) => tensor.dtype(),
Self::AdInt(tensor) => tensor.dtype(),
Self::Bool(tensor) => tensor.dtype(),
Self::AdBool(tensor) => tensor.dtype(),
Self::Quantized(tensor) => tensor.dtype(),
Self::AdQuantized(tensor) => tensor.dtype(),
    }
}
fn shape(&self) -> Shape {
    match self {
Self::Float(tensor) => tensor.shape(),
Self::AdFloat(tensor) => tensor.shape(),
Self::Int(tensor) => tensor.shape(),
Self::AdInt(tensor) => tensor.shape(),
Self::Bool(tensor) => tensor.shape(),
Self::AdBool(tensor) => tensor.shape(),
Self::Quantized(tensor) => tensor.shape(),
Self::AdQuantized(tensor) => tensor.shape(),
    }
}
fn rank(&self) -> usize {
    match self {
Self::Float(tensor) => tensor.rank(),
Self::AdFloat(tensor) => tensor.rank(),
Self::Int(tensor) => tensor.rank(),
Self::AdInt(tensor) => tensor.rank(),
Self::Bool(tensor) => tensor.rank(),
Self::AdBool(tensor) => tensor.rank(),
Self::Quantized(tensor) => tensor.rank(),
Self::AdQuantized(tensor) => tensor.rank(),
    }
}
fn can_mut(&self) -> bool {
    match self {
Self::Float(tensor) => tensor.can_mut(),
Self::AdFloat(tensor) => tensor.can_mut(),
Self::Int(tensor) => tensor.can_mut(),
Self::AdInt(tensor) => tensor.can_mut(),
Self::Bool(tensor) => tensor.can_mut(),
Self::AdBool(tensor) => tensor.can_mut(),
Self::Quantized(tensor) => tensor.can_mut(),
Self::AdQuantized(tensor) => tensor.can_mut(),
    }
}
}

impl<B: Backend> QTensorPrimitive for BackendTensor<B> {
fn scheme(&self) -> &QuantScheme {
    match self {
        Self::Quantized(tensor) => tensor.scheme(),
        Self::AdQuantized(tensor) => tensor.primitive.scheme(),
        _ => panic!("Expected a quantized tensor"),
    }
}
fn acc_precision(&self) -> QuantAcc {
    match self {
        Self::Quantized(tensor) => tensor.acc_precision(),
        Self::AdQuantized(tensor) => tensor.primitive.acc_precision(),
        _ => panic!("Expected a quantized tensor"),
    }
}
fn propagation(&self) -> QuantPropagation {
    match self {
        Self::Quantized(tensor) => tensor.propagation(),
        Self::AdQuantized(tensor) => tensor.primitive.propagation(),
        _ => panic!("Expected a quantized tensor"),
    }
}
}

#[derive(Clone, Debug)]
pub enum DispatchTensor {
Host(BackendTensor<backends::Host>),
#[cfg(feature = "cuda")]
Cuda(BackendTensor<backends::Cuda>),
#[cfg(feature = "cpu")]
Cpu(BackendTensor<backends::Cpu>),
#[cfg(feature = "wgpu")]
Wgpu(BackendTensor<backends::Wgpu>),
#[cfg(feature = "rocm")]
Rocm(BackendTensor<backends::Rocm>),
#[cfg(feature = "tch")]
LibTorch(BackendTensor<backends::LibTorch>),
#[cfg(feature = "remote")]
Remote(BackendTensor<backends::Remote>),
#[cfg(feature = "capture")]
Capture(BackendTensor<backends::Capture>),
}

impl DispatchTensor {
    /// Identifier to declare a captured tensor as a graph boundary.
    #[cfg(feature = "capture")]
    pub fn capture_tensor_id(&self) -> Option<ruda_tensor_capture::TensorId> {
        match self {
            Self::Capture(BackendTensor::Float(tensor) | BackendTensor::Int(tensor)
                | BackendTensor::Bool(tensor) | BackendTensor::Quantized(tensor)) => Some(tensor.id()),
            _ => None,
        }
    }

    pub fn context(&self) -> AutodiffContext {
        match self {
Self::Host(tensor) => tensor.context(),
#[cfg(feature = "cuda")]
Self::Cuda(tensor) => tensor.context(),
#[cfg(feature = "cpu")]
Self::Cpu(tensor) => tensor.context(),
#[cfg(feature = "wgpu")]
Self::Wgpu(tensor) => tensor.context(),
#[cfg(feature = "rocm")]
Self::Rocm(tensor) => tensor.context(),
#[cfg(feature = "tch")]
Self::LibTorch(tensor) => tensor.context(),
#[cfg(feature = "remote")]
Self::Remote(tensor) => tensor.context(),
#[cfg(feature = "capture")]
Self::Capture(tensor) => tensor.context(),
        }
    }

    pub fn with_context(self, context: AutodiffContext) -> Self {
        match self {
Self::Host(tensor) => Self::Host(tensor.with_context(context)),
#[cfg(feature = "cuda")]
Self::Cuda(tensor) => Self::Cuda(tensor.with_context(context)),
#[cfg(feature = "cpu")]
Self::Cpu(tensor) => Self::Cpu(tensor.with_context(context)),
#[cfg(feature = "wgpu")]
Self::Wgpu(tensor) => Self::Wgpu(tensor.with_context(context)),
#[cfg(feature = "rocm")]
Self::Rocm(tensor) => Self::Rocm(tensor.with_context(context)),
#[cfg(feature = "tch")]
Self::LibTorch(tensor) => Self::LibTorch(tensor.with_context(context)),
#[cfg(feature = "remote")]
Self::Remote(tensor) => Self::Remote(tensor.with_context(context)),
#[cfg(feature = "capture")]
Self::Capture(tensor) => {
    assert_eq!(context, AutodiffContext::Disabled, "Capture does not support autodiff");
    Self::Capture(tensor.with_context(context))
},
        }
    }

    pub fn device(&self) -> DispatchDevice {
        let device = match self {
Self::Host(tensor) => DispatchDevice::Host(tensor.device()),
#[cfg(feature = "cuda")]
Self::Cuda(tensor) => DispatchDevice::Cuda(tensor.device()),
#[cfg(feature = "cpu")]
Self::Cpu(tensor) => DispatchDevice::Cpu(tensor.device()),
#[cfg(feature = "wgpu")]
Self::Wgpu(tensor) => DispatchDevice::Wgpu(tensor.device()),
#[cfg(feature = "rocm")]
Self::Rocm(tensor) => DispatchDevice::Rocm(tensor.device()),
#[cfg(feature = "tch")]
Self::LibTorch(tensor) => DispatchDevice::LibTorch(tensor.device()),
#[cfg(feature = "remote")]
Self::Remote(tensor) => DispatchDevice::Remote(tensor.device()),
#[cfg(feature = "capture")]
Self::Capture(tensor) => DispatchDevice::Capture(tensor.device()),
        };
        device.with_context(self.context())
    }
}

impl TensorMetadata for DispatchTensor {
fn dtype(&self) -> DType {
    match self {
Self::Host(tensor) => tensor.dtype(),
#[cfg(feature = "cuda")]
Self::Cuda(tensor) => tensor.dtype(),
#[cfg(feature = "cpu")]
Self::Cpu(tensor) => tensor.dtype(),
#[cfg(feature = "wgpu")]
Self::Wgpu(tensor) => tensor.dtype(),
#[cfg(feature = "rocm")]
Self::Rocm(tensor) => tensor.dtype(),
#[cfg(feature = "tch")]
Self::LibTorch(tensor) => tensor.dtype(),
#[cfg(feature = "remote")]
Self::Remote(tensor) => tensor.dtype(),
#[cfg(feature = "capture")]
Self::Capture(tensor) => tensor.dtype(),
    }
}
fn shape(&self) -> Shape {
    match self {
Self::Host(tensor) => tensor.shape(),
#[cfg(feature = "cuda")]
Self::Cuda(tensor) => tensor.shape(),
#[cfg(feature = "cpu")]
Self::Cpu(tensor) => tensor.shape(),
#[cfg(feature = "wgpu")]
Self::Wgpu(tensor) => tensor.shape(),
#[cfg(feature = "rocm")]
Self::Rocm(tensor) => tensor.shape(),
#[cfg(feature = "tch")]
Self::LibTorch(tensor) => tensor.shape(),
#[cfg(feature = "remote")]
Self::Remote(tensor) => tensor.shape(),
#[cfg(feature = "capture")]
Self::Capture(tensor) => tensor.shape(),
    }
}
fn rank(&self) -> usize {
    match self {
Self::Host(tensor) => tensor.rank(),
#[cfg(feature = "cuda")]
Self::Cuda(tensor) => tensor.rank(),
#[cfg(feature = "cpu")]
Self::Cpu(tensor) => tensor.rank(),
#[cfg(feature = "wgpu")]
Self::Wgpu(tensor) => tensor.rank(),
#[cfg(feature = "rocm")]
Self::Rocm(tensor) => tensor.rank(),
#[cfg(feature = "tch")]
Self::LibTorch(tensor) => tensor.rank(),
#[cfg(feature = "remote")]
Self::Remote(tensor) => tensor.rank(),
#[cfg(feature = "capture")]
Self::Capture(tensor) => tensor.rank(),
    }
}
fn can_mut(&self) -> bool {
    match self {
Self::Host(tensor) => tensor.can_mut(),
#[cfg(feature = "cuda")]
Self::Cuda(tensor) => tensor.can_mut(),
#[cfg(feature = "cpu")]
Self::Cpu(tensor) => tensor.can_mut(),
#[cfg(feature = "wgpu")]
Self::Wgpu(tensor) => tensor.can_mut(),
#[cfg(feature = "rocm")]
Self::Rocm(tensor) => tensor.can_mut(),
#[cfg(feature = "tch")]
Self::LibTorch(tensor) => tensor.can_mut(),
#[cfg(feature = "remote")]
Self::Remote(tensor) => tensor.can_mut(),
#[cfg(feature = "capture")]
Self::Capture(tensor) => tensor.can_mut(),
    }
}
}

impl QTensorPrimitive for DispatchTensor {
fn scheme(&self) -> &QuantScheme {
    match self {
Self::Host(tensor) => tensor.scheme(),
#[cfg(feature = "cuda")]
Self::Cuda(tensor) => tensor.scheme(),
#[cfg(feature = "cpu")]
Self::Cpu(tensor) => tensor.scheme(),
#[cfg(feature = "wgpu")]
Self::Wgpu(tensor) => tensor.scheme(),
#[cfg(feature = "rocm")]
Self::Rocm(tensor) => tensor.scheme(),
#[cfg(feature = "tch")]
Self::LibTorch(tensor) => tensor.scheme(),
#[cfg(feature = "remote")]
Self::Remote(tensor) => tensor.scheme(),
#[cfg(feature = "capture")]
Self::Capture(tensor) => tensor.scheme(),
    }
}
fn acc_precision(&self) -> QuantAcc {
    match self {
Self::Host(tensor) => tensor.acc_precision(),
#[cfg(feature = "cuda")]
Self::Cuda(tensor) => tensor.acc_precision(),
#[cfg(feature = "cpu")]
Self::Cpu(tensor) => tensor.acc_precision(),
#[cfg(feature = "wgpu")]
Self::Wgpu(tensor) => tensor.acc_precision(),
#[cfg(feature = "rocm")]
Self::Rocm(tensor) => tensor.acc_precision(),
#[cfg(feature = "tch")]
Self::LibTorch(tensor) => tensor.acc_precision(),
#[cfg(feature = "remote")]
Self::Remote(tensor) => tensor.acc_precision(),
#[cfg(feature = "capture")]
Self::Capture(tensor) => tensor.acc_precision(),
    }
}
fn propagation(&self) -> QuantPropagation {
    match self {
Self::Host(tensor) => tensor.propagation(),
#[cfg(feature = "cuda")]
Self::Cuda(tensor) => tensor.propagation(),
#[cfg(feature = "cpu")]
Self::Cpu(tensor) => tensor.propagation(),
#[cfg(feature = "wgpu")]
Self::Wgpu(tensor) => tensor.propagation(),
#[cfg(feature = "rocm")]
Self::Rocm(tensor) => tensor.propagation(),
#[cfg(feature = "tch")]
Self::LibTorch(tensor) => tensor.propagation(),
#[cfg(feature = "remote")]
Self::Remote(tensor) => tensor.propagation(),
#[cfg(feature = "capture")]
Self::Capture(tensor) => tensor.propagation(),
    }
}
}
