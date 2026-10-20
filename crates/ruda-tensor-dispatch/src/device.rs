use alloc::boxed::Box;
use ruda_core::device::Device;
use ruda_tensor::{AutodiffContext, DeviceId, DeviceOps, GradientCheckpointingStrategy};

#[derive(Clone, Debug)]
pub enum DispatchDevice {
Host(ruda_tensor_host::HostDevice),
#[cfg(feature = "cuda")]
Cuda(ruda_tensor_device::cuda::CudaDevice),
#[cfg(feature = "cpu")]
Cpu(ruda_driver_cpu::CpuDevice),
#[cfg(feature = "wgpu")]
Wgpu(ruda_tensor_wgpu::WgpuDevice),
#[cfg(feature = "rocm")]
Rocm(ruda_tensor_rocm::RocmDevice),
#[cfg(feature = "tch")]
LibTorch(ruda_tensor_tch::LibTorchDevice),
#[cfg(feature = "remote")]
Remote(ruda_tensor_remote::RemoteDevice),
#[cfg(feature = "capture")]
Capture(ruda_tensor_capture::CaptureDevice),
    Autodiff(Box<DispatchDevice>, GradientCheckpointingStrategy),
}

impl DispatchDevice {
    #[cfg(feature = "throughput")]
    pub fn performance_stats(
        &self,
        keys: &[crate::ThroughputKey],
    ) -> alloc::vec::Vec<Result<crate::ThroughputValue, crate::ThroughputError>> {
        #[allow(unused_imports)]
        use ruda_kernel::library::throughput::device_throughput;
        let _ = keys;
        match self {
            Self::Host(_) => alloc::vec::Vec::new(),
            #[cfg(feature = "cuda")]
            Self::Cuda(device) => device_throughput::<ruda_tensor_device::cuda::CudaRuntime>(device, keys),
            #[cfg(feature = "cpu")]
            Self::Cpu(device) => device_throughput::<ruda_driver_cpu::CpuRuntime>(device, keys),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(device) => device_throughput::<ruda_tensor_wgpu::WgpuRuntime>(device, keys),
            #[cfg(feature = "rocm")]
            Self::Rocm(device) => device_throughput::<ruda_tensor_rocm::HipRuntime>(device, keys),
            #[cfg(feature = "tch")]
            Self::LibTorch(_) => alloc::vec::Vec::new(),
            #[cfg(feature = "remote")]
            Self::Remote(_) => alloc::vec::Vec::new(),
            #[cfg(feature = "capture")]
            Self::Capture(_) => alloc::vec::Vec::new(),
            Self::Autodiff(device, _) => device.performance_stats(keys),
        }
    }

    pub fn identity(&self) -> Option<ruda_core::ir::DeviceIdentity> {
        match self.inner() {
            Self::Host(_) => None,
            #[cfg(feature = "cuda")]
            Self::Cuda(device) => Some(device.identity()),
            #[cfg(feature = "cpu")]
            Self::Cpu(device) => Some(device.identity()),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(device) => Some(device.identity()),
            #[cfg(feature = "rocm")]
            Self::Rocm(device) => Some(device.identity()),
            #[cfg(feature = "tch")]
            Self::LibTorch(_) => None,
            #[cfg(feature = "remote")]
            Self::Remote(_) => None,
            #[cfg(feature = "capture")]
            Self::Capture(_) => None,
            Self::Autodiff(..) => unreachable!(),
        }
    }

    #[cfg(feature = "capture")]
    pub fn capture() -> Self { Self::Capture(Default::default()) }

    #[cfg(feature = "capture")]
    pub fn capture_scope(&self, capture: impl FnOnce(ruda_tensor_capture::CaptureScope) -> ruda_tensor_capture::CompletedCaptureScope)
        -> Result<ruda_tensor_capture::CapturedGraph, ruda_tensor_capture::CaptureError> {
        match self {
            Self::Capture(device) => device.capture_scope(capture),
            _ => Err(ruda_tensor_capture::CaptureError::InvalidDevice),
        }
    }

    pub fn autodiff(device: impl Into<Self>) -> Self {
        Self::autodiff_with_gradient_checkpointing(device, GradientCheckpointingStrategy::Disabled)
    }

    pub fn autodiff_with_gradient_checkpointing(device: impl Into<Self>, strategy: GradientCheckpointingStrategy) -> Self {
        let device = device.into();
        assert!(!matches!(strategy, GradientCheckpointingStrategy::Custom(_)),
            "Dispatch accepts Disabled or Balanced checkpointing");
        #[cfg(feature = "capture")]
        assert!(!matches!(device.inner(), Self::Capture(_)), "Capture does not support autodiff");
        Self::Autodiff(Box::new(device.inner().clone()), strategy)
    }

    pub fn context(&self) -> AutodiffContext {
        match self {
            Self::Autodiff(_, strategy) => AutodiffContext::Enabled(*strategy),
            _ => AutodiffContext::Disabled,
        }
    }

    pub fn gradient_checkpointing_strategy(&self) -> GradientCheckpointingStrategy {
        match self.context() {
            AutodiffContext::Enabled(strategy) => strategy,
            AutodiffContext::Disabled => GradientCheckpointingStrategy::Disabled,
        }
    }

    pub fn with_context(&self, context: AutodiffContext) -> Self {
        match context {
            AutodiffContext::Disabled => self.inner().clone(),
            AutodiffContext::Enabled(strategy) => Self::autodiff_with_gradient_checkpointing(self.clone(), strategy),
        }
    }
}

impl PartialEq for DispatchDevice {
    fn eq(&self, other: &Self) -> bool {
        match (self.inner(), other.inner()) {
(Self::Host(a), Self::Host(b)) => a == b,
#[cfg(feature = "cuda")]
(Self::Cuda(a), Self::Cuda(b)) => a == b,
#[cfg(feature = "cpu")]
(Self::Cpu(a), Self::Cpu(b)) => a == b,
#[cfg(feature = "wgpu")]
(Self::Wgpu(a), Self::Wgpu(b)) => a == b,
#[cfg(feature = "rocm")]
(Self::Rocm(a), Self::Rocm(b)) => a == b,
#[cfg(feature = "tch")]
(Self::LibTorch(a), Self::LibTorch(b)) => a == b,
#[cfg(feature = "remote")]
(Self::Remote(a), Self::Remote(b)) => a == b,
#[cfg(feature = "capture")]
(Self::Capture(a), Self::Capture(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for DispatchDevice {}

impl DeviceOps for DispatchDevice {
    fn defaults(&self) -> Option<ruda_tensor::DeviceSettings> {
        let settings = match self.inner() {
            Self::Host(device) => ruda_tensor::get_device_settings::<crate::backends::Host>(device),
            #[cfg(feature = "cuda")]
            Self::Cuda(device) => ruda_tensor::get_device_settings::<crate::backends::Cuda>(device),
            #[cfg(feature = "cpu")]
            Self::Cpu(device) => ruda_tensor::get_device_settings::<crate::backends::Cpu>(device),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(device) => ruda_tensor::get_device_settings::<crate::backends::Wgpu>(device),
            #[cfg(feature = "rocm")]
            Self::Rocm(device) => ruda_tensor::get_device_settings::<crate::backends::Rocm>(device),
            #[cfg(feature = "tch")]
            Self::LibTorch(device) => ruda_tensor::get_device_settings::<crate::backends::LibTorch>(device),
            #[cfg(feature = "remote")]
            Self::Remote(device) => ruda_tensor::get_device_settings::<crate::backends::Remote>(device),
            #[cfg(feature = "capture")]
            Self::Capture(device) => ruda_tensor::get_device_settings::<crate::backends::Capture>(device),
            Self::Autodiff(..) => unreachable!(),
        };
        Some(settings)
    }

    fn inner(&self) -> &Self {
        match self { Self::Autodiff(device, _) => device.inner(), _ => self }
    }
}

impl Default for DispatchDevice {
    fn default() -> Self {
        #[cfg(feature = "std")]
        if let Ok(name) = std::env::var("RUDA_DEVICE") {
            return match name.to_ascii_lowercase().as_str() {
"host" => Self::Host(Default::default()),
#[cfg(feature = "cuda")]
"cuda" => Self::Cuda(Default::default()),
#[cfg(feature = "cpu")]
"cpu" => Self::Cpu(Default::default()),
#[cfg(feature = "wgpu")]
"wgpu" => Self::Wgpu(Default::default()),
#[cfg(feature = "metal")]
"metal" => Self::Wgpu(ruda_tensor_wgpu::WgpuDevice::default().on(ruda_tensor_wgpu::WgpuBackend::Metal)),
#[cfg(feature = "vulkan")]
"vulkan" => Self::Wgpu(ruda_tensor_wgpu::WgpuDevice::default().on(ruda_tensor_wgpu::WgpuBackend::Vulkan)),
#[cfg(feature = "webgpu")]
"webgpu" => Self::Wgpu(ruda_tensor_wgpu::WgpuDevice::default().on(ruda_tensor_wgpu::WgpuBackend::WebGpu)),
#[cfg(feature = "rocm")]
"rocm" => Self::Rocm(Default::default()),
#[cfg(feature = "tch")]
"tch" | "libtorch" => Self::LibTorch(Default::default()),
#[cfg(feature = "remote")]
"remote" => Self::Remote(Default::default()),
                _ => panic!("Unknown or disabled RUDA_DEVICE: {name}"),
            };
        }
#[cfg(feature = "cuda")]
{ return Self::Cuda(Default::default()); }
#[cfg(feature = "metal")]
{ return Self::Wgpu(Default::default()); }
#[cfg(feature = "rocm")]
{ return Self::Rocm(Default::default()); }
#[cfg(feature = "vulkan")]
{ return Self::Wgpu(Default::default()); }
#[cfg(feature = "webgpu")]
{ return Self::Wgpu(Default::default()); }
#[cfg(feature = "wgpu")]
{ return Self::Wgpu(Default::default()); }
#[cfg(feature = "cpu")]
{ return Self::Cpu(Default::default()); }
#[cfg(feature = "tch")]
{ return Self::LibTorch(Default::default()); }
#[cfg(feature = "remote")]
{ return Self::Remote(Default::default()); }
        #[allow(unreachable_code)]
        Self::Host(Default::default())
    }
}

impl Device for DispatchDevice {
    fn to_id(&self) -> DeviceId {
        let (tag, id) = match self.inner() {
Self::Host(device) => (1u16, device.to_id()),
#[cfg(feature = "cuda")]
Self::Cuda(device) => (2u16, device.to_id()),
#[cfg(feature = "cpu")]
Self::Cpu(device) => (3u16, device.to_id()),
#[cfg(feature = "wgpu")]
Self::Wgpu(device) => (4u16, device.to_id()),
#[cfg(feature = "rocm")]
Self::Rocm(device) => (5u16, device.to_id()),
#[cfg(feature = "tch")]
Self::LibTorch(device) => (6u16, device.to_id()),
#[cfg(feature = "remote")]
Self::Remote(device) => (7u16, device.to_id()),
#[cfg(feature = "capture")]
Self::Capture(device) => (8u16, device.to_id()),
            Self::Autodiff(..) => unreachable!(),
        };
        assert!(id.type_id < 256, "Native device type exceeds the Dispatch device namespace");
        DeviceId::new((tag << 8) | id.type_id, id.index_id)
    }

    fn from_id(id: DeviceId) -> Self {
        let native = DeviceId::new(id.type_id & 255, id.index_id);
        let device = match (id.type_id & 0x7fff) >> 8 {
1 => Self::Host(<ruda_tensor_host::HostDevice as Device>::from_id(native)),
#[cfg(feature = "cuda")]
2 => Self::Cuda(<ruda_tensor_device::cuda::CudaDevice as Device>::from_id(native)),
#[cfg(feature = "cpu")]
3 => Self::Cpu(<ruda_driver_cpu::CpuDevice as Device>::from_id(native)),
#[cfg(feature = "wgpu")]
4 => Self::Wgpu(<ruda_tensor_wgpu::WgpuDevice as Device>::from_id(native)),
#[cfg(feature = "rocm")]
5 => Self::Rocm(<ruda_tensor_rocm::RocmDevice as Device>::from_id(native)),
#[cfg(feature = "tch")]
6 => Self::LibTorch(<ruda_tensor_tch::LibTorchDevice as Device>::from_id(native)),
#[cfg(feature = "remote")]
7 => Self::Remote(<ruda_tensor_remote::RemoteDevice as Device>::from_id(native)),
#[cfg(feature = "capture")]
8 => Self::Capture(<ruda_tensor_capture::CaptureDevice as Device>::from_id(native)),
            _ => panic!("Unknown or disabled Dispatch device: {id}"),
        };
        if id.type_id & 0x8000 != 0 { Self::autodiff(device) } else { device }
    }
}

impl From<ruda_tensor_host::HostDevice> for DispatchDevice {
    fn from(device: ruda_tensor_host::HostDevice) -> Self { Self::Host(device) }
}
#[cfg(feature = "cuda")]
impl From<ruda_tensor_device::cuda::CudaDevice> for DispatchDevice {
    fn from(device: ruda_tensor_device::cuda::CudaDevice) -> Self { Self::Cuda(device) }
}
#[cfg(feature = "cpu")]
impl From<ruda_driver_cpu::CpuDevice> for DispatchDevice {
    fn from(device: ruda_driver_cpu::CpuDevice) -> Self { Self::Cpu(device) }
}
#[cfg(feature = "wgpu")]
impl From<ruda_tensor_wgpu::WgpuDevice> for DispatchDevice {
    fn from(device: ruda_tensor_wgpu::WgpuDevice) -> Self { Self::Wgpu(device) }
}
#[cfg(feature = "rocm")]
impl From<ruda_tensor_rocm::RocmDevice> for DispatchDevice {
    fn from(device: ruda_tensor_rocm::RocmDevice) -> Self { Self::Rocm(device) }
}
#[cfg(feature = "tch")]
impl From<ruda_tensor_tch::LibTorchDevice> for DispatchDevice {
    fn from(device: ruda_tensor_tch::LibTorchDevice) -> Self { Self::LibTorch(device) }
}
#[cfg(feature = "remote")]
impl From<ruda_tensor_remote::RemoteDevice> for DispatchDevice {
    fn from(device: ruda_tensor_remote::RemoteDevice) -> Self { Self::Remote(device) }
}

#[cfg(feature = "capture")]
impl From<ruda_tensor_capture::CaptureDevice> for DispatchDevice {
    fn from(device: ruda_tensor_capture::CaptureDevice) -> Self { Self::Capture(device) }
}
