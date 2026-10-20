//! A device of any linked runtime, with IDs independent of build features.

use alloc::vec::Vec;
use ruda_core::{device::Device as DeviceIdentity, tensor::{Shape, Strides}};
use ruda_runtime::runtime::{backend::Runtime, client::Client};
pub use ruda_core::device::DeviceId;
pub use ruda_runtime::runtime::{
    backend::{DeviceUtilization, UtilizationUnavailable},
    device::{AmdDevice, CpuDevice, CudaDevice, MetalDevice, WgpuBackend, WgpuDevice, WgpuDeviceKind},
};

/// A device selector that remains representable even when its runtime is not linked.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Device {
    /// CUDA device.
    Cuda(CudaDevice),
    /// HIP device.
    Hip(AmdDevice),
    /// Native Metal device, distinct from WGPU Metal.
    Metal(MetalDevice),
    /// WGPU adapter and graphics API.
    Wgpu(WgpuDevice),
    /// Native CPU device.
    Cpu(CpuDevice),
}

/// Stable tags; runtime-local kinds occupy the low five type-ID bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum RuntimeId {
    /// CUDA.
    Cuda = 0,
    /// HIP.
    Hip = 1,
    /// Native Metal.
    Metal = 2,
    /// WGPU.
    Wgpu = 3,
    /// Native CPU.
    Cpu = 4,
}

impl TryFrom<u16> for RuntimeId {
    type Error = u16;
    fn try_from(value: u16) -> Result<Self, u16> {
        match value {
            0 => Ok(Self::Cuda), 1 => Ok(Self::Hip), 2 => Ok(Self::Metal),
            3 => Ok(Self::Wgpu), 4 => Ok(Self::Cpu), other => Err(other),
        }
    }
}

impl RuntimeId {
    /// Bits reserved for a caller nesting this ID in another encoding.
    pub const OUTER_MASK: u16 = 0xff00;
    /// Read the runtime tag without consuming the caller's outer bits.
    pub fn of_device_id(id: DeviceId) -> Result<Self, u16> { Self::try_from((id.type_id >> 5) & 7) }
    /// Remove the runtime and outer tags.
    pub fn strip(id: DeviceId) -> DeviceId { DeviceId::new(id.type_id & 31, id.index_id) }
    /// Stamp a runtime without altering outer bits.
    pub fn stamp(self, id: DeviceId) -> DeviceId {
        DeviceId::new((id.type_id & !0xe0) | ((self as u16) << 5), id.index_id)
    }
    fn is_linked(self) -> bool {
        match self {
            Self::Cuda => cfg!(feature = "cuda"),
            Self::Hip => cfg!(feature = "hip"),
            Self::Wgpu => cfg!(feature = "wgpu"),
            Self::Cpu => cfg!(feature = "cpu"),
            Self::Metal => cfg!(all(feature = "metal", target_vendor = "apple")),
        }
    }
    fn default_device(self) -> Device {
        match self {
            Self::Cuda => Device::Cuda(Default::default()),
            Self::Hip => Device::Hip(Default::default()),
            Self::Metal => Device::Metal(Default::default()),
            Self::Wgpu => Device::Wgpu(Default::default()),
            Self::Cpu => Device::Cpu(Default::default()),
        }
    }
    fn enumerate_all(self) -> Vec<Device> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda => crate::cuda::CudaDevice::enumerate().into_iter().map(Device::from).collect(),
            #[cfg(feature = "hip")]
            Self::Hip => crate::hip::AmdDevice::enumerate().into_iter().map(Device::from).collect(),
            #[cfg(all(feature = "metal", target_vendor = "apple"))]
            Self::Metal => crate::metal::MetalRuntime::enumerate_all_devices(&()).into_iter()
                .map(|id| Device::Metal(MetalDevice::from_id(id))).collect(),
            #[cfg(feature = "wgpu")]
            Self::Wgpu => crate::wgpu::WgpuDevice::enumerate().into_iter().map(Device::from).collect(),
            #[cfg(feature = "cpu")]
            Self::Cpu => crate::cpu::CpuDevice::enumerate().into_iter().map(Device::from).collect(),
            _ => Vec::new(),
        }
    }
    fn enumerate_devices_like(self, id: DeviceId) -> Vec<DeviceId> {
        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        if self == Self::Metal {
            return crate::metal::MetalRuntime::enumerate_devices(id.type_id, &());
        }
        #[cfg(feature = "wgpu")]
        if self == Self::Wgpu {
            return native_wgpu(&WgpuDevice::from_id(id)).enumerate_like().into_iter()
                .map(|device| match Device::from(device) {
                    Device::Wgpu(device) => device.to_id(),
                    _ => unreachable!(),
                }).collect();
        }
        self.enumerate_all().into_iter().map(|device| Self::strip(device.to_id())).collect()
    }
    fn is_available(self) -> bool {
        self.enumerate_all().iter().any(|device| {
            !matches!(device, Device::Wgpu(WgpuDevice { kind: WgpuDeviceKind::Cpu, .. }))
        })
    }
}

/// A missing build feature or an unavailable hardware selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceUnavailable {
    /// This runtime is not linked into the build.
    NotLinked(RuntimeId),
    /// The runtime has only `available` devices of this kind.
    NoSuchDevice {
        /// Runtime queried.
        runtime: RuntimeId,
        /// Number of available devices of this kind.
        available: usize,
    },
}

impl core::fmt::Display for DeviceUnavailable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotLinked(runtime) => write!(f, "this build does not link the {runtime:?} runtime"),
            Self::NoSuchDevice { runtime, available } =>
                write!(f, "the {runtime:?} runtime has no such device; {available} of that kind are available"),
        }
    }
}
impl core::error::Error for DeviceUnavailable {}

impl Device {
    /// Name the native CPU.
    pub fn cpu() -> Result<Self, DeviceUnavailable> { Self::named(Self::Cpu(CpuDevice)) }
    /// Name a visible CUDA ordinal.
    pub fn cuda(index: usize) -> Result<Self, DeviceUnavailable> { Self::named(Self::Cuda(CudaDevice::new(index))) }
    /// Name a visible HIP ordinal.
    pub fn rocm(index: usize) -> Result<Self, DeviceUnavailable> { Self::named(Self::Hip(AmdDevice::new(index))) }
    /// Name a native Metal selector without substituting WGPU Metal.
    pub fn metal(kind: MetalDevice) -> Result<Self, DeviceUnavailable> {
        if matches!(kind, MetalDevice::Existing(_)) { Self::linked(Self::Metal(kind)) }
        else { Self::named(Self::Metal(kind)) }
    }
    /// Name a WGPU adapter on the runtime-selected graphics API.
    pub fn wgpu(kind: WgpuDeviceKind) -> Result<Self, DeviceUnavailable> {
        Self::wgpu_device(WgpuDevice::new(kind))
    }
    /// Name a WGPU Vulkan adapter.
    pub fn vulkan(kind: WgpuDeviceKind) -> Result<Self, DeviceUnavailable> {
        Self::wgpu_device(WgpuDevice::new(kind).on(WgpuBackend::Vulkan))
    }
    /// Name a WGPU DirectX 12 adapter.
    pub fn dx12(kind: WgpuDeviceKind) -> Result<Self, DeviceUnavailable> {
        Self::wgpu_device(WgpuDevice::new(kind).on(WgpuBackend::Dx12))
    }
    /// Name a WGPU OpenGL adapter.
    pub fn gl(kind: WgpuDeviceKind) -> Result<Self, DeviceUnavailable> {
        Self::wgpu_device(WgpuDevice::new(kind).on(WgpuBackend::Gl))
    }
    /// Name a browser WebGPU adapter.
    pub fn webgpu(kind: WgpuDeviceKind) -> Result<Self, DeviceUnavailable> {
        Self::wgpu_device(WgpuDevice::new(kind).on(WgpuBackend::WebGpu))
    }
    /// Name WGPU's Metal path, not the native Metal runtime.
    pub fn metal_msl(kind: WgpuDeviceKind) -> Result<Self, DeviceUnavailable> {
        Self::wgpu_device(WgpuDevice::new(kind).on(WgpuBackend::Metal))
    }
    /// Pin a WGPU selector to one graphics API.
    pub fn on(self, backend: WgpuBackend) -> Result<Self, DeviceUnavailable> {
        match self {
            Self::Wgpu(device) => Self::wgpu_device(device.on(backend)),
            other => Err(DeviceUnavailable::NoSuchDevice { runtime: other.runtime(), available: 0 }),
        }
    }
    fn wgpu_device(device: WgpuDevice) -> Result<Self, DeviceUnavailable> {
        if matches!(device.kind, WgpuDeviceKind::Existing(_)) {
            Self::linked(Self::Wgpu(device))
        } else { Self::named(Self::Wgpu(device)) }
    }
    fn linked(device: Self) -> Result<Self, DeviceUnavailable> {
        if device.runtime().is_linked() { Ok(device) }
        else { Err(DeviceUnavailable::NotLinked(device.runtime())) }
    }
    fn named(device: Self) -> Result<Self, DeviceUnavailable> {
        let device = Self::linked(device)?;
        let found = match &device {
            #[cfg(feature = "cuda")]
            Self::Cuda(device) => crate::cuda::CudaDevice::new(device.index).find(),
            #[cfg(feature = "hip")]
            Self::Hip(device) => crate::hip::AmdDevice::new(device.index).find(),
            #[cfg(all(feature = "metal", target_vendor = "apple"))]
            Self::Metal(device) => {
                let id = device.to_id();
                let devices = crate::metal::MetalRuntime::enumerate_devices(id.type_id, &());
                if devices.contains(&id) { Ok(()) } else { Err(devices.len()) }
            }
            #[cfg(feature = "wgpu")]
            Self::Wgpu(device) => native_wgpu(device).find(),
            #[cfg(feature = "cpu")]
            Self::Cpu(_) => Ok(()),
            _ => Err(0),
        };
        found.map(|()| device.clone()).map_err(|available| DeviceUnavailable::NoSuchDevice {
            runtime: device.runtime(), available,
        })
    }
    /// Reach the selected service, initializing it on first use.
    pub fn client(&self) -> Client {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(device) => crate::cuda::CudaRuntime::client(&crate::cuda::CudaDevice::new(device.index)).erase(),
            #[cfg(feature = "hip")]
            Self::Hip(device) => crate::hip::HipRuntime::client(&crate::hip::AmdDevice::new(device.index)).erase(),
            #[cfg(all(feature = "metal", target_vendor = "apple"))]
            Self::Metal(device) => crate::metal::MetalRuntime::client(device).erase(),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(device) => crate::wgpu::WgpuRuntime::client(&native_wgpu(device)).erase(),
            #[cfg(feature = "cpu")]
            Self::Cpu(_) => crate::cpu::CpuRuntime::client(&crate::cpu::CpuDevice).erase(),
            other => panic!("{other:?} belongs to a runtime this build does not link"),
        }
    }
    /// Query layout compatibility without initializing a device client.
    pub fn can_read_tensor(&self, shape: &Shape, strides: &Strides) -> bool {
        let _ = (shape, strides);
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => crate::cuda::CudaRuntime::can_read_tensor(shape, strides),
            #[cfg(feature = "hip")]
            Self::Hip(_) => crate::hip::HipRuntime::can_read_tensor(shape, strides),
            #[cfg(all(feature = "metal", target_vendor = "apple"))]
            Self::Metal(_) => crate::metal::MetalRuntime::can_read_tensor(shape, strides),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(_) => crate::wgpu::WgpuRuntime::can_read_tensor(shape, strides),
            #[cfg(feature = "cpu")]
            Self::Cpu(_) => crate::cpu::CpuRuntime::can_read_tensor(shape, strides),
            other => panic!("{other:?} belongs to a runtime this build does not link"),
        }
    }
    /// Enumerate peers of a global ID, retaining outer nesting bits.
    pub fn enumerate(id: DeviceId) -> Vec<DeviceId> {
        let Ok(runtime) = RuntimeId::of_device_id(id) else { return Vec::new(); };
        let outer = id.type_id & RuntimeId::OUTER_MASK;
        runtime.enumerate_devices_like(RuntimeId::strip(id)).into_iter()
            .map(|id| runtime.stamp(DeviceId::new(id.type_id | outer, id.index_id))).collect()
    }
    /// Enumerate every adapter of every linked runtime.
    pub fn enumerate_all() -> Vec<Self> {
        [RuntimeId::Cuda, RuntimeId::Hip, RuntimeId::Metal, RuntimeId::Wgpu, RuntimeId::Cpu]
            .into_iter().flat_map(RuntimeId::enumerate_all).collect()
    }
    /// Runtime represented by this selector.
    pub fn runtime(&self) -> RuntimeId {
        match self {
            Self::Cuda(_) => RuntimeId::Cuda, Self::Hip(_) => RuntimeId::Hip,
            Self::Metal(_) => RuntimeId::Metal, Self::Wgpu(_) => RuntimeId::Wgpu,
            Self::Cpu(_) => RuntimeId::Cpu,
        }
    }
    /// Query the runtime's actual driver/platform utilization counter.
    pub fn utilization(&self) -> Result<DeviceUtilization, UtilizationUnavailable> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(device) => crate::cuda::CudaRuntime::utilization(&crate::cuda::CudaDevice::new(device.index)),
            #[cfg(feature = "hip")]
            Self::Hip(device) => crate::hip::HipRuntime::utilization(&crate::hip::AmdDevice::new(device.index)),
            #[cfg(all(feature = "metal", target_vendor = "apple"))]
            Self::Metal(device) => crate::metal::MetalRuntime::utilization(device),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(device) => crate::wgpu::WgpuRuntime::utilization(&native_wgpu(device)),
            #[cfg(feature = "cpu")]
            Self::Cpu(_) => crate::cpu::CpuRuntime::utilization(&crate::cpu::CpuDevice),
            _ => Err(UtilizationUnavailable::RuntimeNotLinked),
        }
    }
    /// Decode a global catalogue ID.
    pub fn from_id(id: DeviceId) -> Self {
        let inner = RuntimeId::strip(id);
        match RuntimeId::of_device_id(id) {
            Ok(RuntimeId::Cuda) => Self::Cuda(CudaDevice::from_id(inner)),
            Ok(RuntimeId::Hip) => Self::Hip(AmdDevice::from_id(inner)),
            Ok(RuntimeId::Metal) => Self::Metal(MetalDevice::from_id(inner)),
            Ok(RuntimeId::Wgpu) => Self::Wgpu(WgpuDevice::from_id(inner)),
            Ok(RuntimeId::Cpu) => Self::Cpu(CpuDevice::from_id(inner)),
            Err(tag) => panic!("device id {id} names unknown runtime tag {tag}"),
        }
    }
    /// Encode a global catalogue ID, distinct from native driver-local IDs.
    pub fn to_id(&self) -> DeviceId {
        let inner = match self {
            Self::Cuda(device) => device.to_id(), Self::Hip(device) => device.to_id(),
            Self::Metal(device) => device.to_id(), Self::Wgpu(device) => device.to_id(),
            Self::Cpu(device) => device.to_id(),
        };
        self.runtime().stamp(inner)
    }
}

#[cfg(any(feature = "cuda", feature = "hip", feature = "wgpu", feature = "cpu", all(feature = "metal", target_vendor = "apple")))]
static DEFAULT_DEVICE: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(u32::MAX);

#[cfg(any(feature = "cuda", feature = "hip", feature = "wgpu", feature = "cpu", all(feature = "metal", target_vendor = "apple")))]
impl Default for Device {
    fn default() -> Self {
        use core::sync::atomic::Ordering;
        let cached = DEFAULT_DEVICE.load(Ordering::Relaxed);
        if cached != u32::MAX { return Self::from_id(DeviceId::new((cached >> 16) as u16, cached as u16)); }
        let linked = [RuntimeId::Cuda, RuntimeId::Hip, RuntimeId::Metal, RuntimeId::Wgpu, RuntimeId::Cpu]
            .into_iter().filter(|runtime| runtime.is_linked()).collect::<Vec<_>>();
        let (last, rest) = linked.split_last().expect("At least one runtime is linked");
        let runtime = rest.iter().find(|runtime| runtime.is_available()).unwrap_or(last);
        let device = runtime.default_device();
        let id = device.to_id();
        DEFAULT_DEVICE.store(((id.type_id as u32) << 16) | id.index_id as u32, Ordering::Relaxed);
        device
    }
}

#[cfg(any(feature = "cuda", feature = "hip", feature = "wgpu", feature = "cpu", all(feature = "metal", target_vendor = "apple")))]
impl DeviceIdentity for Device {
    fn from_id(id: DeviceId) -> Self { Self::from_id(id) }
    fn to_id(&self) -> DeviceId { Self::to_id(self) }
}

impl From<CudaDevice> for Device { fn from(device: CudaDevice) -> Self { Self::Cuda(device) } }
impl From<AmdDevice> for Device { fn from(device: AmdDevice) -> Self { Self::Hip(device) } }
impl From<MetalDevice> for Device { fn from(device: MetalDevice) -> Self { Self::Metal(device) } }
impl From<WgpuDevice> for Device { fn from(device: WgpuDevice) -> Self { Self::Wgpu(device) } }
impl From<CpuDevice> for Device { fn from(device: CpuDevice) -> Self { Self::Cpu(device) } }

#[cfg(feature = "cuda")]
impl From<crate::cuda::CudaDevice> for Device {
    fn from(device: crate::cuda::CudaDevice) -> Self { Self::Cuda(CudaDevice::new(device.index)) }
}
#[cfg(feature = "hip")]
impl From<crate::hip::AmdDevice> for Device {
    fn from(device: crate::hip::AmdDevice) -> Self { Self::Hip(AmdDevice::new(device.index)) }
}
#[cfg(feature = "cpu")]
impl From<crate::cpu::CpuDevice> for Device {
    fn from(_: crate::cpu::CpuDevice) -> Self { Self::Cpu(CpuDevice) }
}

#[cfg(feature = "wgpu")]
fn native_wgpu(device: &WgpuDevice) -> crate::wgpu::WgpuDevice {
    use crate::wgpu::{WgpuBackend as Backend, WgpuDeviceKind as Kind};
    let kind = match device.kind {
        WgpuDeviceKind::DiscreteGpu(i) => Kind::DiscreteGpu(i),
        WgpuDeviceKind::IntegratedGpu(i) => Kind::IntegratedGpu(i),
        WgpuDeviceKind::VirtualGpu(i) => Kind::VirtualGpu(i),
        WgpuDeviceKind::Cpu => Kind::Cpu,
        WgpuDeviceKind::DefaultDevice => Kind::DefaultDevice,
        WgpuDeviceKind::Existing(i) => Kind::Existing(i),
        WgpuDeviceKind::Other(i) => Kind::Other(i),
    };
    let backend = match device.backend {
        WgpuBackend::Auto => Backend::Auto, WgpuBackend::Vulkan => Backend::Vulkan,
        WgpuBackend::Metal => Backend::Metal, WgpuBackend::Dx12 => Backend::Dx12,
        WgpuBackend::Gl => Backend::Gl, WgpuBackend::WebGpu => Backend::WebGpu,
    };
    crate::wgpu::WgpuDevice::new(kind).on(backend)
}

#[cfg(feature = "wgpu")]
impl From<crate::wgpu::WgpuDevice> for Device {
    fn from(device: crate::wgpu::WgpuDevice) -> Self {
        use crate::wgpu::{WgpuBackend as Backend, WgpuDeviceKind as Kind};
        let kind = match device.kind() {
            Kind::DiscreteGpu(i) => WgpuDeviceKind::DiscreteGpu(i),
            Kind::IntegratedGpu(i) => WgpuDeviceKind::IntegratedGpu(i),
            Kind::VirtualGpu(i) => WgpuDeviceKind::VirtualGpu(i),
            Kind::Cpu => WgpuDeviceKind::Cpu,
            Kind::DefaultDevice => WgpuDeviceKind::DefaultDevice,
            Kind::Existing(i) => WgpuDeviceKind::Existing(i),
            Kind::Other(i) => WgpuDeviceKind::Other(i),
        };
        let backend = match device.backend() {
            Backend::Auto => WgpuBackend::Auto, Backend::Vulkan => WgpuBackend::Vulkan,
            Backend::Metal => WgpuBackend::Metal, Backend::Dx12 => WgpuBackend::Dx12,
            Backend::Gl => WgpuBackend::Gl, Backend::WebGpu => WgpuBackend::WebGpu,
        };
        Self::Wgpu(WgpuDevice::new(kind).on(backend))
    }
}
