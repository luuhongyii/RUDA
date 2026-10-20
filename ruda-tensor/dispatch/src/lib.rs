#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[macro_use]
mod macros;
mod backend;
#[cfg(feature = "distributed")]
mod distributed;
mod context;
mod device;
mod ops;
mod tensor;
mod transfer;
#[cfg(feature = "remote-server")]
mod ir;
#[cfg(all(feature = "remote-server", not(target_family = "wasm")))]
pub mod remote_server;

pub use backend::Dispatch;
pub use device::DispatchDevice;
#[cfg(feature = "throughput")]
pub use ruda_runtime::runtime::throughput::{
    CmmaDims, ComputeCmmaConfig, MemoryAccess, MemoryCurve, MemoryPoint, MemorySpec,
    ThroughputError, ThroughputKey, ThroughputMode, ThroughputValue,
};
pub use tensor::{BackendTensor, DispatchTensor};
pub use ruda_core::ir::{DeviceIdentity, PhysicalDevice, PciAddress, PciAddressError, PciVendor, AdapterLuid, RegistryEntryId};
#[cfg(feature = "capture")]
pub use ruda_tensor_capture as capture;

pub mod devices {
    pub use ruda_tensor_host::HostDevice;
    #[cfg(feature = "cuda")]
    pub use ruda_tensor_device::cuda::CudaDevice;
    #[cfg(feature = "cpu")]
    pub use ruda_driver_cpu::CpuDevice;
    #[cfg(feature = "rocm")]
    pub use ruda_tensor_rocm::RocmDevice;
    #[cfg(feature = "wgpu")]
    pub use ruda_tensor_wgpu::{AutoCompiler, WgpuBackend, WgpuDevice, WgpuDeviceKind, init_setup_async};
    #[cfg(feature = "wgpu")]
    pub mod wgpu {
        pub use ruda_tensor_wgpu::{MemoryConfiguration, RuntimeOptions, WgpuBackend, WgpuDevice,
            WgpuDeviceKind, WgpuInitError, WgpuSetup, try_init_device, try_init_device_with_api,
            try_init_setup, try_init_setup_async};
        pub use ruda_tensor_wgpu::graphics::*;
    }
    #[cfg(feature = "tch")]
    pub use ruda_tensor_tch::LibTorchDevice;
    #[cfg(feature = "remote")]
    pub use ruda_tensor_remote::RemoteDevice;
    #[cfg(feature = "capture")]
    pub use ruda_tensor_capture::CaptureDevice;
}

pub mod backends {
pub type Host = ruda_tensor_host::Host;
#[cfg(all(feature = "cuda", not(feature = "fusion")))]
pub type Cuda = ruda_tensor_device::cuda::CudaNative;
#[cfg(all(feature = "cuda", feature = "fusion"))]
pub type Cuda = ruda_fusion::Fusion<ruda_tensor_device::cuda::CudaNative>;
#[cfg(all(feature = "cpu", not(feature = "fusion")))]
pub type Cpu = ruda_tensor_device::DeviceBackend<ruda_driver_cpu::CpuRuntime, f32, i32, u8>;
#[cfg(all(feature = "cpu", feature = "fusion"))]
pub type Cpu = ruda_fusion::Fusion<ruda_tensor_device::DeviceBackend<ruda_driver_cpu::CpuRuntime, f32, i32, u8>>;
#[cfg(all(feature = "wgpu", not(feature = "fusion")))]
pub type Wgpu = ruda_tensor_wgpu::DeviceBackend<ruda_tensor_wgpu::WgpuRuntime, f32, i32, u32>;
#[cfg(all(feature = "wgpu", feature = "fusion"))]
pub type Wgpu = ruda_fusion::Fusion<ruda_tensor_wgpu::DeviceBackend<ruda_tensor_wgpu::WgpuRuntime, f32, i32, u32>>;
#[cfg(all(feature = "rocm", not(feature = "fusion")))]
pub type Rocm = ruda_tensor_rocm::RocmNative;
#[cfg(all(feature = "rocm", feature = "fusion"))]
pub type Rocm = ruda_fusion::Fusion<ruda_tensor_rocm::RocmNative>;
#[cfg(feature = "tch")]
pub type LibTorch = ruda_tensor_tch::LibTorch;
#[cfg(feature = "remote")]
pub type Remote = ruda_tensor_remote::RemoteBackend;
#[cfg(feature = "capture")]
pub type Capture = ruda_tensor_capture::CaptureBackend;
}
