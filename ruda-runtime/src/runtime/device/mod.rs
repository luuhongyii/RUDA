//! Catalogue selectors independent of linked runtime driver crates.

mod cpu;
mod cuda;
mod hip;
mod metal;
mod wgpu;

pub use cpu::CpuDevice;
pub use cuda::CudaDevice;
pub use hip::AmdDevice;
pub use metal::MetalDevice;
pub use wgpu::{WgpuBackend, WgpuDevice, WgpuDeviceKind};

pub use ruda_core::device::DeviceId;

/// A device index as a [`DeviceId`] carries it.
///
/// Kept at the largest the id holds rather than wrapped: wrapped, an index
/// past the end lands on a device that exists; kept, it names one no machine
/// has, and naming it is refused.
fn index_id(index: usize) -> u16 {
    index.min(u16::MAX as usize) as u16
}
