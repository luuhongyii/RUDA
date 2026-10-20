#![no_std]
//! Ruda GPU kernel API and portable runtime.

extern crate self as ruda;

#[cfg(feature = "runtime")]
extern crate alloc;

pub use ruda_runtime::*;

#[cfg(feature = "runtime")]
pub mod device;
#[cfg(feature = "runtime")]
pub use device::Device;

#[cfg(feature = "kernel")]
pub use ruda_kernel::dsl;
#[cfg(feature = "kernel")]
pub use ruda_kernel::dsl::prelude;

#[cfg(feature = "cuda")]
pub use ruda_driver_cuda as cuda;
#[cfg(feature = "hip")]
pub use ruda_driver_hip as hip;
#[cfg(feature = "wgpu")]
pub use ruda_driver_wgpu as wgpu;
#[cfg(feature = "cpu")]
pub use ruda_driver_cpu as cpu;
#[cfg(all(feature = "metal", target_vendor = "apple"))]
pub use ruda_driver_metal as metal;



