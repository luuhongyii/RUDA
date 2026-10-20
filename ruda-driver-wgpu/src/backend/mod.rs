mod base;
mod wgsl;
mod identity;
mod memory;

#[cfg(feature = "spirv")]
pub mod vulkan;

#[cfg(all(feature = "msl", target_os = "macos"))]
pub mod metal;

pub use base::*;
pub(crate) use identity::identity;
pub(crate) use memory::register_memory_capacity;
