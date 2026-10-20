#![cfg(target_vendor = "apple")]

mod compilation;
pub mod device;
pub mod memory;
mod runtime;
mod server;
mod stream;

pub use device::{MetalDevice, register_device};
pub use runtime::{MetalCompiler, MetalRuntime};
pub use server::MetalServer;

