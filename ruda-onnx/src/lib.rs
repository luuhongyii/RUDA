//! Import ONNX graphs as native, backend-generic RUDA modules.

mod adapt;
pub mod import;
pub use import::*;
pub mod pack;
