//! Convert ONNX models into native RUDA Rust source code and Rudapack weight files.

mod logger;

/// Native RUDA code generation.
pub mod codegen;

/// Extension traits used by generated RUDA models.
pub mod ext;

mod formatter;
mod model_gen;

pub use formatter::*;
pub use model_gen::*;
