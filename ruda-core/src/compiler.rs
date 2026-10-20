use crate::{kernel::KernelDefinition, launch::ExecutionMode};
use alloc::string::String;
use crate::backtrace::BackTrace;
use crate::ir::{ElemType, StorageType};
use thiserror::Error;
pub use crate::poison::DevicePoison;

/// JIT compilation error.
#[derive(Error, Clone)]
#[cfg_attr(any(compiler_std_io, feature = "environment-persistence"), derive(serde::Serialize, serde::Deserialize))]
pub enum CompilationError {
    /// An instruction isn't supported.
    #[error(
        "An unsupported instruction caused the compilation to fail\nCaused by:\n  {reason}\nBacktrace:\n{backtrace}"
    )]
    UnsupportedInstruction {
        /// The caused of the error.
        reason: String,
        /// The backtrace for this error.
        #[cfg_attr(any(compiler_std_io, feature = "environment-persistence"), serde(skip))]
        backtrace: BackTrace,
    },

    /// A generic compilation error.
    #[error(
        "An error caused the compilation to fail\nCaused by:\n  {reason}\nBacktrace:\n{backtrace}"
    )]
    Generic {
        /// The error context.
        reason: String,
        /// The backtrace for this error.
        #[cfg_attr(any(compiler_std_io, feature = "environment-persistence"), serde(skip))]
        backtrace: BackTrace,
    },
    /// A generic compilation error.
    #[error(
        "A validation error caused the compilation to fail\nCaused by:\n  {reason}\nBacktrace:\n{backtrace}"
    )]
    Validation {
        /// The error context.
        reason: String,
        /// The backtrace for this error.
        #[cfg_attr(any(compiler_std_io, feature = "environment-persistence"), serde(skip))]
        backtrace: BackTrace,
    },
    /// Kernel expansion panicked before a target representation was produced.
    #[error("Kernel expansion panicked\nCaused by:\n  {reason}\nBacktrace:\n{backtrace}")]
    ExpansionPanicked {
        /// The panic payload, when textual.
        reason: String,
        /// The captured backtrace.
        #[cfg_attr(any(compiler_std_io, feature = "environment-persistence"), serde(skip))]
        backtrace: BackTrace,
    },
    /// The driver cannot load a kernel because the device/context is poisoned.
    #[error("The device is poisoned, the kernel could not be loaded\n{0}")]
    DevicePoisoned(#[from] DevicePoison),
}

impl CompilationError {
    /// Whether a driver fault has made the device unusable.
    pub fn is_device_poisoned(&self) -> bool {
        matches!(self, Self::DevicePoisoned(_))
    }

    /// A compiler refusal can reject a candidate; an expansion panic or device fault cannot.
    pub fn is_refusal(&self) -> bool {
        !matches!(self, Self::ExpansionPanicked { .. } | Self::DevicePoisoned(_))
    }
}

impl core::fmt::Debug for CompilationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self}")
    }
}

/// Compiles the representation into its own representation that can be formatted into tokens.
pub trait Compiler: 'static + Clone + core::fmt::Debug {
    /// The representation for the compiled code.
    type Representation: core::fmt::Display;
    /// The compilation options used to configure the compiler
    type CompilationOptions: Send + Default + core::fmt::Debug;

    /// Compiles the [kernel definition](KernelDefinition) into the compiler's representation.
    fn compile(
        &mut self,
        kernel: KernelDefinition,
        compilation_options: &Self::CompilationOptions,
        mode: ExecutionMode,
        addr_type: StorageType,
    ) -> Result<Self::Representation, CompilationError>;

    /// The size of the given element in bytes.
    fn elem_size(&self, elem: ElemType) -> usize;

    /// Compiled payload accesses, or `None` when every argument must be treated as read-write.
    fn buffer_io(&self, _repr: &Self::Representation) -> Option<crate::kernel::KernelIo> {
        None
    }

    /// The default extension for the runtime's kernel/shader code.
    /// Might change based on which compiler is used.
    fn extension(&self) -> &'static str;

    /// Language accepted by a hand-written precompiled kernel.
    fn lang_tag(&self) -> &'static str {
        match self.extension() {
            "spv" => "spirv",
            other => other,
        }
    }
}

mod options;
pub use options::{WgpuCompilationOptions, VulkanCompilationOptions};
