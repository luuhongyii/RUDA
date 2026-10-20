use crate::runtime::{kernel::{CompiledKernel, KernelMetadata}, server::ExecutionMode};
use ruda_core::ir::StorageType;
pub use ruda_core::compiler::{CompilationError, Compiler};

mod cache;
mod artifacts;
pub use cache::*;
pub use artifacts::CompilationArtifacts;

/// Kernel trait with the `ComputeShader` that will be compiled and cached based on the
/// provided id.
pub trait RudaTask<C: Compiler>: KernelMetadata + Send + Sync {
    /// Export the shared IR when this task originates from a kernel definition.
    /// Source-only tasks do not have an IR representation.
    fn kernel_definition(&self) -> Option<ruda_core::kernel::KernelDefinition> {
        None
    }

    /// Expand with panic-to-compilation-error conversion on unwind-capable runtimes.
    fn try_kernel_definition(&self) -> Result<Option<ruda_core::kernel::KernelDefinition>, CompilationError> {
        catch_expansion(|| self.kernel_definition())
    }

    /// Compile while retaining code in an active compilation record.
    fn compile_recorded(
        &self,
        compiler: &mut C,
        options: &C::CompilationOptions,
        mode: ExecutionMode,
        address_type: StorageType,
        recording: &mut CompilationRecording,
    ) -> Result<CompiledKernel<C>, CompilationError> {
        let compiled = self.compile(compiler, options, mode, address_type)?;
        recording.source(&compiled.source);
        Ok(compiled)
    }

    /// Compile a kernel and return the compiled form with an optional non-text representation
    fn compile(
        &self,
        compiler: &mut C,
        compilation_options: &C::CompilationOptions,
        mode: ExecutionMode,
        address_type: StorageType,
    ) -> Result<CompiledKernel<C>, CompilationError>;
}

pub(crate) fn catch_expansion<T>(expand: impl FnOnce() -> T) -> Result<T, CompilationError> {
    #[cfg(feature = "runtime-std")]
    {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(expand)).map_err(|payload| {
            let reason = if let Some(text) = payload.downcast_ref::<alloc::string::String>() {
                text.clone()
            } else if let Some(text) = payload.downcast_ref::<&str>() {
                (*text).into()
            } else {
                "non-text kernel expansion panic".into()
            };
            CompilationError::ExpansionPanicked {
                reason,
                backtrace: ruda_core::backtrace::BackTrace::capture(),
            }
        })
    }
    #[cfg(not(feature = "runtime-std"))]
    { Ok(expand()) }
}
