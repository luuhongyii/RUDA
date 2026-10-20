use alloc::{
    boxed::Box,
    string::{String, ToString},
};
use core::{
    fmt::Display,
    marker::PhantomData,
    sync::atomic::{AtomicI8, Ordering},
};

use ruda_core::format::format_str;
use ruda_core::ir::StorageType;

use crate::runtime::{
    compiler::{CompilationError, Compiler, RudaTask},
    config::{RudaRuntimeConfig, RuntimeConfig, compilation::CompilationLogLevel},
    id::KernelId,
    server::{RudaDim, ExecutionMode},
};

/// Implement this trait to create a [kernel definition](KernelDefinition).
pub trait KernelMetadata: Send + Sync + 'static {
    /// Name of the kernel for debugging.
    fn name(&self) -> &'static str {
        core::any::type_name::<Self>()
    }

    /// Identifier for the kernel, used for caching kernel compilation.
    fn id(&self) -> KernelId;

    /// Type of addresses in this kernel
    fn address_type(&self) -> StorageType;
}

pub use ruda_core::kernel::{BufferIOAttr, KernelIo, KernelArg, KernelDefinition, KernelOptions, ScalarKernelArg, Visibility};

/// A kernel, compiled in the target language
pub struct CompiledKernel<C: Compiler> {
    /// Compiled payload accesses, falling back to launch declarations when absent.
    pub io: Option<ruda_core::kernel::KernelIo>,
    /// The name of the kernel entrypoint.
    /// For example
    ///
    /// ```text
    /// #[ruda(launch)]
    /// fn gelu_array<F: Float, R: Runtime>() {}
    /// ```
    ///
    /// would have the entrypoint name "`gelu_array`".
    pub entrypoint_name: String,

    /// A fully qualified debug name of the kernel.
    ///
    /// For example
    ///
    /// ```text
    /// #[ruda(launch)]
    /// fn gelu_array<F: Float, R: Runtime>() {}
    /// ```
    ///
    /// would have a debug name such as
    ///
    /// ```text
    /// gelu::gelu_array::GeluArray<
    ///    ruda_kernel::dsl::frontend::element::float::F32,
    ///    ruda_driver_cuda::runtime::CudaRuntime,
    /// >
    /// ```
    pub debug_name: Option<&'static str>,

    /// Source code of the kernel
    pub source: String,
    /// In-memory representation of the kernel
    pub repr: Option<C::Representation>,
    /// Size of a ruda for the compiled kernel
    pub ruda_dim: RudaDim,
    /// Extra debugging information about the compiled kernel.
    pub debug_info: Option<DebugInformation>,
}

/// Extra debugging information about the compiled kernel.
#[derive(new)]
pub struct DebugInformation {
    /// The language tag of the source..
    pub lang_tag: &'static str,
    /// The compilation id.
    pub id: KernelId,
}

/// Kernel that can be defined
pub trait RudaKernel: KernelMetadata {
    /// Define the kernel for compilation
    fn define(&self) -> KernelDefinition;

    /// Target-language source instead of compiling the definition's body.
    fn source(&self) -> Option<PrecompiledSource> { None }
}

/// Hand-written source. The kernel ID must distinguish different source contents.
pub struct PrecompiledSource {
    /// Source in the target compiler's language.
    pub source: String,
    /// Entrypoint exported by the source.
    pub entrypoint_name: String,
    /// Must match the selected compiler's language tag.
    pub lang: &'static str,
}

impl KernelMetadata for Box<dyn RudaKernel> {
    fn name(&self) -> &'static str { self.as_ref().name() }
    fn id(&self) -> KernelId { self.as_ref().id() }
    fn address_type(&self) -> StorageType { self.as_ref().address_type() }
}

impl RudaKernel for Box<dyn RudaKernel> {
    fn define(&self) -> KernelDefinition { self.as_ref().define() }
    fn source(&self) -> Option<PrecompiledSource> { self.as_ref().source() }
}

/// Wraps a [`RudaKernel`] to allow it be compiled.
pub struct KernelTask<C: Compiler, K: RudaKernel> {
    kernel_definition: K,
    _compiler: PhantomData<fn() -> C>,
}

/// Generic [`RudaTask`] for compiling kernels
pub struct RudaTaskKernel<C: Compiler> {
    /// The inner compilation task being wrapped
    pub task: Box<dyn RudaTask<C>>,
}

impl<C: Compiler, K: RudaKernel> KernelTask<C, K> {
    /// Create a new kernel task
    pub fn new(kernel_definition: K) -> Self {
        Self {
            kernel_definition,
            _compiler: PhantomData,
        }
    }

    fn compile_definition(
        &self,
        definition: KernelDefinition,
        compiler: &mut C,
        options: &C::CompilationOptions,
        mode: ExecutionMode,
        addr_type: StorageType,
    ) -> Result<CompiledKernel<C>, CompilationError> {
        let entrypoint_name = definition.options.kernel_name.clone();
        let ruda_dim = definition.ruda_dim;
        if let Some(source) = self.kernel_definition.source() {
            if source.lang != compiler.lang_tag() {
                return Err(CompilationError::Generic {
                    reason: alloc::format!("kernel `{}` carries {} source, but this compiler expects {}",
                        self.name(), source.lang, compiler.lang_tag()),
                    backtrace: ruda_core::backtrace::BackTrace::capture(),
                });
            }
            return Ok(CompiledKernel {
                io: None,
                entrypoint_name: source.entrypoint_name,
                debug_name: Some(self.kernel_definition.name()),
                source: source.source,
                repr: None,
                ruda_dim,
                debug_info: None,
            });
        }
        let repr = compiler.compile(definition, options, mode, addr_type)?;
        Ok(CompiledKernel {
            io: compiler.buffer_io(&repr),
            entrypoint_name,
            debug_name: Some(self.kernel_definition.name()),
            source: repr.to_string(),
            repr: Some(repr),
            ruda_dim,
            debug_info: None,
        })
    }
}

impl<C: Compiler, K: RudaKernel> RudaTask<C> for KernelTask<C, K> {
    fn kernel_definition(&self) -> Option<KernelDefinition> {
        if self.kernel_definition.source().is_some() { None } else { Some(self.kernel_definition.define()) }
    }

    fn compile(
        &self,
        compiler: &mut C,
        compilation_options: &C::CompilationOptions,
        mode: ExecutionMode,
        addr_type: StorageType,
    ) -> Result<CompiledKernel<C>, CompilationError> {
        let gpu_ir = crate::runtime::compiler::catch_expansion(|| self.kernel_definition.define())?;
        self.compile_definition(gpu_ir, compiler, compilation_options, mode, addr_type)
    }

    fn compile_recorded(
        &self,
        compiler: &mut C,
        options: &C::CompilationOptions,
        mode: ExecutionMode,
        addr_type: StorageType,
        recording: &mut crate::runtime::compiler::CompilationRecording,
    ) -> Result<CompiledKernel<C>, CompilationError> {
        let definition = crate::runtime::compiler::catch_expansion(|| self.kernel_definition.define())?;
        recording.defined(&definition);
        let compiled = self.compile_definition(definition, compiler, options, mode, addr_type)?;
        recording.source(&compiled.source);
        Ok(compiled)
    }
}

impl<C: Compiler, K: RudaKernel> KernelMetadata for KernelTask<C, K> {
    // Forward ID to underlying kernel definition.
    fn id(&self) -> KernelId {
        self.kernel_definition.id()
    }

    // Forward name to underlying kernel definition.
    fn name(&self) -> &'static str {
        self.kernel_definition.name()
    }

    fn address_type(&self) -> StorageType {
        self.kernel_definition.address_type()
    }
}

impl<C: Compiler> KernelMetadata for Box<dyn RudaTask<C>> {
    // Deref and use existing ID.
    fn id(&self) -> KernelId {
        self.as_ref().id()
    }

    // Deref and use existing name.
    fn name(&self) -> &'static str {
        self.as_ref().name()
    }

    fn address_type(&self) -> StorageType {
        self.as_ref().address_type()
    }
}

static COMPILATION_LEVEL: AtomicI8 = AtomicI8::new(-1);

fn compilation_level() -> u8 {
    let compilation_level = COMPILATION_LEVEL.load(Ordering::Relaxed);
    if compilation_level == -1 {
        let val = match RudaRuntimeConfig::get().compilation.logger.level {
            CompilationLogLevel::Full => 2,
            CompilationLogLevel::Disabled => 0,
            CompilationLogLevel::Basic => 1,
        };

        COMPILATION_LEVEL.store(val, Ordering::Relaxed);
        val as u8
    } else {
        compilation_level as u8
    }
}

impl<C: Compiler> Display for CompiledKernel<C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match compilation_level() {
            2 => self.format_full(f),
            _ => self.format_basic(f),
        }
    }
}

impl<C: Compiler> CompiledKernel<C> {
    fn format_basic(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("[Compiling kernel]")?;
        if let Some(name) = self.debug_name {
            if name.len() <= 32 {
                f.write_fmt(format_args!(" {name}"))?;
            } else {
                f.write_fmt(format_args!(" {}", name.split('<').next().unwrap_or("")))?;
            }
        }

        Ok(())
    }

    fn format_full(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("[START_KERNEL_COMPILATION]")?;

        if let Some(name) = self.debug_name {
            if name.len() <= 32 {
                f.write_fmt(format_args!("\nname: {name}"))?;
            } else {
                let name = format_str(name, &[('<', '>')], false);
                f.write_fmt(format_args!("\nname: {name}"))?;
            }
        }

        if let Some(info) = &self.debug_info {
            f.write_fmt(format_args!("\nid: {:#?}", info.id))?;
        }

        f.write_fmt(format_args!(
            "
source:
```{}
{}
```
[END_KERNEL_COMPILATION]
",
            self.debug_info
                .as_ref()
                .map(|info| info.lang_tag)
                .unwrap_or(""),
            self.source
        ))
    }
}
