
use ruda_core::backtrace::BackTrace;
use ruda::runtime::compiler::CompilationError;
use ruda_compiler::mlir::shared_memories::SharedMemories;
pub use ruda_compiler::mlir::register_supported_types;

use ruda_kernel::dsl::{
    Compiler,
    ir::{self, StorageType},
    prelude::KernelDefinition,
    server::ExecutionMode,
};
use crate::execution::jit::MlirEngine;
use ruda::runtime::config::{RudaRuntimeConfig, RuntimeConfig};
pub use ruda::runtime::config::compilation::F16Evaluation;

mod f16_evaluation;

#[derive(Clone, Debug, Default)]
pub struct MlirCompiler {}

#[derive(Clone, Default, Debug)]
pub struct MlirCompilerOptions {
    pub f16_evaluation: F16Evaluation,
}

impl MlirCompilerOptions {
    pub(crate) fn for_host() -> Self {
        Self {
            f16_evaluation: RudaRuntimeConfig::get().compilation.f16_evaluation
                .unwrap_or_else(|| F16Evaluation::for_native_f16(host_has_f16_arithmetic())),
        }
    }
}

fn host_has_f16_arithmetic() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    { std::arch::is_x86_feature_detected!("avx512fp16") }
    #[cfg(target_arch = "aarch64")]
    { std::arch::is_aarch64_feature_detected!("fp16") }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    { false }
}

impl Compiler for MlirCompiler {
    type Representation = MlirEngine;

    type CompilationOptions = MlirCompilerOptions;

    fn compile(
        &mut self,
        mut kernel: KernelDefinition,
        compilation_options: &Self::CompilationOptions,
        mode: ExecutionMode, // TODO support this by adding array bound checking
        addr_type: StorageType,
    ) -> Result<Self::Representation, CompilationError> {
        kernel.validate_typed_memory().map_err(|reason| CompilationError::Validation { reason, backtrace: BackTrace::capture() })?;
        ir::PointerValueLowering::new(&kernel.body)
            .map_err(|reason| CompilationError::Validation {
                reason,
                backtrace: BackTrace::capture(),
            })?;
        let errors = kernel.body.pop_errors();
        if !errors.is_empty() {
            let mut reason = "Can't compile mlir kernel".to_string();
            for error in errors {
                reason += error.as_str();
                reason += "\n";
            }

            return Err(CompilationError::Validation {
                reason,
                backtrace: BackTrace::capture(),
            });
        }

        f16_evaluation::evaluate(&mut kernel.body, compilation_options.f16_evaluation);

        #[cfg(feature = "mlir-dump")]
        dump_scope(&kernel.body, &kernel.options.kernel_name);
        let mut opt = ruda_kernel::dsl::lowering::mlir::optimize(&kernel, mode);
        opt.lower_pointer_values()
            .map_err(|reason| CompilationError::Validation {
                reason,
                backtrace: BackTrace::capture(),
            })?;

        let mut shared_memories = SharedMemories::default();
        shared_memories.visit(&opt);

        #[cfg(feature = "mlir-dump")]
        dump_opt(&opt, &kernel.options.kernel_name);
        Ok(MlirEngine::from_ruda_ir(
            kernel,
            &opt,
            shared_memories,
            addr_type,
        ))
    }

    fn elem_size(&self, elem: ir::ElemType) -> usize {
        elem.size()
    }

    fn extension(&self) -> &'static str {
        "mlir"
    }

    fn buffer_io(&self, repr: &Self::Representation) -> Option<ruda_core::kernel::KernelIo> {
        Some(repr.0.io.clone())
    }
}

#[cfg(feature = "mlir-dump")]
fn dump_scope(scope: &ruda_kernel::dsl::prelude::Scope, name: &str) {
    use std::fs;

    if let Ok(dir) = std::env::var("RUDA_DEBUG_MLIR") {
        let path = format!("{dir}/{name}");
        let _ = fs::create_dir(&path);
        fs::write(format!("{path}/ruda.ir.txt"), format!("{}", scope)).unwrap();
    }
}

#[cfg(feature = "mlir-dump")]
fn dump_opt(opt: &ruda_compiler::optimizer::Optimizer, name: &str) {
    if let Ok(dir) = std::env::var("RUDA_DEBUG_MLIR") {
        use std::fs;
        let path = format!("{dir}/{name}");
        let _ = fs::create_dir(&path);
        fs::write(format!("{path}/ruda-compiler.ir.txt"), format!("{}", opt)).unwrap();
        fs::write(
            format!("{path}/ruda-compiler.ir.dot"),
            format!("{}", opt.dot_viz()),
        )
        .unwrap();
    }
}
