use crate::memory::arguments::MlirData;

use super::symbols::register_external_function;
use super::barrier::BarrierGuard;
use ruda_compiler::mlir::shared_memories::SharedMemories;
use ruda_compiler::optimizer::Optimizer;

use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use ruda_compiler::mlir::module::Module;
use ruda_kernel::dsl::{ir::StorageType, prelude::KernelDefinition};
use melior::{
    Context, ExecutionEngine,
    dialect::DialectRegistry,
    utility::{register_all_dialects, register_all_llvm_translations, register_all_passes},
};

pub struct MlirKernel {
    pub io: ruda_core::kernel::KernelIo,
    execution_engine: ExecutionEngine,
    pub shared_memories: SharedMemories,
    pub needs_parallelism: bool,
}

unsafe impl Send for MlirKernel {}
unsafe impl Sync for MlirKernel {}

#[derive(Clone)]
pub struct MlirEngine(pub Arc<MlirKernel>);

impl Debug for MlirEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Work in progress")
    }
}

impl Display for MlirEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "work in progress")
    }
}

impl MlirEngine {
    pub fn from_ruda_ir(
        kernel: KernelDefinition,
        opt: &Optimizer,
        shared_memories: SharedMemories,
        addr_type: StorageType,
    ) -> Self {
        kernel.validate_typed_memory().expect("valid typed memory kernel");
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        register_all_passes();

        let context = Context::new();
        register_all_llvm_translations(&context);
        context.enable_multi_threading(false);
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();

        let mut module = Module::new(&context, kernel.options.kernel_name.clone());

        module.visit_kernel(&kernel, opt, &shared_memories, addr_type);
        let needs_parallelism = module.needs_parallelism();

        module.run_pass();

        let execution_engine = module.into_execution_engine();
        register_external_function(&execution_engine);
        let kernel = MlirKernel {
            io: opt.buffer_io(&kernel.buffers, &kernel.tensor_maps),
            execution_engine,
            shared_memories,
            needs_parallelism,
        };
        let mlir_kernel = Arc::new(kernel);
        Self(mlir_kernel)
    }

    pub fn dump_object(&self, path: &str) {
        self.0.execution_engine.dump_to_object_file(path);
    }

    /// # Safety
    /// MLIR kernel needs valid reference to memory and will segfault if bad pointer are sent.
    #[inline(always)]
    pub unsafe fn run_kernel(&mut self, mlir_data: &mut MlirData) {
        let units = mlir_data.builtin.dims[..3].iter().product();
        let _barrier = BarrierGuard::enter(mlir_data.shared_mlir_data.barrier.clone(), units);
        unsafe {
            self.0
                .execution_engine
                .invoke_packed("kernel", &mut mlir_data.args_second_indirection)
                .unwrap_or_else(|err| {
                    log::error!("MLIR kernel invocation failed: {err}");
                    panic!("{err}");
                });
        }
    }
}
