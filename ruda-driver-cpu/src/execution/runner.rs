use super::{
    affinity::{self, CoreId},
    compute_task::ComputeTask,
    progress::{CpuSequence, LaunchCompletion},
    schedule::BindingsResource,
    worker::Worker,
};
use crate::{
    CpuCompiler,
    compilation::{F16Evaluation, MlirCompiler, MlirCompilerOptions},
    memory::arguments::MlirData,
    execution::jit::MlirEngine,
    execution::schedule::ScheduleTask,
};
use ruda_kernel::dsl::{
    RudaDim, ExecutionMode, MemoryConfiguration, ir::MemoryDeviceProperties,
    prelude::CompiledKernel,
};
use ruda::runtime::{
    compiler::{CompilationError, RudaTask},
    id::KernelId,
    logging::ServerLogger,
    memory_management::{MemoryManagement, MemoryManagementOptions},
    storage::BytesStorage,
};
use std::{
    collections::HashMap,
    fmt::Debug,
    sync::{Arc, atomic::AtomicU64},
};
use sysinfo::System;

/// The kernel runner is responsible to manage shared memory as well as threads to execute kernels.
///
/// A single kernel runner is currently used for all kernels.
/// To register work, you have to use the execution queue.
pub struct KernelRunner {
    workers: Vec<Worker>,
    cores: Vec<CoreId>,
    parallel: Arc<AtomicU64>,
    active: Vec<Arc<LaunchCompletion>>,
    compilation_cache: HashMap<(KernelId, F16Evaluation), CpuKernel>,
    compilation_options: MlirCompilerOptions,
    memory_management_shared_memory: MemoryManagement<BytesStorage>,
}

/// A compiled cpu kernel.
pub struct CpuKernel {
    pub(crate) mlir: Arc<CompiledKernel<MlirCompiler>>,
}

impl CpuKernel {
    pub fn new(kernel: CompiledKernel<MlirCompiler>) -> Self {
        Self {
            mlir: Arc::new(kernel),
        }
    }
}

impl Debug for CpuKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuKernel")
            .field("entrypoint_name", &self.mlir.entrypoint_name)
            .field("debug_name", &self.mlir.debug_name)
            .finish()
    }
}

impl Debug for KernelRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", &self.workers)
    }
}

impl KernelRunner {
    pub fn new(logger: Arc<ServerLogger>) -> Self {
        let system = System::new_all();
        let total_memory = system
            .cgroup_limits()
            .map(|g| g.total_memory)
            .unwrap_or(system.total_memory());
        let max_shared_memory_size = total_memory as usize;

        const ALIGNMENT: u64 = 4;
        let memory_properties = MemoryDeviceProperties::new(max_shared_memory_size as u64, ALIGNMENT)
            .with_max_memory(total_memory);

        let memory_management_shared_memory = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &memory_properties,
            MemoryConfiguration::ExclusivePages,
            logger,
            MemoryManagementOptions::new("Shared Memory"),
        );

        let cores = affinity::active_cores();
        let workers = (0..affinity::worker_count(&cores))
            .map(|index| Worker::new(cores.get(index).copied()))
            .collect();

        let compilation_cache = HashMap::new();

        KernelRunner {
            workers,
            cores,
            parallel: Arc::new(AtomicU64::new(0)),
            active: Vec::new(),
            compilation_cache,
            compilation_options: MlirCompilerOptions::for_host(),
            memory_management_shared_memory,
        }
    }
    pub fn prepare(
        &mut self,
        kernel: Box<dyn RudaTask<CpuCompiler>>,
        ruda_count: [u32; 3],
        bindings: BindingsResource,
        kind: ExecutionMode,
    ) -> Result<ScheduleTask, CompilationError> {
        let mut kernel_id = kernel.id();
        kernel_id.mode(kind);
        let mut execution = ruda::runtime::server::KernelExecution::from_declared_bindings(
            kernel_id.clone(), bindings.bindings.clone(), Vec::new(), bindings.declared_io.clone(), None,
        );
        let kernel_id = (kernel_id, self.compilation_options.f16_evaluation);
        let kernel = if let Some(kernel) = self.compilation_cache.get(&kernel_id) {
            kernel
        } else {
            let kernel = kernel.compile(
                &mut Default::default(),
                &self.compilation_options,
                kind,
                kernel.address_type(),
            ).map_err(|error| {
                execution.fail(ruda::runtime::server::LaunchError::CompilationError(error.clone()).into());
                error
            })?;
            if kernel.repr.is_none() {
                let error = CompilationError::UnsupportedInstruction {
                    reason: "CPU execution requires an in-process MLIR engine; source-only kernels cannot be loaded".into(),
                    backtrace: ruda_core::backtrace::BackTrace::capture(),
                };
                execution.fail(ruda::runtime::server::LaunchError::CompilationError(error.clone()).into());
                return Err(error);
            }
            self.compilation_cache
                .insert(kernel_id.clone(), CpuKernel::new(kernel));
            self.compilation_cache
                .get_mut(&kernel_id)
                .expect("Just inserted")
        };

        let ruda_dim = kernel.mlir.ruda_dim;
        execution.set_io(kernel.mlir.io.as_ref());

        let mlir_engine = kernel.mlir.repr.clone().unwrap();

        let task = ScheduleTask::Execute {
            execution,
            mlir_engine,
            bindings,
            kind,
            ruda_dim,
            ruda_count,
        };

        Ok(task)
    }

    pub fn execute_data(
        &mut self,
        mlir_engine: MlirEngine,
        resources: BindingsResource,
        kind: ExecutionMode,
        ruda_dim: RudaDim,
        ruda_count: [u32; 3],
    ) {
        let mut sequence = CpuSequence::default();
        self.dispatch_data(
            mlir_engine, resources, kind, ruda_dim, ruda_count, &mut sequence, None,
        );
        sequence.wait();
        self.reclaim_completed();
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_data(
        &mut self,
        mlir_engine: MlirEngine,
        resources: BindingsResource,
        kind: ExecutionMode,
        ruda_dim: RudaDim,
        ruda_count: [u32; 3],
        sequence: &mut CpuSequence,
        execution: Option<ruda::runtime::server::KernelExecution>,
    ) {
        let ruda_dim_size = ruda_dim.num_elems();
        if ruda_dim_size == 0 || ruda_count.contains(&0) {
            if let Some(execution) = execution { execution.cancel(); }
            return;
        }
        self.reclaim_completed();
        let needs_parallelism = mlir_engine.0.needs_parallelism;

        if needs_parallelism && ruda_dim_size > self.workers.len() as u32 {
            for index in self.workers.len()..ruda_dim_size as usize {
                let core = self.cores.get(index % self.cores.len().max(1)).copied();
                self.workers.push(Worker::new(core));
            }
        }

        let mut mlir_data = MlirData::new(
            resources,
            &mlir_engine.0.shared_memories,
            &mut self.memory_management_shared_memory,
        );
        mlir_data.builtin.set_ruda_dim(ruda_dim);
        mlir_data.builtin.set_ruda_count(ruda_count);

        let completion = sequence.reserve(
            ruda_dim_size, execution, needs_parallelism.then(|| self.parallel.clone()),
        );
        self.active.push(completion.clone());
        let mut index = 0;
        for unit_pos_x in 0..ruda_dim.x {
            for unit_pos_y in 0..ruda_dim.y {
                for unit_pos_z in 0..ruda_dim.z {
                    let unit_pos = [unit_pos_x, unit_pos_y, unit_pos_z];
                    // All units of a barrier launch need distinct workers. The
                    // launch-wide admission owner keeps independently-ready
                    // streams from occupying incompatible halves of that group.
                    let target = if needs_parallelism {
                        index
                    } else {
                        self.workers.iter().enumerate()
                            .min_by_key(|(_, worker)| worker.load())
                            .map(|(index, _)| index).unwrap()
                    };
                    let worker = &mut self.workers[target];
                    let mlir_engine = mlir_engine.clone();
                    let mlir_data = mlir_data.clone();

                    let compute_task = ComputeTask {
                        mlir_engine,
                        mlir_data,
                        unit_pos,
                        kind,
                    };
                    worker.submit(compute_task, completion.clone());
                    index += 1;
                }
            }
        }
    }

    pub(crate) fn flush(&mut self) {
        for launch in &self.active {
            launch.wait();
        }
        self.active.clear();
    }

    pub(crate) fn reclaim_completed(&mut self) {
        self.active.retain(|launch| !launch.is_complete());
    }
}
