use super::*;

impl CpuServer {
    pub(super) fn prepare_bindings(&mut self, bindings: KernelArguments) -> Result<BindingsResource, IoError> {
        // Store all the resources we'll be using. This could be eliminated if
        // there was a way to tie the lifetime of the resource to the memory handle.
        let resources = bindings
            .buffers
            .iter()
            .map(|binding| {
                let stream = self.scheduler.stream(&binding.stream);
                stream
                    .memory_management
                    .get_resource(binding.memory.clone(), binding.offset_start, binding.offset_end)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(BindingsResource {
            bindings: bindings.buffers,
            declared_io: bindings.declared_io,
            resources,
            info: bindings.info,
        })
    }

    pub(super) fn prepare_task(
        &mut self,
        kernel: Box<dyn RudaTask<CpuCompiler>>,
        count: RudaCount,
        bindings: KernelArguments,
        kind: ExecutionMode,
        mut execution: ruda::runtime::server::KernelExecution,
    ) -> Result<Option<ScheduleTask>, ServerError> {
        let kernel = match self.prepare_kernel(kernel, kind) {
            Ok(kernel) => kernel,
            Err(error) => return Err(execution.fail(
                ruda::runtime::server::LaunchError::CompilationError(error).into(),
            )),
        };
        execution.set_io(kernel.mlir.io.as_ref());
        let ruda_dim = kernel.mlir.ruda_dim;
        let mlir_engine = kernel.mlir.repr.clone().unwrap();
        execution.check()?;
        let ruda_count = match count {
            RudaCount::Static(x, y, z) => [x, y, z],
            RudaCount::Dynamic(binding) => {
                // The grid is a host read: submit its producers before inspecting it.
                self.scheduler.execute_streams(vec![binding.stream]);
                let stream = self.scheduler.stream(&binding.stream);
                if let Err(error) = stream.sync() { return Err(execution.fail(error)); }
                let resource = match stream
                    .memory_management
                    .get_resource(binding.memory, binding.offset_start, binding.offset_end)
                {
                    Ok(resource) => resource,
                    Err(error) => return Err(execution.fail(error.into())),
                };
                let bytes = resource.read();
                let Some(data) = bytes.get(..12) else {
                    return Err(execution.fail(IoError::NotFound {
                        reason: "Indirect dispatch requires three u32 values".into(),
                        backtrace: BackTrace::capture(),
                    }.into()));
                };
                [
                    u32::from_ne_bytes(data[0..4].try_into().unwrap()),
                    u32::from_ne_bytes(data[4..8].try_into().unwrap()),
                    u32::from_ne_bytes(data[8..12].try_into().unwrap()),
                ]
            }
        };
        if ruda_count.contains(&0) {
            execution.cancel();
            return Ok(None);
        }
        let bindings = match self.prepare_bindings(bindings) {
            Ok(bindings) => bindings,
            Err(error) => return Err(execution.fail(error.into())),
        };

        Ok(Some(ScheduleTask::Execute {
            execution,
            mlir_engine,
            bindings,
            kind,
            ruda_dim,
            ruda_count,
        }))
    }

    pub(super) fn prepare_kernel(
        &mut self,
        kernel: Box<dyn RudaTask<CpuCompiler>>,
        kind: ExecutionMode,
    ) -> Result<&CpuKernel, CompilationError> {
        let mut kernel_id = kernel.id();
        kernel_id.mode(kind);
        let kernel_id = (kernel_id, self.compilation_options.f16_evaluation);
        if !self.compilation_cache.contains_key(&kernel_id) {
            let kernel = kernel.compile(
                &mut Default::default(),
                &self.compilation_options,
                kind,
                kernel.address_type(),
            )?;
            if kernel.repr.is_none() {
                return Err(CompilationError::UnsupportedInstruction {
                    reason: "CPU execution requires an in-process MLIR engine; source-only kernels cannot be loaded".into(),
                    backtrace: BackTrace::capture(),
                });
            }
            self.compilation_cache
                .insert(kernel_id.clone(), CpuKernel::new(kernel));
        }
        Ok(self.compilation_cache.get(&kernel_id).expect("Compiled kernel"))
    }
}
