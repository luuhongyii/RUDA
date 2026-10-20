use super::*;
use ruda_compiler::ptx::{PtxCompilationOptions, PtxCompiler, PtxTarget};
use ruda_core::compiler::Compiler;

impl CudaContext {
    pub(super) fn compile_direct_ptx(
        &mut self,
        kernel_id: &KernelId,
        kernel: &dyn RudaTask<CudaCompiler>,
        mode: ExecutionMode,
        logger: Arc<ServerLogger>,
        hash: Option<StableHash>,
        version: (u32, u32),
        mut recording: CompilationRecording,
    ) -> Result<(), LaunchError> {
        let definition = kernel.try_kernel_definition()?.ok_or_else(|| CompilationError::UnsupportedInstruction {
            reason: "Direct PTX requires Kernel IR; source-only tasks cannot be compiled by this backend".into(),
            backtrace: BackTrace::capture(),
        })?;
        recording.defined(&definition);
        let representation = PtxCompiler.compile(
            definition,
            &PtxCompilationOptions {
                target: Some(PtxTarget {
                    version,
                    sm: self.arch.version,
                }),
            },
            mode,
            kernel.address_type(),
        )?;
        let shared_mem_bytes = representation.shared_memory_bytes;
        let dynamic_metadata_index = representation.dynamic_metadata_index;
        let max = self.properties.hardware.max_shared_memory_size;
        if shared_mem_bytes > max {
            return Err(ResourceLimitError::SharedMemory {
                requested: shared_mem_bytes,
                max,
                backtrace: BackTrace::capture(),
            }
            .into());
        }
        let compiled = ruda::runtime::kernel::CompiledKernel::<PtxCompiler> {
            io: Some(representation.io.clone()),
            entrypoint_name: representation.entrypoint.clone(),
            debug_name: Some(kernel.name()),
            source: representation.source.clone(),
            ruda_dim: representation.ruda_dim,
            repr: Some(representation),
            debug_info: logger
                .compilation_activated()
                .then(|| DebugInformation::new("ptx", kernel_id.clone())),
        };
        logger.log_compilation(&compiled);
        recording.source(&compiled.source);
        if let Some((entry, stored)) = self.artifacts.rekey(kernel_id, &compiled.source) {
            self.load_ptx(
                entry.ptx, kernel_id.clone(), entry.entrypoint_name, compiled.ruda_dim,
                entry.shared_mem_bytes, entry.dynamic_metadata_index,
                compiled.io.clone(),
            )?;
            recording.rekeyed(stored);
            return Ok(());
        }
        let ptx: Vec<c_char> = CString::new(compiled.source.as_str())
            .map_err(|err| CompilationError::Validation {
                reason: format!("Invalid PTX source: {err}"),
                backtrace: BackTrace::capture(),
            })?
            .into_bytes_with_nul()
            .into_iter()
            .map(|byte| byte as c_char)
            .collect();
        let stored = self.artifacts.enabled() && self.artifacts.insert(kernel_id, &compiled.source, PtxCacheEntry {
            io: compiled.io.clone(),
            entrypoint_name: compiled.entrypoint_name.clone(),
            shared_mem_bytes,
            ptx: ptx.clone(),
            dynamic_metadata_index,
        });
        if let (Some(cache), Some(hash)) = (&mut self.ptx_cache, hash) {
            if let Err(err) = cache.insert(
                hash,
                PtxCacheEntry {
                    io: compiled.io.clone(),
                    entrypoint_name: compiled.entrypoint_name.clone(),
                    shared_mem_bytes,
                    ptx: ptx.clone(),
                    dynamic_metadata_index,
                },
            ) {
                log::warn!("Unable to save the direct PTX {err:?}");
            }
        }
        self.load_ptx(
            ptx,
            kernel_id.clone(),
            compiled.entrypoint_name,
            compiled.ruda_dim,
            shared_mem_bytes,
            dynamic_metadata_index,
            compiled.io,
        )?;
        recording.compiled(stored);
        Ok(())
    }
}
