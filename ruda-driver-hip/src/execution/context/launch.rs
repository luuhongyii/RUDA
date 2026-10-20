use super::*;

impl HipContext {
    /// Executes a task on the given stream.
    pub fn execute_task(
        &mut self,
        stream: &mut Stream,
        kernel_id: KernelId,
        dispatch_count: (u32, u32, u32),
        resources: &[GpuResource],
    ) -> Result<(), LaunchError> {
        let mut bindings = resources
            .iter()
            .map(|memory| memory.binding)
            .collect::<Vec<_>>();

        let kernel = self.module_names.get(&kernel_id).unwrap();
        let ruda_dim = kernel.ruda_dim;

        // SAFETY: `kernel.func` is a valid function handle from a loaded module.
        // `stream.sys` is a valid HIP stream. `bindings` contains valid device pointers
        // for all kernel arguments. The dispatch and ruda dimensions are validated by
        // the caller.
        unsafe {
            let status = ruda_hip_sys::hipModuleLaunchKernel(
                kernel.func,
                dispatch_count.0,
                dispatch_count.1,
                dispatch_count.2,
                ruda_dim.x,
                ruda_dim.y,
                ruda_dim.z,
                // Shared memory is collected into a single buffer, with each shared memory being
                // an offset pointer
                kernel.shared_mem_bytes as u32,
                stream.sys,
                bindings.as_mut_ptr(),
                std::ptr::null_mut(),
            );

            if status != HIP_SUCCESS {
                Err(crate::execution::status::launch("hipModuleLaunchKernel", status))
            } else {
                Ok(())
            }
        }
    }
}
