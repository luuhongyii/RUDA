use super::storage::gpu::{GpuResource, GpuStorage};
use crate::{
    execution::{command::Command, context::HipContext, fence::Fence, stream::HipStreamBackend},
    runtime::HipCompiler,
};
use ruda_core::{bytes::Bytes, future::DynFut, profile::ProfileDuration, stream_id::StreamId};
use ruda_kernel::dsl::{
    MemoryConfiguration,
    backtrace::BackTrace,
    future,
    ir::MemoryDeviceProperties,
    prelude::*,
    server::{
        Binding, CopyDescriptor, KernelArguments, ProfileError, ProfilingToken,
        ServerCommunication, ServerError, ServerUtilities, StreamErrorMode,
    },
};
use ruda::runtime::{
    allocator::PitchedMemoryLayoutPolicy,
    compiler::RudaTask,
    config::{RudaRuntimeConfig, RuntimeConfig},
    logging::ServerLogger,
    memory_management::{ManagedMemoryHandle, MemoryAllocationMode, MemoryUsage},
    server::ComputeServer,
    storage::{ComputeStorage, ManagedResource},
    stream::MultiStream,
};
use std::sync::Arc;

#[derive(Debug)]
pub struct HipServer {
    ctx: HipContext,
    streams: MultiStream<HipStreamBackend>,
    utilities: Arc<ServerUtilities<Self>>,
}

// SAFETY: `HipServer` is only accessed from one thread at a time via the `DeviceHandle`
// (which serializes access through either a mutex or a dedicated runner thread depending
// on the selected channel feature). The HIP context and streams it manages are never
// shared across threads without synchronization.
unsafe impl Send for HipServer {}

impl ComputeServer for HipServer {
    type Kernel = Box<dyn RudaTask<HipCompiler>>;
    type Storage = GpuStorage;
    type MemoryLayoutPolicy = PitchedMemoryLayoutPolicy;
    type Info = ();

    fn logger(&self) -> Arc<ServerLogger> {
        self.streams.logger.clone()
    }

    fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }

    fn staging(&mut self, sizes: &[usize], stream_id: StreamId) -> Result<Vec<Bytes>, ServerError> {
        let mut command = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        )?;

        Ok(sizes
            .iter()
            .map(|size| command.reserve_cpu(*size, true, None))
            .collect())
    }

    fn initialize_memory(&mut self, memory: ManagedMemoryHandle, size: u64, stream_id: StreamId) {
        let mut command = match self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        ) {
            Ok(val) => val,
            Err(err) => {
                memory.binding().set_failure(0, size, err);
                return;
            }
        };

        let reserved = match command.reserve(size) {
            Ok(reserved) => reserved,
            Err(error) => {
                let error: ServerError = error.into();
                memory.binding().set_failure(0, size, error.clone());
                command.error(error);
                return;
            }
        };
        command.bind(reserved, memory);
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        if let Err(error) = Binding::check_all(descriptors.iter().map(|descriptor| &descriptor.handle)) {
            return Box::pin(async move { Err(error) });
        }
        match self.command(
            stream_id,
            descriptors.iter().map(|d| &d.handle),
            StreamErrorMode {
                ignore: false,
                flush: true,
            },
        ) {
            Ok(mut command) => Box::pin(command.read_async(descriptors)),
            Err(err) => Box::pin(async move { Err(err) }),
        }
    }

    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, stream_id: StreamId) {
        for (descriptor, data) in descriptors {
            if data.is_empty() || descriptor.shape.contains(&0) { continue; }
            match self.command(
                stream_id,
                core::iter::once(&descriptor.handle),
                StreamErrorMode { ignore: true, flush: false },
            ) {
                Ok(mut command) => {
                    if let Err(error) = command.write_to_gpu(descriptor, data) {
                        command.error(error.into());
                    }
                }
                Err(error) => {
                    self.ctx.timestamps.error(ProfileError::Server(Box::new(error.clone())));
                    ruda::runtime::server::WriteScope::new(vec![descriptor.handle]).fail(error);
                }
            }
        }
    }

    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: RudaCount,
        bindings: KernelArguments,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) {
        if let Err(err) = self.launch_checked(kernel, count, bindings, mode, stream_id) {
            self.ctx.timestamps.error(ProfileError::Server(Box::new(err.clone())));
        }
    }

    unsafe fn launch_with_mode(
        &mut self,
        kernel: Self::Kernel,
        count: RudaCount,
        bindings: KernelArguments,
        mode: ExecutionMode,
        stream_id: StreamId,
        launch_mode: ruda::runtime::dry_run::LaunchMode,
    ) {
        if !launch_mode.is_skipped() {
            unsafe { self.launch(kernel, count, bindings, mode, stream_id) };
            return;
        }
        let compiled = (|| -> Result<(), ServerError> {
            let mut kernel_id = kernel.id();
            kernel_id.mode(mode);
            self.ctx.refresh_compilation_environment();
            if !self.ctx.module_names.contains_key(&kernel_id) {
                self.ctx.compile_kernel(&kernel_id, kernel, mode, self.streams.logger.clone())
                    .map_err(ServerError::Launch)?;
            }
            Ok(())
        })();
        if let Err(err) = compiled {
            self.ctx.timestamps.error(ProfileError::Server(Box::new(err.clone())));
            log::error!("HIP compile-only launch: {err}");
        }
    }

    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        let mut command = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: false,
                flush: true,
            },
        )?;

        let current = command.streams.current();
        current.drop_queue.flush(|| Fence::new(current.sys));
        current.memory_management_gpu.storage().flush();

        Ok(())
    }

    fn sync(&mut self, stream_id: StreamId) -> DynFut<Result<(), ServerError>> {
        let command = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: false,
                flush: true,
            },
        );

        match command {
            Ok(mut command) => command.sync(),
            Err(err) => Box::pin(async { Err(err) }),
        }
    }

    fn start_profile(&mut self, stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        let stream = self.profiled_stream(stream_id)?;
        self.ctx.timestamps.start(stream)
    }

    fn end_profile(
        &mut self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        let stream = match self.profiled_stream(stream_id) {
            Ok(stream) => stream,
            Err(error) => {
                self.ctx.timestamps.abandon(token);
                return Err(ProfileError::Server(Box::new(error)));
            }
        };
        self.ctx.timestamps.stop(stream, token)
    }

    fn abandon_profile(&mut self, _stream_id: StreamId, token: ProfilingToken) {
        self.ctx.timestamps.abandon(token);
    }

    fn get_resource(
        &mut self,
        binding: Binding,
        stream_id: StreamId,
    ) -> Result<ManagedResource<GpuResource>, ServerError> {
        let mut command = self.command(
            stream_id,
            [&binding].into_iter(),
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        )?;
        let memory = binding.memory.clone();
        let resource = command.resource(binding)?;

        Ok(ManagedResource::new(memory, resource))
    }

    fn memory_usage(&mut self, stream_id: StreamId) -> Result<MemoryUsage, ServerError> {
        let mut command = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: false,
                flush: false,
            },
        )?;
        Ok(command.memory_usage())
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) {
        let mut command = match self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        ) {
            Ok(val) => val,
            // Server is in error.
            Err(_) => return,
        };
        command.memory_cleanup()
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId) {
        let mut command = match self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        ) {
            Ok(val) => val,
            Err(err) => unreachable!("{err:?}"),
        };
        command.allocation_mode(mode)
    }

    fn memory_compact(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        let mut command = self.command_no_inputs(
            stream_id,
            StreamErrorMode { ignore: false, flush: false },
        )?;
        command.memory_compact()?;
        Ok(())
    }
}

impl ServerCommunication for HipServer {
    const SERVER_COMM_ENABLED: bool = false;
}

impl HipServer {
    /// Create a new hip server.
    pub(crate) fn new(
        ctx: HipContext,
        mem_props: MemoryDeviceProperties,
        mem_config: MemoryConfiguration,
        mem_alignment: usize,
        is_integrated: bool,
        utilities: ServerUtilities<Self>,
    ) -> Self {
        let config = RudaRuntimeConfig::get();
        let max_streams = config.streaming.max_streams;

        Self {
            ctx,
            streams: MultiStream::new(
                utilities.logger.clone(),
                HipStreamBackend::new(
                    mem_props,
                    mem_config,
                    mem_alignment,
                    is_integrated,
                    utilities.logger.clone(),
                ),
                max_streams,
            ),
            utilities: Arc::new(utilities),
        }
    }

    fn profiled_stream(&mut self, stream_id: StreamId) -> Result<ruda_hip_sys::hipStream_t, ServerError> {
        let mut command = self.command_no_inputs(stream_id, StreamErrorMode { ignore: false, flush: true })?;
        Ok(command.streams.current().sys)
    }

    fn command_no_inputs(
        &mut self,
        stream_id: StreamId,
        mode: StreamErrorMode,
    ) -> Result<Command<'_>, ServerError> {
        self.command(stream_id, [].into_iter(), mode)
    }

    fn command<'a>(
        &mut self,
        stream_id: StreamId,
        handles: impl Iterator<Item = &'a Binding>,
        mode: StreamErrorMode,
    ) -> Result<Command<'_>, ServerError> {
        let streams = self.streams.resolve(stream_id, handles, !mode.ignore)?;

        Ok(Command::new(self.utilities.service, &mut self.ctx, streams))
    }

    fn launch_checked(
        &mut self,
        kernel: Box<dyn RudaTask<HipCompiler>>,
        count: RudaCount,
        bindings: KernelArguments,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) -> Result<(), ServerError> {
        let mut kernel_id = kernel.id();
        let logger = self.streams.logger.clone();
        kernel_id.mode(mode);
        let mut execution = ruda::runtime::server::KernelExecution::new(kernel_id.clone(), &bindings, &count);
        let result = (|| -> Result<(), ServerError> {
        let mut command = self.command(
            stream_id,
            bindings.buffers.iter().chain(bindings.tensor_maps.iter().map(|map| &map.binding))
                .chain(match &count { RudaCount::Dynamic(binding) => Some(binding), _ => None }),
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        )?;
        let io = command.prepare_kernel(&kernel_id, kernel, mode, logger)?;
        execution.set_io(io.as_ref());
        execution.check()?;

        let count = match count {
            RudaCount::Static(x, y, z) => (x, y, z),
            // TODO: HIP doesn't have an exact equivalent of dynamic dispatch. Instead, kernels are free to launch other kernels.
            // One option is to create a dummy kernel with 1 thread that launches the real kernel with the dynamic dispatch settings.
            // For now, just read the dispatch settings from the buffer.
            RudaCount::Dynamic(binding) => {
                let data = future::block_on(command.read_async(vec![CopyDescriptor::new(
                    binding,
                    [3].into(),
                    [1].into(),
                    4,
                )]))?;
                let data = bytemuck::cast_slice(&data[0]);
                assert!(
                    data.len() == 3,
                    "Dynamic ruda count should contain 3 values"
                );
                (data[0], data[1], data[2])
            }
        };
        if count.0 == 0 || count.1 == 0 || count.2 == 0 {
            execution.skip();
            return Ok(());
        }
        execution.enter();

        let KernelArguments {
            buffers,
            info,
            tensor_maps,
            ..
        } = bindings;

        debug_assert!(tensor_maps.is_empty(), "Can't use tensor maps on HIP");

        let info = command
            .create_with_data(bytemuck::cast_slice(&info.data))?;

        let mut resources: Vec<_> = buffers
            .into_iter()
            .map(|b| command.resource(b))
            .collect::<Result<_, _>>()?;

        resources.push(
            command
                .resource(info.binding())?,
        );

        command.kernel(kernel_id, count, &resources)?;

        Ok(())
        })();
        execution.finish(result)
    }

    pub(crate) fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }
}
