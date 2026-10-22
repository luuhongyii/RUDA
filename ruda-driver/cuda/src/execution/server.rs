mod interop;
mod graph;
mod capture;
mod collective;
use interop::InteropState;
use super::storage::gpu::{GpuResource, GpuStorage};
use crate::{
    CudaCompiler,
    execution::{
        command::Command,
        communication::{get_nccl_comm_id, get_nccl_dtype_count, to_nccl_op},
        context::CudaContext,
        stream::CudaStreamBackend,
        sync::Fence,
    },
};
use ruda_core::{
    backtrace::BackTrace, bytes::Bytes, profile::ProfileDuration, stream_id::StreamId,
};
use ruda_kernel::dsl::{
    MemoryConfiguration,
    device::DeviceId,
    future::{self, DynFut},
    ir::{ElemType, FloatKind, IntKind, MemoryDeviceProperties, StorageType, UIntKind},
    prelude::*,
    server::{
        Binding, CommunicationId, CopyDescriptor, Handle, KernelArguments, LaunchError,
        ProfileError, ProfilingToken, ReduceOperation, ServerCommunication, ServerError,
        ServerUtilities, StreamErrorMode, TensorMapBinding, TensorMapMeta,
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
use cudarc::driver::sys::{
    CUstream_st, CUtensorMapDataType, CUtensorMapFloatOOBfill, CUtensorMapInterleave,
    CUtensorMapL2promotion, CUtensorMapSwizzle, cuTensorMapEncodeIm2col, cuTensorMapEncodeTiled,
};
use std::{
    collections::{HashMap, hash_map::Entry},
    ffi::c_void,
    mem::MaybeUninit,
    sync::Arc,
};

pub(crate) const MB: usize = 1024 * 1024;

#[derive(Debug)]
pub struct CudaServer {
    captures: crate::execution::capture::Captures,
    // Graphs must release executables/buffer pins BEFORE the context unloads modules.
    graphs: graph::GraphRegistry,
    ctx: CudaContext,
    interop: InteropState,
    device_id: DeviceId,
    streams: MultiStream<CudaStreamBackend>,
    utilities: Arc<ServerUtilities<Self>>,
    comm_stream: *mut CUstream_st,
    communicators: HashMap<CommunicationId, *mut cudarc::nccl::sys::ncclComm>,
}

// SAFETY: `CudaServer` is only accessed from one thread at a time via the `DeviceHandle`,
// which serializes all server access. The CUDA context, streams, and NCCL communicators
// it manages are never shared across threads without synchronization.
unsafe impl Send for CudaServer {}

impl CudaServer {
    pub(crate) fn finish_for_shutdown(&mut self) {
        let captures: Vec<_> = self.captures.active.iter().chain(&self.captures.preparing).copied().collect();
        for stream in captures {
            let _ = self.capture_stop(stream, true);
        }
        // The service queue and every upstream producer have already drained.
        // This is process teardown, not an operator or Backend::flush fence.
        unsafe {
            use cudarc::driver::sys::*;
            assert_eq!(cuCtxSetCurrent(self.ctx.context), CUresult::CUDA_SUCCESS,
                "CUDA shutdown context selection failed");
            assert_eq!(cuCtxSynchronize(), CUresult::CUDA_SUCCESS,
                "CUDA shutdown synchronization failed");
        }
    }
}

impl ComputeServer for CudaServer {
    type Kernel = Box<dyn RudaTask<CudaCompiler>>;
    type Storage = GpuStorage;
    type MemoryLayoutPolicy = PitchedMemoryLayoutPolicy;
    type Info = ();

    fn logger(&self) -> Arc<ServerLogger> {
        self.streams.logger.clone()
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

    fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
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

    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, stream_id: StreamId) {
        for (descriptor, data) in descriptors {
            if data.is_empty() || descriptor.shape.contains(&0) { continue; }
            let failed = match self.command(
                stream_id,
                core::iter::once(&descriptor.handle),
                StreamErrorMode { ignore: true, flush: false },
            ) {
                Ok(mut command) => {
                    if let Err(error) = command.write_to_gpu(descriptor, data) {
                        command.error(error.into());
                    }
                    None
                }
                Err(error) => Some((descriptor.handle, error)),
            };
            if let Some((handle, error)) = failed {
                self.ctx.timestamps.error(ProfileError::Server(Box::new(error.clone())));
                ruda::runtime::server::WriteScope::new(vec![handle]).fail(error);
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
        if let Err(err) = self.launch_checked(kernel, count, bindings, mode, stream_id, None) {
            self.ctx.timestamps.error(ProfileError::Server(Box::new(err.clone())));
            if self.capture_owner(stream_id).is_some_and(|owner| owner != stream_id) { return; }
            let mut stream = match self.streams.resolve(stream_id, [].into_iter(), false) {
                Ok(stream) => stream,
                Err(error) => { log::error!("CUDA launch reporting: {error}"); return; }
            };
            let stream = stream.current();
            if stream.capture.recording && stream.capture.owner == Some(stream_id) && stream.capture.failed.is_none() {
                stream.capture.failed = Some(err.clone());
            }
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
            self.set_current_checked()?;
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
            log::error!("CUDA compile-only launch: {err}");
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
        if current.capture.recording { return Ok(()); }
        current.drop_queue.flush(|| Fence::new(current.sys));
        current.memory_management_gpu.storage().flush();

        Ok(())
    }

    fn graph_prepare(&mut self, stream: StreamId) -> Result<(), ServerError> { self.capture_prepare(stream) }
    fn graph_start_capture(&mut self, stream: StreamId) -> Result<(), ServerError> { self.capture_start(stream) }
    fn graph_stop_capture(&mut self, stream: StreamId) -> Result<u64, ServerError> { self.capture_stop(stream, false) }
    fn graph_abort_capture(&mut self, stream: StreamId) { let _ = self.capture_stop(stream, true); }
    unsafe fn graph_replay(&mut self, stream: StreamId, graph: u64) -> Result<(), ServerError> { self.capture_replay(stream, graph) }
    fn graph_release(&mut self, _stream: StreamId, graph: u64) { self.capture_release(graph); }

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
            Err(err) => unreachable!("{err:?}"),
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

impl ServerCommunication for CudaServer {
    const SERVER_COMM_ENABLED: bool = true;

    fn comm_init(&mut self, device_ids: Vec<DeviceId>) -> Result<(), ServerError> {
        let id = CommunicationId::from(device_ids.clone());
        if let Entry::Vacant(e) = self.communicators.entry(id.clone()) {
            let mut comm = MaybeUninit::uninit();
            let mut device_ids = device_ids.clone();
            device_ids.sort();
            let rank = device_ids
                .iter()
                .position(|id| id.index_id == self.device_id.index_id)
                .expect("Device's peer id should be in the list of device ids.");
            let nccl_comm_id = get_nccl_comm_id(device_ids.clone());

            // SAFETY: `comm` is a valid `MaybeUninit`. `nccl_comm_id` is a unique communicator ID
            // shared across all participating ranks. `rank` is this device's position in the
            // group. `comm_init_rank` initializes the communicator, making `assume_init` valid.
            unsafe {
                cudarc::nccl::result::comm_init_rank(
                    comm.as_mut_ptr(),
                    device_ids.len() as i32,
                    nccl_comm_id,
                    rank as i32,
                )
                .map_err(|e| ServerError::Generic {
                    reason: format!("NCCL comm_init_rank failed: {e:?}"),
                    backtrace: BackTrace::capture(),
                })?;
                e.insert(comm.assume_init());
            }

            let mut initialized_comms = self.utilities.initialized_comms.write().unwrap();
            initialized_comms.insert(id);
        }

        Ok(())
    }

    fn all_reduce(
        &mut self,
        src: Binding,
        dst: Binding,
        dtype: ElemType,
        stream_id: StreamId,
        op: ReduceOperation,
        device_ids: Vec<DeviceId>,
    ) -> Result<(), ServerError> {
        let writes = [dst.clone()];
        let result = (|| {
        if let Some(error) = src.failure().or_else(|| dst.failure()) { return Err(error); }
        // We create a command on the server to retrieve the correct resource of the source and the destination
        // from the memory pools.
        if src.stream != dst.stream {
            for stream in [src.stream, dst.stream].iter() {
                let mut command = self.command_no_inputs(
                    *stream,
                    StreamErrorMode {
                        ignore: false,
                        flush: false,
                    },
                )?;
                command.error(ServerError::Generic {
                    reason: "Source and destination should be on the same stream.".into(),
                    backtrace: BackTrace::capture(),
                });
            }
        }

        let mut command_src = self.command(
            stream_id,
            [&src, &dst].into_iter(),
            StreamErrorMode {
                ignore: false,
                flush: false,
            },
        )?;
        command_src.streams.current().capture.record_write(&dst);
        let resource_src = command_src.resource(src)?;
        let resource_dst = command_src.resource(dst)?;

        let stream = command_src.streams.current().sys;

        // We need to free the command before accessing communicators.
        core::mem::drop(command_src);

        // Wait for data to be ready on compute stream.
        let communication = self.collective_stream(stream_id, stream)?;

        // Get the communicator.
        let comm = self
            .communicators
            .get(&CommunicationId::from(device_ids))
            .ok_or_else(|| graph::error("Collective communicator is not initialized"))?;

        // Perform the `cudarc::nccl::result::all_reduce` operation.
        let (nccl_dtype, count) = get_nccl_dtype_count(dtype, resource_src.size);
        // SAFETY: `resource_src.ptr` and `resource_dst.ptr` are valid device pointers.
        // `comm` is a valid NCCL communicator initialized via `comm_init_rank`.
        // `communication` is the ordinary or capture-owned collective stream.

        unsafe {
            cudarc::nccl::result::all_reduce(
                resource_src.ptr as *const _,
                resource_dst.ptr as *mut _,
                count,
                nccl_dtype,
                to_nccl_op(op),
                *comm,
                communication as _,
            )
            .map_err(|e| ServerError::Generic {
                reason: format!("NCCL all_reduce failed: {e:?}"),
                backtrace: BackTrace::capture(),
            })?;
        }

        Ok(())
        })();
        self.collective_result(stream_id, &writes, result)
    }

    fn sync_collective(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        let result = (|| {
        let mut command = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        )?;
        let stream = command.streams.current().sys;

        drop(command);

        self.collective_sync_stream(stream_id, stream)
        })();
        self.collective_result(stream_id, &[], result)
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(desc)))]
    fn send(
        &mut self,
        desc: CopyDescriptor,
        dtype: ElemType,
        stream_id: StreamId,
        device_id_dst: DeviceId,
    ) -> Result<(), ServerError> {
        let result = (|| {
        if let Some(error) = desc.handle.failure() { return Err(error); }
        let binding = desc.handle.clone();

        // We create a command on the source server to retrieve the correct resource from the
        // source memory pools. We also make sure the current stream is aligned with the stream of
        // the binding, where the data was first allocated.
        let mut command = self.command(
            stream_id,
            [&desc.handle].into_iter(),
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        )?;
        let resource = command.resource(binding.clone())?;
        let stream = command.streams.current().sys;

        // We need to free the command before creating another one.
        core::mem::drop(command);

        // Wait for data to be ready on compute stream.
        let communication = self.collective_stream(stream_id, stream)?;

        // Get the communicator.
        let mut device_ids = vec![device_id_dst, self.device_id];
        device_ids.sort();
        let comm_id = CommunicationId::from(device_ids.clone());
        let comm = self
            .communicators
            .get(&comm_id)
            .ok_or_else(|| graph::error("Send communicator is not initialized"))?;

        let rank_dst = device_ids
            .iter()
            .position(|id| id.index_id != self.device_id.index_id)
            .unwrap() as i32;

        // Perform the `send` operation.
        let (nccl_dtype, count) = get_nccl_dtype_count(dtype, resource.size);
        // SAFETY: `resource.ptr` is a valid device pointer.
        // `comm` is a valid NCCL communicator initialized via `comm_init_rank`.
        // `communication` is the ordinary or capture-owned collective stream.
        unsafe {
            cudarc::nccl::result::send(
                resource.ptr as *const _,
                count,
                nccl_dtype,
                rank_dst,
                *comm,
                communication as _,
            )
            .map_err(|e| ServerError::Generic {
                reason: format!("NCCL send failed: {e:?}"),
                backtrace: BackTrace::capture(),
            })?;
        }

        Ok(())
        })();
        self.collective_result(stream_id, &[], result)
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace"))]
    fn recv(
        &mut self,
        handle: Handle,
        dtype: ElemType,
        stream_id: StreamId,
        device_id_src: DeviceId,
    ) -> Result<(), ServerError> {
        let writes = [handle.clone().binding()];
        let result = (|| {
        // We create a new command on the destination server to reserve the necessary GPU memory.
        let mut command_dst = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        )?;

        command_dst.streams.current().capture.record_write(&writes[0]);
        let memory = command_dst.reserve(handle.size())?;
        command_dst.bind(memory, handle.memory.clone());

        let resource_dst = command_dst.resource(handle.binding())?;
        let stream = command_dst.streams.current().sys;

        core::mem::drop(command_dst);
        let communication = self.collective_stream(stream_id, stream)?;

        // Get the communicator.
        let mut device_ids = vec![device_id_src, self.device_id];
        device_ids.sort();
        let comm_id = CommunicationId::from(device_ids.clone());
        let comm = self
            .communicators
            .get(&comm_id)
            .ok_or_else(|| graph::error("Receive communicator is not initialized"))?;

        let rank_src = device_ids
            .iter()
            .position(|id| id.index_id != self.device_id.index_id)
            .unwrap() as i32;

        // Perform the `recv` operation.
        let (nccl_dtype, count) = get_nccl_dtype_count(dtype, resource_dst.size);
        // SAFETY: `resource.ptr` is a valid device pointer.
        // `comm` is a valid NCCL communicator initialized via `comm_init_rank`.
        // `communication` is the ordinary or capture-owned collective stream.
        unsafe {
            cudarc::nccl::result::recv(
                resource_dst.ptr as *mut _,
                count,
                nccl_dtype,
                rank_src,
                *comm,
                communication as _,
            )
            .map_err(|e| ServerError::Generic {
                reason: format!("NCCL recv failed: {e:?}"),
                backtrace: BackTrace::capture(),
            })?;
        }

        Ok(())
        })();
        self.collective_result(stream_id, &writes, result)
    }
}

impl CudaServer {
    /// Create a new cuda server.
    pub(crate) fn new(
        ctx: CudaContext,
        mem_props: MemoryDeviceProperties,
        mem_config: MemoryConfiguration,
        mem_alignment: usize,
        device_id: DeviceId,
        utilities: ServerUtilities<Self>,
    ) -> Self {
        let config = RudaRuntimeConfig::get();
        let max_streams = config.streaming.max_streams;

        ctx.unsafe_set_current().unwrap();

        let comm_stream = cudarc::driver::result::stream::create(
            cudarc::driver::result::stream::StreamKind::NonBlocking,
        )
        .expect("Can create a new stream.");

        let interop = InteropState::new(ctx.context as usize,max_streams);
        Self {
            graphs: graph::GraphRegistry::default(),
            captures: Default::default(),
            ctx,
            interop,
            device_id,
            streams: MultiStream::new(
                utilities.logger.clone(),
                CudaStreamBackend::new(
                    mem_props,
                    mem_config,
                    mem_alignment,
                    utilities.logger.clone(),
                ),
                max_streams,
            ),
            utilities: Arc::new(utilities),
            comm_stream,
            communicators: HashMap::default(),
        }
    }

    fn command_no_inputs(
        &mut self,
        stream_id: StreamId,
        mode: StreamErrorMode,
    ) -> Result<Command<'_>, ServerError> {
        self.command(stream_id, [].into_iter(), mode)
    }

    fn profiled_stream(&mut self, stream_id: StreamId) -> Result<cudarc::driver::sys::CUstream, ServerError> {
        if !self.captures.active.is_empty() {
            return Err(graph::error("Profiling cannot record events while the device is recording a graph"));
        }
        let mut command = self.command_no_inputs(stream_id, StreamErrorMode { ignore: false, flush: true })?;
        Ok(command.streams.current().sys)
    }

    fn set_current_checked(&self) -> Result<(), ServerError> {
        // Set once per service command, rather than assuming a cached current
        // context survives external calls or device-service thread migration.
        // Report a driver failure to the caller instead of unwinding the service.
        self.ctx.unsafe_set_current()
            .map_err(|error| super::status::server("cuCtxSetCurrent", error))
    }

    fn command<'a>(
        &mut self,
        stream_id: StreamId,
        handles: impl Iterator<Item = &'a Binding>,
        mode: StreamErrorMode,
    ) -> Result<Command<'_>, ServerError> {
        self.set_current_checked()?;

        if let Some(owner) = self.capture_owner(stream_id) {
            if owner != stream_id {
                return Err(graph::error("The physical stream belongs to another logical stream's capture"));
            }
        }

        let mut handles = handles;
        let captured_bindings: Vec<_> = if self.captures.active.is_empty() {
            Vec::new()
        } else {
            handles.by_ref().collect()
        };
        let mut dependency_failure = None;
        let mut captured_dependencies = Vec::new();
        let capture_group = self.capture_group(stream_id);
        if !captured_bindings.is_empty() {
            let origins: Vec<_> = captured_bindings.iter().map(|binding| self.capture_owner(binding.stream)).collect();
            self.streams.for_each_initialized(|stream| {
                if !stream.capture.recording || stream.capture.owner == Some(stream_id) { return; }
                for (binding, origin) in captured_bindings.iter().zip(&origins) {
                    let pending_allocation = *origin == stream.capture.owner
                        && stream.memory_management_gpu.get_cursor(binding.memory.clone())
                        .is_ok_and(|cursor| cursor > stream.capture.entry_cursor);
                    let start = binding.offset_start.unwrap_or(0);
                    let end = binding.size - binding.offset_end.unwrap_or(0);
                    let pending_write = stream.capture.written.keys().any(|(id, written_start, written_end)| {
                        *id == binding.memory.id() && start < *written_end && *written_start < end
                    });
                    if pending_allocation || pending_write {
                        if capture_group.is_some() && stream.capture.root == capture_group {
                            captured_dependencies.push(stream.sys);
                        } else {
                            dependency_failure.get_or_insert_with(|| graph::error(
                                "A different capture graph has not executed this buffer's writes"));
                        }
                        break;
                    }
                }
            });
        }
        if let Some(failure) = dependency_failure {
            self.capture_failure(stream_id, &failure);
            return Err(failure);
        }
        if !captured_dependencies.is_empty() {
            let destination = self.streams.resolve(stream_id, [].into_iter(), false)?.current().sys;
            for source in captured_dependencies {
                if let Err(failure) = super::capture::dependency(source, destination) {
                    self.capture_failure(stream_id, &failure);
                    return Err(failure);
                }
            }
        }
        let mut streams = self.streams.resolve(stream_id, captured_bindings.into_iter().chain(handles), !mode.ignore)?;
        for &owner in &self.captures.active {
            streams.for_each_initialized(|stream| {
                let forbid = stream.capture.root == Some(owner) && stream.capture.recording;
                stream.memory_management_gpu.begin_capture(owner, forbid);
                stream.memory_management_cpu.begin_capture(owner, forbid);
            });
        }
        Ok(Command::new(self.utilities.service, &mut self.ctx, streams))
    }

    fn launch_checked(
        &mut self,
        kernel: Box<dyn RudaTask<CudaCompiler>>,
        count: RudaCount,
        bindings: KernelArguments,
        mode: ExecutionMode,
        stream_id: StreamId,
        mut graph: Option<&mut crate::execution::graph::KernelGraph>,
    ) -> Result<(), ServerError> {
        // Retain allocation/view handles before the invocation consumes them.
        // Tensor maps and dynamic grids are rejected by explicit graph preflight.
        let graph_bindings = graph.as_ref().map(|_| bindings.buffers.clone());
        let mut kernel_id = kernel.id();
        let logger = self.streams.logger.clone();
        kernel_id.mode(mode);
        let mut execution = ruda::runtime::server::KernelExecution::new(kernel_id.clone(), &bindings, &count);
        let result = (|| -> Result<(), ServerError> {
        let grid_constants = self
            .ctx
            .compilation_options
            .supports_features
            .grid_constants;
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
        for binding in execution.writes() { command.streams.current().capture.record_write(binding); }
        execution.check()?;

        let count = match count {
            RudaCount::Static(x, y, z) => (x, y, z),
            // TODO: CUDA doesn't have an exact equivalent of dynamic dispatch. Instead, kernels are free to launch other kernels.
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

        if graph.is_none() && (count.0 == 0 || count.1 == 0 || count.2 == 0) {
            execution.skip();
            return Ok(());
        }
        if graph.is_none() {
            execution.enter();
        }

        let (info_const, info_binding) = if grid_constants {
            let info = &bindings.info;

            let mut handle = Option::None;
            if info.dynamic_metadata_offset < info.data.len() {
                let dyn_meta = &bytemuck::cast_slice(&info.data[info.dynamic_metadata_offset..]);
                handle = Some(command.create_with_data(dyn_meta)?);
            }

            (Some(info.data.as_ptr() as *mut c_void), handle)
        } else {
            let mut handle = Option::None;
            if !bindings.info.data.is_empty() {
                handle = Some(command.create_with_data(bytemuck::cast_slice(&bindings.info.data))?);
            }
            (None, handle)
        };

        let mut resources = bindings
            .tensor_maps
            .iter()
            .map(|it| it.binding.clone())
            .chain(bindings.buffers)
            .map(|binding| command.resource(binding))
            .collect::<Result<Vec<_>, _>>()?;

        let mut tensor_maps = Vec::with_capacity(bindings.tensor_maps.len());

        for TensorMapBinding { map, binding } in bindings.tensor_maps.into_iter() {
            let resource = command
                .resource(binding)?;
            let device_ptr = resource.ptr as *mut c_void;

            let mut map_ptr = MaybeUninit::zeroed();

            let shape: Vec<_> = map
                .metadata
                .shape()
                .iter()
                .rev()
                .map(|s| *s as u64)
                .collect();
            let strides: Vec<_> = map
                .metadata
                .strides()
                .iter()
                .rev()
                .skip(1)
                .map(|s| *s as u64 * map.storage_ty.size() as u64)
                .collect();
            let elem_stride: Vec<_> = map.elem_stride.iter().rev().map(|s| *s as u32).collect();

            match &map.format {
                // SAFETY: `map_ptr` is a zeroed `MaybeUninit<CUtensorMap>`. `device_ptr` is a
                // valid device pointer. Shape, strides, tile_size, and elem_stride vectors
                // are constructed from validated metadata and outlive this call.
                TensorMapFormat::Tiled(TiledArgs { tile_size }) => unsafe {
                    let tile_size: Vec<_> =
                        tile_size.iter().rev().copied().map(|s| s as u32).collect();

                    cuTensorMapEncodeTiled(
                        map_ptr.as_mut_ptr(),
                        elem_to_tensor_map_type(map.storage_ty),
                        map.metadata.rank() as u32,
                        device_ptr,
                        shape.as_ptr(),
                        strides.as_ptr(),
                        tile_size.as_ptr(),
                        elem_stride.as_ptr(),
                        interleave_to_cuda(map.interleave),
                        swizzle_to_cuda(map.swizzle),
                        prefetch_to_cuda(map.prefetch),
                        oob_to_cuda(map.oob_fill),
                    )
                    .result()
                    .map_err(|err| {
                        let generic_err =
                            check_tma_generic(&map, device_ptr, &shape, &strides, &elem_stride)
                                .err();
                        let tiled_err = check_tma_tiled(&map, &tile_size).err();
                        generic_err
                            .or(tiled_err)
                            .unwrap_or_else(|| LaunchError::Unknown {
                                reason: format!("{err}"),
                                backtrace: BackTrace::capture(),
                            })
                    })?;
                },
                // SAFETY: Same invariants as `Tiled` above. Additionally, `lower_corner` and
                // `upper_corner` are valid pixel box bounds derived from the tensor map args.
                TensorMapFormat::Im2col(args) => unsafe {
                    let lower_corner: Vec<_> =
                        args.pixel_box_lower_corner.iter().rev().copied().collect();
                    let upper_corner: Vec<_> =
                        args.pixel_box_upper_corner.iter().rev().copied().collect();

                    cuTensorMapEncodeIm2col(
                        map_ptr.as_mut_ptr(),
                        elem_to_tensor_map_type(map.storage_ty),
                        map.metadata.rank() as u32,
                        device_ptr,
                        shape.as_ptr(),
                        strides.as_ptr(),
                        lower_corner.as_ptr(),
                        upper_corner.as_ptr(),
                        args.channels_per_pixel,
                        args.pixels_per_column,
                        elem_stride.as_ptr(),
                        interleave_to_cuda(map.interleave),
                        swizzle_to_cuda(map.swizzle),
                        prefetch_to_cuda(map.prefetch),
                        oob_to_cuda(map.oob_fill),
                    )
                    .result()
                    .map_err(|err| {
                        let generic_err =
                            check_tma_generic(&map, device_ptr, &shape, &strides, &elem_stride)
                                .err();
                        let tiled_err = check_tma_im2col(
                            &map,
                            &lower_corner,
                            &upper_corner,
                            args.channels_per_pixel,
                            args.pixels_per_column,
                        )
                        .err();
                        generic_err
                            .or(tiled_err)
                            .unwrap_or_else(|| LaunchError::Unknown {
                                reason: format!("{err}"),
                                backtrace: BackTrace::capture(),
                            })
                    })?;
                },
                // SAFETY: Same invariants as `Im2col` above. Requires CUDA 12.8+.
                #[cfg(cuda_12080)]
                TensorMapFormat::Im2colWide(args) => unsafe {
                    use cudarc::driver::sys::{
                        CUtensorMapIm2ColWideMode, cuTensorMapEncodeIm2colWide,
                    };
                    cuTensorMapEncodeIm2colWide(
                        map_ptr.as_mut_ptr(),
                        elem_to_tensor_map_type(map.storage_ty),
                        map.metadata.rank() as u32,
                        device_ptr,
                        shape.as_ptr(),
                        strides.as_ptr(),
                        args.pixel_box_lower_corner_width,
                        args.pixel_box_upper_corner_width,
                        args.channels_per_pixel,
                        args.pixels_per_column,
                        elem_stride.as_ptr(),
                        interleave_to_cuda(map.interleave),
                        CUtensorMapIm2ColWideMode::CU_TENSOR_MAP_IM2COL_WIDE_MODE_W,
                        swizzle_to_cuda(map.swizzle),
                        prefetch_to_cuda(map.prefetch),
                        oob_to_cuda(map.oob_fill),
                    )
                    .result()
                    .map_err(|err| {
                        let generic_err =
                            check_tma_generic(&map, device_ptr, &shape, &strides, &elem_stride)
                                .err();
                        generic_err.unwrap_or_else(|| LaunchError::Unknown {
                            reason: format!("{err}"),
                            backtrace: BackTrace::capture(),
                        })
                    })?;
                },
                #[cfg(not(cuda_12080))]
                TensorMapFormat::Im2colWide(_) => {
                    return Err(LaunchError::Unknown {
                        reason: "CUDA version 12.8 required for tensor map format Im2colWide"
                            .into(),
                        backtrace: BackTrace::capture(),
                    }
                    .into());
                }
            };
            // SAFETY: `map_ptr` was fully initialized by one of the `cuTensorMapEncode*`
            // calls above, which all succeeded (errors are propagated before reaching here).
            let binding = unsafe { map_ptr.assume_init() };
            tensor_maps.push(binding);
        }

        let info_pin = if graph.is_some() {
            info_binding.as_ref().map(|handle| handle.clone().binding())
        } else { None };
        resources.extend(
            info_binding
                .into_iter()
                .map(|s| command.resource(s.binding()).expect("Resource to exist")),
        );

        if let Some(graph) = graph.as_deref_mut() {
            graph.prepare_metadata(&bindings.info, info_pin.clone(), grid_constants)?;
            let mut pins = graph_bindings.unwrap_or_default();
            pins.extend(info_pin);
            if pins.len() != resources.len() {
                return Err(crate::execution::graph::error("graph buffer retention layout mismatch"));
            }
            for (binding, resource) in pins.into_iter().zip(resources.iter()) {
                graph.retain(binding, resource.ptr)?;
            }
        }
        let non_producing = graph.is_some();
        command.kernel(
            kernel_id,
            count,
            &tensor_maps,
            &resources,
            info_const,
            graph,
        )?;
        if non_producing { execution.skip(); }

        Ok(())
        })();
        execution.finish(result)
    }

    pub(crate) fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }
}

fn elem_to_tensor_map_type(ty: StorageType) -> CUtensorMapDataType {
    use cudarc::driver::sys::CUtensorMapDataType::*;
    match ty {
        // packed fp4 should be treated as single 4-bit values to simplify indexing/shape handling
        // So a tile of width 16 with fp4 elements is 8 x fp4x2 elements wide.
        #[cfg(cuda_12080)]
        StorageType::Packed(ty, 2) if ty.size_bits() == 4 => CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
        StorageType::Scalar(ElemType::Float(kind)) => match kind {
            // There's no special handling for FP8, so load as u8. `0u8 == 0.0` when reinterpreting.
            FloatKind::E2M1 // single fp4s are padded to a full byte
            | FloatKind::E4M3
            | FloatKind::E5M2
            | FloatKind::UE8M0
            | FloatKind::E2M3
            | FloatKind::E3M2 => CU_TENSOR_MAP_DATA_TYPE_UINT8,
            FloatKind::F16 => CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            FloatKind::BF16 => CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            FloatKind::Flex32 | FloatKind::F32 => CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            FloatKind::TF32 => CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,
            FloatKind::F64 => CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
        },
        StorageType::Scalar(ElemType::Int(kind)) => match kind {
            // UInt is fine because zero bits and size is the same between both
            IntKind::I8 => CU_TENSOR_MAP_DATA_TYPE_UINT8,
            IntKind::I16 => CU_TENSOR_MAP_DATA_TYPE_UINT16,
            IntKind::I32 => CU_TENSOR_MAP_DATA_TYPE_INT32,
            IntKind::I64 => CU_TENSOR_MAP_DATA_TYPE_INT64,
        },
        StorageType::Scalar(ElemType::UInt(kind)) => match kind {
            UIntKind::U8 => CU_TENSOR_MAP_DATA_TYPE_UINT8,
            UIntKind::U16 => CU_TENSOR_MAP_DATA_TYPE_UINT16,
            UIntKind::U32 => CU_TENSOR_MAP_DATA_TYPE_UINT32,
            UIntKind::U64 => CU_TENSOR_MAP_DATA_TYPE_UINT64,
        },
        _ => unimplemented!("Not supported for tensor map type"),
    }
}

fn interleave_to_cuda(interleave: TensorMapInterleave) -> CUtensorMapInterleave {
    use cudarc::driver::sys::CUtensorMapInterleave::*;
    match interleave {
        TensorMapInterleave::None => CU_TENSOR_MAP_INTERLEAVE_NONE,
        TensorMapInterleave::B16 => CU_TENSOR_MAP_INTERLEAVE_16B,
        TensorMapInterleave::B32 => CU_TENSOR_MAP_INTERLEAVE_32B,
    }
}

fn swizzle_to_cuda(swizzle: TensorMapSwizzle) -> CUtensorMapSwizzle {
    use cudarc::driver::sys::CUtensorMapSwizzle::*;
    match swizzle {
        TensorMapSwizzle::None => CU_TENSOR_MAP_SWIZZLE_NONE,
        TensorMapSwizzle::B32 => CU_TENSOR_MAP_SWIZZLE_32B,
        TensorMapSwizzle::B64 => CU_TENSOR_MAP_SWIZZLE_64B,
        TensorMapSwizzle::B128 => CU_TENSOR_MAP_SWIZZLE_128B,
        #[cfg(cuda_12080)]
        TensorMapSwizzle::B128Atom32B => CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B,
        #[cfg(cuda_12080)]
        TensorMapSwizzle::B128Atom32BFlip8B => CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B,
        #[cfg(cuda_12080)]
        TensorMapSwizzle::B128Atom64B => CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B,
        #[cfg(not(cuda_12080))]
        _ => unimplemented!("Swizzle atomicity requires CUDA 12.8 or higher"),
    }
}

fn prefetch_to_cuda(prefetch: TensorMapPrefetch) -> CUtensorMapL2promotion {
    use cudarc::driver::sys::CUtensorMapL2promotion::*;
    match prefetch {
        TensorMapPrefetch::None => CU_TENSOR_MAP_L2_PROMOTION_NONE,
        TensorMapPrefetch::B64 => CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
        TensorMapPrefetch::B128 => CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        TensorMapPrefetch::B256 => CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
    }
}

fn oob_to_cuda(fill: OobFill) -> CUtensorMapFloatOOBfill {
    use cudarc::driver::sys::CUtensorMapFloatOOBfill::*;
    match fill {
        OobFill::Zero => CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        OobFill::NaN => CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA,
    }
}

macro_rules! launch_check {
    ($assertion: expr, $($arg:tt)+) => {
        if $assertion {
            Ok(())
        } else {
            Err(LaunchError::Unknown {
                reason: format!($($arg)*),
                backtrace: BackTrace::capture(),
            })
        }
    };
}

fn check_tma_generic(
    map: &TensorMapMeta,
    device_ptr: *mut c_void,
    shape: &[u64],
    strides: &[u64],
    elem_strides: &[u32],
) -> Result<(), LaunchError> {
    // globalAddress invariants
    launch_check!(
        (device_ptr as usize).is_multiple_of(16),
        "Tensor pointer must be 16 byte aligned"
    )?;
    if !matches!(map.interleave, TensorMapInterleave::None) {
        launch_check!(
            (device_ptr as usize).is_multiple_of(32),
            "Tensor pointer must be 32 byte aligned"
        )?;
    }

    // tensorRank invariants
    launch_check!(
        (1..=5).contains(&map.metadata.rank()),
        "Rank must be between 1 and 5"
    )?;
    launch_check!(
        matches!(map.interleave, TensorMapInterleave::None) || map.metadata.rank() >= 3,
        "When interleave is enabled, rank must be >= 3"
    )?;

    // globalDim invariants
    launch_check!(
        shape.iter().all(|it| *it <= u32::MAX as u64),
        "Shape must be <= u32::MAX"
    )?;
    #[cfg(cuda_12080)]
    if matches!(map.storage_ty, StorageType::Packed(ty, 2) if ty.size_bits() == 4) {
        launch_check!(
            shape[0].is_multiple_of(2),
            "Packed tensor map must have multiple of 2 for the innermost dimension"
        )?;
    }

    // globalStrides invariants
    launch_check!(
        strides.iter().all(|it| it.is_multiple_of(16)),
        "Strides must be 16 byte aligned"
    )?;
    if matches!(map.interleave, TensorMapInterleave::B32) {
        launch_check!(
            strides.iter().all(|it| it.is_multiple_of(32)),
            "Strides must be 32 byte aligned when interleave is B32"
        )?;
    }

    // elementStrides invariants
    launch_check!(
        elem_strides.iter().all(|it| *it > 0 && *it <= 8),
        "Element strides must be non-zero and <= 8"
    )?;
    if matches!(map.interleave, TensorMapInterleave::None) {
        launch_check!(
            elem_strides[0] == 1,
            "Innermost element stride is ignored without interleaving"
        )?;
    }

    // oobFill invariants
    if matches!(map.oob_fill, OobFill::NaN) {
        launch_check!(
            map.storage_ty.is_float(),
            "NaN fill is only supported for float types"
        )?;
    }

    Ok(())
}

fn check_tma_tiled(map: &TensorMapMeta, tile_size: &[u32]) -> Result<(), LaunchError> {
    launch_check!(
        tile_size.len() == map.metadata.rank(),
        "Tile shape should match rank"
    )?;
    launch_check!(
        tile_size.iter().all(|it| *it > 0 && *it <= 256),
        "Tile shape must be non-zero and <= 256"
    )?;
    let tile_size_0_bytes = tile_size[0] as usize * map.storage_ty.size();
    if matches!(map.interleave, TensorMapInterleave::None) {
        let max_tile_bytes = match map.swizzle {
            TensorMapSwizzle::None => usize::MAX,
            TensorMapSwizzle::B32 => 32,
            TensorMapSwizzle::B64 => 64,
            TensorMapSwizzle::B128
            | TensorMapSwizzle::B128Atom32B
            | TensorMapSwizzle::B128Atom32BFlip8B
            | TensorMapSwizzle::B128Atom64B => 128,
        };
        launch_check!(
            tile_size_0_bytes <= max_tile_bytes,
            "Innermost tile dim must be <= swizzle size"
        )?;
    }
    if matches!(map.interleave, TensorMapInterleave::B32) {
        launch_check!(
            map.swizzle == TensorMapSwizzle::B32,
            "If interleave is B32, swizzle must be B32"
        )?;
    }

    Ok(())
}

fn check_tma_im2col(
    map: &TensorMapMeta,
    lower_corner: &[i32],
    upper_corner: &[i32],
    channels_per_pixel: u32,
    pixels_per_column: u32,
) -> Result<(), LaunchError> {
    launch_check!(
        lower_corner.len() == map.metadata.rank() - 2,
        "Lower corner must be rank - 2 elements"
    )?;
    launch_check!(
        upper_corner.len() == map.metadata.rank() - 2,
        "Upper corner must be rank - 2 elements"
    )?;

    launch_check!(
        map.metadata.rank() >= 3 && map.metadata.rank() <= 5,
        "im2col requires rank to be between 3 and 5"
    )?;

    let (range_lower, range_upper) = match map.metadata.rank() {
        3 => (-32768, 32767),
        4 => (-128, 127),
        5 => (-16, 15),
        _ => unreachable!(),
    };
    launch_check!(
        lower_corner
            .iter()
            .all(|it| *it >= range_lower && *it <= range_upper),
        "Lower corner must be in range [{range_lower}, {range_upper}] for {}D im2col",
        map.metadata.rank()
    )?;
    launch_check!(
        upper_corner
            .iter()
            .all(|it| *it >= range_lower && *it <= range_upper),
        "Upper corner must be in range [{range_lower}, {range_upper}] for {}D im2col",
        map.metadata.rank()
    )?;

    launch_check!(
        channels_per_pixel <= 256,
        "Channels per pixel must be <= 256"
    )?;
    launch_check!(
        pixels_per_column <= 1024,
        "Pixels per column must be <= 1024"
    )?;

    Ok(())
}
