use crate::{
    CpuCompiler,
    compilation::{F16Evaluation, MlirCompilerOptions},
    execution::{
        runner::CpuKernel,
        schedule::{BindingsResource, ScheduleTask, ScheduledCpuBackend},
    },
};
use ruda_core::{
    backtrace::BackTrace, bytes::Bytes, profile::ProfileDuration, stream_id::StreamId,
};
use ruda_kernel::dsl::{
    CompilationError, RudaCount, ExecutionMode, MemoryConfiguration, MemoryUsage,
    future::DynFut,
    ir::MemoryDeviceProperties,
    server::{
        Binding, ComputeServer, CopyDescriptor, IoError, KernelArguments, ProfileError,
        ProfilingToken, ServerCommunication, ServerError, ServerUtilities, StreamErrorMode,
    },
    zspace::{Shape, Strides, strides},
};
use ruda::runtime::{
    allocator::ContiguousMemoryLayoutPolicy,
    compiler::RudaTask,
    config::{RudaRuntimeConfig, RuntimeConfig},
    id::KernelId,
    logging::ServerLogger,
    memory_management::{ManagedMemoryHandle, MemoryAllocationMode},
    storage::{BytesStorage, ComputeStorage, ManagedResource},
    stream::scheduler::{SchedulerMultiStream, SchedulerMultiStreamOptions, SchedulerStrategy},
};
use std::{collections::HashMap, sync::Arc};

#[derive(Debug)]
pub struct CpuServer {
    scheduler: SchedulerMultiStream<ScheduledCpuBackend>,
    utilities: Arc<ServerUtilities<CpuServer>>,
    compilation_cache: HashMap<(KernelId, F16Evaluation), CpuKernel>,
    compilation_options: MlirCompilerOptions,
    // A buffer that can be used to store stream id without extra allocations.
    streams_pool: Vec<StreamId>,
}

impl CpuServer {
    pub fn new(
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        utilities: Arc<ServerUtilities<CpuServer>>,
    ) -> Self {
        let backend =
            ScheduledCpuBackend::new(memory_properties, memory_config, utilities.logger.clone());
        let config = RudaRuntimeConfig::get();
        let max_streams = config.streaming.max_streams;

        let scheduler = SchedulerMultiStream::new(
            utilities.logger.clone(),
            backend,
            SchedulerMultiStreamOptions {
                max_streams,
                max_tasks: 8,
                strategy: SchedulerStrategy::Interleave,
            },
        );

        Self {
            scheduler,
            utilities,
            compilation_cache: HashMap::new(),
            compilation_options: MlirCompilerOptions::for_host(),
            streams_pool: Vec::new(),
        }
    }

    pub(crate) fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }
}

impl ComputeServer for CpuServer {
    type Kernel = Box<dyn RudaTask<CpuCompiler>>;
    type Storage = BytesStorage;
    type MemoryLayoutPolicy = ContiguousMemoryLayoutPolicy;
    type Info = ();

    fn logger(&self) -> Arc<ServerLogger> {
        self.scheduler.logger.clone()
    }

    fn staging(
        &mut self,
        _sizes: &[usize],
        _stream_id: StreamId,
    ) -> Result<Vec<Bytes>, ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }

    fn initialize_memory(&mut self, memory: ManagedMemoryHandle, size: u64, stream_id: StreamId) {
        let stream = self.scheduler.stream(&stream_id);
        let reserved = match stream.empty(size) {
            Ok(reserved) => reserved,
            Err(error) => {
                let error: ServerError = error.into();
                memory.binding().set_failure(0, size, error.clone());
                stream.error(error);
                return;
            }
        };
        stream.bind(reserved, memory);
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        let mut streams = vec![stream_id];
        let mut results = Vec::with_capacity(descriptors.len());
        let mut resources = Vec::with_capacity(descriptors.len());

        // Finish producer tasks before acquiring host read leases. Acquiring
        // leases first would conflict with queued writes to the same ranges.
        for desc in &descriptors {
            if !streams.contains(&desc.handle.stream) {
                streams.push(desc.handle.stream);
            }
        }
        self.scheduler.execute_streams(streams.clone());
        for id in streams {
            if let Err(err) = self.scheduler.stream(&id).sync() {
                return Box::pin(async move { Err(err) });
            }
        }
        if let Err(error) = Binding::check_all(descriptors.iter().map(|descriptor| &descriptor.handle)) {
            return Box::pin(async move { Err(error) });
        }
        for desc in descriptors {
            let stream = self.scheduler.stream(&desc.handle.stream);
            results.push(stream.read_async(desc));
        }

        Box::pin(async move {
            for result in results {
                match result.await {
                    Ok(val) => resources.push(val),
                    Err(err) => return Err(err.into()),
                }
            }

            Ok(resources)
        })
    }

    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, stream_id: StreamId) {
        for (desc, data) in descriptors {
            if data.is_empty() || desc.shape.contains(&0) { continue; }
            let destination = desc.handle.clone();
            let mut scope = ruda::runtime::server::WriteScope::deferred(vec![destination.clone()]);
            let owner = desc.handle.stream;
            let stream = self.scheduler.stream(&desc.handle.stream);
            let resource = if contiguous_strides(&desc.shape) != desc.strides {
                Err(IoError::UnsupportedStrides {
                    backtrace: BackTrace::capture(),
                })
            } else {
                stream.get_resource(desc.handle)
            };
            let task = match resource {
                Ok(buffer) => ScheduleTask::Write { data, buffer, destination },
                Err(error) => {
                    let error = scope.fail(ServerError::Io(error));
                    self.scheduler.stream(&stream_id).error(error);
                    continue;
                }
            };
            scope.enter();
            self.scheduler.register(stream_id, task, &[owner]);
            scope.complete();
        }
    }

    fn memory_usage(&mut self, stream_id: StreamId) -> Result<MemoryUsage, ServerError> {
        let stream = self.scheduler.stream(&stream_id);
        Ok(stream.memory_management.memory_usage())
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) {
        let stream = self.scheduler.stream(&stream_id);
        stream.memory_management.cleanup(true)
    }

    fn memory_compact(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.flush(StreamErrorMode { ignore: false, flush: false })?;
        stream.memory_management.compact()?;
        Ok(())
    }

    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: RudaCount,
        bindings: KernelArguments,
        kind: ExecutionMode,
        stream_id: StreamId,
    ) {
        unsafe {
            self.launch_with_mode(kernel, count, bindings, kind, stream_id, ruda::runtime::dry_run::LaunchMode::Execute)
        };
    }

    unsafe fn launch_with_mode(
        &mut self,
        kernel: Self::Kernel,
        count: RudaCount,
        bindings: KernelArguments,
        kind: ExecutionMode,
        stream_id: StreamId,
        launch_mode: ruda::runtime::dry_run::LaunchMode,
    ) {
        if launch_mode.is_skipped() {
            if let Err(error) = self.prepare_kernel(kernel, kind) {
                self.scheduler.stream(&stream_id).error(
                    ruda::runtime::server::LaunchError::CompilationError(error).into(),
                );
            }
            return;
        }
        let mut kernel_id = kernel.id();
        kernel_id.mode(kind);
        let execution = ruda::runtime::server::KernelExecution::new(kernel_id, &bindings, &count);
        self.streams_pool.clear();
        bindings
            .buffers
            .iter()
            .for_each(|b| self.streams_pool.push(b.stream));
        if let RudaCount::Dynamic(binding) = &count {
            self.streams_pool.push(binding.stream);
        }
        let mut task = match self.prepare_task(kernel, count, bindings, kind, execution) {
            Ok(Some(task)) => task,
            Ok(None) => return,
            Err(error) => {
                self.scheduler.stream(&stream_id).error(error);
                return;
            }
        };
        let scope = match &mut task {
            ScheduleTask::Execute { execution, .. } => match execution.begin_submission() {
                Ok(scope) => scope,
                Err(error) => {
                    self.scheduler.stream(&stream_id).error(error);
                    return;
                }
            },
            ScheduleTask::Write { .. } => unreachable!("prepare_task only constructs kernels"),
        };
        self.scheduler.register(stream_id, task, &self.streams_pool);
        if let Some(scope) = scope { scope.complete(); }
    }

    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.flush(ruda_kernel::dsl::server::StreamErrorMode {
            ignore: false,
            flush: true,
        })
    }

    fn sync(&mut self, stream_id: StreamId) -> DynFut<Result<(), ServerError>> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        let result = stream.sync();

        Box::pin(async move { result })
    }

    fn start_profile(&mut self, stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.start_profile()
    }

    fn end_profile(
        &mut self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.end_profile(token)
    }

    fn abandon_profile(&mut self, stream_id: StreamId, token: ProfilingToken) {
        self.scheduler.stream(&stream_id).timestamps.abandon(token);
    }

    fn get_resource(
        &mut self,
        binding: Binding,
        stream_id: StreamId,
    ) -> Result<ManagedResource<<Self::Storage as ComputeStorage>::Resource>, ServerError> {
        let mut streams = vec![stream_id];
        if binding.stream != stream_id {
            streams.push(binding.stream);
        }
        self.scheduler.execute_streams(streams);

        let stream = self.scheduler.stream(&binding.stream);
        let memory = binding.memory.clone();
        let resource = stream.get_resource(binding)?;

        Ok(ManagedResource::new(memory, resource))
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId) {
        let stream = self.scheduler.stream(&stream_id);
        stream.allocation_mode(mode);
    }
}

impl ServerCommunication for CpuServer {
    const SERVER_COMM_ENABLED: bool = false;
}

pub(crate) fn contiguous_strides(shape: &Shape) -> Strides {
    let rank = shape.len();
    let mut strides = strides![1; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

mod preparation;
