use crate::memory::storage::{WgpuResource, WgpuStorage};
use crate::schedule::{BindingsResource, ScheduleTask, ScheduledWgpuBackend};
use crate::{AutoCompiler, AutoRepresentation};
use alloc::sync::Arc;
use ruda_core::{
    backtrace::BackTrace,
    bytes::Bytes,
    profile::{ProfileDuration, TimingMethod},
    stream_id::StreamId,
};
use ruda_kernel::dsl::server::{Binding, StreamErrorMode};
use ruda_kernel::dsl::zspace::Shape;
use ruda_kernel::dsl::{
    MemoryConfiguration, WgpuCompilationOptions,
    future::DynFut,
    prelude::*,
    server::{
        CopyDescriptor, IoError, KernelArguments, LaunchError, ProfileError, ProfilingToken,
        ResourceLimitError, ServerCommunication, ServerError, ServerUtilities,
    },
    zspace::{Strides, strides},
};
#[cfg(feature = "spirv")]
use ruda_kernel::dsl::{cache::CacheOption, compilation_cache::CompilationCache, hash::StableHash};
use ruda_core::ir::MemoryDeviceProperties;
use ruda::runtime::allocator::ContiguousMemoryLayoutPolicy;
use ruda::runtime::memory_management::{ManagedMemoryHandle, MemoryUsage};
use ruda::runtime::{
    compiler::RudaTask,
    config::{RudaRuntimeConfig, RuntimeConfig},
    logging::ServerLogger,
    memory_management::MemoryAllocationMode,
    server::ComputeServer,
    storage::ManagedResource,
    stream::scheduler::{SchedulerMultiStream, SchedulerMultiStreamOptions, SchedulerStrategy},
    validation::{validate_ruda_dim, validate_units},
};
use ruda::runtime::compiler::{CompilationCache as PipelineCache, CompilationRecording};
#[cfg(feature = "spirv")]
use ruda::runtime::compiler::CompilationArtifacts;
use crate::compilation::PreparedPipeline;

/// Wgpu compute server.
#[derive(Debug)]
pub struct WgpuServer {
    pub(crate) device: wgpu::Device,
    // A buffer that can be used to store stream id without extra allocations.
    streams_pool: Vec<StreamId>,
    pipelines: PipelineCache<KernelId, Arc<PreparedPipeline>>,
    scheduler: SchedulerMultiStream<ScheduledWgpuBackend>,
    #[cfg(feature = "spirv")]
    pub(crate) spirv_cache:
        Option<CompilationCache<(u64, StableHash), ruda_compiler::spirv::SpirvCacheEntry>>,
    #[cfg(feature = "spirv")]
    spirv_artifacts: CompilationArtifacts<ruda_compiler::spirv::SpirvCacheEntry>,
    pub compilation_options: WgpuCompilationOptions,
    pub(crate) backend: wgpu::Backend,
    pub(crate) utilities: Arc<ServerUtilities<Self>>,
}

impl ServerCommunication for WgpuServer {
    const SERVER_COMM_ENABLED: bool = false;
}

impl WgpuServer {
    /// Create a new server.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        compilation_options: WgpuCompilationOptions,
        device: wgpu::Device,
        queue: wgpu::Queue,
        tasks_max: usize,
        backend: wgpu::Backend,
        timing_method: TimingMethod,
        mut utilities: ServerUtilities<Self>,
    ) -> Self {
        utilities.runtime_name = Some(match compiler(backend, &compilation_options) {
            AutoCompiler::Wgsl(_) => "wgpu<wgsl>",
            #[cfg(feature = "spirv")]
            AutoCompiler::SpirV(_) => "wgpu<spirv>",
            #[cfg(feature = "msl")]
            AutoCompiler::Msl(_) => "wgpu<msl>",
        });
        let backend_scheduler = ScheduledWgpuBackend::new(
            device.clone(),
            queue.clone(),
            memory_properties,
            memory_config,
            timing_method,
            backend,
            tasks_max,
            utilities.logger.clone(),
        );

        let config = RudaRuntimeConfig::get();
        let max_streams = config.streaming.max_streams;
        #[cfg(feature = "spirv")]
        let spirv_artifacts = CompilationArtifacts::primary_only(
            "wgpu-spirv",
            &format!("{}-properties-{:016x}",
                utilities.properties.identity.fingerprint, utilities.properties_hash),
        );
        #[cfg(feature = "spirv")]
        let pipelines = PipelineCache::bound(spirv_artifacts.enabled());
        #[cfg(not(feature = "spirv"))]
        let pipelines = PipelineCache::unbound();

        Self {
            compilation_options,
            streams_pool: Vec::new(),
            device,
            pipelines,
            scheduler: SchedulerMultiStream::new(
                utilities.logger.clone(),
                backend_scheduler,
                SchedulerMultiStreamOptions {
                    max_streams,
                    max_tasks: tasks_max,
                    strategy: SchedulerStrategy::Interleave,
                },
            ),
            #[cfg(feature = "spirv")]
            spirv_cache: {
                let config = ruda::runtime::config::RudaRuntimeConfig::get();
                if let Some(cache) = &config.compilation.cache
                    && !spirv_artifacts.enabled()
                {
                    let root = cache.root();
                    Some(CompilationCache::new(
                        "spirv",
                        CacheOption::default().name(utilities.properties.identity.fingerprint.clone()).root(root),
                    ))
                } else {
                    None
                }
            },
            #[cfg(feature = "spirv")]
            spirv_artifacts,
            backend,
            utilities: Arc::new(utilities),
        }
    }

    fn prepare_bindings(&mut self, bindings: KernelArguments) -> Result<BindingsResource, IoError> {
        // Store all the resources we'll be using. This could be eliminated if
        // there was a way to tie the lifetime of the resource to the memory handle.
        let mut resources = Vec::with_capacity(bindings.buffers.len());

        for b in bindings.buffers.into_iter() {
            let stream = self.scheduler.stream(&b.stream);
            let resource = stream.mem_manage.get_resource(b)?;
            resources.push(resource);
        }

        Ok(BindingsResource {
            resources,
            info: bindings.info,
        })
    }

    fn pipeline(
        &mut self,
        kernel: <Self as ComputeServer>::Kernel,
        bindings: &KernelArguments,
        mode: ExecutionMode,
    ) -> Result<Arc<PreparedPipeline>, LaunchError> {
        let mut kernel_id = kernel.id();
        kernel_id.mode(mode);

        if let Some(pipeline) = self.pipelines.get(&kernel_id) {
            return Ok(pipeline.clone());
        }

        let mut recording = CompilationRecording::new(&kernel_id);
        #[cfg(feature = "spirv")]
        if let Some(entry) = self.spirv_artifacts.get(&kernel_id) {
            let repr = crate::AutoRepresentationRef::SpirV(&entry.kernel);
            let module = self.create_module(&entry.entrypoint_name, crate::backend::ModuleSource::SpirV(&entry.kernel), mode)?;
            let pipeline = self.create_pipeline(&entry.entrypoint_name, Some(repr), module, bindings);
            self.pipelines.insert(kernel_id, pipeline.clone());
            recording.loaded();
            return Ok(pipeline);
        }
        let cached = self.load_cached_pipeline(&kernel_id, bindings, mode)?;

        if let Some(Ok(pipeline)) = cached {
            self.pipelines.insert(kernel_id, pipeline.clone());
            return Ok(pipeline);
        }

        validate_ruda_dim(&self.utilities.properties, &kernel_id)?;
        validate_units(&self.utilities.properties, &kernel_id)?;

        let mut compiler = compiler(self.backend, &self.compilation_options);
        let mut compiled = compiler.compile_recorded(self, kernel, mode, &mut recording)?;

        if self.scheduler.logger.compilation_activated() {
            compiled.debug_info = Some(DebugInformation::new(
                compiler.lang_tag(),
                kernel_id.clone(),
            ));
        }
        self.scheduler.logger.log_compilation(&compiled);
        recording.source(&compiled.source);

        self.validate_shared(&compiled.repr)?;

        // /!\ Do not delete the following commented code.
        // This is useful while working on the metal compiler.
        // Also the errors are printed nicely which is not the case when this is the runtime
        // that does it.
        // println!("SOURCE:\n{}", compiled.source);
        // {
        //     // Write shader in metal file then compile it for error
        //     std::fs::write("shader.metal", &compiled.source).expect("should write to file");
        //     let _status = std::process::Command::new("xcrun")
        //         .args(vec![
        //             "-sdk",
        //             "macosx",
        //             "metal",
        //             "-o",
        //             "shader.ir",
        //             "-c",
        //             "shader.metal",
        //             "-w",
        //         ])
        //         .status()
        //         .expect("should launch the command");
        //     // std::process::exit(status.code().unwrap());
        // }
        let repr = compiled.repr.as_ref().map(|it| it.as_ref());
        let source = crate::backend::ModuleSource::resolve(
            repr, compiler.lang_tag(), &compiled.source, compiled.ruda_dim,
        )?;
        let module = self.create_module(&compiled.entrypoint_name, source, mode)?;
        let pipeline = self.create_pipeline(&compiled.entrypoint_name, repr, module, bindings);
        self.pipelines.insert(kernel_id.clone(), pipeline.clone());

        let mut stored = false;
        #[cfg(feature = "spirv")]
        if (self.spirv_artifacts.enabled() || matches!(cached, Some(Err(_))))
            && let Some(crate::AutoRepresentation::SpirV(kernel)) = compiled.repr
        {
            let entry = ruda_compiler::spirv::SpirvCacheEntry::new(compiled.entrypoint_name, kernel);
            if self.spirv_artifacts.enabled() {
                stored = self.spirv_artifacts.insert(&kernel_id, &compiled.source, entry);
            } else if let Some(Err(key)) = cached {
                if let Err(err) = self.spirv_cache.as_mut().unwrap().insert(key, entry) {
                    log::warn!("Unable to save the SPIR-V {err:?}");
                }
            }
        }
        recording.compiled(stored);

        Ok(pipeline)
    }

    fn validate_shared(&self, repr: &Option<crate::AutoRepresentation>) -> Result<(), LaunchError> {
        let shared_bytes = repr.as_ref().map(|repr| match repr {
            AutoRepresentation::Wgsl(repr) => repr.shared_memory_bytes(),
            #[cfg(feature = "msl")]
            AutoRepresentation::Msl(repr) => repr.shared_memory_size(),
            #[cfg(feature = "spirv")]
            AutoRepresentation::SpirV(repr) => repr.shared_size,
        });
        let max_smem = self.utilities.properties.hardware.max_shared_memory_size;
        if let Some(shared_bytes) = shared_bytes
            && shared_bytes > max_smem
        {
            Err(ResourceLimitError::SharedMemory {
                requested: shared_bytes,
                max: max_smem,
                backtrace: BackTrace::capture(),
            }
            .into())
        } else {
            Ok(())
        }
    }
}

impl ComputeServer for WgpuServer {
    type Kernel = Box<dyn RudaTask<AutoCompiler>>;
    type Storage = WgpuStorage;
    type MemoryLayoutPolicy = ContiguousMemoryLayoutPolicy;
    type Info = wgpu::Backend;

    fn logger(&self) -> Arc<ServerLogger> {
        self.scheduler.logger.clone()
    }

    fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }

    fn staging(
        &mut self,
        _sizes: &[usize],
        _stream_id: StreamId,
    ) -> Result<Vec<Bytes>, ServerError> {
        // TODO: Check if using a staging buffer is useful here.
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
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
        stream.mem_manage.bind(reserved, memory);
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        if let Err(error) = Binding::check_all(descriptors.iter().map(|descriptor| &descriptor.handle)) {
            return Box::pin(async move { Err(error) });
        }
        let bindings: Vec<_> = descriptors.iter().map(|descriptor| descriptor.handle.clone()).collect();
        let mut streams = vec![stream_id];
        let mut resources = Vec::with_capacity(descriptors.len());
        for desc in descriptors {
            if contiguous_strides(&desc.shape) != desc.strides {
                return Box::pin(async {
                    Err(IoError::UnsupportedStrides {
                        backtrace: BackTrace::capture(),
                    }
                    .into())
                });
            }
            if !streams.contains(&desc.handle.stream) {
                streams.push(desc.handle.stream);
            }
            let stream = self.scheduler.stream(&desc.handle.stream);
            let resource = match stream.mem_manage.get_resource(desc.handle) {
                Ok(val) => val,
                Err(err) => return Box::pin(async move { Err(err.into()) }),
            };
            resources.push((resource, desc.shape, desc.elem_size));
        }

        self.scheduler.execute_streams(streams);

        if let Err(error) = Binding::check_all(bindings.iter()) {
            return Box::pin(async move { Err(error) });
        }
        let stream = self.scheduler.stream(&stream_id);
        stream.read_resources(resources)
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
                stream.mem_manage.get_resource(desc.handle)
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

    fn get_resource(
        &mut self,
        binding: Binding,
        stream_id: StreamId,
    ) -> Result<ManagedResource<WgpuResource>, ServerError> {
        let mut streams = vec![stream_id];
        if binding.stream != stream_id {
            streams.push(binding.stream);
        }
        self.scheduler.execute_streams(streams);
        let stream = self.scheduler.stream(&binding.stream);
        let memory = binding.memory.clone();
        let resource = stream.mem_manage.get_resource(binding)?;

        Ok(ManagedResource::new(memory, resource))
    }

    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: RudaCount,
        args: KernelArguments,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) {
        unsafe {
            self.launch_with_mode(kernel, count, args, mode, stream_id, ruda::runtime::dry_run::LaunchMode::Execute)
        };
    }

    unsafe fn launch_with_mode(
        &mut self,
        kernel: Self::Kernel,
        count: RudaCount,
        args: KernelArguments,
        mode: ExecutionMode,
        stream_id: StreamId,
        launch_mode: ruda::runtime::dry_run::LaunchMode,
    ) {
        let mut kernel_id = kernel.id();
        kernel_id.mode(mode);
        let mut execution = ruda::runtime::server::KernelExecution::new(kernel_id, &args, &count);
        if launch_mode.is_skipped() { execution.skip(); }
        self.streams_pool.clear();
        self.streams_pool.extend(args.buffers.iter().map(|binding| binding.stream));
        self.streams_pool.extend(args.tensor_maps.iter().map(|map| map.binding.stream));
        if let RudaCount::Dynamic(binding) = &count {
            self.streams_pool.push(binding.stream);
        }
        let pipeline = match self.scheduler.stream(&stream_id).check_device()
            .and_then(|()| self.pipeline(kernel, &args, mode).map_err(ServerError::Launch)) {
            Ok(val) => val,
            Err(error) => {
                let error = execution.fail(error);
                self.scheduler.stream(&stream_id).error(error);
                return;
            }
        };

        if launch_mode.is_skipped() {
            return;
        }
        execution.set_io(pipeline.io.as_ref());
        if let Err(error) = execution.check() {
            self.scheduler.stream(&stream_id).error(error);
            return;
        }
        if count.is_empty() {
            execution.cancel();
            return;
        }

        let count_resource = match &count {
            RudaCount::Dynamic(binding) => {
                let stream = self.scheduler.stream(&binding.stream);
                match stream.mem_manage.get_resource(binding.clone()) {
                    Ok(resource) => Some(resource),
                    Err(error) => {
                        let error = execution.fail(ServerError::Io(error));
                        self.scheduler.stream(&stream_id).error(error);
                        return;
                    }
                }
            }
            _ => None,
        };

        let resources = match self.prepare_bindings(args) {
            Ok(val) => val,
            Err(err) => {
                let error = execution.fail(ServerError::Io(err));
                self.scheduler.stream(&stream_id).error(error);
                return;
            }
        };
        let scope = match execution.begin_submission() {
            Ok(scope) => scope,
            Err(error) => {
                self.scheduler.stream(&stream_id).error(error);
                return;
            }
        };
        let task = ScheduleTask::Execute {
            pipeline: pipeline.pipeline.clone(),
            count,
            count_resource,
            resources,
            execution,
        };

        self.scheduler.register(stream_id, task, &self.streams_pool);
        if let Some(scope) = scope { scope.complete(); }
    }

    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);

        let stream = self.scheduler.stream(&stream_id);

        stream.flush(StreamErrorMode {
            ignore: false,
            flush: true,
        })
    }

    /// Returns the total time of GPU work this sync completes.
    fn sync(&mut self, stream_id: StreamId) -> DynFut<Result<(), ServerError>> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);

        stream.sync()
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
        self.scheduler.stream(&stream_id).abandon_profile(token);
    }

    fn memory_usage(&mut self, stream_id: StreamId) -> Result<MemoryUsage, ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        Ok(stream.mem_manage.memory_usage())
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.mem_manage.memory_cleanup(true);
    }

    fn memory_compact(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.flush(StreamErrorMode { ignore: false, flush: false })?;
        stream.mem_manage.memory_compact()?;
        Ok(())
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId) {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.mem_manage.mode(mode);
    }
}

fn compiler(backend: wgpu::Backend, options: &WgpuCompilationOptions) -> AutoCompiler {
    let _ = options; // Unused without `spirv` feature
    match backend {
        #[cfg(feature = "spirv")]
        wgpu::Backend::Vulkan if options.supports_vulkan => AutoCompiler::SpirV(ruda_kernel::dsl::lowering::spirv::compiler()),
        #[cfg(feature = "msl")]
        wgpu::Backend::Metal if options.supports_msl => AutoCompiler::Msl(Default::default()),
        _ => AutoCompiler::Wgsl(Default::default()),
    }
}

pub(crate) fn contiguous_strides(shape: &Shape) -> Strides {
    let rank = shape.len();
    let mut strides = strides![1; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
