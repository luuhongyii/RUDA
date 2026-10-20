use crate::{
    MetalCompiler, compilation::{MetalContext, Pipeline},
    memory::{MetalResource, MetalStorage, io_error, wait_command},
    stream::MetalStreamBackend,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLComputeCommandEncoder, MTLDevice, MTLResourceOptions, MTLSize};
use ruda::runtime::{
    allocator::ContiguousMemoryLayoutPolicy, compiler::RudaTask,
    config::{RudaRuntimeConfig, RuntimeConfig}, dry_run::LaunchMode,
    logging::ServerLogger,
    memory_management::{ManagedMemoryHandle, MemoryAllocationMode, MemoryConfiguration, MemoryUsage},
    server::{
        Binding, ComputeServer, CopyDescriptor, ExecutionMode, KernelArguments, KernelExecution,
        ProfileError, ProfilingToken, RudaCount, ServerCommunication, ServerError, ServerUtilities, WriteScope,
    },
    storage::ManagedResource, stream::{MultiStream, ResolvedStreams}, timestamp_profiler::TimestampProfiler,
};
use ruda_core::{
    backtrace::BackTrace, bytes::Bytes, future::DynFut, ir::MemoryDeviceProperties,
    profile::{Instant, ProfileDuration, ProfileTicks}, stream_id::StreamId,
    tensor::striding::has_pitched_row_major_strides,
};
use std::{ptr::NonNull, sync::Arc, time::Duration};

#[derive(Debug)]
pub struct MetalServer {
    context: MetalContext,
    streams: MultiStream<MetalStreamBackend>,
    utilities: Arc<ServerUtilities<Self>>,
    timestamps: TimestampProfiler,
}

// DeviceHandle serializes every server operation. Native command buffers retain their
// resources and completion handlers synchronize any data accessed outside that thread.
unsafe impl Send for MetalServer {}

impl MetalServer {
    pub fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>, memory: MemoryDeviceProperties,
        configuration: MemoryConfiguration, utilities: ServerUtilities<Self>,
    ) -> Self {
        let context = MetalContext::new(device.clone(), utilities.properties.clone());
        let backend = MetalStreamBackend::new(device, memory, configuration, utilities.logger.clone());
        let streams = MultiStream::new(
            utilities.logger.clone(), backend, RudaRuntimeConfig::get().streaming.max_streams,
        );
        Self { context, streams, utilities: Arc::new(utilities), timestamps: TimestampProfiler::default() }
    }

    fn failure(&mut self, error: &ServerError) {
        self.timestamps.error(ProfileError::Server(Box::new(error.clone())));
    }

    fn finish_all(&mut self) -> Result<(), ServerError> {
        let mut first = None;
        self.streams.for_each_initialized(|stream| {
            if let Err(error) = stream.finish() {
                if first.is_none() { first = Some(error); }
            }
        });
        first.map_or(Ok(()), Err)
    }

    fn launch_impl(
        &mut self, kernel: Box<dyn RudaTask<MetalCompiler>>, count: RudaCount,
        bindings: KernelArguments, mode: ExecutionMode, stream_id: StreamId, launch_mode: LaunchMode,
    ) -> Result<(), ServerError> {
        if matches!(&count, RudaCount::Static(x, y, z) if *x == 0 || *y == 0 || *z == 0) {
            return Ok(());
        }
        let mut id = kernel.id();
        id.mode(mode);
        let mut execution = (!launch_mode.is_skipped()).then(|| KernelExecution::new(id.clone(), &bindings, &count));
        let compiled = self.context.compile(&id, kernel, mode, self.utilities.logger.clone());
        let compiled = match compiled {
            Ok(compiled) => compiled,
            Err(error) => {
                let error = ServerError::from(error);
                return Err(match execution.as_mut() { Some(execution) => execution.fail(error), None => error });
            }
        };
        let Some(mut execution) = execution.take() else { return Ok(()); };
        execution.set_io(compiled.io.as_ref());
        execution.check()?;
        execution.enter();
        let result = self.dispatch(compiled, count, bindings, stream_id);
        execution.finish(result)
    }

    fn dispatch(
        &mut self, compiled: Pipeline, count: RudaCount, bindings: KernelArguments, stream_id: StreamId,
    ) -> Result<(), ServerError> {
        if !bindings.tensor_maps.is_empty() {
            return Err(io_error("Tensor maps are not supported by native Metal").into());
        }
        if bindings.buffers.len() > self.utilities.properties.hardware.max_bindings as usize {
            return Err(io_error("Native Metal buffer binding limit exceeded").into());
        }
        let indirect = match &count { RudaCount::Dynamic(binding) => Some(binding), _ => None };
        let mut streams = self.streams.resolve(stream_id, bindings.buffers.iter().chain(indirect), true)?;
        let mut resources = Vec::with_capacity(bindings.buffers.len());
        let mut bytes = 0usize;
        for binding in &bindings.buffers {
            let resource = resource(&mut streams, binding)?;
            bytes = bytes.saturating_add(resource.size as usize);
            resources.push(resource);
        }
        let indirect_resource = indirect.map(|binding| resource(&mut streams, binding)).transpose()?;
        if let Some(resource) = &indirect_resource {
            if resource.size < 12 || resource.offset % 4 != 0 {
                return Err(io_error("Metal indirect dispatch requires three aligned u32 counts").into());
            }
        }
        let stream = streams.current();
        let device = stream.device.clone();
        let active = stream.encoder()?;
        active.encoder.setComputePipelineState(&compiled.pipeline);
        for (index, resource) in resources.iter().enumerate() {
            // The view belongs to this device and is retained through command completion.
            unsafe { active.encoder.setBuffer_offset_atIndex(Some(&resource.buffer), resource.offset as usize, index); }
        }
        if !bindings.info.data.is_empty() {
            let info: &[u8] = bytemuck::cast_slice(&bindings.info.data);
            let index = resources.len();
            let pointer = NonNull::new(info.as_ptr() as *mut _).expect("nonempty metadata");
            if info.len() <= 4096 {
                // setBytes copies metadata into the encoder before returning.
                unsafe { active.encoder.setBytes_length_atIndex(pointer, info.len(), index); }
            } else {
                // Metal copies the source bytes into a retained shared allocation.
                let buffer = unsafe {
                    device.newBufferWithBytes_length_options(pointer, info.len(), MTLResourceOptions::StorageModeShared)
                }.ok_or_else(|| io_error(format!("Metal metadata allocation failed: {} bytes", info.len())))?;
                unsafe { active.encoder.setBuffer_offset_atIndex(Some(&buffer), 0, index); }
                active.temporaries.push(buffer);
            }
        }
        let dim = compiled.ruda_dim;
        let threads = MTLSize { width: dim.x as usize, height: dim.y as usize, depth: dim.z as usize };
        match &count {
            RudaCount::Static(x, y, z) => active.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: *x as usize, height: *y as usize, depth: *z as usize }, threads,
            ),
            RudaCount::Dynamic(_) => {
                let resource = indirect_resource.as_ref().expect("indirect resource resolved");
                // The three-u32 argument range was checked and is retained below.
                unsafe {
                    active.encoder.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                        &resource.buffer, resource.offset as usize, threads,
                    );
                }
            }
        }
        active.resources.extend(resources);
        active.resources.extend(indirect_resource);
        active.bindings.extend(bindings.buffers);
        if let RudaCount::Dynamic(binding) = count { active.bindings.push(binding); }
        stream.batch_ops += 1;
        stream.batch_bytes = stream.batch_bytes.saturating_add(bytes);
        if stream.batch_ops > stream.max_ops || (stream.batch_bytes >> 20) > stream.max_mb {
            stream.flush_checked()?;
        }
        Ok(())
    }
}

fn resource(streams: &mut ResolvedStreams<'_, MetalStreamBackend>, binding: &Binding) -> Result<MetalResource, ServerError> {
    let size = binding.size().checked_sub(binding.offset_start.unwrap_or(0))
        .and_then(|size| size.checked_sub(binding.offset_end.unwrap_or(0)))
        .ok_or_else(|| io_error("Metal binding offsets exceed its allocation"))?;
    let mut resource = streams.get(&binding.stream).memory.get_resource(
        binding.memory.clone(), binding.offset_start, binding.offset_end,
    )?;
    if size > resource.size { return Err(io_error("Metal binding exceeds its reservation").into()); }
    resource.size = size;
    Ok(resource)
}

struct CopyLayout { width: usize, rows: usize, pitch: usize, bytes: usize }

impl CopyLayout {
    fn new(descriptor: &CopyDescriptor, resource: &MetalResource) -> Result<Self, ServerError> {
        let shape = &descriptor.shape;
        let rank = shape.len();
        if rank == descriptor.strides.len() && shape.contains(&0) {
            return Ok(Self { width: 0, rows: 0, pitch: 0, bytes: 0 });
        }
        if rank != descriptor.strides.len() || (rank > 0 && !has_pitched_row_major_strides(shape, &descriptor.strides)) {
            return Err(io_error("Metal host copy requires pitched row-major strides").into());
        }
        let overflow = || io_error("Metal host copy size overflow");
        let width = shape.last().copied().unwrap_or(1).checked_mul(descriptor.elem_size).ok_or_else(overflow)?;
        let rows = if rank <= 1 { 1 } else {
            shape[..rank - 1].iter().try_fold(1usize, |count, size| count.checked_mul(*size)).ok_or_else(overflow)?
        };
        let pitch = if rank <= 1 { width } else {
            descriptor.strides[rank - 2].checked_mul(descriptor.elem_size).ok_or_else(overflow)?
        };
        let bytes = rows.checked_mul(width).ok_or_else(overflow)?;
        let span = if bytes == 0 { 0 } else {
            (rows - 1).checked_mul(pitch).and_then(|prefix| prefix.checked_add(width)).ok_or_else(overflow)?
        };
        if span as u64 > resource.size || resource.offset.checked_add(span as u64)
            .is_none_or(|end| end > resource.buffer.length() as u64)
        {
            return Err(io_error("Metal host copy exceeds its buffer view").into());
        }
        Ok(Self { width, rows, pitch, bytes })
    }
}

impl ServerCommunication for MetalServer { const SERVER_COMM_ENABLED: bool = false; }

impl ComputeServer for MetalServer {
    type Kernel = Box<dyn RudaTask<MetalCompiler>>;
    type Info = ();
    type Storage = MetalStorage;
    type MemoryLayoutPolicy = ContiguousMemoryLayoutPolicy;

    fn logger(&self) -> Arc<ServerLogger> { self.utilities.logger.clone() }
    fn utilities(&self) -> Arc<ServerUtilities<Self>> { self.utilities.clone() }

    fn initialize_memory(&mut self, memory: ManagedMemoryHandle, size: u64, stream_id: StreamId) {
        let result = (|| -> Result<(), ServerError> {
            let pending = self.streams.resolve(stream_id, std::iter::empty(), true)?
                .current().memory.relocation_pending(size);
            if pending { self.finish_all()?; }
            let mut streams = self.streams.resolve(stream_id, std::iter::empty(), true)?;
            let cursor = streams.cursor;
            let allocator = &mut streams.current().memory;
            let reserved = allocator.reserve(size)?;
            allocator.bind(reserved, memory.clone(), cursor)?;
            Ok(())
        })();
        if let Err(error) = result {
            memory.binding().set_failure(0, size, error.clone());
            self.failure(&error);
        }
    }

    fn read(&mut self, descriptors: Vec<CopyDescriptor>, stream_id: StreamId) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        let result = (|| {
            Binding::check_all(descriptors.iter().map(|descriptor| &descriptor.handle))?;
            let mut streams = self.streams.resolve(stream_id, descriptors.iter().map(|descriptor| &descriptor.handle), true)?;
            streams.current().finish()?;
            descriptors.iter().map(|descriptor| {
                let resource = resource(&mut streams, &descriptor.handle)?;
                let layout = CopyLayout::new(descriptor, &resource)?;
                let mut bytes = vec![0u8; layout.bytes];
                if layout.bytes != 0 {
                    let pointer = resource.buffer.contents().as_ptr().cast::<u8>();
                    for row in 0..layout.rows {
                        // Completion was awaited; each pitched row lies in the checked view.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                pointer.add(resource.offset as usize + row * layout.pitch),
                                bytes.as_mut_ptr().add(row * layout.width), layout.width,
                            );
                        }
                    }
                }
                Ok(Bytes::from_bytes_vec(bytes))
            }).collect()
        })();
        Box::pin(async move { result })
    }

    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, stream_id: StreamId) {
        for (descriptor, bytes) in descriptors {
            if descriptor.shape.contains(&0) { continue; }
            let mut scope = WriteScope::deferred(vec![descriptor.handle.clone()]);
            let result = (|| {
                let mut streams = self.streams.resolve(stream_id, std::iter::once(&descriptor.handle), false)?;
                streams.current().finish()?;
                let resource = resource(&mut streams, &descriptor.handle)?;
                let layout = CopyLayout::new(&descriptor, &resource)?;
                if bytes.len() != layout.bytes {
                    return Err(io_error("Metal host copy byte count does not match its shape").into());
                }
                scope.enter();
                if layout.bytes != 0 {
                    let pointer = resource.buffer.contents().as_ptr().cast::<u8>();
                    for row in 0..layout.rows {
                        // Prior device work completed and the destination range is checked.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                bytes.as_ptr().add(row * layout.width),
                                pointer.add(resource.offset as usize + row * layout.pitch), layout.width,
                            );
                        }
                    }
                }
                Ok(())
            })();
            if let Err(error) = scope.finish(result) { self.failure(&error); }
        }
    }

    fn sync(&mut self, stream_id: StreamId) -> DynFut<Result<(), ServerError>> {
        let event = self.streams.resolve(stream_id, std::iter::empty(), false)
            .and_then(|mut streams| streams.current().flush_checked());
        Box::pin(async move { event?.wait() })
    }

    fn get_resource(&mut self, binding: Binding, stream_id: StreamId) -> Result<ManagedResource<MetalResource>, ServerError> {
        Binding::check_all(std::iter::once(&binding))?;
        let mut streams = self.streams.resolve(stream_id, std::iter::once(&binding), true)?;
        let resource = resource(&mut streams, &binding)?;
        Ok(ManagedResource::new(binding.memory, resource))
    }

    unsafe fn launch(
        &mut self, kernel: Self::Kernel, count: RudaCount, bindings: KernelArguments,
        mode: ExecutionMode, stream_id: StreamId,
    ) {
        if let Err(error) = self.launch_impl(kernel, count, bindings, mode, stream_id, LaunchMode::Execute) {
            self.failure(&error);
        }
    }

    unsafe fn launch_with_mode(
        &mut self, kernel: Self::Kernel, count: RudaCount, bindings: KernelArguments,
        mode: ExecutionMode, stream_id: StreamId, launch_mode: LaunchMode,
    ) {
        if let Err(error) = self.launch_impl(kernel, count, bindings, mode, stream_id, launch_mode) {
            self.failure(&error);
        }
    }

    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        self.streams.resolve(stream_id, std::iter::empty(), false)?.current().flush_checked()?;
        Ok(())
    }

    fn memory_usage(&mut self, stream_id: StreamId) -> Result<MemoryUsage, ServerError> {
        Ok(self.streams.resolve(stream_id, std::iter::empty(), false)?.current().memory.memory_usage())
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) {
        let result = self.finish_all().and_then(|()| {
            self.streams.resolve(stream_id, std::iter::empty(), false)?.current().memory.cleanup(true);
            Ok(())
        });
        if let Err(error) = result { self.failure(&error); }
    }

    fn memory_compact(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        self.finish_all()?;
        self.streams.resolve(stream_id, std::iter::empty(), true)?.current().memory.compact()?;
        Ok(())
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId) {
        match self.streams.resolve(stream_id, std::iter::empty(), false) {
            Ok(mut streams) => streams.current().memory.mode(mode),
            Err(error) => self.timestamps.error(ProfileError::Server(Box::new(error))),
        }
    }

    fn start_profile(&mut self, stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        let mut streams = self.streams.resolve(stream_id, std::iter::empty(), true)?;
        let stream = streams.current();
        if stream.profiling.is_some() { return Err(io_error("Native Metal does not support nested profiling windows").into()); }
        stream.finish()?;
        stream.profiling = Some(Vec::new());
        Ok(self.timestamps.start())
    }

    fn end_profile(&mut self, stream_id: StreamId, token: ProfilingToken) -> Result<ProfileDuration, ProfileError> {
        let result = (|| -> Result<_, ServerError> {
            let mut streams = self.streams.resolve(stream_id, std::iter::empty(), false)?;
            let stream = streams.current();
            let finished = stream.finish();
            let buffers = stream.profiling.take().unwrap_or_default();
            finished?;
            Ok(buffers)
        })();
        if let Err(error) = &result { self.failure(error); }
        self.timestamps.stop(token)?;
        let buffers = result.map_err(|error| ProfileError::Server(Box::new(error)))?;
        let mut start = f64::INFINITY;
        let mut end = f64::NEG_INFINITY;
        for buffer in &buffers {
            wait_command(buffer).map_err(|error| ProfileError::Server(Box::new(error.into())))?;
            start = start.min(buffer.GPUStartTime());
            end = end.max(buffer.GPUEndTime());
        }
        let span = if buffers.is_empty() { Duration::ZERO } else {
            if !start.is_finite() || !end.is_finite() || end < start {
                return Err(ProfileError::NotMeasured { backtrace: BackTrace::capture() });
            }
            Duration::from_secs_f64(end - start)
        };
        let start = Instant::now();
        let ticks = ProfileTicks::from_start_end(start, start + span);
        Ok(ProfileDuration::new_device_time(async move { ticks }))
    }

    fn abandon_profile(&mut self, stream_id: StreamId, token: ProfilingToken) {
        if let Ok(mut streams) = self.streams.resolve(stream_id, std::iter::empty(), false) {
            streams.current().profiling = None;
        }
        self.timestamps.abandon(token);
    }
}

