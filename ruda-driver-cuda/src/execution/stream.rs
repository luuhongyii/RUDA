use crate::execution::{
    storage::{
        cpu::{PINNED_MEMORY_ALIGNMENT, PinnedMemoryStorage},
        gpu::GpuStorage,
    },
    sync::Fence,
};
use ruda_kernel::dsl::{
    MemoryConfiguration,
    ir::MemoryDeviceProperties,
    server::{Binding, ServerError},
};
use ruda::runtime::{
    logging::ServerLogger,
    memory_management::{
        MemoryAllocationMode, MemoryManagement, MemoryManagementOptions, drop_queue,
    },
    stream::EventStreamBackend,
};
use std::sync::Arc;

#[derive(Debug)]
pub struct Stream {
    pub(crate) capture: super::capture::CaptureState,
    pub sys: cudarc::driver::sys::CUstream,
    pub memory_management_gpu: MemoryManagement<GpuStorage>,
    pub memory_management_cpu: MemoryManagement<PinnedMemoryStorage>,
    pub drop_queue: drop_queue::PendingDropQueue<Fence>,
}

impl drop_queue::Fence for Fence {
    fn sync(self) {
        let _ = self.wait_sync().ok();
    }
    fn wait(self) -> Result<(), ServerError> { self.wait_sync() }
}

#[derive(new, Debug)]
pub struct CudaStreamBackend {
    mem_props: MemoryDeviceProperties,
    mem_config: MemoryConfiguration,
    mem_alignment: usize,
    logger: Arc<ServerLogger>,
}

impl EventStreamBackend for CudaStreamBackend {
    type Stream = Stream;
    type Event = Fence;

    fn create_stream(&self) -> Self::Stream {
        self.try_create_stream().unwrap_or_else(|error| panic!("CUDA stream creation failed: {error}"))
    }

    fn try_create_stream(&self) -> Result<Self::Stream, ServerError> {
        let stream = cudarc::driver::result::stream::create(
            cudarc::driver::result::stream::StreamKind::NonBlocking,
        )
        .map_err(|error| super::status::server("cuStreamCreate", error))?;

        Ok(self.wrap_stream(stream))
    }

    fn flush(stream: &mut Self::Stream) -> Self::Event {
        Fence::new(stream.sys)
    }

    fn dependency_event(stream: &mut Self::Stream) -> Option<Self::Event> {
        if stream.capture.recording { None } else { Some(Fence::new(stream.sys)) }
    }

    fn dependency_event_for(stream: &mut Self::Stream, recording: bool) -> Result<Option<Self::Event>, ServerError> {
        if recording && !stream.capture.recording {
            super::capture::checked("cuStreamSynchronize capture input", unsafe {
                cudarc::driver::sys::cuStreamSynchronize(stream.sys)
            })?;
            Ok(None)
        } else {
            Ok(Self::dependency_event(stream))
        }
    }

    fn dependency_failed(stream: &mut Self::Stream, error: &ServerError) {
        if stream.capture.recording { stream.capture.failed.get_or_insert_with(|| error.clone()); }
    }

    fn dependency_cursor(stream: &Self::Stream, cursor: u64) -> u64 {
        if stream.capture.recording { stream.capture.entry_cursor } else { cursor }
    }

    fn wait_event(stream: &mut Self::Stream, event: Self::Event) {
        if let Err(error) = event.wait_async(stream.sys) {
            log::error!("CUDA event dependency failed: {error}");
        }
    }

    fn wait_event_checked(stream: &mut Self::Stream, event: Self::Event) -> Result<(), ServerError> {
        event.wait_async(stream.sys)
    }

    fn wait_event_sync(event: Self::Event) -> Result<(), ServerError> {
        event.wait_sync()
    }

    fn handle_cursor(stream: &Self::Stream, binding: &Binding) -> u64 {
        stream
            .memory_management_gpu
            .get_cursor(binding.memory.clone())
            .unwrap_or(0)
    }

    fn is_healthy(_stream: &Self::Stream) -> bool {
        true
    }

    fn is_recording(stream: &Self::Stream) -> bool { stream.capture.recording }
}

impl CudaStreamBackend {
    /// Create a real CUDA nonblocking priority stream. CUDA performs native
    /// clamping to its device priority range, as with CUDA stream APIs.
    pub(crate) fn create_stream_with_priority(&self, priority: i32) -> Result<Stream, ServerError> {
        let mut stream = std::ptr::null_mut();
        let status = unsafe { cudarc::driver::sys::cuStreamCreateWithPriority(
            &mut stream, cudarc::driver::sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32, priority) };
        if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(super::status::server("cuStreamCreateWithPriority", cudarc::driver::DriverError(status)));
        }
        Ok(self.wrap_stream(stream))
    }

    fn wrap_stream(&self, stream: cudarc::driver::sys::CUstream) -> Stream {
        let storage = GpuStorage::new(self.mem_alignment, stream);
        let memory_management_gpu = MemoryManagement::from_configuration(
            storage,
            &self.mem_props,
            self.mem_config.clone(),
            self.logger.clone(),
            MemoryManagementOptions::new("Main GPU Memory"),
        );
        // Pinned host buffers keep fixed addresses even when GPU memory is adaptive.
        let host_memory_config = if self.mem_config.is_adaptive() {
            MemoryConfiguration::default()
        } else {
            self.mem_config.clone()
        };
        let memory_management_cpu = MemoryManagement::from_configuration(
            PinnedMemoryStorage::new(),
            &MemoryDeviceProperties::new(self.mem_props.max_page_size, PINNED_MEMORY_ALIGNMENT as u64),
            host_memory_config,
            self.logger.clone(),
            MemoryManagementOptions::new("Pinned CPU Memory").mode(MemoryAllocationMode::Auto),
        );

        Stream {
            capture: Default::default(),
            sys: stream,
            memory_management_gpu,
            memory_management_cpu,
            drop_queue: Default::default(),
        }
    }
}
