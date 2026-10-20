use crate::memory::allocation::CpuAllocController;
use crate::execution::{
    progress::CpuSequence, queue::CpuExecutionQueue, schedule::ScheduleTask,
};
use ruda_core::{bytes::Bytes, profile::ProfileDuration};
use ruda_kernel::dsl::{
    MemoryConfiguration,
    ir::MemoryDeviceProperties,
    server::{
        Binding, CopyDescriptor, IoError, ProfileError, ProfilingToken, ServerError,
        StreamErrorMode,
    },
};
use ruda::runtime::{
    logging::ServerLogger,
    memory_management::{
        ManagedMemoryHandle, MemoryAllocationMode, MemoryManagement, MemoryManagementOptions,
    },
    storage::{BytesResource, BytesStorage},
    timestamp_profiler::TimestampProfiler,
};
use std::sync::Arc;

pub struct CpuStream {
    queue: CpuExecutionQueue,
    sequence: CpuSequence,
    pub(crate) memory_management: MemoryManagement<BytesStorage>,
    pub(crate) timestamps: TimestampProfiler,
}

impl core::fmt::Debug for CpuStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuStream").finish()
    }
}

impl CpuStream {
    pub fn new(
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        logger: Arc<ServerLogger>,
    ) -> Self {
        let queue = CpuExecutionQueue::get(logger.clone());
        let storage = if memory_config.is_adaptive() {
            let relocation_queue = queue.clone();
            BytesStorage::default().with_relocation_barrier(move || relocation_queue.flush())
        } else {
            BytesStorage::default()
        };
        let memory_management = MemoryManagement::from_configuration(
            storage,
            &memory_properties,
            memory_config,
            logger.clone(),
            MemoryManagementOptions::new("Main CPU"),
        );

        Self {
            memory_management,
            timestamps: TimestampProfiler::default(),
            queue,
            sequence: CpuSequence::default(),
        }
    }

    pub fn enqueue_task(&mut self, task: ScheduleTask) {
        if let Err(error) = self.queue.add_to_stream(task, &mut self.sequence) { self.error(error); }
    }

    pub fn flush(&mut self, _mode: StreamErrorMode) -> Result<(), ServerError> {
        self.sequence.wait();
        self.queue.reclaim_completed();
        Ok(())
    }

    /// Returns whether the stream can accept new tasks.
    pub fn is_healthy(&self) -> bool {
        true
    }

    /// Invalidate active measurements without poisoning unrelated stream work.
    pub fn error(&mut self, error: ServerError) {
        self.timestamps.error(ProfileError::Server(Box::new(error.clone())));
    }

    /// Allocates a new empty buffer using the main memory pool.
    pub fn empty(&mut self, size: u64) -> Result<ManagedMemoryHandle, IoError> {
        self.memory_management.reserve(size)
    }

    /// Maps handles to their corresponding buffers.
    pub fn bind(&mut self, reserved: ManagedMemoryHandle, new: ManagedMemoryHandle) {
        self.memory_management.bind(reserved, new, 0).unwrap();
    }

    /// Producer tasks must be flushed by CpuServer before host leases are taken.
    pub fn read_async(
        &mut self,
        descriptor: CopyDescriptor,
    ) -> impl Future<Output = Result<Bytes, IoError>> + Send + use<> {
        fn inner(
            mem: &mut MemoryManagement<BytesStorage>,
            descriptor: CopyDescriptor,
        ) -> Result<Bytes, IoError> {
            let len = descriptor.handle.size_in_used() as usize;
            let controller = Box::new(CpuAllocController::init(descriptor.handle, mem)?);
            // SAFETY:
            // - The binding has initialized memory for at least `len` bytes.
            Ok(unsafe { Bytes::from_controller(controller, len) })
        }

        let res = inner(&mut self.memory_management, descriptor);

        async move { res }
    }

    pub fn sync(&mut self) -> Result<(), ServerError> {
        self.flush(StreamErrorMode {
            ignore: false,
            flush: true,
        })
    }

    pub fn start_profile(&mut self) -> Result<ProfilingToken, ServerError> {
        self.sync()?;

        Ok(self.timestamps.start())
    }

    pub fn end_profile(&mut self, token: ProfilingToken) -> Result<ProfileDuration, ProfileError> {
        if let Err(err) = self.sync() {
            self.timestamps.error(ProfileError::Server(Box::new(err)));
        }

        self.timestamps.stop(token)
    }

    pub fn allocation_mode(&mut self, mode: MemoryAllocationMode) {
        self.memory_management.mode(mode);
    }

    pub fn get_resource(&mut self, binding: Binding) -> Result<BytesResource, IoError> {
        self.memory_management.get_resource(
            binding.memory,
            binding.offset_start,
            binding.offset_end,
        )
    }
}
