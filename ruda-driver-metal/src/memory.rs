use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus,
    MTLCommandEncoder, MTLCommandQueue, MTLDevice, MTLResourceOptions,
};
use ruda::runtime::{
    memory_management::{ManagedMemoryBinding, MemoryResourcePin},
    server::IoError,
    storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization},
};
use ruda_core::backtrace::BackTrace;
use std::{collections::HashMap, sync::{Arc, Mutex}};

pub(crate) fn io_error(reason: impl Into<String>) -> IoError {
    IoError::Unknown { description: reason.into(), backtrace: BackTrace::capture() }
}

pub(crate) fn wait_command(command: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), IoError> {
    command.waitUntilCompleted();
    std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
    if command.status() == MTLCommandBufferStatus::Error {
        return Err(io_error(command.error().map(|error| error.localizedDescription().to_string())
            .unwrap_or_else(|| "Metal command buffer failed".into())));
    }
    Ok(())
}

/// All queues which can access this server's allocations.
#[derive(Default, Debug)]
pub(crate) struct Queues(Mutex<Vec<Retained<ProtocolObject<dyn MTLCommandQueue>>>>);

// Metal queues support concurrent command-buffer creation; registration is protected by a mutex.
unsafe impl Send for Queues {}
unsafe impl Sync for Queues {}

impl Queues {
    pub(crate) fn register(&self, queue: Retained<ProtocolObject<dyn MTLCommandQueue>>) {
        self.0.lock().unwrap_or_else(|error| error.into_inner()).push(queue);
    }

    fn barrier(&self) -> Result<(), IoError> {
        let queues = self.0.lock().unwrap_or_else(|error| error.into_inner());
        let mut commands = Vec::with_capacity(queues.len());
        for queue in queues.iter() {
            let command = queue.commandBuffer().ok_or_else(|| io_error("Metal queue barrier allocation failed"))?;
            command.commit();
            commands.push(command);
        }
        for command in commands { wait_command(&command)?; }
        Ok(())
    }
}

/// A native buffer view. A returned adaptive view retains its relocation lease.
#[derive(Debug, Clone)]
pub struct MetalResource {
    pub buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub offset: u64,
    pub size: u64,
    pin: Option<MemoryResourcePin>,
}

// MTLBuffer references are thread-safe; host accesses are ordered by stream events.
unsafe impl Send for MetalResource {}
unsafe impl Sync for MetalResource {}

#[derive(Debug)]
pub struct MetalStorage {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    copy_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    queues: Arc<Queues>,
    buffers: HashMap<StorageId, Retained<ProtocolObject<dyn MTLBuffer>>>,
    pending_copies: Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
}

// Access is serialized by the compute server. Queues and buffers are native retained objects.
unsafe impl Send for MetalStorage {}

impl MetalStorage {
    pub(crate) fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        queues: Arc<Queues>,
    ) -> Result<Self, IoError> {
        let copy_queue = device.newCommandQueue().ok_or_else(|| io_error("Metal copy queue allocation failed"))?;
        queues.register(copy_queue.clone());
        Ok(Self { device, copy_queue, queues, buffers: HashMap::new(), pending_copies: Vec::new() })
    }
}

impl ComputeStorage for MetalStorage {
    type Resource = MetalResource;

    fn alignment(&self) -> usize { 256 }

    fn get(&mut self, handle: &StorageHandle) -> MetalResource {
        MetalResource {
            buffer: self.buffers.get(&handle.id).expect("Metal allocation not found").clone(),
            offset: handle.offset(), size: handle.size(), pin: None,
        }
    }

    fn get_pinned(&mut self, handle: &StorageHandle, binding: ManagedMemoryBinding) -> MetalResource {
        let mut resource = self.get(handle);
        resource.pin = Some(binding.pin());
        resource
    }

    fn supports_relocation(&self) -> bool { true }

    fn relocation_barrier(&mut self) -> Result<(), IoError> {
        // The server submits every outstanding encoder before invoking the allocator.
        self.queues.barrier()
    }

    fn relocation_copy(&mut self, source: &StorageHandle, target: &StorageHandle) -> Result<(), IoError> {
        self.relocation_copy_batch(core::iter::once((source, target)))
    }

    fn relocation_copy_batch<'a>(
        &mut self, copies: impl IntoIterator<Item = (&'a StorageHandle, &'a StorageHandle)>,
    ) -> Result<(), IoError> {
        let mut copies = copies.into_iter().peekable();
        if copies.peek().is_none() { return Ok(()); }
        let mut resources = Vec::new();
        for (source, target) in copies {
            let source = self.get(source);
            let target = self.get(target);
            let size = source.size.checked_next_multiple_of(4)
                .ok_or_else(|| io_error("Metal relocation size overflow"))?;
            if source.size != target.size || source.offset % 4 != 0 || target.offset % 4 != 0
                || source.offset.checked_add(size).is_none_or(|end| end > source.buffer.length() as u64)
                || target.offset.checked_add(size).is_none_or(|end| end > target.buffer.length() as u64)
            {
                return Err(io_error("Metal relocation copy exceeds its buffer or is unaligned"));
            }
            resources.push((source, target, size));
        }
        let command = self.copy_queue.commandBuffer().ok_or_else(|| io_error("Metal copy command allocation failed"))?;
        let encoder = command.blitCommandEncoder().ok_or_else(|| io_error("Metal blit encoder allocation failed"))?;
        for (source, target, size) in resources {
            // Reservations are disjoint and every byte range was checked above.
            unsafe {
                encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &source.buffer, source.offset as usize, &target.buffer, target.offset as usize, size as usize,
                );
            }
        }
        encoder.endEncoding();
        command.commit();
        self.pending_copies.push(command);
        Ok(())
    }

    fn relocation_complete(&mut self) -> Result<(), IoError> {
        // Retain all command buffers if any wait fails.
        for command in &self.pending_copies { wait_command(command)?; }
        self.pending_copies.clear();
        Ok(())
    }

    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        let length = usize::try_from(size).map_err(|_| io_error("Metal allocation size exceeds usize"))?;
        let buffer = self.device.newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| io_error(format!("Unable to allocate {size} bytes of Metal memory")))?;
        let id = StorageId::new();
        self.buffers.insert(id, buffer);
        Ok(StorageHandle::new(id, StorageUtilization { offset: 0, size }))
    }

    fn dealloc(&mut self, id: StorageId) { self.buffers.remove(&id); }
    fn flush(&mut self) {}
}

