use ruda_core::{backtrace::BackTrace, device::ServiceId, stream_id::StreamId};
use ruda_core::tensor::{Shape, Strides};
use alloc::{format, vec::Vec};
use core::borrow::Borrow;

use crate::runtime::{
    memory_management::{ErrorGraph, FailureId, ManagedMemoryBinding, ManagedMemoryHandle, Skipped, WeakMemoryBinding},
    server::{CopyDescriptor, IoError, ServerError},
};

/// Server handle containing the [memory handle](crate::runtime::server::Handle).
pub struct Handle {
    /// The service whose memory this handle addresses.
    pub service: ServiceId,
    /// Memory handle.
    pub memory: ManagedMemoryHandle,
    /// Memory offset in bytes.
    pub offset_start: Option<u64>,
    /// Memory offset in bytes.
    pub offset_end: Option<u64>,
    /// The stream where the data was created.
    pub stream: StreamId,
    /// Length of the underlying buffer ignoring offsets
    pub(crate) size: u64,
}

impl core::fmt::Debug for Handle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Handle")
            .field("service", &self.service)
            .field("id", &self.memory)
            .field("offset_start", &self.offset_start)
            .field("offset_end", &self.offset_end)
            .field("stream", &self.stream)
            .field("size", &self.size)
            .finish()
    }
}

impl Clone for Handle {
    fn clone(&self) -> Self {
        Self {
            service: self.service,
            memory: self.memory.clone(),
            offset_start: self.offset_start,
            offset_end: self.offset_end,
            stream: self.stream,
            size: self.size,
        }
    }
}

impl Handle {
    /// Creates a new handle of the given size.
    pub fn from_memory(id: ManagedMemoryHandle, service: ServiceId, stream: StreamId, size: u64) -> Self {
        Self {
            service,
            memory: id,
            offset_start: None,
            offset_end: None,
            stream,
            size,
        }
    }
    /// Creates a new handle of the given size.
    pub fn new(service: ServiceId, stream: StreamId, size: u64) -> Self {
        Self {
            service,
            memory: ManagedMemoryHandle::new(),
            offset_start: None,
            offset_end: None,
            stream,
            size,
        }
    }
    /// Checks whether the handle can be mutated in-place without affecting other computation.
    pub fn can_mut(&self) -> bool {
        self.memory.can_mut()
    }

    /// Returns the [`Binding`] corresponding to the current handle.
    pub fn binding(self) -> Binding {
        Binding {
            service: self.service,
            memory: self.memory.binding(),
            offset_start: self.offset_start,
            offset_end: self.offset_end,
            stream: self.stream,
            size: self.size,
        }
    }

    /// Add to the current offset in bytes.
    pub fn offset_start(mut self, offset: u64) -> Self {
        if let Some(val) = &mut self.offset_start {
            *val += offset;
        } else {
            self.offset_start = Some(offset);
        }

        self
    }
    /// Add to the current offset in bytes.
    pub fn offset_end(mut self, offset: u64) -> Self {
        if let Some(val) = &mut self.offset_end {
            *val += offset;
        } else {
            self.offset_end = Some(offset);
        }

        self
    }

    /// Convert the [handle](Handle) into a [binding](Binding) with shape and stride metadata.
    pub fn copy_descriptor(
        self,
        shape: Shape,
        strides: Strides,
        elem_size: usize,
    ) -> CopyDescriptor {
        CopyDescriptor {
            shape,
            strides,
            elem_size,
            handle: self.binding(),
        }
    }
    /// Get the size of the handle, in bytes, accounting for offsets
    pub fn size_in_used(&self) -> u64 {
        self.size - self.offset_start.unwrap_or(0) - self.offset_end.unwrap_or(0)
    }
    /// Get the total size of the handle, in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }
}

/// A binding represents a [Handle] that is bound to managed memory.
///
/// The memory used is known by the compute server.
/// A binding is only valid after being initlized with [`super::ComputeServer::initialize_bindings`]
///
/// # Notes
///
/// A binding is detached from a [`Handle`], meaning that is won't affect [`Handle::can_mut`].
#[derive(Clone, Debug)]
pub struct Binding {
    /// The service whose memory this binding addresses.
    pub service: ServiceId,
    /// The id of the handle the binding is bound to.
    pub memory: ManagedMemoryBinding,
    /// Memory offset in bytes.
    pub offset_start: Option<u64>,
    /// Memory offset in bytes.
    pub offset_end: Option<u64>,
    /// The stream where the data was created.
    pub stream: StreamId,
    /// Length of the underlying buffer ignoring offsets
    pub size: u64,
}

/// A view and its service identity without retaining the allocation.
#[derive(Clone, Debug)]
pub struct WeakBinding {
    memory: WeakMemoryBinding,
    service: ServiceId,
    offset_start: Option<u64>,
    offset_end: Option<u64>,
    stream: StreamId,
    size: u64,
}

impl WeakBinding {
    /// Restore the same view while the original allocation still exists.
    pub fn upgrade(&self) -> Option<Binding> {
        Some(Binding {
            memory: self.memory.upgrade()?,
            service: self.service,
            offset_start: self.offset_start,
            offset_end: self.offset_end,
            stream: self.stream,
            size: self.size,
        })
    }
}

impl Binding {
    /// Keep the view's metadata without retaining its allocation.
    pub fn downgrade(&self) -> WeakBinding {
        WeakBinding {
            memory: self.memory.downgrade(),
            service: self.service,
            offset_start: self.offset_start,
            offset_end: self.offset_end,
            stream: self.stream,
            size: self.size,
        }
    }

    pub(crate) fn range(&self) -> Result<core::ops::Range<u64>, ServerError> {
        let start = self.offset_start.unwrap_or(0);
        let end = self.size.checked_sub(self.offset_end.unwrap_or(0)).filter(|end| start <= *end)
            .ok_or_else(|| IoError::NotFound {
                reason: "Binding offsets exceed the allocation size".into(),
                backtrace: BackTrace::capture(),
            })?;
        Ok(start..end)
    }

    /// Each root claiming this view, independent of the issuing stream.
    pub fn failures(&self) -> Vec<ErrorGraph> {
        self.range().map(|range| self.memory.failures(range.start, range.end)).unwrap_or_default()
    }

    /// Attach a shared root to this view.
    pub fn claim(&self, failure: ErrorGraph) {
        if let Ok(range) = self.range() {
            self.memory.claim(range.start, range.end, failure);
        }
    }

    /// One failed write can leave several views unwritten, all with the same root.
    pub fn fail_all<I: IntoIterator>(bindings: I, error: ServerError)
    where
        I::Item: Borrow<Binding>,
    {
        let mut failure = None;
        for binding in bindings {
            let binding = binding.borrow();
            if binding.range().is_ok_and(|range| !range.is_empty() || !binding.memory.is_allocated()) {
                let root = failure.get_or_insert_with(|| ErrorGraph::new(error.clone()));
                binding.claim(root.clone());
            }
        }
    }

    /// Carry the first failed input's root through a skipped launch.
    pub fn propagate_failure<'a>(
        kernel: crate::runtime::id::KernelId,
        reads: impl IntoIterator<Item = &'a Binding>,
        written: &[Binding],
    ) -> Option<ServerError> {
        for read in reads {
            if let Some(failure) = read.failures().into_iter().next() {
                failure.skipped(Skipped {
                    kernel,
                    needed: read.memory.id(),
                    produced: written.iter().map(|binding| binding.memory.id()).collect(),
                });
                for binding in written {
                    binding.claim(failure.clone());
                }
                return Some(failure.report(read.memory.id()));
            }
        }
        None
    }

    /// Report each root once, even when it claims several buffers or views.
    pub fn check_all<'a>(bindings: impl IntoIterator<Item = &'a Binding>) -> Result<(), ServerError> {
        let mut seen = Vec::new();
        let mut unallocated = Vec::new();
        let mut errors = Vec::new();
        for binding in bindings {
            let range = match binding.range() {
                Ok(range) => range,
                Err(error) => { errors.push(error); continue; }
            };
            let failures = binding.memory.failures(range.start, range.end);
            if failures.is_empty() && !binding.memory.is_allocated()
                && !unallocated.contains(&binding.memory.id())
            {
                unallocated.push(binding.memory.id());
                errors.push(IoError::NotFound {
                    reason: format!("memory {:?} was never allocated", binding.memory.id()).into(),
                    backtrace: BackTrace::capture(),
                }.into());
            }
            for failure in failures {
                if !seen.contains(&failure.id()) {
                    seen.push(failure.id());
                    errors.push(failure.report(binding.memory.id()));
                }
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(ServerError::Several { errors, backtrace: BackTrace::capture() })
        }
    }

    /// Failure of any byte in this view, independent of the stream reading it.
    pub fn failure(&self) -> Option<super::ServerError> {
        match self.range() {
            Ok(range) => self.memory.failure(range.start, range.end),
            Err(error) => Some(error),
        }
    }

    /// Claim this view when its contents were not produced.
    pub fn set_failure(&self, error: super::ServerError) {
        Self::fail_all(core::iter::once(self), error);
    }

    /// Release claims after the complete view has an accepted writer again.
    pub fn written(&self) {
        if let Ok(range) = self.range() {
            self.memory.clear_failure(range.start, range.end);
        }
    }

    /// Release this producer's claim without clearing any later producer.
    pub fn release(&self, failure: FailureId) {
        if let Ok(range) = self.range() {
            self.memory.release_failure(range.start, range.end, failure);
        }
    }

    /// Get the size of the handle, in bytes, accounting for offsets
    pub fn size_in_used(&self) -> u64 {
        self.size - self.offset_start.unwrap_or(0) - self.offset_end.unwrap_or(0)
    }
    /// Get the total size of the handle, in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }
}
