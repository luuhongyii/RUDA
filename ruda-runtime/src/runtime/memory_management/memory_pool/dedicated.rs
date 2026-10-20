use alloc::vec::Vec;
use ruda_core::backtrace::BackTrace;

use super::{ManagedMemoryBinding, ManagedMemoryHandle, PageMapping, Slice, calculate_padding};
use crate::runtime::{
    memory_management::{MemoryLocation, MemoryUsage},
    server::IoError,
    storage::{ComputeStorage, StorageUtilization},
};

pub(crate) struct DedicatedAllocations {
    slices: Vec<Slice>,
    alignment: u64,
    max_alloc_size: u64,
}

impl DedicatedAllocations {
    pub(crate) fn new(max_alloc_size: u64, alignment: u64) -> Self {
        Self { slices: Vec::new(), alignment, max_alloc_size }
    }

    pub(crate) fn alloc<S: ComputeStorage>(
        &mut self,
        storage: &mut S,
        size: u64,
    ) -> Result<ManagedMemoryHandle, IoError> {
        let padding = calculate_padding(size, self.alignment);
        let allocation_size = size.checked_add(padding)
            .filter(|size| *size <= self.max_alloc_size)
            .ok_or_else(|| IoError::BufferTooBig { size, backtrace: BackTrace::capture() })?;
        let index = u32::try_from(self.slices.len()).map_err(|_| IoError::Unknown {
            description: "Dedicated allocation table is full".into(),
            backtrace: BackTrace::capture(),
        })?;
        let mapping = PageMapping::current();
        let storage_handle = mapping.allocate(storage, allocation_size)?;
        let mut slice = Slice::new(storage_handle, padding);
        slice.mapped = mapping.is_mapped();
        slice.storage.utilization = StorageUtilization { offset: 0, size };
        slice.descriptor().update_location(MemoryLocation::dedicated(index));
        let handle = slice.handle.clone();
        self.slices.push(slice);
        Ok(handle)
    }

    pub(crate) fn find(&self, binding: &ManagedMemoryBinding) -> Result<&Slice, IoError> {
        self.slices.get(binding.descriptor().slice())
            .filter(|slice| slice.descriptor() == binding.descriptor())
            .ok_or_else(|| IoError::NotFound {
                reason: "Dedicated allocation no longer exists".into(),
                backtrace: BackTrace::capture(),
            })
    }

    pub(crate) fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        assigned: ManagedMemoryHandle,
        cursor: u64,
    ) -> Result<(), IoError> {
        let location = reserved.descriptor().location();
        let slice = self.slices.get_mut(location.slice as usize)
            .filter(|slice| slice.descriptor() == reserved.descriptor())
            .ok_or_else(|| IoError::NotFound {
                reason: "Dedicated reservation no longer exists".into(),
                backtrace: BackTrace::capture(),
            })?;
        assigned.descriptor().update_location(location);
        slice.cursor = cursor;
        slice.handle = assigned;
        Ok(())
    }

    pub(crate) fn materialize<S: ComputeStorage>(
        &mut self, storage: &mut S, binding: &ManagedMemoryBinding,
    ) -> Result<(), IoError> {
        self.find(binding)?;
        let slice = &mut self.slices[binding.descriptor().slice()];
        let size = slice.effective_size();
        slice.materialize(storage, size)
    }

    pub(crate) fn reclaim<S: ComputeStorage>(&mut self, storage: &mut S) {
        let mut next = 0u32;
        let mut reclaimed = false;
        self.slices.retain(|slice| {
            if slice.is_free() {
                if slice.mapped { storage.dealloc(slice.storage.id); }
                reclaimed = true;
                false
            } else {
                slice.descriptor().update_slice(next);
                next += 1;
                true
            }
        });
        if reclaimed { storage.flush(); }
    }

    pub(crate) fn memory_usage(&self) -> MemoryUsage {
        let mut usage = MemoryUsage {
            number_allocs: 0, bytes_in_use: 0, bytes_padding: 0, bytes_reserved: 0,
        };
        for slice in &self.slices {
            usage.bytes_reserved += slice.effective_size();
            if !slice.is_free() {
                usage.number_allocs += 1;
                usage.bytes_in_use += slice.storage.size();
                usage.bytes_padding += slice.padding;
            }
        }
        usage
    }
}
