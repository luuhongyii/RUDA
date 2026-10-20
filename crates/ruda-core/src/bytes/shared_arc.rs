//! Allocation controller that shares another [`Bytes`] through an [`Arc`].
//!
//! Unlike the other controllers, [`SharedAllocationController`] does not own an
//! allocation directly. Instead it composes over an existing [`Bytes`] held
//! behind an [`Arc`], optionally viewing only a sub-range of it. This makes
//! cloning and splitting cheap (reference counted, zero-copy) at the cost of
//! never being able to detach the allocation into a `Vec` (so
//! [`Bytes::try_into_vec`](super::Bytes::try_into_vec) always fails on shared
//! bytes).
//!
//! # Example
//!
//! ```
//! use ruda_core::bytes::Bytes;
//!
//! let bytes = Bytes::from_elems(vec![1u32, 2, 3, 4]);
//! let shared = bytes.shared();
//! // Cloning is cheap, both clones reference the same backing allocation.
//! let clone = shared.clone();
//! assert_eq!(&shared[..], &clone[..]);
//! ```

use super::{
    AccessError, AccessPolicy, AllocationController, AllocationProperty, Bytes, SplitError,
    default_controller::{MAX_ALIGN, NativeAllocationController},
};
use crate::stub::Arc;
use alloc::boxed::Box;
use core::mem::MaybeUninit;
use spin::Once;

/// Allocation controller that shares a view into another [`Bytes`] behind an [`Arc`].
///
/// The shared content is lazily copied into a private buffer on first mutable access
/// (copy-on-write), behind a [`Once`] so the private buffer is materialized exactly once.
pub struct SharedAllocationController {
    /// The shared underlying bytes.
    inner: Arc<Bytes>,
    /// Offset, in bytes, of this view into `inner`.
    offset: usize,
    /// Length, in bytes, of this view.
    len: usize,
    /// Lazily initialized private buffer (copy-on-write).
    controller: Once<Box<dyn AllocationController>>,
}

impl SharedAllocationController {
    /// Create a controller viewing `inner[offset..offset + len]`.
    pub(crate) fn new(inner: Arc<Bytes>, offset: usize, len: usize) -> Self {
        debug_assert!(
            offset + len <= inner.len(),
            "shared view must stay within the bounds of the inner bytes"
        );
        Self {
            inner,
            offset,
            len,
            controller: Once::new(),
        }
    }

    /// The shared view, valid as long as no copy-on-write has occurred.
    fn view(&self) -> &[u8] {
        &self.inner[self.offset..self.offset + self.len]
    }

    /// Copy the shared view into a private, writable native allocation on first call.
    /// Called lazily on first mutable access (copy-on-write).
    fn init_mutable(&self) -> Result<(), AccessError> {
        self.controller.try_call_once(|| {
            let bytes = self.inner.read(super::Reader::new())?;
            let data = &bytes[self.offset..self.offset + self.len];
            let controller = NativeAllocationController::alloc_with_data(data, MAX_ALIGN)
                .map_err(|error| AccessError::Read(alloc::format!(
                    "failed to allocate copy-on-write buffer: {error:?}"
                )))?;
            Ok::<Box<dyn AllocationController>, AccessError>(Box::new(controller))
        })?;
        Ok(())
    }
}

impl AllocationController for SharedAllocationController {
    fn memory(&self) -> &[MaybeUninit<u8>] {
        self.memory_with_policy(AccessPolicy::default()).expect("shared: host access failed")
    }

    unsafe fn memory_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        unsafe { self.memory_mut_with_policy(AccessPolicy::default()) }
            .expect("shared: mutable host access failed")
    }

    fn alloc_align(&self) -> usize {
        if self.controller.is_completed() {
            MAX_ALIGN
        } else {
            // Report the inner allocation's alignment. `try_into_vec` still
            // fails because `try_detach` is never implemented for shared bytes.
            if self.offset == 0 { self.inner.align() }
            else { self.inner.align().min(1usize << self.offset.trailing_zeros()) }
        }
    }

    fn property(&self) -> AllocationProperty {
        self.inner.property()
    }

    // The view length, known without copy-on-write (never materializes).
    fn capacity(&self) -> usize {
        self.len
    }

    fn memory_with_policy(&self, policy: AccessPolicy) -> Result<&[MaybeUninit<u8>], AccessError> {
        match self.controller.get() {
            // After copy-on-write, read from the private controller.
            Some(controller) => controller.memory_with_policy(policy),
            None => {
                let bytes = self.inner.read(super::Reader { policy })?;
                let slice = &bytes[self.offset..self.offset + self.len];
                // SAFETY: `&[u8]` and `&[MaybeUninit<u8>]` share a layout, and every
                // byte of the shared view is initialized.
                Ok(unsafe { core::slice::from_raw_parts(slice.as_ptr().cast(), slice.len()) })
            }
        }
    }

    unsafe fn memory_mut_with_policy(
        &mut self,
        policy: AccessPolicy,
    ) -> Result<&mut [MaybeUninit<u8>], AccessError> {
        // Mutating still-shared data requires copying it into a private buffer first.
        if !self.controller.is_completed() && !policy.copy_allowed() {
            return Err(AccessError::WouldCopy);
        }
        // Trigger copy-on-write so we never mutate the shared allocation.
        self.init_mutable()?;

        // SAFETY: `init_mutable` guarantees the private controller is set, and `&mut self` is
        // exclusive.
        let controller = self
            .controller
            .get_mut()
            .expect("controller must be set after init_mutable");
        unsafe { controller.memory_mut_with_policy(policy) }
    }

    fn split(
        &mut self,
        offset: usize,
    ) -> Result<(Box<dyn AllocationController>, Box<dyn AllocationController>), SplitError> {
        if self.controller.is_completed() {
            // After copy-on-write the private buffer is no longer shared.
            return Err(SplitError::Unsupported);
        }
        // Use `>` (not `>=`) to allow boundary splits where one side is empty.
        if offset > self.len {
            return Err(SplitError::InvalidOffset);
        }

        let left = SharedAllocationController::new(self.inner.clone(), self.offset, offset);
        let right = SharedAllocationController::new(
            self.inner.clone(),
            self.offset + offset,
            self.len - offset,
        );

        Ok((Box::new(left), Box::new(right)))
    }

    fn view(&self, start: usize, end: usize) -> Option<Box<dyn AllocationController>> {
        if self.controller.is_completed() {
            // After copy-on-write the private buffer is no longer shared.
            return None;
        }
        if start > end || end > self.len {
            return None;
        }

        Some(Box::new(SharedAllocationController::new(
            self.inner.clone(),
            self.offset + start,
            end - start,
        )))
    }

    fn duplicate(&self) -> Option<Box<dyn AllocationController>> {
        if self.controller.is_completed() {
            // After mutation the private buffer can't be shared cheaply.
            return None;
        }

        Some(Box::new(SharedAllocationController::new(
            self.inner.clone(),
            self.offset,
            self.len,
        )))
    }

    unsafe fn copy_into(&self, buf: &mut [u8]) {
        match self.controller.get() {
            Some(controller) => {
                let memory = controller
                    .memory_with_policy(AccessPolicy::default())
                    .expect("shared: host access failed");
                let copy_len = buf.len().min(memory.len());
                // SAFETY: every byte of the private buffer up to its length is initialized.
                let data =
                    unsafe { core::slice::from_raw_parts(memory.as_ptr().cast::<u8>(), copy_len) };
                buf[..copy_len].copy_from_slice(data);
            }
            None => {
                let src = self.view();
                let copy_len = buf.len().min(src.len());
                buf[..copy_len].copy_from_slice(&src[..copy_len]);
            }
        }
    }
}
