use crate::execution::status;
use ruda_kernel::dsl::server::ServerError;
use ruda_hip_sys::HIP_SUCCESS;

/// A fence is simply an [event](hipEvent_t) created on a [stream](hipStream_t) that you can wait
/// until completion.
///
/// This is useful for doing synchronization outside of the compute server, which is normally
/// locked by a mutex or a channel. This allows the server to continue accepting other tasks.
pub struct Fence {
    event: ruda_hip_sys::hipEvent_t,
    error: Option<ServerError>,
}

// If we don't close the stream or destroy the event, it is safe.
//
// # Safety
//
// Since streams are never closed and we destroy the event after waiting, which consumes the
// [Fence], it is safe.
unsafe impl Send for Fence {}

impl Drop for Fence {
    fn drop(&mut self) {
        if !self.event.is_null() {
            let _ = unsafe { ruda_hip_sys::hipEventDestroy(self.event) };
        }
    }
}

impl Fence {
    /// Create a new [Fence] on the given stream.
    ///
    /// # Notes
    ///
    /// The [stream](hipStream_t) must be initialized.
    pub fn new(stream: ruda_hip_sys::hipStream_t) -> Self {
        let mut event: ruda_hip_sys::hipEvent_t = std::ptr::null_mut();
        // SAFETY: `stream` must be a valid, initialized HIP stream (enforced by the doc
        // contract). A failed create/record is retained for the subsequent wait.
        unsafe {
            let status = ruda_hip_sys::hipEventCreateWithFlags(
                &mut event,
                ruda_hip_sys::hipEventDefault,
            );
            if status != HIP_SUCCESS {
                return Self { event: core::ptr::null_mut(),
                    error: Some(status::server("hipEventCreateWithFlags", status)) };
            }
            let status = ruda_hip_sys::hipEventRecord(event, stream);
            Self {
                event,
                error: (status != HIP_SUCCESS).then(|| status::server("hipEventRecord", status)),
            }
        }
    }

    /// Wait for the [Fence] to be reached, ensuring that all previous tasks enqueued to the
    /// [stream](hipStream_t) are completed.
    ///
    /// # Notes
    ///
    /// The [stream](hipStream_t) must be initialized.
    #[allow(unused)]
    pub fn wait_async(mut self, stream: ruda_hip_sys::hipStream_t) -> Result<(), ServerError> {
        if let Some(error) = self.error.take() { return Err(error); }
        // SAFETY: `self.event` is a valid event created in `Fence::new`. `stream` must be
        // a valid HIP stream. The event is destroyed after the wait, and `self` is consumed
        // so the event cannot be used again.
        unsafe {
            let status = ruda_hip_sys::hipStreamWaitEvent(stream, self.event, 0);
            status::checked("hipStreamWaitEvent", status)?;
        }
        self.destroy()
    }

    /// Wait for the [Fence] to be reached, ensuring that all previous tasks enqueued to the
    /// [stream](hipStream_t) are completed.
    pub fn wait_sync(mut self) -> Result<(), ServerError> {
        if let Some(error) = self.error.take() { return Err(error); }
        // SAFETY: `self.event` is a valid event created in `Fence::new`. We synchronize
        // (block) until the event completes, then destroy it. `self` is consumed so the
        // event cannot be double-freed.
        unsafe {
            let status = ruda_hip_sys::hipEventSynchronize(self.event);

            status::checked("hipEventSynchronize", status)?;
        }
        self.destroy()
    }

    fn destroy(mut self) -> Result<(), ServerError> {
        let event = core::mem::replace(&mut self.event, core::ptr::null_mut());
        status::checked("hipEventDestroy", unsafe { ruda_hip_sys::hipEventDestroy(event) })
    }
}
