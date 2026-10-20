use super::{backend::Runtime, client::ComputeClient, server::{ComputeServer, ServerError}};
use alloc::sync::Arc;
use ruda_core::{backtrace::BackTrace, device_handle::DeviceHandle, stream_id::StreamId};

/// A graph executable owned by the device service and fixed to its capture stream.
pub struct RuntimeGraph<R: Runtime>(Arc<GraphOwner<R>>);

struct GraphOwner<R: Runtime> {
    client: ComputeClient<R>,
    id: u64,
}

impl<R: Runtime> Clone for RuntimeGraph<R> {
    fn clone(&self) -> Self { Self(self.0.clone()) }
}

fn queue_error(error: impl core::fmt::Debug) -> ServerError {
    ServerError::Generic { reason: alloc::format!("Graph device queue: {error:?}"), backtrace: BackTrace::capture() }
}

impl<R: Runtime> RuntimeGraph<R> {
    /// Enqueue the recorded work without a completion fence.
    ///
    /// # Safety
    /// Captured buffers must retain their meaning and have no concurrent accesses.
    pub unsafe fn replay(&self) -> Result<(), ServerError> {
        if super::dry_run::launch_mode().is_skipped() { return Ok(()); }
        let stream = self.0.client.execution_stream();
        let id = self.0.id;
        DeviceHandle::<R::Server>::new(self.0.client.device_id())
            .submit_blocking(move |server| unsafe { server.graph_replay(stream, id) })
            .map_err(queue_error)?
    }
}

impl<R: Runtime> Drop for GraphOwner<R> {
    fn drop(&mut self) {
        let stream = self.client.execution_stream();
        let id = self.id;
        let _ = DeviceHandle::<R::Server>::new(self.client.device_id())
            .submit_blocking(move |server| server.graph_release(stream, id));
    }
}

impl<R: Runtime> ComputeClient<R> {
    fn capture_command<T: Send + 'static>(&self, command: impl FnOnce(&mut R::Server, StreamId) -> T + Send + 'static) -> Result<T, ServerError> {
        let stream = self.execution_stream();
        DeviceHandle::<R::Server>::new(self.device_id())
            .submit_blocking(move |server| command(server, stream)).map_err(queue_error)
    }

    /// Prepare a hardware-capture warmup on the current stream.
    pub fn graph_prepare(&self) -> Result<(), ServerError> {
        self.capture_command(|server, stream| server.graph_prepare(stream))?
    }

    /// Begin recording warmed work on the current stream.
    pub fn graph_start_capture(&self) -> Result<(), ServerError> {
        self.capture_command(|server, stream| server.graph_start_capture(stream))?
    }

    /// Seal the recording and retain the executable until its last clone drops.
    pub fn graph_stop_capture(&self) -> Result<RuntimeGraph<R>, ServerError> {
        let client = self.fixed_execution_queue();
        let id = client.capture_command(|server, stream| server.graph_stop_capture(stream))??;
        Ok(RuntimeGraph(Arc::new(GraphOwner { client, id })))
    }

    /// Abort a recording, including closure unwind paths.
    pub fn graph_abort_capture(&self) {
        let _ = self.capture_command(|server, stream| server.graph_abort_capture(stream));
    }
}
