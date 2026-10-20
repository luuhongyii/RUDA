use super::ExecutionError;
use alloc::sync::Arc;

/// A backend-owned executable with its allocation and stream lifetimes attached.
pub trait HardwareGraph: Send + Sync + 'static {
    /// Enqueue a replay on the original capture stream.
    ///
    /// # Safety
    /// Inputs must remain valid and accesses on other streams must be synchronized.
    unsafe fn replay(&self) -> Result<(), ExecutionError>;
}

/// Shared ownership of an opaque backend graph.
#[derive(Clone)]
pub struct BackendGraph(Arc<dyn HardwareGraph>);

impl BackendGraph {
    /// Wrap a backend's executable and its retained resources.
    pub fn new(graph: impl HardwareGraph) -> Self { Self(Arc::new(graph)) }

    /// Enqueue replay without a device-completion fence.
    ///
    /// # Safety
    /// The captured buffers must remain valid, stable, and free of concurrent access.
    pub unsafe fn replay(&self) -> Result<(), ExecutionError> {
        unsafe { self.0.replay() }
    }
}
