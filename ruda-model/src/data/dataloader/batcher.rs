use ruda_tensor::api::{Device, backend::Backend};

#[cfg(test)]
use crate::TestBackend;

/// A trait for batching items of type `I` into items of type `O`.
pub trait Batcher<B: Backend, I, O>: Send + Sync {
    fn checkpoint(&self) -> Result<Vec<u8>, String> {
        Err("This batcher does not implement continuation; use StatelessBatcher only for a stateless batcher".into())
    }
    fn restore_checkpoint(&self, _state: &[u8]) -> Result<(), String> {
        Err("This batcher does not implement continuation".into())
    }
    /// Batches the given items on the specified device.
    ///
    /// # Arguments
    ///
    /// * `items` - The items to batch.
    /// * `device` - The backend device to use.
    ///
    /// # Returns
    ///
    /// The batched items.
    fn batch(&self, items: Vec<I>, device: &Device<B>) -> O;
}

/// Declares that a batcher has no mutable state other than the backend RNG.
#[derive(Clone)]
pub struct StatelessBatcher<T>(pub T);
impl<B: Backend, I, O, T: Batcher<B, I, O>> Batcher<B, I, O> for StatelessBatcher<T> {
    fn batch(&self, items: Vec<I>, device: &Device<B>) -> O { self.0.batch(items, device) }
    fn checkpoint(&self) -> Result<Vec<u8>, String> { Ok(Vec::new()) }
    fn restore_checkpoint(&self, state: &[u8]) -> Result<(), String> {
        if state.is_empty() { Ok(()) } else { Err("Stateless batcher checkpoint is not empty".into()) }
    }
}

/// Test batcher
#[cfg(test)]
#[derive(new, Clone)]
pub struct TestBatcher;

#[cfg(test)]
impl<I> Batcher<TestBackend, I, Vec<I>> for TestBatcher {
    fn batch(&self, items: Vec<I>, _device: &Device<TestBackend>) -> Vec<I> {
        items
    }
}
