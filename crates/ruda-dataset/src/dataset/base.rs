use std::sync::Arc;

use crate::DatasetIterator;

/// The dataset trait defines a basic collection of items with a predefined size.
pub trait Dataset<I>: Send + Sync {
    /// Whether concurrent reads preserve both item values and checkpoint state
    /// independently of the order in which reads execute.
    fn is_read_order_independent(&self) -> bool { false }
    /// Capture mutable source/transform state at a quiescent batch boundary.
    fn checkpoint(&self) -> Result<Vec<u8>, String> {
        Err("This dataset does not implement continuation; use StatelessDataset only for a stateless source".into())
    }
    /// Restore the mutable state against the same underlying dataset contents.
    fn restore_checkpoint(&self, _state: &[u8]) -> Result<(), String> {
        Err("This dataset does not implement continuation".into())
    }
    /// Gets the item at the given index.
    fn get(&self, index: usize) -> Option<I>;

    /// Fetch actual items in requested order, preserving repeated indices.
    /// None means at least one requested item is absent; no filler is returned.
    /// Storage-backed implementations may combine I/O without changing order.
    fn get_many(&self, indices: &[usize]) -> Option<Vec<I>> {
        indices.iter().map(|&index| self.get(index)).collect()
    }

    /// Gets the number of items in the dataset.
    fn len(&self) -> usize;

    /// Checks if the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns an iterator over the dataset.
    fn iter(&self) -> DatasetIterator<'_, I>
    where
        Self: Sized,
    {
        DatasetIterator::new(self)
    }
}

impl<D, I> Dataset<I> for Arc<D>
where
    D: Dataset<I>,
{
    fn is_read_order_independent(&self) -> bool { self.as_ref().is_read_order_independent() }
    fn checkpoint(&self) -> Result<Vec<u8>, String> { self.as_ref().checkpoint() }
    fn restore_checkpoint(&self, state: &[u8]) -> Result<(), String> { self.as_ref().restore_checkpoint(state) }
    fn get(&self, index: usize) -> Option<I> {
        self.as_ref().get(index)
    }

    fn get_many(&self, indices: &[usize]) -> Option<Vec<I>> {
        self.as_ref().get_many(indices)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

impl<I> Dataset<I> for Arc<dyn Dataset<I>> {
    fn is_read_order_independent(&self) -> bool { self.as_ref().is_read_order_independent() }
    fn checkpoint(&self) -> Result<Vec<u8>, String> { self.as_ref().checkpoint() }
    fn restore_checkpoint(&self, state: &[u8]) -> Result<(), String> { self.as_ref().restore_checkpoint(state) }
    fn get(&self, index: usize) -> Option<I> {
        self.as_ref().get(index)
    }

    fn get_many(&self, indices: &[usize]) -> Option<Vec<I>> {
        self.as_ref().get_many(indices)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

impl<D, I> Dataset<I> for Box<D>
where
    D: Dataset<I>,
{
    fn is_read_order_independent(&self) -> bool { self.as_ref().is_read_order_independent() }
    fn checkpoint(&self) -> Result<Vec<u8>, String> { self.as_ref().checkpoint() }
    fn restore_checkpoint(&self, state: &[u8]) -> Result<(), String> { self.as_ref().restore_checkpoint(state) }
    fn get(&self, index: usize) -> Option<I> {
        self.as_ref().get(index)
    }

    fn get_many(&self, indices: &[usize]) -> Option<Vec<I>> {
        self.as_ref().get_many(indices)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

impl<I> Dataset<I> for Box<dyn Dataset<I>> {
    fn is_read_order_independent(&self) -> bool { self.as_ref().is_read_order_independent() }
    fn checkpoint(&self) -> Result<Vec<u8>, String> { self.as_ref().checkpoint() }
    fn restore_checkpoint(&self, state: &[u8]) -> Result<(), String> { self.as_ref().restore_checkpoint(state) }
    fn get(&self, index: usize) -> Option<I> {
        self.as_ref().get(index)
    }

    fn get_many(&self, indices: &[usize]) -> Option<Vec<I>> {
        self.as_ref().get_many(indices)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

/// Explicit declaration that indexed reads have no mutable state or private RNG.
/// Dataset contents must remain unchanged across continuation.
#[derive(Clone)]
pub struct StatelessDataset<D>(pub D);

impl<I, D: Dataset<I>> Dataset<I> for StatelessDataset<D> {
    fn is_read_order_independent(&self) -> bool { true }
    fn get(&self, index: usize) -> Option<I> { self.0.get(index) }
    fn get_many(&self, indices: &[usize]) -> Option<Vec<I>> { self.0.get_many(indices) }
    fn len(&self) -> usize { self.0.len() }
    fn checkpoint(&self) -> Result<Vec<u8>, String> { Ok(Vec::new()) }
    fn restore_checkpoint(&self, state: &[u8]) -> Result<(), String> {
        if state.is_empty() { Ok(()) } else { Err("Stateless dataset checkpoint is not empty".into()) }
    }
}
