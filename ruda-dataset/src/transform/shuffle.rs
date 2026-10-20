use crate::Dataset;
use crate::transform::{RngSource, SelectionDataset};
use std::sync::RwLock;

/// A Shuffled a dataset.
///
/// This is a thin wrapper around a [SelectionDataset] which selects and shuffles
/// the full indices of the original dataset.
///
/// Consider using [SelectionDataset] if you are only interested in
/// shuffling mechanisms.
///
/// Consider using [sampler dataset](crate::transform::SamplerDataset) if you
/// want a probability distribution which is computed lazily.
pub struct ShuffledDataset<D, I>
where
    D: Dataset<I>,
    I: Clone + Send + Sync,
{
    wrapped: RwLock<SelectionDataset<D, I>>,
}

impl<D, I> ShuffledDataset<D, I>
where
    D: Dataset<I>,
    I: Clone + Send + Sync,
{
    /// Creates a new selection dataset with shuffled indices.
    ///
    /// This is a thin wrapper around `SelectionDataset::new_shuffled`.
    ///
    /// # Arguments
    ///
    /// * `dataset` - The original dataset to select from.
    /// * `rng_source` - The source of the random number generator.
    ///
    /// # Returns
    ///
    /// A new `ShuffledDataset`.
    pub fn new<R>(dataset: D, rng_source: R) -> Self
    where
        R: Into<RngSource>,
    {
        Self {
            wrapped: RwLock::new(SelectionDataset::new_shuffled(dataset, rng_source)),
        }
    }

    /// Creates a new selection dataset with shuffled indices using a fixed seed.
    ///
    /// This is a thin wrapper around `SelectionDataset::new_shuffled_with_seed`.
    ///
    /// # Arguments
    ///
    /// * `dataset` - The original dataset to select from.
    /// * `seed` - A fixed seed for the random number generator.
    ///
    /// # Returns
    ///
    /// A new `ShuffledDataset`.
    #[deprecated(since = "0.19.0", note = "Use `new(dataset, seed)` instead`")]
    pub fn with_seed(dataset: D, seed: u64) -> Self {
        Self::new(dataset, seed)
    }
}

impl<D, I> Dataset<I> for ShuffledDataset<D, I>
where
    D: Dataset<I>,
    I: Clone + Send + Sync,
{
    fn is_read_order_independent(&self) -> bool {
        self.wrapped.read().unwrap().is_read_order_independent()
    }
    fn checkpoint(&self) -> Result<Vec<u8>, String> {
        self.wrapped.read().map_err(|_| "Shuffled dataset state lock is poisoned")?.checkpoint()
    }
    fn restore_checkpoint(&self, state: &[u8]) -> Result<(), String> {
        let (version, indices, source): (u32, Vec<usize>, Vec<u8>) =
            serde_json::from_slice(state).map_err(|error| error.to_string())?;
        let mut wrapped = self.wrapped.write().map_err(|_| "Shuffled dataset state lock is poisoned")?;
        let len = wrapped.wrapped.len();
        if version != 1 || indices.len() != len {
            return Err("Shuffled dataset checkpoint does not match the source length".into());
        }
        let mut seen = vec![false; len];
        for &index in &indices {
            let Some(present) = seen.get_mut(index) else {
                return Err("Shuffled dataset checkpoint contains an out-of-range index".into());
            };
            if *present {
                return Err("Shuffled dataset checkpoint is not a permutation".into());
            }
            *present = true;
        }
        wrapped.wrapped.restore_checkpoint(&source)?;
        wrapped.indices = indices;
        Ok(())
    }

    fn get(&self, index: usize) -> Option<I> {
        self.wrapped.read().unwrap().get(index)
    }

    fn get_many(&self, indices: &[usize]) -> Option<Vec<I>> {
        self.wrapped.read().unwrap().get_many(indices)
    }

    fn len(&self) -> usize {
        self.wrapped.read().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FakeDataset;
    use crate::transform::selection::shuffled_indices;
    use rand::SeedableRng;
    use rand::prelude::StdRng;

    #[test]
    fn test_shuffled_dataset() {
        let dataset = FakeDataset::<String>::new(27);
        let source_items = dataset.iter().collect::<Vec<_>>();

        let seed = 42;

        #[allow(deprecated)]
        let shuffled = ShuffledDataset::with_seed(dataset, seed);

        let mut rng = StdRng::seed_from_u64(seed);
        let indices = shuffled_indices(source_items.len(), &mut rng);

        assert_eq!(shuffled.len(), source_items.len());

        let expected_items: Vec<_> = indices
            .iter()
            .map(|&i| source_items[i].to_string())
            .collect();
        assert_eq!(&shuffled.iter().collect::<Vec<_>>(), &expected_items);
    }
}
