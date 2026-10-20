use crate::Dataset;
use std::marker::PhantomData;

/// Basic mapper trait to be used with the [mapper dataset](MapperDataset).
pub trait Mapper<I, O>: Send + Sync {
    /// Whether outputs and continuation state are independent of mapping order.
    fn is_read_order_independent(&self) -> bool { false }
    /// Capture private transform state, including any private RNG.
    fn checkpoint(&self) -> Result<Vec<u8>, String> {
        Err("This mapper does not implement continuation; use StatelessMapper only for a stateless mapper".into())
    }
    /// Restore transform state before the next input is mapped.
    fn restore_checkpoint(&self, _state: &[u8]) -> Result<(), String> {
        Err("This mapper does not implement continuation".into())
    }
    /// Maps an item of type I to an item of type O.
    fn map(&self, item: &I) -> O;
}

/// Declares that mapping has no mutable state or private RNG.
#[derive(Clone)]
pub struct StatelessMapper<M>(pub M);

impl<I, O, M: Mapper<I, O>> Mapper<I, O> for StatelessMapper<M> {
    fn is_read_order_independent(&self) -> bool { true }
    fn checkpoint(&self) -> Result<Vec<u8>, String> { Ok(Vec::new()) }
    fn restore_checkpoint(&self, state: &[u8]) -> Result<(), String> {
        if state.is_empty() { Ok(()) } else { Err("Stateless mapper checkpoint is not empty".into()) }
    }
    fn map(&self, item: &I) -> O { self.0.map(item) }
}

/// Dataset mapping each element in an inner dataset to another element type lazily.
#[derive(new)]
pub struct MapperDataset<D, M, I> {
    dataset: D,
    mapper: M,
    input: PhantomData<I>,
}

impl<D, M, I, O> Dataset<O> for MapperDataset<D, M, I>
where
    D: Dataset<I>,
    M: Mapper<I, O> + Send + Sync,
    I: Send + Sync,
    O: Send + Sync,
{
    fn is_read_order_independent(&self) -> bool {
        self.dataset.is_read_order_independent() && self.mapper.is_read_order_independent()
    }

    fn checkpoint(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec(&(1u32, self.dataset.checkpoint()?, self.mapper.checkpoint()?))
            .map_err(|error| error.to_string())
    }

    fn restore_checkpoint(&self, state: &[u8]) -> Result<(), String> {
        let (version, dataset, mapper): (u32, Vec<u8>, Vec<u8>) =
            serde_json::from_slice(state).map_err(|error| error.to_string())?;
        if version != 1 { return Err("Unsupported mapper dataset checkpoint".into()); }
        self.dataset.restore_checkpoint(&dataset)?;
        self.mapper.restore_checkpoint(&mapper)
    }

    fn get(&self, index: usize) -> Option<O> {
        let item = self.dataset.get(index);
        item.map(|item| self.mapper.map(&item))
    }

    fn get_many(&self, indices: &[usize]) -> Option<Vec<O>> {
        let items = self.dataset.get_many(indices)?;
        if items.len() != indices.len() {
            return None;
        }
        Some(items.iter().map(|item| self.mapper.map(item)).collect())
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{InMemDataset, test_data};

    #[test]
    pub fn given_mapper_dataset_when_iterate_should_iterate_though_all_map_items() {
        struct StringToFirstChar;

        impl Mapper<String, String> for StringToFirstChar {
            fn map(&self, item: &String) -> String {
                let mut item = item.clone();
                item.truncate(1);
                item
            }
        }

        let items_original = test_data::string_items();
        let dataset = InMemDataset::new(items_original);
        let dataset = MapperDataset::new(dataset, StringToFirstChar);

        let items: Vec<String> = dataset.iter().collect();

        assert_eq!(vec!["1", "2", "3", "4"], items);
    }
}
