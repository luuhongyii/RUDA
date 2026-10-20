use crate::Dataset;

/// Compose multiple datasets together to create a bigger one.
#[derive(new)]
pub struct ComposedDataset<D> {
    datasets: Vec<D>,
}

impl<D, I> Dataset<I> for ComposedDataset<D>
where
    D: Dataset<I>,
    I: Clone,
{
    fn is_read_order_independent(&self) -> bool {
        self.datasets.iter().all(|dataset| dataset.is_read_order_independent())
    }

    fn checkpoint(&self) -> Result<Vec<u8>, String> {
        let sources = self.datasets.iter().map(|dataset| {
            Ok((dataset.len(), dataset.checkpoint()?))
        }).collect::<Result<Vec<_>, String>>()?;
        serde_json::to_vec(&(1u32, sources)).map_err(|error| error.to_string())
    }

    fn restore_checkpoint(&self, state: &[u8]) -> Result<(), String> {
        let (version, sources): (u32, Vec<(usize, Vec<u8>)>) =
            serde_json::from_slice(state).map_err(|error| error.to_string())?;
        if version != 1 || sources.len() != self.datasets.len()
            || sources.iter().zip(&self.datasets).any(|((len, _), dataset)| *len != dataset.len())
        { return Err("Composed dataset sources differ from checkpoint".into()); }
        for ((_, state), dataset) in sources.iter().zip(&self.datasets) {
            dataset.restore_checkpoint(state)?;
        }
        Ok(())
    }

    fn get(&self, index: usize) -> Option<I> {
        let mut current_index = 0;
        for dataset in self.datasets.iter() {
            if index < dataset.len() + current_index {
                return dataset.get(index - current_index);
            }
            current_index += dataset.len();
        }
        None
    }

    fn get_many(&self, indices: &[usize]) -> Option<Vec<I>> {
        if indices.is_empty() {
            return Some(Vec::new());
        }

        let mut offset = 0;
        let ends: Vec<_> = self.datasets
            .iter()
            .map(|dataset| {
                offset += dataset.len();
                offset
            })
            .collect();
        if indices.iter().any(|&index| index >= offset) {
            return None;
        }
        let mut items = Vec::with_capacity(indices.len());
        let mut local_indices = Vec::new();
        let mut start = 0;
        while start < indices.len() {
            let dataset = ends.partition_point(|&end| end <= indices[start]);
            let offset = if dataset == 0 { 0 } else { ends[dataset - 1] };
            let mut end = start + 1;
            while end < indices.len() && indices[end] >= offset && indices[end] < ends[dataset] {
                end += 1;
            }
            let translated = if offset == 0 {
                &indices[start..end]
            } else {
                local_indices.clear();
                local_indices.extend(indices[start..end].iter().map(|&index| index - offset));
                local_indices.as_slice()
            };
            let batch = self.datasets[dataset].get_many(translated)?;
            if batch.len() != translated.len() {
                return None;
            }
            items.extend(batch);
            start = end;
        }
        Some(items)
    }

    fn len(&self) -> usize {
        let mut total = 0;
        for dataset in self.datasets.iter() {
            total += dataset.len();
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FakeDataset;

    #[test]
    fn test_composed_dataset() {
        let dataset1 = FakeDataset::<String>::new(10);
        let dataset2 = FakeDataset::<String>::new(5);

        let items1 = dataset1.iter().collect::<Vec<_>>();
        let items2 = dataset2.iter().collect::<Vec<_>>();

        let composed = ComposedDataset::new(vec![dataset1, dataset2]);

        assert_eq!(composed.len(), 15);

        let expected_items: Vec<String> = items1.iter().chain(items2.iter()).cloned().collect();

        let items = composed.iter().collect::<Vec<_>>();

        assert_eq!(items, expected_items);
    }
}
