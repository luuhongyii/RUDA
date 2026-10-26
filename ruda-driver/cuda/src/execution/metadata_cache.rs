use std::collections::{HashMap, VecDeque};

/// Exact-byte, bounded storage for immutable launch metadata on one stream.
/// Values must never be exposed to graph scalar updates or user writes.
#[derive(Debug)]
pub(crate) struct MetadataCache<T> {
    entries: HashMap<Vec<u8>, T>,
    order: VecDeque<Vec<u8>>,
    bytes: usize,
}

const MAX_ENTRIES: usize = 2048;
const MAX_BYTES: usize = 1024 * 1024;

impl<T> Default for MetadataCache<T> {
    fn default() -> Self {
        Self { entries: HashMap::new(), order: VecDeque::new(), bytes: 0 }
    }
}

impl<T> MetadataCache<T> {
    pub fn get(&self, data: &[u8]) -> Option<&T> { self.entries.get(data) }

    pub fn insert(&mut self, data: &[u8], value: T) {
        if data.len() > MAX_BYTES || self.entries.contains_key(data) { return; }
        while self.entries.len() >= MAX_ENTRIES || self.bytes + data.len() > MAX_BYTES {
            let oldest = self.order.pop_front().expect("nonempty metadata cache");
            self.bytes -= oldest.len();
            self.entries.remove(oldest.as_slice());
        }
        self.bytes += data.len();
        self.order.push_back(data.to_vec());
        self.entries.insert(data.to_vec(), value);
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.bytes = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_bytes_include_length_and_scalar_bits() {
        let mut cache = MetadataCache::default();
        for (index, key) in [vec![], vec![0], vec![0, 0], vec![0, 1], vec![1, 0]].iter().enumerate() {
            cache.insert(key, index);
            assert_eq!(cache.get(key), Some(&index));
        }
        cache.insert(&0.0f32.to_bits().to_ne_bytes(), 10);
        cache.insert(&(-0.0f32).to_bits().to_ne_bytes(), 11);
        assert_eq!(cache.get(&0.0f32.to_bits().to_ne_bytes()), Some(&10));
        assert_eq!(cache.get(&(-0.0f32).to_bits().to_ne_bytes()), Some(&11));
        assert_eq!(cache.get(&[2]), None);
    }

    #[test]
    fn duplicate_keeps_original_value_and_capacity() {
        let mut cache = MetadataCache::default();
        cache.insert(&[1, 2], 7);
        for _ in 0..MAX_ENTRIES * 2 { cache.insert(&[1, 2], 9); }
        assert_eq!(cache.get(&[1, 2]), Some(&7));
        assert_eq!((cache.entries.len(), cache.order.len(), cache.bytes), (1, 1, 2));
    }

    #[test]
    fn entry_limit_evicts_oldest_without_changing_live_values() {
        let mut cache = MetadataCache::default();
        for i in 0..=MAX_ENTRIES { cache.insert(&i.to_ne_bytes(), i); }
        assert_eq!(cache.get(&0usize.to_ne_bytes()), None);
        for i in 1..=MAX_ENTRIES { assert_eq!(cache.get(&i.to_ne_bytes()), Some(&i)); }
        assert_eq!(cache.entries.len(), MAX_ENTRIES);
        assert_eq!(cache.bytes, MAX_ENTRIES * size_of::<usize>());
    }

    #[test]
    fn byte_limit_evicts_and_oversized_values_do_not_flush() {
        let mut cache = MetadataCache::default();
        let first = vec![1; MAX_BYTES / 2 + 1];
        let second = vec![2; MAX_BYTES / 2 + 1];
        cache.insert(&first, 1);
        cache.insert(&second, 2);
        assert_eq!(cache.get(&first), None);
        assert_eq!(cache.get(&second), Some(&2));
        cache.insert(&vec![3; MAX_BYTES + 1], 3);
        assert_eq!(cache.get(&second), Some(&2));
        assert_eq!(cache.bytes, second.len());
    }

    #[test]
    fn independent_stream_caches_and_cleanup_are_isolated() {
        let mut first = MetadataCache::default();
        let mut second = MetadataCache::default();
        first.insert(&[1], 10);
        second.insert(&[1], 20);
        assert_eq!(first.get(&[1]), Some(&10));
        assert_eq!(second.get(&[1]), Some(&20));
        first.clear();
        assert_eq!(first.get(&[1]), None);
        assert_eq!((first.entries.len(), first.order.len(), first.bytes), (0, 0, 0));
        assert_eq!(second.get(&[1]), Some(&20));
    }
}
