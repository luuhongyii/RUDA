use super::cache::{Record, split_scoped_key};
use ruda_core::{
    bytes::Bytes,
    environment::{self, Namespace, persistence::{Storage, Insertion, Origin, storage}},
};
use std::{boxed::Box, io, string::ToString, sync::Mutex, vec::Vec};

struct BoundStorage {
    generation: u32,
    storage: Box<dyn Storage>,
}

/// The legacy bounded payload, stored in a versioned, bundle-exportable environment table.
pub(super) struct EnvironmentCache {
    namespace: Namespace,
    capacity: usize,
    state: Mutex<Option<BoundStorage>>,
}

impl EnvironmentCache {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            namespace: Namespace::scoped("stack-autotune", "records-v1"),
            capacity,
            state: Mutex::new(None),
        }
    }

    pub(super) fn table(&self) -> &str { self.namespace.as_str() }

    fn access<T>(
        &self, generation: u32, action: impl FnOnce(&dyn Storage) -> io::Result<T>,
    ) -> io::Result<Option<T>> {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if generation != environment::generation() { return Ok(None); }
        if state.as_ref().is_none_or(|state| state.generation != generation) {
            let storage = storage::open(self.namespace.as_str());
            // A switch during open must not label the new table as belonging to an old tune.
            if generation != environment::generation() { return Ok(None); }
            *state = Some(BoundStorage { generation, storage });
        }
        let bound = state.as_ref().expect("opened environment table");
        // The handle remains bound to that table even if another thread switches now.
        action(bound.storage.as_ref()).map(Some)
    }

    pub(super) fn load(&self, scoped: &str) -> io::Result<Option<Record>> {
        let (generation, key) = split_scoped_key(scoped)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "unscoped environment tuning key"))?;
        self.access(generation, |storage| {
            let Some(bytes) = storage.get(key.as_bytes()) else { return Ok(None); };
            let mut record = match Record::decode(&bytes) {
                Ok(record) => record,
                Err(error) => { storage.purge_key(key.as_bytes()); return Err(error); }
            };
            if record.key != key { storage.purge_key(key.as_bytes()); return Ok(None); }
            record.key = scoped.to_string();
            Ok(Some(record))
        }).map(Option::flatten)
    }

    pub(super) fn save(&self, record: &Record) -> io::Result<bool> {
        let (generation, key) = split_scoped_key(&record.key)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "unscoped environment tuning key"))?;
        let mut durable = record.clone();
        durable.key = key.to_string();
        let bytes = durable.encode()?;
        self.access(generation, |storage| {
            let accepted = match storage.insert(key.as_bytes(), Bytes::from_bytes_vec(bytes.clone()), Origin::Local) {
                Insertion::Stored => true,
                Insertion::Conflict(previous) => &previous[..] == bytes.as_slice(),
                Insertion::Failed(reason) => return Err(io::Error::other(reason)),
            };
            if accepted { self.prune(storage); }
            Ok(accepted)
        }).map(|accepted| accepted.unwrap_or(false))
    }

    pub(super) fn remove(&self, scoped: &str) -> io::Result<()> {
        let (generation, key) = split_scoped_key(scoped)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "unscoped environment tuning key"))?;
        self.access(generation, |storage| { storage.purge_key(key.as_bytes()); Ok(()) })?;
        Ok(())
    }

    fn prune(&self, storage: &dyn Storage) {
        let mut entries = Vec::new();
        storage.scan(&mut |key, value| {
            let created = Record::decode(&value).ok()
                .filter(|record| record.key.as_bytes() == key)
                .map(|record| record.created);
            entries.push((created, key.to_vec()));
        });
        entries.sort_unstable();
        let extra = entries.len().saturating_sub(self.capacity);
        // Never touch storage from inside its scan visitor: tables may share a lock.
        for (_, key) in entries.into_iter().take(extra) { storage.purge_key(&key); }
    }
}
