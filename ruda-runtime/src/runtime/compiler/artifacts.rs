use super::{KernelCacheKey, build_id_hash, compilation_store, store_compiled};
use crate::runtime::id::KernelId;
use ruda_core::environment::persistence::{Store, StoreValue};
use ruda_core::hash::{StableHash, StableHasher};

/// Build-keyed artifacts with a second index for source-identical kernels.
#[derive(Debug)]
pub struct CompilationArtifacts<V> {
    primary: Option<Store<KernelCacheKey, V>>,
    sources: Option<Store<StableHash, KernelCacheKey>>,
    build: StableHash,
}

impl<V: StoreValue> CompilationArtifacts<V> {
    pub fn new(backend: &'static str, fingerprint: &str) -> Self {
        let mut artifacts = Self::primary_only(backend, fingerprint);
        artifacts.sources = compilation_store(backend, alloc::format!("{fingerprint}/source-index"));
        artifacts
    }

    /// Backends that cache final IR without a separate source compiler need only this index.
    pub fn primary_only(backend: &'static str, fingerprint: &str) -> Self {
        Self {
            primary: compilation_store(backend, fingerprint),
            sources: None,
            build: build_id_hash(),
        }
    }

    pub fn enabled(&self) -> bool {
        self.primary.is_some()
    }

    pub fn key(&self, kernel: &KernelId) -> KernelCacheKey {
        KernelCacheKey::new(kernel, self.build)
    }

    pub fn get(&mut self, kernel: &KernelId) -> Option<V> {
        if !self.enabled() { return None; }
        let key = self.key(kernel);
        self.primary.as_mut()?.remove(&key)
    }

    /// Reuse identical compiler input without rerunning the backend compiler.
    pub fn rekey(&mut self, kernel: &KernelId, source: &str) -> Option<(V, bool)> {
        if self.sources.is_none() { return None; }
        let key = self.key(kernel);
        let hash = StableHasher::hash_one(&source);
        let old_key = *self.sources.as_mut()?.get_mut(&hash)?;
        let primary = self.primary.as_mut()?;
        let value = primary.remove(&old_key)?;
        if old_key == key {
            return Some((value, true));
        }
        let stored = store_compiled(primary, key, value.clone());
        if stored {
            primary.purge_key(&old_key);
            let sources = self.sources.as_mut().unwrap();
            sources.purge_key(&hash);
            store_compiled(sources, hash, key);
        }
        Some((value, stored))
    }

    /// Store the artifact and the exact source used by the backend compiler.
    pub fn insert(&mut self, kernel: &KernelId, source: &str, value: V) -> bool {
        let key = self.key(kernel);
        let Some(primary) = self.primary.as_mut() else { return false; };
        let stored = store_compiled(primary, key, value);
        if stored && let Some(sources) = self.sources.as_mut() {
            let hash = StableHasher::hash_one(&source);
            if sources.get_mut(&hash).is_some_and(|existing| *existing != key) {
                sources.purge_key(&hash);
            }
            store_compiled(sources, hash, key);
        }
        stored
    }
}
