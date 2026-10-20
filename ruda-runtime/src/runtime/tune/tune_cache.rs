#[cfg(serializable)]
use alloc::vec::Vec;

#[cfg(all(std_io, not(persistence)))]
use ruda_core::cache::{Cache, CacheError};
#[cfg(persistence)]
use ruda_core::environment::persistence::{Store, StoreOptions, CacheOption, Namespace, StoreError};
#[cfg(serializable)]
use serde::{Deserialize, Serialize};

use super::{AutotuneError, AutotuneKey, AutotuneOutcome};
use alloc::string::String;
use hashbrown::HashMap;

#[derive(Debug)]
pub(crate) enum CacheEntry {
    Done {
        checksum: ChecksumState,
        fastest_index: usize,
    },
    Pending,
}

#[derive(Debug)]
#[allow(dead_code)] // Some variants are not created when the cache isn't saved.
pub(crate) enum ChecksumState {
    Match,
    NoMatch,
    ToBeVerified(String),
}

/// Persistent cache key
#[cfg(serializable)]
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone, Hash)]
pub struct PersistentCacheKey<K> {
    /// Operation key.
    pub key: K,
    /// Candidate-list checksum.
    pub checksum: String,
}

/// Persistent cache entry
#[cfg(serializable)]
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
pub struct PersistentCacheValue {
    /// Selected candidate index.
    pub fastest_index: usize,
    /// Measured and rejected candidate outcomes.
    pub results: Vec<AutotuneResult>,
    /// Measured resource ceilings, absent in legacy entries.
    #[serde(default)]
    pub bounds: Option<super::Bounds>,
    /// Derived early-exit limit, absent in legacy entries.
    #[serde(default)]
    pub limit: Option<core::time::Duration>,
}

#[cfg_attr(serializable, derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
/// The result of an autotune job.
pub struct AutotuneResult {
    /// The measured outcome or candidate failure.
    pub outcome: Result<AutotuneOutcome, AutotuneError>,
}

impl AutotuneResult {
    pub(crate) fn error(error: AutotuneError) -> Self {
        Self {
            outcome: Err(error),
        }
    }
    pub(crate) fn success(outcome: AutotuneOutcome) -> Self {
        Self {
            outcome: Ok(outcome),
        }
    }
}

impl Eq for AutotuneResult {}
impl PartialEq for AutotuneResult {
    fn eq(&self, other: &Self) -> bool {
        match (&self.outcome, &other.outcome) {
            (Ok(lhs), Ok(rhs)) => lhs == rhs,
            (Ok(_), Err(_)) => false,
            (Err(_), Ok(_)) => false,
            // We don't have to check the error
            (Err(_), Err(_)) => true,
        }
    }
}

/// Use to find and reuse the best kernel for some input
#[derive(Debug)]
pub(crate) struct TuneCache<K> {
    in_memory_cache: HashMap<K, CacheEntry>,
    #[cfg(all(std_io, not(persistence)))]
    persistent_cache: Option<Cache<PersistentCacheKey<K>, PersistentCacheValue>>,
    #[cfg(persistence)]
    persistent_cache: Option<Store<PersistentCacheKey<K>, PersistentCacheValue>>,
    #[cfg(persistence)]
    table: String,
    #[cfg(persistence)]
    generation: u32,
    #[cfg(persistence)]
    hydrated: bool,
}

/// Result of the cache try
#[derive(Debug)]
pub enum TuneCacheResult {
    /// An operation is found.
    Hit {
        /// The index of the fastest operation to execute.
        fastest_index: usize,
    },
    /// The operation might be cached, but we don't know yet whether the checksum is valid.
    Unchecked,
    /// A tuning job is in flight for this key — the worker hasn't published a result yet.
    /// The receiver wakes (with `Err(RecvError)`) when the worker commits the result. Native
    /// callers `block_on` it and re-query; wasm callers drop it and fall back.
    Pending,
    /// No operation is found yet.
    Miss,
}

impl<K: AutotuneKey> TuneCache<K> {
    pub(crate) fn new(
        #[cfg_attr(not(serializable), allow(unused_variables))] name: &str,
        #[cfg_attr(not(serializable), allow(unused_variables))] device_id: &str,
    ) -> Self {
        #[cfg(persistence)]
        {
            use crate::runtime::config::{RudaRuntimeConfig, RuntimeConfig};
            let config = RudaRuntimeConfig::get();
            let namespace = Namespace::scoped("autotune", alloc::format!("{device_id}/{name}"));
            let table = namespace.as_str().into();
            let generation = ruda_core::environment::generation();
            let persistent_cache = (!config.autotune.disable_cache).then(|| {
                Store::new(StoreOptions::new().storage(namespace).cache(CacheOption::Lazy))
            });
            let mut cache = Self {
                in_memory_cache: HashMap::new(),
                hydrated: persistent_cache.is_none(),
                persistent_cache,
                table,
                generation,
            };
            cache.load();
            cache
        }
        #[cfg(all(std_io, not(persistence)))]
        {
            use crate::runtime::config::RuntimeConfig;
            use std::format;

            let config = crate::runtime::config::RudaRuntimeConfig::get();
            let persistent_cache = (!config.autotune.disable_cache).then(|| {
                let root = config.autotune.cache.root();
                Cache::new(
                    format!("{device_id}/{name}"),
                    ruda_core::cache::CacheOption::default().root(root).name("autotune"),
                )
            });
            let mut cache = TuneCache {
                in_memory_cache: HashMap::new(),
                persistent_cache,
            };
            cache.load();
            cache
        }

        #[cfg(not(serializable))]
        {
            TuneCache {
                in_memory_cache: HashMap::new(),
            }
        }
    }

    pub fn fastest(&self, key: &K) -> TuneCacheResult {
        #[cfg(persistence)]
        if self.environment_switched() {
            return TuneCacheResult::Miss;
        }
        let Some(val) = self.in_memory_cache.get(key) else {
            return TuneCacheResult::Miss;
        };

        let CacheEntry::Done {
            checksum,
            fastest_index,
        } = val
        else {
            // Pending: clone the receiver so the caller can subscribe to the in-flight tune.
            let CacheEntry::Pending = val else {
                unreachable!()
            };
            return TuneCacheResult::Pending;
        };

        if cfg!(serializable) {
            match checksum {
                ChecksumState::ToBeVerified(..) => TuneCacheResult::Unchecked, // Don't know yet.
                ChecksumState::NoMatch => TuneCacheResult::Miss,               // Can't use this.
                ChecksumState::Match => TuneCacheResult::Hit {
                    fastest_index: *fastest_index,
                },
            }
        } else {
            // Clippy;
            let _ = checksum;
            TuneCacheResult::Hit {
                fastest_index: *fastest_index,
            }
        }
    }

    #[cfg(serializable)]
    pub fn validate_checksum(&mut self, key: &K, checksum: &str) -> TuneCacheResult {
        self.sync_environment();
        let Some(val) = self.in_memory_cache.get_mut(key) else {
            return TuneCacheResult::Miss;
        };

        if let CacheEntry::Done {
            checksum: checksum_state,
            ..
        } = val
            && let ChecksumState::ToBeVerified(checksum_expected) = checksum_state
        {
            if checksum_expected == checksum {
                *checksum_state = ChecksumState::Match;
            } else {
                *checksum_state = ChecksumState::NoMatch;
            }
        }

        self.fastest(key)
    }

    pub(crate) fn sync_environment(&mut self) {
        #[cfg(persistence)]
        {
            if self.environment_switched() {
                self.generation = ruda_core::environment::generation();
                self.in_memory_cache.clear();
                self.hydrated = false;
            }
            if !self.hydrated {
                self.load();
            }
        }
    }

    #[cfg(persistence)]
    fn environment_switched(&self) -> bool {
        self.persistent_cache.is_some()
            && self.generation != ruda_core::environment::generation()
    }

    #[cfg(persistence)]
    pub(crate) fn table(&self) -> &str {
        &self.table
    }

    #[cfg(persistence)]
    pub(crate) fn persistent_enabled(&self) -> bool {
        self.persistent_cache.is_some()
    }

    /// Mark a key as being tuned. Used by [`Tuner::tune`] under the cache mutex so that
    /// concurrent callers see [`TuneCacheResult::Pending`] and wait on the same job instead of
    /// starting a second one. Returns `(Sender, Receiver)`:
    pub(crate) fn mark_pending(&mut self, key: K) {
        self.sync_environment();
        self.in_memory_cache.insert(key, CacheEntry::Pending);
    }

    pub(crate) fn cache_insert(&mut self, key: K, fastest_index: usize) {
        self.sync_environment();
        self.in_memory_cache.insert(
            key,
            CacheEntry::Done {
                checksum: ChecksumState::Match,
                fastest_index,
            },
        );
    }
}

#[cfg(serializable)]
impl<K: AutotuneKey> TuneCache<K> {
    pub(crate) fn persistent_cache_insert(
        &mut self,
        key: K,
        checksum: String,
        fastest_index: usize,
        results: Vec<AutotuneResult>,
        bounds: Option<super::Bounds>,
        limit: Option<core::time::Duration>,
    ) -> bool {
        self.sync_environment();
        let Some(cache) = self.persistent_cache.as_mut() else { return false; };
        if let Err(err) = cache.insert(
            PersistentCacheKey { key, checksum },
            PersistentCacheValue {
                fastest_index,
                results,
                bounds,
                limit,
            },
        ) {
            match err {
                #[cfg(all(std_io, not(persistence)))]
                CacheError::DuplicatedKey {
                    key,
                    value_previous,
                    value_updated,
                } => {
                    log::warn!(
                        "Autotune the same function multiple times for key {key:?} => old {value_previous:?}, new {value_updated:?}"
                    );
                }
                #[cfg(all(std_io, not(persistence)))]
                CacheError::KeyOutOfSync { .. } => {
                    // This is OK.
                }
                #[cfg(persistence)]
                StoreError::DuplicatedKey { key, value_previous, value_updated } => {
                    log::warn!("Autotune the same function multiple times for key {key:?} => old {value_previous:?}, new {value_updated:?}");
                }
                #[cfg(persistence)]
                StoreError::KeyOutOfSync { .. } => {}
                #[cfg(persistence)]
                StoreError::Backend { key, error } => {
                    log::warn!("Autotune cache write failed for {key:?}: {error}");
                }
            }
            return false;
        }
        true
    }

    /// Load persistent entries without retaining a second copy in the storage layer.
    pub(crate) fn load(&mut self) {
        #[cfg(persistence)]
        if self.hydrated {
            return;
        }
        let Some(cache) = self.persistent_cache.as_mut() else { return; };
        let mut loaded = 0;
        #[cfg(persistence)]
        cache.scan(|key, value| {
            loaded += 1;
            self.in_memory_cache.entry(key.key).or_insert(CacheEntry::Done {
                checksum: ChecksumState::ToBeVerified(key.checksum),
                fastest_index: value.fastest_index,
            });
        });
        #[cfg(all(std_io, not(persistence)))]
        cache.for_each(|key, value| {
            loaded += 1;
            self.in_memory_cache.insert(key.key.clone(), CacheEntry::Done {
                checksum: ChecksumState::ToBeVerified(key.checksum.clone()),
                fastest_index: value.fastest_index,
            });
        });
        #[cfg(persistence)]
        { self.hydrated = true; }
        log::info!("Loaded {loaded} autotune cached entries");
    }
}
