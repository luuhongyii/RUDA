use alloc::{format, string::String, sync::Arc};
use hashbrown::HashMap;
use ruda_core::ir::{DeviceIdentity, DeviceProperties};
use spin::Mutex;

use super::{ThroughputKey, ThroughputValue};
#[cfg(all(std_io, not(persistence)))]
use ruda_core::cache::{Cache, CacheOption};
#[cfg(persistence)]
use ruda_core::environment::persistence::{Namespace, Store, StoreOptions};

static GLOBAL_CACHE: Mutex<Option<HashMap<String, Arc<Mutex<ThroughputCache>>>>> =
    Mutex::new(None);

/// Successful throughput measurements, shared by identical device parts.
pub struct ThroughputCache {
    #[cfg(persistence)]
    cache: Store<ThroughputKey, ThroughputValue>,
    #[cfg(not(serializable))]
    cache: HashMap<ThroughputKey, ThroughputValue>,
    #[cfg(all(std_io, not(persistence)))]
    cache: Cache<ThroughputKey, ThroughputValue>,
}

impl ThroughputCache {
    pub fn get_for_device(runtime: &str, properties: &DeviceProperties) -> Arc<Mutex<Self>> {
        let name = device_key(
            runtime,
            &properties.identity,
            properties.memory.max_page_size,
            properties.hardware.num_cpu_cores
                .or(properties.hardware.num_streaming_multiprocessors)
                .unwrap_or(0),
        );
        let mut caches = GLOBAL_CACHE.lock();
        caches.get_or_insert_with(HashMap::new)
            .entry(name.clone())
            .or_insert_with(|| Arc::new(Mutex::new(Self::new(&name))))
            .clone()
    }

    pub fn new(#[cfg_attr(not(serializable), allow(unused_variables))] name: &str) -> Self {
        #[cfg(persistence)]
        {
            use crate::runtime::config::{RudaRuntimeConfig, RuntimeConfig};
            let config = RudaRuntimeConfig::get();
            let options = if config.throughput.disable_cache {
                StoreOptions::new()
            } else {
                drop_earlier_environment_generations();
                StoreOptions::new().storage(Namespace::scoped(
                    "throughput", format!("probe-v{}/{name}", super::PROBE_VERSION),
                ))
            };
            Self { cache: Store::new(options) }
        }
        #[cfg(not(serializable))]
        { Self { cache: HashMap::new() } }
        #[cfg(all(std_io, not(persistence)))]
        {
            let root = dirs::cache_dir().unwrap_or_else(|| {
                dirs::home_dir().expect("A cache or home directory must exist").join(".cache")
            });
            let scope = root.join("ruda-throughput").join(env!("CARGO_PKG_VERSION"));
            drop_earlier_generations(&scope);
            let name = segment(name);
            Self {
                cache: Cache::new(
                    format!("probe-v{}/{name}", super::PROBE_VERSION),
                    CacheOption::default().root(root).name("ruda-throughput")
                        .version(env!("CARGO_PKG_VERSION")),
                ),
            }
        }
    }

    pub fn insert(&mut self, key: ThroughputKey, value: ThroughputValue) {
        #[cfg(persistence)]
        if let Err(error) = self.cache.insert(key, value) {
            log::warn!("Throughput environment write declined: {}", error.reason());
        }
        #[cfg(all(std_io, not(persistence)))]
        if let Err(error) = self.cache.insert(key, value) {
            log::warn!("Concurrent throughput measurement, keeping the existing value: {error:?}");
        }
        #[cfg(not(serializable))]
        self.cache.insert(key, value);
    }

    pub fn get(&self, key: &ThroughputKey) -> Option<&ThroughputValue> {
        self.cache.get(key)
    }

    /// Reopen and rehydrate the active environment before a measurement lookup.
    pub fn get_current(&mut self, key: &ThroughputKey) -> Option<&ThroughputValue> {
        #[cfg(persistence)]
        {
            if self.cache.namespace().is_some() {
                drop_earlier_environment_generations();
            }
            self.cache.get_mut(key).map(|value| &*value)
        }
        #[cfg(not(persistence))]
        { self.cache.get(key) }
    }
}

fn segment(text: &str) -> String {
    text.replace(|c: char| !c.is_ascii_alphanumeric(), "-")
}

fn device_key(runtime: &str, identity: &DeviceIdentity, capacity: u64, parallelism: u32) -> String {
    format!("{}_{}_{}_mem{capacity}_par{parallelism}",
        segment(runtime), segment(&identity.fingerprint), segment(&identity.name))
}

#[cfg(all(std_io, not(persistence)))]
fn drop_earlier_generations(scope: &std::path::Path) {
    use core::sync::atomic::{AtomicBool, Ordering};
    static DROPPED: AtomicBool = AtomicBool::new(false);
    if DROPPED.swap(true, Ordering::Relaxed) { return; }
    let Ok(entries) = std::fs::read_dir(scope) else { return; };
    for entry in entries.flatten() {
        if !entry.file_type().is_ok_and(|kind| kind.is_dir()) { continue; }
        let name = entry.file_name();
        let Some(generation) = name.to_str()
            .and_then(|name| name.strip_prefix("probe-v"))
            .and_then(|number| number.parse::<u32>().ok()) else { continue; };
        if generation < super::PROBE_VERSION {
            if let Err(error) = std::fs::remove_dir_all(entry.path()) {
                log::warn!("Could not remove throughput cache generation {generation}: {error}");
            }
        }
    }
}

#[cfg(persistence)]
fn drop_earlier_environment_generations() {
    use ruda_core::environment::{self, persistence};
    static DROPPED: Mutex<Option<u32>> = Mutex::new(None);
    let mut dropped = DROPPED.lock();
    let generation = environment::generation();
    if *dropped == Some(generation) { return; }
    let scope = Namespace::scoped("throughput", "");
    for namespace in persistence::namespaces() {
        let Some(suffix) = namespace.strip_prefix(scope.as_str()) else { continue; };
        let version = suffix.strip_prefix("probe-v")
            .and_then(|text| text.split('/').next())
            .and_then(|text| text.parse::<u32>().ok()).unwrap_or(0);
        if version < super::PROBE_VERSION {
            persistence::open(&namespace).purge();
        }
    }
    *dropped = Some(generation);
}
