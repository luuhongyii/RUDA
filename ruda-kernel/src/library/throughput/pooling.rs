use core::any::TypeId;
use ruda_core::device::DeviceId;
pub(super) type ServiceId = (TypeId, DeviceId);

pub(super) fn service_id<R: Runtime>(client: &ComputeClient<R>) -> ServiceId {
    (TypeId::of::<R::Server>(), client.device_id())
}
use hashbrown::HashMap;
use spin::Mutex;
use ruda::runtime::{backend::Runtime, client::ComputeClient};

/// How many sweeps are holding each device's pools open.
static POOLED_PROBES: Mutex<Option<HashMap<ServiceId, usize>>> = Mutex::new(None);

/// A device whose probes leave their pools with the allocator, for as long as
/// one of these is alive, since faulting the next one back in costs more than
/// the measurement. Counted per device, so a sweep that ends cannot release the
/// pool another is still measuring against.
pub(super) struct PooledProbes {
    service: ServiceId,
}

impl PooledProbes {
    pub(super) fn enter<R: Runtime>(client: &ComputeClient<R>) -> Self {
        Self::enter_service(service_id(client))
    }

    fn enter_service(service: ServiceId) -> Self {
        let mut pooled = POOLED_PROBES.lock();

        *pooled
            .get_or_insert_with(HashMap::new)
            .entry(service)
            .or_insert(0) += 1;

        Self { service }
    }

    pub(super) fn cleanup_unless_held<R: Runtime>(client: &ComputeClient<R>) {
        // A cleanup refused because a stream records a graph leaves the memory
        // for the next one; the sweep needs nothing from it.
        Self::cleanup_unless_held_by(service_id(client), || {
            let _ = client.memory_cleanup();
        });
    }

    /// `release` runs with the lock dropped: it blocks on the device thread,
    /// which takes this lock itself when it autotunes.
    fn cleanup_unless_held_by(service: ServiceId, release: impl FnOnce()) {
        let held = {
            let pooled = POOLED_PROBES.lock();
            Self::held_by(&pooled, service)
        };

        if !held {
            release();
        }
    }

    fn held_by(pooled: &Option<HashMap<ServiceId, usize>>, service: ServiceId) -> bool {
        pooled
            .as_ref()
            .is_some_and(|sweeps| sweeps.contains_key(&service))
    }
}

impl Drop for PooledProbes {
    fn drop(&mut self) {
        let mut pooled = POOLED_PROBES.lock();
        let Some(sweeps) = pooled.as_mut() else {
            return;
        };

        if let Some(holders) = sweeps.get_mut(&self.service) {
            *holders -= 1;

            if *holders == 0 {
                sweeps.remove(&self.service);
            }
        }
    }
}
