#[cfg(any(all(feature = "monitoring-amdgpu-busy-percent", target_os = "linux"), all(feature = "monitoring-intel-idle-residency", target_os = "linux"), all(feature = "monitoring-nvml", target_os = "linux"), all(feature = "monitoring-gpu-engine-counters", target_os = "windows"), all(feature = "monitoring-io-accelerator", target_os = "macos")))]
use alloc::{sync::Arc, vec::Vec};
#[cfg(any(all(feature = "monitoring-amdgpu-busy-percent", target_os = "linux"), all(feature = "monitoring-intel-idle-residency", target_os = "linux"), all(feature = "monitoring-nvml", target_os = "linux"), all(feature = "monitoring-gpu-engine-counters", target_os = "windows"), all(feature = "monitoring-io-accelerator", target_os = "macos")))]
use std::sync::{Mutex, PoisonError};

use ruda_core::ir::PhysicalDevice;

#[cfg(any(all(feature = "monitoring-amdgpu-busy-percent", target_os = "linux"), all(feature = "monitoring-intel-idle-residency", target_os = "linux"), all(feature = "monitoring-nvml", target_os = "linux"), all(feature = "monitoring-gpu-engine-counters", target_os = "windows"), all(feature = "monitoring-io-accelerator", target_os = "macos")))]
use super::{opened::OpenedCounter, source::UtilizationSource};
use crate::runtime::monitoring::{DeviceUtilization, UtilizationUnavailable};

/// The counters of the cards the GPU runtimes drive, each opened the first time its card is asked
/// and kept for the life of the process: opening is the costly half, loading NVML or opening a
/// query.
pub struct CardCounters;

#[cfg(any(all(feature = "monitoring-amdgpu-busy-percent", target_os = "linux"), all(feature = "monitoring-intel-idle-residency", target_os = "linux"), all(feature = "monitoring-nvml", target_os = "linux"), all(feature = "monitoring-gpu-engine-counters", target_os = "windows"), all(feature = "monitoring-io-accelerator", target_os = "macos")))]
static OPENED_COUNTERS: Mutex<Vec<(UtilizationSource, Arc<OpenedCounter>)>> =
    Mutex::new(Vec::new());

impl CardCounters {
    /// How busy `card` is, the card a runtime reports its device runs on. The card and the
    /// platform decide which counter answers, never the runtime: an NVIDIA card reads the same
    /// through CUDA as through wgpu.
    #[cfg(any(all(feature = "monitoring-amdgpu-busy-percent", target_os = "linux"), all(feature = "monitoring-intel-idle-residency", target_os = "linux"), all(feature = "monitoring-nvml", target_os = "linux"), all(feature = "monitoring-gpu-engine-counters", target_os = "windows"), all(feature = "monitoring-io-accelerator", target_os = "macos")))]
    pub fn read(
        card: Option<&PhysicalDevice>,
    ) -> Result<DeviceUtilization, UtilizationUnavailable> {
        Self::opened_for(UtilizationSource::of_card(card)).read()
    }

    /// How busy `card` is: never known to a build that compiles no counter for this platform.
    #[cfg(not(any(all(feature = "monitoring-amdgpu-busy-percent", target_os = "linux"), all(feature = "monitoring-intel-idle-residency", target_os = "linux"), all(feature = "monitoring-nvml", target_os = "linux"), all(feature = "monitoring-gpu-engine-counters", target_os = "windows"), all(feature = "monitoring-io-accelerator", target_os = "macos"))))]
    pub fn read(
        card: Option<&PhysicalDevice>,
    ) -> Result<DeviceUtilization, UtilizationUnavailable> {
        Err(match card {
            Some(card) => UtilizationUnavailable::NoCounterForCard(card.vendor),
            None => UtilizationUnavailable::NoCard,
        })
    }

    /// Keyed by the counter rather than the device, so the devices of every runtime on one card
    /// share it.
    #[cfg(any(all(feature = "monitoring-amdgpu-busy-percent", target_os = "linux"), all(feature = "monitoring-intel-idle-residency", target_os = "linux"), all(feature = "monitoring-nvml", target_os = "linux"), all(feature = "monitoring-gpu-engine-counters", target_os = "windows"), all(feature = "monitoring-io-accelerator", target_os = "macos")))]
    fn opened_for(source: UtilizationSource) -> Arc<OpenedCounter> {
        let mut opened_counters = OPENED_COUNTERS
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some((_, opened)) = opened_counters
            .iter()
            .find(|(opened_source, _)| *opened_source == source)
        {
            return opened.clone();
        }
        let opened = Arc::new(OpenedCounter::open(source.clone()));
        opened_counters.push((source, opened.clone()));
        opened
    }
}
