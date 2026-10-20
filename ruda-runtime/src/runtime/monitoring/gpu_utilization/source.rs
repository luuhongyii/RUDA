#[cfg(all(feature = "monitoring-gpu-engine-counters", target_os = "windows"))]
use ruda_core::ir::AdapterLuid;
use ruda_core::ir::PhysicalDevice;
#[cfg(all(feature = "monitoring-io-accelerator", target_os = "macos"))]
use ruda_core::ir::RegistryEntryId;
#[cfg(any(all(feature = "monitoring-amdgpu-busy-percent", target_os = "linux"), all(feature = "monitoring-intel-idle-residency", target_os = "linux"), all(feature = "monitoring-nvml", target_os = "linux")))]
use ruda_core::ir::{PciAddress, PciVendor};

use crate::runtime::monitoring::UtilizationUnavailable;

/// The counter a device's utilization is read from, among the ones this build compiles. The card
/// and the platform decide it, never the runtime: an NVIDIA card driven through wgpu reads the same
/// counter as through CUDA.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UtilizationSource {
    #[cfg(all(feature = "monitoring-nvml", target_os = "linux"))]
    Nvml(PciAddress),
    #[cfg(all(feature = "monitoring-amdgpu-busy-percent", target_os = "linux"))]
    AmdgpuBusyPercentFile(PciAddress),
    #[cfg(all(feature = "monitoring-intel-idle-residency", target_os = "linux"))]
    IntelIdleResidencyFiles(PciAddress),
    #[cfg(all(feature = "monitoring-gpu-engine-counters", target_os = "windows"))]
    GpuEngineCounters(AdapterLuid),
    /// Without a registry entry id, the machine's only accelerator.
    #[cfg(all(feature = "monitoring-io-accelerator", target_os = "macos"))]
    IoAcceleratorStatistics(Option<RegistryEntryId>),
    Unavailable(UtilizationUnavailable),
}

impl UtilizationSource {
    pub fn of_card(card: Option<&PhysicalDevice>) -> Self {
        match card {
            Some(card) => Self::counter_this_build_keeps_for(card),
            None => Self::Unavailable(UtilizationUnavailable::NoCard),
        }
    }

    #[cfg(any(all(feature = "monitoring-amdgpu-busy-percent", target_os = "linux"), all(feature = "monitoring-intel-idle-residency", target_os = "linux"), all(feature = "monitoring-nvml", target_os = "linux")))]
    fn counter_this_build_keeps_for(card: &PhysicalDevice) -> Self {
        let counter_at: fn(PciAddress) -> Self = match card.vendor {
            #[cfg(all(feature = "monitoring-nvml", target_os = "linux"))]
            Some(PciVendor::Nvidia) => Self::Nvml,
            #[cfg(all(feature = "monitoring-amdgpu-busy-percent", target_os = "linux"))]
            Some(PciVendor::Amd) => Self::AmdgpuBusyPercentFile,
            #[cfg(all(feature = "monitoring-intel-idle-residency", target_os = "linux"))]
            Some(PciVendor::Intel) => Self::IntelIdleResidencyFiles,
            vendor => return Self::Unavailable(UtilizationUnavailable::NoCounterForCard(vendor)),
        };
        match card.pci_address {
            Some(pci_address) => counter_at(pci_address),
            None => Self::Unavailable(UtilizationUnavailable::CardAddressNotReported),
        }
    }

    #[cfg(all(feature = "monitoring-gpu-engine-counters", target_os = "windows"))]
    fn counter_this_build_keeps_for(card: &PhysicalDevice) -> Self {
        match card.luid {
            Some(luid) => Self::GpuEngineCounters(luid),
            None => Self::Unavailable(UtilizationUnavailable::CardAddressNotReported),
        }
    }

    #[cfg(all(feature = "monitoring-io-accelerator", target_os = "macos"))]
    fn counter_this_build_keeps_for(card: &PhysicalDevice) -> Self {
        Self::IoAcceleratorStatistics(card.registry_entry_id)
    }
}
