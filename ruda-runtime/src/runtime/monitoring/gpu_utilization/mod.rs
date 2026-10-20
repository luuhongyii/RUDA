mod card_counters;

#[cfg(any(all(feature = "monitoring-amdgpu-busy-percent", target_os = "linux"), all(feature = "monitoring-intel-idle-residency", target_os = "linux"), all(feature = "monitoring-nvml", target_os = "linux"), all(feature = "monitoring-gpu-engine-counters", target_os = "windows"), all(feature = "monitoring-io-accelerator", target_os = "macos")))]
mod opened;
#[cfg(any(all(feature = "monitoring-amdgpu-busy-percent", target_os = "linux"), all(feature = "monitoring-intel-idle-residency", target_os = "linux"), all(feature = "monitoring-nvml", target_os = "linux"), all(feature = "monitoring-gpu-engine-counters", target_os = "windows"), all(feature = "monitoring-io-accelerator", target_os = "macos")))]
mod source;

#[cfg(all(feature = "monitoring-amdgpu-busy-percent", target_os = "linux"))]
mod amdgpu_busy_percent;
#[cfg(all(feature = "monitoring-intel-idle-residency", target_os = "linux"))]
mod intel_idle_residency;
#[cfg(all(feature = "monitoring-nvml", target_os = "linux"))]
mod nvml;

#[cfg(all(feature = "monitoring-gpu-engine-counters", target_os = "windows"))]
mod gpu_engine_counters;
#[cfg(any(all(feature = "monitoring-gpu-engine-counters", target_os = "windows")))]
mod gpu_engine_instance;

#[cfg(all(feature = "monitoring-io-accelerator", target_os = "macos"))]
mod io_accelerator;

pub use card_counters::CardCounters;
