use crate::{
    AutoCompiler, AutoGraphicsApi, GraphicsApi, WgpuBackend, WgpuDevice, WgpuDeviceKind, backend, execution::WgpuServer,
    contiguous_strides,
};
use ruda_core::device::{Device, DeviceService};
use ruda_core::{future, profile::TimingMethod};
use ruda_kernel::dsl::device::{DeviceId, ServerUtilitiesHandle};
use ruda_kernel::dsl::server::ServerUtilities;
use ruda_kernel::dsl::zspace::{Shape, Strides};
use ruda_kernel::dsl::{Runtime, ir::TargetProperties};
use ruda_core::ir::{DeviceProperties, HardwareProperties, MemoryDeviceProperties};
use ruda::runtime::allocator::ContiguousMemoryLayoutPolicy;
#[cfg(not(feature = "vulkan-validate"))]
use ruda::runtime::logging::ProfileLevel;
pub use ruda::runtime::memory_management::MemoryConfiguration;
use ruda::runtime::{client::ComputeClient, logging::ServerLogger};
use wgpu::{InstanceFlags, RequestAdapterOptions};

/// Runtime that uses the [wgpu] crate with the wgsl compiler. This is used in the Wgpu backend.
/// For advanced configuration, use [`init_setup`] to pass in runtime options or to select a
/// specific graphics API.
#[derive(Debug, Clone)]
pub struct WgpuRuntime;

impl Runtime for WgpuRuntime {
    type Compiler = AutoCompiler;
    type Server = WgpuServer;
    type Device = WgpuDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self> {
        ComputeClient::load(device)
    }

    fn name(client: &ComputeClient<Self>) -> &'static str {
        client.runtime_name().expect("Wgpu server must report its selected compiler")
    }

    fn utilization(
        device: &Self::Device,
    ) -> Result<ruda::runtime::backend::DeviceUtilization, ruda::runtime::backend::UtilizationUnavailable> {
        ruda::runtime::monitoring::gpu_utilization::CardCounters::read(
            Self::client(device).properties().identity.physical.as_ref(),
        )
    }

    fn max_ruda_count() -> (u32, u32, u32) {
        let max_dim = u16::MAX as u32;
        (max_dim, max_dim, max_dim)
    }

    fn can_read_tensor(shape: &Shape, strides: &Strides) -> bool {
        if shape.is_empty() {
            return true;
        }

        for (&expected, &stride) in contiguous_strides(shape).iter().zip(strides.iter()) {
            if expected != stride {
                return false;
            }
        }

        true
    }

    fn target_properties() -> TargetProperties {
        TargetProperties {
            // Values are irrelevant, since no wgsl backends currently support manual mma
            mma: Default::default(),
        }
    }

    fn enumerate_devices(type_id: u16, info: &wgpu::Backend) -> Vec<DeviceId> {
        let selection = WgpuDevice::from_id(DeviceId::new(type_id, 0));
        if selection.backend() != WgpuBackend::Auto
            && !adapters::backend_candidates(selection.backend()).contains(info)
        {
            return Vec::new();
        }
        Self::enumerate_all_devices(info).into_iter()
            .filter(|id| (type_id & 7) == 4 || (id.type_id & 7) == (type_id & 7))
            .collect()
    }

    fn enumerate_all_devices(info: &wgpu::Backend) -> Vec<DeviceId> {
        let api = match info {
            wgpu::Backend::Vulkan => WgpuBackend::Vulkan,
            wgpu::Backend::Metal => WgpuBackend::Metal,
            wgpu::Backend::Dx12 => WgpuBackend::Dx12,
            wgpu::Backend::Gl => WgpuBackend::Gl,
            wgpu::Backend::BrowserWebGpu => WgpuBackend::WebGpu,
            wgpu::Backend::Noop => return Vec::new(),
        };
        #[cfg(target_family = "wasm")]
        {
            vec![WgpuDevice::DefaultDevice.on(api).to_id()]
        }
        #[cfg(not(target_family = "wasm"))]
        {
            if !wgpu::Instance::enabled_backend_features().contains((*info).into()) {
                return Vec::new();
            }
            let instance = adapters::instance_for(*info);
            let found = future::block_on(instance.enumerate_adapters((*info).into()));
            let mut indices = [0usize; 7];
            found.into_iter().filter_map(|adapter| {
                let kind = match adapter.get_info().device_type {
                    wgpu::DeviceType::DiscreteGpu => 0,
                    wgpu::DeviceType::IntegratedGpu => 1,
                    wgpu::DeviceType::VirtualGpu => 2,
                    wgpu::DeviceType::Cpu => 3,
                    wgpu::DeviceType::Other => 6,
                };
                let index = indices[kind];
                indices[kind] += 1;
                if index > WgpuDeviceKind::MAX_INDEX || (kind == 3 && index != 0) {
                    return None;
                }
                Some(WgpuDevice::from_id(DeviceId::new(kind as u16, index as u16)).on(api).to_id())
            }).collect()
        }
    }
}

/// The values that control how a WGPU Runtime will perform its calculations.
pub struct RuntimeOptions {
    /// Control the amount of compute tasks to be aggregated into a single GPU command.
    pub tasks_max: usize,
    /// Configures the memory management.
    pub memory_config: MemoryConfiguration,
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        Self::try_default().expect("Invalid Wgpu runtime options")
    }
}

impl RuntimeOptions {
    pub fn try_default() -> Result<Self, WgpuInitError> {
        Self::with_tasks_max(None)
    }

    pub fn with_tasks_max(tasks_max: Option<usize>) -> Result<Self, WgpuInitError> {
        let tasks_max = match tasks_max {
            Some(value) => value,
            None => match std::env::var("RUDA_WGPU_MAX_TASKS") {
                Ok(value) => value.parse::<usize>().map_err(|_| WgpuInitError::InvalidConfiguration {
                    message: format!("RUDA_WGPU_MAX_TASKS must be a positive integer, got {value:?}"),
                })?,
                Err(std::env::VarError::NotPresent) => 32,
                Err(err) => return Err(WgpuInitError::InvalidConfiguration { message: err.to_string() }),
            },
        };
        let options = Self {
            tasks_max,
            memory_config: MemoryConfiguration::default(),
        };
        options.validate()?;
        Ok(options)
    }

    pub(super) fn validate(&self) -> Result<(), WgpuInitError> {
        if self.tasks_max == 0 {
            return Err(WgpuInitError::InvalidConfiguration {
                message: "tasks_max must be greater than zero".into(),
            });
        }
        Ok(())
    }
}

pub(crate) mod adapters;
mod device_service;
mod error;
mod setup;

pub use error::WgpuInitError;
pub use setup::{WgpuSetup, init_device, init_setup, init_setup_async, try_init_device,
    try_init_device_with_api, try_init_setup, try_init_setup_async};
pub(crate) use setup::try_create_setup_for_device;
pub(crate) use device_service::try_create_server;
