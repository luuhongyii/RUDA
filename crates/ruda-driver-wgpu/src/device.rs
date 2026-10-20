use ruda_core::device::{Device, DeviceId};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Default)]
pub enum WgpuBackend {
    #[default]
    Auto,
    Vulkan,
    Metal,
    Dx12,
    Gl,
    WebGpu,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Default)]
pub enum WgpuDeviceKind {
    DiscreteGpu(usize),
    IntegratedGpu(usize),
    VirtualGpu(usize),
    Cpu,
    #[default]
    DefaultDevice,
    Existing(u32),
    Other(usize),
}

impl WgpuDeviceKind {
    pub const MAX_INDEX: usize = u16::MAX as usize;
}

/// The device struct when using the `wgpu` backend.
///
/// Note that you need to provide the device index when using a GPU backend.
///
/// # Example
///
/// ```ignore
/// use ruda_driver_wgpu::WgpuDevice;
///
/// let device_gpu_1 = WgpuDevice::DiscreteGpu(0); // First discrete GPU found.
/// let device_gpu_2 = WgpuDevice::DiscreteGpu(1);  // Second discrete GPU found.
/// ```
#[derive(Clone, Debug, Hash, PartialEq, Eq, Default)]
pub enum WgpuDevice {
    /// Discrete GPU with the given index. The index is the index of the discrete GPU in the list
    /// of all discrete GPUs found on the system.
    DiscreteGpu(usize),

    /// Integrated GPU with the given index. The index is the index of the integrated GPU in the
    /// list of all integrated GPUs found on the system.
    IntegratedGpu(usize),

    /// Virtual GPU with the given index. The index is the index of the virtual GPU in the list of
    /// all virtual GPUs found on the system.
    VirtualGpu(usize),

    /// CPU.
    Cpu,

    /// The best available device found with the current [graphics API](crate::GraphicsApi).
    ///
    /// This will prioritize GPUs wgpu recognizes as "high power". Additionally, you can override this using
    /// the `RUDA_WGPU_DEFAULT_DEVICE` environment variable. This variable is spelled as if i was a `WgpuDevice`,
    /// so for example `RUDA_WGPU_DEFAULT_DEVICE=IntegratedGpu(1)` or `RUDA_WGPU_DEFAULT_DEVICE=Cpu`
    #[default]
    DefaultDevice,

    /// Deprecated, use [`DefaultDevice`](WgpuDevice::DefaultDevice).
    #[deprecated]
    BestAvailable,

    /// Use an externally created, existing, wgpu setup. This is helpful when using `Ruda` in conjunction
    /// with some existing wgpu setup (eg. egui or bevy), as resources can be transferred in & out of `Ruda`.
    ///
    /// # Notes
    ///
    /// This can be initialized with [`init_device`](crate::runtime::init_device).
    Existing(u32),

    /// An adapter reported by the graphics API as an unclassified device.
    Other(usize),

    /// A hardware selector pinned to a graphics API. Construct with [`Self::on`].
    On(WgpuDeviceKind, WgpuBackend),
}

impl WgpuDevice {
    pub fn enumerate() -> Vec<Self> {
        Self::enumerate_on(WgpuBackend::Auto)
    }

    pub fn enumerate_on(backend: WgpuBackend) -> Vec<Self> {
        #[cfg(target_family = "wasm")]
        {
            match backend {
                WgpuBackend::Auto | WgpuBackend::WebGpu => vec![Self::DiscreteGpu(0).on(backend)],
                _ => Vec::new(),
            }
        }
        #[cfg(not(target_family = "wasm"))]
        {
            let device = Self::default().on(backend);
            let Ok(api) = ruda_core::future::block_on(crate::runtime::adapters::try_resolve_backend(&device)) else {
                return Vec::new();
            };
            <crate::WgpuRuntime as ruda_kernel::dsl::Runtime>::enumerate_all_devices(&api)
                .into_iter().map(|id| Self::from_id(id).on(backend)).collect()
        }
    }

    pub fn enumerate_like(&self) -> Vec<Self> {
        let reachable = Self::enumerate_on(self.backend());
        if self.kind() == WgpuDeviceKind::DefaultDevice && !reachable.is_empty() {
            return vec![Self::DefaultDevice.on(self.backend())];
        }
        reachable.into_iter().filter(|device| {
            core::mem::discriminant(&device.kind()) == core::mem::discriminant(&self.kind())
        }).collect()
    }

    pub fn find(&self) -> Result<(), usize> {
        #[cfg(target_family = "wasm")]
        if self.kind() == WgpuDeviceKind::IntegratedGpu(0) && !Self::enumerate_on(self.backend()).is_empty() {
            return Ok(());
        }
        let devices = self.enumerate_like();
        if devices.iter().any(|device| device.kind() == self.kind()) { Ok(()) } else { Err(devices.len()) }
    }

    pub fn identity(&self) -> ruda_core::ir::DeviceIdentity {
        <crate::WgpuRuntime as ruda_kernel::dsl::Runtime>::client(self).properties().identity.clone()
    }

    pub fn new(kind: WgpuDeviceKind) -> Self {
        match kind {
            WgpuDeviceKind::DiscreteGpu(i) => Self::DiscreteGpu(i),
            WgpuDeviceKind::IntegratedGpu(i) => Self::IntegratedGpu(i),
            WgpuDeviceKind::VirtualGpu(i) => Self::VirtualGpu(i),
            WgpuDeviceKind::Cpu => Self::Cpu,
            WgpuDeviceKind::DefaultDevice => Self::DefaultDevice,
            WgpuDeviceKind::Existing(i) => Self::Existing(i),
            WgpuDeviceKind::Other(i) => Self::Other(i),
        }
    }

    pub fn kind(&self) -> WgpuDeviceKind {
        #[allow(deprecated)]
        match *self {
            Self::DiscreteGpu(i) => WgpuDeviceKind::DiscreteGpu(i),
            Self::IntegratedGpu(i) => WgpuDeviceKind::IntegratedGpu(i),
            Self::VirtualGpu(i) => WgpuDeviceKind::VirtualGpu(i),
            Self::Cpu => WgpuDeviceKind::Cpu,
            Self::DefaultDevice | Self::BestAvailable => WgpuDeviceKind::DefaultDevice,
            Self::Existing(i) => WgpuDeviceKind::Existing(i),
            Self::Other(i) => WgpuDeviceKind::Other(i),
            Self::On(kind, _) => kind,
        }
    }

    pub fn backend(&self) -> WgpuBackend {
        match *self {
            Self::On(WgpuDeviceKind::Existing(_), _) => WgpuBackend::Auto,
            Self::On(_, backend) => backend,
            _ => WgpuBackend::Auto,
        }
    }

    pub fn on(self, backend: WgpuBackend) -> Self {
        let kind = self.kind();
        if backend == WgpuBackend::Auto || matches!(kind, WgpuDeviceKind::Existing(_)) {
            Self::new(kind)
        } else {
            Self::On(kind, backend)
        }
    }
}

impl Device for WgpuDevice {
    fn from_id(device_id: DeviceId) -> Self {
        let device = match device_id.type_id & 7 {
            0 => Self::DiscreteGpu(device_id.index_id as usize),
            1 => Self::IntegratedGpu(device_id.index_id as usize),
            2 => Self::VirtualGpu(device_id.index_id as usize),
            3 => Self::Cpu,
            4 => Self::DefaultDevice,
            5 => Self::Existing(device_id.index_id as u32),
            6 => Self::Other(device_id.index_id as usize),
            _ => Self::DefaultDevice,
        };
        device.on(match device_id.type_id >> 3 {
            1 => WgpuBackend::Vulkan,
            2 => WgpuBackend::Metal,
            3 => WgpuBackend::Dx12,
            4 => WgpuBackend::Gl,
            5 => WgpuBackend::WebGpu,
            _ => WgpuBackend::Auto,
        })
    }

    fn to_id(&self) -> DeviceId {
        let (kind, index) = match self.kind() {
            WgpuDeviceKind::DiscreteGpu(i) => (0, i),
            WgpuDeviceKind::IntegratedGpu(i) => (1, i),
            WgpuDeviceKind::VirtualGpu(i) => (2, i),
            WgpuDeviceKind::Cpu => (3, 0),
            WgpuDeviceKind::DefaultDevice => (4, 0),
            WgpuDeviceKind::Existing(i) => (5, i as usize),
            WgpuDeviceKind::Other(i) => (6, i),
        };
        let index = u16::try_from(index).expect("Wgpu device index exceeds the device ID range");
        let backend = match self.backend() {
            WgpuBackend::Auto => 0,
            WgpuBackend::Vulkan => 1,
            WgpuBackend::Metal => 2,
            WgpuBackend::Dx12 => 3,
            WgpuBackend::Gl => 4,
            WgpuBackend::WebGpu => 5,
        };
        DeviceId::new(kind | (backend << 3), index)
    }
}
