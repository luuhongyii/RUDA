use super::*;

/// A complete setup used to run wgpu.
///
/// These can either be created with [`init_setup`] or [`init_setup_async`].
#[derive(Clone, Debug)]
pub struct WgpuSetup {
    /// The underlying wgpu instance.
    pub instance: wgpu::Instance,
    /// The selected 'adapter'. This corresponds to a physical device.
    pub adapter: wgpu::Adapter,
    /// The wgpu device Ruda will use. Nb: There can only be one device per adapter.
    pub device: wgpu::Device,
    /// The queue Ruda commands will be submitted to.
    pub queue: wgpu::Queue,
    /// The backend used by the setup.
    pub backend: wgpu::Backend,
}

/// Create a [`WgpuDevice`] on an existing [`WgpuSetup`].
/// Useful when you want to share a device between `Ruda` and other wgpu-dependent libraries.
///
/// # Note
///
/// Please **do not** to call on the same [`setup`](WgpuSetup) more than once.
///
/// This function generates a new, globally unique ID for the device every time it is called,
/// even if called on the same device multiple times.
pub fn init_device(setup: WgpuSetup, options: RuntimeOptions) -> WgpuDevice {
    try_init_device(setup, options).expect("Unable to initialize Wgpu device")
}

pub fn try_init_device(setup: WgpuSetup, options: RuntimeOptions) -> Result<WgpuDevice, WgpuInitError> {
    try_init_device_with_api::<AutoGraphicsApi>(setup, options)
}

pub fn try_init_device_with_api<G: GraphicsApi>(
    setup: WgpuSetup,
    options: RuntimeOptions,
) -> Result<WgpuDevice, WgpuInitError> {
    use core::sync::atomic::{AtomicU32, Ordering};
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    let requested = G::backend_for(&WgpuDevice::default());
    let id = COUNTER.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| {
        (id <= u16::MAX as u32).then_some(id + 1)
    }).map_err(|_| WgpuInitError::Registration { message: "Existing Wgpu device IDs exhausted".into() })?;
    let device = WgpuDevice::Existing(id);
    let server = try_create_server(device.to_id(), setup, options, requested)?;
    register(&device, server)?;
    Ok(device)
}

/// Like [`init_setup_async`], but synchronous.
/// On wasm, it is necessary to use [`init_setup_async`] instead.
pub fn init_setup<G: GraphicsApi>(device: &WgpuDevice, options: RuntimeOptions) -> WgpuSetup {
    try_init_setup::<G>(device, options).expect("Unable to initialize Wgpu setup")
}

pub fn try_init_setup<G: GraphicsApi>(
    device: &WgpuDevice,
    options: RuntimeOptions,
) -> Result<WgpuSetup, WgpuInitError> {
    #[cfg(target_family = "wasm")]
    {
        let _ = (device, options);
        Err(WgpuInitError::InvalidConfiguration {
            message: "Synchronous Wgpu setup is unsupported on wasm; use try_init_setup_async".into(),
        })
    }
    #[cfg(not(target_family = "wasm"))]
    {
        future::block_on(try_init_setup_async::<G>(device, options))
    }
}

/// Initialize a client on the given device with the given options.
/// This function is useful to configure the runtime options
/// or to pick a different graphics API.
pub async fn init_setup_async<G: GraphicsApi>(
    device: &WgpuDevice,
    options: RuntimeOptions,
) -> WgpuSetup {
    try_init_setup_async::<G>(device, options).await.expect("Unable to initialize Wgpu setup")
}

pub async fn try_init_setup_async<G: GraphicsApi>(
    device: &WgpuDevice,
    options: RuntimeOptions,
) -> Result<WgpuSetup, WgpuInitError> {
    options.validate()?;
    let requested = G::backend_for(device);
    if device.backend() != WgpuBackend::Auto && device.backend() != requested {
        return Err(WgpuInitError::InvalidConfiguration {
            message: format!("Graphics API {requested:?} conflicts with device API {:?}", device.backend()),
        });
    }
    let selected = device.clone().on(requested);
    let setup = try_create_setup_for_device(&selected).await?;
    let server = try_create_server(device.to_id(), setup.clone(), options, requested)?;
    register(device, server)?;
    Ok(setup)
}

fn register(device: &WgpuDevice, server: WgpuServer) -> Result<(), WgpuInitError> {
    ComputeClient::<WgpuRuntime>::try_init(device, server)
        .map(|_| ())
        .map_err(|err| WgpuInitError::Registration { message: format!("{err:?}") })
}

pub(crate) async fn try_create_setup_for_device(device: &WgpuDevice) -> Result<WgpuSetup, WgpuInitError> {
    let selected = if device.kind() == WgpuDeviceKind::DefaultDevice {
        super::adapters::try_device_override()?
            .map(|kind| WgpuDevice::new(kind).on(device.backend()))
    } else {
        None
    };
    let device = selected.as_ref().unwrap_or(device);
    match device.kind() {
        WgpuDeviceKind::DiscreteGpu(index)
        | WgpuDeviceKind::IntegratedGpu(index)
        | WgpuDeviceKind::VirtualGpu(index)
        | WgpuDeviceKind::Other(index) if index > WgpuDeviceKind::MAX_INDEX => {
            return Err(WgpuInitError::InvalidConfiguration {
                message: format!("Adapter index {index} exceeds {}", WgpuDeviceKind::MAX_INDEX),
            });
        }
        WgpuDeviceKind::Existing(_) => {
            return Err(WgpuInitError::InvalidConfiguration { message: "an Existing ID cannot acquire an adapter".into() });
        }
        _ => {}
    }
    let backend = super::adapters::try_resolve_backend(device).await?;
    let (instance, adapter) = super::adapters::request_adapter(device, backend).await?;
    let (device, queue) = backend::try_request_device(&adapter).await?;
    let backend = adapter.get_info().backend;
    log::info!("Created wgpu compute server on adapter {:?}", adapter.get_info());
    Ok(WgpuSetup { instance, adapter, device, queue, backend })
}

impl WgpuSetup {
    pub(super) fn validate(&self, requested: WgpuBackend) -> Result<(), WgpuInitError> {
        if requested != WgpuBackend::Auto && !super::adapters::backend_candidates(requested).contains(&self.backend) {
            return Err(WgpuInitError::InvalidConfiguration {
                message: format!("Setup backend {:?} does not match {requested:?}", self.backend),
            });
        }
        if self.backend == wgpu::Backend::Noop
            || self.backend != self.adapter.get_info().backend
            || self.backend != self.device.adapter_info().backend
        {
            return Err(WgpuInitError::InvalidConfiguration {
                message: "Setup must contain a device and adapter on the declared graphics API".into(),
            });
        }
        let limits = self.device.limits();
        if limits.max_compute_invocations_per_workgroup == 0 || limits.max_storage_buffers_per_shader_stage < 2 {
            return Err(WgpuInitError::InvalidConfiguration {
                message: "Wgpu compute requires workgroup support and at least two storage-buffer bindings".into(),
            });
        }
        Ok(())
    }
}
