use super::*;

pub(crate) fn backend_candidates(backend: WgpuBackend) -> Vec<wgpu::Backend> {
    match backend {
        WgpuBackend::Auto => AutoGraphicsApi::chain(),
        WgpuBackend::Vulkan => vec![wgpu::Backend::Vulkan],
        WgpuBackend::Metal => vec![wgpu::Backend::Metal],
        WgpuBackend::Dx12 => vec![wgpu::Backend::Dx12],
        WgpuBackend::Gl => vec![wgpu::Backend::Gl],
        WgpuBackend::WebGpu => vec![wgpu::Backend::BrowserWebGpu],
    }
}

pub(crate) fn instance_for(backend: wgpu::Backend) -> wgpu::Instance {
    #[cfg(not(feature = "vulkan-validate"))]
    let instance_flags = {
        let debug = ServerLogger::default();
        match (debug.profile_level(), debug.compilation_activated()) {
            (Some(ProfileLevel::Full), _) => InstanceFlags::advanced_debugging(),
            (_, true) => InstanceFlags::debugging(),
            (_, false) => InstanceFlags::default(),
        }
    };
    #[cfg(feature = "vulkan-validate")]
    let instance_flags = InstanceFlags::advanced_debugging();
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: backend.into(),
        flags: instance_flags,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    })
}

pub(crate) async fn try_resolve_backend(device: &WgpuDevice) -> Result<wgpu::Backend, WgpuInitError> {
    let enabled = wgpu::Instance::enabled_backend_features();
    let candidates = backend_candidates(device.backend());
    if !candidates.iter().any(|api| enabled.contains((*api).into())) {
        return Err(WgpuInitError::UnsupportedGraphicsApi { api: device.backend() });
    }
    let mut fallback = None;
    for api in candidates {
        if !enabled.contains(api.into()) {
            continue;
        }
        #[cfg(not(target_family = "wasm"))]
        {
            let adapters = instance_for(api).enumerate_adapters(api.into()).await;
            if adapters.iter().any(|adapter| adapter.get_info().device_type != wgpu::DeviceType::Cpu) {
                return Ok(api);
            }
            if fallback.is_none() && !adapters.is_empty() {
                fallback = Some(api);
            }
        }
        #[cfg(target_family = "wasm")]
        if api == wgpu::Backend::BrowserWebGpu {
            fallback = Some(api);
            break;
        }
    }
    fallback.ok_or_else(|| WgpuInitError::AdapterUnavailable {
        device: device.clone(),
        message: "no adapter found on the requested graphics API".into(),
    })
}

pub(crate) fn resolve_backend(backend: WgpuBackend) -> wgpu::Backend {
    #[cfg(not(target_family = "wasm"))]
    {
        future::block_on(try_resolve_backend(&WgpuDevice::default().on(backend)))
            .expect("Unable to select a Wgpu graphics API")
    }
    #[cfg(target_family = "wasm")]
    {
        backend_candidates(backend).into_iter()
            .find(|api| *api == wgpu::Backend::BrowserWebGpu
                && wgpu::Instance::enabled_backend_features().contains((*api).into()))
            .expect("The requested Wgpu graphics API is unavailable on wasm")
    }
}

pub(super) async fn request_adapter(
    device: &WgpuDevice,
    backend: wgpu::Backend,
) -> Result<(wgpu::Instance, wgpu::Adapter), WgpuInitError> {
    let instance = instance_for(backend);
    let adapter = match device.kind() {
        #[cfg(not(target_family = "wasm"))]
        WgpuDeviceKind::DiscreteGpu(num)
        | WgpuDeviceKind::IntegratedGpu(num)
        | WgpuDeviceKind::VirtualGpu(num)
        | WgpuDeviceKind::Other(num) => {
            select_from_adapter_list(num, &instance, device, backend).await
        }
        #[cfg(not(target_family = "wasm"))]
        WgpuDeviceKind::Cpu => select_from_adapter_list(0, &instance, device, backend).await,
        #[cfg(target_family = "wasm")]
        WgpuDeviceKind::IntegratedGpu(_) => {
            request_adapter_with_preference(&instance, wgpu::PowerPreference::LowPower).await
        }
        WgpuDeviceKind::Existing(_) => {
            return Err(WgpuInitError::InvalidConfiguration {
                message: "an Existing ID cannot acquire an adapter".into(),
            });
        }
        _ => request_adapter_with_preference(&instance, wgpu::PowerPreference::HighPerformance).await,
    }.map_err(|message| WgpuInitError::AdapterUnavailable { device: device.clone(), message })?;
    log::info!("Using adapter {:?}", adapter.get_info());
    Ok((instance, adapter))
}

async fn request_adapter_with_preference(
    instance: &wgpu::Instance,
    power_preference: wgpu::PowerPreference,
) -> Result<wgpu::Adapter, String> {
    instance.request_adapter(&RequestAdapterOptions {
        power_preference,
        force_fallback_adapter: false,
        compatible_surface: None,
        ..RequestAdapterOptions::default()
    }).await.map_err(|err| err.to_string())
}

#[cfg(not(target_family = "wasm"))]
async fn select_from_adapter_list(
    num: usize,
    instance: &wgpu::Instance,
    device: &WgpuDevice,
    backend: wgpu::Backend,
) -> Result<wgpu::Adapter, String> {
    let adapters = instance.enumerate_adapters(backend.into()).await;
    let found = adapters.iter().map(|adapter| adapter.get_info()).collect::<Vec<_>>();
    adapters.into_iter().filter(|adapter| {
        let kind = adapter.get_info().device_type;
        match device.kind() {
            WgpuDeviceKind::DiscreteGpu(_) => kind == wgpu::DeviceType::DiscreteGpu,
            WgpuDeviceKind::IntegratedGpu(_) => kind == wgpu::DeviceType::IntegratedGpu,
            WgpuDeviceKind::VirtualGpu(_) => kind == wgpu::DeviceType::VirtualGpu,
            WgpuDeviceKind::Other(_) => kind == wgpu::DeviceType::Other,
            WgpuDeviceKind::Cpu => kind == wgpu::DeviceType::Cpu,
            WgpuDeviceKind::DefaultDevice => true,
            WgpuDeviceKind::Existing(_) => false,
        }
    }).nth(num).ok_or_else(|| format!("No adapter matching {device:?}; available adapters: {found:?}"))
}

pub(super) fn try_device_override() -> Result<Option<WgpuDeviceKind>, WgpuInitError> {
    let value = match std::env::var("RUDA_WGPU_DEFAULT_DEVICE") {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => return Ok(None),
        Err(err) => return Err(WgpuInitError::InvalidConfiguration { message: err.to_string() }),
    };
    let parsed = if value == "Cpu" {
        Some(WgpuDeviceKind::Cpu)
    } else {
        [
            ("DiscreteGpu(", WgpuDeviceKind::DiscreteGpu as fn(usize) -> WgpuDeviceKind),
            ("IntegratedGpu(", WgpuDeviceKind::IntegratedGpu),
            ("VirtualGpu(", WgpuDeviceKind::VirtualGpu),
            ("Other(", WgpuDeviceKind::Other),
        ].into_iter().find_map(|(prefix, kind)| {
            value.strip_prefix(prefix)?.strip_suffix(')')?.parse().ok().map(kind)
        })
    };
    parsed.map(Some).ok_or_else(|| WgpuInitError::InvalidConfiguration {
        message: format!("Invalid RUDA_WGPU_DEFAULT_DEVICE: {value}"),
    })
}
