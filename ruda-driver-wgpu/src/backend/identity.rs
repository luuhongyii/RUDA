use ruda_core::ir::{DeviceIdentity, PhysicalDevice};

pub(crate) fn identity(adapter: &wgpu::Adapter, device: &wgpu::Device) -> DeviceIdentity {
    let info = adapter.get_info();
    let physical = if info.device_type == wgpu::DeviceType::Cpu {
        None
    } else {
        let mut physical = PhysicalDevice::default();
        physical.vendor = (info.vendor != 0).then(|| info.vendor.into());
        if info.backend == wgpu::Backend::Vulkan {
            physical.pci_address = info.device_pci_bus_id.parse().ok();
        }
        #[cfg(feature = "spirv")]
        describe_vulkan(adapter, &mut physical);
        #[cfg(windows)]
        describe_dx12(adapter, &mut physical);
        #[cfg(target_vendor = "apple")]
        describe_metal(device, &mut physical);
        Some(physical)
    };
    let _ = device;
    DeviceIdentity {
        name: info.name,
        fingerprint: format!("spirv_{}_{}", info.vendor, info.device),
        physical,
    }
}

#[cfg(feature = "spirv")]
fn describe_vulkan(adapter: &wgpu::Adapter, physical: &mut PhysicalDevice) {
    use ash::vk;
    let Some(adapter) = (unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }) else { return; };
    let instance = adapter.shared_instance();
    if instance.instance_api_version() < vk::API_VERSION_1_1
        || adapter.physical_device_capabilities().properties().api_version < vk::API_VERSION_1_1
    { return; }
    let mut ids = vk::PhysicalDeviceIDProperties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut ids);
    unsafe {
        instance.raw_instance().get_physical_device_properties2(adapter.raw_physical_device(), &mut properties);
    }
    if ids.device_luid_valid == vk::TRUE {
        physical.luid = Some(ruda_core::ir::AdapterLuid::new(ids.device_luid));
    }
}

#[cfg(windows)]
fn describe_dx12(adapter: &wgpu::Adapter, physical: &mut PhysicalDevice) {
    let Some(adapter) = (unsafe { adapter.as_hal::<wgpu::hal::api::Dx12>() }) else { return; };
    let Ok(description) = (unsafe { adapter.raw_adapter().GetDesc2() }) else { return; };
    let luid = description.AdapterLuid;
    physical.luid = Some(ruda_core::ir::AdapterLuid::from_parts(luid.LowPart, luid.HighPart));
}

#[cfg(target_vendor = "apple")]
fn describe_metal(device: &wgpu::Device, physical: &mut PhysicalDevice) {
    use objc2_metal::MTLDevice;
    let Some(device) = (unsafe { device.as_hal::<wgpu::hal::api::Metal>() }) else { return; };
    physical.registry_entry_id = Some(ruda_core::ir::RegistryEntryId::new(device.raw_device().registryID()));
}
