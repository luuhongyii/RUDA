use ruda_core::ir::MemoryDeviceProperties;

pub(crate) fn register_memory_capacity(
    adapter: &wgpu::Adapter,
    device: &wgpu::Device,
    properties: &mut MemoryDeviceProperties,
) {
    #[cfg(feature = "spirv")]
    if let Some(adapter) = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() } {
        register_vulkan_capacity(&adapter, properties);
    }

    #[cfg(target_vendor = "apple")]
    {
        use objc2_metal::MTLDevice;
        // SAFETY: The guarded HAL device is borrowed only for this property query.
        if let Some(device) = unsafe { device.as_hal::<wgpu::hal::api::Metal>() } {
            properties.set_max_memory(device.raw_device().recommendedMaxWorkingSetSize());
        }
    }
    let _ = (adapter, device, properties);
}

#[cfg(feature = "spirv")]
pub(super) fn register_vulkan_capacity(
    adapter: &wgpu::hal::vulkan::Adapter,
    properties: &mut MemoryDeviceProperties,
) {
    let instance = adapter.shared_instance().raw_instance();
    // SAFETY: The physical device belongs to this live HAL adapter and instance.
    let memory = unsafe {
        instance.get_physical_device_memory_properties(adapter.raw_physical_device())
    };
    if let Some(capacity) = memory.memory_heaps
        .iter()
        .take(memory.memory_heap_count as usize)
        .filter(|heap| heap.flags.contains(ash::vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|heap| heap.size)
        .max()
    {
        properties.set_max_memory(capacity);
    }
}
