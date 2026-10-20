use super::*;

impl DeviceService for WgpuServer {
    fn init(device_id: ruda_core::device::DeviceId) -> Self {
        let device = WgpuDevice::from_id(device_id);
        let options = RuntimeOptions::try_default().expect("Invalid Wgpu runtime options");
        let setup = future::block_on(try_create_setup_for_device(&device))
            .expect("Unable to acquire Wgpu device");
        try_create_server(device_id, setup, options, device.backend()).expect("Unable to initialize Wgpu server")
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        self.utilities.clone() as ServerUtilitiesHandle
    }
}

pub(crate) fn try_create_server(
    device_id: ruda_core::device::DeviceId,
    setup: WgpuSetup,
    options: RuntimeOptions,
    requested_backend: WgpuBackend,
) -> Result<WgpuServer, WgpuInitError> {
    options.validate()?;
    setup.validate(requested_backend)?;
    let limits = setup.device.limits();
    let adapter_limits = setup.adapter.limits();
    let mut adapter_info = setup.adapter.get_info();

    // Workaround: WebGPU reports some "fake" subgroup info atm, as it's not really supported yet.
    // However, some algorithms do rely on having this information eg. ruPRIM uses max subgroup size _even_ when
    // subgroups aren't used. For now, just override with the maximum range of subgroups possible.
    if adapter_info.subgroup_min_size == 0 && adapter_info.subgroup_max_size == 0 {
        // There is in theory nothing limiting the size to go below 8 but in practice 8 is the minimum found anywhere.
        adapter_info.subgroup_min_size = 8;
        // This is a hard limit of GPU APIs (subgroup ballot returns 4 * 32 bits).
        adapter_info.subgroup_max_size = 128;
    }

    let mem_props = MemoryDeviceProperties::new(limits.max_storage_buffer_binding_size.min(limits.max_buffer_size), limits.min_uniform_buffer_offset_alignment.max(limits.min_storage_buffer_offset_alignment) as u64);
    let max_count = adapter_limits.max_compute_workgroups_per_dimension;
    let hardware_props = HardwareProperties {
        load_width: 128,
        vector_register_count: None,
        // On Apple Silicon, the plane size is 32,
        // though the minimum and maximum differ.
        // https://github.com/gpuweb/gpuweb/issues/3950
        #[cfg(apple_silicon)]
        plane_size_min: 32,
        #[cfg(not(apple_silicon))]
        plane_size_min: adapter_info.subgroup_min_size,
        #[cfg(apple_silicon)]
        plane_size_max: 32,
        #[cfg(not(apple_silicon))]
        plane_size_max: adapter_info.subgroup_max_size,
        // wgpu uses an additional buffer for variable-length buffers,
        // so we have to use one buffer less on our side to make room for that wgpu internal buffer.
        // See: https://github.com/gfx-rs/wgpu/blob/a9638c8e3ac09ce4f27ac171f8175671e30365fd/wgpu-hal/src/metal/device.rs#L799
        max_bindings: limits
            .max_storage_buffers_per_shader_stage
            .saturating_sub(1),
        max_shared_memory_size: limits.max_compute_workgroup_storage_size as usize,
        max_ruda_count: (max_count, max_count, max_count),
        max_units_per_ruda: adapter_limits.max_compute_invocations_per_workgroup,
        max_ruda_dim: (
            adapter_limits.max_compute_workgroup_size_x,
            adapter_limits.max_compute_workgroup_size_y,
            adapter_limits.max_compute_workgroup_size_z,
        ),
        num_streaming_multiprocessors: None,
        num_tensor_cores: None,
        min_tensor_cores_dim: None,
        num_cpu_cores: None, // TODO: Check if device is CPU.
        last_level_cache_size: None,
        max_vector_size: 4,
        ruda_mma_reserved_shared_memory: 0,
    };

    let mut compilation_options = Default::default();

    let features = setup.device.features();

    let time_measurement = if features.contains(wgpu::Features::TIMESTAMP_QUERY) {
        TimingMethod::Device
    } else {
        TimingMethod::System
    };

    let mut device_props = DeviceProperties::new(
        Default::default(),
        mem_props,
        hardware_props,
        time_measurement,
    ).with_identity(backend::identity(&setup.adapter, &setup.device));
    backend::register_memory_capacity(&setup.adapter, &setup.device, &mut device_props.memory);

    #[cfg(not(all(target_os = "macos", feature = "msl")))]
    {
        if features.contains(wgpu::Features::SUBGROUP)
            && setup.adapter.get_info().device_type != wgpu::DeviceType::Cpu
        {
            use ruda_core::ir::features::Plane;

            device_props.features.plane.insert(Plane::Ops);
        }
    }

    backend::register_features(
        &setup.adapter,
        &mut device_props,
        &mut compilation_options,
        &options.memory_config,
    );

    if compilation_options.supports_vulkan || compilation_options.supports_msl {
        device_props.features.plane.insert(ruda_core::ir::features::Plane::NonUniformControlFlow);
    }
    #[cfg(feature = "msl")]
    if requested_backend == WgpuBackend::Metal && !compilation_options.supports_msl {
        return Err(WgpuInitError::InvalidConfiguration {
            message: "Native MSL compilation is unavailable for the requested Metal device".into(),
        });
    }

    let logger = alloc::sync::Arc::new(ServerLogger::default());

    let allocator = ContiguousMemoryLayoutPolicy::new(device_props.memory.alignment as usize);
    Ok(WgpuServer::new(
        device_props.memory.clone(),
        options.memory_config,
        compilation_options,
        setup.device.clone(),
        setup.queue,
        options.tasks_max,
        setup.backend,
        time_measurement,
        ServerUtilities::new(
            ruda_core::device::ServiceId::of::<WgpuServer>(device_id),
            device_props, logger, setup.backend, allocator,
        ),
    ))
}
