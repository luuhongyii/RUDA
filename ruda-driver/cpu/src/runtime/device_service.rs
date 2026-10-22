use super::*;
use ruda_core::ir::DeviceIdentity;

impl DeviceService for CpuServer {
    fn init(device_id: ruda_core::device::DeviceId) -> Self {
        let options = RuntimeOptions::default();
        let max_ruda_dim = (u32::MAX, u32::MAX, u32::MAX);
        let max_ruda_count = (u32::MAX, u32::MAX, u32::MAX);
        let system = System::new_all();
        let total_memory = system
            .cgroup_limits()
            .map(|g| g.total_memory)
            .unwrap_or(system.total_memory());
        let max_shared_memory_size = total_memory as usize;
        let logger = ruda_core::stub::Arc::new(ServerLogger::default());

        let available_parallelism = crate::execution::affinity::worker_count(
            &crate::execution::affinity::active_cores(),
        );

        let (load_width, vector_register_count) = host_vector_registers();
        let f16_evaluation = crate::compilation::MlirCompilerOptions::for_host().f16_evaluation;

        let topology = HardwareProperties {
            load_width,
            vector_register_count,
            plane_size_min: 1,
            plane_size_max: 1,
            max_bindings: u32::MAX,
            max_shared_memory_size,
            max_ruda_count,
            num_cpu_cores: Some(available_parallelism as u32),
            last_level_cache_size: super::cache::last_level_cache_size(),
            max_units_per_ruda: u32::MAX,
            max_ruda_dim,
            num_streaming_multiprocessors: None,
            num_tensor_cores: None,
            min_tensor_cores_dim: None,
            max_vector_size: VectorSize::MAX,
            ruda_mma_reserved_shared_memory: 0,
        };

        const ALIGNMENT: u64 = 8;

        let mem_properties = MemoryDeviceProperties::new(max_shared_memory_size as u64, ALIGNMENT)
            .with_max_memory(total_memory);

        let mut device_props = DeviceProperties::new(
            Features {
                unaligned_io: true,
                ..Default::default()
            },
            mem_properties.clone(),
            topology,
            TimingMethod::Device,
        ).with_identity(DeviceIdentity {
            name: system.cpus().first().map(|cpu| cpu.brand().trim())
                .filter(|brand| !brand.is_empty()).map(String::from)
                .unwrap_or_else(|| format!("CPU {}", std::env::consts::ARCH)),
            fingerprint: format!("cpu_mlir_{}_load-{load_width}_f16-{f16_evaluation}", std::env::consts::ARCH),
            physical: None,
        });
        register_supported_types(&mut device_props);

        let utilities = ServerUtilities::new(
            ruda_core::device::ServiceId::of::<Self>(device_id),
            device_props,
            logger,
            (),
            ContiguousMemoryLayoutPolicy::new(ALIGNMENT as usize),
        );
        CpuServer::new(mem_properties, options.memory_config, Arc::new(utilities))
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        self.utilities() as ServerUtilitiesHandle
    }
}

fn host_vector_registers() -> (u32, Option<u32>) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") { return (512, Some(32)); }
        if std::arch::is_x86_feature_detected!("avx") { return (256, Some(16)); }
        return (128, Some(16));
    }
    #[cfg(target_arch = "aarch64")]
    { (128, Some(32)) }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    { (128, None) }
}
