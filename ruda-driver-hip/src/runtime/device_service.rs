use super::*;
use ruda_core::ir::{DeviceIdentity, PciVendor, PhysicalDevice};

impl DeviceService for HipServer {
    fn init(device_id: ruda_core::device::DeviceId) -> Self {
        let device = AmdDevice::from_id(device_id);

        #[allow(unused_assignments)]
        let mut prop_warp_size = 0;
        #[allow(unused_assignments)]
        let mut prop_arch_name = String::new();
        let prop_name;
        let mut physical = PhysicalDevice::default();
        physical.vendor = Some(PciVendor::Amd);
        #[allow(unused_assignments)]
        let mut prop_max_shared_memory_size = 0;
        #[allow(unused_assignments)]
        let mut max_ruda_count = (1, 1, 1);
        #[allow(unused_assignments)]
        let mut prop_max_threads = 0;
        let mut max_ruda_dim = (1, 1, 1);
        let mut mem_alignment = 32;
        // SAFETY: Calling HIP FFI to query device properties. The `MaybeUninit` is
        // initialized by `hipGetDevicePropertiesR0600` on success (asserted below), so
        // `assume_init()` is valid. The device index is validated by the `AmdDevice` constructor.
        unsafe {
            let mut ll_device_props = MaybeUninit::uninit();
            let status = ruda_hip_sys::hipGetDevicePropertiesR0600(
                ll_device_props.as_mut_ptr(),
                device.index as ruda_hip_sys::hipDevice_t,
            );
            assert_eq!(status, HIP_SUCCESS, "Should get device properties");
            let ll_device_props = ll_device_props.assume_init();
            prop_warp_size = ll_device_props.warpSize;
            prop_arch_name = CStr::from_ptr(ll_device_props.gcnArchName.as_ptr())
                .to_string_lossy().into_owned();
            prop_name = CStr::from_ptr(ll_device_props.name.as_ptr()).to_string_lossy().into_owned();
            #[cfg(windows)]
            {
                let luid = ll_device_props.luid.map(|byte| byte as u8);
                physical.luid = (luid != [0; 8]).then(|| ruda_core::ir::AdapterLuid::new(luid));
            }
            prop_max_shared_memory_size = ll_device_props.sharedMemPerBlock;
            max_ruda_count = (
                ll_device_props.maxGridSize[0] as u32,
                ll_device_props.maxGridSize[1] as u32,
                ll_device_props.maxGridSize[2] as u32,
            );
            prop_max_threads = ll_device_props.maxThreadsPerBlock as u32;
            max_ruda_dim.0 = ll_device_props.maxThreadsDim[0] as u32;
            max_ruda_dim.1 = ll_device_props.maxThreadsDim[1] as u32;
            max_ruda_dim.2 = ll_device_props.maxThreadsDim[2] as u32;

            // Just to be sure we check both.
            mem_alignment = usize::max(mem_alignment, ll_device_props.textureAlignment);
            mem_alignment = usize::max(mem_alignment, ll_device_props.surfaceAlignment);
        };
        let mut bus_id = [0u8; 32];
        let status = unsafe {
            ruda_hip_sys::hipDeviceGetPCIBusId(bus_id.as_mut_ptr().cast(), bus_id.len() as _, device.index as _)
        };
        if status == HIP_SUCCESS {
            physical.pci_address = CStr::from_bytes_until_nul(&bus_id).ok()
                .and_then(|id| id.to_str().ok()?.parse().ok());
        }
        let normalized_arch_name = prop_arch_name.split(':').next().unwrap_or(&prop_arch_name);
        let arch = AMDArchitecture::parse(normalized_arch_name).unwrap();
        assert_eq!(prop_warp_size as u32, arch.warp_size());

        // SAFETY: Calling HIP FFI to set the active device and configure spin-wait scheduling
        // for the current thread. The device index has been validated above by a successful
        // `hipGetDevicePropertiesR0600` call.
        unsafe {
            let status = ruda_hip_sys::hipSetDevice(device.index as ruda_hip_sys::hipDevice_t);
            hipSetDeviceFlags(hipDeviceScheduleSpin);

            assert_eq!(
                status, HIP_SUCCESS,
                "Should set the default device for the current thread"
            );
        }

        // SAFETY: Calling HIP FFI to query device memory info. The pointers to `free` and
        // `total` are valid, uniquely borrowed mutable stack variables; HIP writes the values
        // through them on success (asserted below).
        let max_memory = unsafe {
            let mut free: usize = 0;
            let mut total: usize = 0;
            let status = ruda_hip_sys::hipMemGetInfo(
                &mut free,
                &mut total,
            );
            assert_eq!(
                status, HIP_SUCCESS,
                "Should get the available memory of the device"
            );
            total
        };
        let mem_properties = MemoryDeviceProperties::new(max_memory as u64 / 4, mem_alignment as u64)
            .with_max_memory(max_memory as u64);

        let supported_wmma_combinations = HipWmmaCompiler::supported_wmma_combinations(&arch);
        let supported_mma_combinations = HipWmmaCompiler::supported_mma_combinations(&arch);
        let supported_scaled_mma_combinations =
            HipWmmaCompiler::supported_scaled_mma_combinations(&arch);

        let topology = HardwareProperties {
            load_width: 128,
            vector_register_count: None,
            plane_size_min: prop_warp_size as u32,
            plane_size_max: prop_warp_size as u32,
            max_bindings: crate::device::AMD_MAX_BINDINGS,
            max_shared_memory_size: prop_max_shared_memory_size,
            max_ruda_count,
            max_units_per_ruda: prop_max_threads,
            max_ruda_dim,
            num_streaming_multiprocessors: None,
            num_tensor_cores: None,
            min_tensor_cores_dim: if supported_wmma_combinations.is_empty() {
                None
            } else {
                Some(16)
            },
            num_cpu_cores: None,
            last_level_cache_size: None,
            max_vector_size: VectorSize::MAX,
            ruda_mma_reserved_shared_memory: 0,
        };

        let mut device_props = DeviceProperties::new(
            Default::default(),
            mem_properties.clone(),
            topology,
            TimingMethod::Device,
        ).with_identity(DeviceIdentity {
            name: prop_name,
            fingerprint: format!("hip-subgroup-v11-{prop_arch_name}"),
            physical: Some(physical),
        });
        register_supported_types(&mut device_props);

        // TODO look into unsafeAtomicAdd (https://github.com/ROCm/HIP/issues/3573120)
        // device_props.register_feature(Feature::Type(Elem::AtomicFloat(FloatKind::F16)));
        // device_props.register_feature(Feature::Type(Elem::AtomicFloat(FloatKind::BF16)));

        device_props.features.memory_reinterpret = true;
        device_props.features.alignment = true;
        device_props.features.device_memory_scope = true;
        device_props.features.plane.insert(Plane::Ops);
        device_props.features.plane.insert(Plane::Sync);
        device_props.features.plane.insert(Plane::Packed);
        device_props
            .features
            .plane
            .insert(Plane::NonUniformControlFlow);

        register_wmma_features(supported_wmma_combinations, &mut device_props);
        register_mma_features(supported_mma_combinations, &mut device_props);
        register_scaled_mma_features(supported_scaled_mma_combinations, &mut device_props);

        let comp_opts = CompilationOptions {
            warp_size: arch.warp_size(),
            supports_features: CppSupportedFeatures {
                fast_math: true,
                device_memory_scope: true,
                ..Default::default()
            },
        };
        let hip_ctx = HipContext::new(comp_opts, device_props.clone());
        let logger = Arc::new(ServerLogger::default());
        let policy = PitchedMemoryLayoutPolicy::new(device_props.memory.alignment as usize);
        let utilities = ServerUtilities::new(
            ruda_core::device::ServiceId::of::<Self>(device_id), device_props, logger, (), policy,
        );
        let options = RuntimeOptions::default();

        // SAFETY: `is_integrated_gpu` calls HIP FFI functions with a valid device index.
        let is_integrated = unsafe { is_integrated_gpu(device_id.index_id as i32) };

        HipServer::new(
            hip_ctx,
            mem_properties,
            options.memory_config,
            mem_alignment,
            is_integrated,
            utilities,
        )
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        self.utilities() as ServerUtilitiesHandle
    }
}
