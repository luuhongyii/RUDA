use ash::vk::{API_VERSION_1_1, KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME, MemoryHeapFlags};
use ruda_kernel::dsl::{
    ExecutionMode, MemoryConfiguration, WgpuCompilationOptions,
    ir::{AddressType, ElemType, FloatKind, IntKind, UIntKind},
    prelude::{CompiledKernel, Visibility},
    server::{ComputeServer, KernelArguments},
};
use ruda_core::ir::{DeviceProperties, Type, features::*};
use ruda::runtime::compiler::CompilationError;
use ruda_compiler::spirv::{GLCompute, SpirvCompiler, SpirvKernel};
use features::ExtendedFeatures;
use ash::{
    khr::cooperative_matrix,
    vk::{ComponentTypeKHR, DeviceCreateInfo, DeviceQueueCreateInfo, ScopeKHR, TRUE},
};
use wgpu::{
    DeviceDescriptor, Features, Limits,
    hal::{
        self,
        vulkan::{self, InstanceShared},
    },
};

use crate::{AutoCompiler, WgpuServer};

mod features;
mod extensions;
use extensions::{BFLOAT16_KHR, FLOAT8_E4M3_EXT, FLOAT8_E5M2_EXT};

pub type VkSpirvCompiler = SpirvCompiler<GLCompute>;

pub fn bindings(
    repr: &SpirvKernel,
    bindings: &KernelArguments,
) -> (Vec<Visibility>, Option<Visibility>, bool) {
    let buffers: Vec<_> = repr.bindings.clone();
    let meta = (!bindings.info.data.is_empty()).then_some(Visibility::Read);
    (buffers, meta, repr.uniform_info)
}

pub async fn request_vulkan_device(adapter: &wgpu::Adapter) -> Option<(wgpu::Device, wgpu::Queue)> {
    try_request_vulkan_device(adapter).await.expect("Unable to request Vulkan device")
}

pub async fn try_request_vulkan_device(adapter: &wgpu::Adapter) -> Result<Option<(wgpu::Device, wgpu::Queue)>, crate::WgpuInitError> {
    let limits = adapter.limits();
    let features = adapter
        .features()
        .difference(Features::MAPPABLE_PRIMARY_BUFFERS);
    unsafe {
        let hal_adapter = adapter.as_hal::<hal::api::Vulkan>().ok_or_else(|| crate::WgpuInitError::InvalidConfiguration {
            message: "Native Vulkan initialization requires a Vulkan adapter".into(),
        })?;
        request_device(adapter, &hal_adapter, features, limits)
    }
}

pub fn register_vulkan_features(
    adapter: &wgpu::Adapter,
    props: &mut DeviceProperties,
    comp_options: &mut WgpuCompilationOptions,
    memory_config: &MemoryConfiguration,
) -> bool {
    let features = adapter.features();
    unsafe {
        if let Some(adapter) = adapter.as_hal::<hal::api::Vulkan>() {
            register_features(&adapter, props, features, comp_options, memory_config)
        } else {
            false
        }
    }
}

/// Request device with required features, plus CMMA if available.
fn request_device(
    wgpu_adapter: &wgpu::Adapter,
    adapter: &vulkan::Adapter,
    features: Features,
    mut limits: Limits,
) -> Result<Option<(wgpu::Device, wgpu::Queue)>, crate::WgpuInitError> {
    let ash = adapter.shared_instance();

    // Can't even query for required features without `PhysicalDeviceFeatures2`
    if ash.instance_api_version() < API_VERSION_1_1
        && !ash
            .extensions()
            .contains(&KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME)
    {
        return Ok(None);
    }

    let mut extended_feat = ExtendedFeatures::from_adapter(ash.raw_instance(), adapter, features);

    if !extended_feat.has_required_features() {
        return Ok(None);
    }

    let extensions = adapter.required_device_extensions(features);
    let mut phys_features = adapter.physical_device_features(&extensions, features);

    if let Some(index_64) = &extended_feat.index_64
        && index_64.shader64_bit_indexing == TRUE
    {
        limits.max_storage_buffer_binding_size = u64::MAX;
    }

    let supported_feat = unsafe {
        ash.raw_instance()
            .get_physical_device_features(adapter.raw_physical_device())
    };

    let family_index = 0; //TODO
    let family_info = DeviceQueueCreateInfo::default()
        .queue_family_index(family_index)
        .queue_priorities(&[1.0]);
    let family_infos = [family_info];

    let str_pointers = extended_feat
        .extensions
        .iter()
        .map(|&s| {
            // Safe because `device_extensions` entries have static lifetime.
            s.as_ptr()
        })
        .collect::<Vec<_>>();

    let pre_info = DeviceCreateInfo::default()
        .queue_create_infos(&family_infos)
        .enabled_extension_names(&str_pointers);
    let mut info = phys_features.add_to_device_create(pre_info);
    info = info.enabled_features(&supported_feat);
    info = extended_feat.add_to_device_create(info);

    let vk_device = unsafe {
        ash.raw_instance()
            .create_device(adapter.raw_physical_device(), &info, None)
            .map_err(|err| crate::WgpuInitError::RequestDevice { message: format!("Vulkan device: {err}") })?
    };

    // The default is MemoryHints::Performance, which tries to do some bigger
    // block allocations. However, we already batch allocations, so we
    // can use MemoryHints::MemoryUsage to lower memory usage.
    let memory_hints = wgpu::MemoryHints::MemoryUsage;
    let owned_device = vk_device.clone();
    let drop_device: hal::DropCallback = Box::new(move || unsafe { owned_device.destroy_device(None) });
    let device = unsafe {
        adapter
            .device_from_raw(
                vk_device,
                Some(drop_device),
                &extensions,
                features,
                &limits,
                &memory_hints,
                family_info.queue_family_index,
                0,
            )
            .map_err(|err| crate::WgpuInitError::RequestDevice { message: format!("Vulkan HAL device: {err}") })?
    };

    let descriptor = DeviceDescriptor {
        label: None,
        required_features: features,
        required_limits: limits,
        memory_hints,
        trace: wgpu::Trace::Off,
        // SAFETY: Enabling experimental passthrough shaders.
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
    };

    unsafe {
        Ok(Some(
            wgpu_adapter
                .create_device_from_hal(device, &descriptor)
                .map_err(|err| crate::WgpuInitError::RequestDevice { message: format!("Vulkan Wgpu device: {err}") })?,
        ))
    }
}

/// Request device's supported features
fn register_features(
    adapter: &vulkan::Adapter,
    props: &mut DeviceProperties,
    features: Features,
    comp_options: &mut WgpuCompilationOptions,
    memory_config: &MemoryConfiguration,
) -> bool {
    let ash = adapter.shared_instance();
    super::memory::register_vulkan_capacity(adapter, &mut props.memory);

    // Can't even query for required features without `PhysicalDeviceFeatures2`
    if ash.instance_api_version() < API_VERSION_1_1
        && !ash
            .extensions()
            .contains(&KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME)
    {
        return false;
    }

    let extended_feat = ExtendedFeatures::from_adapter(ash.raw_instance(), adapter, features);

    if !extended_feat.has_required_features() {
        return false;
    }

    log::debug!("Supported Vulkan features: {extended_feat:#?}");

    register_types(props, &extended_feat);

    comp_options.supports_vulkan = true;
    props.features.device_memory_scope = true;
    comp_options.supports_u64 = extended_feat.core.shader_int64 == TRUE;
    comp_options.vulkan.max_spirv_version = extended_feat.max_spirv_version;

    props.features.plane.insert(Plane::Sync);

    if let Some(uniform_standard_layout) = extended_feat.uniform_standard_layout
        && uniform_standard_layout.uniform_buffer_standard_layout == TRUE
    {
        comp_options.vulkan.supports_uniform_standard_layout = true;
    }

    if let Some(uniform_unsized_array) = extended_feat.uniform_unsized_array
        && uniform_unsized_array.shader_uniform_buffer_unsized_array == TRUE
    {
        comp_options.vulkan.supports_uniform_unsized_array = true;
    }

    if let Some(float_controls2) = &extended_feat.float_controls2
        && float_controls2.shader_float_controls2 == TRUE
    {
        comp_options.vulkan.supports_fp_fast_math = true;
    }

    if let Some(wg_explicit_layout) = &extended_feat.wg_explicit_layout
        && wg_explicit_layout.workgroup_memory_explicit_layout == TRUE
    {
        comp_options.vulkan.supports_explicit_smem = true;
    }

    if let Some(maintenance_9) = &extended_feat.maintenance_9
        && maintenance_9.maintenance9 == TRUE
    {
        comp_options.vulkan.supports_arbitrary_bitwise = true;
    }
    comp_options.vulkan.supports_variable_pointers = extended_feat.variable_pointers
        .is_some_and(|features| features.variable_pointers == TRUE);

    if let Some(index_64) = &extended_feat.index_64
        && index_64.shader64_bit_indexing == TRUE
    {
        if let Some(heap_size) = props.memory.max_memory() {
            let max_page_size = if memory_config.uses_subslices() {
                heap_size / 4
            } else {
                heap_size
            };
            props.memory.max_page_size = max_page_size;
        }
    }

    if extended_feat.cooperative_matrix.is_some_and(|it| it.cooperative_matrix == TRUE) {
        register_cmma(ash, adapter, props);
        props.features.matmul.cmma_elementwise =
            extended_feat.max_spirv_version >= (1, 6)
                && extended_feat.nv_cooperative_matrix2.is_some_and(|it| {
                    it.cooperative_matrix_per_element_operations == TRUE
                });
        props.features.matmul.cmma_tensor_addressing =
            extended_feat.max_spirv_version >= (1, 6)
                && extended_feat.nv_cooperative_matrix2.is_some_and(|it| {
                    it.cooperative_matrix_tensor_addressing == TRUE
                });
        if extended_feat.nv_cooperative_matrix2.is_some_and(|it| {
            it.cooperative_matrix_workgroup_scope == TRUE
                && it.cooperative_matrix_flexible_dimensions == TRUE
        }) {
            register_ruda_mma(ash, adapter, props);
        }
    }

    true
}

fn register_types(props: &mut DeviceProperties, ext_feat: &ExtendedFeatures<'_>) {
    use ruda_kernel::dsl::ir::{ElemType, FloatKind, IntKind, StorageType};

    props.register_address_type(AddressType::U32);

    if let Some(index_64) = &ext_feat.index_64
        && index_64.shader64_bit_indexing == TRUE
    {
        props.register_address_type(AddressType::U64);
    }

    let default_types = [
        ElemType::UInt(UIntKind::U32),
        ElemType::Int(IntKind::I32),
        ElemType::Float(FloatKind::F32),
        ElemType::Bool,
    ];

    let default_atomic_types = [ElemType::Int(IntKind::I32), ElemType::UInt(UIntKind::U32)];

    let storage16 = ext_feat
        .buf_16
        .is_some_and(|it| it.uniform_and_storage_buffer16_bit_access == TRUE);
    let storage8 = ext_feat
        .buf_16
        .is_some_and(|it| it.uniform_and_storage_buffer16_bit_access == TRUE);

    for ty in default_types {
        props.register_type_usage(ty, TypeUsage::all());
    }

    for ty in default_atomic_types {
        props.register_atomic_type_usage(Type::new(StorageType::Atomic(ty)), AtomicUsage::all());
    }

    if ext_feat.core.shader_int64 == TRUE {
        props.register_type_usage(ElemType::UInt(UIntKind::U64), TypeUsage::all());
        props.register_type_usage(ElemType::Int(IntKind::I64), TypeUsage::all());
    }

    if ext_feat.core.shader_int16 == TRUE {
        props.register_type_usage(
            ElemType::UInt(UIntKind::U16),
            TypeUsage::maybe_store(storage16),
        );
        props.register_type_usage(
            ElemType::Int(IntKind::I16),
            TypeUsage::maybe_store(storage16),
        );
    }

    if ext_feat.core.shader_float64 == TRUE {
        props.register_type_usage(ElemType::Float(FloatKind::F64), TypeUsage::all());
    }

    if let Some(atomic_int64) = ext_feat.atomic_int64
        && atomic_int64.shader_buffer_int64_atomics == TRUE
    {
        props.register_atomic_type_usage(
            Type::new(StorageType::Atomic(ElemType::Int(IntKind::I64))),
            AtomicUsage::all(),
        );
        props.register_atomic_type_usage(
            Type::new(StorageType::Atomic(ElemType::UInt(UIntKind::U64))),
            AtomicUsage::all(),
        );
    }

    if let Some(float16_int8) = ext_feat.float16_int8 {
        if float16_int8.shader_float16 == TRUE {
            props.register_type_usage(
                ElemType::Float(FloatKind::F16),
                TypeUsage::maybe_store(storage16),
            );
        }
        if float16_int8.shader_int8 == TRUE {
            props.register_type_usage(ElemType::Int(IntKind::I8), TypeUsage::maybe_store(storage8));
            props.register_type_usage(
                ElemType::UInt(UIntKind::U8),
                TypeUsage::maybe_store(storage8),
            );
        }
    }

    if let Some(bfloat16) = ext_feat.bfloat16 {
        if bfloat16.shader_b_float16_type == TRUE {
            props.register_type_usage(ElemType::Float(FloatKind::BF16), TypeUsage::Conversion);
            if storage16 {
                props.register_type_usage(ElemType::Float(FloatKind::BF16), TypeUsage::Buffer);
            }
        }
        if bfloat16.shader_b_float16_dot_product == TRUE {
            props.register_type_usage(ElemType::Float(FloatKind::BF16), TypeUsage::DotProduct);
        }
    }

    if let Some(float8) = ext_feat.float8
        && float8.shader_float8 == TRUE
    {
        props.register_type_usage(ElemType::Float(FloatKind::E4M3), TypeUsage::Conversion);
        props.register_type_usage(ElemType::Float(FloatKind::E5M2), TypeUsage::Conversion);
        if storage8 {
            props.register_type_usage(ElemType::Float(FloatKind::E4M3), TypeUsage::Buffer);
            props.register_type_usage(ElemType::Float(FloatKind::E5M2), TypeUsage::Buffer);
        }
    }

    if let Some(atomic_float) = ext_feat.atomic_float {
        if atomic_float.shader_buffer_float32_atomics == TRUE {
            props.register_atomic_type_usage(
                Type::new(StorageType::Atomic(ElemType::Float(FloatKind::F32))),
                AtomicUsage::LoadStore,
            );
        }
        if atomic_float.shader_buffer_float32_atomic_add == TRUE {
            props.register_atomic_type_usage(
                Type::new(StorageType::Atomic(ElemType::Float(FloatKind::F32))),
                AtomicUsage::Add,
            );
        }
        if atomic_float.shader_buffer_float64_atomics == TRUE {
            props.register_atomic_type_usage(
                Type::new(StorageType::Atomic(ElemType::Float(FloatKind::F64))),
                AtomicUsage::LoadStore,
            );
        }
        if atomic_float.shader_buffer_float64_atomic_add == TRUE {
            props.register_atomic_type_usage(
                Type::new(StorageType::Atomic(ElemType::Float(FloatKind::F64))),
                AtomicUsage::Add,
            );
        }
    }

    if let Some(atomic_float) = ext_feat.atomic_float2 {
        if atomic_float.shader_buffer_float16_atomics == TRUE {
            props.register_atomic_type_usage(
                Type::new(StorageType::Atomic(ElemType::Float(FloatKind::F16))),
                AtomicUsage::LoadStore,
            );
        }
        if atomic_float.shader_buffer_float16_atomic_add == TRUE {
            props.register_atomic_type_usage(
                Type::new(StorageType::Atomic(ElemType::Float(FloatKind::F16))),
                AtomicUsage::Add,
            );
        }
        if atomic_float.shader_buffer_float16_atomic_min_max == TRUE {
            props.register_atomic_type_usage(
                Type::new(StorageType::Atomic(ElemType::Float(FloatKind::F16))),
                AtomicUsage::MinMax,
            );
        }
        if atomic_float.shader_buffer_float32_atomic_min_max == TRUE {
            props.register_atomic_type_usage(
                Type::new(StorageType::Atomic(ElemType::Float(FloatKind::F32))),
                AtomicUsage::MinMax,
            );
        }
        if atomic_float.shader_buffer_float64_atomic_min_max == TRUE {
            props.register_atomic_type_usage(
                Type::new(StorageType::Atomic(ElemType::Float(FloatKind::F64))),
                AtomicUsage::MinMax,
            );
        }
    }

    // Hardcoded support for f16 vectorized to 2 or 4 elements
    if let Some(nv_atomic_float_vector) = ext_feat.nv_atomic_float_vector
        && nv_atomic_float_vector.shader_float16_vector_atomics == TRUE
    {
        props.register_atomic_type_usage(
            Type::new(StorageType::Atomic(ElemType::Float(FloatKind::F16))).with_vector_size(2),
            AtomicUsage::all(),
        );
        props.register_atomic_type_usage(
            Type::new(StorageType::Atomic(ElemType::Float(FloatKind::F16))).with_vector_size(4),
            AtomicUsage::all(),
        );
    }
}

fn register_cmma(ash: &InstanceShared, adapter: &vulkan::Adapter, props: &mut DeviceProperties) {
    let cmma = cooperative_matrix::Instance::new(ash.entry(), ash.raw_instance());
    let properties = unsafe {
        cmma.get_physical_device_cooperative_matrix_properties(
            adapter.raw_physical_device(),
        )
        .unwrap()
    };
    let sizes = properties
        .into_iter()
        .filter(|it| {
            it.saturating_accumulation == 0
                && it.result_type == it.c_type
                && it.scope == ScopeKHR::SUBGROUP
        })
        .filter_map(|it| {
            let mut min_current = props.hardware.min_tensor_cores_dim.unwrap_or(it.m_size);
            min_current = u32::min(min_current, it.m_size);
            min_current = u32::min(min_current, it.n_size);
            min_current = u32::min(min_current, it.k_size);
            props.hardware.min_tensor_cores_dim = Some(min_current);

            Some(MmaConfig {
                a_type: convert_type(it.a_type)?.into(),
                b_type: convert_type(it.b_type)?.into(),
                cd_type: convert_type(it.c_type)?.into(),
                m: it.m_size,
                k: it.k_size,
                n: it.n_size,
            })
        })
        .collect::<Vec<_>>();
    log::debug!("Supported CMMA sizes: {sizes:#?}");

    for size in sizes {
        props.features.matmul.cmma.insert(size);
    }
}

fn register_ruda_mma(ash: &InstanceShared, adapter: &vulkan::Adapter, props: &mut DeviceProperties) {
    let configurations = unsafe {
        extensions::cooperative_matrix_flexible_dimensions(
            ash.entry(), ash.raw_instance(), adapter.raw_physical_device(),
        )
    }.expect("Unable to query workgroup cooperative matrix configurations");
    let mut properties = extensions::PhysicalDeviceCooperativeMatrix2PropertiesNV::default();
    let mut query = ash::vk::PhysicalDeviceProperties2::default().push_next(&mut properties);
    unsafe {
        ash.raw_instance().get_physical_device_properties2(adapter.raw_physical_device(), &mut query);
    }
    let max_dim = properties.cooperative_matrix_flexible_dimensions_max_dimension;
    props.features.matmul.ruda_mma = configurations.into_iter()
        .filter(|it| it.scope == ScopeKHR::WORKGROUP
            && it.c_type == it.result_type
            && it.saturating_accumulation == 0)
        .filter_map(|it| Some(RudaMmaConfig {
            a_type: convert_type(it.a_type)?.into(),
            b_type: convert_type(it.b_type)?.into(),
            cd_type: convert_type(it.c_type)?.into(),
            m_granularity: it.m_granularity,
            m_max: max_dim,
            n_granularity: it.n_granularity,
            n_max: max_dim,
            k_granularity: it.k_granularity,
            k_max: max_dim,
            units_per_block: Some(it.workgroup_invocations),
        }))
        .collect();
    props.hardware.ruda_mma_reserved_shared_memory =
        properties.cooperative_matrix_workgroup_scope_reserved_shared_memory as usize;
    log::debug!("Supported workgroup MMA configurations: {:#?}", props.features.matmul.ruda_mma);
}

fn convert_type(vk_ty: ComponentTypeKHR) -> Option<ElemType> {
    let ty = match vk_ty {
        FLOAT8_E4M3_EXT => ElemType::Float(FloatKind::E4M3),
        FLOAT8_E5M2_EXT => ElemType::Float(FloatKind::E5M2),
        ComponentTypeKHR::FLOAT16 => ElemType::Float(FloatKind::F16),
        BFLOAT16_KHR => ElemType::Float(FloatKind::BF16),
        ComponentTypeKHR::FLOAT32 => ElemType::Float(FloatKind::F32),
        ComponentTypeKHR::FLOAT64 => ElemType::Float(FloatKind::F64),
        ComponentTypeKHR::SINT8 => ElemType::Int(IntKind::I8),
        ComponentTypeKHR::SINT16 => ElemType::Int(IntKind::I16),
        ComponentTypeKHR::SINT32 => ElemType::Int(IntKind::I32),
        ComponentTypeKHR::SINT64 => ElemType::Int(IntKind::I64),
        ComponentTypeKHR::UINT8 => ElemType::UInt(UIntKind::U8),
        ComponentTypeKHR::UINT16 => ElemType::UInt(UIntKind::U16),
        ComponentTypeKHR::UINT32 => ElemType::UInt(UIntKind::U32),
        ComponentTypeKHR::UINT64 => ElemType::UInt(UIntKind::U64),
        _ => None?,
    };
    Some(ty)
}

/// Check robustness, compile, and optionally dump SPIR-V
pub(crate) fn compile(
    dyn_comp: &mut AutoCompiler,
    server: &mut WgpuServer,
    kernel: <WgpuServer as ComputeServer>::Kernel,
    mode: ExecutionMode,
) -> Result<CompiledKernel<AutoCompiler>, CompilationError> {
    log::debug!("Compiling {}", kernel.name());
    let compiled = kernel.compile(
        dyn_comp,
        &server.compilation_options,
        mode,
        kernel.address_type(),
    )?;
    #[cfg(feature = "spirv-dump")]
    dump_spirv(&compiled, kernel.name(), kernel.id());
    Ok(compiled)
}

pub(crate) fn compile_recorded(
    compiler: &mut AutoCompiler,
    server: &mut WgpuServer,
    kernel: <WgpuServer as ComputeServer>::Kernel,
    mode: ExecutionMode,
    recording: &mut ruda::runtime::compiler::CompilationRecording,
) -> Result<CompiledKernel<AutoCompiler>, CompilationError> {
    let compiled = kernel.compile_recorded(
        compiler, &server.compilation_options, mode, kernel.address_type(), recording,
    )?;
    #[cfg(feature = "spirv-dump")]
    dump_spirv(&compiled, kernel.name(), kernel.id());
    Ok(compiled)
}

#[cfg(feature = "spirv-dump")]
fn dump_spirv(
    compiled: &CompiledKernel<AutoCompiler>,
    name: &str,
    id: ruda::runtime::id::KernelId,
) {
    use std::{
        fs,
        hash::{DefaultHasher, Hash, Hasher},
    };

    if let Ok(dir) = std::env::var("RUDA_DEBUG_SPIRV")
        && let Some(repr) = compiled.repr.as_ref().and_then(|repr| repr.as_spirv())
    {
        let name = name
            .split("<")
            .take_while(|it| !it.ends_with("Runtime"))
            .map(|it| it.split(">").next().unwrap())
            .map(|it| it.split("::").last().unwrap())
            .collect::<Vec<_>>()
            .join("_");
        let mut hash = DefaultHasher::new();
        id.hash(&mut hash);
        let id = hash.finish();
        let name = sanitize_filename::sanitize_with_options(
            format!("{name}_{id:#x}"),
            sanitize_filename::Options {
                replacement: "_",
                ..Default::default()
            },
        );
        let kernel = &repr.assembled_module;
        let kernel = kernel
            .iter()
            .flat_map(|it| it.to_le_bytes())
            .collect::<Vec<_>>();
        fs::write(format!("{dir}/{name}.spv"), kernel).unwrap();
        if let Some(optimizer) = &repr.optimizer {
            fs::write(format!("{dir}/{name}.ir.txt"), format!("{}", optimizer)).unwrap();
            fs::write(
                format!("{dir}/{name}.ir.dot"),
                format!("{}", optimizer.dot_viz()),
            )
            .unwrap();
        }
    }
}
