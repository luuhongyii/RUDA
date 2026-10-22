use ash::vk::{Bool32, ComponentTypeKHR, ExtendsDeviceCreateInfo,
    ExtendsPhysicalDeviceFeatures2, StructureType, TaggedStructure};
use std::{ffi::{CStr, c_void}, marker::PhantomData, ptr};

macro_rules! feature {
    ($name:ident, $tag:expr, $($field:ident),+ $(,)?) => {
        #[repr(C)]
        #[derive(Clone, Copy, Debug)]
        pub struct $name<'a> {
            pub s_type: StructureType,
            pub p_next: *mut c_void,
            $(pub $field: Bool32,)+
            _marker: PhantomData<&'a mut c_void>,
        }
        impl Default for $name<'_> {
            fn default() -> Self {
                Self { s_type: Self::STRUCTURE_TYPE, p_next: ptr::null_mut(),
                    $($field: 0,)+ _marker: PhantomData }
            }
        }
        unsafe impl Send for $name<'_> {}
        unsafe impl Sync for $name<'_> {}
        unsafe impl TaggedStructure for $name<'_> {
            const STRUCTURE_TYPE: StructureType = StructureType::from_raw($tag);
        }
        unsafe impl ExtendsPhysicalDeviceFeatures2 for $name<'_> {}
        unsafe impl ExtendsDeviceCreateInfo for $name<'_> {}
    };
}

feature!(PhysicalDeviceShaderBfloat16FeaturesKHR, 1_000_141_000,
    shader_b_float16_type, shader_b_float16_dot_product, shader_b_float16_cooperative_matrix);
feature!(PhysicalDeviceShaderFloat8FeaturesEXT, 1_000_567_000,
    shader_float8, shader_float8_cooperative_matrix);
feature!(PhysicalDeviceMaintenance8FeaturesKHR, 1_000_574_000, maintenance8);
feature!(PhysicalDeviceMaintenance9FeaturesKHR, 1_000_584_000, maintenance9);
feature!(PhysicalDeviceShader64BitIndexingFeaturesEXT, 1_000_627_000, shader64_bit_indexing);
feature!(PhysicalDeviceShaderUniformBufferUnsizedArrayFeaturesEXT, 1_000_642_000,
    shader_uniform_buffer_unsized_array);
feature!(PhysicalDeviceCooperativeMatrix2FeaturesNV, 1_000_593_000,
    cooperative_matrix_workgroup_scope, cooperative_matrix_flexible_dimensions,
    cooperative_matrix_reductions, cooperative_matrix_conversions,
    cooperative_matrix_per_element_operations, cooperative_matrix_tensor_addressing,
    cooperative_matrix_block_loads);

pub const KHR_SHADER_BFLOAT16_NAME: &CStr = c"VK_KHR_shader_bfloat16";
pub const EXT_SHADER_FLOAT8_NAME: &CStr = c"VK_EXT_shader_float8";
pub const KHR_MAINTENANCE8_NAME: &CStr = c"VK_KHR_maintenance8";
pub const KHR_MAINTENANCE9_NAME: &CStr = c"VK_KHR_maintenance9";
pub const EXT_SHADER_64BIT_INDEXING_NAME: &CStr = c"VK_EXT_shader_64bit_indexing";
pub const EXT_SHADER_UNIFORM_BUFFER_UNSIZED_ARRAY_NAME: &CStr = c"VK_EXT_shader_uniform_buffer_unsized_array";
pub const NV_COOPERATIVE_MATRIX2_NAME: &CStr = c"VK_NV_cooperative_matrix2";

pub const BFLOAT16_KHR: ComponentTypeKHR = ComponentTypeKHR::from_raw(1_000_141_000);
pub const FLOAT8_E4M3_EXT: ComponentTypeKHR = ComponentTypeKHR::from_raw(1_000_491_002);
pub const FLOAT8_E5M2_EXT: ComponentTypeKHR = ComponentTypeKHR::from_raw(1_000_491_003);

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PhysicalDeviceCooperativeMatrix2PropertiesNV<'a> {
    pub s_type: StructureType,
    pub p_next: *mut c_void,
    pub cooperative_matrix_workgroup_scope_max_workgroup_size: u32,
    pub cooperative_matrix_flexible_dimensions_max_dimension: u32,
    pub cooperative_matrix_workgroup_scope_reserved_shared_memory: u32,
    _marker: PhantomData<&'a mut c_void>,
}

impl Default for PhysicalDeviceCooperativeMatrix2PropertiesNV<'_> {
    fn default() -> Self {
        Self {
            s_type: Self::STRUCTURE_TYPE,
            p_next: ptr::null_mut(),
            cooperative_matrix_workgroup_scope_max_workgroup_size: 0,
            cooperative_matrix_flexible_dimensions_max_dimension: 0,
            cooperative_matrix_workgroup_scope_reserved_shared_memory: 0,
            _marker: PhantomData,
        }
    }
}

unsafe impl TaggedStructure for PhysicalDeviceCooperativeMatrix2PropertiesNV<'_> {
    const STRUCTURE_TYPE: StructureType = StructureType::from_raw(1_000_593_002);
}
unsafe impl ash::vk::ExtendsPhysicalDeviceProperties2 for PhysicalDeviceCooperativeMatrix2PropertiesNV<'_> {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CooperativeMatrixFlexibleDimensionsPropertiesNV<'a> {
    pub s_type: StructureType,
    pub p_next: *mut c_void,
    pub m_granularity: u32,
    pub n_granularity: u32,
    pub k_granularity: u32,
    pub a_type: ComponentTypeKHR,
    pub b_type: ComponentTypeKHR,
    pub c_type: ComponentTypeKHR,
    pub result_type: ComponentTypeKHR,
    pub saturating_accumulation: Bool32,
    pub scope: ash::vk::ScopeKHR,
    pub workgroup_invocations: u32,
    _marker: PhantomData<&'a mut c_void>,
}

impl Default for CooperativeMatrixFlexibleDimensionsPropertiesNV<'_> {
    fn default() -> Self {
        Self {
            s_type: Self::STRUCTURE_TYPE,
            p_next: ptr::null_mut(),
            m_granularity: 0,
            n_granularity: 0,
            k_granularity: 0,
            a_type: ComponentTypeKHR::default(),
            b_type: ComponentTypeKHR::default(),
            c_type: ComponentTypeKHR::default(),
            result_type: ComponentTypeKHR::default(),
            saturating_accumulation: 0,
            scope: ash::vk::ScopeKHR::default(),
            workgroup_invocations: 0,
            _marker: PhantomData,
        }
    }
}

unsafe impl TaggedStructure for CooperativeMatrixFlexibleDimensionsPropertiesNV<'_> {
    const STRUCTURE_TYPE: StructureType = StructureType::from_raw(1_000_593_001);
}

pub unsafe fn cooperative_matrix_flexible_dimensions(
    entry: &ash::Entry,
    instance: &ash::Instance,
    physical_device: ash::vk::PhysicalDevice,
) -> Result<Vec<CooperativeMatrixFlexibleDimensionsPropertiesNV<'static>>, ash::vk::Result> {
    type Query = unsafe extern "system" fn(
        ash::vk::PhysicalDevice,
        *mut u32,
        *mut CooperativeMatrixFlexibleDimensionsPropertiesNV<'_>,
    ) -> ash::vk::Result;
    let symbol = unsafe {
        entry.get_instance_proc_addr(
            instance.handle(),
            c"vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV".as_ptr(),
        )
    }.ok_or(ash::vk::Result::ERROR_EXTENSION_NOT_PRESENT)?;
    let query: Query = unsafe { std::mem::transmute(symbol) };
    loop {
        let mut count = 0;
        unsafe { query(physical_device, &mut count, ptr::null_mut()) }.result()?;
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut properties = vec![Default::default(); count as usize];
        let result = unsafe { query(physical_device, &mut count, properties.as_mut_ptr()) };
        if result == ash::vk::Result::INCOMPLETE {
            continue;
        }
        result.result()?;
        properties.truncate(count as usize);
        return Ok(properties);
    }
}
