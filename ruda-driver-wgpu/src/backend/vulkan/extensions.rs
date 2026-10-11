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

pub const KHR_SHADER_BFLOAT16_NAME: &CStr = c"VK_KHR_shader_bfloat16";
pub const EXT_SHADER_FLOAT8_NAME: &CStr = c"VK_EXT_shader_float8";
pub const KHR_MAINTENANCE8_NAME: &CStr = c"VK_KHR_maintenance8";
pub const KHR_MAINTENANCE9_NAME: &CStr = c"VK_KHR_maintenance9";
pub const EXT_SHADER_64BIT_INDEXING_NAME: &CStr = c"VK_EXT_shader_64bit_indexing";
pub const EXT_SHADER_UNIFORM_BUFFER_UNSIZED_ARRAY_NAME: &CStr = c"VK_EXT_shader_uniform_buffer_unsized_array";

pub const BFLOAT16_KHR: ComponentTypeKHR = ComponentTypeKHR::from_raw(1_000_141_000);
pub const FLOAT8_E4M3_EXT: ComponentTypeKHR = ComponentTypeKHR::from_raw(1_000_491_002);
pub const FLOAT8_E5M2_EXT: ComponentTypeKHR = ComponentTypeKHR::from_raw(1_000_491_003);
