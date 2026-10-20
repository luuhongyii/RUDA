// We cannot put this struct in ruda-driver-wgpu crate due to circular dependencies.
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuCompilationOptions {
    pub supports_u64: bool,
    /// Whether the Vulkan compiler is supported or we need to fall back to WGSL
    pub supports_vulkan: bool,
    pub supports_msl: bool,
    /// Whether native MSL provides device-scope availability and visibility fences.
    pub supports_msl_device_memory_scope: bool,

    pub vulkan: VulkanCompilationOptions,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct VulkanCompilationOptions {
    pub supports_fp_fast_math: bool,
    pub supports_explicit_smem: bool,
    pub supports_arbitrary_bitwise: bool,
    pub supports_uniform_standard_layout: bool,
    pub supports_uniform_unsized_array: bool,
    pub supports_variable_pointers: bool,

    pub max_spirv_version: (u8, u8),
}
