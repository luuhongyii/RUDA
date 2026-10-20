use super::wgsl;
use crate::AutoRepresentationRef;
use crate::WgpuServer;
use ruda_kernel::dsl::MemoryConfiguration;
use ruda_kernel::dsl::{
    ExecutionMode, WgpuCompilationOptions, hash::StableHash, server::KernelArguments,
};
use ruda_core::ir::DeviceProperties;
use ruda::runtime::{compiler::CompilationError, id::KernelId};
use crate::compilation::PreparedPipeline;
use std::{borrow::Cow, sync::Arc};
use wgpu::{
    Adapter, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType,
    Device, PipelineLayoutDescriptor, Queue, ShaderModule, ShaderModuleDescriptor,
    ShaderStages,
};

#[cfg(feature = "spirv")]
use super::vulkan;

#[cfg(all(feature = "msl", target_os = "macos"))]
use super::metal;
#[cfg(all(feature = "msl", target_os = "macos"))]
use ruda_kernel::dsl::lowering::cpp::metal as cpp_metal;

mod pipeline;

/// Shader input after reconciling a compiled representation with its language tag.
pub enum ModuleSource<'a> {
    #[cfg(feature = "spirv")]
    SpirV(&'a ruda_compiler::spirv::SpirvKernel),
    #[cfg(all(feature = "msl", target_os = "macos"))]
    Msl(&'a str, ruda_core::launch::RudaDim),
    Wgsl(&'a str),
}

impl<'a> ModuleSource<'a> {
    pub fn resolve(
        repr: Option<AutoRepresentationRef<'a>>,
        lang: &str,
        source: &'a str,
        dim: ruda_core::launch::RudaDim,
    ) -> Result<Self, CompilationError> {
        match repr {
            #[cfg(feature = "spirv")]
            Some(AutoRepresentationRef::SpirV(repr)) => Ok(Self::SpirV(repr)),
            Some(AutoRepresentationRef::Wgsl(_)) => Ok(Self::Wgsl(source)),
            #[cfg(feature = "msl")]
            Some(AutoRepresentationRef::Msl(_)) => Self::msl(source, dim),
            None => match lang {
                "wgsl" => Ok(Self::Wgsl(source)),
                "msl" => Self::msl(source, dim),
                other => Err(CompilationError::Generic {
                    reason: format!("wgpu has no text passthrough for a precompiled `{other}` kernel"),
                    backtrace: ruda_core::backtrace::BackTrace::capture(),
                }),
            },
        }
    }

    #[cfg(all(feature = "msl", target_os = "macos"))]
    fn msl(source: &'a str, dim: ruda_core::launch::RudaDim) -> Result<Self, CompilationError> {
        Ok(Self::Msl(source, dim))
    }

    #[cfg(not(all(feature = "msl", target_os = "macos")))]
    fn msl(_source: &'a str, _dim: ruda_core::launch::RudaDim) -> Result<Self, CompilationError> {
        Err(CompilationError::Generic {
            reason: "MSL passthrough is only available on macOS".into(),
            backtrace: ruda_core::backtrace::BackTrace::capture(),
        })
    }
}

pub async fn request_device(adapter: &Adapter) -> (Device, Queue) {
    try_request_device(adapter).await.expect("Unable to request Wgpu device")
}

pub async fn try_request_device(adapter: &Adapter) -> Result<(Device, Queue), crate::WgpuInitError> {
    if let Some(result) = request_vulkan_device(adapter).await? {
        return Ok(result);
    }
    if let Some(result) = request_metal_device(adapter).await? {
        return Ok(result);
    }
    wgsl::try_request_device(adapter).await
}

#[cfg(feature = "spirv")]
async fn request_vulkan_device(adapter: &Adapter) -> Result<Option<(Device, Queue)>, crate::WgpuInitError> {
    if is_vulkan(adapter) {
        vulkan::try_request_vulkan_device(adapter).await
    } else {
        Ok(None)
    }
}

#[cfg(not(feature = "spirv"))]
async fn request_vulkan_device(_adapter: &Adapter) -> Result<Option<(Device, Queue)>, crate::WgpuInitError> {
    Ok(None)
}

#[cfg(all(feature = "msl", target_os = "macos"))]
async fn request_metal_device(adapter: &Adapter) -> Result<Option<(Device, Queue)>, crate::WgpuInitError> {
    if is_metal(adapter) {
        metal::try_request_metal_device(adapter).await.map(Some)
    } else {
        Ok(None)
    }
}

#[cfg(not(all(feature = "msl", target_os = "macos")))]
async fn request_metal_device(_adapter: &Adapter) -> Result<Option<(Device, Queue)>, crate::WgpuInitError> {
    Ok(None)
}

pub fn register_features(
    adapter: &Adapter,
    props: &mut DeviceProperties,
    comp_options: &mut WgpuCompilationOptions,
    memory_config: &MemoryConfiguration,
) {
    if register_vulkan_features(adapter, props, comp_options, memory_config) {
        return;
    }
    if register_metal_features(adapter, props, comp_options, memory_config) {
        return;
    }
    wgsl::register_wgsl_features(adapter, props, comp_options);
}

#[cfg(feature = "spirv")]
pub fn register_vulkan_features(
    adapter: &Adapter,
    props: &mut DeviceProperties,
    comp_options: &mut WgpuCompilationOptions,
    memory_config: &MemoryConfiguration,
) -> bool {
    if is_vulkan(adapter) {
        vulkan::register_vulkan_features(adapter, props, comp_options, memory_config)
    } else {
        false
    }
}

#[cfg(not(feature = "spirv"))]
pub fn register_vulkan_features(
    _adapter: &Adapter,
    _props: &mut DeviceProperties,
    _comp_options: &mut WgpuCompilationOptions,
    _memory_config: &MemoryConfiguration,
) -> bool {
    false
}

#[cfg(all(feature = "msl", target_os = "macos"))]
pub fn register_metal_features(
    adapter: &Adapter,
    props: &mut DeviceProperties,
    comp_options: &mut WgpuCompilationOptions,
    _memory_config: &MemoryConfiguration,
) -> bool {
    if is_metal(adapter) {
        metal::register_metal_features(adapter, props, comp_options);
        comp_options.supports_msl
    } else {
        false
    }
}

#[cfg(not(all(feature = "msl", target_os = "macos")))]
pub fn register_metal_features(
    _adapter: &Adapter,
    _props: &mut DeviceProperties,
    _comp_options: &mut WgpuCompilationOptions,
    _memory_config: &MemoryConfiguration,
) -> bool {
    false
}

#[cfg(feature = "spirv")]
fn is_vulkan(adapter: &Adapter) -> bool {
    unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>().is_some() }
}

#[cfg(all(feature = "msl", target_os = "macos"))]
fn is_metal(adapter: &Adapter) -> bool {
    unsafe { adapter.as_hal::<wgpu::hal::api::Metal>().is_some() }
}
