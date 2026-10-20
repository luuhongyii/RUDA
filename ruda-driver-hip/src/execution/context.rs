use super::storage::gpu::GpuResource;
use crate::runtime::HipCompiler;
use crate::{execution::stream::Stream, runtime::HipComputeKernel};
use ruda_core::backtrace::BackTrace;
use ruda_core::cache::CacheOption;
use ruda_core::hash::StableHash;
use ruda_kernel::dsl::{
    compilation_cache::CompilationCache,
    server::ResourceLimitError,
    {ir::DeviceProperties, prelude::*},
};
use ruda_compiler::cpp::formatter::format_cpp;
use ruda_compiler::cpp::shared::CompilationOptions;
use ruda_hip_sys::{HIP_SUCCESS, get_hip_include_path, hiprtcResult_HIPRTC_SUCCESS};
use super::profile::EventProfiler;
use ruda::runtime::{
    compiler::CompilationError,
    validation::{validate_ruda_dim, validate_units},
};
use ruda::runtime::{compiler::RudaTask, logging::ServerLogger};
use ruda::runtime::compiler::{CompilationArtifacts, CompilationRecording};
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::ffi::CStr;
use std::ffi::CString;
use std::sync::Arc;

#[derive(Debug)]
pub(crate) struct HipContext {
    pub module_names: HashMap<KernelId, HipCompiledKernel>,
    pub timestamps: EventProfiler,
    pub compilation_options: CompilationOptions,
    pub properties: DeviceProperties,
    pub compilation_cache: Option<CompilationCache<StableHash, CompilationCacheEntry>>,
    artifacts: CompilationArtifacts<CompilationCacheEntry>,
    cache_generation: Option<u32>,
    retired_modules: Vec<HipCompiledKernel>,
}

#[derive(Debug)]
pub struct HipCompiledKernel {
    pub(crate) io: Option<ruda_core::kernel::KernelIo>,
    _module: ruda_hip_sys::hipModule_t,
    func: ruda_hip_sys::hipFunction_t,
    ruda_dim: RudaDim,
    shared_mem_bytes: usize,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
pub struct CompilationCacheEntry {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    io: Option<ruda_core::kernel::KernelIo>,
    entrypoint_name: String,
    shared_mem_bytes: usize,
    binary: Vec<i8>,
}

impl HipContext {
    pub fn new(compilation_options: CompilationOptions, properties: DeviceProperties) -> Self {
        let artifacts = CompilationArtifacts::new("hip-kernel", &properties.identity.fingerprint);
        let cache_generation = artifacts.enabled().then(ruda_core::environment::generation);
        Self {
            module_names: HashMap::new(),
            timestamps: EventProfiler::default(),
            compilation_options,
            compilation_cache: {
                use ruda::runtime::config::RuntimeConfig;
                let config = ruda::runtime::config::RudaRuntimeConfig::get();
                if let Some(cache) = &config.compilation.cache
                    && !artifacts.enabled()
                {
                    let root = cache.root();
                    Some(CompilationCache::new(
                        "hip-kernel",
                        CacheOption::default().name(properties.identity.fingerprint.clone()).root(root),
                    ))
                } else {
                    None
                }
            },
            properties,
            artifacts,
            cache_generation,
            retired_modules: Vec::new(),
        }
    }

    pub(super) fn refresh_compilation_environment(&mut self) {
        let Some(generation) = self.cache_generation else { return; };
        let current = ruda_core::environment::generation();
        if generation != current {
            self.retired_modules.extend(self.module_names.drain().map(|(_, kernel)| kernel));
            self.cache_generation = Some(current);
        }
    }
}

mod compilation;
mod launch;
