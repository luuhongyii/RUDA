use crate::MetalCompiler;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCompileOptions, MTLComputePipelineState, MTLDevice, MTLLanguageVersion, MTLLibrary,
    MTLMathFloatingPointFunctions, MTLMathMode,
};
use ruda::runtime::{
    compiler::{CompilationArtifacts, CompilationCache, CompilationError, CompilationRecording, RudaTask},
    config::{RudaRuntimeConfig, RuntimeConfig}, id::KernelId, kernel::DebugInformation, logging::ServerLogger,
    server::{ExecutionMode, LaunchError, ResourceLimitError, RudaDim},
    validation::{validate_ruda_dim, validate_units},
};
use ruda_compiler::cpp::shared::{CompilationOptions, CppSupportedFeatures};
use ruda_core::{
    backtrace::BackTrace, cache::CacheOption,
    compilation_cache::CompilationCache as LegacyCache, hash::StableHash,
    ir::DeviceProperties, kernel::KernelIo,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub(crate) struct Pipeline {
    pub pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub ruda_dim: RudaDim,
    pub io: Option<KernelIo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MslCacheEntry {
    entrypoint: String,
    source: String,
    ruda_dim: (u32, u32, u32),
    shared_memory: usize,
    #[serde(default)]
    io: Option<KernelIo>,
}

#[derive(Debug)]
pub(crate) struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    native_options: Retained<MTLCompileOptions>,
    options: CompilationOptions,
    pipelines: CompilationCache<KernelId, Pipeline>,
    artifacts: CompilationArtifacts<MslCacheEntry>,
    legacy: Option<LegacyCache<StableHash, MslCacheEntry>>,
    properties: DeviceProperties,
}

impl MetalContext {
    pub fn new(device: Retained<ProtocolObject<dyn MTLDevice>>, properties: DeviceProperties) -> Self {
        let native_options = MTLCompileOptions::new();
        native_options.setLanguageVersion(MTLLanguageVersion::Version3_2);
        native_options.setMathMode(MTLMathMode::Safe);
        native_options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        let artifacts = CompilationArtifacts::primary_only("metal", &properties.identity.fingerprint);
        let legacy = RudaRuntimeConfig::get().compilation.cache.as_ref()
            .filter(|_| !artifacts.enabled())
            .map(|cache| LegacyCache::new("metal",
                CacheOption::default().name(properties.identity.fingerprint.clone()).root(cache.root())));
        Self {
            device, native_options, properties, legacy,
            pipelines: CompilationCache::bound(artifacts.enabled()), artifacts,
            options: CompilationOptions {
                warp_size: 32,
                supports_features: CppSupportedFeatures {
                    fast_math: true, device_memory_scope: true, ..Default::default()
                },
            },
        }
    }

    pub fn compile(
        &mut self, id: &KernelId, task: Box<dyn RudaTask<MetalCompiler>>,
        mode: ExecutionMode, logger: Arc<ServerLogger>,
    ) -> Result<Pipeline, LaunchError> {
        if let Some(pipeline) = self.pipelines.get(id) { return Ok(pipeline.clone()); }
        validate_ruda_dim(&self.properties, id)?;
        validate_units(&self.properties, id)?;
        let mut recording = CompilationRecording::new(id);
        let cached = self.artifacts.get(id)
            .or_else(|| self.legacy.as_mut().and_then(|cache| cache.get(&id.stable_hash()).cloned()));
        if let Some(entry) = cached {
            let pipeline = self.create_pipeline(&entry)?;
            self.pipelines.insert(id.clone(), pipeline.clone());
            recording.loaded();
            return Ok(pipeline);
        }
        let mut compiled = task.compile_recorded(
            &mut MetalCompiler::default(), &self.options, mode, task.address_type(), &mut recording,
        )?;
        if logger.compilation_activated() {
            compiled.debug_info = Some(DebugInformation::new("msl", id.clone()));
        }
        logger.log_compilation(&compiled);
        let entry = MslCacheEntry {
            source: compiled.source, entrypoint: compiled.entrypoint_name,
            ruda_dim: (compiled.ruda_dim.x, compiled.ruda_dim.y, compiled.ruda_dim.z),
            shared_memory: compiled.repr.as_ref().map_or(0, |repr| repr.shared_memory_size()),
            io: compiled.io,
        };
        self.validate_shared(entry.shared_memory)?;
        let pipeline = self.create_pipeline(&entry)?;
        let stored = self.artifacts.insert(id, &entry.source, entry.clone());
        if let Some(cache) = self.legacy.as_mut() {
            if let Err(error) = cache.insert(id.stable_hash(), entry) {
                log::warn!("Unable to persist Metal source: {error:?}");
            }
        }
        recording.compiled(stored);
        self.pipelines.insert(id.clone(), pipeline.clone());
        Ok(pipeline)
    }

    fn validate_shared(&self, requested: usize) -> Result<(), LaunchError> {
        let max = self.properties.hardware.max_shared_memory_size;
        if requested > max {
            return Err(ResourceLimitError::SharedMemory { requested, max, backtrace: BackTrace::capture() }.into());
        }
        Ok(())
    }

    fn create_pipeline(&self, entry: &MslCacheEntry) -> Result<Pipeline, LaunchError> {
        self.validate_shared(entry.shared_memory)?;
        let error = |reason| CompilationError::Generic { reason, backtrace: BackTrace::capture() };
        let library = self.device.newLibraryWithSource_options_error(
            &NSString::from_str(&entry.source), Some(&self.native_options),
        ).map_err(|err| error(format!("MSL compilation: {}", err.localizedDescription())))?;
        let function = library.newFunctionWithName(&NSString::from_str(&entry.entrypoint))
            .ok_or_else(|| error(format!("MSL entry point not found: {}", entry.entrypoint)))?;
        let pipeline = self.device.newComputePipelineStateWithFunction_error(&function)
            .map_err(|err| error(format!("Metal pipeline creation: {}", err.localizedDescription())))?;
        let dim: RudaDim = entry.ruda_dim.into();
        let requested = u64::from(dim.x) * u64::from(dim.y) * u64::from(dim.z);
        let max = pipeline.maxTotalThreadsPerThreadgroup();
        if requested > max as u64 {
            return Err(error(format!("Metal threadgroup requests {requested} threads; pipeline limit is {max}")).into());
        }
        Ok(Pipeline { pipeline, ruda_dim: dim, io: entry.io.clone() })
    }
}

