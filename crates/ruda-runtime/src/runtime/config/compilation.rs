#[cfg(std_io)]
use super::cache::CacheConfig;
use super::logger::{LogLevel, LoggerConfig};

/// Configuration for compilation settings in `Ruda`.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompilationConfig {
    /// Logger configuration for compilation logs, using binary log levels.
    #[serde(default)]
    pub logger: LoggerConfig<CompilationLogLevel>,
    /// Cache location for storing compiled kernels.
    #[serde(default)]
    #[cfg(std_io)]
    pub cache: Option<CacheConfig>,
    /// Cache compiled artifacts in the active named environment. The legacy
    /// `cache` path remains unchanged when this option is false.
    #[serde(default)]
    pub environment_cache: bool,
    /// Controls whether kernel launches enforce bounds checks.
    #[serde(default)]
    pub check_mode: BoundsCheckMode,
    /// CPU half-precision evaluation policy. `None` selects by native f16 support.
    #[serde(default)]
    pub f16_evaluation: Option<F16Evaluation>,
}

/// Rounding boundaries for CPU f16 arithmetic; external storage stays f16.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum F16Evaluation {
    /// Round each arithmetic result to f16.
    #[serde(rename = "per-operation")]
    PerOperation,
    /// Keep arithmetic intermediates in f32 until a storage or non-arithmetic use.
    #[default]
    #[serde(rename = "chain")]
    Chain,
    /// Also widen private accumulators when doing so removes more conversions.
    #[serde(rename = "accumulators")]
    Accumulators,
}

impl F16Evaluation {
    pub fn for_native_f16(native: bool) -> Self {
        if native { Self::PerOperation } else { Self::Chain }
    }
}

impl core::fmt::Display for F16Evaluation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::PerOperation => "per-operation",
            Self::Chain => "chain",
            Self::Accumulators => "accumulators",
        })
    }
}

/// Bounds checks options.
#[derive(Default, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum BoundsCheckMode {
    #[serde(rename = "enforce")]
    /// Always enforce bounds checks on every kernel launch.
    Enforce,
    #[serde(rename = "validate")]
    /// Always enforce bounds checks on every kernel launch, and validate unchecked kernels for OOB.
    Validate,
    /// Enforce bounds checking on standard launches, but skip checks on
    /// explicitly unchecked launches for better performance.
    #[default]
    #[serde(rename = "auto")]
    Auto,
}

/// Log levels for compilation in `Ruda`.
#[derive(Default, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum CompilationLogLevel {
    /// Compilation logging is disabled.
    #[default]
    #[serde(rename = "disabled")]
    Disabled,

    /// Basic compilation information is logged such as when kernels are compiled.
    #[serde(rename = "basic")]
    Basic,

    /// Full compilation details are logged including source code.
    #[serde(rename = "full")]
    Full,
}

impl LogLevel for CompilationLogLevel {}
