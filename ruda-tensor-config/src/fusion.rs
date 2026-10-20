use ruda_core::config::logger::{LogLevel, LoggerConfig};

/// Configuration for operation fusion in Ruda.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FusionConfig {
    /// Logger configuration for fusion logs.
    #[serde(default)]
    pub logger: LoggerConfig<FusionLogLevel>,

    /// Beam search configuration used when exploring fusion opportunities.
    #[serde(default)]
    pub beam_search: BeamSearchConfig,

    /// Optional operation limit for a router-cached graph.
    #[serde(default)]
    pub max_graph_size: Option<usize>,

    /// Close a router graph after this many operations without a new best fusion score.
    #[serde(default = "default_growth_patience")]
    pub growth_patience: usize,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self { logger: LoggerConfig::default(), beam_search: BeamSearchConfig::default(),
            max_graph_size: None, growth_patience: default_growth_patience() }
    }
}

fn default_growth_patience() -> usize { 32 }

/// Beam search configuration controlling how the fusion optimizer explores independent blocks
/// of operations.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BeamSearchConfig {
    /// Maximum number of independent blocks explored during the fusion search.
    ///
    /// Higher values can find better fusion opportunities at the cost of more cache misses
    /// in the fusion cache.
    #[serde(default = "default_max_blocks")]
    pub max_blocks: usize,

    /// Maximum newly built optimizations per stream; cached plans remain usable.
    /// `None` keeps exploration unlimited. Further cache misses execute without caching.
    #[serde(default)]
    pub max_explorations: Option<usize>,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            max_blocks: default_max_blocks(),
            max_explorations: None,
        }
    }
}

fn default_max_blocks() -> usize {
    5
}

/// Log levels for fusion logging.
#[derive(
    Default,
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum FusionLogLevel {
    /// Fusion logging is disabled.
    #[default]
    #[serde(rename = "disabled")]
    Disabled,

    /// Log the final execution strategy selected per stream (single vs composed).
    #[serde(rename = "basic")]
    Basic,

    /// Log block merge/split decisions and cache hit/miss events.
    #[serde(rename = "medium")]
    Medium,

    /// Log every registration, rejection and scoring decision.
    #[serde(rename = "full")]
    Full,
}

impl LogLevel for FusionLogLevel {}
