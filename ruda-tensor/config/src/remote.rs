use ruda_core::config::logger::{LogLevel, LoggerConfig};

/// Remote wire batching and logging configuration.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RemoteConfig {
    /// Sinks and verbosity for remote graph traffic observations.
    #[serde(default)]
    pub logger: LoggerConfig<RemoteLogLevel>,
    /// Flush after this many buffered tasks.
    #[serde(default = "default_flush_threshold")]
    pub flush_threshold: usize,
    /// Flush after this many bytes of buffered tensor payloads.
    #[serde(default = "default_flush_bytes_threshold")]
    pub flush_bytes_threshold: usize,
    /// Segmentation offload for Iroh endpoints created by the library.
    #[serde(default)]
    pub iroh_segmentation_offload: bool,
}

impl Default for RemoteConfig {
    fn default() -> Self {
        Self { logger: LoggerConfig::default(), flush_threshold: default_flush_threshold(),
            flush_bytes_threshold: default_flush_bytes_threshold(), iroh_segmentation_offload: false }
    }
}
fn default_flush_threshold() -> usize { 4 }
fn default_flush_bytes_threshold() -> usize { 1024 * 1024 }

/// Verbosity of remote graph traffic logs.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub enum RemoteLogLevel {
    /// No remote traffic logs.
    #[default]
    #[serde(rename = "disabled")]
    Disabled,
    /// A summary for each additional MiB of graph payload saved.
    #[serde(rename = "basic")]
    Basic,
    /// Every registration and replay, including operation categories.
    #[serde(rename = "full")]
    Full,
}
impl LogLevel for RemoteLogLevel {}
