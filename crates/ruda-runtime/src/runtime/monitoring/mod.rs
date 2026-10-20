//! Device utilization reported by the card driver or host platform.

/// Physical-card counters shared across runtime backends.
pub mod gpu_utilization;
mod utilization;

pub use utilization::*;
