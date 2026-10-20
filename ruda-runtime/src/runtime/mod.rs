#![warn(missing_docs)]

//! `Ruda` runtime crate that helps creating high performance async runtimes.

/// Various identifier types used in `Ruda`.
pub mod id;

/// Kernel related traits.
pub mod kernel;

/// Stream related utilities.
pub mod stream;

/// Compute client module.
pub mod client;

/// Runtime-independent device selectors and their stable IDs.
pub mod device;

/// Compile workloads without dispatching them, while retaining real tuning measurements.
pub mod dry_run;

/// Stream-captured executable graphs and device-service lifetime management.
pub mod capture;

/// Autotune module
pub mod tune;

/// Device throughput measurements and resource ceilings.
pub mod throughput;

/// Device utilization counters from the driver or host platform.
pub mod monitoring;

/// Memory management module.
pub mod memory_management;
/// Compute server module.
pub mod server;
/// Compute Storage module.
pub mod storage;

/// `Ruda` config module.
pub mod config;

pub use ruda_core::benchmark;
pub use ruda_core::environment;
pub use ruda_core::poison;

/// Logging utilities to be used by a compute server.
pub mod logging;

/// TMA-related runtime types
pub mod tma;

/// Compiler trait and related types
pub mod compiler;
/// Runtime trait and related types
pub mod backend;
/// Device buffer contracts for native normalization dispatch.
pub mod normalization;
/// Simple system profiling using timestamps.
pub mod timestamp_profiler;

/// Validation utils for shared properties
pub mod validation;

/// Allocators moddule.
pub mod allocator;

pub use crate::{local_tuner, storage_id_type};
