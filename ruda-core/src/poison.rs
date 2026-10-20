//! Shared payload for a device the driver can no longer execute work on.

use alloc::string::String;
use crate::backtrace::BackTrace;
use thiserror::Error;

/// An unrecoverable device/context fault, rather than a refused kernel configuration.
#[derive(Error, Clone)]
#[cfg_attr(any(compiler_std_io, feature = "environment-persistence"), derive(serde::Serialize, serde::Deserialize))]
#[error("{reason}\nBacktrace:\n{backtrace}")]
pub struct DevicePoison {
    /// The driver operation and status, or device-lost notification.
    pub reason: String,
    /// Where the failure was observed.
    #[cfg_attr(any(compiler_std_io, feature = "environment-persistence"), serde(skip))]
    pub backtrace: BackTrace,
}

impl DevicePoison {
    /// Capture a device fault at its observation point.
    pub fn new(reason: impl Into<String>) -> Self {
        Self { reason: reason.into(), backtrace: BackTrace::capture() }
    }
}

impl core::fmt::Debug for DevicePoison {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self}")
    }
}
