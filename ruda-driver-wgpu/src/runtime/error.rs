use crate::{WgpuBackend, WgpuDevice};

#[derive(Debug)]
#[non_exhaustive]
pub enum WgpuInitError {
    InvalidConfiguration { message: String },
    UnsupportedGraphicsApi { api: WgpuBackend },
    AdapterUnavailable { device: WgpuDevice, message: String },
    RequestDevice { message: String },
    Registration { message: String },
}

impl core::fmt::Display for WgpuInitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidConfiguration { message } => write!(f, "Invalid Wgpu configuration: {message}"),
            Self::UnsupportedGraphicsApi { api } => write!(f, "Unsupported graphics API: {api:?}"),
            Self::AdapterUnavailable { device, message } => write!(f, "No adapter for {device:?}: {message}"),
            Self::RequestDevice { message } => write!(f, "Unable to create Wgpu device: {message}"),
            Self::Registration { message } => write!(f, "Unable to register Wgpu device: {message}"),
        }
    }
}

impl std::error::Error for WgpuInitError {}
