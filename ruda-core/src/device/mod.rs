mod base;

pub use base::*;
pub use handle::register_device_service_shutdown;

pub(crate) mod handle;

#[cfg(feature = "communication")]
mod communication;
#[cfg(feature = "communication")]
pub use communication::CommunicationId;
