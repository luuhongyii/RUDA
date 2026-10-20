mod task;
pub use task::{ConnectionId, RemoteDeviceInfo, SessionId};

#[allow(unused_imports)]
pub(crate) use task::*;

/// We define the communication protocol here
#[cfg(all(not(feature = "iroh"), not(target_family = "wasm")))]
pub(crate) type RemoteProtocol = ruda_communication::websocket::WebSocket;
#[cfg(all(feature = "iroh", not(target_family = "wasm")))]
pub(crate) type RemoteProtocol = ruda_communication::transport::TransportProtocol;
#[cfg(all(feature = "iroh", target_family = "wasm"))]
pub(crate) type RemoteProtocol = ruda_communication::iroh::Iroh;
#[cfg(all(feature = "client", not(feature = "iroh"), target_family = "wasm"))]
compile_error!("The browser Remote client requires the iroh feature");
