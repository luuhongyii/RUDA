pub(crate) mod processor;
pub(crate) mod session;
pub(crate) mod stream;

mod base;
mod backend;
mod transport;
mod local_comm;
mod authorize;
#[cfg(feature = "iroh")]
mod iroh;
#[cfg(feature = "iroh")]
pub use iroh::{IrohRemoteProtocol, serve_iroh};
pub use authorize::{AllowAll, AuthorizationRequest, ClientId, EmptyToken, PeerAuthorizer, ServerSettings, TokenAuthorizer};

pub use base::{start_websocket, start_websocket_async, start_websocket_on, start_websocket_on_async};
pub use base::{RemoteServer, ServeError, start_websocket_devices_on_async};
pub use backend::BackendServer;
pub use transport::Transport;
pub use ruda_tensor_router::CustomOpRegistry;
pub use ruda_communication::websocket::WsServer as WebSocketTransport;
#[cfg(feature = "distributed")]
pub use base::start_websocket_distributed_on_async;
