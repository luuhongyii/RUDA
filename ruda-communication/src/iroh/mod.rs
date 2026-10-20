mod channel;
mod host;
mod identity;
mod node;
#[cfg(not(target_family = "wasm"))]
mod server;

pub use channel::{IrohChannel, IrohClient, IrohError};
pub use host::{IrohHost, IrohRelays};
pub use identity::IrohIdentity;
pub use node::{IrohNode, RUDA_REMOTE_ALPN};
#[cfg(not(target_family = "wasm"))]
pub use server::{IrohRoutes, IrohServer, IrohTransport};
pub use ::iroh::protocol::{AcceptError, ProtocolHandler};
pub use ::iroh::endpoint::Connection;
pub use ::iroh::{Endpoint, EndpointAddr, EndpointId, RelayUrl};

#[derive(Clone)]
pub struct Iroh;

impl crate::Protocol for Iroh {
    type Client = IrohClient;
    #[cfg(not(target_family = "wasm"))]
    type Server = IrohServer;

    #[cfg(not(target_family = "wasm"))]
    fn connector(server: &IrohServer) -> crate::ClientConnector<IrohClient> {
        let node = server.node().clone();
        std::sync::Arc::new(move |address, route| {
            let node = node.clone();
            Box::pin(async move { IrohClient::connect_from(&node, address, &route).await.ok() })
        })
    }
}
