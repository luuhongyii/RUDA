use crate::{Address, ClientConnector, CommunicationChannel, CommunicationError, Message, PeerId, Protocol, ProtocolClient, ProtocolServer};
use crate::iroh::{IrohChannel, IrohClient, IrohServer};
use crate::websocket::{WsClient, WsClientChannel, WsServer, WsServerChannel};
use ruda_core::future::DynFut;

#[derive(Clone)]
pub struct TransportProtocol;
pub struct TransportClient;
pub enum TransportServer { WebSocket(WsServer), Iroh(IrohServer) }
pub enum TransportChannel<W> { WebSocket(W), Iroh(IrohChannel) }

#[derive(Debug)]
pub struct TransportError(String);
impl CommunicationError for TransportError {}
impl core::fmt::Display for TransportError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { f.write_str(&self.0) }
}
impl std::error::Error for TransportError {}
fn transport_error(error: impl core::fmt::Debug) -> TransportError { TransportError(format!("{error:?}")) }

impl Protocol for TransportProtocol {
    type Client = TransportClient;
    type Server = TransportServer;

    fn connector(server: &TransportServer) -> ClientConnector<TransportClient> {
        let node = match server { TransportServer::Iroh(server) => Some(server.node().clone()), _ => None };
        std::sync::Arc::new(move |address, route| {
            let node = node.clone();
            Box::pin(async move {
                match (node, address.iroh.is_some()) {
                    (Some(node), true) => IrohClient::connect_from(&node, address, &route).await.ok().map(TransportChannel::Iroh),
                    (None, false) => WsClient::connect(address, &route).await.map(TransportChannel::WebSocket),
                    _ => None,
                }
            })
        })
    }
}

impl ProtocolClient for TransportClient {
    type Channel = TransportChannel<WsClientChannel>;
    type Error = TransportError;
    fn connect(address: Address, route: &str) -> DynFut<Option<Self::Channel>> {
        let route = route.to_owned();
        Box::pin(async move {
            if address.iroh.is_some() {
                IrohClient::connect(address, &route).await.map(TransportChannel::Iroh)
            } else {
                WsClient::connect(address, &route).await.map(TransportChannel::WebSocket)
            }
        })
    }
}

impl<W: CommunicationChannel> CommunicationChannel for TransportChannel<W> {
    type Error = TransportError;
    fn peer_addr(&self) -> Option<std::net::SocketAddr> {
        match self { Self::WebSocket(channel) => channel.peer_addr(), Self::Iroh(_) => None }
    }
    fn authenticated_peer(&self) -> Option<PeerId> {
        match self { Self::WebSocket(channel) => channel.authenticated_peer(), Self::Iroh(channel) => channel.authenticated_peer() }
    }
    async fn send(&mut self, message: Message) -> Result<(), TransportError> {
        match self { Self::WebSocket(channel) => channel.send(message).await.map_err(transport_error), Self::Iroh(channel) => channel.send(message).await.map_err(transport_error) }
    }
    async fn recv(&mut self) -> Result<Option<Message>, TransportError> {
        match self { Self::WebSocket(channel) => channel.recv().await.map_err(transport_error), Self::Iroh(channel) => channel.recv().await.map_err(transport_error) }
    }
    async fn close(&mut self) -> Result<(), TransportError> {
        match self { Self::WebSocket(channel) => channel.close().await.map_err(transport_error), Self::Iroh(channel) => channel.close().await.map_err(transport_error) }
    }
}

impl ProtocolServer for TransportServer {
    type Channel = TransportChannel<WsServerChannel>;
    type Error = TransportError;
    fn route<C, Fut>(self, path: &str, callback: C) -> Self
    where C: FnOnce(Self::Channel) -> Fut + Clone + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static {
        match self {
            Self::WebSocket(server) => Self::WebSocket(server.route(path, move |channel| callback(TransportChannel::WebSocket(channel)))),
            Self::Iroh(server) => Self::Iroh(server.route(path, move |channel| callback(TransportChannel::Iroh(channel)))),
        }
    }
    async fn serve<F>(self, shutdown: F) -> Result<(), TransportError>
    where F: Future<Output = ()> + Send + 'static {
        match self {
            Self::WebSocket(server) => server.serve(shutdown).await.map_err(transport_error),
            Self::Iroh(server) => server.serve(shutdown).await.map_err(transport_error),
        }
    }
}
impl From<WsServer> for TransportServer { fn from(server: WsServer) -> Self { Self::WebSocket(server) } }
impl From<IrohServer> for TransportServer { fn from(server: IrohServer) -> Self { Self::Iroh(server) } }
