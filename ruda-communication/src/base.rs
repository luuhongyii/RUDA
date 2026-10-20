#[cfg(not(target_family = "wasm"))]
pub type ChannelFuture<T> = ruda_core::future::DynFut<T>;
#[cfg(target_family = "wasm")]
pub type ChannelFuture<T> = core::pin::Pin<Box<dyn Future<Output = T>>>;

#[cfg(not(target_family = "wasm"))]
pub trait MaybeSend: Send {}
#[cfg(not(target_family = "wasm"))]
impl<T: Send + ?Sized> MaybeSend for T {}
#[cfg(target_family = "wasm")]
pub trait MaybeSend {}
#[cfg(target_family = "wasm")]
impl<T: ?Sized> MaybeSend for T {}
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::str::FromStr;

/// Allows nodes to find each other
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct Address {
    pub(crate) inner: String,
    #[cfg(feature = "iroh")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) iroh: Option<crate::iroh::IrohHost>,
}

impl FromStr for Address {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some(id) = s.strip_prefix("iroh://") {
            #[cfg(feature = "iroh")]
            return id.parse::<::iroh::EndpointId>().map(crate::iroh::IrohHost::new)
                .map(Self::from).map_err(|error| error.to_string());
            #[cfg(not(feature = "iroh"))]
            { let _ = id; return Err("Iroh addresses require the iroh feature".into()); }
        }
        Ok(Self {
            inner: s.to_string(),
            #[cfg(feature = "iroh")]
            iroh: None,
        })
    }
}

/// Stable peer identity, independent of dialing hints and the client's own endpoint.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PeerId { WebSocket(String), Iroh([u8; 32]) }

impl Address {
    pub fn peer_id(&self) -> PeerId {
        #[cfg(feature = "iroh")]
        if let Some(host) = &self.iroh { return PeerId::Iroh(*host.id().as_bytes()); }
        PeerId::WebSocket(self.inner.clone())
    }

    pub fn same_peer(&self, other: &Self) -> bool { self.peer_id() == other.peer_id() }

    /// Explicit client identity, if the application supplies its own endpoint.
    pub fn application_peer_id(&self) -> Option<PeerId> {
        #[cfg(feature = "iroh")]
        if let Some(host) = &self.iroh {
            return host.application_endpoint_id().map(|id| PeerId::Iroh(*id.as_bytes()));
        }
        None
    }

    pub fn validate(&self) -> Result<(), String> {
        #[cfg(feature = "iroh")]
        if let Some(host) = &self.iroh { return host.validate(); }
        Ok(())
    }
}

#[cfg(feature = "iroh")]
impl From<crate::iroh::IrohHost> for Address {
    fn from(host: crate::iroh::IrohHost) -> Self {
        Self { inner: format!("iroh://{}", host.id()), iroh: Some(host) }
    }
}

#[cfg(not(target_family = "wasm"))]
pub type ClientConnector<C> = std::sync::Arc<dyn Fn(Address, String)
    -> ChannelFuture<Option<<C as ProtocolClient>::Channel>> + Send + Sync>;

impl Display for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
}

/// The protocol used for the communications.
pub trait Protocol: Clone + Send + Sync + 'static {
    /// The client implementation for the current protocol.
    type Client: ProtocolClient;
    /// The server implementation for the current protocol.
    #[cfg(not(target_family = "wasm"))]
    type Server: ProtocolServer;

    /// The server's outbound connector, preserving its transport identity for peer transfers.
    #[cfg(not(target_family = "wasm"))]
    fn connector(_server: &Self::Server) -> ClientConnector<Self::Client> {
        std::sync::Arc::new(|address, route| Self::Client::connect(address, &route))
    }
}

/// Error that happens during a communication.
pub trait CommunicationError: Debug + Send + 'static {}

/// The client is only used to create a [channel](CommunicationChannel), which should be use to
/// transmit information with the [server](ProtocolServer).
pub trait ProtocolClient: Send + Sync + 'static {
    /// Channel used by this protocol.
    type Channel: CommunicationChannel<Error = Self::Error>;
    /// The error type.
    type Error: CommunicationError;

    /// Opens a new [channel](CommunicationChannel) with the current protocol at the given
    /// [address](Address) and route.
    ///
    /// * `address` - Address to connect to
    /// * `route` - The name of the route (no slashes)
    ///
    /// Returns None if the connection can't be done.
    fn connect(address: Address, route: &str) -> ChannelFuture<Option<Self::Channel>>;
}

/// Data sent and received by the client and server.
#[derive(new)]
pub struct Message {
    /// The data is always encoded as bytes.
    pub data: bytes::Bytes,
}

/// Defines how to create a server that respond to a [channel](CommunicationChannel).
pub trait ProtocolServer: Sized + Send + Sync + 'static {
    /// Channel used by this protocol.
    type Channel: CommunicationChannel<Error = Self::Error>;
    /// The error type.
    type Error: CommunicationError;

    /// Defines an endpoint with the function that responds.
    /// TODO Docs: does it need a slash?
    fn route<C, Fut>(self, path: &str, callback: C) -> Self
    where
        C: FnOnce(Self::Channel) -> Fut + Clone + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static;

    /// Start the server.
    fn serve<F>(
        self,
        shutdown: F,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'static
    where
        F: Future<Output = ()> + Send + 'static;
}

/// Handles communications.
pub trait CommunicationChannel: MaybeSend + 'static {
    type Error: CommunicationError;

    /// Address supplied by the transport, when available; not an authenticated identity.
    fn peer_addr(&self) -> Option<std::net::SocketAddr> { None }

    /// Identity authenticated by the transport, rather than supplied in an application message.
    fn authenticated_peer(&self) -> Option<PeerId> { None }

    /// Send a [message](Message) on the channel.
    fn send(
        &mut self,
        message: Message,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>> + MaybeSend;

    /// Receive a [message](Message) on the channel and returns a new [response message](Message).
    fn recv(
        &mut self,
    ) -> impl std::future::Future<Output = Result<Option<Message>, Self::Error>> + MaybeSend;

    fn close(&mut self) -> impl std::future::Future<Output = Result<(), Self::Error>> + MaybeSend;
}
