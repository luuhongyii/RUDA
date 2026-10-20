use ruda_communication::websocket::{WebSocket, WsServer};
use ruda_tensor::graph::BackendIr;
use ruda_tensor_router::Runner;
use super::{RemoteServer, ServerSettings, ServeError};

/// A native server transport, accepting WebSocket or feature-enabled Iroh configuration.
#[derive(Debug)]
pub struct Transport(TransportKind);

#[derive(Debug)]
enum TransportKind {
    WebSocket(WsServer),
    #[cfg(feature = "iroh")]
    Iroh(ruda_communication::iroh::IrohTransport),
}

impl From<WsServer> for Transport {
    fn from(server: WsServer) -> Self { Self(TransportKind::WebSocket(server)) }
}

#[cfg(feature = "iroh")]
impl From<ruda_communication::iroh::IrohTransport> for Transport {
    fn from(transport: ruda_communication::iroh::IrohTransport) -> Self { Self(TransportKind::Iroh(transport)) }
}

pub(crate) enum Listener {
    WebSocket(WsServer),
    #[cfg(feature = "iroh")]
    Iroh(ruda_communication::iroh::IrohServer),
}

impl Transport {
    pub(crate) async fn bind(self) -> Result<Listener, ServeError> {
        match self.0 {
            TransportKind::WebSocket(server) => server.bind().await.map(Listener::WebSocket)
                .map_err(|error| ServeError::Bind(format!("{error:?}"))),
            #[cfg(feature = "iroh")]
            TransportKind::Iroh(transport) => transport.bind().await.map(Listener::Iroh)
                .map_err(|error| ServeError::Bind(error.to_string())),
        }
    }
}

impl Listener {
    pub(crate) async fn serve<B: BackendIr>(self, runners: Vec<Runner<B>>, settings: ServerSettings,
        shutdown: impl Future<Output = ()> + Send + 'static) -> Result<(), ServeError> {
        match self {
            Self::WebSocket(server) => RemoteServer::<B, WebSocket>::serve(runners, server, settings, shutdown).await,
            #[cfg(feature = "iroh")]
            Self::Iroh(server) => RemoteServer::<B, ruda_communication::iroh::Iroh>::serve(runners, server, settings, shutdown).await,
        }
    }
}
