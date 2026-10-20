use ruda_communication::iroh::{AcceptError, Connection, Iroh, IrohRoutes, IrohServer, IrohTransport, ProtocolHandler};
use ruda_tensor::graph::BackendIr;
use ruda_tensor_router::Runner;
use super::{RemoteServer, ServerSettings, ServeError, base::ServerState};

pub struct IrohRemoteProtocol<B: BackendIr> { routes: IrohRoutes, state: ServerState<B, Iroh> }

impl<B: BackendIr> core::fmt::Debug for IrohRemoteProtocol<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { f.debug_struct("IrohRemoteProtocol").finish_non_exhaustive() }
}

impl<B: BackendIr> ProtocolHandler for IrohRemoteProtocol<B> {
    async fn accept(&self, connection: Connection) -> Result<(), AcceptError> {
        self.state.start_logger();
        self.routes.accept(connection).await
    }
    async fn shutdown(&self) {
        self.state.cancellation.cancel();
        self.routes.shutdown().await;
        self.state.shutdown().await;
    }
}

impl<B: BackendIr> RemoteServer<B, Iroh> {
    /// Attach configured runners to an application's existing Iroh router.
    pub fn into_protocol(runners: Vec<Runner<B>>, server: IrohServer, settings: ServerSettings)
        -> Result<IrohRemoteProtocol<B>, ServeError> {
        let (server, state) = Self::prepare(runners, server, settings)?;
        Ok(IrohRemoteProtocol { routes: server.into_protocol(), state })
    }
}

pub async fn serve_iroh<B: BackendIr>(runners: Vec<Runner<B>>, transport: IrohTransport,
    settings: ServerSettings, shutdown: impl Future<Output = ()> + Send + 'static) -> Result<(), ServeError> {
    let server = transport.bind().await.map_err(|error| ServeError::Transport(error.to_string()))?;
    RemoteServer::<B, Iroh>::serve(runners, server, settings, shutdown).await
}
