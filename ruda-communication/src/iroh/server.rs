use std::{collections::HashMap, sync::Arc};
use ::iroh::{EndpointAddr, EndpointId, endpoint::{BindOpts, Connection}, protocol::{AcceptError, ProtocolHandler, Router}};
use ruda_core::future::DynFut;
use tokio_util::sync::CancellationToken;
use crate::{CommunicationChannel, ProtocolServer};
use super::{IrohChannel, IrohError, IrohIdentity, IrohNode, IrohRelays, RUDA_REMOTE_ALPN, node::StreamHeader};

#[derive(Clone, Debug)]
pub struct IrohTransport { identity: Box<IrohIdentity>, relays: IrohRelays, port: Option<u16>, segmentation_offload: Option<bool> }
impl IrohTransport {
    pub fn new(identity: IrohIdentity) -> Self { Self { identity: Box::new(identity), relays: IrohRelays::default(), port: None, segmentation_offload: None } }
    pub fn with_relays(mut self, relays: IrohRelays) -> Self { self.relays = relays; self }
    pub fn with_port(mut self, port: u16) -> Self { self.port = Some(port); self }
    pub fn with_segmentation_offload(mut self, enabled: bool) -> Self { self.segmentation_offload = Some(enabled); self }
    pub fn id(&self) -> EndpointId { self.identity.id() }

    pub async fn bind(self) -> Result<IrohServer, IrohError> {
        let segmentation_offload = self.segmentation_offload.unwrap_or_else(|| ruda_tensor_config::config().remote.iroh_segmentation_offload);
        let mut builder = self.relays.builder(segmentation_offload).secret_key(self.identity.secret_key()).alpns(vec![RUDA_REMOTE_ALPN.to_vec()]);
        if let Some(port) = self.port {
            builder = builder.clear_ip_transports().bind_addr(format!("0.0.0.0:{port}"))
                .and_then(|builder| builder.bind_addr_with_opts(format!("[::]:{port}"), BindOpts::default().set_is_required(false)))
                .map_err(IrohError::from_error)?;
        }
        let endpoint = builder.bind().await.map_err(IrohError::from_error)?;
        Ok(IrohServer::new(IrohNode::for_endpoint(endpoint)?))
    }
}

type Route = Arc<dyn Fn(IrohChannel) -> DynFut<()> + Send + Sync>;

pub struct IrohServer { node: IrohNode, routes: HashMap<String, Route> }
impl IrohServer {
    pub fn new(node: IrohNode) -> Self { Self { node, routes: HashMap::new() } }
    pub fn node(&self) -> &IrohNode { &self.node }
    pub fn address(&self) -> EndpointAddr { self.node.endpoint().addr() }

    /// Register the returned handler in an application's router under RUDA_REMOTE_ALPN.
    pub fn into_protocol(self) -> IrohRoutes {
        IrohRoutes { routes: self.routes, shutdown: CancellationToken::new() }
    }
}

pub struct IrohRoutes { routes: HashMap<String, Route>, shutdown: CancellationToken }
impl core::fmt::Debug for IrohRoutes {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { f.debug_struct("IrohRoutes").field("routes", &self.routes.keys()).finish() }
}
impl Drop for IrohRoutes { fn drop(&mut self) { self.shutdown.cancel(); } }
impl ProtocolHandler for IrohRoutes {
    async fn accept(&self, connection: Connection) -> Result<(), AcceptError> {
        let mut streams = tokio::task::JoinSet::new();
        loop {
            let next = tokio::select! {
                _ = self.shutdown.cancelled() => break,
                _ = streams.join_next(), if !streams.is_empty() => continue,
                next = connection.accept_bi() => next,
            };
            let (send, recv) = match next {
                Ok(stream) => stream,
                Err(_) if connection.close_reason().is_some() => break,
                Err(error) => return Err(AcceptError::from_err(std::io::Error::other(error.to_string()))),
            };
            let mut channel = IrohChannel::new(connection.clone(), send, recv);
            let routes = self.routes.clone();
            let shutdown = self.shutdown.clone();
            streams.spawn(async move {
                let session = async {
                    let Some(message) = channel.recv().await? else { return Ok(()); };
                    let header: StreamHeader = rmp_serde::from_slice(&message.data).map_err(IrohError::from_error)?;
                    if header.version != 1 { return Err(IrohError("Incompatible Iroh stream version".into())); }
                    let callback = routes.get(&header.route).ok_or_else(|| IrohError("Unknown Iroh route".into()))?;
                    callback(channel).await;
                    Ok::<(), IrohError>(())
                };
                tokio::select! {
                    _ = shutdown.cancelled() => {},
                    result = session => if let Err(error) = result { log::warn!("Iroh stream failed: {error}"); },
                }
            });
        }
        while streams.join_next().await.is_some() {}
        Ok(())
    }

    async fn shutdown(&self) { self.shutdown.cancel(); }
}

impl ProtocolServer for IrohServer {
    type Channel = IrohChannel;
    type Error = IrohError;

    fn route<C, Fut>(mut self, path: &str, callback: C) -> Self
    where C: FnOnce(IrohChannel) -> Fut + Clone + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static {
        self.routes.insert(path.trim_start_matches('/').to_owned(), Arc::new(move |channel| Box::pin(callback.clone()(channel))));
        self
    }

    async fn serve<F>(self, shutdown: F) -> Result<(), IrohError>
    where F: Future<Output = ()> + Send + 'static {
        let endpoint = self.node.endpoint().clone();
        let closed = endpoint.closed();
        let router = RouterGuard(Some(Router::builder(endpoint.clone()).accept(RUDA_REMOTE_ALPN, self.into_protocol()).spawn()));
        let result = tokio::select! {
            _ = shutdown => Ok(()),
            _ = closed => Err(IrohError("Iroh endpoint closed while serving".into())),
        };
        router.shutdown().await?;
        result
    }
}

struct RouterGuard(Option<Router>);
impl RouterGuard {
    async fn shutdown(mut self) -> Result<(), IrohError> {
        if let Some(router) = self.0.take() { router.shutdown().await.map_err(IrohError::from_error)?; }
        Ok(())
    }
}
impl Drop for RouterGuard {
    fn drop(&mut self) {
        if let Some(router) = self.0.take() && let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move { let _ = router.shutdown().await; });
        }
    }
}
