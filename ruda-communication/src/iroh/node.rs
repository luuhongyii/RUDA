use std::{collections::HashMap, sync::{Arc, LazyLock, Mutex as StdMutex, OnceLock, Weak}};
use ::iroh::{Endpoint, EndpointAddr, EndpointId, endpoint::Connection};
use tokio::sync::{Mutex, OnceCell};
use super::{IrohChannel, IrohError, IrohRelays};
use crate::{CommunicationChannel, Message};

pub const RUDA_REMOTE_ALPN: &[u8] = b"ruda/remote/1";

struct NodeInner {
    endpoint: Endpoint,
    connections: Mutex<HashMap<EndpointId, Arc<OnceCell<Connection>>>>,
}

#[derive(Clone)]
pub struct IrohNode(Arc<NodeInner>);

static APP_NODES: LazyLock<StdMutex<HashMap<EndpointId, Weak<NodeInner>>>> = LazyLock::new(|| StdMutex::new(HashMap::new()));
static OWNED_NODES: LazyLock<StdMutex<HashMap<(IrohRelays, bool), Arc<OnceCell<IrohNode>>>>> = LazyLock::new(|| StdMutex::new(HashMap::new()));
#[cfg(not(target_family = "wasm"))]
static OWNED_RUNTIME: OnceLock<Result<tokio::runtime::Runtime, String>> = OnceLock::new();

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct StreamHeader { pub version: u16, pub route: String }

impl IrohNode {
    pub fn for_endpoint(endpoint: Endpoint) -> Result<Self, IrohError> {
        let mut nodes = APP_NODES.lock().unwrap();
        nodes.retain(|_, node| node.strong_count() != 0);
        if let Some(existing) = nodes.get(&endpoint.id()).and_then(Weak::upgrade)
            && !existing.endpoint.is_closed() {
            #[cfg(not(target_family = "wasm"))]
            if existing.endpoint.bound_sockets() != endpoint.bound_sockets() {
                return Err(IrohError("Two live Iroh endpoints use the same identity on different sockets".into()));
            }
            return Ok(Self(existing));
        }
        let node = Self(Arc::new(NodeInner { endpoint, connections: Mutex::new(HashMap::new()) }));
        nodes.insert(node.id(), Arc::downgrade(&node.0));
        Ok(node)
    }

    pub(crate) async fn for_relays(relays: IrohRelays, segmentation_offload: bool) -> Result<Self, IrohError> {
        #[cfg(not(target_family = "wasm"))]
        let runtime = OWNED_RUNTIME.get_or_init(|| tokio::runtime::Builder::new_multi_thread()
            .enable_all().build().map_err(|error| error.to_string())).as_ref().map_err(|error| IrohError(error.clone()))?;
        loop {
            let key = (relays.clone(), segmentation_offload);
            let cell = OWNED_NODES.lock().unwrap().entry(key.clone()).or_default().clone();
            if let Some(node) = cell.get() && node.endpoint().is_closed() {
                let mut nodes = OWNED_NODES.lock().unwrap();
                if nodes.get(&key).is_some_and(|current| Arc::ptr_eq(current, &cell)) { nodes.remove(&key); }
                continue;
            }
            let relays = relays.clone();
            #[cfg(not(target_family = "wasm"))]
            return runtime.spawn(async move {
                cell.get_or_try_init(|| async move {
                    let endpoint = relays.builder(segmentation_offload).bind().await.map_err(IrohError::from_error)?;
                    Self::for_endpoint(endpoint)
                }).await.cloned()
            }).await.map_err(IrohError::from_error)?;
            #[cfg(target_family = "wasm")]
            return cell.get_or_try_init(|| async move {
                let endpoint = relays.builder(segmentation_offload).bind().await.map_err(IrohError::from_error)?;
                Self::for_endpoint(endpoint)
            }).await.cloned();
        }
    }

    pub fn id(&self) -> EndpointId { self.0.endpoint.id() }
    pub fn endpoint(&self) -> &Endpoint { &self.0.endpoint }

    pub(crate) async fn open(&self, peer: EndpointAddr, route: &str) -> Result<IrohChannel, IrohError> {
        let connection = self.connection(peer).await?;
        let (send, recv) = connection.open_bi().await.map_err(IrohError::from_error)?;
        let mut channel = IrohChannel::new(connection, send, recv);
        channel.retain_node(self.clone());
        let header = StreamHeader { version: 1, route: route.trim_start_matches('/').into() };
        channel.send(Message::new(rmp_serde::to_vec(&header).map_err(IrohError::from_error)?.into())).await?;
        Ok(channel)
    }

    async fn connection(&self, peer: EndpointAddr) -> Result<Connection, IrohError> {
        loop {
            let cell = self.0.connections.lock().await.entry(peer.id).or_default().clone();
            if let Some(connection) = cell.get() && connection.close_reason().is_some() {
                let mut connections = self.0.connections.lock().await;
                if connections.get(&peer.id).is_some_and(|current| Arc::ptr_eq(current, &cell)) { connections.remove(&peer.id); }
                continue;
            }
            return cell.get_or_try_init(|| async {
                self.endpoint().connect(peer.clone(), RUDA_REMOTE_ALPN).await.map_err(IrohError::from_error)
            }).await.cloned();
        }
    }
}

impl core::fmt::Debug for IrohNode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { f.debug_struct("IrohNode").field("id", &self.id()).finish() }
}
