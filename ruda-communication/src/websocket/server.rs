use std::{net::{SocketAddr, TcpListener}, sync::Arc};

use crate::{
    base::{CommunicationChannel, CommunicationError, Message, ProtocolServer},
    util::init_logging,
};
use axum::{
    Router,
    extract::{
        ConnectInfo, State, WebSocketUpgrade,
        ws::{self, WebSocket},
    },
    routing::get,
};
use futures::StreamExt;

#[derive(Clone, Debug)]
pub struct WsServer {
    address: SocketAddr,
    router: Router<()>,
    listener: Option<Arc<TcpListener>>,
}

pub struct WsServerChannel {
    inner: WebSocket,
    peer: SocketAddr,
}

impl WsServer {
    /// Listen on the loopback interface only.
    pub fn new(port: u16) -> Self {
        Self::with_address(SocketAddr::from(([127, 0, 0, 1], port)))
    }

    /// Listen on an explicit address, e.g. `0.0.0.0:<port>` to accept other hosts.
    ///
    /// The transport has no authentication or TLS; only expose it on trusted networks.
    pub fn with_address(address: SocketAddr) -> Self {
        Self {
            address,
            router: Router::new(),
            listener: None,
        }
    }

    /// Serve an already-bound socket without releasing and rebinding its port.
    pub fn from_listener(listener: TcpListener) -> std::io::Result<Self> {
        let address = listener.local_addr()?;
        listener.set_nonblocking(true)?;
        Ok(Self { address, router: Router::new(), listener: Some(Arc::new(listener)) })
    }

    /// Bind before installing routes or shutdown handlers.
    pub async fn bind(mut self) -> Result<Self, WsServerError> {
        if self.listener.is_none() {
            let listener = tokio::net::TcpListener::bind(self.address).await?;
            self.address = listener.local_addr()?;
            self.listener = Some(Arc::new(listener.into_std()?));
        }
        Ok(self)
    }

    /// Listen on every interface.
    pub fn all_interfaces(port: u16) -> Self {
        Self::with_address(SocketAddr::from(([0, 0, 0, 0], port)))
    }
}

impl ProtocolServer for WsServer {
    type Channel = WsServerChannel;
    type Error = WsServerError;

    async fn serve<F>(self, shutdown: F) -> Result<(), Self::Error>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        init_logging();

        let server = self.bind().await?;
        let address = server.address;
        log::info!("Starting server {address}");

        let listener = match Arc::try_unwrap(server.listener.expect("bound WebSocket listener")) {
            Ok(listener) => listener,
            Err(listener) => listener.try_clone()?,
        };
        let listener = tokio::net::TcpListener::from_std(listener)?;

        axum::serve(
            listener,
            server.router
                .into_make_service_with_connect_info::<SocketAddr>(),
        )
        .with_graceful_shutdown(shutdown)
        .await?;

        Ok(())
    }

    fn route<C, Fut>(mut self, path: &str, callback: C) -> Self
    where
        C: FnOnce(WsServerChannel) -> Fut + Clone + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        // Format path: should start with a /
        let path = if path.starts_with("/") {
            path.to_owned()
        } else {
            format!("/{path}")
        };

        let method = get(|ws: WebSocketUpgrade, ConnectInfo(peer): ConnectInfo<SocketAddr>, _: State<()>| async move {
            ws.on_upgrade(async move |socket| {
                callback(WsServerChannel { inner: socket, peer }).await;
            })
        });

        self.router = self.router.route(&path, method);

        self
    }
}

impl CommunicationChannel for WsServerChannel {
    type Error = WsServerError;
    fn peer_addr(&self) -> Option<SocketAddr> { Some(self.peer) }

    async fn send(&mut self, message: Message) -> Result<(), WsServerError> {
        self.inner.send(ws::Message::Binary(message.data)).await?;

        Ok(())
    }

    async fn recv(&mut self) -> Result<Option<Message>, WsServerError> {
        match self.inner.next().await {
            Some(next) => match next {
                Ok(ws::Message::Binary(data)) => Ok(Some(Message { data })),
                Ok(ws::Message::Close(_close_frame)) => Ok(None),
                Err(err) => Err(WsServerError::Axum(err)),
                msg => Err(WsServerError::UnknownMessage(format!("{msg:?}"))),
            },
            None => Ok(None),
        }
    }

    async fn close(&mut self) -> Result<(), WsServerError> {
        let reason = "Peer is closing".to_string();

        self.inner
            .send(ws::Message::Close(Some(ws::CloseFrame {
                code: 1000, // code: Normal
                reason: reason.clone().into(),
            })))
            .await?;

        Ok(())
    }
}

#[derive(Debug)]
pub enum WsServerError {
    Io(std::io::Error),
    Axum(axum::Error),
    UnknownMessage(String),
    Other(String),
}

impl CommunicationError for WsServerError {}

impl From<std::io::Error> for WsServerError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<axum::Error> for WsServerError {
    fn from(err: axum::Error) -> Self {
        Self::Axum(err)
    }
}
