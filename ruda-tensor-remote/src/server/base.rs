use ruda_communication::{
    CommunicationChannel, Message, Protocol, ProtocolServer,
    data_service::{TensorDataServer, TensorDataService}, util::os_shutdown_signal,
    websocket::{WebSocket, WsServer},
};
use std::{marker::PhantomData, net::SocketAddr, sync::Arc};
use tokio_util::sync::CancellationToken;
use ruda_tensor::tensor::Device;
use ruda_tensor::graph::BackendIr;
use ruda_tensor_router::Runner;
use futures_util::{stream::FuturesUnordered, StreamExt};
use crate::shared::{ComputeTask, SessionId, SessionReply, SessionRefusal, Task, TaskResponse, PROTOCOL_VERSION};
use super::{ClientId, ServerSettings, session::{SessionManager, SessionError}};

#[derive(Debug)]
pub enum ServeError {
    NoDevices,
    TooManyDevices,
    MixedBackends,
    UnsupportedDevice { reason: String },
    Bind(String),
    Transport(String),
    Runtime(String),
    SignalHandler(std::io::Error),
}
impl core::fmt::Display for ServeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoDevices => f.write_str("A remote server needs at least one device"),
            Self::TooManyDevices => f.write_str("Remote device catalog exceeds the device-index namespace"),
            Self::MixedBackends => f.write_str("A remote server's devices must share one backend family"),
            Self::UnsupportedDevice { reason } => write!(f, "Cannot host remote sessions: {reason}"),
            Self::Bind(reason) => write!(f, "Cannot bind the remote server: {reason}"),
            Self::Transport(reason) => write!(f, "Remote server transport failed: {reason}"),
            Self::Runtime(reason) => write!(f, "Remote server runtime failed: {reason}"),
            Self::SignalHandler(error) => write!(f, "Cannot install shutdown signal handlers: {error}"),
        }
    }
}
impl std::error::Error for ServeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self { Self::SignalHandler(error) => Some(error), _ => None }
    }
}

struct BoundSession {
    id: SessionId,
    versioned: bool,
    receiver: Option<tokio::sync::mpsc::Receiver<tokio::sync::mpsc::Receiver<TaskResponse>>>,
    closed: CancellationToken,
}

pub struct RemoteServer<B: BackendIr, P: Protocol> { _b: PhantomData<B>, _n: PhantomData<P> }

pub(crate) struct ServerState<B: BackendIr, P: Protocol> {
    pub(crate) cancellation: CancellationToken,
    sessions: Arc<SessionManager<B, P>>,
    data: Arc<TensorDataService<B, P>>,
    logger: std::sync::Mutex<Option<ruda_core::future::DynFut<()>>>,
}
impl<B: BackendIr, P: Protocol> ServerState<B, P> {
    pub(crate) fn start_logger(&self) {
        if let Ok(runtime) = tokio::runtime::Handle::try_current()
            && let Some(logger) = self.logger.lock().unwrap().take() { runtime.spawn(logger); }
    }
    pub(crate) async fn shutdown(&self) {
        self.cancellation.cancel();
        self.sessions.close_all().await;
        self.data.shutdown().await;
    }
}
impl<B: BackendIr, P: Protocol> Drop for ServerState<B, P> {
    fn drop(&mut self) {
        self.cancellation.cancel();
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            let sessions = self.sessions.clone();
            let data = self.data.clone();
            runtime.spawn(async move { sessions.close_all().await; data.shutdown().await; });
        }
    }
}

impl<B: BackendIr, P: Protocol> RemoteServer<B, P> {
    pub async fn start(device: Device<B>, server: P::Server) {
        Self::start_with_runner(Runner::new(device), server).await;
    }

    pub async fn start_with_runner(runner: Runner<B>, server: P::Server) {
        Self::start_with_runners(vec![runner], server).await;
    }

    pub async fn start_with_runners(runners: Vec<Runner<B>>, server: P::Server) {
        Self::start_with_settings(runners, server, ServerSettings::default()).await;
    }

    /// Start with an explicit authorizer; configured runners retain their custom operations.
    pub async fn start_with_settings(runners: Vec<Runner<B>>, server: P::Server, settings: ServerSettings) {
        if let Err(error) = Self::serve(runners, server, settings, os_shutdown_signal()).await {
            log::error!("{error}");
        }
    }

    /// Serve until the supplied shutdown future resolves, then close all sessions and data links.
    pub async fn serve(runners: Vec<Runner<B>>, server: P::Server, settings: ServerSettings,
        shutdown: impl Future<Output = ()> + Send + 'static) -> Result<(), ServeError> {
        let (server, state) = Self::prepare(runners, server, settings)?;
        state.start_logger();
        let stop = state.cancellation.clone();
        let result = server.serve(async move { shutdown.await; stop.cancel(); }).await;
        state.shutdown().await;
        result.map_err(|error| ServeError::Transport(format!("{error:?}")))
    }

    pub(crate) fn prepare(runners: Vec<Runner<B>>, server: P::Server, settings: ServerSettings)
        -> Result<(P::Server, ServerState<B, P>), ServeError> {
        if runners.is_empty() { return Err(ServeError::NoDevices); }
        if runners.len() > u16::MAX as usize + 1 { return Err(ServeError::TooManyDevices); }
        let cancellation = CancellationToken::new();
        let logger = crate::metrics::logger_task(&settings.telemetry, crate::metrics::MetricSide::Server)
            .map(|logger| Box::pin(logger) as ruda_core::future::DynFut<()>);
        let data_service = Arc::new(TensorDataService::<B, P>::new(cancellation.clone()).with_connector(P::connector(&server)));
        let session_manager = Arc::new(SessionManager::<B, P>::with_runners(runners, data_service.clone())
            .with_settings(settings, cancellation.clone()));
        let server = server.route("/response", {
            let manager = session_manager.clone();
            move |stream| Self::handle_socket_response(manager, stream)
        }).route("/request", {
            let manager = session_manager.clone();
            move |stream| Self::handle_socket_request(manager, stream)
        }).route_tensor_data_service(data_service.clone());
        Ok((server, ServerState { cancellation, sessions: session_manager, data: data_service, logger: std::sync::Mutex::new(logger) }))
    }

    async fn reply(socket: &mut <P::Server as ProtocolServer>::Channel, reply: SessionReply) -> bool {
        let Ok(bytes) = rmp_serde::to_vec(&reply) else { return false; };
        socket.send(Message::new(bytes.into())).await.is_ok()
    }

    async fn initialize(manager: &SessionManager<B, P>, socket: &mut <P::Server as ProtocolServer>::Channel,
        request: bool) -> Option<BoundSession> {
        let shutdown = manager.shutdown_token();
        let message = tokio::select! {
            _ = shutdown.cancelled() => return None,
            message = socket.recv() => match message { Ok(Some(message)) => message, _ => return None },
        };
        let (id, index, credential, versioned) = match rmp_serde::from_slice::<Task>(&message.data) {
            Ok(Task::InitSession(init)) => {
                if init.version != PROTOCOL_VERSION {
                    Self::reply(socket, SessionReply::Refused(SessionRefusal::IncompatibleProtocol { server_version: PROTOCOL_VERSION })).await;
                    return None;
                }
                (init.session_id, init.device_index, crate::Credential::from(init.authorization), true)
            }
            Ok(Task::Init(id)) => (id, 0, crate::Credential::default(), false),
            Ok(Task::InitDevice(id, index)) => (id, index, crate::Credential::default(), false),
            _ => {
                Self::reply(socket, SessionReply::Refused(SessionRefusal::IncompatibleProtocol { server_version: PROTOCOL_VERSION })).await;
                return None;
            }
        };
        let client = match socket.authenticated_peer() {
            Some(ruda_communication::PeerId::Iroh(id)) => ClientId::Iroh(id),
            _ => socket.peer_addr().map(ClientId::WebSocket).unwrap_or(ClientId::Unknown),
        };
        if let Err(refusal) = manager.authorize(client, index, &credential) {
            if versioned { Self::reply(socket, SessionReply::Refused(refusal)).await; }
            return None;
        }
        let bound = if request {
            manager.bind_request_from(id, index, versioned, socket.authenticated_peer()).await.map(|(info, closed)| (None, info, closed))
        } else {
            manager.bind_response_from(id, index, versioned, socket.authenticated_peer()).await.map(|(receiver, info, closed)| (Some(receiver), info, closed))
        };
        match bound {
            Ok((receiver, info, closed)) => {
                if versioned && !Self::reply(socket, SessionReply::Accepted { version: PROTOCOL_VERSION, device: info }).await {
                    manager.close(Some(id)).await;
                    return None;
                }
                Some(BoundSession { id, receiver, closed, versioned })
            }
            Err(error) => {
                let refusal = match error {
                    SessionError::InvalidDeviceIndex(_) => SessionRefusal::NoSuchDevice { device_count: manager.device_count() },
                    _ => SessionRefusal::InvalidSession,
                };
                if versioned { Self::reply(socket, SessionReply::Refused(refusal)).await; }
                None
            }
        }
    }

    async fn handle_socket_response(session_manager: Arc<SessionManager<B, P>>, mut socket: <P::Server as ProtocolServer>::Channel) {
        let Some(bound) = Self::initialize(&session_manager, &mut socket, false).await else { return; };
        let id = bound.id;
        let mut receiver = bound.receiver.expect("response binding");
        let shutdown = session_manager.shutdown_token();
        let mut responses = FuturesUnordered::new();
        let mut receiving = true;
        loop {
            if !receiving && responses.is_empty() { break; }
            let response = tokio::select! {
                _ = shutdown.cancelled() => break,
                _ = bound.closed.cancelled() => break,
                _ = socket.recv() => break,
                callback = receiver.recv(), if receiving => {
                    match callback {
                        Some(mut callback) => responses.push(async move { callback.recv().await }),
                        None => receiving = false,
                    }
                    continue;
                }
                response = responses.next(), if !responses.is_empty() => response.flatten(),
            };
            let Some(response) = response else { break; };
            let bytes = match rmp_serde::to_vec(&response) {
                Ok(bytes) => bytes,
                Err(error) => { log::error!("Cannot serialize remote response: {error}"); break; }
            };
            let sent = tokio::select! {
                _ = shutdown.cancelled() => break,
                _ = bound.closed.cancelled() => break,
                sent = socket.send(Message::new(bytes.into())) => sent,
            };
            if sent.is_err() { break; }
        }
        drop(receiver);
        session_manager.close(Some(id)).await;
    }

    async fn handle_socket_request(session_manager: Arc<SessionManager<B, P>>, mut socket: <P::Server as ProtocolServer>::Channel) {
        let Some(bound) = Self::initialize(&session_manager, &mut socket, true).await else { return; };
        let mut session_id = Some(bound.id);
        let shutdown = session_manager.shutdown_token();
        let mut queued = std::collections::VecDeque::new();
        let mut acknowledge_close = false;
        loop {
            if shutdown.is_cancelled() || bound.closed.is_cancelled() { break; }
            let task = if let Some(task) = queued.pop_front() { task } else {
                let packet = tokio::select! {
                    _ = shutdown.cancelled() => break,
                    _ = bound.closed.cancelled() => break,
                    packet = socket.recv() => packet,
                };
                let msg = match packet { Ok(Some(message)) => message, _ => break };
                let task = match rmp_serde::from_slice::<Task>(&msg.data) {
                    Ok(task) => task,
                    Err(error) => { log::warn!("Invalid remote task: {error}"); break; }
                };
                if let Task::Batch(tasks) = task {
                    queued.extend(tasks);
                    continue;
                }
                task
            };
            if let Task::Close(id) = task {
                if session_id != Some(id) { log::warn!("Remote connection attempted to close another session"); }
                else { acknowledge_close = bound.versioned; }
                break;
            }
            let (stream, connection_id, task) = match session_manager.stream(&mut session_id, task).await {
                Ok(Some(value)) => value,
                Ok(None) => continue,
                Err(error) => { log::warn!("Invalid request in session {session_id:?}: {error:?}"); break; }
            };
            let result = match task {
                ComputeTask::RegisterOperation(op) => stream.register_operation(op).await,
                ComputeTask::RegisterTensor(id, data) => stream.register_tensor(id, data).await,
                ComputeTask::ReadTensor(tensor) => stream.read_tensor(connection_id, tensor).await,
                ComputeTask::SyncBackend => stream.sync(connection_id).await,
                ComputeTask::RegisterTensorRemote(tensor, new_id) => {
                    stream.register_tensor_remote(tensor, new_id).await
                }
                ComputeTask::RegisterTensorLocal { transfer_id, new_id } => {
                    stream.register_tensor_local(transfer_id, new_id).await
                }
                ComputeTask::ExposeTensorLocal { tensor, transfer_id } => {
                    stream.expose_tensor_local(tensor, transfer_id).await
                }
                ComputeTask::ExposeTensorRemote {
                    tensor,
                    count,
                    transfer_id,
                } => {
                    stream
                        .expose_tensor_remote(tensor, count, transfer_id)
                        .await
                }
                ComputeTask::Seed(seed) => stream.seed(seed).await,
                ComputeTask::ExposeTensorCapability { tensor, count, capability, target } => {
                    stream.expose_tensor_capability(tensor, count, capability, target).await
                }
                ComputeTask::RegisterTensorCapability { address, capability, new_id } => {
                    stream.register_tensor_capability(address, capability, new_id).await
                }
                ComputeTask::DTypeUsage(dtype) => stream.dtype_usage(connection_id, dtype).await,
                ComputeTask::RegisterGraph(graph) => stream.register_graph(connection_id, graph).await,
                ComputeTask::ExecuteGraph(graph, bindings) => stream.execute_graph(connection_id, graph, bindings).await,
                ComputeTask::RemoveGraph(graph) => stream.remove_graph(connection_id, graph).await,
                ComputeTask::QueueGraph { id, graph, bindings } => {
                    let graph = match session_manager.queued_graph(bound.id, id, Some(graph)).await {
                        Ok(graph) => graph,
                        Err(error) => { log::warn!("Remote graph registration failed: {error:?}"); break; }
                    };
                    stream.queue_graph_replay(id, graph, bindings).await
                }
                ComputeTask::QueueGraphReplay { id, bindings } => {
                    let graph = match session_manager.queued_graph(bound.id, id, None).await {
                        Ok(graph) => graph,
                        Err(error) => { log::warn!("Remote graph lookup failed: {error:?}"); break; }
                    };
                    stream.queue_graph_replay(id, graph, bindings).await
                }
                ComputeTask::QueueGraphRelease(id) => {
                    if let Err(error) = session_manager.release_queued_graph(bound.id, id).await {
                        log::warn!("Remote graph release failed: {error:?}");
                        break;
                    }
                    Ok(())
                }
                ComputeTask::QueueAlias { destination, source } => stream.queue_alias(destination, source).await,
                ComputeTask::RegisterAlias { destination, source } => stream.register_alias(connection_id, destination, source).await,
                ComputeTask::FlushBackend => stream.flush(connection_id).await,
                ComputeTask::ProfileStart => stream.profile_start(connection_id).await,
                ComputeTask::ProfileEnd(token, options) => stream.profile_end(connection_id, token, options).await,
                ComputeTask::ProfileAbandon(token) => stream.profile_abandon(token).await,
                ComputeTask::DeviceInfo => {
                    let Some(info) = session_manager.device_info(session_id.expect("initialized request session")).await else { break; };
                    stream.device_info(connection_id, info).await
                }
                ComputeTask::SyncCollective => stream.sync_collective(connection_id).await,
            };
            if result.is_err() {
                log::warn!("Stream of session {session_id:?} stopped, closing.");
                break;
            }
        }
        session_manager.close(session_id).await;
        if acknowledge_close && let Ok(bytes) = rmp_serde::to_vec(&crate::shared::SessionClosed { session: bound.id }) {
            let _ = socket.send(Message::new(bytes.into())).await;
        }
    }
}

/// Start the server on the given loopback port and [device](Device).
pub async fn start_websocket_async<B: BackendIr>(device: Device<B>, port: u16) {
    start_websocket_on_async::<B>(device, SocketAddr::from(([127, 0, 0, 1], port))).await;
}

/// Start the server on an explicit address, e.g. `0.0.0.0:3000` to accept other hosts.
///
/// The transport has no authentication or TLS; only expose it on trusted networks.
pub async fn start_websocket_on_async<B: BackendIr>(device: Device<B>, address: SocketAddr) {
    let server = WsServer::with_address(address);
    RemoteServer::<B, WebSocket>::start(device, server).await;
}

/// Host an ordered list of devices on one WebSocket endpoint.
pub async fn start_websocket_devices_on_async<B: BackendIr>(devices: Vec<Device<B>>, address: SocketAddr) {
    let runners = devices.into_iter().map(Runner::<B>::new).collect();
    RemoteServer::<B, WebSocket>::start_with_runners(runners, WsServer::with_address(address)).await;
}

/// Host native-collective-capable devices without introducing host-staged collectives.
#[cfg(feature = "distributed")]
pub async fn start_websocket_distributed_on_async<B: BackendIr + ruda_tensor::distributed::DistributedBackend>(
    devices: Vec<Device<B>>, address: SocketAddr,
) {
    let runners = devices.into_iter().map(Runner::<B>::new_distributed).collect();
    RemoteServer::<B, WebSocket>::start_with_runners(runners, WsServer::with_address(address)).await;
}

#[tokio::main]
/// Start the server on the given loopback port and [device](Device).
pub async fn start_websocket<B: BackendIr>(device: Device<B>, port: u16) {
    start_websocket_async::<B>(device, port).await;
}

#[tokio::main]
/// Start the server on an explicit address and [device](Device).
pub async fn start_websocket_on<B: BackendIr>(device: Device<B>, address: SocketAddr) {
    start_websocket_on_async::<B>(device, address).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::SessionId;
    use ruda_communication::{Address, ProtocolClient, websocket::{WsClient, WsClientChannel}};
    use ruda_tensor_host::Host;
    use std::time::Duration;

    async fn test_server() -> (Arc<SessionManager<Host, WebSocket>>, Address,
        CancellationToken, tokio::task::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let cancel = CancellationToken::new();
        let data_service = Arc::new(TensorDataService::new(cancel.clone()));
        let manager = Arc::new(SessionManager::new(Default::default(), data_service));
        let server = WsServer::new(port)
            .route("/request", {
                let manager = manager.clone();
                move |socket| RemoteServer::<Host, WebSocket>::handle_socket_request(manager, socket)
            })
            .route("/response", {
                let manager = manager.clone();
                move |socket| RemoteServer::<Host, WebSocket>::handle_socket_response(manager, socket)
            });
        let shutdown = cancel.clone();
        let worker = tokio::spawn(async move {
            server.serve(async move { shutdown.cancelled().await }).await.unwrap();
        });
        (manager, format!("ws://127.0.0.1:{port}").parse().unwrap(), cancel, worker)
    }

    async fn connect(address: Address, route: &str) -> WsClientChannel {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Some(socket) = WsClient::connect(address.clone(), route).await {
                    return socket;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }).await.expect("test server did not become ready")
    }

    async fn send(socket: &mut WsClientChannel, task: Task) {
        socket.send(Message::new(rmp_serde::to_vec(&task).unwrap().into())).await.unwrap();
    }

    async fn wait_count(manager: &SessionManager<Host, WebSocket>, count: usize) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while manager.session_count().await != count { tokio::task::yield_now().await; }
        }).await.expect("session cleanup did not finish");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn websocket_close_cannot_target_another_session() {
        let (manager, address, cancel, worker) = test_server().await;
        let victim = SessionId::new();
        let mut current = None;
        manager.stream(&mut current, Task::Init(victim)).await.unwrap();
        let mut request = connect(address, "request").await;
        send(&mut request, Task::Init(SessionId::new())).await;
        wait_count(&manager, 2).await;
        send(&mut request, Task::Close(victim)).await;
        wait_count(&manager, 1).await;
        assert!(manager.stream(&mut current, Task::Compute(ComputeTask::Seed(1),
            crate::shared::ConnectionId::new(0, ruda_core::id::StreamId::current())))
            .await.unwrap().is_some());
        manager.close(current).await;
        drop(request);
        cancel.cancel();
        worker.await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn idle_responder_disconnect_removes_session() {
        let (manager, address, cancel, worker) = test_server().await;
        let mut response = connect(address, "response").await;
        send(&mut response, Task::Init(SessionId::new())).await;
        wait_count(&manager, 1).await;
        response.close().await.unwrap();
        wait_count(&manager, 0).await;
        drop(response);
        cancel.cancel();
        worker.await.unwrap();
    }
}
