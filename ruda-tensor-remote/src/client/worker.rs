use super::{RemoteClient, RemoteRuntime, ConnectError, connection::{Connection, ConnectionGuard}, runner::RemoteDevice};
use crate::shared::{ConnectionId, SessionId, SessionInit, SessionReply, PROTOCOL_VERSION, Task, TaskResponse, TaskResponseContent};
use ruda_communication::{CommunicationChannel, Message, ProtocolClient};
use std::{collections::HashMap, marker::PhantomData, sync::Arc};
use crate::telemetry::{GraphNamespace, GraphOp, TelemetryEvent, TelemetryProbe, serialized_len};

struct GraphRegistration { ops: Vec<GraphOp>, bytes: usize }
struct ObservedSession { session: SessionId, probe: TelemetryProbe }
impl Drop for ObservedSession {
    fn drop(&mut self) { self.probe.emit(|| TelemetryEvent::SessionClosed { session: self.session }); }
}

pub type CallbackSender = async_channel::Sender<TaskResponseContent>;

#[derive(Debug)]
pub enum ClientRequest {
    Flush,
    WithSyncCallback(Task, CallbackSender),
    WithoutCallback(Task),
}

struct PendingRequests(async_channel::Receiver<ClientRequest>);
impl Drop for PendingRequests {
    fn drop(&mut self) {
        self.0.close();
        while self.0.try_recv().is_ok() {}
    }
}

pub(crate) struct ClientWorker<C: ProtocolClient> {
    requests: HashMap<ConnectionId, CallbackSender>,
    graphs: HashMap<ConnectionId, GraphRegistration>,
    probe: TelemetryProbe,
    session: Option<SessionId>,
    _p: PhantomData<C>,
}

impl<C: ProtocolClient> ClientWorker<C> {
    async fn on_response(&mut self, response: TaskResponse) -> Result<(), ConnectError> {
        let callback = self.requests.remove(&response.id).ok_or_else(|| ConnectError::ConnectionClosed {
            reason: "The server returned an unknown or duplicate response identifier".into(),
        })?;
        if let Some(graph) = self.graphs.remove(&response.id)
            && let TaskResponseContent::RegisterGraph(Ok(id)) = &response.content
            && let Some(session) = self.session {
            self.probe.emit(|| TelemetryEvent::GraphRegistered { session, graph: *id,
                namespace: GraphNamespace::Acknowledged, ops: graph.ops, bytes: graph.bytes });
        }
        let _ = callback.send(response.content).await;
        Ok(())
    }

    fn register_callback(&mut self, id: ConnectionId, callback: CallbackSender) -> Result<(), ConnectError> {
        if self.requests.contains_key(&id) {
            return Err(ConnectError::ConnectionClosed { reason: "Duplicate pending request identifier".into() });
        }
        self.requests.insert(id, callback);
        Ok(())
    }

    pub fn start(device: RemoteDevice) -> RemoteClient {
        let device = device.refreshed();
        let runtime = Arc::new(RemoteRuntime::session());
        #[cfg(not(target_family = "wasm"))]
        let (sender, receiver) = async_channel::bounded(10);
        #[cfg(target_family = "wasm")]
        let (sender, receiver) = async_channel::unbounded();
        let session_id = SessionId::new();
        let client = RemoteClient::new(device.clone(), sender, runtime.clone(), session_id);
        let connection = client.connection.clone();
        let probe = client.probe.clone();
        if let Some(logger) = crate::metrics::logger_task(&probe, crate::metrics::MetricSide::Client) { runtime.spawn(logger); }
        let guard = ConnectionGuard(connection.clone());
        let task_runtime = runtime.clone();
        runtime.spawn(async move {
            let _task_runtime = task_runtime;
            let _guard = guard;
            let result = Self::run(device, session_id, PendingRequests(receiver), connection.clone(), probe).await;
            connection.fail(result.err().unwrap_or_else(|| ConnectError::ConnectionClosed {
                reason: "The client closed its request queue".into(),
            }));
        });
        client
    }

    async fn handshake(channel: &mut C::Channel, device: &RemoteDevice, session_id: SessionId)
        -> Result<crate::RemoteDeviceInfo, ConnectError> {
        let init = Task::InitSession(SessionInit {
            version: PROTOCOL_VERSION, session_id, device_index: device.device_index,
            authorization: device.credential.as_bytes().to_vec(),
        });
        let bytes = rmp_serde::to_vec(&init).map_err(|error| ConnectError::Handshake { reason: error.to_string() })?;
        channel.send(Message::new(bytes.into())).await
            .map_err(|error| ConnectError::Handshake { reason: format!("Cannot send initialization: {error:?}") })?;
        let reply = channel.recv().await
            .map_err(|error| ConnectError::Handshake { reason: format!("Cannot receive initialization: {error:?}") })?
            .ok_or_else(|| ConnectError::Handshake { reason: "Server closed before acknowledging initialization".into() })?;
        match rmp_serde::from_slice::<SessionReply>(&reply.data)
            .map_err(|error| ConnectError::Handshake { reason: format!("Invalid initialization reply: {error}") })? {
            SessionReply::Refused(refusal) => Err(refusal.into()),
            SessionReply::Accepted { version, device: info } => {
                if version != PROTOCOL_VERSION {
                    return Err(ConnectError::IncompatibleProtocol { client_version: PROTOCOL_VERSION, server_version: version });
                }
                if info.device_index != device.device_index || info.device_index >= info.device_count {
                    return Err(ConnectError::Handshake { reason: "Server acknowledged a different or invalid device".into() });
                }
                Ok(info)
            }
        }
    }

    async fn run(device: RemoteDevice, session_id: SessionId, pending: PendingRequests, connection: Arc<Connection>, probe: TelemetryProbe)
        -> Result<(), ConnectError> {
        device.address.validate().map_err(|reason| ConnectError::InvalidConfiguration { reason })?;
        let mut stream_request = C::connect(device.address.clone(), "request").await
            .ok_or_else(|| ConnectError::Unreachable { reason: format!("Cannot connect to {}", device.address) })?;
        let info = Self::handshake(&mut stream_request, &device, session_id).await?;
        let mut stream_response = C::connect(device.address.clone(), "response").await
            .ok_or_else(|| ConnectError::Handshake { reason: "Cannot connect the response channel".into() })?;
        let response_info = Self::handshake(&mut stream_response, &device, session_id).await?;
        if info != response_info {
            return Err(ConnectError::Handshake { reason: "Session connections disagree on the hosted device".into() });
        }
        connection.ready(info);
        probe.emit(|| TelemetryEvent::SessionOpened { session: session_id, device: device.device_index });
        let _observed_session = ObservedSession { session: session_id, probe: probe.clone() };
        let state = Arc::new(tokio::sync::Mutex::new(Self { probe: probe.clone(), session: Some(session_id), ..Self::default() }));
        let response_state = state.clone();
        let closing = std::sync::atomic::AtomicBool::new(false);
        let responses = async move {
            loop {
                let message = stream_response.recv().await
                    .map_err(|error| ConnectError::ConnectionClosed { reason: format!("Response receive failed: {error:?}") })?
                    .ok_or_else(|| ConnectError::ConnectionClosed { reason: "Server closed its response channel".into() })?;
                let response = rmp_serde::from_slice::<TaskResponse>(&message.data)
                    .map_err(|error| ConnectError::ConnectionClosed { reason: format!("Cannot decode response: {error}") })?;
                response_state.lock().await.on_response(response).await?;
            }
        };
        let requests = async {
            let mut batch = super::batch::OutgoingBatch::new();
            while let Ok(request) = pending.0.recv().await {
                let wants_response = matches!(&request, ClientRequest::WithSyncCallback(..));
                let task = match request {
                    ClientRequest::Flush => { Self::send_batch(&mut stream_request, &mut batch).await?; continue; }
                    ClientRequest::WithSyncCallback(task, callback) => {
                        if let Task::Compute(_, id) = &task {
                            let mut state = state.lock().await;
                            state.register_callback(*id, callback)?;
                            if probe.is_active() && let Task::Compute(crate::shared::ComputeTask::RegisterGraph(graph), _) = &task {
                                state.graphs.insert(*id, GraphRegistration {
                                    ops: graph.operations.iter().map(GraphOp::summarize).collect(), bytes: serialized_len(&graph.operations),
                                });
                            }
                        }
                        task
                    }
                    ClientRequest::WithoutCallback(task) => task,
                };
                let close = matches!(&task, Task::Close(_));
                if close { closing.store(true, std::sync::atomic::Ordering::Release); }
                if let Task::Compute(compute, id) = &task {
                    use crate::shared::ComputeTask;
                    match compute {
                        ComputeTask::RegisterOperation(op) => probe.emit(|| TelemetryEvent::unfused_op(session_id, id.stream_id, op)),
                        ComputeTask::ExecuteGraph(graph, bindings) => probe.emit(|| TelemetryEvent::graph_executed(session_id, *graph,
                            GraphNamespace::Acknowledged, id.stream_id, serialized_len(bindings))),
                        ComputeTask::QueueGraph { id: graph_id, graph, bindings } => {
                            probe.emit(|| TelemetryEvent::GraphRegistered { session: session_id, graph: *graph_id,
                                namespace: GraphNamespace::Queued,
                                ops: graph.operations.iter().map(GraphOp::summarize).collect(), bytes: serialized_len(&graph.operations) });
                            probe.emit(|| TelemetryEvent::graph_executed(session_id, *graph_id, GraphNamespace::Queued, id.stream_id, serialized_len(bindings)));
                        }
                        ComputeTask::QueueGraphReplay { id: graph_id, bindings } => probe.emit(||
                            TelemetryEvent::graph_executed(session_id, *graph_id, GraphNamespace::Queued, id.stream_id, serialized_len(bindings))),
                        ComputeTask::ReadTensor(_) => probe.emit(|| TelemetryEvent::Read { session: session_id, request: *id }),
                        ComputeTask::SyncBackend => probe.emit(|| TelemetryEvent::Sync { session: session_id, request: *id }),
                        _ => {},
                    }
                }
                let force = wants_response || super::batch::needs_flush(&task);
                if batch.push(task) || force {
                    Self::send_batch(&mut stream_request, &mut batch).await?;
                }
                if close {
                    let acknowledgement = stream_request.recv().await
                        .map_err(|error| ConnectError::ConnectionClosed { reason: format!("Close acknowledgement failed: {error:?}") })?
                        .ok_or_else(|| ConnectError::ConnectionClosed { reason: "Server disconnected before draining the session".into() })?;
                    let closed: crate::shared::SessionClosed = rmp_serde::from_slice(&acknowledgement.data)
                        .map_err(|error| ConnectError::ConnectionClosed { reason: format!("Invalid close acknowledgement: {error}") })?;
                    if closed.session != session_id { return Err(ConnectError::InvalidSession); }
                    return Ok(());
                }
            }
            Self::send_batch(&mut stream_request, &mut batch).await?;
            Ok(())
        };
        tokio::pin!(requests);
        tokio::select! {
            result = responses => {
                if closing.load(std::sync::atomic::Ordering::Acquire) { requests.await } else { result }
            },
            result = &mut requests => result,
        }
    }

    async fn send_batch(channel: &mut C::Channel, batch: &mut super::batch::OutgoingBatch) -> Result<(), ConnectError> {
        let Some(task) = batch.take() else { return Ok(()); };
        let bytes = rmp_serde::to_vec(&task).map_err(|error| ConnectError::ConnectionClosed { reason: error.to_string() })?;
        channel.send(Message::new(bytes.into())).await
            .map_err(|error| ConnectError::ConnectionClosed { reason: format!("Request send failed: {error:?}") })
    }
}

impl<C: ProtocolClient> Default for ClientWorker<C> {
    fn default() -> Self { Self { requests: HashMap::new(), graphs: HashMap::new(), probe: TelemetryProbe::disabled(), session: None, _p: PhantomData } }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::ComputeTask;
    use ruda_communication::{
        ProtocolServer,
        websocket::{WsClient, WsServer},
    };
    use ruda_core::id::StreamId;
    use std::time::Duration;
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn failed_connection_releases_queued_callbacks() {
        let (sender, receiver) = async_channel::bounded(2);
        let mut callbacks = Vec::new();
        for position in 0..2 {
            let (callback, response) = async_channel::bounded(1);
            sender
                .send(ClientRequest::WithSyncCallback(
                    Task::Compute(
                        ComputeTask::SyncBackend,
                        ConnectionId::new(position, StreamId::current()),
                    ),
                    callback,
                ))
                .await
                .unwrap();
            callbacks.push(response);
        }

        // An invalid scheme makes the real WebSocket client fail before reading the queue.
        ClientWorker::<WsClient>::run(
            "http://127.0.0.1".parse().unwrap(),
            SessionId::new(),
            PendingRequests(receiver),
        )
        .await;
        assert!(sender.is_closed());
        for callback in callbacks {
            assert!(
                tokio::time::timeout(Duration::from_secs(5), callback.recv())
                    .await
                    .expect("queued callback was retained after connection failure")
                    .is_err()
            );
        }
    }

    #[tokio::test]
    async fn cancelled_callback_does_not_stop_other_responses() {
        let mut worker = ClientWorker::<WsClient>::default();
        let (cancelled, receiver) = async_channel::bounded(1);
        drop(receiver);
        let cancelled_id = ConnectionId::new(0, StreamId::current());
        worker.register_callback(cancelled_id, cancelled);
        worker
            .on_response(TaskResponse {
                id: cancelled_id,
                content: TaskResponseContent::SyncBackend(Ok(())),
            })
            .await;

        let (callback, receiver) = async_channel::bounded(1);
        let id = ConnectionId::new(1, StreamId::current());
        worker.register_callback(id, callback);
        worker
            .on_response(TaskResponse {
                id,
                content: TaskResponseContent::SyncBackend(Ok(())),
            })
            .await;
        assert!(matches!(
            receiver.recv().await.unwrap(),
            TaskResponseContent::SyncBackend(Ok(()))
        ));
        assert!(worker.requests.is_empty());
    }

    #[test]
    fn response_disconnect_releases_pending_callback() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let address: Address = format!("ws://127.0.0.1:{port}").parse().unwrap();
        let shutdown = CancellationToken::new();
        let disconnect = Arc::new(tokio::sync::Notify::new());
        let server = WsServer::new(port)
            .route("/ready", |_| async {})
            .route("/request", {
                let disconnect = disconnect.clone();
                move |mut socket| async move {
                    while let Some(message) = socket.recv().await.unwrap() {
                        if matches!(
                            rmp_serde::from_slice::<Task>(&message.data).unwrap(),
                            Task::Compute(ComputeTask::SyncBackend, _)
                        ) {
                            disconnect.notify_one();
                            break;
                        }
                    }
                }
            })
            .route("/response", move |mut socket| async move {
                let message = socket.recv().await.unwrap().unwrap();
                assert!(matches!(
                    rmp_serde::from_slice::<Task>(&message.data).unwrap(),
                    Task::Init(_)
                ));
                disconnect.notified().await;
                socket.close().await.unwrap();
            });
        let stop = shutdown.clone();
        let server = rt.spawn(server.serve(async move { stop.cancelled().await }));
        rt.block_on(async {
            tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    if WsClient::connect(address.clone(), "ready").await.is_some() {
                        break;
                    }
                    assert!(
                        !server.is_finished(),
                        "test server stopped before becoming ready"
                    );
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .unwrap();
        });

        let client = RemoteClient::init::<WsClient>(RemoteDevice::new(&address.to_string()));
        let result = rt.block_on(async {
            tokio::time::timeout(
                Duration::from_secs(5),
                client.sender.send_async(ComputeTask::SyncBackend),
            )
            .await
        });
        // Keep the client alive while awaiting the result: dropping it must not be what
        // releases a callback retained by a stopped worker.
        assert!(
            result
                .expect("callback was retained after response disconnect")
                .is_err()
        );
        drop(client);
        shutdown.cancel();
        rt.block_on(async {
            tokio::time::timeout(Duration::from_secs(5), server)
                .await
                .unwrap()
                .unwrap()
                .unwrap();
        });
    }
}
