use ruda_communication::{
    CommunicationChannel, Message, Protocol, ProtocolServer,
    data_service::{TensorDataServer, TensorDataService},
    util::os_shutdown_signal,
    websocket::{WebSocket, WsServer},
};
use std::{marker::PhantomData, sync::Arc};
use tokio_util::sync::CancellationToken;

use ruda_tensor::tensor::Device;
use ruda_tensor::graph::BackendIr;

use crate::shared::{ComputeTask, Task};

use super::session::SessionManager;

pub struct RemoteServer<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    _b: PhantomData<B>,
    _n: PhantomData<P>,
}

impl<B, P> RemoteServer<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    /// Start the server on the given address.
    pub async fn start(device: Device<B>, server: P::Server) {
        let cancel_token = CancellationToken::new();
        let data_service = Arc::new(TensorDataService::<B, P>::new(cancel_token));
        let session_manager = Arc::new(SessionManager::<B, P>::new(device, data_service.clone()));

        let _server = server
            .route("/response", {
                let session_manager = session_manager.clone();
                move |stream| Self::handle_socket_response(session_manager, stream)
            })
            .route("/request", {
                let session_manager = session_manager.clone();
                move |stream| Self::handle_socket_request(session_manager, stream)
            })
            .route_tensor_data_service(data_service)
            .serve(os_shutdown_signal())
            .await;
    }

    async fn handle_socket_response(
        session_manager: Arc<SessionManager<B, P>>,
        mut socket: <P::Server as ProtocolServer>::Channel,
    ) {
        log::info!("[Response Handler] On new connection.");

        let packet = socket.recv().await;
        let msg = match packet {
            Ok(Some(msg)) => msg,
            Ok(None) => {
                log::info!("Response stream closed");
                return;
            }
            Err(e) => {
                log::info!("Response stream error on init: {e:?}");
                return;
            }
        };

        let id = match rmp_serde::from_slice::<Task>(&msg.data) {
            Ok(Task::Init(session_id)) => session_id,
            msg => {
                log::error!("Message is not a valid initialization task {msg:?}");
                return;
            }
        };

        let Some(mut receiver) = session_manager.register_responder(id).await else {
            log::error!("Session {id} already has a responder");
            return;
        };

        log::info!("Response handler connection active");

        loop {
            let callback = tokio::select! {
                _ = socket.recv() => break,
                callback = receiver.recv() => callback,
            };
            let Some(mut callback) = callback else { break };
            let response = tokio::select! {
                _ = socket.recv() => break,
                response = callback.recv() => response,
            };
            let Some(response) = response else { break };
            let bytes = match rmp_serde::to_vec(&response) {
                Ok(bytes) => bytes,
                Err(err) => {
                    log::error!("Can't serialize response for session {id}: {err:?}");
                    break;
                }
            };
            if let Err(err) = socket.send(Message::new(bytes.into())).await {
                log::info!("Response stream error for session {id}: {err:?}");
                break;
            }
        }
        drop(receiver);
        session_manager.close(Some(id)).await;
    }

    async fn handle_socket_request(
        session_manager: Arc<SessionManager<B, P>>,
        mut socket: <P::Server as ProtocolServer>::Channel,
    ) {
        log::info!("[Request Handler] On new connection.");
        let mut session_id = None;

        loop {
            let packet = socket.recv().await;
            let msg = match packet {
                Ok(Some(msg)) => msg,
                Ok(None) => {
                    log::info!("Request stream closed");
                    break;
                }
                Err(e) => {
                    log::info!("Request stream error: {e:?}, Closing.");
                    break;
                }
            };

            let task = match rmp_serde::from_slice::<Task>(&msg.data) {
                Ok(val) => val,
                Err(err) => {
                    log::info!("Only bytes message in the json format are supported {err:?}");
                    break;
                }
            };

            if let Task::Close(id) = task {
                if session_id != Some(id) {
                    log::warn!("Rejecting close of session {id} from connection of {session_id:?}");
                }
                break;
            }

            let (stream, connection_id, task) =
                match session_manager.stream(&mut session_id, task).await {
                    Ok(Some(val)) => val,
                    Ok(None) => {
                        log::info!("Ops session activated {session_id:?}");
                        continue;
                    }
                    Err(err) => {
                        log::warn!("Invalid request for session {session_id:?}: {err:?}");
                        break;
                    }
                };

            let result = match task {
                ComputeTask::RegisterOperation(op) => stream.register_operation(op).await,
                ComputeTask::RegisterTensor(id, data) => stream.register_tensor(id, data).await,
                ComputeTask::ReadTensor(tensor) => stream.read_tensor(connection_id, tensor).await,
                ComputeTask::SyncBackend => stream.sync(connection_id).await,
                ComputeTask::RegisterTensorRemote(tensor, new_id) => {
                    stream.register_tensor_remote(tensor, new_id).await
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
                ComputeTask::DTypeUsage(dtype) => stream.dtype_usage(connection_id, dtype).await,
                ComputeTask::RegisterGraph(graph) => stream.register_graph(connection_id, graph).await,
                ComputeTask::ExecuteGraph(graph, bindings) => stream.execute_graph(connection_id, graph, bindings).await,
                ComputeTask::RemoveGraph(graph) => stream.remove_graph(connection_id, graph).await,
            };
            if result.is_err() {
                log::warn!("Stream of session {session_id:?} stopped, closing.");
                break;
            }
        }

        log::info!("Closing session {session_id:?}");
        session_manager.close(session_id).await;
    }
}

/// Start the server on the given port and [device](Device).
pub async fn start_websocket_async<B: BackendIr>(device: Device<B>, port: u16) {
    let server = WsServer::new(port);
    RemoteServer::<B, WebSocket>::start(device, server).await;
}

#[tokio::main]
/// Start the server on the given port and [device](Device).
pub async fn start_websocket<B: BackendIr>(device: Device<B>, port: u16) {
    start_websocket_async::<B>(device, port).await;
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
