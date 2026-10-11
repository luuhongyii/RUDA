use ruda_communication::{
    CommunicationChannel, Message, Protocol, ProtocolServer,
    data_service::{TensorDataServer, TensorDataService},
    util::os_shutdown_signal,
    websocket::{WebSocket, WsServer},
};
use std::{marker::PhantomData, net::SocketAddr, sync::Arc};
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

        while let Some(mut callback) = receiver.recv().await {
            let Some(response) = callback.recv().await else {
                log::info!("Processor dropped a pending response for session {id}");
                break;
            };
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
        // The responder going away leaves the request side unable to deliver results.
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

            // A connection may only close the session it initialized.
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
