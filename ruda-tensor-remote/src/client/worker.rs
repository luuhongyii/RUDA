use super::{RemoteClient, runner::RemoteDevice};
use crate::shared::{ConnectionId, SessionId, Task, TaskResponse, TaskResponseContent};
use ruda_communication::{Address, CommunicationChannel, Message, ProtocolClient};
use std::{collections::HashMap, marker::PhantomData, sync::Arc};

pub type CallbackSender = async_channel::Sender<TaskResponseContent>;

#[derive(Debug)]
pub enum ClientRequest {
    WithSyncCallback(Task, CallbackSender),
    WithoutCallback(Task),
}

struct PendingRequests(async_channel::Receiver<ClientRequest>);

impl Drop for PendingRequests {
    fn drop(&mut self) {
        self.0.close();
        // Closing a channel alone leaves queued callbacks alive while senders still exist.
        while self.0.try_recv().is_ok() {}
    }
}

pub(crate) struct ClientWorker<C: ProtocolClient> {
    requests: HashMap<ConnectionId, CallbackSender>,
    _p: PhantomData<C>,
}

impl<C: ProtocolClient> ClientWorker<C> {
    async fn on_response(&mut self, response: TaskResponse) {
        match self.requests.remove(&response.id) {
            Some(request) => {
                let _ = request.send(response.content).await;
            }
            None => {
                panic!("Can't ignore message from the server.");
            }
        }
    }

    fn register_callback(&mut self, id: ConnectionId, callback: CallbackSender) {
        self.requests.insert(id, callback);
    }
}

impl<C: ProtocolClient> ClientWorker<C> {
    pub fn start(device: RemoteDevice) -> RemoteClient {
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_io()
                .build()
                .unwrap(),
        );

        let (sender, rec) = async_channel::bounded(10);

        let session_id = SessionId::new();
        let address = device.address.clone();

        runtime.spawn(Self::run(address, session_id, PendingRequests(rec)));

        RemoteClient::new(device, sender, runtime, session_id)
    }

    #[allow(deprecated)]
    async fn run(address: Address, session_id: SessionId, pending: PendingRequests) {
        log::info!("Connecting to {} ...", address.clone());
        let Some(mut stream_request) = C::connect(address.clone(), "request").await else {
            log::warn!("Cannot connect to request stream at {address}");
            return;
        };
        let Some(mut stream_response) = C::connect(address.clone(), "response").await else {
            log::warn!("Cannot connect to response stream at {address}");
            return;
        };

        let state = Arc::new(tokio::sync::Mutex::new(ClientWorker::<C>::default()));

        // Init the connection.
        let bytes: bytes::Bytes = rmp_serde::to_vec(&Task::Init(session_id))
            .expect("Can serialize tasks to bytes.")
            .into();
        stream_request
            .send(Message::new(bytes.clone()))
            .await
            .expect("Can send the message over the comms channel.");
        stream_response
            .send(Message::new(bytes))
            .await
            .expect("Can send the message on the websocket.");

        // Async worker loading callbacks from the server.
        let state_ws = state.clone();
        let responses = async move {
            while let Ok(msg) = stream_response.recv().await {
                let msg = match msg {
                    Some(msg) => msg,
                    None => {
                        log::warn!("Closed connection");
                        return;
                    }
                };

                let response: TaskResponse = rmp_serde::from_slice(&msg.data)
                    .expect("Can deserialize messages from the websocket.");
                let mut state = state_ws.lock().await;
                state.on_response(response).await;
            }
        };

        // Channel async worker sending operations to the server.
        let requests = async {
            while let Ok(req) = pending.0.recv().await {
                let task = match req {
                    ClientRequest::WithSyncCallback(task, callback) => {
                        if let Task::Compute(_content, id) = &task {
                            let mut state = state.lock().await;
                            state.register_callback(*id, callback);
                        }
                        task
                    }
                    ClientRequest::WithoutCallback(task) => task,
                };
                let bytes = rmp_serde::to_vec(&task)
                    .expect("Can serialize tasks to bytes.")
                    .into();
                stream_request
                    .send(Message::new(bytes))
                    .await
                    .expect("Can send the message on the websocket.");
            }
        };
        // Either stream ending must release both registered and still-queued callbacks.
        tokio::select! {
            _ = responses => {},
            _ = requests => {},
        }
    }
}

impl<C: ProtocolClient> Default for ClientWorker<C> {
    fn default() -> Self {
        Self {
            requests: Default::default(),
            _p: PhantomData,
        }
    }
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
