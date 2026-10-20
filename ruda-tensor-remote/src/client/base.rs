pub use super::RemoteDevice;
use super::worker::{ClientRequest, ClientWorker};
use crate::shared::{ComputeTask, ConnectionId, LocalTransferId, SessionId, Task, TaskResponseContent};
use async_channel::{RecvError, SendError, Sender};
use ruda_communication::ProtocolClient;
use ruda_tensor::graph::TensorId;
use ruda_tensor::ProfileToken;
use ruda_core::id::StreamId;
use std::{
    future::Future,
    collections::HashMap,
    sync::{Arc, Mutex, atomic::AtomicU64},
};

pub(crate) struct RemoteRuntime {
    #[cfg(not(target_family = "wasm"))]
    inner: Option<tokio::runtime::Runtime>,
}
impl RemoteRuntime {
    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn new(runtime: tokio::runtime::Runtime) -> Self { Self { inner: Some(runtime) } }
    pub(crate) fn session() -> Self {
        #[cfg(not(target_family = "wasm"))]
        { Self::new(tokio::runtime::Builder::new_multi_thread().enable_io().enable_time().build().unwrap()) }
        #[cfg(target_family = "wasm")]
        { Self {} }
    }
    pub(crate) fn block_on<F: Future>(&self, future: F) -> F::Output {
        ruda_core::future::block_on(future)
    }
    #[cfg(target_family = "wasm")]
    pub(crate) fn spawn(&self, future: impl Future<Output = ()> + 'static) {
        wasm_bindgen_futures::spawn_local(future);
    }
}
#[cfg(not(target_family = "wasm"))]
impl core::ops::Deref for RemoteRuntime {
    type Target = tokio::runtime::Runtime;
    fn deref(&self) -> &Self::Target { self.inner.as_ref().expect("live remote runtime") }
}
impl Drop for RemoteRuntime {
    fn drop(&mut self) {
        #[cfg(not(target_family = "wasm"))]
        if let Some(runtime) = self.inner.take() { runtime.shutdown_background(); }
    }
}

#[derive(Default)]
pub(crate) struct ProfileWindows {
    pub(crate) counter: u64,
    pub(crate) open: HashMap<u64, (StreamId, ProfileToken)>,
}

#[derive(Clone)]
pub struct RemoteClient {
    pub(crate) device: RemoteDevice,
    pub(crate) sender: Arc<RemoteSender>,
    pub(crate) runtime: Arc<RemoteRuntime>,
    pub(crate) profiles: Arc<Mutex<ProfileWindows>>,
    pub(crate) info: Arc<std::sync::OnceLock<crate::RemoteDeviceInfo>>,
    pub(crate) connection: Arc<super::connection::Connection>,
    pub(crate) probe: crate::telemetry::TelemetryProbe,
}

impl RemoteClient {
    pub fn init<C: ProtocolClient>(device: RemoteDevice) -> Self {
        ClientWorker::<C>::start(device)
    }

    pub(crate) fn new(
        device: RemoteDevice,
        sender: Sender<ClientRequest>,
        runtime: Arc<RemoteRuntime>,
        session_id: SessionId,
    ) -> Self {
        Self {
            device,
            runtime,
            profiles: Arc::new(Mutex::new(ProfileWindows::default())),
            info: Arc::new(std::sync::OnceLock::new()),
            connection: Arc::new(super::connection::Connection::new()),
            probe: crate::telemetry::TelemetryProbe::new(crate::telemetry::CHANNEL_CAPACITY),
            sender: Arc::new(RemoteSender {
                sender,
                position_counter: AtomicU64::new(0),
                tensor_id_counter: AtomicU64::new(0),
                graph_id_counter: AtomicU64::new(0),
                local_transfer_counter: AtomicU64::new(0),
                session_id,
            }),
        }
    }

    pub fn telemetry(&self) -> crate::telemetry::TelemetryProbe { self.probe.clone() }
}

pub(crate) struct RemoteSender {
    sender: Sender<ClientRequest>,
    position_counter: AtomicU64,
    tensor_id_counter: AtomicU64,
    graph_id_counter: AtomicU64,
    local_transfer_counter: AtomicU64,
    session_id: SessionId,
}

#[allow(unused)]
#[derive(Debug)]
pub enum RemoteSendError {
    SendError(SendError<ClientRequest>),
    RecvError(RecvError),
}

impl RemoteSender {
    fn enqueue(&self, request: ClientRequest) -> Result<(), SendError<ClientRequest>> {
        #[cfg(not(target_family = "wasm"))]
        { self.sender.send_blocking(request) }
        #[cfg(target_family = "wasm")]
        { self.sender.try_send(request).map_err(|error| SendError(error.into_inner())) }
    }

    pub(crate) fn flush_queue(&self) -> Result<(), SendError<ClientRequest>> { self.enqueue(ClientRequest::Flush) }
    pub(crate) fn new_graph_id(&self) -> Option<ruda_tensor::graph::GraphId> {
        self.graph_id_counter.fetch_update(std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed, |value| value.checked_add(1))
            .ok().map(ruda_tensor::graph::GraphId)
    }
    pub(crate) fn new_local_transfer(&self, target: &Self) -> LocalTransferId {
        LocalTransferId {
            source: self.session_id,
            target: target.session_id,
            sequence: self.local_transfer_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// Generate a new unique (for this [`RemoteSender`] [`TensorId`].
    pub(crate) fn new_tensor_id(&self) -> TensorId {
        TensorId::new(
            self.tensor_id_counter
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// Give the next operation sequence number.
    fn next_position(&self) -> u64 {
        self.position_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn send(&self, task: ComputeTask) {
        self.send_on(task, StreamId::current()).unwrap();
    }

    pub(crate) fn send_on(&self, task: ComputeTask, stream_id: StreamId)
        -> Result<(), SendError<ClientRequest>> {
        self.enqueue(ClientRequest::WithoutCallback(Task::Compute(
            task, ConnectionId::new(self.next_position(), stream_id),
        )))
    }

    pub(crate) fn send_async(
        &self,
        task: ComputeTask,
    ) -> impl Future<Output = Result<TaskResponseContent, RemoteSendError>> + Send + use<> {
        self.send_async_on(task, StreamId::current())
    }

    pub(crate) fn send_async_on(&self, task: ComputeTask, stream_id: StreamId)
        -> impl Future<Output = Result<TaskResponseContent, RemoteSendError>> + Send + use<> {
        let position = self.next_position();
        let (tx, rx) = async_channel::bounded(1);
        let sent = self.enqueue(ClientRequest::WithSyncCallback(
            Task::Compute(task, ConnectionId::new(position, stream_id)), tx,
        ));

        async move {
            sent.map_err(RemoteSendError::SendError)?;
            match rx.recv().await {
                Ok(response) => Ok(response),
                Err(e) => Err(RemoteSendError::RecvError(e)),
            }
        }
    }

    pub(crate) fn close(&mut self) {
        let close_task = ClientRequest::WithoutCallback(Task::Close(self.session_id));
        let _ = self.enqueue(close_task);
    }
}

impl Drop for RemoteSender {
    fn drop(&mut self) {
        self.close();
    }
}
