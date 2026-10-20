use ruda_tensor::tensor::Device;
use ruda_communication::{Protocol, data_service::TensorDataService};
use ruda_tensor::graph::{BackendIr, GraphId, GraphIr};
use ruda_tensor_router::{Runner, RunnerClient};
use ruda_tensor::DeviceOps;
use ruda_core::id::StreamId;
use std::{collections::HashMap, sync::Arc};
use tokio::sync::{
    Mutex,
    mpsc::{Receiver, Sender},
};
use tokio_util::sync::CancellationToken;

use crate::shared::{ComputeTask, ConnectionId, SessionId, SessionRefusal, Task, TaskResponse};

use super::{AuthorizationRequest, ClientId, ServerSettings, local_comm::LocalCommService, stream::Stream};

/// A request violated the session protocol; the connection that sent it is closed.
#[derive(Debug)]
pub enum SessionError {
    /// The first message on a request connection was not [`Task::Init`].
    NotInitialized,
    /// A non-compute task arrived after initialization.
    UnexpectedTask,
    /// The session was closed or never registered.
    UnknownSession(SessionId),
    InvalidDeviceIndex(u32),
    DeviceMismatch(SessionId),
    InvalidCollectiveDevice(u16),
    AlreadyBound(SessionId),
    InvalidGraph(String),
    UnknownQueuedGraph(GraphId),
    DuplicateQueuedGraph(GraphId),
}

/// A session manager control the creation of sessions.
///
/// Each session manages its own stream, spawning one thread per stream to mimic the same behavior
/// a native backend would have. Sessions share the device but not the tensor or graph context,
/// since every client numbers its tensors from zero.
pub struct SessionManager<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    runners: Vec<Runner<B>>,
    sessions: Mutex<HashMap<SessionId, Session<B, P>>>,
    data_service: Arc<TensorDataService<B, P>>,
    local_comm: Arc<LocalCommService<B>>,
    settings: ServerSettings,
    shutdown: CancellationToken,
    closing: Arc<std::sync::Mutex<HashMap<SessionId, CancellationToken>>>,
}

struct ClosingSession {
    id: SessionId,
    completed: CancellationToken,
    closing: Arc<std::sync::Mutex<HashMap<SessionId, CancellationToken>>>,
}
impl Drop for ClosingSession {
    fn drop(&mut self) {
        self.closing.lock().unwrap().remove(&self.id);
        self.completed.cancel();
    }
}

struct Session<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    id: SessionId,
    runner: Runner<B>,
    info: crate::RemoteDeviceInfo,
    streams: HashMap<StreamId, Stream<B, P>>,
    queued_graphs: HashMap<GraphId, Arc<ruda_tensor_router::CachedGraph>>,
    sender: Sender<Receiver<TaskResponse>>,
    receiver: Option<Receiver<Receiver<TaskResponse>>>,
    data_service: Arc<TensorDataService<B, P>>,
    local_comm: Arc<LocalCommService<B>>,
    local_closed: bool,
    closed: CancellationToken,
    request_bound: bool,
    versioned: bool,
    peer: Option<ruda_communication::PeerId>,
    probe: crate::telemetry::TelemetryProbe,
}

impl<B, P> SessionManager<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    pub fn new(device: Device<B>, data_service: Arc<TensorDataService<B, P>>) -> Self {
        Self::with_runner(Runner::new(device), data_service)
    }

    /// Create sessions from a configured runner, retaining its custom operation handlers.
    pub fn with_runner(runner: Runner<B>, data_service: Arc<TensorDataService<B, P>>) -> Self {
        Self::with_runners(vec![runner], data_service)
    }

    /// Host an explicitly ordered set of configured devices.
    pub fn with_runners(runners: Vec<Runner<B>>, data_service: Arc<TensorDataService<B, P>>) -> Self {
        assert!(!runners.is_empty(), "A remote host needs at least one device");
        assert!(runners.len() <= u16::MAX as usize + 1, "Remote device catalog exceeds the collective index namespace");
        Self {
            runners,
            sessions: Mutex::new(Default::default()),
            data_service,
            local_comm: Arc::new(LocalCommService::new()),
            settings: ServerSettings::default(),
            shutdown: CancellationToken::new(),
            closing: Arc::new(std::sync::Mutex::new(HashMap::new())),
        }
    }

    pub(crate) fn with_settings(mut self, settings: ServerSettings, shutdown: CancellationToken) -> Self {
        self.settings = settings;
        self.shutdown = shutdown;
        self
    }

    pub(crate) fn shutdown_token(&self) -> CancellationToken { self.shutdown.clone() }
    pub(crate) fn device_count(&self) -> u32 { self.runners.len() as u32 }

    pub(crate) fn authorize(&self, client: ClientId, device_index: u32, credential: &crate::Credential)
        -> Result<(), SessionRefusal> {
        self.settings.authorizer.authorize(AuthorizationRequest { client, device_index, credential })
            .map_err(|reason| { log::warn!("Remote session refused: {reason}"); SessionRefusal::Unauthorized })
    }

    pub(crate) async fn bind_request(&self, id: SessionId, index: u32, versioned: bool)
        -> Result<(crate::RemoteDeviceInfo, CancellationToken), SessionError> {
        self.bind_request_from(id, index, versioned, None).await
    }

    pub(crate) async fn bind_request_from(&self, id: SessionId, index: u32, versioned: bool,
        peer: Option<ruda_communication::PeerId>) -> Result<(crate::RemoteDeviceInfo, CancellationToken), SessionError> {
        let mut sessions = self.sessions.lock().await;
        if let Some(session) = sessions.get(&id) {
            if session.request_bound || session.versioned != versioned || session.peer != peer { return Err(SessionError::AlreadyBound(id)); }
        }
        let session = self.register_session(&mut sessions, id, index)?;
        session.request_bound = true;
        session.versioned = versioned;
        session.peer = peer;
        Ok((session.info.clone(), session.closed.clone()))
    }

    pub(crate) async fn bind_response(&self, id: SessionId, index: u32, versioned: bool)
        -> Result<(Receiver<Receiver<TaskResponse>>, crate::RemoteDeviceInfo, CancellationToken), SessionError> {
        self.bind_response_from(id, index, versioned, None).await
    }

    pub(crate) async fn bind_response_from(&self, id: SessionId, index: u32, versioned: bool,
        peer: Option<ruda_communication::PeerId>) -> Result<(Receiver<Receiver<TaskResponse>>, crate::RemoteDeviceInfo, CancellationToken), SessionError> {
        let mut sessions = self.sessions.lock().await;
        let session = if versioned {
            let session = sessions.get_mut(&id).ok_or(SessionError::UnknownSession(id))?;
            if !session.versioned || !session.request_bound || session.peer != peer { return Err(SessionError::AlreadyBound(id)); }
            if session.info.device_index != index { return Err(SessionError::DeviceMismatch(id)); }
            session
        } else {
            if sessions.get(&id).is_some_and(|session| session.peer != peer) { return Err(SessionError::AlreadyBound(id)); }
            let session = self.register_session(&mut sessions, id, index)?;
            if session.versioned { return Err(SessionError::AlreadyBound(id)); }
            session.peer = peer;
            session
        };
        let receiver = session.init_responder().ok_or(SessionError::AlreadyBound(id))?;
        Ok((receiver, session.info.clone(), session.closed.clone()))
    }

    /// Register a new responder for the session. Only one responder can exist for a session for
    /// now; `None` is returned if the session already has one.
    pub async fn register_responder(
        &self,
        session_id: SessionId,
    ) -> Option<Receiver<Receiver<TaskResponse>>> {
        self.register_responder_on(session_id, 0).await.ok().flatten()
    }

    pub async fn register_responder_on(&self, session_id: SessionId, device_index: u32)
        -> Result<Option<Receiver<Receiver<TaskResponse>>>, SessionError> {
        log::info!("Register responder for session {session_id}");
        match self.bind_response(session_id, device_index, false).await {
            Ok((receiver, _, _)) => Ok(Some(receiver)),
            Err(SessionError::AlreadyBound(_)) => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// Get the stream for the current session and task.
    pub async fn stream(
        &self,
        session_id: &mut Option<SessionId>,
        task: Task,
    ) -> Result<Option<(Stream<B, P>, ConnectionId, ComputeTask)>, SessionError> {
        let session_id = match session_id {
            Some(id) => *id,
            None => match task {
                Task::Init(id) => {
                    log::info!("Init requester for session {id}");
                    self.bind_request(id, 0, false).await?;
                    *session_id = Some(id);
                    return Ok(None);
                }
                Task::InitDevice(id, index) => {
                    self.bind_request(id, index, false).await?;
                    *session_id = Some(id);
                    return Ok(None);
                }
                _ => return Err(SessionError::NotInitialized),
            },
        };

        let mut sessions = self.sessions.lock().await;
        let session = sessions
            .get_mut(&session_id)
            .ok_or(SessionError::UnknownSession(session_id))?;
        let (mut task, connection_id) = match task {
            Task::Compute(task, connection_id) => (task, connection_id),
            _ => return Err(SessionError::UnexpectedTask),
        };
        #[cfg(feature = "distributed")]
        match &mut task {
            ComputeTask::RegisterOperation(operation) => self.resolve_collectives(operation)?,
            ComputeTask::RegisterGraph(graph) | ComputeTask::QueueGraph { graph, .. } => {
                for operation in &mut graph.operations { self.resolve_collectives(operation)?; }
            }
            _ => {}
        }
        let stream = session.select(connection_id.stream_id).await;
        Ok(Some((stream, connection_id, task)))
    }

    pub(crate) async fn queued_graph(&self, session: SessionId, id: GraphId, graph: Option<GraphIr>)
        -> Result<Arc<ruda_tensor_router::CachedGraph>, SessionError> {
        let mut sessions = self.sessions.lock().await;
        let session = sessions.get_mut(&session).ok_or(SessionError::UnknownSession(session))?;
        if let Some(graph) = graph {
            if session.queued_graphs.contains_key(&id) { return Err(SessionError::DuplicateQueuedGraph(id)); }
            let observed = session.probe.is_active().then(|| (
                graph.operations.iter().map(crate::telemetry::GraphOp::summarize).collect(),
                crate::telemetry::serialized_len(&graph.operations),
            ));
            let graph = Arc::new(ruda_tensor_router::CachedGraph::new(graph)
                .map_err(|error| SessionError::InvalidGraph(format!("{error:?}")))?);
            session.queued_graphs.insert(id, graph.clone());
            if let Some((ops, bytes)) = observed {
                session.probe.emit(|| crate::telemetry::TelemetryEvent::GraphRegistered {
                    session: session.id, graph: id, namespace: crate::telemetry::GraphNamespace::Queued, ops, bytes,
                });
            }
            Ok(graph)
        } else {
            session.queued_graphs.get(&id).cloned().ok_or(SessionError::UnknownQueuedGraph(id))
        }
    }

    pub(crate) async fn release_queued_graph(&self, session: SessionId, id: GraphId) -> Result<(), SessionError> {
        let mut sessions = self.sessions.lock().await;
        let session = sessions.get_mut(&session).ok_or(SessionError::UnknownSession(session))?;
        session.queued_graphs.remove(&id).ok_or(SessionError::UnknownQueuedGraph(id))?;
        Ok(())
    }

    /// Close the session with the given id.
    ///
    /// The session is removed before its streams are drained so that slow cleanup never blocks
    /// other sessions.
    pub async fn close(&self, session_id: Option<SessionId>) {
        let Some(id) = session_id else {
            return;
        };
        let (completed, session) = {
            let mut sessions = self.sessions.lock().await;
            let mut closing = self.closing.lock().unwrap();
            if let Some(completed) = closing.get(&id) {
                (completed.clone(), None)
            } else if let Some(mut session) = sessions.remove(&id) {
                session.close_local_transfers();
                let completed = CancellationToken::new();
                closing.insert(id, completed.clone());
                (completed, Some(session))
            } else { return; }
        };
        if let Some(mut session) = session {
            let completion = ClosingSession { id, completed: completed.clone(), closing: self.closing.clone() };
            tokio::spawn(async move {
                let _completion = completion;
                session.close().await;
            });
        }
        completed.cancelled().await;
    }

    pub(crate) async fn close_all(&self) {
        self.shutdown.cancel();
        let ids = {
            let mut active = self.sessions.lock().await;
            for session in active.values_mut() { session.close_local_transfers(); }
            active.keys().copied().collect::<Vec<_>>()
        };
        futures_util::future::join_all(ids.into_iter().map(|id| self.close(Some(id)))).await;
        let closing: Vec<_> = self.closing.lock().unwrap().values().cloned().collect();
        futures_util::future::join_all(closing.into_iter().map(|completed| async move {
            completed.cancelled().await;
        })).await;
    }

    /// Number of sessions currently registered.
    pub async fn session_count(&self) -> usize {
        self.sessions.lock().await.len()
    }

    pub async fn device_info(&self, session: SessionId) -> Option<crate::RemoteDeviceInfo> {
        self.sessions.lock().await.get(&session).map(|session| session.info.clone())
    }

    #[cfg(feature = "distributed")]
    fn resolve_collectives(&self, operation: &mut ruda_tensor::graph::OperationIr) -> Result<(), SessionError> {
        use ruda_tensor::graph::{OperationIr, DistributedOperationIr};
        if let OperationIr::Distributed(DistributedOperationIr::AllReduce(desc)) = operation {
            for (kind, index) in &mut desc.device_ids {
                if *kind != 0 { return Err(SessionError::InvalidCollectiveDevice(*index)); }
                let runner = self.runners.get(*index as usize).ok_or(SessionError::InvalidCollectiveDevice(*index))?;
                let native = runner.device().id();
                *kind = native.type_id;
                *index = native.index_id;
            }
        }
        Ok(())
    }

    fn register_session<'a>(
        &self,
        sessions: &'a mut HashMap<SessionId, Session<B, P>>,
        id: SessionId,
        device_index: u32,
    ) -> Result<&'a mut Session<B, P>, SessionError> {
        if self.shutdown.is_cancelled() { return Err(SessionError::UnknownSession(id)); }
        if self.closing.lock().unwrap().contains_key(&id) { return Err(SessionError::AlreadyBound(id)); }
        let runner = self.runners.get(device_index as usize).ok_or(SessionError::InvalidDeviceIndex(device_index))?;
        let session = sessions.entry(id).or_insert_with(|| {
            log::info!("Creating a new session {id}");

            let device = runner.device();
            let settings = ruda_tensor::get_device_settings::<B>(&device);
            let info = crate::RemoteDeviceInfo {
                device_index, device_count: self.runners.len() as u32, name: B::name(&device),
                float_dtype: settings.float_dtype, int_dtype: settings.int_dtype, bool_dtype: settings.bool_dtype,
                native_collectives: runner.has_native_collectives(),
            };
            self.local_comm.register_session(id);
            self.settings.telemetry.emit(|| crate::telemetry::TelemetryEvent::SessionOpened { session: id, device: device_index });
            Session::new(id, runner.fork(), info, self.data_service.clone(), self.local_comm.clone(), self.settings.telemetry.clone())
        });
        if session.info.device_index != device_index { return Err(SessionError::DeviceMismatch(id)); }
        Ok(session)
    }
}

impl<B, P> Session<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    fn new(id: SessionId, runner: Runner<B>, info: crate::RemoteDeviceInfo,
        data_service: Arc<TensorDataService<B, P>>, local_comm: Arc<LocalCommService<B>>, probe: crate::telemetry::TelemetryProbe) -> Self {
        let (sender, receiver) = tokio::sync::mpsc::channel(1);

        Self {
            id,
            runner,
            info,
            streams: Default::default(),
            queued_graphs: HashMap::new(),
            sender,
            receiver: Some(receiver),
            data_service,
            local_comm,
            local_closed: false,
            closed: CancellationToken::new(),
            request_bound: false,
            versioned: false,
            peer: None,
            probe,
        }
    }

    fn init_responder(&mut self) -> Option<Receiver<Receiver<TaskResponse>>> {
        self.receiver.take()
    }

    /// Select the current [stream](Stream) based on the given task.
    async fn select(&mut self, stream_id: StreamId) -> Stream<B, P> {
        // We return the stream.
        match self.streams.get(&stream_id) {
            Some(stream) => stream.clone(),
            None => {
                let stream = Stream::<B, P>::new_with_local_comm(
                    self.runner.clone(),
                    self.sender.clone(),
                    self.data_service.clone(),
                    self.id,
                    self.local_comm.clone(),
                    self.closed.clone(),
                    super::processor::WorkerProbe { session: self.id, stream: stream_id, probe: self.probe.clone() },
                )
                .await;
                self.streams.insert(stream_id, stream.clone());
                stream
            }
        }
    }

    fn close_local_transfers(&mut self) {
        if !self.local_closed {
            self.local_closed = true;
            self.closed.cancel();
            self.local_comm.close_session(self.id);
        }
    }

    // Close all streams created in the session.
    async fn close(&mut self) {
        self.close_local_transfers();
        for (id, stream) in self.streams.drain() {
            log::info!("Closing stream {id}");
            stream.close().await;
        }
    }
}

impl<B: BackendIr, P: Protocol> Drop for Session<B, P> {
    fn drop(&mut self) {
        self.close_local_transfers();
        self.probe.emit(|| crate::telemetry::TelemetryEvent::SessionClosed { session: self.id });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruda_communication::websocket::WebSocket;
    use ruda_tensor_host::Host;
    use tokio_util::sync::CancellationToken;

    fn manager() -> SessionManager<Host, WebSocket> {
        let data_service = Arc::new(TensorDataService::new(CancellationToken::new()));
        SessionManager::new(Default::default(), data_service)
    }

    fn seed_task() -> Task {
        Task::Compute(ComputeTask::Seed(0), ConnectionId::new(0, StreamId::current()))
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn closed_sessions_are_removed() {
        let manager = manager();
        for _ in 0..16 {
            let id = SessionId::new();
            let mut current = None;
            assert!(manager.stream(&mut current, Task::Init(id)).await.unwrap().is_none());
            let _responder = manager.register_responder(id).await.unwrap();
            let (stream, _, _) = manager.stream(&mut current, seed_task()).await.unwrap().unwrap();
            stream.seed(1).await.unwrap();
            manager.close(current).await;
        }
        assert_eq!(manager.session_count().await, 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn closing_one_session_keeps_others_usable() {
        let manager = manager();
        let (mut first, mut second) = (None, None);
        manager.stream(&mut first, Task::Init(SessionId::new())).await.unwrap();
        manager.stream(&mut second, Task::Init(SessionId::new())).await.unwrap();

        manager.close(first).await;
        manager.close(first).await;

        let (stream, _, _) = manager.stream(&mut second, seed_task()).await.unwrap().unwrap();
        stream.seed(1).await.unwrap();
        assert!(matches!(
            manager.stream(&mut first, seed_task()).await,
            Err(SessionError::UnknownSession(_))
        ));
        assert_eq!(manager.session_count().await, 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn requests_before_init_are_rejected() {
        let manager = manager();
        let mut current = None;
        assert!(matches!(
            manager.stream(&mut current, seed_task()).await,
            Err(SessionError::NotInitialized)
        ));
        assert_eq!(manager.session_count().await, 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn second_responder_is_refused() {
        let manager = manager();
        let id = SessionId::new();
        assert!(manager.register_responder(id).await.is_some());
        assert!(manager.register_responder(id).await.is_none());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn sessions_with_equal_tensor_ids_keep_separate_values() {
        use ruda_tensor::{DType, Shape, TensorData};
        use ruda_tensor::graph::{TensorId, TensorIr, TensorStatus};
        use ruda_tensor_router::RunnerClient;

        let manager = manager();
        let (a, b) = (SessionId::new(), SessionId::new());
        let (mut first, mut second) = (None, None);
        manager.stream(&mut first, Task::Init(a)).await.unwrap();
        manager.stream(&mut second, Task::Init(b)).await.unwrap();
        let (runner_a, runner_b) = {
            let sessions = manager.sessions.lock().await;
            (sessions[&a].runner.clone(), sessions[&b].runner.clone())
        };
        let id = TensorId::new(0);
        runner_a.register_tensor_data_id(id, TensorData::new(vec![1.5f32], [1]));
        runner_b.register_tensor_data_id(id, TensorData::new(vec![9.0f32], [1]));
        let tensor = TensorIr { id, shape: Shape::new([1]), dtype: DType::F32,
            status: TensorStatus::ReadOnly };
        assert_eq!(runner_a.read_tensor_async(tensor.clone()).await.unwrap()
            .to_vec::<f32>().unwrap(), vec![1.5]);
        manager.close(first).await;
        assert_eq!(runner_b.read_tensor_async(tensor).await.unwrap()
            .to_vec::<f32>().unwrap(), vec![9.0]);
        manager.close(second).await;
    }
}
