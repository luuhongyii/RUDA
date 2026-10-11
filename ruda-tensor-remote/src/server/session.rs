use ruda_tensor::tensor::Device;
use ruda_communication::{Protocol, data_service::TensorDataService};
use ruda_tensor::graph::BackendIr;
use ruda_tensor_router::Runner;
use ruda_core::id::StreamId;
use std::{collections::HashMap, sync::Arc};
use tokio::sync::{
    Mutex,
    mpsc::{Receiver, Sender},
};

use crate::shared::{ComputeTask, ConnectionId, SessionId, Task, TaskResponse};

use super::stream::Stream;

/// A request violated the session protocol; the connection that sent it is closed.
#[derive(Debug)]
pub enum SessionError {
    /// The first message on a request connection was not [`Task::Init`].
    NotInitialized,
    /// A non-compute task arrived after initialization.
    UnexpectedTask,
    /// The session was closed or never registered.
    UnknownSession(SessionId),
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
    runner: Runner<B>,
    sessions: Mutex<HashMap<SessionId, Session<B, P>>>,
    data_service: Arc<TensorDataService<B, P>>,
}

struct Session<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    runner: Runner<B>,
    streams: HashMap<StreamId, Stream<B, P>>,
    sender: Sender<Receiver<TaskResponse>>,
    receiver: Option<Receiver<Receiver<TaskResponse>>>,
    data_service: Arc<TensorDataService<B, P>>,
}

impl<B, P> SessionManager<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    pub fn new(device: Device<B>, data_service: Arc<TensorDataService<B, P>>) -> Self {
        Self {
            runner: Runner::new(device),
            sessions: Mutex::new(Default::default()),
            data_service,
        }
    }

    /// Register a new responder for the session. Only one responder can exist for a session for
    /// now; `None` is returned if the session already has one.
    pub async fn register_responder(
        &self,
        session_id: SessionId,
    ) -> Option<Receiver<Receiver<TaskResponse>>> {
        log::info!("Register responder for session {session_id}");
        let mut sessions = self.sessions.lock().await;
        self.register_session(&mut sessions, session_id)
            .init_responder()
    }

    /// Get the stream for the current session and task.
    pub async fn stream(
        &self,
        session_id: &mut Option<SessionId>,
        task: Task,
    ) -> Result<Option<(Stream<B, P>, ConnectionId, ComputeTask)>, SessionError> {
        let mut sessions = self.sessions.lock().await;

        let session_id = match session_id {
            Some(id) => *id,
            None => match task {
                Task::Init(id) => {
                    log::info!("Init requester for session {id}");
                    *session_id = Some(id);
                    self.register_session(&mut sessions, id);
                    return Ok(None);
                }
                _ => return Err(SessionError::NotInitialized),
            },
        };

        let session = sessions
            .get_mut(&session_id)
            .ok_or(SessionError::UnknownSession(session_id))?;
        let (task, connection_id) = match task {
            Task::Compute(task, connection_id) => (task, connection_id),
            _ => return Err(SessionError::UnexpectedTask),
        };
        let stream = session.select(connection_id.stream_id).await;
        Ok(Some((stream, connection_id, task)))
    }

    /// Close the session with the given id.
    ///
    /// The session is removed before its streams are drained so that slow cleanup never blocks
    /// other sessions.
    pub async fn close(&self, session_id: Option<SessionId>) {
        let Some(id) = session_id else {
            return;
        };
        let session = self.sessions.lock().await.remove(&id);
        if let Some(mut session) = session {
            session.close().await;
        }
    }

    /// Number of sessions currently registered.
    pub async fn session_count(&self) -> usize {
        self.sessions.lock().await.len()
    }

    fn register_session<'a>(
        &self,
        sessions: &'a mut HashMap<SessionId, Session<B, P>>,
        id: SessionId,
    ) -> &'a mut Session<B, P> {
        sessions.entry(id).or_insert_with(|| {
            log::info!("Creating a new session {id}");

            Session::new(self.runner.fork(), self.data_service.clone())
        })
    }
}

impl<B, P> Session<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    fn new(runner: Runner<B>, data_service: Arc<TensorDataService<B, P>>) -> Self {
        let (sender, receiver) = tokio::sync::mpsc::channel(1);

        Self {
            runner,
            streams: Default::default(),
            sender,
            receiver: Some(receiver),
            data_service,
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
                let stream = Stream::<B, P>::new(
                    self.runner.clone(),
                    self.sender.clone(),
                    self.data_service.clone(),
                )
                .await;
                self.streams.insert(stream_id, stream.clone());
                stream
            }
        }
    }

    // Close all streams created in the session.
    async fn close(&mut self) {
        for (id, stream) in self.streams.drain() {
            log::info!("Closing stream {id}");
            stream.close().await;
        }
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
}
