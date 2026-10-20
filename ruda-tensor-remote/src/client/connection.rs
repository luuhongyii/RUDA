use std::sync::Mutex;
use tokio::sync::Notify;
use super::ConnectError;
use crate::RemoteDeviceInfo;

enum State { Connecting, Ready(RemoteDeviceInfo), Failed(ConnectError) }

pub(crate) struct Connection {
    state: Mutex<State>,
    changed: Notify,
}

impl Connection {
    pub(crate) fn new() -> Self { Self { state: Mutex::new(State::Connecting), changed: Notify::new() } }
    pub(crate) fn ready(&self, info: RemoteDeviceInfo) {
        let mut state = self.state.lock().unwrap();
        if matches!(*state, State::Connecting) { *state = State::Ready(info); }
        drop(state);
        self.changed.notify_waiters();
    }
    pub(crate) fn fail(&self, error: ConnectError) {
        let mut state = self.state.lock().unwrap();
        if !matches!(*state, State::Failed(_)) { *state = State::Failed(error); }
        drop(state);
        self.changed.notify_waiters();
    }
    pub(crate) async fn wait(&self) -> Result<RemoteDeviceInfo, ConnectError> {
        loop {
            let notified = self.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            match &*self.state.lock().unwrap() {
                State::Ready(info) => return Ok(info.clone()),
                State::Failed(error) => return Err(error.clone()),
                State::Connecting => {},
            }
            notified.await;
        }
    }
}

pub(crate) struct ConnectionGuard(pub std::sync::Arc<Connection>);
impl Drop for ConnectionGuard {
    fn drop(&mut self) { self.0.fail(ConnectError::Interrupted); }
}
