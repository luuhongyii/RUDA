use crate::shared::{LocalTransferId, SessionId};
use ruda_tensor::{ExecutionError, graph::{BackendIr, HandleKind}};
use std::{collections::{HashMap, HashSet, hash_map::Entry}, sync::Mutex};
use tokio::sync::oneshot;

enum Transfer<B: BackendIr> {
    Exposed(HandleKind<B>),
    Waiting(oneshot::Sender<HandleKind<B>>),
}

struct State<B: BackendIr> {
    sessions: HashSet<SessionId>,
    transfers: HashMap<LocalTransferId, Transfer<B>>,
}

/// Owns native primitives only until the destination takes them or an endpoint closes.
pub(crate) struct LocalCommService<B: BackendIr> {
    state: Mutex<State<B>>,
}

fn transfer_error(reason: &str) -> ExecutionError {
    ExecutionError::WithContext { reason: reason.into() }
}

impl<B: BackendIr> LocalCommService<B> {
    pub(crate) fn new() -> Self {
        Self { state: Mutex::new(State { sessions: HashSet::new(), transfers: HashMap::new() }) }
    }

    pub(crate) fn register_session(&self, session: SessionId) {
        self.state.lock().unwrap().sessions.insert(session);
    }

    pub(crate) fn close_session(&self, session: SessionId) {
        let removed = {
            let mut state = self.state.lock().unwrap();
            state.sessions.remove(&session);
            let ids: Vec<_> = state.transfers.keys().copied()
                .filter(|id| id.source == session || id.target == session).collect();
            ids.into_iter().filter_map(|id| state.transfers.remove(&id)).collect::<Vec<_>>()
        };
        // Dropping senders wakes waiting workers; dropping primitives releases device storage.
        drop(removed);
    }

    pub(crate) fn expose(&self, source: SessionId, id: LocalTransferId, tensor: HandleKind<B>)
        -> Result<(), ExecutionError> {
        let mut state = self.state.lock().unwrap();
        if source != id.source || !state.sessions.contains(&id.source) || !state.sessions.contains(&id.target) {
            return Err(transfer_error("Local transfer endpoint is not an active session"));
        }
        match state.transfers.entry(id) {
            Entry::Vacant(entry) => { entry.insert(Transfer::Exposed(tensor)); Ok(()) }
            Entry::Occupied(entry) => {
                if matches!(entry.get(), Transfer::Exposed(_)) {
                    return Err(transfer_error("Local transfer was exposed more than once"));
                }
                let Transfer::Waiting(sender) = entry.remove() else { unreachable!() };
                sender.send(tensor).map_err(|_| transfer_error("Local transfer destination stopped waiting"))
            }
        }
    }

    pub(crate) async fn take(&self, target: SessionId, id: LocalTransferId)
        -> Result<HandleKind<B>, ExecutionError> {
        let receiver = {
            let mut state = self.state.lock().unwrap();
            if target != id.target || !state.sessions.contains(&id.source) || !state.sessions.contains(&id.target) {
                return Err(transfer_error("Local transfer endpoint is not an active session"));
            }
            match state.transfers.entry(id) {
                Entry::Occupied(entry) => {
                    if matches!(entry.get(), Transfer::Waiting(_)) {
                        return Err(transfer_error("Local transfer already has a destination"));
                    }
                    let Transfer::Exposed(tensor) = entry.remove() else { unreachable!() };
                    return Ok(tensor);
                }
                Entry::Vacant(entry) => {
                    let (sender, receiver) = oneshot::channel();
                    entry.insert(Transfer::Waiting(sender));
                    receiver
                }
            }
        };
        let _waiting = Waiting { service: self, id };
        receiver.await.map_err(|_| transfer_error("Local transfer endpoint closed before handoff"))
    }
}

struct Waiting<'a, B: BackendIr> {
    service: &'a LocalCommService<B>,
    id: LocalTransferId,
}

impl<B: BackendIr> Drop for Waiting<'_, B> {
    fn drop(&mut self) {
        let mut state = self.service.state.lock().unwrap();
        if matches!(state.transfers.get(&self.id), Some(Transfer::Waiting(_))) {
            state.transfers.remove(&self.id);
        }
    }
}
