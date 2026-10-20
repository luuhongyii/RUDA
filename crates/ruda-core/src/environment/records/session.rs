//! The session in progress, where there is a database to record into.

use super::{
    RecordEffect, RecordLevel, Records, RecordsConfig, SESSIONS, Session, SessionId, Stamp, Stamped,
};
use crate::bytes::Bytes;
use crate::environment::persistence::{Database, Insertion, Origin, encode};
use crate::environment::sync::{AtomicU8, LazyLock, Mutex, Ordering};
use alloc::collections::VecDeque;
use alloc::string::{String, ToString};
use core::time::Duration;
use serde::Serialize;
use std::time::Instant;

/// The level, apart from the rest of the state: [`super::enabled`] is
/// asked on paths that must not take a lock.
static LEVEL: AtomicU8 = AtomicU8::new(RecordLevel::Basic as u8);

/// How many encoded bytes a session holds before it changes anything.
/// Past it the oldest record goes first: what led up to a change is kept,
/// and a session that never changes anything cannot grow without bound.
const PENDING_BUDGET: usize = 1 << 20;

struct State {
    keep_sessions: Option<u32>,
    label: Option<String>,
    current: Option<Current>,
}

/// The session in progress, bound to the environment it was started in.
struct Current {
    session: Session,
    database: Database,
    generation: u32,
    started: Instant,
    next_seq: u64,
    /// What the session recorded before it changed anything, in order:
    /// written if it does, dropped with it if it never does. Empty once
    /// the session is kept, and never past [`PENDING_BUDGET`].
    pending: VecDeque<Row>,
    /// The encoded size of [`Self::pending`].
    pending_bytes: usize,
    /// Whether the session changed the environment, and so is written.
    kept: bool,
}

/// One encoded record: its namespace, key and value.
struct Row {
    namespace: String,
    key: Bytes,
    value: Bytes,
}

impl Row {
    fn bytes(&self) -> usize {
        self.namespace.len() + self.key.len() + self.value.len()
    }
}

impl Current {
    /// Hold `row` until the session changes something, dropping the
    /// oldest rows held past the budget.
    fn hold(&mut self, row: Row) {
        self.pending_bytes += row.bytes();
        self.pending.push_back(row);
        while self.pending_bytes > PENDING_BUDGET {
            let Some(oldest) = self.pending.pop_front() else {
                break;
            };
            self.pending_bytes -= oldest.bytes();
        }
    }
}

static STATE: LazyLock<Mutex<State>> = LazyLock::new(|| {
    Mutex::new(State {
        keep_sessions: None,
        label: None,
        current: None,
    })
});

pub(crate) fn configure(config: RecordsConfig) {
    LEVEL.store(config.level as u8, Ordering::Relaxed);
    STATE.lock().keep_sessions = config.keep_sessions;
}

pub(crate) fn level() -> RecordLevel {
    match LEVEL.load(Ordering::Relaxed) {
        0 => RecordLevel::Off,
        1 => RecordLevel::Basic,
        _ => RecordLevel::Full,
    }
}

pub(crate) fn label(label: String) {
    let mut state = STATE.lock();
    state.label = Some(label.clone());
    if let Some(current) = state.current.as_mut() {
        current.session.label = Some(label);
        if current.kept {
            write_session(current);
        }
    }
}

pub(crate) fn stamp() -> Option<Stamp> {
    let mut state = STATE.lock();
    let current = current(&mut state)?;
    let stamp = Stamp {
        session: current.session.id,
        seq: current.next_seq,
        offset: current.started.elapsed(),
    };
    current.next_seq += 1;
    Some(stamp)
}

/// The session clock's reading, when `session` is still the one in
/// progress.
pub(crate) fn offset(session: SessionId) -> Option<Duration> {
    let state = STATE.lock();
    let current = state.current.as_ref()?;
    let live =
        current.session.id == session && current.generation == crate::environment::generation();
    live.then(|| current.started.elapsed())
}

pub(crate) fn write<V: Serialize>(
    namespace: &str,
    effect: RecordEffect,
    stamped: &Stamped<V>,
) -> bool {
    let row = Row {
        namespace: namespace.into(),
        key: encode(&(stamped.stamp.session, stamped.stamp.seq)),
        value: encode(stamped),
    };
    let mut state = STATE.lock();
    let keep_sessions = state.keep_sessions;
    let Some(current) = current(&mut state) else {
        return false;
    };
    // A record stamped in a session the environment has since switched
    // away from belongs to neither.
    if current.session.id != stamped.stamp.session {
        return false;
    }
    match (current.kept, effect) {
        (true, _) => insert(&current.database, &row),
        (false, RecordEffect::Observed) => {
            current.hold(row);
            true
        }
        (false, RecordEffect::Changed) => {
            keep(current, keep_sessions);
            insert(&current.database, &row)
        }
    }
}

/// The session changed the environment: write it, with what it recorded
/// before, and make room for it among the sessions kept.
fn keep(current: &mut Current, keep_sessions: Option<u32>) {
    if let Some(keep) = keep_sessions {
        Records::new(&current.database).prune(keep.saturating_sub(1) as usize);
    }
    current.kept = true;
    write_session(current);
    current.pending_bytes = 0;
    for row in core::mem::take(&mut current.pending) {
        insert(&current.database, &row);
    }
}

fn insert(database: &Database, row: &Row) -> bool {
    matches!(
        database.insert(&row.namespace, &row.key, &row.value, Origin::Local),
        Insertion::Stored
    )
}

/// The session of the active environment, started if there is none or the
/// environment switched since. `None` when recording is off or the
/// environment has no database.
fn current(state: &mut State) -> Option<&mut Current> {
    if level() == RecordLevel::Off {
        return None;
    }
    let generation = crate::environment::generation();
    let stale = state
        .current
        .as_ref()
        .is_none_or(|current| current.generation != generation);
    if stale {
        // A session replaced before it changed anything drops what it
        // was holding with it.
        state.current = Some(Current {
            session: start_session(state.label.clone()),
            database: Database::open_active()?,
            generation,
            started: Instant::now(),
            next_seq: 0,
            pending: VecDeque::new(),
            pending_bytes: 0,
            kept: false,
        });
    }
    state.current.as_mut()
}

fn write_session(current: &Current) {
    let key = encode(&current.session.id);
    current
        .database
        .replace(SESSIONS, &key, &encode(&current.session), Origin::Local);
}

/// The session a process starts now, under `label`.
fn start_session(label: Option<String>) -> Session {
    let started = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    Session {
        // Nanoseconds are unique enough between the processes sharing one
        // environment, and sort sessions by start without a lookup.
        id: SessionId(started.as_nanos() as u64),
        started_unix_ms: started.as_millis() as u64,
        ruda_version: env!("CARGO_PKG_VERSION").to_string(),
        label,
        process: std::process::id(),
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
    }
}
