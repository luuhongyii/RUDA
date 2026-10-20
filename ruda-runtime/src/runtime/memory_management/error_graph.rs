use alloc::{boxed::Box, format, sync::Arc, vec::Vec};
use core::num::NonZeroU64;
use ruda_core::backtrace::BackTrace;

use super::ManagedMemoryId;
use crate::runtime::{id::KernelId, server::ServerError};

/// Stable identity of a root failure, shared by every allocation it claims.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FailureId(NonZeroU64);

impl core::fmt::Display for FailureId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// A skipped launch records identities, never bindings that would retain allocations.
#[derive(Debug, Clone)]
pub struct Skipped {
    pub kernel: KernelId,
    pub needed: ManagedMemoryId,
    pub produced: Vec<ManagedMemoryId>,
}

#[derive(Debug)]
struct State {
    error: ServerError,
    claimed: u32,
    skipped: Vec<Skipped>,
    skipped_total: u64,
}

#[derive(Debug)]
struct Root {
    id: FailureId,
    state: spin::Mutex<State>,
}

/// Shared root and dependency history carried by failed allocations.
///
/// The last carrier releases the root without retaining intermediate buffers.
/// Descriptor-owned references also cover allocations that never acquired storage.
#[derive(Debug, Clone)]
pub struct ErrorGraph(Arc<Root>);

impl ErrorGraph {
    pub const MAX_SKIPPED: usize = 16;

    pub fn new(error: ServerError) -> Self {
        static MINTED: spin::Mutex<u64> = spin::Mutex::new(0);
        let id = {
            let mut minted = MINTED.lock();
            *minted = minted.checked_add(1).expect("failure identity space exhausted");
            FailureId(NonZeroU64::new(*minted).expect("failure ids start at one"))
        };
        Self(Arc::new(Root {
            id,
            state: spin::Mutex::new(State {
                error, claimed: 0, skipped: Vec::new(), skipped_total: 0,
            }),
        }))
    }

    pub fn id(&self) -> FailureId {
        self.0.id
    }

    pub fn error(&self) -> ServerError {
        self.0.state.lock().error.clone()
    }

    pub fn replace(&self, error: ServerError) {
        self.0.state.lock().error = error;
    }

    pub fn skipped(&self, record: Skipped) {
        let mut state = self.0.state.lock();
        state.skipped_total = state.skipped_total.saturating_add(1);
        if state.skipped.len() == Self::MAX_SKIPPED {
            state.skipped.remove(0);
        }
        state.skipped.push(record);
    }

    pub fn report(&self, memory: ManagedMemoryId) -> ServerError {
        let state = self.0.state.lock();
        let mut target = memory;
        let mut upper = state.skipped.len();
        let mut chain = Vec::new();
        while let Some(found) = state.skipped[..upper].iter()
            .rposition(|record| record.produced.contains(&target))
        {
            let record = &state.skipped[found];
            chain.push(format!(
                "skipped `{}`: it needed memory {:?}, which carried the failure",
                record.kernel.type_name(), record.needed,
            ));
            target = record.needed;
            upper = found;
        }
        let dropped = state.skipped_total.saturating_sub(state.skipped.len() as u64);
        if !chain.is_empty() && dropped > 0 {
            chain.push(format!(
                "({dropped} older skip record(s) were dropped; the walk may stop before the root)"
            ));
        }
        ServerError::Unwritten {
            failure: self.id().0.get(),
            claimed: state.claimed,
            chain,
            root: Box::new(state.error.clone()),
            backtrace: BackTrace::capture(),
        }
    }

    pub(crate) fn tag(&self) {
        let mut state = self.0.state.lock();
        state.claimed = state.claimed.checked_add(1).expect("too many failure carriers");
    }

    pub(crate) fn untag(&self) {
        let mut state = self.0.state.lock();
        debug_assert!(state.claimed > 0, "failure carrier released twice");
        state.claimed = state.claimed.saturating_sub(1);
    }
}
