use alloc::vec::Vec;

use super::{Binding, ServerError};
use crate::runtime::memory_management::ErrorGraph;

/// Provisional claims installed immediately before work is submitted or executed.
#[derive(Debug)]
#[must_use]
pub struct WriteScope {
    written: Vec<Binding>,
    provisional: Option<ErrorGraph>,
    armed: bool,
}

impl WriteScope {
    /// Claim the write set. Dropping an unfinished scope leaves TornDown behind.
    pub fn new(written: Vec<Binding>) -> Self {
        let mut scope = Self::deferred(written);
        scope.enter();
        scope
    }

    /// Retain a write until its queued position is reached.
    pub fn deferred(written: Vec<Binding>) -> Self {
        Self { written, provisional: None, armed: true }
    }

    /// Install claims only after preceding queued producers have been submitted.
    pub fn enter(&mut self) {
        if !self.armed || self.provisional.is_some() { return; }
        for binding in &self.written {
            if binding.range().is_ok_and(|range| !range.is_empty() || !binding.memory.is_allocated()) {
                let root = self.provisional.get_or_insert_with(|| ErrorGraph::new(ServerError::TornDown));
                binding.claim(root.clone());
            }
        }
    }

    /// The exact views owned by this write.
    pub fn bindings(&self) -> &[Binding] { &self.written }

    /// Settle this write only; claims by newer writers are left untouched.
    pub fn complete(mut self) {
        self.enter();
        self.armed = false;
        if let Some(root) = self.provisional.take() {
            for binding in &self.written { binding.release(root.id()); }
        }
    }

    /// Replace the provisional root while retaining its identity and every carrier.
    pub fn fail(mut self, error: ServerError) -> ServerError {
        self.enter();
        self.armed = false;
        if let Some(root) = self.provisional.take() { root.replace(error.clone()); }
        error
    }

    /// Settle a submission result.
    pub fn finish(self, result: Result<(), ServerError>) -> Result<(), ServerError> {
        match result {
            Ok(()) => { self.complete(); Ok(()) }
            Err(error) => Err(self.fail(error)),
        }
    }
}

impl Drop for WriteScope {
    fn drop(&mut self) {
        if self.armed { self.enter(); }
    }
}
