use alloc::sync::Arc;
use core::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Debug, Default)]
pub(crate) struct PageRef(Arc<AtomicUsize>);

impl PageRef {
    pub(crate) fn guarded(&self) -> bool {
        self.0.load(Ordering::Acquire) != 0
    }

    pub(crate) fn guard(&self) -> PageGuard {
        self.0.fetch_add(1, Ordering::AcqRel);
        PageGuard(self.clone())
    }
}

/// Keeps a recorded graph's entire allocation page out of reuse and relocation.
#[derive(Debug)]
pub struct PageGuard(PageRef);

impl Drop for PageGuard {
    fn drop(&mut self) {
        self.0.0.fetch_sub(1, Ordering::AcqRel);
    }
}
