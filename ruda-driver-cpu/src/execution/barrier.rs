use std::{
    cell::RefCell,
    sync::{Arc, atomic::{AtomicU32, Ordering}},
};

#[derive(Default)]
pub(crate) struct RudaBarrier {
    arrived: AtomicU32,
    exited: AtomicU32,
}

struct BarrierContext {
    state: Arc<RudaBarrier>,
    units: u32,
}

thread_local! {
    static ACTIVE_BARRIER: RefCell<Option<BarrierContext>> = const { RefCell::new(None) };
}

pub(crate) struct BarrierGuard(Option<BarrierContext>);

impl BarrierGuard {
    pub(crate) fn enter(state: Arc<RudaBarrier>, units: u32) -> Self {
        Self(ACTIVE_BARRIER.with(|active| active.replace(Some(BarrierContext { state, units }))))
    }
}

impl Drop for BarrierGuard {
    fn drop(&mut self) {
        ACTIVE_BARRIER.with(|active| active.replace(self.0.take()));
    }
}

pub extern "C" fn sync_ruda() {
    ACTIVE_BARRIER.with(|active| {
        let active = active.borrow();
        let Some(BarrierContext { state, units }) = active.as_ref() else { return };
        if *units <= 1 { return; }
        while state.exited.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
        state.arrived.fetch_add(1, Ordering::AcqRel);
        while state.arrived.load(Ordering::Acquire) < *units {
            std::hint::spin_loop();
        }
        if state.exited.fetch_add(1, Ordering::AcqRel) + 1 == *units {
            state.arrived.store(0, Ordering::Release);
            state.exited.store(0, Ordering::Release);
        }
    });
}
