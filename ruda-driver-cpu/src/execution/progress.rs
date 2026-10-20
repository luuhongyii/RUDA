use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU32, AtomicU64, Ordering},
};

use ruda::runtime::server::KernelExecution;

static NEXT_PARALLEL_LAUNCH: AtomicU64 = AtomicU64::new(1);

#[derive(Default)]
pub(crate) struct CpuSequence {
    completed: Arc<AtomicU64>,
    next: u64,
}

impl CpuSequence {
    pub(crate) fn reserve(
        &mut self,
        units: u32,
        execution: Option<KernelExecution>,
        parallel: Option<Arc<AtomicU64>>,
    ) -> Arc<LaunchCompletion> {
        let prerequisite = self.next;
        self.next += u64::from(units);
        Arc::new(LaunchCompletion {
            completed: self.completed.clone(),
            prerequisite,
            end: self.next,
            remaining: AtomicU32::new(units),
            execution: Mutex::new(execution),
            parallel_id: parallel.as_ref().map_or(0, |_| {
                NEXT_PARALLEL_LAUNCH.fetch_add(1, Ordering::Relaxed)
            }),
            parallel,
        })
    }

    pub(crate) fn wait(&self) {
        wait_for(&self.completed, self.next);
    }
}

pub(crate) struct LaunchCompletion {
    completed: Arc<AtomicU64>,
    prerequisite: u64,
    end: u64,
    remaining: AtomicU32,
    execution: Mutex<Option<KernelExecution>>,
    parallel_id: u64,
    parallel: Option<Arc<AtomicU64>>,
}

impl LaunchCompletion {
    pub(crate) fn is_ready(&self) -> bool {
        if self.completed.load(Ordering::Acquire) < self.prerequisite {
            return false;
        }
        let Some(parallel) = &self.parallel else { return true };
        match parallel.compare_exchange(
            0, self.parallel_id, Ordering::AcqRel, Ordering::Acquire,
        ) {
            Ok(_) => true,
            Err(owner) => owner == self.parallel_id,
        }
    }

    pub(crate) fn unit_complete(&self) {
        if self.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
            if let Some(execution) = self.execution.lock().unwrap().take() {
                execution.complete();
            }
            if let Some(parallel) = &self.parallel {
                parallel.store(0, Ordering::Release);
            }
            self.completed.fetch_add(self.end - self.prerequisite, Ordering::Release);
        }
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.completed.load(Ordering::Acquire) >= self.end
    }

    pub(crate) fn wait(&self) {
        wait_for(&self.completed, self.end);
    }
}

fn wait_for(completed: &AtomicU64, end: u64) {
    let mut spins = 0;
    while completed.load(Ordering::Acquire) < end {
        if spins < 1_000 {
            spins += 1;
            std::hint::spin_loop();
        } else {
            std::thread::yield_now();
        }
    }
}
