use crate::execution::{
    progress::CpuSequence,
    runner::KernelRunner,
    schedule::ScheduleTask,
};
use ruda::runtime::logging::ServerLogger;
use ruda::runtime::server::ServerError;
use std::sync::{Arc, Mutex, OnceLock};

static INSTANCE: OnceLock<CpuExecutionQueue> = OnceLock::new();

#[derive(Clone)]
/// The shared CPU dispatcher; each stream supplies its own completion sequence.
pub struct CpuExecutionQueue {
    runner: Arc<Mutex<KernelRunner>>,
    direct_sequence: Arc<Mutex<CpuSequence>>,
}

impl CpuExecutionQueue {
    /// Adds a new task to the queue.
    pub fn add(&self, task: ScheduleTask) -> Result<(), ServerError> {
        self.add_to_stream(task, &mut self.direct_sequence.lock().unwrap())
    }

    pub(crate) fn add_to_stream(
        &self,
        mut task: ScheduleTask,
        sequence: &mut CpuSequence,
    ) -> Result<(), ServerError> {
        // The service normally settled submission before this queue is reached.
        // Direct runner users still get the same submission-time dependency check.
        let scope = match &mut task {
            ScheduleTask::Execute { execution, ruda_count, ruda_dim, .. } => {
                if ruda_count.contains(&0) || ruda_dim.num_elems() == 0 {
                    execution.skip();
                    return Ok(());
                }
                execution.begin_submission()?
            }
            ScheduleTask::Write { .. } => None,
        };
        match task {
            ScheduleTask::Write { data, mut buffer, destination: _destination } => {
                sequence.wait();
                buffer.write().copy_from_slice(&data);
            }
            ScheduleTask::Execute {
                execution, mlir_engine, bindings, kind, ruda_dim, ruda_count,
            } => {
                if !mlir_engine.0.shared_memories.0.is_empty() {
                    sequence.wait();
                }
                self.runner.lock().unwrap().dispatch_data(
                    mlir_engine, bindings, kind, ruda_dim, ruda_count, sequence, Some(execution),
                );
            }
        }
        if let Some(scope) = scope { scope.complete(); }
        Ok(())
    }

    /// Flushes the queue, making sure all enqueued tasks before this point are executed.
    pub fn flush(&self) {
        self.runner.lock().unwrap().flush();
    }

    pub(crate) fn reclaim_completed(&self) {
        self.runner.lock().unwrap().reclaim_completed();
    }

    /// Resolves the global execution queue instance.
    pub fn get(logger: Arc<ServerLogger>) -> Self {
        INSTANCE.get_or_init(|| Self::init(logger)).clone()
    }

    fn init(logger: Arc<ServerLogger>) -> Self {
        Self {
            runner: Arc::new(Mutex::new(KernelRunner::new(logger))),
            direct_sequence: Arc::new(Mutex::new(CpuSequence::default())),
        }
    }
}
