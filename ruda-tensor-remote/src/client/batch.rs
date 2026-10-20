use crate::shared::{ComputeTask, Task};

pub(crate) struct OutgoingBatch { tasks: Vec<Task>, bytes: usize, tasks_limit: usize, bytes_limit: usize }

impl OutgoingBatch {
    pub(crate) fn new() -> Self {
        let config = ruda_tensor_config::config();
        Self { tasks: Vec::new(), bytes: 0, tasks_limit: config.remote.flush_threshold,
            bytes_limit: config.remote.flush_bytes_threshold }
    }

    pub(crate) fn push(&mut self, task: Task) -> bool {
        if let Task::Compute(ComputeTask::RegisterTensor(_, data), _) = &task {
            self.bytes = self.bytes.saturating_add(data.bytes.len());
        }
        self.tasks.push(task);
        self.tasks.len() >= self.tasks_limit || self.bytes >= self.bytes_limit
    }

    pub(crate) fn take(&mut self) -> Option<Task> {
        if self.tasks.is_empty() { return None; }
        self.bytes = 0;
        Some(Task::Batch(core::mem::take(&mut self.tasks)))
    }
}

pub(crate) fn needs_flush(task: &Task) -> bool {
    match task {
        Task::Close(_) => true,
        Task::Compute(task, _) => matches!(task,
            ComputeTask::ExposeTensorLocal { .. } | ComputeTask::RegisterTensorLocal { .. }
            | ComputeTask::ExposeTensorRemote { .. } | ComputeTask::RegisterTensorRemote(..)
            | ComputeTask::ExposeTensorCapability { .. } | ComputeTask::RegisterTensorCapability { .. }
            | ComputeTask::ProfileAbandon(_) | ComputeTask::QueueGraphRelease(_)),
        _ => false,
    }
}
