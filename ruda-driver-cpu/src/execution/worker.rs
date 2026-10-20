use std::collections::VecDeque;
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use super::compute_task::{ComputeTask, Message};
use super::affinity::{self, CoreId};
use super::progress::LaunchCompletion;

pub const MAX_STACK_SIZE: usize = 16 * 1024 * 1024;
pub const DEFAULT_STACK_SIZE: usize = 64 * 1024 * 1024;

fn resolve_stack_size() -> usize {
    if let Ok(value) = std::env::var("RUDA_CPU_STACK_SIZE")
        && let Ok(bytes) = value.parse::<usize>()
    {
        return bytes.max(MAX_STACK_SIZE);
    }
    if let Ok(value) = std::env::var("RUDA_CPU_STACK_MB")
        && let Ok(mb) = value.parse::<usize>()
    {
        return (mb.saturating_mul(1024 * 1024)).max(MAX_STACK_SIZE);
    }
    DEFAULT_STACK_SIZE
}

#[derive(Debug)]
pub struct Worker {
    tx: mpsc::Sender<PendingTask>,
    queued: Arc<AtomicUsize>,
}

impl Default for Worker {
    fn default() -> Self {
        Self::new(None)
    }
}

impl Worker {
    pub(crate) fn new(core: Option<CoreId>) -> Self {
        let (tx, rx) = mpsc::channel();
        let queued = Arc::new(AtomicUsize::new(0));
        let inner_worker = InnerWorker { rx, queued: queued.clone() };
        thread::Builder::new()
            .stack_size(resolve_stack_size())
            .spawn(move || {
                if let Some(core) = core {
                    affinity::set_for_current(core);
                }
                inner_worker.work();
            })
            .unwrap();
        Self { tx, queued }
    }

    pub fn send_task(&mut self, compute_task: ComputeTask) {
        self.enqueue(compute_task, None);
    }

    pub(crate) fn submit(&mut self, compute_task: ComputeTask, completion: Arc<LaunchCompletion>) {
        self.enqueue(compute_task, Some(completion));
    }

    pub(crate) fn load(&self) -> usize {
        self.queued.load(Ordering::Relaxed)
    }

    fn enqueue(&mut self, compute_task: ComputeTask, completion: Option<Arc<LaunchCompletion>>) {
        self.queued.fetch_add(1, Ordering::Relaxed);
        self.tx.send(PendingTask {
            message: Message::ComputeTask(compute_task),
            completion,
        }).unwrap();
    }

    pub fn send_stop(&mut self, callback: mpsc::Sender<()>) {
        self.tx.send(PendingTask { message: Message::EndTask(callback), completion: None }).unwrap();
    }
}

struct PendingTask {
    message: Message,
    completion: Option<Arc<LaunchCompletion>>,
}

struct InnerWorker {
    rx: mpsc::Receiver<PendingTask>,
    queued: Arc<AtomicUsize>,
}

impl InnerWorker {
    fn work(self) {
        const IDLE_POLL: Duration = Duration::from_micros(200);
        let mut pending: VecDeque<PendingTask> = VecDeque::new();
        loop {
            if pending.is_empty() {
                let start = Instant::now();
                loop {
                    match self.rx.try_recv() {
                        Ok(task) => {
                            pending.push_back(task);
                            break;
                        }
                        Err(mpsc::TryRecvError::Disconnected) => return,
                        Err(mpsc::TryRecvError::Empty) if start.elapsed() < IDLE_POLL => {
                            thread::yield_now();
                        }
                        Err(mpsc::TryRecvError::Empty) => {
                            match self.rx.recv() {
                                Ok(task) => pending.push_back(task),
                                Err(_) => return,
                            }
                            break;
                        }
                    }
                }
            } else if let Ok(task) = self.rx.try_recv() {
                pending.push_back(task);
            }

            let mut ready = None;
            for (index, task) in pending.iter().enumerate() {
                if matches!(&task.message, Message::EndTask(_)) {
                    if index == 0 { ready = Some(index); }
                    break;
                }
                if task.completion.as_ref().is_none_or(|completion| completion.is_ready()) {
                    ready = Some(index);
                    break;
                }
            }
            let Some(index) = ready else {
                thread::yield_now();
                continue;
            };
            let task = pending.remove(index).unwrap();
            match task.message {
                Message::ComputeTask(compute_task) => {
                    compute_task.compute();
                    if let Some(completion) = task.completion {
                        completion.unit_complete();
                    }
                    self.queued.fetch_sub(1, Ordering::Relaxed);
                }
                Message::EndTask(end_task) => end_task.send(()).unwrap(),
            }
        }
    }
}
