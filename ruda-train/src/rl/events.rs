use super::EpisodeSummary;
use crate::{ItemLazy, LearnerSummary, TrainingError};
use ruda_model::{data::dataloader::Progress, tensor::StreamId};
use std::{
    sync::mpsc::{self, Receiver, SyncSender},
    thread::JoinHandle,
};

pub struct EvaluationItem<T> {
    pub item: T,
    pub progress: Progress,
    pub iteration: Option<usize>,
}
impl<T> EvaluationItem<T> {
    pub fn new(item: T, progress: Progress, iteration: Option<usize>) -> Self {
        Self {
            item,
            progress,
            iteration,
        }
    }
    fn map<U>(
        self,
        prepare: &mut impl FnMut(T) -> Result<U, TrainingError>,
    ) -> Result<EvaluationItem<U>, TrainingError> {
        Ok(EvaluationItem::new(
            prepare(self.item)?,
            self.progress,
            self.iteration,
        ))
    }
}
impl<T: ItemLazy> ItemLazy for EvaluationItem<T> {
    type Synced = EvaluationItem<T::Synced>;
    fn sync(self) -> Result<Self::Synced, TrainingError> {
        self.map(&mut T::sync)
    }
}
pub enum RLEvent<T, A> {
    Start {
        total_items: usize,
        label: Option<String>,
    },
    TrainStep(EvaluationItem<T>),
    EnvStep(EvaluationItem<A>),
    EpisodeEnd(EvaluationItem<EpisodeSummary>),
    End(Option<LearnerSummary>),
}
pub enum AgentEvaluationEvent<A> {
    Start(usize),
    EnvStep(EvaluationItem<A>),
    EpisodeEnd(EvaluationItem<EpisodeSummary>),
    End,
}
pub trait RLEventProcessor<T, A>: Send {
    fn process_train(&mut self, event: RLEvent<T, A>) -> Result<(), TrainingError>;
    fn process_valid(&mut self, event: AgentEvaluationEvent<A>) -> Result<(), TrainingError>;
    fn flush(&mut self) -> Result<(), TrainingError> {
        Ok(())
    }
}
pub struct LazyRLProcessor<P>(pub P);
impl<T: ItemLazy, A: ItemLazy, P: RLEventProcessor<T::Synced, A::Synced>> RLEventProcessor<T, A>
    for LazyRLProcessor<P>
{
    fn process_train(&mut self, event: RLEvent<T, A>) -> Result<(), TrainingError> {
        self.0
            .process_train(prepare_train(event, &mut T::sync, &mut A::sync)?)
    }
    fn process_valid(&mut self, event: AgentEvaluationEvent<A>) -> Result<(), TrainingError> {
        self.0.process_valid(prepare_valid(event, &mut A::sync)?)
    }
    fn flush(&mut self) -> Result<(), TrainingError> {
        self.0.flush()
    }
}
fn prepare_train<T, A, U, V>(
    event: RLEvent<T, A>,
    train: &mut impl FnMut(T) -> Result<U, TrainingError>,
    action: &mut impl FnMut(A) -> Result<V, TrainingError>,
) -> Result<RLEvent<U, V>, TrainingError> {
    Ok(match event {
        RLEvent::Start { total_items, label } => RLEvent::Start { total_items, label },
        RLEvent::TrainStep(item) => RLEvent::TrainStep(item.map(train)?),
        RLEvent::EnvStep(item) => RLEvent::EnvStep(item.map(action)?),
        RLEvent::EpisodeEnd(item) => RLEvent::EpisodeEnd(item),
        RLEvent::End(summary) => RLEvent::End(summary),
    })
}
fn prepare_valid<A, V>(
    event: AgentEvaluationEvent<A>,
    action: &mut impl FnMut(A) -> Result<V, TrainingError>,
) -> Result<AgentEvaluationEvent<V>, TrainingError> {
    Ok(match event {
        AgentEvaluationEvent::Start(count) => AgentEvaluationEvent::Start(count),
        AgentEvaluationEvent::EnvStep(item) => AgentEvaluationEvent::EnvStep(item.map(action)?),
        AgentEvaluationEvent::EpisodeEnd(item) => AgentEvaluationEvent::EpisodeEnd(item),
        AgentEvaluationEvent::End => AgentEvaluationEvent::End,
    })
}
enum Message<T, A> {
    Train(StreamId, RLEvent<T, A>),
    Valid(StreamId, AgentEvaluationEvent<A>),
    Flush(SyncSender<()>),
}
pub struct AsyncRLProcessor<T, A> {
    sender: Option<SyncSender<Message<T, A>>>,
    errors: Receiver<TrainingError>,
    worker: Option<JoinHandle<()>>,
}
impl<T: Send + 'static, A: Send + 'static> AsyncRLProcessor<T, A> {
    pub fn new<P: RLEventProcessor<T, A> + 'static>(processor: P) -> Self {
        Self::with_capacity(processor, 1)
    }
    pub fn with_capacity<P: RLEventProcessor<T, A> + 'static>(
        processor: P,
        capacity: usize,
    ) -> Self {
        Self::prepared(processor, capacity, Ok, Ok)
    }
    pub fn with_lazy<P>(processor: P) -> Self
    where
        T: ItemLazy,
        A: ItemLazy,
        P: RLEventProcessor<T::Synced, A::Synced> + 'static,
    {
        Self::with_lazy_capacity(processor, 1)
    }
    pub fn with_lazy_capacity<P>(processor: P, capacity: usize) -> Self
    where
        T: ItemLazy,
        A: ItemLazy,
        P: RLEventProcessor<T::Synced, A::Synced> + 'static,
    {
        Self::prepared(processor, capacity, T::sync, A::sync)
    }
    fn prepared<U: Send + 'static, V: Send + 'static, P, F, G>(
        mut processor: P,
        capacity: usize,
        mut train: F,
        mut action: G,
    ) -> Self
    where
        P: RLEventProcessor<U, V> + 'static,
        F: FnMut(T) -> Result<U, TrainingError> + Send + 'static,
        G: FnMut(A) -> Result<V, TrainingError> + Send + 'static,
    {
        let (sender, receiver) = mpsc::sync_channel(capacity);
        let (error_sender, errors) = mpsc::channel();
        let worker = std::thread::Builder::new()
            .name("ruda-rl-metrics".into())
            .spawn(move || {
                while let Ok(message) = receiver.recv() {
                    let result = match message {
                        Message::Train(stream, event) => stream
                            .executes(|| prepare_train(event, &mut train, &mut action))
                            .and_then(|event| processor.process_train(event)),
                        Message::Valid(stream, event) => stream
                            .executes(|| prepare_valid(event, &mut action))
                            .and_then(|event| processor.process_valid(event)),
                        Message::Flush(reply) => {
                            if let Err(error) = processor.flush() {
                                let _ = error_sender.send(error);
                            }
                            let _ = reply.send(());
                            continue;
                        }
                    };
                    if let Err(error) = result {
                        let _ = error_sender.send(error);
                    }
                }
            })
            .expect("Can start RL metrics worker");
        Self {
            sender: Some(sender),
            errors,
            worker: Some(worker),
        }
    }
    fn check_errors(&self) -> Result<(), TrainingError> {
        let errors: Vec<_> = self.errors.try_iter().map(|error| error.0).collect();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(TrainingError(errors.join("\n")))
        }
    }
    fn died(&mut self) -> ! {
        match self.worker.take().map(JoinHandle::join) {
            Some(Err(payload)) => std::panic::resume_unwind(payload),
            _ => panic!("RL metrics worker stopped unexpectedly"),
        }
    }
    fn send(&mut self, message: Message<T, A>) {
        if self
            .sender
            .as_ref()
            .expect("RL processor is open")
            .send(message)
            .is_err()
        {
            self.died();
        }
    }
    pub fn finish(mut self) -> Result<(), TrainingError> {
        self.flush()
    }
}
impl<T: Send + 'static, A: Send + 'static> RLEventProcessor<T, A> for AsyncRLProcessor<T, A> {
    fn process_train(&mut self, event: RLEvent<T, A>) -> Result<(), TrainingError> {
        self.send(Message::Train(StreamId::current(), event));
        self.check_errors()
    }
    fn process_valid(&mut self, event: AgentEvaluationEvent<A>) -> Result<(), TrainingError> {
        self.send(Message::Valid(StreamId::current(), event));
        self.check_errors()
    }
    fn flush(&mut self) -> Result<(), TrainingError> {
        let (sender, receiver) = mpsc::sync_channel(1);
        self.send(Message::Flush(sender));
        if receiver.recv().is_err() {
            self.died();
        }
        self.check_errors()
    }
}
impl<T, A> Drop for AsyncRLProcessor<T, A> {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
