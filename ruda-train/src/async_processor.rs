use crate::{EventProcessorTraining, ItemLazy, LearnerEvent, RunEvent, TrainingError, TrainingItem};
use ruda_model::tensor::StreamId;
use std::{
    sync::mpsc::{self, Receiver, SyncSender},
    thread::JoinHandle,
};

enum Message<T, V> {
    Run(RunEvent),
    Train(StreamId, LearnerEvent<T>),
    Valid(StreamId, LearnerEvent<V>),
    Flush(SyncSender<()>),
}

/// Ordered, bounded event processing on a separate thread. `flush` waits for all
/// preceding events and returns every unreported worker error in delivery order.
pub struct AsyncProcessorTraining<T, V> {
    sender: Option<SyncSender<Message<T, V>>>,
    errors: Receiver<TrainingError>,
    worker: Option<JoinHandle<()>>,
}
impl<T: Send + 'static, V: Send + 'static> AsyncProcessorTraining<T, V> {
    pub fn new<P: EventProcessorTraining<T, V> + 'static>(processor: P) -> Self {
        Self::with_capacity(processor, 1)
    }
    pub fn with_capacity<P: EventProcessorTraining<T, V> + 'static>(
        processor: P,
        capacity: usize,
    ) -> Self {
        Self::prepared(processor, capacity, Ok, Ok)
    }
    pub fn with_lazy<P>(processor: P) -> Self
    where
        T: ItemLazy,
        V: ItemLazy,
        P: EventProcessorTraining<T::Synced, V::Synced> + 'static,
    {
        Self::with_lazy_capacity(processor, 1)
    }
    pub fn with_lazy_capacity<P>(processor: P, capacity: usize) -> Self
    where
        T: ItemLazy,
        V: ItemLazy,
        P: EventProcessorTraining<T::Synced, V::Synced> + 'static,
    {
        Self::prepared(processor, capacity, T::sync, V::sync)
    }
    fn prepared<U: Send + 'static, W: Send + 'static, P, F, G>(
        mut processor: P,
        capacity: usize,
        mut train: F,
        mut valid: G,
    ) -> Self
    where
        P: EventProcessorTraining<U, W> + 'static,
        F: FnMut(T) -> Result<U, TrainingError> + Send + 'static,
        G: FnMut(V) -> Result<W, TrainingError> + Send + 'static,
    {
        let (sender, receiver) = mpsc::sync_channel(capacity);
        let (error_sender, errors) = mpsc::channel();
        let worker = std::thread::Builder::new()
            .name("ruda-metrics".into())
            .spawn(move || {
                while let Ok(message) = receiver.recv() {
                    let result = match message {
                        Message::Run(event) => processor.process_run(event),
                        Message::Train(stream, event) => prepare_event(stream, event, &mut train)
                            .and_then(|event| processor.process_train(event)),
                        Message::Valid(stream, event) => prepare_event(stream, event, &mut valid)
                            .and_then(|event| processor.process_valid(event)),
                        Message::Flush(callback) => {
                            if let Err(error) = processor.flush() {
                                let _ = error_sender.send(error);
                            }
                            let _ = callback.send(());
                            continue;
                        }
                    };
                    if let Err(error) = result {
                        let _ = error_sender.send(error);
                    }
                }
            })
            .expect("Can start metric processing thread");
        Self {
            sender: Some(sender),
            errors,
            worker: Some(worker),
        }
    }
    pub fn finish(mut self) -> Result<(), TrainingError> {
        self.flush()
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
            _ => panic!("The metric processing worker stopped unexpectedly"),
        }
    }
    fn send(&mut self, message: Message<T, V>) {
        if self
            .sender
            .as_ref()
            .expect("Metric processor is open")
            .send(message)
            .is_err()
        {
            self.died();
        }
    }
}
impl<T: Send + 'static, V: Send + 'static> EventProcessorTraining<T, V>
    for AsyncProcessorTraining<T, V>
{
    fn process_run(&mut self, event: RunEvent) -> Result<(), TrainingError> {
        self.send(Message::Run(event));
        self.check_errors()
    }
    fn process_train(&mut self, event: LearnerEvent<T>) -> Result<(), TrainingError> {
        self.send(Message::Train(StreamId::current(), event));
        self.check_errors()
    }
    fn process_valid(&mut self, event: LearnerEvent<V>) -> Result<(), TrainingError> {
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

fn prepare_event<T, U>(
    stream: StreamId,
    event: LearnerEvent<T>,
    prepare: &mut impl FnMut(T) -> Result<U, TrainingError>,
) -> Result<LearnerEvent<U>, TrainingError> {
    Ok(match event {
        LearnerEvent::ProcessedItem(item) => LearnerEvent::ProcessedItem(TrainingItem {
            item: stream.executes(|| prepare(item.item))?,
            progress: item.progress,
            iteration: item.iteration,
            learning_rate: item.learning_rate,
        }),
        LearnerEvent::StartSplit { epoch, total_items } => {
            LearnerEvent::StartSplit { epoch, total_items }
        }
        LearnerEvent::EndSplit { epoch, completed } => LearnerEvent::EndSplit { epoch, completed },
        LearnerEvent::EndEpoch(epoch) => LearnerEvent::EndEpoch(epoch),
    })
}
impl<T, V> Drop for AsyncProcessorTraining<T, V> {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
