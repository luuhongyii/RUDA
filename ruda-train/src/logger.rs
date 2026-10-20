pub use crate::MetricLogger;
mod progress;
pub use progress::*;
use crate::metric::{MetricDefinition, MetricId, MetricSnapshot, NumericEntry};
use crate::{MetricUpdate, Split, TrainingError};
use std::{
    collections::{BTreeMap, HashMap},
    fmt::Display,
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, mpsc},
    thread::JoinHandle,
};

pub trait Logger<T>: Send {
    fn log(&mut self, item: T) -> Result<(), TrainingError>;
    fn flush(&mut self) -> Result<(), TrainingError> {
        Ok(())
    }
}

pub struct FileLogger {
    writer: BufWriter<File>,
}
impl FileLogger {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, TrainingError> {
        Ok(Self {
            writer: BufWriter::new(
                OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(path)?,
            ),
        })
    }
}
impl<T: Display> Logger<T> for FileLogger {
    fn log(&mut self, item: T) -> Result<(), TrainingError> {
        Ok(writeln!(self.writer, "{item}")?)
    }
    fn flush(&mut self) -> Result<(), TrainingError> {
        Ok(self.writer.flush()?)
    }
}

#[derive(Clone, Default)]
pub struct InMemoryLogger {
    values: Arc<Mutex<Vec<String>>>,
}
impl InMemoryLogger {
    pub fn values(&self) -> Vec<String> {
        self.values.lock().unwrap().clone()
    }
}
impl<T: Display> Logger<T> for InMemoryLogger {
    fn log(&mut self, item: T) -> Result<(), TrainingError> {
        self.values.lock().unwrap().push(item.to_string());
        Ok(())
    }
}

enum Message<T> {
    Log(T),
    Sync(mpsc::SyncSender<()>),
}
pub struct AsyncLogger<T> {
    sender: Option<mpsc::Sender<Message<T>>>,
    errors: mpsc::Receiver<TrainingError>,
    worker: Option<JoinHandle<()>>,
}
impl<T: Send + 'static> AsyncLogger<T> {
    pub fn new<L: Logger<T> + 'static>(mut logger: L) -> Self {
        let (sender, receiver) = mpsc::channel();
        let (error_sender, errors) = mpsc::channel();
        let worker = std::thread::Builder::new()
            .name("ruda-logger".into())
            .spawn(move || {
                while let Ok(message) = receiver.recv() {
                    match message {
                        Message::Log(item) => {
                            if let Err(error) = logger.log(item) {
                                let _ = error_sender.send(error);
                            }
                        }
                        Message::Sync(callback) => {
                            if let Err(error) = logger.flush() {
                                let _ = error_sender.send(error);
                            }
                            let _ = callback.send(());
                        }
                    }
                }
                if let Err(error) = logger.flush() {
                    let _ = error_sender.send(error);
                }
            })
            .expect("Can start logger thread");
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
    fn worker_failed(&mut self) -> TrainingError {
        if let Some(worker) = self.worker.take() {
            if let Err(payload) = worker.join() {
                std::panic::resume_unwind(payload);
            }
        }
        TrainingError("Logger worker stopped unexpectedly".into())
    }
    pub fn sync(&mut self) -> Result<(), TrainingError> {
        let (sender, receiver) = mpsc::sync_channel(1);
        if self
            .sender
            .as_ref()
            .unwrap()
            .send(Message::Sync(sender))
            .is_err()
        {
            return Err(self.worker_failed());
        }
        if receiver.recv().is_err() {
            return Err(self.worker_failed());
        }
        self.check_errors()
    }
    pub fn finish(mut self) -> Result<(), TrainingError> {
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            if let Err(payload) = worker.join() {
                std::panic::resume_unwind(payload);
            }
        }
        self.check_errors()
    }
}
impl<T: Send + 'static> Logger<T> for AsyncLogger<T> {
    fn log(&mut self, item: T) -> Result<(), TrainingError> {
        if self
            .sender
            .as_ref()
            .unwrap()
            .send(Message::Log(item))
            .is_err()
        {
            return Err(self.worker_failed());
        }
        self.check_errors()
    }
    fn flush(&mut self) -> Result<(), TrainingError> {
        self.sync()
    }
}
impl<T> Drop for AsyncLogger<T> {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

pub struct FileMetricLogger {
    directory: PathBuf,
    definitions: HashMap<MetricId, MetricDefinition>,
    writers: HashMap<(Split, Option<String>, MetricId), AsyncLogger<String>>,
    last_epoch: Option<usize>,
    evaluation: bool,
}
impl FileMetricLogger {
    pub(crate) fn read_numeric_file(path: &Path) -> Result<Vec<NumericEntry>, TrainingError> {
        let contents = match std::fs::read_to_string(path) {
            Ok(contents) => contents,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error.into()),
        };
        contents
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| NumericEntry::deserialize(line).map_err(TrainingError))
            .collect()
    }
    pub fn new(directory: impl AsRef<Path>) -> Self {
        Self {
            directory: directory.as_ref().into(),
            definitions: HashMap::new(),
            writers: HashMap::new(),
            last_epoch: None,
            evaluation: false,
        }
    }
    pub fn new_eval(directory: impl AsRef<Path>) -> Self {
        Self {
            evaluation: true,
            ..Self::new(directory)
        }
    }
    fn file_path(&self, name: &str, epoch: usize, split: Split) -> PathBuf {
        self.file_path_named(name, epoch, split, None)
    }
    fn file_path_named(
        &self,
        name: &str,
        epoch: usize,
        split: Split,
        tag: Option<&str>,
    ) -> PathBuf {
        if let Some(tag) = tag {
            let mut directory = self.directory.join("test").join(format_tag(tag));
            if !self.evaluation {
                directory = directory.join(format!("epoch-{epoch}"));
            }
            return directory.join(format!("{}.log", name.replace(' ', "_")));
        }
        let directory = if self.evaluation {
            self.directory.clone()
        } else {
            let split = match split {
                Split::Train => "train",
                Split::Valid => "valid",
                Split::Test => "test",
            };
            self.directory.join(split).join(format!("epoch-{epoch}"))
        };
        directory.join(format!("{}.log", name.replace(' ', "_")))
    }
    fn write_snapshots(
        &mut self,
        update: &MetricUpdate,
        snapshots: &[MetricSnapshot],
        tag: Option<&str>,
    ) -> Result<(), TrainingError> {
        if !self.evaluation && self.last_epoch != Some(update.epoch) {
            self.flush()?;
            self.writers.clear();
            self.last_epoch = Some(update.epoch);
        }
        if !matches!(update.event.as_str(), "item" | "end_split" | "TrainStep" | "EnvStep" | "EpisodeEnd") {
            return Ok(());
        }
        for snapshot in snapshots {
            if snapshot.serialized.is_not_available() {
                continue;
            }
            let definition = self
                .definitions
                .get(&snapshot.definition.metric_id)
                .ok_or_else(|| TrainingError("Metric definition was not registered".into()))?;
            let key = (
                update.split,
                tag.map(str::to_owned),
                definition.metric_id.clone(),
            );
            if !self.writers.contains_key(&key) {
                let path = self.file_path_named(&definition.name, update.epoch, update.split, tag);
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                self.writers
                    .insert(key.clone(), AsyncLogger::new(FileLogger::new(path)?));
            }
            self.writers
                .get_mut(&key)
                .unwrap()
                .log(snapshot.serialized.serialized.clone())?;
        }
        if update.event == "end_split" {
            self.flush()?;
        }
        Ok(())
    }
}
pub(crate) fn format_tag(tag: &str) -> String {
    tag.trim().replace(' ', "-").to_lowercase()
}
impl MetricLogger for FileMetricLogger {
    fn needs_snapshots(&self) -> bool {
        true
    }
    fn log(&mut self, _: &MetricUpdate) -> Result<(), TrainingError> {
        Err(TrainingError(
            "Serialized metric logging requires metric snapshots".into(),
        ))
    }
    fn log_metric_definition(
        &mut self,
        definition: &MetricDefinition,
    ) -> Result<(), TrainingError> {
        self.definitions
            .insert(definition.metric_id.clone(), definition.clone());
        Ok(())
    }
    fn log_snapshots(
        &mut self,
        update: &MetricUpdate,
        snapshots: &[MetricSnapshot],
    ) -> Result<(), TrainingError> {
        self.write_snapshots(update, snapshots, None)
    }
    fn log_snapshots_named(
        &mut self,
        update: &MetricUpdate,
        snapshots: &[MetricSnapshot],
        name: &str,
    ) -> Result<(), TrainingError> {
        self.write_snapshots(update, snapshots, Some(name))
    }
    fn read_numeric_named(
        &mut self,
        metric: &str,
        epoch: usize,
        name: &str,
    ) -> Result<Vec<NumericEntry>, TrainingError> {
        self.flush()?;
        Self::read_numeric_file(&self.file_path_named(metric, epoch, Split::Test, Some(name)))
    }
    fn read_numeric(
        &mut self,
        name: &str,
        epoch: usize,
        split: Split,
    ) -> Result<Vec<NumericEntry>, TrainingError> {
        self.flush()?;
        Self::read_numeric_file(&self.file_path(name, epoch, split))
    }
    fn flush(&mut self) -> Result<(), TrainingError> {
        let mut errors = Vec::new();
        for writer in self.writers.values_mut() {
            if let Err(error) = writer.sync() {
                errors.push(error.0);
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(TrainingError(errors.join("\n")))
        }
    }
}

#[derive(Default)]
struct MemoryMetrics {
    definitions: HashMap<MetricId, MetricDefinition>,
    values: BTreeMap<(usize, Split, Option<String>, String), Vec<String>>,
}
#[derive(Clone, Default)]
pub struct InMemoryMetricLogger {
    state: Arc<Mutex<MemoryMetrics>>,
}
impl InMemoryMetricLogger {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn entries(&self, name: &str, epoch: usize, split: Split) -> Vec<String> {
        self.state
            .lock()
            .unwrap()
            .values
            .get(&(epoch, split, None, name.into()))
            .cloned()
            .unwrap_or_default()
    }
    pub fn definition(&self, id: &MetricId) -> Option<MetricDefinition> {
        self.state.lock().unwrap().definitions.get(id).cloned()
    }
    pub fn entries_named(&self, metric: &str, epoch: usize, name: &str) -> Vec<String> {
        self.state
            .lock()
            .unwrap()
            .values
            .get(&(epoch, Split::Test, Some(name.into()), metric.into()))
            .cloned()
            .unwrap_or_default()
    }
    fn write_snapshots(
        &mut self,
        update: &MetricUpdate,
        snapshots: &[MetricSnapshot],
        tag: Option<&str>,
    ) -> Result<(), TrainingError> {
        if !matches!(update.event.as_str(), "item" | "end_split" | "TrainStep" | "EnvStep" | "EpisodeEnd") {
            return Ok(());
        }
        let mut state = self.state.lock().unwrap();
        for snapshot in snapshots {
            let name = state
                .definitions
                .get(&snapshot.definition.metric_id)
                .ok_or_else(|| TrainingError("Metric definition was not registered".into()))?
                .name
                .to_string();
            state
                .values
                .entry((update.epoch, update.split, tag.map(str::to_owned), name))
                .or_default()
                .push(snapshot.serialized.serialized.clone());
        }
        Ok(())
    }
}
impl MetricLogger for InMemoryMetricLogger {
    fn needs_snapshots(&self) -> bool {
        true
    }
    fn log(&mut self, _: &MetricUpdate) -> Result<(), TrainingError> {
        Err(TrainingError(
            "Serialized metric logging requires metric snapshots".into(),
        ))
    }
    fn log_metric_definition(
        &mut self,
        definition: &MetricDefinition,
    ) -> Result<(), TrainingError> {
        self.state
            .lock()
            .unwrap()
            .definitions
            .insert(definition.metric_id.clone(), definition.clone());
        Ok(())
    }
    fn log_snapshots(
        &mut self,
        update: &MetricUpdate,
        snapshots: &[MetricSnapshot],
    ) -> Result<(), TrainingError> {
        self.write_snapshots(update, snapshots, None)
    }
    fn log_snapshots_named(
        &mut self,
        update: &MetricUpdate,
        snapshots: &[MetricSnapshot],
        name: &str,
    ) -> Result<(), TrainingError> {
        self.write_snapshots(update, snapshots, Some(name))
    }
    fn read_numeric_named(
        &mut self,
        metric: &str,
        epoch: usize,
        name: &str,
    ) -> Result<Vec<NumericEntry>, TrainingError> {
        Ok(self
            .entries_named(metric, epoch, name)
            .iter()
            .filter_map(|entry| NumericEntry::deserialize(entry).ok())
            .collect())
    }
    fn read_numeric(
        &mut self,
        name: &str,
        epoch: usize,
        split: Split,
    ) -> Result<Vec<NumericEntry>, TrainingError> {
        Ok(self
            .entries(name, epoch, split)
            .iter()
            .filter_map(|entry| NumericEntry::deserialize(entry).ok())
            .collect())
    }
}
