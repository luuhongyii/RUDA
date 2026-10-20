use crate::TrainingError;
use ruda_model::{
    record::{FileRecorder, Record},
    tensor::backend::Backend,
};
use std::{
    marker::PhantomData,
    path::{Path, PathBuf},
};

pub trait Checkpointer<R>: Send {
    fn save(&mut self, epoch: usize, record: R) -> Result<(), TrainingError>;
    fn restore(&mut self, epoch: usize) -> Result<R, TrainingError>;
    fn delete(&mut self, epoch: usize) -> Result<(), TrainingError>;
    /// Stored epochs in this checkpointer's namespace, including previous runs.
    fn epochs(&self) -> Result<Vec<usize>, TrainingError>;
}

/// Uses the recorder's atomic file replacement for the whole training record.
pub struct FileCheckpointer<B: Backend, R: Record<B>, F: FileRecorder<B>> {
    directory: PathBuf,
    recorder: F,
    device: B::Device,
    marker: PhantomData<fn() -> R>,
}

impl<B: Backend, R: Record<B>, F: FileRecorder<B>> FileCheckpointer<B, R, F> {
    pub fn new(directory: impl AsRef<Path>, recorder: F, device: B::Device) -> Self {
        Self {
            directory: directory.as_ref().to_owned(),
            recorder,
            device,
            marker: PhantomData,
        }
    }
    fn path(&self, epoch: usize) -> PathBuf {
        self.directory.join(format!("epoch-{epoch}"))
    }
}

impl<B: Backend, R: Record<B>, F: FileRecorder<B>> Checkpointer<R> for FileCheckpointer<B, R, F> {
    fn save(&mut self, epoch: usize, record: R) -> Result<(), TrainingError> {
        Ok(self.recorder.record(record, self.path(epoch))?)
    }
    fn restore(&mut self, epoch: usize) -> Result<R, TrainingError> {
        Ok(self.recorder.load(self.path(epoch), &self.device)?)
    }
    fn delete(&mut self, epoch: usize) -> Result<(), TrainingError> {
        let mut path = self.path(epoch);
        path.set_extension(F::file_extension());
        match std::fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        }
    }
    fn epochs(&self) -> Result<Vec<usize>, TrainingError> {
        let directory = match std::fs::read_dir(&self.directory) {
            Ok(directory) => directory,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error.into()),
        };
        let suffix = format!(".{}", F::file_extension());
        let mut epochs = Vec::new();
        for entry in directory {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some(number) = name
                .strip_prefix("epoch-")
                .and_then(|name| name.strip_suffix(&suffix))
            else {
                continue;
            };
            if let Ok(epoch) = number.parse::<usize>() {
                if name == format!("epoch-{epoch}{suffix}") {
                    epochs.push(epoch);
                }
            }
        }
        epochs.sort_unstable();
        Ok(epochs)
    }
}

/// Retains the newly saved checkpoint and the newest remaining epoch files.
/// Existing checkpoints are included after process restart; deletion follows a successful save.
pub struct KeepLastNCheckpoints<C> {
    inner: C,
    keep: usize,
}

impl<C> KeepLastNCheckpoints<C> {
    pub fn new(inner: C, keep: usize) -> Self {
        assert!(keep > 0, "At least one checkpoint must be retained");
        Self { inner, keep }
    }
}
impl<R, C: Checkpointer<R>> Checkpointer<R> for KeepLastNCheckpoints<C> {
    fn save(&mut self, epoch: usize, record: R) -> Result<(), TrainingError> {
        let mut saved = self.inner.epochs()?;
        saved.sort_unstable();
        saved.dedup();
        saved.retain(|saved| *saved != epoch);
        self.inner.save(epoch, record)?;
        saved.push(epoch);
        for old in saved.iter().take(saved.len().saturating_sub(self.keep)) {
            self.inner.delete(*old)?;
        }
        Ok(())
    }
    fn restore(&mut self, epoch: usize) -> Result<R, TrainingError> {
        self.inner.restore(epoch)
    }
    fn delete(&mut self, epoch: usize) -> Result<(), TrainingError> {
        self.inner.delete(epoch)
    }
    fn epochs(&self) -> Result<Vec<usize>, TrainingError> {
        self.inner.epochs()
    }
}
