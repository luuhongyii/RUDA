use crate::{EpochState, TrainingError};
use ruda_model::{
    data::dataloader::DataLoaderState,
    module::{AutodiffModule, ModuleDTypeRecord},
    record::{NativePrecisionSettings, PrecisionSettings, Record},
    tensor::backend::{AutodiffBackend, Backend},
};
use ruda_optim::{Optimizer, lr_scheduler::LrScheduler, training::TrainingRecord};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

pub(crate) fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>, TrainingError> {
    bincode::serde::encode_to_vec(value, bincode::config::standard())
        .map_err(|error| TrainingError(error.to_string()))
}

pub(crate) fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, TrainingError> {
    let (value, read) = bincode::serde::decode_from_slice(bytes, bincode::config::standard())
        .map_err(|error| TrainingError(error.to_string()))?;
    if read != bytes.len() {
        return Err(TrainingError("Trailing continuation state bytes".into()));
    }
    Ok(value)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContinuationPhase {
    Train,
    Valid,
    EpochEnd,
    Finished,
}

/// Position immediately after a completed batch or epoch transition.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchContinuation {
    pub version: u32,
    pub epoch: usize,
    pub phase: ContinuationPhase,
    pub counters: EpochState,
    pub batches: usize,
    pub iteration: usize,
    pub accumulated: usize,
    pub accumulation: usize,
    pub train: Option<DataLoaderState>,
    pub valid: Option<DataLoaderState>,
    pub train_source: Vec<u8>,
    pub valid_source: Vec<u8>,
    pub backend_rng: Vec<u8>,
    pub inner_backend_rng: Vec<u8>,
    pub processor: Vec<u8>,
    pub event_store: Vec<u8>,
    pub early_stopping: Option<Vec<u8>>,
    pub stopped_early: bool,
}

impl<B: Backend> Record<B> for BatchContinuation {
    type Item<P: PrecisionSettings> = Self;
    fn into_item<P: PrecisionSettings>(self) -> Self { self }
    fn from_item<P: PrecisionSettings>(item: Self, _: &B::Device) -> Self { item }
}

pub(crate) type BatchRecordInner<B, M, O, S> =
    TrainingRecord<B, M, O, S, (ModuleDTypeRecord, BatchContinuation)>;

/// Atomic model/optimizer/scheduler/gradient and continuation payload.
/// Component tensors retain their original values and storage dtypes independently
/// of the recorder's compact settings.
pub struct BatchLearnerRecord<B, M, O, S>(pub(crate) BatchRecordInner<B, M, O, S>)
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: Optimizer<M, B>,
    S: LrScheduler;

impl<B, M, O, S> Record<B> for BatchLearnerRecord<B, M, O, S>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: Optimizer<M, B>,
    S: LrScheduler,
{
    type Item<P: PrecisionSettings> =
        <BatchRecordInner<B, M, O, S> as Record<B>>::Item<NativePrecisionSettings>;
    fn into_item<P: PrecisionSettings>(self) -> Self::Item<P> {
        self.0.into_item::<NativePrecisionSettings>()
    }
    fn from_item<P: PrecisionSettings>(item: Self::Item<P>, device: &B::Device) -> Self {
        Self(BatchRecordInner::from_item::<NativePrecisionSettings>(item, device))
    }
}
