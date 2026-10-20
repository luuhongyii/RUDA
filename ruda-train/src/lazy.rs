use crate::{
    ClassificationOutput, MultiLabelClassificationOutput, RegressionOutput, SequenceOutput,
    TrainingError, TrainingItem,
};
use ruda_model::tensor::backend::{AutodiffBackend, Backend};

/// Prepare an output on its producing stream before metrics consume it.
/// `sync` submits pending work, not a GPU-completion fence or a tensor readback.
pub trait ItemLazy: Send + 'static {
    type Synced: Send + 'static;
    fn sync(self) -> Result<Self::Synced, TrainingError>;
}

/// Outputs still using an autodiff backend can be handed to the metric worker
/// with this wrapper; all tensor fields are converted to the inner backend.
pub struct AutodiffItem<T>(pub T);

impl ItemLazy for () {
    type Synced = ();
    fn sync(self) -> Result<Self::Synced, TrainingError> {
        Ok(())
    }
}

impl<T: ItemLazy> ItemLazy for TrainingItem<T> {
    type Synced = TrainingItem<T::Synced>;
    fn sync(self) -> Result<Self::Synced, TrainingError> {
        Ok(TrainingItem {
            item: self.item.sync()?,
            progress: self.progress,
            iteration: self.iteration,
            learning_rate: self.learning_rate,
        })
    }
}

macro_rules! prepare_output {
    ($output:ident) => {
        impl<B: Backend> ItemLazy for $output<B> {
            type Synced = Self;
            fn sync(self) -> Result<Self, TrainingError> {
                B::flush(&self.loss.device()).map_err(|error| TrainingError(error.to_string()))?;
                Ok(self)
            }
        }
        impl<B: AutodiffBackend> ItemLazy for AutodiffItem<$output<B>> {
            type Synced = $output<B::InnerBackend>;
            fn sync(self) -> Result<Self::Synced, TrainingError> {
                B::flush(&self.0.loss.device())
                    .map_err(|error| TrainingError(error.to_string()))?;
                Ok(self.0.into_inner())
            }
        }
    };
}
prepare_output!(RegressionOutput);
prepare_output!(ClassificationOutput);
prepare_output!(MultiLabelClassificationOutput);
prepare_output!(SequenceOutput);
