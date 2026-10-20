//! Backend-generic supervised training, validation, metrics and epoch checkpoints.

mod async_processor;
mod checkpoint;
mod continuation;
mod event;
mod learner;
mod lazy;
pub mod metric;
pub mod logger;
pub mod renderer;
pub mod summary;
mod output;
mod step;
mod training;
mod evaluator;
#[cfg(feature = "rl")]
pub mod rl;

pub use async_processor::*;
pub use checkpoint::*;
pub use continuation::{BatchContinuation, BatchLearnerRecord, ContinuationPhase};
pub use event::*;
pub use learner::*;
pub use lazy::*;
pub use output::*;
pub use step::*;
pub use training::*;
pub use evaluator::*;
pub use summary::{LearnerSummary, LearnerSummaryConfig, MetricSummary, SummaryMetrics};

#[cfg(test)]
mod tests;
