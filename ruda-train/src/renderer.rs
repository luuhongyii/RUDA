use crate::{
    logger::{EvaluationProgressLogger, ProgressSnapshot, TrainingProgressLogger},
    metric::{MetricDefinition, MetricSnapshot, NumericEntry},
};
use ruda_model::data::dataloader::Progress;
use std::{fmt, sync::Arc};

mod boxed;

#[cfg(feature = "tui")]
pub mod tui;

pub fn default_renderer(
    #[cfg_attr(not(feature = "tui"), allow(unused_variables))] interrupter: crate::Interrupter,
    #[cfg_attr(not(feature = "tui"), allow(unused_variables))] checkpoint: Option<usize>,
) -> Box<dyn MetricsRenderer> {
    #[cfg(feature = "tui")]
    {
        use std::io::IsTerminal;
        if std::io::stdout().is_terminal() && std::io::stdin().is_terminal() {
            return Box::new(tui::TuiMetricsRendererWrapper::new(interrupter, checkpoint));
        }
    }
    Box::new(CliMetricsRenderer::new())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvaluationName {
    name: Arc<String>,
}
impl EvaluationName {
    pub fn new(name: impl fmt::Display) -> Self {
        Self {
            name: Arc::new(name.to_string()),
        }
    }
    pub fn as_str(&self) -> &str {
        &self.name
    }
}
impl fmt::Display for EvaluationName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)
    }
}

#[derive(Clone, Debug)]
pub enum MetricState {
    Generic(MetricSnapshot),
    Numeric(MetricSnapshot, Option<NumericEntry>),
}

pub trait MetricsRendererTraining: Send + Sync {
    fn update_train(&mut self, state: MetricState);
    fn update_valid(&mut self, state: MetricState);
    fn on_train_end(&mut self) {}
    fn on_train_end_with_summary(&mut self, summary: crate::LearnerSummary) {
        println!("{summary}");
        self.on_train_end();
    }
}
pub trait MetricsRendererEvaluation: Send + Sync {
    fn update_test(&mut self, name: EvaluationName, state: MetricState);
    fn on_test_end(&mut self) {}
    fn on_test_end_with_summary(&mut self, summary: crate::LearnerSummary) {
        println!("{summary}");
        self.on_test_end();
    }
}
pub trait MetricsRenderer:
    MetricsRendererTraining
    + MetricsRendererEvaluation
    + TrainingProgressLogger
    + EvaluationProgressLogger
{
    fn register_metric(&mut self, definition: MetricDefinition);
}

pub struct CliMetricsRenderer {
    label: String,
    training_progress: ProgressSnapshot,
    eval_progress: ProgressSnapshot,
}
impl Default for CliMetricsRenderer {
    fn default() -> Self {
        Self::new()
    }
}
impl CliMetricsRenderer {
    pub fn new() -> Self {
        let empty = Progress::new(0, 0);
        Self {
            label: "training".into(),
            training_progress: ProgressSnapshot::new(empty.clone(), empty.clone()),
            eval_progress: ProgressSnapshot::new(empty.clone(), empty),
        }
    }
}
impl MetricsRendererTraining for CliMetricsRenderer {
    fn update_train(&mut self, _: MetricState) {}
    fn update_valid(&mut self, _: MetricState) {}
}
impl MetricsRendererEvaluation for CliMetricsRenderer {
    fn update_test(&mut self, _: EvaluationName, _: MetricState) {}
}
impl MetricsRenderer for CliMetricsRenderer {
    fn register_metric(&mut self, _: MetricDefinition) {}
}
impl TrainingProgressLogger for CliMetricsRenderer {
    fn start(
        &mut self,
        total_epochs: usize,
        starting_epoch: usize,
        total_items: Option<usize>,
        label: Option<&str>,
    ) {
        self.training_progress.global = Progress::new(starting_epoch, total_epochs);
        if let Some(total) = total_items {
            self.training_progress.split = Progress::new(0, total);
        }
        self.label = label.unwrap_or("training").into();
        println!("Starting {} for {} epochs.", self.label, total_epochs);
    }
    fn start_split(&mut self, split: &str, total_items: usize) {
        self.training_progress.split = Progress::new(0, total_items);
        println!("Starting split '{split}' with {total_items} items.");
    }
    fn update_split(&mut self, items_processed: usize) {
        self.training_progress.split.items_processed = items_processed;
        if self.training_progress.global.items_total == 0 {
            self.training_progress.global = self.training_progress.split.clone();
        }
        println!("{:?}", self.training_progress);
    }
    fn update_epoch(&mut self, epoch: usize) {
        self.training_progress.global.items_processed = epoch + 1;
    }
    fn end_split(&mut self) {
        println!("Split ended.");
    }
    fn end(&mut self) {
        println!("{} ended.", self.label);
    }
    fn log_event_training(&mut self, _: String) {}
}
impl EvaluationProgressLogger for CliMetricsRenderer {
    fn start_global_progress(&mut self, total_tests: usize) {
        self.eval_progress.global = Progress::new(0, total_tests);
        println!("Starting evaluation with {total_tests} test(s).");
    }
    fn start_test(&mut self, name: &str, total_items: usize) {
        self.eval_progress.global.items_processed += 1;
        self.eval_progress.split = Progress::new(0, total_items);
        println!("Starting test '{name}' with {total_items} items.");
    }
    fn update_test_progress(&mut self, items_processed: usize) {
        self.eval_progress.split.items_processed = items_processed;
        println!("{:?}", self.eval_progress);
    }
    fn end_test(&mut self) {}
    fn end_global_progress(&mut self) {}
    fn log_event_evaluation(&mut self, _: String) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_progress_tracks_epochs_unbounded_steps_and_evaluation() {
        let mut renderer = CliMetricsRenderer::new();
        renderer.start(4, 3, None, Some("resumed"));
        renderer.start_split("train", 5);
        renderer.update_split(2);
        assert_eq!(renderer.training_progress.global.items_processed, 3);
        assert_eq!(renderer.training_progress.split.items_total, 5);
        assert_eq!(renderer.training_progress.split.items_processed, 2);
        renderer.update_epoch(3);
        assert_eq!(renderer.training_progress.global.items_processed, 4);
        renderer.end_split();
        renderer.end();
        renderer.start(0, 0, Some(100), None);
        renderer.update_split(7);
        assert_eq!(renderer.training_progress.global.items_processed, 7);
        assert_eq!(renderer.training_progress.global.items_total, 100);
        renderer.start_global_progress(2);
        renderer.start_test("first", 10);
        renderer.update_test_progress(3);
        renderer.end_test();
        renderer.start_test("second", 4);
        assert_eq!(renderer.eval_progress.global.items_processed, 2);
        assert_eq!(renderer.eval_progress.global.items_total, 2);
        assert_eq!(renderer.eval_progress.split.items_processed, 0);
        renderer.update_test_progress(4);
        assert_eq!(renderer.eval_progress.split.items_total, 4);
        renderer.end_global_progress();
    }
}
