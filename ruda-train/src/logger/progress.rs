use ruda_model::data::dataloader::Progress;

pub trait TrainingProgressLogger: Send {
    fn start(
        &mut self,
        total_epochs: usize,
        starting_epoch: usize,
        total_items: Option<usize>,
        label: Option<&str>,
    );
    fn update_epoch(&mut self, epoch: usize);
    fn start_split(&mut self, split: &str, total_items: usize);
    fn update_split(&mut self, items_processed: usize);
    fn end_split(&mut self);
    fn end(&mut self);
    fn log_event_training(&mut self, event: String);
}

pub trait EvaluationProgressLogger: Send {
    fn start_global_progress(&mut self, total_tests: usize);
    fn start_test(&mut self, name: &str, total_items: usize);
    fn update_test_progress(&mut self, items_processed: usize);
    fn end_test(&mut self);
    fn end_global_progress(&mut self);
    fn log_event_evaluation(&mut self, event: String);
}

#[derive(Clone, Debug)]
pub struct ProgressSnapshot {
    pub global: Progress,
    pub split: Progress,
}
impl ProgressSnapshot {
    pub fn new(global: Progress, split: Progress) -> Self {
        Self { global, split }
    }
}
