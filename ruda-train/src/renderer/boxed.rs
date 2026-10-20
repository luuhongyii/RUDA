use super::*;

impl<R: MetricsRenderer + ?Sized> MetricsRendererTraining for Box<R> {
    fn update_train(&mut self, state: MetricState) {
        (**self).update_train(state);
    }
    fn update_valid(&mut self, state: MetricState) {
        (**self).update_valid(state);
    }
    fn on_train_end(&mut self) {
        (**self).on_train_end();
    }
    fn on_train_end_with_summary(&mut self, summary: crate::LearnerSummary) {
        (**self).on_train_end_with_summary(summary);
    }
}

impl<R: MetricsRenderer + ?Sized> MetricsRendererEvaluation for Box<R> {
    fn update_test(&mut self, name: EvaluationName, state: MetricState) {
        (**self).update_test(name, state);
    }
    fn on_test_end(&mut self) {
        (**self).on_test_end();
    }
    fn on_test_end_with_summary(&mut self, summary: crate::LearnerSummary) {
        (**self).on_test_end_with_summary(summary);
    }
}

impl<R: MetricsRenderer + ?Sized> MetricsRenderer for Box<R> {
    fn register_metric(&mut self, definition: MetricDefinition) {
        (**self).register_metric(definition);
    }
}

impl<R: MetricsRenderer + ?Sized> TrainingProgressLogger for Box<R> {
    fn start(&mut self, epochs: usize, start: usize, items: Option<usize>, label: Option<&str>) {
        (**self).start(epochs, start, items, label);
    }
    fn update_epoch(&mut self, epoch: usize) {
        (**self).update_epoch(epoch);
    }
    fn start_split(&mut self, split: &str, items: usize) {
        (**self).start_split(split, items);
    }
    fn update_split(&mut self, items: usize) {
        (**self).update_split(items);
    }
    fn end_split(&mut self) {
        (**self).end_split();
    }
    fn end(&mut self) {
        (**self).end();
    }
    fn log_event_training(&mut self, event: String) {
        (**self).log_event_training(event);
    }
}

impl<R: MetricsRenderer + ?Sized> EvaluationProgressLogger for Box<R> {
    fn start_global_progress(&mut self, tests: usize) {
        (**self).start_global_progress(tests);
    }
    fn start_test(&mut self, name: &str, items: usize) {
        (**self).start_test(name, items);
    }
    fn update_test_progress(&mut self, items: usize) {
        (**self).update_test_progress(items);
    }
    fn end_test(&mut self) {
        (**self).end_test();
    }
    fn end_global_progress(&mut self) {
        (**self).end_global_progress();
    }
    fn log_event_evaluation(&mut self, event: String) {
        (**self).log_event_evaluation(event);
    }
}
