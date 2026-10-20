use super::*;
use ruda_model::data::{dataset::StatelessDataset, dataloader::batcher::StatelessBatcher};
use ruda_model::record::HalfPrecisionSettings;

fn resumable<B: Backend>(items: &[f32]) -> Arc<dyn DataLoader<B, Tensor<B, 1>>> {
    DataLoaderBuilder::new(StatelessBatcher(Batches))
        .batch_size(2).shuffle(57).num_workers(2)
        .build_resumable(StatelessDataset(Samples(items.to_vec())))
}

struct StopAt<C> {
    inner: C,
    at: usize,
    handle: Interrupter,
}
impl<R, C: Checkpointer<R>> Checkpointer<R> for StopAt<C> {
    fn save(&mut self, batch: usize, record: R) -> Result<(), TrainingError> {
        self.inner.save(batch, record)?;
        if batch == self.at { self.handle.stop(Some("batch checkpoint saved")); }
        Ok(())
    }
    fn restore(&mut self, batch: usize) -> Result<R, TrainingError> { self.inner.restore(batch) }
    fn delete(&mut self, batch: usize) -> Result<(), TrainingError> { self.inner.delete(batch) }
    fn epochs(&self) -> Result<Vec<usize>, TrainingError> { self.inner.epochs() }
}

#[test]
fn exact_continuation_at_every_train_and_valid_batch_with_dropout_and_pending_gradients() {
    let run = |directory: &std::path::Path, stop: usize, restore: Option<usize>| {
        let device = Default::default();
        AD::seed(&device, if restore.is_some() { 9999 } else { 1234 });
        let mut model = Regressor::<AD>::new();
        model.dropout = DropoutConfig::new(0.3).init();
        let learner = Learner::new(model, AdamConfig::new().init(),
            ExponentialLrSchedulerConfig::new(0.01, 0.8).init().unwrap());
        let handle = Interrupter::default();
        let checkpoint = StopAt {
            inner: KeepLastNCheckpoints::new(FileCheckpointer::new(
                directory, BinFileRecorder::<HalfPrecisionSettings>::default(), device,
            ), 1),
            at: stop, handle: handle.clone(),
        };
        let mut training = SupervisedTraining::new(
            resumable(&[1., 2., 3., 4., 5.]), resumable(&[1., 2., 3.]), Default::default(),
        ).num_epochs(3).grads_accumulation(2).with_metrics(metrics())
            .with_interrupter(handle).with_batch_checkpointer(checkpoint, 1);
        if let Some(batch) = restore { training = training.batch_checkpoint(batch); }
        training.fit(learner)
    };
    let baseline_dir = Temp::new();
    let baseline = run(&baseline_dir.0, usize::MAX, None);
    assert!(baseline.error.is_none(), "{:?}", baseline.error);
    for boundary in 1..15 {
        let directory = Temp::new();
        let stopped = run(&directory.0, boundary, None);
        assert!(stopped.error.is_none(), "boundary {boundary}: {:?}", stopped.error);
        assert!(stopped.interrupted.is_some());
        let resumed = run(&directory.0, usize::MAX, Some(boundary));
        assert!(resumed.error.is_none(), "boundary {boundary}: {:?}", resumed.error);
        assert_eq!(resumed.state, baseline.state);
        assert_eq!(scalar(&resumed.model).to_bits(), scalar(&baseline.model).to_bits(), "boundary {boundary}");
        for epoch in 1..=3 {
            for split in [Split::Train, Split::Valid] {
                assert_eq!(resumed.metrics.find_metric("Loss", epoch, split).unwrap().to_bits(),
                    baseline.metrics.find_metric("Loss", epoch, split).unwrap().to_bits(),
                    "boundary {boundary}, epoch {epoch}, split {split:?}");
            }
        }
    }
}
