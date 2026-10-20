use ruda_model::{
    data::{
        dataloader::{DataLoaderBuilder, batcher::Batcher},
        dataset::{
            Dataset,
            transform::PartialDataset,
            vision::{MnistDataset, MnistItem},
        },
    },
    module::Module,
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::{
        Int, Tensor, TensorData,
        backend::{AutodiffBackend, Backend},
    },
};
use ruda_nn::{
    Dropout, DropoutConfig, Linear, LinearConfig, Relu,
    conv::{Conv2d, Conv2dConfig},
    loss::CrossEntropyLossConfig,
    pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
};
use ruda_optim::{AdamConfig, GradientsAccumulator, training::TrainingRecord};
use ruda_train::{
    Checkpointer, ClassificationOutput, EpochState, FileCheckpointer, InferenceStep, Interrupter,
    Learner, MetricLogger, MetricUpdate, MetricsProcessor, Split, SupervisedTraining, TrainOutput,
    TrainStep, TrainingError, evaluate,
    metric::{AccuracyMetric, LossMetric},
};
use serde::{Deserialize, Serialize};
use std::{
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

#[cfg(not(feature = "test-cuda"))]
type Native = ruda_tensor_host::Host;
#[cfg(all(feature = "test-cuda", not(feature = "test-cuda-native")))]
type Native = ruda_tensor_device::cuda::Cuda;
#[cfg(feature = "test-cuda-native")]
type Native = ruda_tensor_device::cuda::CudaNative;
type AD = ruda_autodiff::Autodiff<Native>;
type Error = Box<dyn std::error::Error>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Config {
    num_classes: usize,
    hidden_size: usize,
    dropout: f64,
    epochs: usize,
    batch_size: usize,
    workers: usize,
    seed: u64,
    learning_rate: f64,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            num_classes: 10,
            hidden_size: 512,
            dropout: 0.5,
            epochs: 10,
            batch_size: 64,
            workers: 4,
            seed: 42,
            learning_rate: 1e-4,
        }
    }
}

#[derive(Module, Debug)]
struct Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
}
impl<B: Backend> Model<B> {
    fn new(config: &Config, device: &B::Device) -> Self {
        Self {
            conv1: Conv2dConfig::new([1, 8], [3, 3]).init(device),
            conv2: Conv2dConfig::new([8, 16], [3, 3]).init(device),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            dropout: DropoutConfig::new(config.dropout).init(),
            linear1: LinearConfig::new(16 * 8 * 8, config.hidden_size).init(device),
            linear2: LinearConfig::new(config.hidden_size, config.num_classes).init(device),
            activation: Relu::new(),
        }
    }
    fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [n, h, w] = images.dims();
        let x = self.conv1.forward(images.reshape([n, 1, h, w]));
        let x = self.dropout.forward(x);
        let x = self.conv2.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x).reshape([n, 16 * 8 * 8]);
        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);
        self.linear2.forward(x)
    }
    fn classification(&self, batch: Batch<B>) -> ClassificationOutput<B> {
        let output = self.forward(batch.images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), batch.targets.clone());
        ClassificationOutput::new(loss, output, batch.targets)
    }
}
impl<B: AutodiffBackend> TrainStep<B> for Model<B> {
    type Input = Batch<B>;
    type Output = ClassificationOutput<B::InnerBackend>;
    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let output = self.classification(batch);
        let grads = output.loss.backward();
        TrainOutput::new(self, grads, output.into_inner())
    }
}
impl<B: Backend> InferenceStep<B> for Model<B> {
    type Input = Batch<B>;
    type Output = ClassificationOutput<B>;
    fn step(&self, batch: Self::Input) -> Self::Output {
        self.classification(batch)
    }
}

#[derive(Clone, Debug)]
struct Batch<B: Backend> {
    images: Tensor<B, 3>,
    targets: Tensor<B, 1, Int>,
}
#[derive(Clone)]
struct MnistBatcher;
impl<B: Backend> Batcher<B, MnistItem, Batch<B>> for MnistBatcher {
    fn batch(&self, items: Vec<MnistItem>, device: &B::Device) -> Batch<B> {
        let images = items
            .iter()
            .map(|item| {
                let image = Tensor::<B, 2>::from_data(TensorData::from(item.image), device)
                    .reshape([1, 28, 28]);
                ((image / 255) - 0.1307) / 0.3081
            })
            .collect();
        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_data([item.label as i64], device))
            .collect();
        Batch {
            images: Tensor::cat(images, 0),
            targets: Tensor::cat(targets, 0),
        }
    }
}

struct ProgressLogger {
    writer: std::io::BufWriter<std::fs::File>,
    started: Instant,
}
impl ProgressLogger {
    fn new(path: PathBuf) -> Result<Self, Error> {
        Ok(Self {
            writer: std::io::BufWriter::new(
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)?,
            ),
            started: Instant::now(),
        })
    }
}
impl MetricLogger for ProgressLogger {
    fn log(&mut self, update: &MetricUpdate) -> Result<(), TrainingError> {
        let mut event = serde_json::to_value(update).map_err(|e| TrainingError(e.to_string()))?;
        event["elapsed_seconds"] = serde_json::json!(self.started.elapsed().as_secs_f64());
        event["process_id"] = serde_json::json!(std::process::id());
        writeln!(self.writer, "{event}")?;
        self.writer.flush()?;
        if update.event != "item" || update.iteration % 20 == 0 {
            println!("{event}");
        }
        Ok(())
    }
    fn flush(&mut self) -> Result<(), TrainingError> {
        Ok(self.writer.flush()?)
    }
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), Error> {
    std::fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

fn final_metrics(
    root: &Path,
    epoch: usize,
    split: Split,
) -> Result<Vec<ruda_train::metric::MetricEntry>, Error> {
    for line in std::fs::read_to_string(root.join("metrics.jsonl"))?
        .lines()
        .rev()
    {
        let Ok(update) = serde_json::from_str::<MetricUpdate>(line) else {
            continue;
        };
        if update.epoch == epoch && update.split == split && update.event == "end_split" {
            return Ok(update.metrics);
        }
    }
    Err("No completed epoch metrics were recorded".into())
}

fn train(root: &Path, pilot: bool, until: Option<usize>) -> Result<(), Error> {
    let config = Config::default();
    std::fs::create_dir_all(root)?;
    let config_path = root.join("config.json");
    if config_path.exists() {
        let saved: Config = serde_json::from_slice(&std::fs::read(&config_path)?)?;
        if saved != config {
            return Err("Existing configuration differs from this run".into());
        }
    } else {
        write_json(&config_path, &config)?;
    }
    let mode_path = root.join("mode.json");
    let mode = if pilot { "pilot" } else { "train" };
    if mode_path.exists() {
        let saved: String = serde_json::from_slice(&std::fs::read(&mode_path)?)?;
        if saved != mode {
            return Err("Pilot and full training must use separate directories".into());
        }
    } else {
        write_json(&mode_path, &mode)?;
    }
    let device = Default::default();
    Native::seed(&device, config.seed);
    let train_data = Arc::new(MnistDataset::train());
    let valid_data = Arc::new(MnistDataset::test());
    assert_eq!(train_data.len(), 60_000);
    assert_eq!(valid_data.len(), 10_000);
    let (train_count, valid_count, epochs) = if pilot {
        (512, 256, 1)
    } else {
        (60_000, 10_000, until.unwrap_or(config.epochs))
    };
    if epochs == 0 || epochs > config.epochs {
        return Err("Epoch limit must be in 1..=10".into());
    }
    let train_loader = DataLoaderBuilder::<AD, _, _>::new(MnistBatcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.workers)
        .set_device(device.clone())
        .build(PartialDataset::new(train_data, 0, train_count));
    let valid_loader = DataLoaderBuilder::<Native, _, _>::new(MnistBatcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.workers)
        .set_device(device.clone())
        .build(PartialDataset::new(valid_data, 0, valid_count));
    let model = Model::<AD>::new(&config, &device).fork(&device);
    let optim = AdamConfig::new().init();
    let mut checkpointer = FileCheckpointer::new(
        root.join("checkpoint"),
        BinFileRecorder::<FullPrecisionSettings>::new(),
        device.clone(),
    );
    let initial = TrainingRecord::capture(
        &model,
        &optim,
        &config.learning_rate,
        &GradientsAccumulator::new(),
        EpochState::default(),
    )?;
    let previous = checkpointer.epochs()?.into_iter().max();
    if previous.is_none() {
        checkpointer.save(0, initial)?;
    }
    let resume_epoch = previous.unwrap_or(0);
    if resume_epoch > epochs {
        return Err("Checkpoint exceeds the configured epoch count".into());
    }
    let processor =
        MetricsProcessor::<ClassificationOutput<Native>, ClassificationOutput<Native>>::new()
            .metric_train(AccuracyMetric::<Native>::new())
            .metric_train(LossMetric::<Native>::new())
            .metric_valid(AccuracyMetric::<Native>::new())
            .metric_valid(LossMetric::<Native>::new())
            .with_logger(ProgressLogger::new(root.join("metrics.jsonl"))?);
    write_json(
        &root.join("workload.json"),
        &serde_json::json!({
            "mode": mode, "train_items_per_epoch": train_count, "valid_items_per_epoch": valid_count,
            "epochs": epochs, "resume_epoch": resume_epoch, "batch_size": config.batch_size,
            "remaining_batches": (epochs - resume_epoch) * (train_count.div_ceil(config.batch_size) + valid_count.div_ceil(config.batch_size)),
        }),
    )?;
    let started = Instant::now();
    let result = SupervisedTraining::new(train_loader, valid_loader, device.clone())
        .num_epochs(epochs)
        .with_metrics(processor)
        .with_file_checkpointer(
            root.join("checkpoint"),
            BinFileRecorder::<FullPrecisionSettings>::new(),
            2,
        )
        .checkpoint(resume_epoch)
        .fit(Learner::new(model, optim, config.learning_rate));
    if let Some(error) = result.error {
        return Err(error.into());
    }
    if result.interrupted.is_some() || result.state.completed_epochs != epochs {
        return Err("Training did not finish the requested epochs".into());
    }
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    <BinFileRecorder<_> as Recorder<Native>>::record(
        &recorder,
        result.model.clone().into_record(),
        root.join("model"),
    )?;
    let loaded = Model::<Native>::new(&config, &device)
        .load_record(<BinFileRecorder<_> as Recorder<Native>>::load(
            &recorder,
            root.join("model"),
            &device,
        )?)
        .freeze();
    let item = MnistDataset::test().get(42).ok_or("Missing test item 42")?;
    let batch =
        <MnistBatcher as Batcher<Native, _, _>>::batch(&MnistBatcher, vec![item.clone()], &device);
    let expected = result.model.forward(batch.images.clone()).into_data();
    let restored = loaded.forward(batch.images).into_data();
    if expected != restored {
        return Err("Saved model logits differ after reload".into());
    }
    write_json(
        &root.join("result.json"),
        &serde_json::json!({
            "mode": mode, "state": result.state, "seconds": started.elapsed().as_secs_f64(),
            "train_metrics": final_metrics(root, epochs, Split::Train)?,
            "valid_metrics": final_metrics(root, epochs, Split::Valid)?,
            "model_record_round_trip": true, "resume_epoch": resume_epoch,
        }),
    )?;
    Ok(())
}

fn infer(root: &Path) -> Result<(), Error> {
    let config: Config = serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
    let device = Default::default();
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let model = Model::<Native>::new(&config, &device)
        .load_record(<BinFileRecorder<_> as Recorder<Native>>::load(
            &recorder,
            root.join("model"),
            &device,
        )?)
        .freeze();
    let loader = DataLoaderBuilder::<Native, _, _>::new(MnistBatcher)
        .batch_size(config.batch_size)
        .num_workers(config.workers)
        .set_device(device.clone())
        .build(MnistDataset::test());
    let mut processor = MetricsProcessor::<(), ClassificationOutput<Native>>::new()
        .for_evaluation()
        .metric_valid(AccuracyMetric::<Native>::new())
        .metric_valid(LossMetric::<Native>::new())
        .with_logger(ProgressLogger::new(root.join("inference.jsonl"))?);
    let result = evaluate(model, loader, &device, &mut processor, &Interrupter::new());
    if let Some(error) = result.error {
        return Err(error.into());
    }
    if !result.completed {
        return Err("Inference did not finish".into());
    }
    write_json(
        &root.join("inference.json"),
        &processor.store().metrics(1, Split::Test),
    )
}

fn main() -> Result<(), Error> {
    let args: Vec<_> = std::env::args().collect();
    if !(3..=4).contains(&args.len()) || (args.len() == 4 && args[1] != "train") {
        return Err(
            "Usage: mnist <train|pilot|infer> <artifact-directory> [train-until-epoch]".into(),
        );
    }
    let root = PathBuf::from(&args[2]);
    match args[1].as_str() {
        "train" => train(&root, false, args.get(3).map(|s| s.parse()).transpose()?),
        "pilot" => train(&root, true, None),
        "infer" => infer(&root),
        _ => Err("Expected train, pilot or infer".into()),
    }
}
