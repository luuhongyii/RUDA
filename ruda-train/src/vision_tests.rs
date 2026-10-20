use super::*;
use crate::metric::vision::*;
use ruda_model::tensor::Int;

fn image(values: Vec<f32>, shape: [usize; 4]) -> Tensor<TestBackend, 4> {
    Tensor::from_data(TensorData::new(values, shape), &Default::default())
}
fn near(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 3e-5,
        "actual={actual}, expected={expected}"
    );
}

// Independent direct two-dimensional F64 window oracle (no tensor/conv calls).
fn blur(
    values: &[f64],
    height: usize,
    width: usize,
    size: usize,
    sigma: f64,
    reflect: bool,
) -> Vec<f64> {
    let p = size as isize / 2;
    let weights: Vec<_> = (-p..=p)
        .map(|i| (-(i * i) as f64 / (2.0 * sigma * sigma)).exp())
        .collect();
    let norm = weights.iter().sum::<f64>().powi(2);
    let index = |i: isize, limit: usize| {
        if i < 0 {
            (-i) as usize
        } else if i >= limit as isize {
            (2 * limit as isize - 2 - i) as usize
        } else {
            i as usize
        }
    };
    (0..height * width)
        .map(|i| {
            let (row, col) = ((i / width) as isize, (i % width) as isize);
            let mut sum = 0.0;
            for dy in -p..=p {
                for dx in -p..=p {
                    let (y, x) = (row + dy, col + dx);
                    if !reflect && (x < 0 || y < 0 || x >= width as isize || y >= height as isize) {
                        continue;
                    }
                    sum += values[index(y, height) * width + index(x, width)]
                        * weights[(dy + p) as usize]
                        * weights[(dx + p) as usize]
                        / norm;
                }
            }
            sum
        })
        .collect()
}
fn maps(
    x: &[f64],
    y: &[f64],
    h: usize,
    w: usize,
    size: usize,
    sigma: f64,
    k1: f64,
    k2: f64,
    range: f64,
    reflect: bool,
) -> (f64, f64) {
    let conv = |values: &[f64]| blur(values, h, w, size, sigma, reflect);
    let mx = conv(x);
    let my = conv(y);
    let xx = conv(&x.iter().map(|v| v * v).collect::<Vec<_>>());
    let yy = conv(&y.iter().map(|v| v * v).collect::<Vec<_>>());
    let xy = conv(&x.iter().zip(y).map(|(a, b)| a * b).collect::<Vec<_>>());
    let (c1, c2) = ((k1 * range).powi(2), (k2 * range).powi(2));
    let (mut ssim, mut cs) = (0.0, 0.0);
    for i in 0..x.len() {
        let contrast = (2.0 * (xy[i] - mx[i] * my[i]) + c2)
            / ((xx[i] - mx[i] * mx[i]).max(0.0) + (yy[i] - my[i] * my[i]).max(0.0) + c2);
        let luminance = (2.0 * mx[i] * my[i] + c1) / (mx[i] * mx[i] + my[i] * my[i] + c1);
        cs += contrast;
        ssim += contrast * luminance;
    }
    (ssim / x.len() as f64, cs / x.len() as f64)
}
fn pool(values: &[f64], h: usize, w: usize) -> Vec<f64> {
    (0..h / 2)
        .flat_map(|y| {
            (0..w / 2).map(move |x| {
                (values[2 * y * w + 2 * x]
                    + values[2 * y * w + 2 * x + 1]
                    + values[(2 * y + 1) * w + 2 * x]
                    + values[(2 * y + 1) * w + 2 * x + 1])
                    / 4.0
            })
        })
        .collect()
}
fn multiscale(
    mut x: Vec<f64>,
    mut y: Vec<f64>,
    mut h: usize,
    mut w: usize,
    config: &MsSsimMetricConfig,
) -> f64 {
    let mut value = 1.0;
    for (scale, beta) in config.betas.iter().enumerate() {
        let (ssim, cs) = maps(
            &x,
            &y,
            h,
            w,
            config.kernel_size,
            config.sigma as f64,
            config.k1 as f64,
            config.k2 as f64,
            config.pixel_range as f64,
            true,
        );
        value *= (if scale + 1 == config.betas.len() {
            ssim
        } else {
            cs
        })
        .max(0.0)
        .powf(*beta as f64);
        if scale + 1 < config.betas.len() {
            x = pool(&x, h, w);
            y = pool(&y, h, w);
            h /= 2;
            w /= 2;
        }
    }
    value
}

#[test]
fn psnr_per_image_mean_unequal_batches_and_clear() {
    let mut metric = PsnrMetric::new(PsnrMetricConfig::new(1.0));
    metric
        .update(&PsnrInput::new(
            image(vec![0.0, 0.0], [2, 1, 1, 1]),
            image(vec![0.1, 0.01], [2, 1, 1, 1]),
        ))
        .unwrap();
    near(metric.value().unwrap(), 30.0);
    metric
        .update(&PsnrInput::new(
            image(vec![0.0], [1, 1, 1, 1]),
            image(vec![0.0], [1, 1, 1, 1]),
        ))
        .unwrap();
    near(metric.value().unwrap(), 160.0 / 3.0);
    metric.clear();
    assert!(metric.value().is_none());
    assert_eq!(metric.with_name("reconstruction").name(), "reconstruction");
}
#[test]
fn psnr_rgb_noncontiguous_range_and_custom_floor() {
    let shape = [1, 3, 2, 3];
    let x = image(vec![5.0; 18], shape).swap_dims(2, 3);
    let y = image(vec![6.0; 18], shape).swap_dims(2, 3);
    let mut metric = PsnrMetric::new(PsnrMetricConfig::new(255.0));
    metric.update(&PsnrInput::new(x.clone(), y)).unwrap();
    near(metric.value().unwrap(), 20.0 * 255.0f64.log10());
    let mut metric = PsnrMetric::new(PsnrMetricConfig::new(1.0).with_epsilon(1e-4));
    metric.update(&PsnrInput::new(x.clone(), x)).unwrap();
    near(metric.value().unwrap(), 40.0);
}
#[test]
fn ssim_zero_padded_separable_matches_direct_window_oracle() {
    let shape = [2, 2, 5, 7];
    let size = 35;
    let x: Vec<f32> = (0..140)
        .map(|i| ((i * 7 % 31) as f32) / 15.0 - 1.0)
        .collect();
    let y: Vec<f32> = (0..140)
        .map(|i| ((i * 11 % 23) as f32) / 11.0 - 1.0)
        .collect();
    let config = SsimMetricConfig::new(2.0)
        .with_kernel_size(3)
        .with_sigma(0.9)
        .with_k1_k2(0.02, 0.04);
    let expected = (0..4)
        .map(|n| {
            maps(
                &x[n * size..(n + 1) * size]
                    .iter()
                    .map(|&v| v as f64)
                    .collect::<Vec<_>>(),
                &y[n * size..(n + 1) * size]
                    .iter()
                    .map(|&v| v as f64)
                    .collect::<Vec<_>>(),
                5,
                7,
                3,
                0.9,
                0.02,
                0.04,
                2.0,
                false,
            )
            .0
        })
        .sum::<f64>()
        / 4.0;
    let mut metric = SsimMetric::new(config);
    metric
        .update(&SsimInput::new(image(x, shape), image(y, shape)))
        .unwrap();
    near(metric.value().unwrap(), expected);
    metric.clear();
    assert!(metric.value().is_none());
}
#[test]
fn ssim_identity_negative_scores_and_strides() {
    let mut metric = SsimMetric::new(SsimMetricConfig::new(1.0).with_kernel_size(3));
    let x = image(
        (0..30)
            .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
            .collect(),
        [1, 1, 5, 6],
    )
    .swap_dims(2, 3);
    metric
        .update(&SsimInput::new(x.clone(), x.clone()))
        .unwrap();
    near(metric.value().unwrap(), 1.0);
    metric.clear();
    metric
        .update(&SsimInput::new(x.clone(), x.neg() + 1.0))
        .unwrap();
    assert!(metric.value().unwrap() < 0.0);
    assert_eq!(metric.with_name("ssim").name(), "ssim");
}
#[test]
fn ms_ssim_reflect_odd_shapes_zero_beta_matches_direct_oracle() {
    let shape = [2, 2, 15, 19];
    let plane = 15 * 19;
    let x: Vec<f32> = (0..4 * plane)
        .map(|i| (i * 13 % 71) as f32 / 70.0)
        .collect();
    let y: Vec<f32> = x
        .iter()
        .enumerate()
        .map(|(i, v)| v * 0.73 + (i % 11) as f32 / 55.0)
        .collect();
    let config = MsSsimMetricConfig::new(1.0)
        .with_kernel_size(3)
        .with_sigma(0.8)
        .with_channels(2)
        .with_betas(vec![0.2, 0.0, 0.8]);
    let expected = (0..4)
        .map(|n| {
            multiscale(
                x[n * plane..(n + 1) * plane]
                    .iter()
                    .map(|&v| v as f64)
                    .collect(),
                y[n * plane..(n + 1) * plane]
                    .iter()
                    .map(|&v| v as f64)
                    .collect(),
                15,
                19,
                &config,
            )
        })
        .sum::<f64>()
        / 4.0;
    let mut metric = MsSsimMetric::new(config, &Default::default());
    metric
        .update(&MsSsimInput::new(image(x, shape), image(y, shape)))
        .unwrap();
    near(metric.value().unwrap(), expected);
    metric.clear();
    assert!(metric.value().is_none());
}
#[test]
fn ms_ssim_default_five_scale_rgb_and_batch_weighting() {
    let shape = [1, 3, 176, 176];
    let count = 3 * 176 * 176;
    let x = image(vec![0.25; count], shape);
    let y = image(vec![0.5; count], shape);
    let mut metric = MsSsimMetric::new(MsSsimMetricConfig::new(1.0), &Default::default());
    metric.update(&MsSsimInput::new(x.clone(), y)).unwrap();
    let expected =
        ((2.0 * 0.25 * 0.5 + 0.0001) / (0.25f64.powi(2) + 0.5f64.powi(2) + 0.0001)).powf(0.1333);
    near(metric.value().unwrap(), expected);
    let xx = x.repeat_dim(0, 2);
    metric.update(&MsSsimInput::new(xx.clone(), xx)).unwrap();
    near(metric.value().unwrap(), (expected + 2.0) / 3.0);
    assert_eq!(metric.with_name("ms").name(), "ms");
}
#[test]
fn ms_ssim_negative_contrast_is_clamped_before_fractional_power() {
    let x = image(
        (0..64)
            .map(|i| if (i / 8 + i % 8) % 2 == 0 { 0.0 } else { 1.0 })
            .collect(),
        [1, 1, 8, 8],
    );
    let mut metric = MsSsimMetric::new(
        MsSsimMetricConfig::new(1.0)
            .with_kernel_size(3)
            .with_channels(1)
            .with_betas(vec![0.5]),
        &Default::default(),
    );
    metric
        .update(&MsSsimInput::new(x.clone(), x.clone().neg() + 1.0))
        .unwrap();
    near(metric.value().unwrap(), 0.0);
    let config = MsSsimMetricConfig::new(1.0)
        .with_kernel_size(3)
        .with_channels(1);
    let mut zero_weight = MsSsimMetric::new(
        config.clone().with_betas(vec![0.0, 1.0]),
        &Default::default(),
    );
    zero_weight
        .update(&MsSsimInput::new(x.clone(), x.neg() + 1.0))
        .unwrap();
    near(zero_weight.value().unwrap(), 1.0);
    let mut unnormalized = MsSsimMetric::new(config.with_betas(vec![2.0]), &Default::default());
    unnormalized
        .update(&MsSsimInput::new(
            image(vec![0.25; 9], [1, 1, 3, 3]),
            image(vec![0.5; 9], [1, 1, 3, 3]),
        ))
        .unwrap();
    near(
        unnormalized.value().unwrap(),
        ((0.25 + 0.0001) / (0.3125f64 + 0.0001)).powi(2),
    );
}
#[test]
fn dice_global_counts_not_batch_mean_and_empty_masks() {
    let device = Default::default();
    let mut metric = DiceMetric::<TestBackend, 3>::new();
    metric
        .update(&DiceInput::new(
            Tensor::from_ints([[[1, 0]]], &device),
            Tensor::from_ints([[[1, 0]]], &device),
        ))
        .unwrap();
    metric
        .update(&DiceInput::new(
            Tensor::from_ints([[[1, 1]], [[1, 1]]], &device),
            Tensor::from_ints([[[0, 0]], [[0, 0]]], &device),
        ))
        .unwrap();
    near(metric.value().unwrap(), (2.0 + 1e-7) / (6.0 + 1e-7));
    metric.clear();
    assert!(metric.value().is_none());
    metric
        .update(&DiceInput::new(
            Tensor::zeros([1, 1, 3], &device),
            Tensor::zeros([1, 1, 3], &device),
        ))
        .unwrap();
    near(metric.value().unwrap(), 1.0);
}
#[test]
fn dice_background_selection_rank_five_and_noncontiguous() {
    let device = Default::default();
    let shape = [1, 2, 1, 2, 2];
    let x = Tensor::<TestBackend, 5, Int>::from_data(
        TensorData::new(vec![1i32, 1, 1, 1, 1, 0, 0, 0], shape),
        &device,
    )
    .swap_dims(3, 4);
    let y = Tensor::<TestBackend, 5, Int>::from_data(
        TensorData::new(vec![1i32, 1, 1, 1, 0, 1, 0, 0], shape),
        &device,
    )
    .swap_dims(3, 4);
    let input = DiceInput::new(x, y);
    let mut foreground = DiceMetric::new();
    let mut all = DiceMetric::with_config(DiceMetricConfig {
        include_background: true,
        ..Default::default()
    });
    foreground.update(&input).unwrap();
    all.update(&input).unwrap();
    near(foreground.value().unwrap(), 1e-7 / (2.0 + 1e-7));
    near(all.value().unwrap(), (8.0 + 1e-7) / (10.0 + 1e-7));
}
#[test]
fn vision_configs_and_invalid_shapes_are_rejected() {
    let s = SsimMetricConfig::new(1.0);
    assert_eq!((s.k1, s.k2, s.kernel_size, s.sigma), (0.01, 0.03, 11, 1.5));
    let m = MsSsimMetricConfig::new(1.0);
    assert_eq!(m.betas, vec![0.0448, 0.2856, 0.3001, 0.2363, 0.1333]);
    for beta in [
        vec![],
        vec![0.0],
        vec![-1.0],
        vec![f32::NAN],
        vec![f32::INFINITY],
    ] {
        assert!(
            std::panic::catch_unwind(|| MsSsimMetricConfig::new(1.0).with_betas(beta)).is_err()
        );
    }
    assert!(std::panic::catch_unwind(|| SsimMetricConfig::new(1.0).with_kernel_size(2)).is_err());
    assert!(std::panic::catch_unwind(|| PsnrMetricConfig::new(0.0)).is_err());
    assert!(
        std::panic::catch_unwind(|| DiceMetric::<TestBackend>::with_config(DiceMetricConfig {
            epsilon: f64::NAN,
            include_background: false
        }))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            let mut c = MsSsimMetricConfig::new(1.0);
            c.channels = 0;
            MsSsimMetric::<TestBackend>::new(c, &Default::default())
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| PsnrInput::new(
            image(vec![0.0], [1, 1, 1, 1]),
            image(vec![0.0; 2], [1, 1, 1, 2])
        ))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            let mut metric = MsSsimMetric::new(
                MsSsimMetricConfig::new(1.0).with_channels(1),
                &Default::default(),
            );
            let x = image(vec![0.0; 16], [1, 1, 4, 4]);
            metric.update(&MsSsimInput::new(x.clone(), x)).unwrap();
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            let device = Default::default();
            let mut metric = DiceMetric::<TestBackend>::with_config(DiceMetricConfig {
                include_background: true,
                ..Default::default()
            });
            metric
                .update(&DiceInput::new(
                    Tensor::zeros([1, 1, 2, 2], &device),
                    Tensor::zeros([1, 1, 2, 2], &device),
                ))
                .unwrap();
        })
        .is_err()
    );
}

struct Images {
    output: Tensor<TestBackend, 4>,
    target: Tensor<TestBackend, 4>,
}
impl Adaptor<PsnrInput<TestBackend>> for Images {
    fn adapt(&self) -> PsnrInput<TestBackend> {
        PsnrInput::new(self.output.clone(), self.target.clone())
    }
}
impl Adaptor<SsimInput<TestBackend>> for Images {
    fn adapt(&self) -> SsimInput<TestBackend> {
        SsimInput::new(self.output.clone(), self.target.clone())
    }
}
impl Adaptor<MsSsimInput<TestBackend>> for Images {
    fn adapt(&self) -> MsSsimInput<TestBackend> {
        MsSsimInput::new(self.output.clone(), self.target.clone())
    }
}
#[test]
fn vision_metrics_run_through_async_epoch_processor() {
    let processor = MetricsProcessor::<(), Images>::new()
        .metric_valid(PsnrMetric::new(PsnrMetricConfig::new(1.0)).with_name("psnr"))
        .metric_valid(
            SsimMetric::new(SsimMetricConfig::new(1.0).with_kernel_size(3)).with_name("ssim"),
        )
        .metric_valid(
            MsSsimMetric::new(
                MsSsimMetricConfig::new(1.0)
                    .with_kernel_size(3)
                    .with_channels(1)
                    .with_betas(vec![1.0]),
                &Default::default(),
            )
            .with_name("ms"),
        );
    let store = processor.store();
    let mut processor = AsyncProcessorTraining::new(processor);
    processor
        .process_valid(LearnerEvent::StartSplit {
            epoch: 1,
            total_items: 1,
        })
        .unwrap();
    let output = image(vec![0.25; 16], [1, 1, 4, 4]);
    processor
        .process_valid(LearnerEvent::ProcessedItem(TrainingItem {
            item: Images {
                output: output.clone(),
                target: output,
            },
            progress: ruda_model::data::dataloader::Progress {
                items_processed: 1,
                items_total: 1,
            },
            iteration: 1,
            learning_rate: None,
        }))
        .unwrap();
    processor
        .process_valid(LearnerEvent::EndSplit {
            epoch: 1,
            completed: true,
        })
        .unwrap();
    processor.flush().unwrap();
    let entries = store.metrics(1, Split::Valid);
    assert_eq!(entries.len(), 3);
    for entry in entries {
        near(
            entry.value.unwrap(),
            if entry.name == "psnr" { 100.0 } else { 1.0 },
        );
    }
}
