use super::{Metric, NumericEntry, NumericMetricState};
use crate::TrainingError;
use ruda_model::tensor::{
    ElementConversion, Int, Tensor,
    backend::Backend,
    module::{avg_pool2d, conv2d},
    ops::{ConvOptions, PadMode},
};
use std::marker::PhantomData;

mod afine;
mod dists;
mod fid;
mod lpips;

#[cfg(test)]
mod pretrained_tests;

pub use afine::*;
pub use dists::*;
pub use fid::*;
pub use lpips::*;

macro_rules! image_input {
    ($name:ident) => {
        pub struct $name<B: Backend> {
            outputs: Tensor<B, 4>,
            targets: Tensor<B, 4>,
        }
        impl<B: Backend> $name<B> {
            pub fn new(outputs: Tensor<B, 4>, targets: Tensor<B, 4>) -> Self {
                assert_eq!(outputs.dims(), targets.dims(), "Image shape mismatch");
                Self { outputs, targets }
            }
        }
    };
}
image_input!(PsnrInput);
image_input!(SsimInput);
image_input!(MsSsimInput);

fn scalar<B: Backend>(tensor: Tensor<B, 1>) -> Result<f64, TrainingError> {
    tensor
        .try_into_scalar()
        .map(|x| x.elem::<f64>())
        .map_err(|error| TrainingError(error.to_string()))
}

#[derive(Clone, Copy, Debug)]
pub struct PsnrMetricConfig {
    pub max_pixel_val: f64,
    pub epsilon: f64,
}
impl PsnrMetricConfig {
    pub fn new(max_pixel_val: f64) -> Self {
        assert!(max_pixel_val > 0.0, "max_pixel_val must be positive");
        Self {
            max_pixel_val,
            epsilon: 1e-10,
        }
    }
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        assert!(epsilon > 0.0, "epsilon must be positive");
        self.epsilon = epsilon;
        self
    }
}

#[derive(Clone)]
pub struct PsnrMetric<B: Backend> {
    name: String,
    state: NumericMetricState,
    config: PsnrMetricConfig,
    backend: PhantomData<B>,
}
impl<B: Backend> PsnrMetric<B> {
    pub fn new(config: PsnrMetricConfig) -> Self {
        Self {
            name: format!("PSNR@{}", config.max_pixel_val),
            state: Default::default(),
            config,
            backend: PhantomData,
        }
    }
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.into();
        self
    }
}
impl<B: Backend> Metric for PsnrMetric<B> {
    numeric_state_metric!(Some("dB"), true, 2);
    type Input = PsnrInput<B>;
    fn name(&self) -> &str {
        &self.name
    }
    fn update(&mut self, item: &Self::Input) -> Result<(), TrainingError> {
        let batch = item.outputs.dims()[0];
        let mse = (item.outputs.clone() - item.targets.clone())
            .powi_scalar(2)
            .mean_dims(&[1, 2, 3])
            .flatten::<1>(0, 3)
            .clamp_min(self.config.epsilon);
        let psnr = (mse.recip() * (self.config.max_pixel_val * self.config.max_pixel_val)).log()
            * (10.0 / std::f64::consts::LN_10);
        self.state.update(scalar(psnr.mean())?, batch);
        Ok(())
    }
    fn value(&self) -> Option<f64> {
        self.state.value()
    }
    fn clear(&mut self) {
        self.state.clear();
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SsimMetricConfig {
    pub pixel_range: f32,
    pub k1: f32,
    pub k2: f32,
    pub kernel_size: usize,
    pub sigma: f32,
}
impl SsimMetricConfig {
    pub fn new(pixel_range: f32) -> Self {
        assert!(pixel_range > 0.0, "pixel_range must be positive");
        Self {
            pixel_range,
            k1: 0.01,
            k2: 0.03,
            kernel_size: 11,
            sigma: 1.5,
        }
    }
    pub fn with_k1_k2(mut self, k1: f32, k2: f32) -> Self {
        assert!(k1 > 0.0 && k2 > 0.0, "k1 and k2 must be positive");
        self.k1 = k1;
        self.k2 = k2;
        self
    }
    pub fn with_kernel_size(mut self, size: usize) -> Self {
        assert!(
            size > 0 && size % 2 == 1,
            "kernel_size must be positive and odd"
        );
        self.kernel_size = size;
        self
    }
    pub fn with_sigma(mut self, sigma: f32) -> Self {
        assert!(sigma > 0.0, "sigma must be positive");
        self.sigma = sigma;
        self
    }
}

fn gaussian_conv<B: Backend>(
    input: Tensor<B, 4>,
    horizontal: Tensor<B, 4>,
    vertical: Tensor<B, 4>,
    padding: usize,
) -> Tensor<B, 4> {
    let channels = input.dims()[1];
    let x = conv2d(
        input,
        horizontal,
        None,
        ConvOptions::new([1, 1], [0, padding], [1, 1], channels),
    );
    conv2d(
        x,
        vertical,
        None,
        ConvOptions::new([1, 1], [padding, 0], [1, 1], channels),
    )
}

#[derive(Clone)]
pub struct SsimMetric<B: Backend> {
    name: String,
    state: NumericMetricState,
    config: SsimMetricConfig,
    backend: PhantomData<B>,
}
impl<B: Backend> SsimMetric<B> {
    pub fn new(config: SsimMetricConfig) -> Self {
        Self {
            name: format!(
                "SSIM (dr={}, w={}, σ={})",
                config.pixel_range, config.kernel_size, config.sigma
            ),
            state: Default::default(),
            config,
            backend: PhantomData,
        }
    }
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.into();
        self
    }
}
impl<B: Backend> Metric for SsimMetric<B> {
    numeric_state_metric!(None, true, 4);
    type Input = SsimInput<B>;
    fn name(&self) -> &str {
        &self.name
    }
    fn update(&mut self, item: &Self::Input) -> Result<(), TrainingError> {
        let [batch, channels, height, width] = item.outputs.dims();
        let config = self.config;
        let size = config.kernel_size;
        assert!(
            height >= size && width >= size,
            "Image dimensions must be >= kernel_size"
        );
        let mut kernel = vec![0.0f32; size];
        let mut sum = 0.0f32;
        for (i, v) in kernel.iter_mut().enumerate() {
            let x = i as f32 - (size / 2) as f32;
            *v = (-(x * x) / (2.0 * config.sigma * config.sigma)).exp();
            sum += *v;
        }
        for v in &mut kernel {
            *v /= sum;
        }
        let kernel = Tensor::<B, 1>::from_floats(kernel.as_slice(), &item.outputs.device());
        let horizontal = kernel
            .clone()
            .reshape([1, 1, 1, size])
            .repeat_dim(0, channels);
        let vertical = kernel.reshape([1, 1, size, 1]).repeat_dim(0, channels);
        let conv = |input| gaussian_conv(input, horizontal.clone(), vertical.clone(), size / 2);
        let x = item.outputs.clone();
        let y = item.targets.clone();
        let mu_x = conv(x.clone());
        let mu_y = conv(y.clone());
        let xy = mu_x.clone() * mu_y.clone();
        let xx = mu_x.clone() * mu_x;
        let yy = mu_y.clone() * mu_y;
        let vx = (conv(x.clone() * x.clone()) - xx.clone()).clamp_min(0.0);
        let vy = (conv(y.clone() * y.clone()) - yy.clone()).clamp_min(0.0);
        let cov = conv(x * y) - xy.clone();
        let c1 = (config.k1 * config.pixel_range).powi(2);
        let c2 = (config.k2 * config.pixel_range).powi(2);
        let ssim = ((xy * 2.0f32 + c1) * (cov * 2.0f32 + c2)) / ((xx + yy + c1) * (vx + vy + c2));
        self.state
            .update(scalar(ssim.mean_dims(&[1, 2, 3]).mean())?, batch);
        Ok(())
    }
    fn value(&self) -> Option<f64> {
        self.state.value()
    }
    fn clear(&mut self) {
        self.state.clear();
    }
}

#[derive(Clone, Debug)]
pub struct MsSsimMetricConfig {
    pub k1: f32,
    pub k2: f32,
    pub pixel_range: f32,
    pub kernel_size: usize,
    pub sigma: f32,
    pub channels: usize,
    pub betas: Vec<f32>,
}
impl MsSsimMetricConfig {
    pub fn new(pixel_range: f32) -> Self {
        let config = Self {
            pixel_range,
            k1: 0.01,
            k2: 0.03,
            kernel_size: 11,
            sigma: 1.5,
            channels: 3,
            betas: vec![0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
        };
        config.validate();
        config
    }
    pub fn with_k1_k2(mut self, k1: f32, k2: f32) -> Self {
        self.k1 = k1;
        self.k2 = k2;
        self.validate();
        self
    }
    pub fn with_kernel_size(mut self, size: usize) -> Self {
        self.kernel_size = size;
        self.validate();
        self
    }
    pub fn with_sigma(mut self, sigma: f32) -> Self {
        self.sigma = sigma;
        self.validate();
        self
    }
    pub fn with_channels(mut self, channels: usize) -> Self {
        self.channels = channels;
        self.validate();
        self
    }
    pub fn with_betas(mut self, betas: Vec<f32>) -> Self {
        self.betas = betas;
        self.validate();
        self
    }
    fn validate(&self) {
        for value in [self.pixel_range, self.k1, self.k2, self.sigma] {
            assert!(
                value.is_finite() && value > 0.0,
                "MS-SSIM constants must be finite and positive"
            );
        }
        assert!(
            self.kernel_size > 0 && self.kernel_size % 2 == 1,
            "kernel_size must be positive and odd"
        );
        assert!(self.channels > 0, "channels must be positive");
        assert!(!self.betas.is_empty(), "betas cannot be empty");
        assert!(
            self.betas.iter().all(|b| b.is_finite() && *b >= 0.0),
            "betas must be finite and nonnegative"
        );
        assert!(
            self.betas.iter().any(|b| *b > 0.0),
            "At least one beta must be positive"
        );
    }
}

#[derive(Clone)]
pub struct MsSsimMetric<B: Backend> {
    name: String,
    state: NumericMetricState,
    config: MsSsimMetricConfig,
    horizontal: Tensor<B, 4>,
    vertical: Tensor<B, 4>,
}
impl<B: Backend> MsSsimMetric<B> {
    pub fn new(config: MsSsimMetricConfig, device: &B::Device) -> Self {
        config.validate();
        let size = config.kernel_size;
        let x = Tensor::<B, 1, Int>::arange(0..size as i64, device).float() - (size / 2) as f32;
        let kernel = ((x.clone() * x) / (2.0 * config.sigma * config.sigma))
            .neg()
            .exp();
        let kernel = kernel.clone() / kernel.sum();
        let horizontal = kernel
            .clone()
            .reshape([1, 1, 1, size])
            .repeat_dim(0, config.channels);
        let vertical = kernel
            .reshape([1, 1, size, 1])
            .repeat_dim(0, config.channels);
        Self {
            name: format!(
                "MS-SSIM (pr={}, k={}, σ={})",
                config.pixel_range, size, config.sigma
            ),
            state: Default::default(),
            config,
            horizontal,
            vertical,
        }
    }
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.into();
        self
    }
    fn conv(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let p = self.config.kernel_size / 2;
        gaussian_conv(
            input.pad((p, p, p, p), PadMode::Reflect),
            self.horizontal.clone(),
            self.vertical.clone(),
            0,
        )
    }
}
impl<B: Backend> Metric for MsSsimMetric<B> {
    numeric_state_metric!(None, true, 4);
    type Input = MsSsimInput<B>;
    fn name(&self) -> &str {
        &self.name
    }
    fn update(&mut self, item: &Self::Input) -> Result<(), TrainingError> {
        let [batch, channels, height, width] = item.outputs.dims();
        assert_eq!(channels, self.config.channels, "MS-SSIM channel mismatch");
        let scales = self.config.betas.len();
        let min_size = self.config.kernel_size * 2usize.pow(scales.saturating_sub(1) as u32);
        assert!(
            height >= min_size && width >= min_size,
            "Image too small for MS-SSIM scales"
        );
        let c1 = (self.config.k1 * self.config.pixel_range).powi(2);
        let c2 = (self.config.k2 * self.config.pixel_range).powi(2);
        let mut x = item.outputs.clone();
        let mut y = item.targets.clone();
        let mut product = Tensor::<B, 2>::ones([batch, channels], &item.outputs.device());
        for (scale, beta) in self.config.betas.iter().enumerate() {
            let mx = self.conv(x.clone());
            let my = self.conv(y.clone());
            let xx = mx.clone() * mx.clone();
            let yy = my.clone() * my.clone();
            let vx = (self.conv(x.clone() * x.clone()) - xx.clone()).clamp_min(0.0);
            let vy = (self.conv(y.clone() * y.clone()) - yy.clone()).clamp_min(0.0);
            let cov = self.conv(x.clone() * y.clone()) - mx.clone() * my.clone();
            let cs = (cov * 2.0 + c2) / (vx + vy + c2);
            if scale == scales - 1 {
                let luminance: Tensor<B, 4> = (2 * mx * my + c1) / (xx + yy + c1);
                let mean = (luminance * cs)
                    .mean_dims(&[2, 3])
                    .reshape([batch, channels]);
                product = product * mean.clamp_min(0.0).powf_scalar(*beta);
            } else {
                let mean = cs.mean_dims(&[2, 3]).reshape([batch, channels]);
                product = product * mean.clamp_min(0.0).powf_scalar(*beta);
                x = avg_pool2d(x, [2, 2], [2, 2], [0, 0], false, false);
                y = avg_pool2d(y, [2, 2], [2, 2], [0, 0], false, false);
            }
        }
        self.state
            .update(scalar(product.mean_dim(1).mean())?, batch);
        Ok(())
    }
    fn value(&self) -> Option<f64> {
        self.state.value()
    }
    fn clear(&mut self) {
        self.state.clear();
    }
}

pub struct DiceInput<B: Backend, const D: usize = 4> {
    outputs: Tensor<B, D, Int>,
    targets: Tensor<B, D, Int>,
}
impl<B: Backend, const D: usize> DiceInput<B, D> {
    pub fn new(outputs: Tensor<B, D, Int>, targets: Tensor<B, D, Int>) -> Self {
        assert!(D >= 3, "DiceInput requires at least 3 dimensions");
        assert_eq!(outputs.dims(), targets.dims(), "Dice shape mismatch");
        Self { outputs, targets }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct DiceMetricConfig {
    pub epsilon: f64,
    pub include_background: bool,
}
impl Default for DiceMetricConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-7,
            include_background: false,
        }
    }
}
#[derive(Clone)]
pub struct DiceMetric<B: Backend, const D: usize = 4> {
    name: String,
    config: DiceMetricConfig,
    numerator: f64,
    denominator: f64,
    count: usize,
    current: f64,
    current_count: usize,
    backend: PhantomData<B>,
}
impl<B: Backend, const D: usize> Default for DiceMetric<B, D> {
    fn default() -> Self {
        Self::new()
    }
}
impl<B: Backend, const D: usize> DiceMetric<B, D> {
    pub fn new() -> Self {
        Self::with_config(DiceMetricConfig::default())
    }
    pub fn with_config(config: DiceMetricConfig) -> Self {
        assert!(D >= 3, "DiceMetric requires at least 3 dimensions");
        assert!(
            config.epsilon.is_finite() && config.epsilon > 0.0,
            "epsilon must be finite and positive"
        );
        Self {
            name: format!("{D}D Dice Metric"),
            config,
            numerator: 0.0,
            denominator: 0.0,
            count: 0,
            current: f64::NAN,
            current_count: 0,
            backend: PhantomData,
        }
    }
}
impl<B: Backend, const D: usize> Metric for DiceMetric<B, D> {
    numeric_attributes!(None, true, 4);
    fn current_value(&self) -> Option<NumericEntry> {
        Some(NumericEntry::Aggregated { aggregated_value: self.current, count: self.current_count })
    }
    fn running_value(&self) -> Option<NumericEntry> {
        Some(NumericEntry::Aggregated {
            aggregated_value: (self.numerator + self.config.epsilon) / (self.denominator + self.config.epsilon),
            count: self.count,
        })
    }
    fn final_value(&self) -> NumericEntry {
        NumericEntry::Final((self.numerator + self.config.epsilon) / (self.denominator + self.config.epsilon))
    }
    type Input = DiceInput<B, D>;
    fn name(&self) -> &str {
        &self.name
    }
    fn update(&mut self, item: &Self::Input) -> Result<(), TrainingError> {
        let dims = item.outputs.dims();
        let mut outputs = item.outputs.clone();
        let mut targets = item.targets.clone();
        if !self.config.include_background && dims[1] > 1 {
            let ranges: [_; D] = std::array::from_fn(|axis| {
                if axis == 1 {
                    1..dims[axis]
                } else {
                    0..dims[axis]
                }
            });
            outputs = outputs.slice(ranges.clone());
            targets = targets.slice(ranges);
        } else if self.config.include_background && dims[1] < 2 {
            panic!("Dice metric requires at least 2 classes when including background");
        }
        let read = |tensor: Tensor<B, 1, Int>| {
            tensor
                .try_into_scalar()
                .map(|value| value.elem::<f64>())
                .map_err(|error| TrainingError(error.to_string()))
        };
        let intersection = read((outputs.clone() * targets.clone()).sum())?;
        let outputs_sum = read(outputs.sum())?;
        let targets_sum = read(targets.sum())?;
        self.numerator += 2.0 * intersection;
        self.denominator += outputs_sum + targets_sum;
        self.count += dims[0];
        self.current = (2.0 * intersection + self.config.epsilon)
            / (outputs_sum + targets_sum + self.config.epsilon);
        self.current_count = dims[0];
        Ok(())
    }
    fn value(&self) -> Option<f64> {
        (self.count > 0).then(|| {
            (self.numerator + self.config.epsilon) / (self.denominator + self.config.epsilon)
        })
    }
    fn clear(&mut self) {
        self.numerator = 0.0;
        self.denominator = 0.0;
        self.count = 0;
        self.current = f64::NAN;
        self.current_count = 0;
    }
}
