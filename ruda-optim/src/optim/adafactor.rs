use super::{SimpleOptimizer, adaptor::OptimizerAdaptor, state_tensor};
use crate::{LearningRate, grad_clipping::GradientClippingConfig};
use ruda_model::{
    config::Config,
    module::AutodiffModule,
    record::{PrecisionSettings, Record},
    tensor::{
        DType, FloatDType, Tensor, TensorData,
        backend::{AutodiffBackend, Backend},
    },
};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// Configuration for [`Adafactor`].
///
/// By default the supplied learning rate caps the relative step at `1 / sqrt(step)`,
/// then the parameter RMS scales the update. Disable both `relative_step` and
/// `scale_parameter` to use an absolute learning rate from an external scheduler.
#[derive(Config, Debug)]
pub struct AdafactorConfig {
    /// Positive constant added to squared gradients.
    #[config(default = 1e-30)]
    epsilon_1: f32,
    /// Positive lower bound on the parameter RMS.
    #[config(default = 1e-3)]
    epsilon_2: f32,
    /// Positive upper bound on update RMS, before learning rate scaling.
    #[config(default = 1.0)]
    clip_threshold: f32,
    /// Negative exponent in `beta_2 = 1 - step^decay_rate`.
    #[config(default = -0.8)]
    decay_rate: f64,
    /// Whether to cap the supplied learning rate by `1 / sqrt(step)`.
    #[config(default = true)]
    relative_step: bool,
    /// Whether to multiply the step size by `max(epsilon_2, RMS(parameter))`.
    #[config(default = true)]
    scale_parameter: bool,
    /// Decoupled weight decay coefficient, multiplied by the supplied learning rate.
    #[config(default = 0.0)]
    weight_decay: f32,
    /// Optional gradient clipping applied before the Adafactor update.
    grad_clipping: Option<GradientClippingConfig>,
}

/// Adafactor with factored second moments and update clipping.
///
/// Matrices factor their last two dimensions independently for each leading index.
/// Vectors keep full second moments. Half-precision parameters use FP32 updates and state.
#[derive(Clone)]
pub struct Adafactor {
    epsilon_1: f32,
    epsilon_2: f32,
    clip_threshold: f32,
    decay_rate: f64,
    relative_step: bool,
    scale_parameter: bool,
    weight_decay: f32,
}

/// Adafactor state for one parameter tensor.
#[derive(Clone)]
pub struct AdafactorState<B: Backend, const D: usize> {
    /// Number of updates applied to this parameter.
    pub time: usize,
    /// Row second moment `[..., rows, 1]`, or a full second moment for vectors.
    pub second_moment: Tensor<B, D>,
    /// Column second moment `[..., 1, columns]`, absent for vectors.
    pub column_second_moment: Option<Tensor<B, D>>,
}

impl AdafactorConfig {
    /// Build the per-parameter optimizer without applying gradient clipping.
    ///
    /// Panics for nonpositive/nonfinite epsilons or clipping threshold, a nonnegative
    /// or nonfinite decay exponent, or negative/nonfinite weight decay.
    pub fn build(&self) -> Adafactor {
        assert!(
            self.epsilon_1.is_finite() && self.epsilon_1 > 0.0,
            "Adafactor epsilon_1 must be positive and finite"
        );
        assert!(
            self.epsilon_2.is_finite() && self.epsilon_2 > 0.0,
            "Adafactor epsilon_2 must be positive and finite"
        );
        assert!(
            self.clip_threshold.is_finite() && self.clip_threshold > 0.0,
            "Adafactor clip_threshold must be positive and finite"
        );
        assert!(
            self.decay_rate.is_finite() && self.decay_rate < 0.0,
            "Adafactor decay_rate must be negative and finite"
        );
        assert!(
            self.weight_decay.is_finite() && self.weight_decay >= 0.0,
            "Adafactor weight_decay must be nonnegative and finite"
        );
        Adafactor {
            epsilon_1: self.epsilon_1,
            epsilon_2: self.epsilon_2,
            clip_threshold: self.clip_threshold,
            decay_rate: self.decay_rate,
            relative_step: self.relative_step,
            scale_parameter: self.scale_parameter,
            weight_decay: self.weight_decay,
        }
    }

    /// Initialize a module optimizer with the configured gradient clipping.
    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(
        &self,
    ) -> OptimizerAdaptor<Adafactor, M, B> {
        let mut optimizer = OptimizerAdaptor::from(self.build());
        if let Some(config) = &self.grad_clipping {
            optimizer = optimizer.with_grad_clipping(config.init());
        }
        optimizer
    }
}

impl<B: Backend> SimpleOptimizer<B> for Adafactor {
    type State<const D: usize> = AdafactorState<B, D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        let dtype = tensor.dtype();
        let (tensor, grad) = match dtype {
            DType::F16 | DType::BF16 => (tensor.cast(FloatDType::F32), grad.cast(FloatDType::F32)),
            _ => (tensor, grad),
        };
        let squared_grad = grad.clone().square().add_scalar(self.epsilon_1);
        let (second_moment, column_second_moment) = if D >= 2 {
            (
                squared_grad.clone().mean_dim(D - 1),
                Some(squared_grad.mean_dim(D - 2)),
            )
        } else {
            (squared_grad, None)
        };
        let state = match state {
            Some(mut state) => {
                state.time += 1;
                let new_weight = (state.time as f64).powf(self.decay_rate);
                let old_weight = 1.0 - new_weight;
                state.second_moment = state
                    .second_moment
                    .mul_scalar(old_weight)
                    .add(second_moment.mul_scalar(new_weight));
                state.column_second_moment = column_second_moment.map(|column| {
                    state
                        .column_second_moment
                        .take()
                        .expect("Factored Adafactor state must contain a column second moment")
                        .mul_scalar(old_weight)
                        .add(column.mul_scalar(new_weight))
                });
                state
            }
            None => AdafactorState {
                time: 1,
                second_moment,
                column_second_moment,
            },
        };
        let mut update = match &state.column_second_moment {
            Some(column) => {
                let row = state.second_moment.clone();
                let row_scale = row.clone().div(row.mean_dim(D - 2)).sqrt();
                grad.div(row_scale).div(column.clone().sqrt())
            }
            None => grad.div(state.second_moment.clone().sqrt()),
        };
        let clipping = rms(update.clone())
            .div_scalar(self.clip_threshold)
            .clamp_min(1.0);
        update = update.div(clipping.reshape([1; D]));
        let step_size = if self.relative_step {
            lr.min(1.0 / (state.time as f64).sqrt())
        } else {
            lr
        };
        if self.scale_parameter {
            let scale = rms(tensor.clone()).clamp_min(self.epsilon_2);
            update = update.mul(scale.reshape([1; D]));
        }
        update = update.mul_scalar(step_size);
        let tensor = if self.weight_decay == 0.0 {
            tensor
        } else {
            tensor.mul_scalar(1.0 - lr * self.weight_decay as f64)
        };
        let tensor = tensor - update;
        let tensor = match dtype {
            DType::F16 | DType::BF16 => tensor.cast(FloatDType::from(dtype)),
            _ => tensor,
        };
        (tensor, Some(state))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &B::Device) -> Self::State<D> {
        state.second_moment = state.second_moment.to_device(device);
        state.column_second_moment = state
            .column_second_moment
            .map(|column| column.to_device(device));
        state
    }
}

fn rms<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, 1> {
    tensor.square().mean().sqrt()
}

impl<B: Backend, const D: usize> Record<B> for AdafactorState<B, D> {
    fn map_parameters<M: ruda_model::module::ModuleMapper<B>>(mut self, mapper: &mut M) -> Self {
        self.second_moment = mapper.map_tensor(self.second_moment);
        self.column_second_moment = self.column_second_moment.map(|value| mapper.map_tensor(value));
        self
    }
    type Item<P: PrecisionSettings> = (usize, (DType, TensorData), Option<(DType, TensorData)>);

    fn into_item<P: PrecisionSettings>(self) -> Self::Item<P> {
        (
            self.time,
            state_tensor::into_item::<B, D, P>(self.second_moment),
            self.column_second_moment
                .map(state_tensor::into_item::<B, D, P>),
        )
    }

    fn from_item<P: PrecisionSettings>(item: Self::Item<P>, device: &B::Device) -> Self {
        Self {
            time: item.0,
            second_moment: state_tensor::from_item(item.1, device),
            column_second_moment: item.2.map(|column| state_tensor::from_item(column, device)),
        }
    }
}
