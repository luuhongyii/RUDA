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

/// Configuration for the [`Lamb`] optimizer.
#[derive(Config, Debug)]
pub struct LambConfig {
    /// Exponential decay rate for first moments.
    #[config(default = 0.9)]
    beta_1: f32,
    /// Exponential decay rate for second moments.
    #[config(default = 0.999)]
    beta_2: f32,
    /// Value added to the denominator for numerical stability.
    #[config(default = 1e-6)]
    epsilon: f32,
    /// Weight decay added to the Adam update before layer-wise adaptation.
    #[config(default = 0.0)]
    weight_decay: f32,
    /// Whether to apply the layer-wise trust ratio.
    #[config(default = true)]
    use_trust_ratio: bool,
    /// Optional gradient clipping configuration.
    grad_clipping: Option<GradientClippingConfig>,
}

/// Layer-wise Adaptive Moments optimizer.
#[derive(Clone)]
pub struct Lamb {
    beta_1: f32,
    beta_2: f32,
    epsilon: f32,
    weight_decay: f32,
    use_trust_ratio: bool,
}

/// LAMB state for one parameter tensor.
#[derive(Clone)]
pub struct LambState<B: Backend, const D: usize> {
    /// Number of updates applied to the parameter.
    pub time: usize,
    /// Exponential moving average of gradients.
    pub moment_1: Tensor<B, D>,
    /// Exponential moving average of squared gradients.
    pub moment_2: Tensor<B, D>,
}

impl LambConfig {
    /// Build the per-parameter optimizer without applying gradient clipping.
    pub fn build(&self) -> Lamb {
        Lamb {
            beta_1: self.beta_1,
            beta_2: self.beta_2,
            epsilon: self.epsilon,
            weight_decay: self.weight_decay,
            use_trust_ratio: self.use_trust_ratio,
        }
    }

    /// Initialize a module optimizer with the configured gradient clipping.
    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(&self) -> OptimizerAdaptor<Lamb, M, B> {
        let mut optimizer = OptimizerAdaptor::from(self.build());
        if let Some(config) = &self.grad_clipping {
            optimizer = optimizer.with_grad_clipping(config.init());
        }
        optimizer
    }
}

impl<B: Backend> SimpleOptimizer<B> for Lamb {
    type State<const D: usize> = LambState<B, D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        let factor_1 = 1.0 - self.beta_1;
        let factor_2 = 1.0 - self.beta_2;
        let state = match state {
            Some(mut state) => {
                state.moment_1 = state
                    .moment_1
                    .mul_scalar(self.beta_1)
                    .add(grad.clone().mul_scalar(factor_1));
                state.moment_2 = state
                    .moment_2
                    .mul_scalar(self.beta_2)
                    .add(grad.square().mul_scalar(factor_2));
                state.time += 1;
                state
            }
            None => LambState {
                time: 1,
                moment_1: grad.clone().mul_scalar(factor_1),
                moment_2: grad.square().mul_scalar(factor_2),
            },
        };
        let time = state.time as i32;
        let moment_1 = state
            .moment_1
            .clone()
            .div_scalar(1.0 - self.beta_1.powi(time));
        let moment_2 = state
            .moment_2
            .clone()
            .div_scalar(1.0 - self.beta_2.powi(time));
        let mut update = moment_1.div(moment_2.sqrt().add_scalar(self.epsilon));
        if self.weight_decay != 0.0 {
            update = update.add(tensor.clone().mul_scalar(self.weight_decay));
        }
        if self.use_trust_ratio {
            let parameter_norm = tensor.clone().square().sum().sqrt();
            let update_norm = update.clone().square().sum().sqrt();
            let valid_norms = parameter_norm
                .clone()
                .greater_elem(0.0)
                .bool_and(update_norm.clone().greater_elem(0.0));
            let min_positive = update
                .dtype()
                .finfo()
                .unwrap_or(FloatDType::F32.finfo())
                .min_positive;
            let ratio = parameter_norm.div(update_norm.clamp_min(min_positive));
            let trust_ratio = ratio.ones_like().mask_where(valid_norms, ratio);
            update = update.mul(trust_ratio.reshape([1; D]));
        }
        (tensor - update.mul_scalar(lr), Some(state))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &B::Device) -> Self::State<D> {
        state.moment_1 = state.moment_1.to_device(device);
        state.moment_2 = state.moment_2.to_device(device);
        state
    }
}

impl<B: Backend, const D: usize> Record<B> for LambState<B, D> {
    fn map_parameters<M: ruda_model::module::ModuleMapper<B>>(mut self, mapper: &mut M) -> Self {
        self.moment_1 = mapper.map_tensor(self.moment_1);
        self.moment_2 = mapper.map_tensor(self.moment_2);
        self
    }
    type Item<P: PrecisionSettings> = (usize, (DType, TensorData), (DType, TensorData));

    fn into_item<P: PrecisionSettings>(self) -> Self::Item<P> {
        (
            self.time,
            state_tensor::into_item::<B, D, P>(self.moment_1),
            state_tensor::into_item::<B, D, P>(self.moment_2),
        )
    }

    fn from_item<P: PrecisionSettings>(item: Self::Item<P>, device: &B::Device) -> Self {
        Self {
            time: item.0,
            moment_1: state_tensor::from_item(item.1, device),
            moment_2: state_tensor::from_item(item.2, device),
        }
    }
}
