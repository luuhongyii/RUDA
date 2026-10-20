use super::{SimpleOptimizer, adaptor::OptimizerAdaptor, state_tensor};
use crate::{LearningRate, grad_clipping::GradientClippingConfig};
use ruda_model::{
    config::Config,
    module::AutodiffModule,
    record::{PrecisionSettings, Record},
    tensor::{
        DType, Tensor, TensorData,
        backend::{AutodiffBackend, Backend},
    },
};

/// Configuration for the [`Lion`] optimizer.
#[derive(Config, Debug)]
pub struct LionConfig {
    /// Interpolation factor used to compute the update direction.
    #[config(default = 0.9)]
    beta_1: f32,
    /// Decay factor for the momentum state.
    #[config(default = 0.99)]
    beta_2: f32,
    /// Decoupled weight decay factor.
    #[config(default = 0.0)]
    weight_decay: f32,
    /// Optional gradient clipping configuration.
    grad_clipping: Option<GradientClippingConfig>,
}

/// Lion sign-momentum optimizer with decoupled weight decay.
#[derive(Clone)]
pub struct Lion {
    beta_1: f32,
    beta_2: f32,
    weight_decay: f32,
}

/// Lion state for one parameter tensor.
#[derive(Clone, new)]
pub struct LionState<B: Backend, const D: usize> {
    /// Exponential moving average of gradients.
    pub momentum: Tensor<B, D>,
}

impl LionConfig {
    /// Build the per-parameter optimizer without applying gradient clipping.
    pub fn build(&self) -> Lion {
        Lion {
            beta_1: self.beta_1,
            beta_2: self.beta_2,
            weight_decay: self.weight_decay,
        }
    }

    /// Initialize a module optimizer with the configured gradient clipping.
    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(&self) -> OptimizerAdaptor<Lion, M, B> {
        let mut optimizer = OptimizerAdaptor::from(self.build());
        if let Some(config) = &self.grad_clipping {
            optimizer = optimizer.with_grad_clipping(config.init());
        }
        optimizer
    }
}

impl<B: Backend> SimpleOptimizer<B> for Lion {
    type State<const D: usize> = LionState<B, D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        let (update, momentum) = match state {
            Some(state) => {
                let update = state
                    .momentum
                    .clone()
                    .mul_scalar(self.beta_1)
                    .add(grad.clone().mul_scalar(1.0 - self.beta_1))
                    .sign();
                let momentum = state
                    .momentum
                    .mul_scalar(self.beta_2)
                    .add(grad.mul_scalar(1.0 - self.beta_2));
                (update, momentum)
            }
            None => (
                grad.clone().mul_scalar(1.0 - self.beta_1).sign(),
                grad.mul_scalar(1.0 - self.beta_2),
            ),
        };
        let decay = 1.0 - lr * self.weight_decay as f64;
        let tensor = if decay == 1.0 {
            tensor
        } else {
            tensor.mul_scalar(decay)
        };
        (
            tensor - update.mul_scalar(lr),
            Some(LionState::new(momentum)),
        )
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &B::Device) -> Self::State<D> {
        state.momentum = state.momentum.to_device(device);
        state
    }
}

impl<B: Backend, const D: usize> Record<B> for LionState<B, D> {
    fn map_parameters<M: ruda_model::module::ModuleMapper<B>>(mut self, mapper: &mut M) -> Self {
        self.momentum = mapper.map_tensor(self.momentum);
        self
    }
    type Item<P: PrecisionSettings> = (DType, TensorData);

    fn into_item<P: PrecisionSettings>(self) -> Self::Item<P> {
        state_tensor::into_item::<B, D, P>(self.momentum)
    }

    fn from_item<P: PrecisionSettings>(item: Self::Item<P>, device: &B::Device) -> Self {
        Self::new(state_tensor::from_item(item, device))
    }
}
