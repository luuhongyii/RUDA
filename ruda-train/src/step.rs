use ruda_model::{
    module::{AutodiffModule, Module, Reparameterization, Reparameterized},
    tensor::backend::{AutodiffBackend, Backend},
};
use ruda_optim::{GradientsParams, LearningRate, MultiGradientsParams, Optimizer};

/// Model output and gradients from one forward/backward pass.
pub struct TrainOutput<T> {
    pub grads: GradientsParams,
    pub item: T,
}

impl<T> TrainOutput<T> {
    pub fn new<B: AutodiffBackend, M: AutodiffModule<B>>(
        module: &M,
        grads: B::Gradients,
        item: T,
    ) -> Self {
        Self {
            grads: GradientsParams::from_grads(grads, module),
            item,
        }
    }
}

/// The model controls its loss, backward pass and optional optimizer hook.
pub trait TrainStep<B: AutodiffBackend>: AutodiffModule<B> {
    type Input: Send + 'static;
    type Output: Send + 'static;

    fn step(&self, item: Self::Input) -> TrainOutput<Self::Output>;

    fn optimize<O: Optimizer<Self, B>>(
        self,
        optim: &mut O,
        lr: LearningRate,
        grads: GradientsParams,
    ) -> Self {
        optim.step(lr, self, grads)
    }
}

/// Implemented on the non-autodiff model used for validation or evaluation.
pub trait InferenceStep<B: Backend> {
    type Input: Send + 'static;
    type Output: Send + 'static;
    fn step(&self, item: Self::Input) -> Self::Output;
}

impl<B, M, R> TrainStep<B> for Reparameterized<M, R, B>
where
    B: AutodiffBackend,
    M: TrainStep<B>,
    R: Reparameterization<B> + AutodiffModule<B>,
    R::InnerModule: Reparameterization<B::InnerBackend>,
{
    type Input = M::Input;
    type Output = M::Output;

    fn step(&self, item: Self::Input) -> TrainOutput<Self::Output> {
        M::step(self, item)
    }

    fn optimize<O: Optimizer<Self, B>>(self, optim: &mut O, lr: LearningRate, grads: GradientsParams) -> Self {
        let module = (*self).clone();
        let mut bridge = ReparameterizedOptimizer { optim: optim.clone(), model: self };
        let module = module.optimize(&mut bridge, lr, grads);
        *optim = bridge.optim;
        bridge.model.with_module(module)
    }
}

#[derive(Clone)]
struct ReparameterizedOptimizer<O, M, R, B: Backend> {
    optim: O,
    model: Reparameterized<M, R, B>,
}

impl<B, M, R, O> Optimizer<M, B> for ReparameterizedOptimizer<O, M, R, B>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    R: Reparameterization<B> + AutodiffModule<B>,
    R::InnerModule: Reparameterization<B::InnerBackend>,
    O: Optimizer<Reparameterized<M, R, B>, B>,
{
    type Record = O::Record;

    fn step(&mut self, lr: LearningRate, module: M, grads: GradientsParams) -> M {
        self.model = self.optim.step(lr, self.model.clone().with_module(module), grads);
        (*self.model).clone()
    }

    fn step_multi(&mut self, lr: LearningRate, module: M, grads: MultiGradientsParams) -> M {
        self.model = self.optim.step_multi(lr, self.model.clone().with_module(module), grads);
        (*self.model).clone()
    }

    fn to_record(&self) -> Self::Record { self.optim.to_record() }

    fn load_record(mut self, record: Self::Record) -> Self {
        self.optim = self.optim.load_record(record);
        self
    }
}

impl<B, M, R> InferenceStep<B> for Reparameterized<M, R, B>
where B: Backend, M: Module<B> + InferenceStep<B>, R: Reparameterization<B> {
    type Input = M::Input;
    type Output = M::Output;
    fn step(&self, item: Self::Input) -> Self::Output { M::step(self, item) }
}

pub struct RegressionOutput<B: Backend> {
    pub loss: ruda_model::Tensor<B, 1>,
    pub output: ruda_model::Tensor<B, 2>,
    pub targets: ruda_model::Tensor<B, 2>,
}
impl<B: Backend> crate::metric::Adaptor<crate::metric::LossInput<B>> for RegressionOutput<B> {
    fn adapt(&self) -> crate::metric::LossInput<B> {
        crate::metric::LossInput {
            tensor: self.loss.clone(),
        }
    }
}

pub struct ClassificationOutput<B: Backend> {
    pub loss: ruda_model::Tensor<B, 1>,
    pub output: ruda_model::Tensor<B, 2>,
    pub targets: ruda_model::Tensor<B, 1, ruda_model::tensor::Int>,
}
impl<B: Backend> crate::metric::Adaptor<crate::metric::LossInput<B>> for ClassificationOutput<B> {
    fn adapt(&self) -> crate::metric::LossInput<B> {
        crate::metric::LossInput {
            tensor: self.loss.clone(),
        }
    }
}
impl<B: Backend> crate::metric::Adaptor<crate::metric::AccuracyInput<B>>
    for ClassificationOutput<B>
{
    fn adapt(&self) -> crate::metric::AccuracyInput<B> {
        crate::metric::AccuracyInput {
            predictions: self.output.clone(),
            targets: self.targets.clone(),
        }
    }
}
