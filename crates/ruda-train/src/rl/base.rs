use super::TransitionBatch;
use crate::{
    ItemLazy, TrainingError,
    metric::{Adaptor, CumulativeRewardInput, EpisodeLengthInput},
};
use ruda_model::tensor::{Int, Tensor, backend::Backend};

pub struct StepResult<S> {
    pub next_state: S,
    pub reward: f64,
    pub done: bool,
    pub truncated: bool,
}
pub trait Environment {
    type State;
    type Action;
    const MAX_STEPS: usize;
    fn state(&self) -> Self::State;
    fn step(&mut self, action: Self::Action) -> StepResult<Self::State>;
    fn reset(&mut self);
}
pub trait EnvironmentInit<E: Environment>: Clone {
    fn init(&self) -> E;
}
impl<E: Environment, F: Fn() -> E + Clone> EnvironmentInit<E> for F {
    fn init(&self) -> E {
        self()
    }
}
pub trait ToObservation<B: Backend, O> {
    fn to_observation(&self, device: &B::Device) -> O;
}
pub trait ToAction<B: Backend, A> {
    fn to_action(&self, device: &B::Device) -> A;
}
pub trait PolicyState: Clone + Send + 'static {
    type Record: Send + 'static;
    fn into_record(self) -> Self::Record;
    fn load_record(&self, record: Self::Record) -> Self;
}
pub trait Batchable: Sized {
    fn batch(items: Vec<Self>) -> Self;
    fn unbatch(self) -> Vec<Self>;
}
macro_rules! tensor_batchable {
    ($($kind:ty),*) => {$ (
        impl<B: Backend, const D: usize> Batchable for Tensor<B, D, $kind> {
            fn batch(items: Vec<Self>) -> Self { Tensor::cat(items, 0) }
            fn unbatch(self) -> Vec<Self> {
                let dims = self.dims();
                (0..dims[0]).map(|i| self.clone().slice(core::array::from_fn::<_, D, _>(|d| {
                    if d == 0 { i..i+1 } else { 0..dims[d] }
                }))).collect()
            }
        }
        impl<B: Backend, const D: usize> ToObservation<B, Self> for Tensor<B, D, $kind> {
            fn to_observation(&self, device: &B::Device) -> Self { self.clone().to_device(device) }
        }
        impl<B: Backend, const D: usize> ToAction<B, Self> for Tensor<B, D, $kind> {
            fn to_action(&self, device: &B::Device) -> Self { self.clone().to_device(device) }
        }
    )*};
}
tensor_batchable!(ruda_model::tensor::Float, Int);

pub trait Policy<B: Backend>: Clone + Send + 'static {
    type Observation: Clone + Send + Batchable + 'static;
    type ActionDistribution: Clone + Send + Batchable + 'static;
    type Action: Clone + Send + Batchable + 'static;
    type ActionContext: Clone + Send + ItemLazy + 'static;
    type PolicyState: PolicyState;
    fn forward(&mut self, observation: Self::Observation) -> Self::ActionDistribution;
    fn action(
        &mut self,
        observation: Self::Observation,
        deterministic: bool,
    ) -> (Self::Action, Vec<Self::ActionContext>);
    fn update(&mut self, state: Self::PolicyState);
    fn state(&self) -> Self::PolicyState;
    fn to_device(self, device: &B::Device) -> Self;
    fn load_record(self, record: <Self::PolicyState as PolicyState>::Record) -> Self;
}
pub struct RLTrainOutput<T, P> {
    pub policy: P,
    pub item: T,
}
pub type LearnerTransitionBatch<B, P> =
    TransitionBatch<B, <P as Policy<B>>::Observation, <P as Policy<B>>::Action>;
pub trait PolicyLearner<B: Backend>: Send + 'static {
    type TrainContext: ItemLazy;
    type InnerPolicy: Policy<B>;
    type Record: Send + 'static;
    fn train(
        &mut self,
        batch: LearnerTransitionBatch<B, Self::InnerPolicy>,
    ) -> RLTrainOutput<Self::TrainContext, <Self::InnerPolicy as Policy<B>>::PolicyState>;
    fn policy(&self) -> Self::InnerPolicy;
    fn update_policy(&mut self, policy: Self::InnerPolicy);
    fn record(&self) -> Self::Record;
    fn load_record(self, record: Self::Record) -> Self;
    fn device(&self) -> B::Device;
}
#[derive(Clone, Debug)]
pub struct EpisodeSummary {
    pub episode_length: usize,
    pub cum_reward: f64,
}
impl ItemLazy for EpisodeSummary {
    type Synced = Self;
    fn sync(self) -> Result<Self, TrainingError> {
        Ok(self)
    }
}
impl Adaptor<CumulativeRewardInput> for EpisodeSummary {
    fn adapt(&self) -> CumulativeRewardInput {
        CumulativeRewardInput::new(self.cum_reward)
    }
}
impl Adaptor<EpisodeLengthInput> for EpisodeSummary {
    fn adapt(&self) -> EpisodeLengthInput {
        EpisodeLengthInput::new(self.episode_length as f64)
    }
}
