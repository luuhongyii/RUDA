use super::{
    AgentEnvAsyncLoop, AgentEnvLoop, AgentEvaluationEvent, AsyncAgentEnvLoopConfig, AsyncPolicy,
    AsyncRLProcessor, Environment, EnvironmentInit, EvaluationItem, LazyRLProcessor,
    MultiAgentEnvLoop, Policy, PolicyLearner, RLCheckpointer, RLEvent, RLEventProcessor,
    RLMetricsProcessor, SliceAccess, ToAction, ToObservation, TransitionBuffer,
};
use crate::{
    Interrupter, Interruption, ItemLazy, LearnerSummary, LearnerSummaryConfig, TrainingError,
    metric::EventStore,
};
use ruda_model::{
    data::dataloader::Progress,
    tensor::{ElementConversion, backend::Backend},
};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OffPolicyConfig {
    pub num_envs: usize,
    pub autobatch_size: usize,
    pub replay_buffer_size: usize,
    pub train_interval: usize,
    pub train_steps: usize,
    pub eval_interval: usize,
    pub eval_episodes: usize,
    pub train_batch_size: usize,
    pub warmup_steps: usize,
}
impl Default for OffPolicyConfig {
    fn default() -> Self {
        Self {
            num_envs: 1,
            autobatch_size: 1,
            replay_buffer_size: 1024,
            train_interval: 1,
            train_steps: 1,
            eval_interval: 10_000,
            eval_episodes: 1,
            train_batch_size: 32,
            warmup_steps: 0,
        }
    }
}
impl OffPolicyConfig {
    pub fn new() -> Self {
        Self::default()
    }
}

pub struct RLTrainingContext<B: Backend, L: PolicyLearner<B>> {
    pub num_steps: usize,
    pub starting_step: usize,
    pub inference_device: B::Device,
    pub interrupter: Interrupter,
    pub processor:
        Box<dyn RLEventProcessor<L::TrainContext, <L::InnerPolicy as Policy<B>>::ActionContext>>,
    pub store: EventStore,
    pub checkpointer: Option<RLCheckpointer<B, L>>,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RLLoopState {
    pub steps_processed: usize,
    pub optimizer_steps: usize,
    pub evaluations: usize,
    pub last_checkpoint: Option<usize>,
}
pub trait RLStrategy<B: Backend, E: Environment, I: EnvironmentInit<E>, L: PolicyLearner<B>>:
    Send
{
    fn train_loop(
        &self,
        learner: &mut L,
        init: I,
        context: &mut RLTrainingContext<B, L>,
    ) -> RLLoopState;
}
pub struct OffPolicyStrategy {
    config: OffPolicyConfig,
}
impl OffPolicyStrategy {
    pub fn new(config: OffPolicyConfig) -> Self {
        Self { config }
    }
}
impl<B, E, I, L> RLStrategy<B, E, I, L> for OffPolicyStrategy
where
    B: Backend,
    E: Environment + 'static,
    I: EnvironmentInit<E> + Send + 'static,
    L: PolicyLearner<B>,
    E::State: Clone + Send + ToObservation<B, <L::InnerPolicy as Policy<B>>::Observation> + 'static,
    E::Action: Clone
        + Send
        + From<<L::InnerPolicy as Policy<B>>::Action>
        + ToAction<B, <L::InnerPolicy as Policy<B>>::Action>
        + 'static,
    <L::InnerPolicy as Policy<B>>::Observation: SliceAccess<B>,
    <L::InnerPolicy as Policy<B>>::Action: SliceAccess<B>,
{
    fn train_loop(
        &self,
        learner: &mut L,
        init: I,
        context: &mut RLTrainingContext<B, L>,
    ) -> RLLoopState {
        let mut runner = MultiAgentEnvLoop::<B, E, L::InnerPolicy>::new(
            self.config.num_envs,
            init.clone(),
            AsyncPolicy::new(
                self.config.num_envs.min(self.config.autobatch_size),
                learner.policy(),
            ),
            false,
            false,
            &context.inference_device,
        );
        let mut validation = AgentEnvAsyncLoop::<B, E, L::InnerPolicy>::new(
            init,
            AsyncPolicy::new(1, learner.policy()),
            AsyncAgentEnvLoopConfig {
                eval: true,
                deterministic: true,
                id: 0,
            },
            &context.inference_device,
        );
        let mut replay = TransitionBuffer::new(self.config.replay_buffer_size, &learner.device());
        let mut next_validation = self.config.eval_interval + context.starting_step - 1;
        let mut progress = Progress::new(context.starting_step, context.num_steps);
        let mut update: Option<<L::InnerPolicy as Policy<B>>::PolicyState> = None;
        let mut state = RLLoopState {
            steps_processed: context.starting_step,
            ..Default::default()
        };
        while progress.items_processed < context.num_steps {
            if context.interrupter.should_stop() {
                break;
            }
            let previous_steps = progress.items_processed;
            let items = runner.run_steps(
                self.config.train_interval,
                context.processor.as_mut(),
                &context.interrupter,
                &mut progress,
            );
            for item in items {
                let transition = item.transition;
                let reward = transition
                    .reward
                    .try_into_scalar()
                    .map(|value| value.elem::<f32>())
                    .map_err(|e| TrainingError(e.to_string()));
                let done = transition
                    .done
                    .try_into_scalar()
                    .map(|value| value.elem::<f32>())
                    .map_err(|e| TrainingError(e.to_string()));
                match (reward, done) {
                    (Ok(reward), Ok(done)) => replay.push(
                        transition.state.to_observation(&runner.device()),
                        transition.next_state.to_observation(&runner.device()),
                        transition.action.to_action(&runner.device()),
                        reward,
                        done > 0.5,
                    ),
                    (Err(error), _) | (_, Err(error)) => {
                        context.interrupter.fail(error);
                        break;
                    }
                }
            }
            state.steps_processed = progress.items_processed;
            if context.interrupter.error().is_some() {
                break;
            }
            if replay.len() >= self.config.train_batch_size
                && progress.items_processed >= self.config.warmup_steps
            {
                if let Some(ref previous) = update {
                    runner.update_policy(previous.clone());
                }
                for _ in 0..self.config.train_steps {
                    let output = learner.train(replay.sample(self.config.train_batch_size));
                    update = Some(learner.policy().state());
                    state.optimizer_steps += 1;
                    context
                        .interrupter
                        .check(context.processor.process_train(RLEvent::TrainStep(
                            EvaluationItem::new(output.item, progress.clone(), None),
                        )));
                }
            }
            if next_validation > previous_steps && next_validation <= progress.items_processed {
                context.interrupter.check(
                    context
                        .processor
                        .process_valid(AgentEvaluationEvent::Start(self.config.eval_episodes)),
                );
                validation.update_policy(learner.policy().state());
                validation.run_episodes(
                    self.config.eval_episodes,
                    context.processor.as_mut(),
                    &context.interrupter,
                    &mut progress,
                );
                context.interrupter.check(context.processor.flush());
                if let Some(checkpointer) = &mut context.checkpointer {
                    match checkpointer.checkpoint(
                        &runner.policy(),
                        learner,
                        next_validation,
                        &context.store,
                    ) {
                        Ok(true) => state.last_checkpoint = Some(next_validation),
                        Ok(false) => {}
                        Err(error) => context.interrupter.fail(error),
                    }
                }
                next_validation += self.config.eval_interval;
                state.evaluations += 1;
                context
                    .interrupter
                    .check(context.processor.process_valid(AgentEvaluationEvent::End));
            }
        }
        state
    }
}
pub struct RLResult<B: Backend, L: PolicyLearner<B>> {
    pub policy: L::InnerPolicy,
    pub learner: L,
    pub state: RLLoopState,
    pub metrics: EventStore,
    pub interrupted: Option<Interruption>,
    pub error: Option<TrainingError>,
    pub summary: Option<LearnerSummary>,
}
pub struct RLTraining<B: Backend, E: Environment, I: EnvironmentInit<E>, L: PolicyLearner<B>> {
    init: I,
    context: RLTrainingContext<B, L>,
    strategy: Box<dyn RLStrategy<B, E, I, L>>,
    restore_step: Option<usize>,
    label: Option<String>,
    summary: Option<LearnerSummaryConfig>,
    marker: PhantomData<fn() -> E>,
}
impl<B, E, I, L> RLTraining<B, E, I, L>
where
    B: Backend,
    E: Environment + 'static,
    I: EnvironmentInit<E> + Send + 'static,
    L: PolicyLearner<B>,
    E::State: Clone + Send + ToObservation<B, <L::InnerPolicy as Policy<B>>::Observation> + 'static,
    E::Action: Clone
        + Send
        + From<<L::InnerPolicy as Policy<B>>::Action>
        + ToAction<B, <L::InnerPolicy as Policy<B>>::Action>
        + 'static,
    <L::InnerPolicy as Policy<B>>::Observation: SliceAccess<B>,
    <L::InnerPolicy as Policy<B>>::Action: SliceAccess<B>,
{
    pub fn new(init: I, device: B::Device) -> Self {
        let metrics = RLMetricsProcessor::new();
        let store = metrics.store();
        Self {
            init,
            context: RLTrainingContext {
                num_steps: 1,
                starting_step: 1,
                inference_device: device,
                interrupter: Interrupter::new(),
                processor: Box::new(LazyRLProcessor(metrics)),
                store,
                checkpointer: None,
            },
            strategy: Box::new(OffPolicyStrategy::new(OffPolicyConfig::new())),
            restore_step: None,
            label: None,
            summary: None,
            marker: PhantomData,
        }
    }
    pub fn num_steps(mut self, steps: usize) -> Self {
        self.context.num_steps = steps;
        self
    }
    pub fn off_policy(mut self, config: OffPolicyConfig) -> Self {
        self.strategy = Box::new(OffPolicyStrategy::new(config));
        self
    }
    pub fn with_strategy<S: RLStrategy<B, E, I, L> + 'static>(mut self, strategy: S) -> Self {
        self.strategy = Box::new(strategy);
        self
    }
    pub fn with_inference_device(mut self, device: B::Device) -> Self {
        self.context.inference_device = device;
        self
    }
    pub fn with_interrupter(mut self, interrupter: Interrupter) -> Self {
        self.context.interrupter = interrupter;
        self
    }
    pub fn interrupter(&self) -> Interrupter {
        self.context.interrupter.clone()
    }
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
    pub fn with_checkpointer(mut self, checkpointer: RLCheckpointer<B, L>) -> Self {
        self.context.checkpointer = Some(checkpointer);
        self
    }
    pub fn checkpoint(mut self, step: usize) -> Self {
        self.restore_step = Some(step);
        self
    }
    pub fn with_summary_config(mut self, config: LearnerSummaryConfig) -> Self {
        self.summary = Some(config);
        self
    }
    pub fn with_event_processor<
        P: RLEventProcessor<L::TrainContext, <L::InnerPolicy as Policy<B>>::ActionContext> + 'static,
    >(
        mut self,
        processor: P,
        store: EventStore,
    ) -> Self {
        self.context.processor = Box::new(processor);
        self.context.store = store;
        self
    }
    pub fn with_metrics(
        self,
        metrics: RLMetricsProcessor<L::TrainContext, <L::InnerPolicy as Policy<B>>::ActionContext>,
    ) -> Self {
        let store = metrics.store();
        self.with_event_processor(metrics, store)
    }
    pub fn with_async_metrics(
        self,
        metrics: RLMetricsProcessor<L::TrainContext, <L::InnerPolicy as Policy<B>>::ActionContext>,
    ) -> Self {
        let store = metrics.store();
        self.with_event_processor(AsyncRLProcessor::new(metrics), store)
    }
    pub fn with_lazy_metrics(
        self,
        metrics: RLMetricsProcessor<
            <L::TrainContext as ItemLazy>::Synced,
            <<L::InnerPolicy as Policy<B>>::ActionContext as ItemLazy>::Synced,
        >,
    ) -> Self {
        let store = metrics.store();
        self.with_event_processor(LazyRLProcessor(metrics), store)
    }
    pub fn with_lazy_async_metrics(
        self,
        metrics: RLMetricsProcessor<
            <L::TrainContext as ItemLazy>::Synced,
            <<L::InnerPolicy as Policy<B>>::ActionContext as ItemLazy>::Synced,
        >,
    ) -> Self {
        let store = metrics.store();
        self.with_event_processor(AsyncRLProcessor::with_lazy(metrics), store)
    }
    pub fn launch(mut self, mut learner: L) -> RLResult<B, L> {
        if let Some(step) = self.restore_step {
            self.context.starting_step = step + 1;
            match self.context.checkpointer.as_mut() {
                Some(checkpointer) => match checkpointer.restore_records(step) {
                    Ok((policy, record)) => {
                        let policy = learner.policy().load_record(policy);
                        learner = learner.load_record(record);
                        learner.update_policy(policy);
                    }
                    Err(error) => self.context.interrupter.fail(error),
                },
                None => self
                    .context
                    .interrupter
                    .fail(TrainingError("RL resume requires a checkpointer".into())),
            }
        }
        self.context
            .interrupter
            .check(self.context.processor.process_train(RLEvent::Start {
                total_items: self.context.num_steps,
                label: self.label,
            }));
        let state = if self.context.interrupter.should_stop() {
            RLLoopState {
                steps_processed: self.context.starting_step,
                ..Default::default()
            }
        } else {
            self.strategy
                .train_loop(&mut learner, self.init, &mut self.context)
        };
        self.context
            .interrupter
            .check(self.context.processor.flush());
        let summary = self.summary.and_then(|config| match config.init() {
            Ok(summary) => Some(summary),
            Err(error) => {
                self.context.interrupter.fail(TrainingError(error));
                None
            }
        });
        self.context.interrupter.check(
            self.context
                .processor
                .process_train(RLEvent::End(summary.clone())),
        );
        self.context
            .interrupter
            .check(self.context.processor.flush());
        RLResult {
            policy: learner.policy(),
            learner,
            state,
            metrics: self.context.store,
            interrupted: self.context.interrupter.interruption(),
            error: self.context.interrupter.error(),
            summary,
        }
    }
}
