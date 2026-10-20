use super::{
    AgentEvaluationEvent, AsyncPolicy, Environment, EnvironmentInit, EpisodeSummary,
    EvaluationItem, Policy, RLEvent, RLEventProcessor, ToObservation, Transition,
};
use crate::Interrupter;
use rand::prelude::SliceRandom;
use ruda_model::{
    data::dataloader::Progress,
    tensor::{Tensor, backend::Backend},
};
use std::{
    any::Any,
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    sync::{
        Arc, Mutex,
        mpsc::{self, Receiver, Sender},
    },
    thread::JoinHandle,
};

#[derive(Clone)]
pub struct TimeStep<B: Backend, S, A, C> {
    pub env_id: usize,
    pub transition: Transition<B, S, A>,
    pub done: bool,
    pub ep_len: usize,
    pub cum_reward: f64,
    pub action_context: C,
}
#[derive(Clone)]
pub struct Trajectory<B: Backend, S, A, C> {
    pub timesteps: Vec<TimeStep<B, S, A, C>>,
}
impl<B: Backend, S, A, C> Trajectory<B, S, A, C> {
    pub fn new(timesteps: Vec<TimeStep<B, S, A, C>>) -> Self {
        Self { timesteps }
    }
}
pub type EnvTimeStep<B, E, P> = TimeStep<
    B,
    <E as Environment>::State,
    <E as Environment>::Action,
    <P as Policy<B>>::ActionContext,
>;
pub type EnvTrajectory<B, E, P> = Trajectory<
    B,
    <E as Environment>::State,
    <E as Environment>::Action,
    <P as Policy<B>>::ActionContext,
>;

pub trait AgentEnvLoop<B: Backend, E: Environment, P: Policy<B>> {
    fn run_steps<T>(
        &mut self,
        num_steps: usize,
        processor: &mut dyn RLEventProcessor<T, P::ActionContext>,
        interrupter: &Interrupter,
        progress: &mut Progress,
    ) -> Vec<EnvTimeStep<B, E, P>>;
    fn run_episodes<T>(
        &mut self,
        num_episodes: usize,
        processor: &mut dyn RLEventProcessor<T, P::ActionContext>,
        interrupter: &Interrupter,
        progress: &mut Progress,
    ) -> Vec<EnvTrajectory<B, E, P>>;
    fn update_policy(&mut self, state: P::PolicyState);
    fn policy(&self) -> P::PolicyState;
    fn device(&self) -> B::Device;
}
fn episode<B: Backend, S, A, C>(step: &TimeStep<B, S, A, C>) -> EpisodeSummary {
    EpisodeSummary {
        episode_length: step.ep_len,
        cum_reward: step.cum_reward,
    }
}
fn emit_step<B: Backend, S, A, C: Clone, T>(
    step: &TimeStep<B, S, A, C>,
    processor: &mut dyn RLEventProcessor<T, C>,
    interrupter: &Interrupter,
    progress: &mut Progress,
) {
    progress.items_processed += 1;
    interrupter.check(
        processor.process_train(RLEvent::EnvStep(EvaluationItem::new(
            step.action_context.clone(),
            progress.clone(),
            None,
        ))),
    );
    if step.done {
        interrupter.check(
            processor.process_train(RLEvent::EpisodeEnd(EvaluationItem::new(
                episode(step),
                progress.clone(),
                None,
            ))),
        );
    }
}
fn emit_episode_step<B: Backend, S, A, C: Clone, T>(
    step: &TimeStep<B, S, A, C>,
    eval: bool,
    processor: &mut dyn RLEventProcessor<T, C>,
    interrupter: &Interrupter,
    step_count: usize,
    episode_count: usize,
    total: usize,
) {
    let action = EvaluationItem::new(
        step.action_context.clone(),
        Progress::new(step_count, step_count),
        None,
    );
    if eval {
        interrupter.check(processor.process_valid(AgentEvaluationEvent::EnvStep(action)));
    } else {
        interrupter.check(processor.process_train(RLEvent::EnvStep(action)));
    }
    if step.done {
        let summary = EvaluationItem::new(episode(step), Progress::new(episode_count, total), None);
        if eval {
            interrupter.check(processor.process_valid(AgentEvaluationEvent::EpisodeEnd(summary)));
        } else {
            interrupter.check(processor.process_train(RLEvent::EpisodeEnd(summary)));
        }
    }
}

pub struct AgentEnvBaseLoop<B: Backend, E: Environment, P: Policy<B>> {
    env: E,
    agent: P,
    eval: bool,
    deterministic: bool,
    current_reward: f64,
    step_num: usize,
    device: B::Device,
}
impl<B: Backend, E: Environment, P: Policy<B>> AgentEnvBaseLoop<B, E, P> {
    pub fn new<I: EnvironmentInit<E>>(
        init: I,
        agent: P,
        eval: bool,
        deterministic: bool,
        device: &B::Device,
    ) -> Self {
        let mut env = init.init();
        env.reset();
        Self {
            env,
            agent: agent.to_device(device),
            eval,
            deterministic,
            current_reward: 0.,
            step_num: 0,
            device: device.clone(),
        }
    }
}
impl<B, E, P> AgentEnvLoop<B, E, P> for AgentEnvBaseLoop<B, E, P>
where
    B: Backend,
    E: Environment,
    P: Policy<B>,
    E::State: Clone + ToObservation<B, P::Observation>,
    E::Action: Clone + From<P::Action>,
{
    fn run_steps<T>(
        &mut self,
        num_steps: usize,
        processor: &mut dyn RLEventProcessor<T, P::ActionContext>,
        interrupter: &Interrupter,
        progress: &mut Progress,
    ) -> Vec<EnvTimeStep<B, E, P>> {
        let mut items = Vec::new();
        let transition_device = B::Device::default();
        for _ in 0..num_steps {
            let state = self.env.state();
            let (action, context) = self
                .agent
                .action(state.to_observation(&self.device), self.deterministic);
            let action = E::Action::from(action);
            let result = self.env.step(action.clone());
            self.current_reward += result.reward;
            self.step_num += 1;
            let step = TimeStep {
                env_id: 0,
                transition: Transition {
                    state,
                    next_state: result.next_state,
                    action,
                    reward: Tensor::from_data([result.reward], &transition_device),
                    done: Tensor::from_data(
                        [if result.done || result.truncated {
                            1f64
                        } else {
                            0f64
                        }],
                        &transition_device,
                    ),
                },
                done: result.done,
                ep_len: self.step_num,
                cum_reward: self.current_reward,
                action_context: context[0].clone(),
            };
            if !self.eval {
                emit_step(&step, processor, interrupter, progress);
            }
            items.push(step);
            if interrupter.should_stop() {
                break;
            }
            if result.done || result.truncated {
                self.env.reset();
                self.current_reward = 0.;
                self.step_num = 0;
            }
        }
        items
    }
    fn run_episodes<T>(
        &mut self,
        num_episodes: usize,
        processor: &mut dyn RLEventProcessor<T, P::ActionContext>,
        interrupter: &Interrupter,
        progress: &mut Progress,
    ) -> Vec<EnvTrajectory<B, E, P>> {
        self.env.reset();
        let mut items = Vec::new();
        for episode_num in 0..num_episodes {
            let mut steps = Vec::new();
            loop {
                let step = self
                    .run_steps(1, processor, interrupter, progress)
                    .remove(0);
                let done = step.done;
                if self.eval {
                    emit_episode_step(
                        &step,
                        true,
                        processor,
                        interrupter,
                        steps.len() + 2,
                        episode_num + 1,
                        num_episodes,
                    );
                }
                steps.push(step);
                if interrupter.should_stop() || done {
                    break;
                }
            }
            items.push(Trajectory::new(steps));
            if interrupter.should_stop() {
                break;
            }
        }
        items
    }
    fn update_policy(&mut self, state: P::PolicyState) {
        self.agent.update(state);
    }
    fn policy(&self) -> P::PolicyState {
        self.agent.state()
    }
    fn device(&self) -> B::Device {
        self.device.clone()
    }
}

enum Request {
    Step,
    Episode,
}
#[derive(Clone, Default)]
struct WorkerFailure(Arc<Mutex<Option<Box<dyn Any + Send>>>>);
impl WorkerFailure {
    fn record(&self, error: Box<dyn Any + Send>) {
        *self.0.lock().unwrap() = Some(error);
    }
    fn resume(&self) -> ! {
        let error = self.take();
        match error {
            Some(error) => resume_unwind(error),
            None => panic!("Environment worker is closed"),
        }
    }
    fn take(&self) -> Option<Box<dyn Any + Send>> {
        self.0.lock().unwrap().take()
    }
}
type WorkerResult<T> = Result<T, WorkerFailure>;
fn receive_worker<T>(receiver: &Receiver<WorkerResult<T>>) -> T {
    match receiver
        .recv()
        .expect("Environment worker returns a result")
    {
        Ok(value) => value,
        Err(failure) => failure.resume(),
    }
}
pub struct AsyncAgentEnvLoopConfig {
    pub eval: bool,
    pub deterministic: bool,
    pub id: usize,
}
struct EnvWorker<B: Backend, S, A, C> {
    request: Option<Sender<Request>>,
    transitions: Receiver<WorkerResult<TimeStep<B, S, A, C>>>,
    trajectories: Receiver<WorkerResult<Trajectory<B, S, A, C>>>,
    failure: WorkerFailure,
    handle: Option<JoinHandle<()>>,
}
impl<B: Backend, S, A, C> EnvWorker<B, S, A, C> {
    fn request(&self, request: Request) {
        if self.request.as_ref().unwrap().send(request).is_err() {
            self.failure.resume();
        }
    }
}
impl<B: Backend, S, A, C> Drop for EnvWorker<B, S, A, C> {
    fn drop(&mut self) {
        self.request.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
fn spawn_env<B, E, I, P>(
    init: I,
    mut agent: AsyncPolicy<B, P>,
    config: AsyncAgentEnvLoopConfig,
    device: B::Device,
    transition_sender: Option<Sender<WorkerResult<EnvTimeStep<B, E, P>>>>,
    trajectory_sender: Option<Sender<WorkerResult<EnvTrajectory<B, E, P>>>>,
) -> EnvWorker<B, E::State, E::Action, P::ActionContext>
where
    B: Backend,
    E: Environment + 'static,
    I: EnvironmentInit<E> + Send + 'static,
    P: Policy<B>,
    E::State: Clone + Send + ToObservation<B, P::Observation> + 'static,
    E::Action: Clone + Send + From<P::Action> + 'static,
{
    let (transitions_tx, transitions) = mpsc::channel();
    let (trajectories_tx, trajectories) = mpsc::channel();
    let (request, receive) = mpsc::channel();
    let transitions_tx = transition_sender.unwrap_or(transitions_tx);
    let trajectories_tx = trajectory_sender.unwrap_or(trajectories_tx);
    agent = agent.to_device(&device);
    let failure = WorkerFailure::default();
    let thread_failure = failure.clone();
    let handle = std::thread::Builder::new()
        .name(format!("ruda-env-{}", config.id))
        .spawn(move || {
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                let mut env = init.init();
                env.reset();
                let mut steps = Vec::new();
                let mut reward = 0.;
                let mut step_num = 0;
                let mut request_episode = false;
                loop {
                    let state = env.state();
                    let (action, context) =
                        agent.action(state.to_observation(&device), config.deterministic);
                    let action = E::Action::from(action);
                    let result = env.step(action.clone());
                    reward += result.reward;
                    step_num += 1;
                    let transition = Transition {
                        state,
                        next_state: result.next_state,
                        action,
                        reward: Tensor::from_data([result.reward], &device),
                        done: Tensor::from_data(
                            [if result.done || result.truncated {
                                1f64
                            } else {
                                0f64
                            }],
                            &device,
                        ),
                    };
                    if !request_episode {
                        agent.decrement_agents(1);
                        let Ok(request) = receive.recv() else { break };
                        agent.increment_agents(1);
                        request_episode = matches!(request, Request::Episode);
                    }
                    let step = TimeStep {
                        env_id: config.id,
                        transition,
                        done: result.done,
                        ep_len: step_num,
                        cum_reward: reward,
                        action_context: context[0].clone(),
                    };
                    steps.push(step.clone());
                    if !request_episode && transitions_tx.send(Ok(step)).is_err() {
                        break;
                    }
                    if result.done || result.truncated {
                        if request_episode {
                            request_episode = false;
                            if trajectories_tx
                                .send(Ok(Trajectory::new(core::mem::take(&mut steps))))
                                .is_err()
                            {
                                break;
                            }
                        }
                        steps.clear();
                        env.reset();
                        reward = 0.;
                        step_num = 0;
                    }
                }
            }));
            if let Err(error) = outcome {
                thread_failure.record(error);
                let _ = transitions_tx.send(Err(thread_failure.clone()));
                let _ = trajectories_tx.send(Err(thread_failure));
                agent.stop_failed_workers();
            }
        })
        .expect("Can start environment worker");
    EnvWorker {
        request: Some(request),
        transitions,
        trajectories,
        failure,
        handle: Some(handle),
    }
}
pub struct AgentEnvAsyncLoop<B: Backend, E: Environment, P: Policy<B>> {
    worker: EnvWorker<B, E::State, E::Action, P::ActionContext>,
    agent: AsyncPolicy<B, P>,
    eval: bool,
    device: B::Device,
}
impl<B, E, P> AgentEnvAsyncLoop<B, E, P>
where
    B: Backend,
    E: Environment + 'static,
    P: Policy<B>,
    E::State: Clone + Send + ToObservation<B, P::Observation> + 'static,
    E::Action: Clone + Send + From<P::Action> + 'static,
{
    pub fn new<I: EnvironmentInit<E> + Send + 'static>(
        init: I,
        agent: AsyncPolicy<B, P>,
        config: AsyncAgentEnvLoopConfig,
        device: &B::Device,
    ) -> Self {
        let eval = config.eval;
        let worker = spawn_env(init, agent.clone(), config, device.clone(), None, None);
        Self {
            worker,
            agent,
            eval,
            device: device.clone(),
        }
    }
}
impl<B, E, P> AgentEnvLoop<B, E, P> for AgentEnvAsyncLoop<B, E, P>
where
    B: Backend,
    E: Environment + 'static,
    P: Policy<B>,
    E::State: Clone + Send + ToObservation<B, P::Observation> + 'static,
    E::Action: Clone + Send + From<P::Action> + 'static,
{
    fn run_steps<T>(
        &mut self,
        num_steps: usize,
        processor: &mut dyn RLEventProcessor<T, P::ActionContext>,
        interrupter: &Interrupter,
        progress: &mut Progress,
    ) -> Vec<EnvTimeStep<B, E, P>> {
        let mut items = Vec::new();
        for _ in 0..num_steps {
            self.worker.request(Request::Step);
            let step = receive_worker(&self.worker.transitions);
            if !self.eval {
                emit_step(&step, processor, interrupter, progress);
            }
            items.push(step);
            if interrupter.should_stop() {
                break;
            }
        }
        items
    }
    fn run_episodes<T>(
        &mut self,
        num_episodes: usize,
        processor: &mut dyn RLEventProcessor<T, P::ActionContext>,
        interrupter: &Interrupter,
        _: &mut Progress,
    ) -> Vec<EnvTrajectory<B, E, P>> {
        let mut items = Vec::new();
        self.agent.increment_agents(1);
        for episode_num in 0..num_episodes {
            self.worker.request(Request::Episode);
            let trajectory = receive_worker(&self.worker.trajectories);
            for (i, step) in trajectory.timesteps.iter().enumerate() {
                emit_episode_step(
                    step,
                    self.eval,
                    processor,
                    interrupter,
                    i,
                    episode_num + 1,
                    num_episodes,
                );
            }
            items.push(trajectory);
            if interrupter.should_stop() {
                break;
            }
        }
        self.agent.decrement_agents(1);
        items
    }
    fn update_policy(&mut self, state: P::PolicyState) {
        self.agent.update(state);
        self.agent.clone().to_device(&self.device);
    }
    fn policy(&self) -> P::PolicyState {
        self.agent.state()
    }
    fn device(&self) -> B::Device {
        self.device.clone()
    }
}

pub struct MultiAgentEnvLoop<B: Backend, E: Environment, P: Policy<B>> {
    workers: Vec<EnvWorker<B, E::State, E::Action, P::ActionContext>>,
    agent: AsyncPolicy<B, P>,
    transitions: Receiver<WorkerResult<EnvTimeStep<B, E, P>>>,
    trajectories: Receiver<WorkerResult<EnvTrajectory<B, E, P>>>,
    eval: bool,
    device: B::Device,
}
impl<B, E, P> MultiAgentEnvLoop<B, E, P>
where
    B: Backend,
    E: Environment + 'static,
    P: Policy<B>,
    E::State: Clone + Send + ToObservation<B, P::Observation> + 'static,
    E::Action: Clone + Send + From<P::Action> + 'static,
{
    pub fn new<I: EnvironmentInit<E> + Send + 'static>(
        num_envs: usize,
        init: I,
        agent: AsyncPolicy<B, P>,
        eval: bool,
        deterministic: bool,
        device: &B::Device,
    ) -> Self {
        let (transitions_tx, transitions) = mpsc::channel();
        let (trajectories_tx, trajectories) = mpsc::channel();
        agent.increment_agents(num_envs);
        let mut workers: Vec<EnvWorker<B, E::State, E::Action, P::ActionContext>> = Vec::new();
        for id in 0..num_envs {
            let worker = catch_unwind(AssertUnwindSafe(|| {
                spawn_env(
                    init.clone(),
                    agent.clone(),
                    AsyncAgentEnvLoopConfig {
                        eval,
                        deterministic,
                        id,
                    },
                    device.clone(),
                    Some(transitions_tx.clone()),
                    Some(trajectories_tx.clone()),
                )
            }));
            match worker {
                Ok(worker) => workers.push(worker),
                Err(error) => {
                    let original = workers.iter().find_map(|worker| worker.failure.take());
                    agent.stop_failed_workers();
                    for worker in &mut workers {
                        worker.request.take();
                    }
                    drop(workers);
                    resume_unwind(original.unwrap_or(error));
                }
            }
        }
        for worker in &workers {
            worker.request(Request::Step);
        }
        Self {
            workers,
            agent,
            transitions,
            trajectories,
            eval,
            device: device.clone(),
        }
    }
}
impl<B, E, P> AgentEnvLoop<B, E, P> for MultiAgentEnvLoop<B, E, P>
where
    B: Backend,
    E: Environment + 'static,
    P: Policy<B>,
    E::State: Clone + Send + ToObservation<B, P::Observation> + 'static,
    E::Action: Clone + Send + From<P::Action> + 'static,
{
    fn run_steps<T>(
        &mut self,
        num_steps: usize,
        processor: &mut dyn RLEventProcessor<T, P::ActionContext>,
        interrupter: &Interrupter,
        progress: &mut Progress,
    ) -> Vec<EnvTimeStep<B, E, P>> {
        let mut items = Vec::new();
        for _ in 0..num_steps {
            let step = receive_worker(&self.transitions);
            self.workers[step.env_id].request(Request::Step);
            if !self.eval {
                emit_step(&step, processor, interrupter, progress);
            }
            items.push(step);
            if interrupter.should_stop() {
                break;
            }
        }
        items
    }
    fn run_episodes<T>(
        &mut self,
        num_episodes: usize,
        processor: &mut dyn RLEventProcessor<T, P::ActionContext>,
        interrupter: &Interrupter,
        _: &mut Progress,
    ) -> Vec<EnvTrajectory<B, E, P>> {
        let mut ids: Vec<_> = (0..self.workers.len()).collect();
        if num_episodes < ids.len() {
            ids.shuffle(&mut rand::rng());
            ids.truncate(num_episodes);
        }
        let requests = ids.len();
        for id in ids {
            self.workers[id].request(Request::Episode);
        }
        let mut items = Vec::new();
        for episode_num in 0..num_episodes {
            let trajectory = receive_worker(&self.trajectories);
            if items.len() + 1 + requests <= num_episodes {
                self.workers[trajectory.timesteps[0].env_id].request(Request::Episode);
            }
            for (i, step) in trajectory.timesteps.iter().enumerate() {
                emit_episode_step(
                    step,
                    self.eval,
                    processor,
                    interrupter,
                    i,
                    episode_num + 1,
                    num_episodes,
                );
            }
            items.push(trajectory);
            if interrupter.should_stop() {
                break;
            }
        }
        items
    }
    fn update_policy(&mut self, state: P::PolicyState) {
        self.agent.update(state);
        self.agent.clone().to_device(&self.device);
    }
    fn policy(&self) -> P::PolicyState {
        self.agent.state()
    }
    fn device(&self) -> B::Device {
        self.device.clone()
    }
}
impl<B: Backend, E: Environment, P: Policy<B>> Drop for MultiAgentEnvLoop<B, E, P> {
    fn drop(&mut self) {
        for worker in &mut self.workers {
            worker.request.take();
        }
    }
}
