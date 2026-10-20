use super::{AD, TestBackend, close};
use crate::{
    logger::{FileMetricLogger as SerializedLogger, InMemoryMetricLogger},
    metric::{
        Adaptor, CumulativeRewardMetric, EpisodeLengthMetric, ExplorationRateInput,
        ExplorationRateMetric, LossMetric,
    },
    rl::*,
    *,
};
use ruda_model::{
    data::dataloader::Progress,
    module::{AutodiffModule, Module, Param},
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{Device, Int, Tensor, TensorData, backend::Backend},
};
use ruda_optim::{GradientsParams, Optimizer, Sgd, SgdConfig, adaptor::OptimizerAdaptor};
use std::{
    path::PathBuf,
    sync::{
        Arc, Barrier, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

#[derive(Module, Debug)]
struct QNet<B: Backend> {
    weights: Param<Tensor<B, 2>>,
}
impl<B: Backend> QNet<B> {
    fn new() -> Self {
        Self {
            weights: Param::from_tensor(Tensor::zeros([2, 2], &Default::default())),
        }
    }
    fn forward(&self, observation: Tensor<B, 2>) -> Tensor<B, 2> {
        observation.matmul(self.weights.val())
    }
}
#[derive(Clone)]
struct QState {
    net: QNet<TestBackend>,
    revision: usize,
}
impl PolicyState for QState {
    type Record = (QNetRecord<TestBackend>, usize);
    fn into_record(self) -> Self::Record {
        (self.net.into_record(), self.revision)
    }
    fn load_record(&self, record: Self::Record) -> Self {
        Self {
            net: self.net.clone().load_record(record.0),
            revision: record.1,
        }
    }
}
#[derive(Clone)]
struct ActionContext {
    exploration: f64,
    revision: usize,
}
impl ItemLazy for ActionContext {
    type Synced = Self;
    fn sync(self) -> Result<Self, TrainingError> {
        Ok(self)
    }
}
impl Adaptor<ExplorationRateInput> for ActionContext {
    fn adapt(&self) -> ExplorationRateInput {
        ExplorationRateInput::new(self.exploration)
    }
}
#[derive(Clone)]
struct QPolicy {
    state: QState,
    cursor: usize,
    batches: Arc<Mutex<Vec<(usize, bool)>>>,
}
impl QPolicy {
    fn new() -> Self {
        Self {
            state: QState {
                net: QNet::new(),
                revision: 0,
            },
            cursor: 0,
            batches: Arc::default(),
        }
    }
}
impl Policy<TestBackend> for QPolicy {
    type Observation = Tensor<TestBackend, 2>;
    type ActionDistribution = Tensor<TestBackend, 2>;
    type Action = Tensor<TestBackend, 2, Int>;
    type ActionContext = ActionContext;
    type PolicyState = QState;
    fn forward(&mut self, observation: Self::Observation) -> Self::ActionDistribution {
        self.state.net.forward(observation)
    }
    fn action(
        &mut self,
        observation: Self::Observation,
        deterministic: bool,
    ) -> (Self::Action, Vec<ActionContext>) {
        let count = observation.dims()[0];
        let device = observation.device();
        self.batches.lock().unwrap().push((count, deterministic));
        let actions = if deterministic {
            self.forward(observation).argmax(1)
        } else {
            let actions: Vec<i64> = (0..count)
                .map(|_| {
                    let action = (self.cursor / 2) % 2;
                    self.cursor += 1;
                    action as i64
                })
                .collect();
            Tensor::from_data(TensorData::new(actions, [count, 1]), &device)
        };
        (
            actions,
            (0..count)
                .map(|_| ActionContext {
                    exploration: if deterministic { 0. } else { 1. },
                    revision: self.state.revision,
                })
                .collect(),
        )
    }
    fn update(&mut self, state: QState) {
        self.state = state;
    }
    fn state(&self) -> QState {
        self.state.clone()
    }
    fn to_device(mut self, device: &Device<TestBackend>) -> Self {
        self.state.net = self.state.net.to_device(device);
        self
    }
    fn load_record(mut self, record: <QState as PolicyState>::Record) -> Self {
        self.state = self.state.load_record(record);
        self
    }
}
type QOptimizer = OptimizerAdaptor<Sgd<TestBackend>, QNet<AD>, AD>;
struct QLearner {
    net: QNet<AD>,
    optimizer: QOptimizer,
    steps: usize,
}
impl QLearner {
    fn new() -> Self {
        Self {
            net: QNet::new(),
            optimizer: SgdConfig::new().init(),
            steps: 0,
        }
    }
}
impl PolicyLearner<TestBackend> for QLearner {
    type TrainContext = RegressionOutput<TestBackend>;
    type InnerPolicy = QPolicy;
    type Record = (
        QNetRecord<AD>,
        <QOptimizer as Optimizer<QNet<AD>, AD>>::Record,
        usize,
    );
    fn train(
        &mut self,
        batch: LearnerTransitionBatch<TestBackend, QPolicy>,
    ) -> RLTrainOutput<Self::TrainContext, QState> {
        let targets = batch.rewards
            + (batch.dones * -1. + 1.)
                * self.net.valid().forward(batch.next_states).max_dim(1)
                * 0.5;
        let predicted = self
            .net
            .forward(Tensor::from_inner(batch.states))
            .gather(1, Tensor::from_inner(batch.actions));
        let loss = (predicted.clone() - Tensor::from_inner(targets.clone()))
            .square()
            .flatten::<1>(0, 1);
        let gradients = GradientsParams::from_grads(loss.clone().mean().backward(), &self.net);
        self.net = self.optimizer.step(0.15, self.net.clone(), gradients);
        self.steps += 1;
        RLTrainOutput {
            policy: self.policy().state(),
            item: RegressionOutput {
                loss: loss.inner(),
                output: predicted.inner(),
                targets,
            },
        }
    }
    fn policy(&self) -> QPolicy {
        QPolicy {
            state: QState {
                net: self.net.valid(),
                revision: self.steps,
            },
            ..QPolicy::new()
        }
    }
    fn update_policy(&mut self, policy: QPolicy) {
        self.net.weights = self
            .net
            .weights
            .clone()
            .map(|_| Tensor::from_inner(policy.state.net.weights.val()).require_grad());
    }
    fn record(&self) -> Self::Record {
        (
            self.net.clone().into_record(),
            self.optimizer.to_record(),
            self.steps,
        )
    }
    fn load_record(mut self, record: Self::Record) -> Self {
        self.net = self.net.load_record(record.0);
        self.optimizer = self.optimizer.load_record(record.1);
        self.steps = record.2;
        self
    }
    fn device(&self) -> Device<TestBackend> {
        self.net.weights.val().device()
    }
}
#[derive(Clone)]
struct Observation(usize);
impl ToObservation<TestBackend, Tensor<TestBackend, 2>> for Observation {
    fn to_observation(&self, device: &Device<TestBackend>) -> Tensor<TestBackend, 2> {
        Tensor::from_data(
            [if self.0 % 2 == 0 {
                [1f32, 0.]
            } else {
                [0f32, 1.]
            }],
            device,
        )
    }
}
#[derive(Clone)]
struct Action(i64);
impl From<Tensor<TestBackend, 2, Int>> for Action {
    fn from(action: Tensor<TestBackend, 2, Int>) -> Self {
        Self(action.into_data().convert::<i64>().to_vec::<i64>().unwrap()[0])
    }
}
impl ToAction<TestBackend, Tensor<TestBackend, 2, Int>> for Action {
    fn to_action(&self, device: &Device<TestBackend>) -> Tensor<TestBackend, 2, Int> {
        Tensor::from_data([[self.0]], device)
    }
}
struct Bandit {
    step: usize,
    truncate: bool,
}
impl Environment for Bandit {
    type State = Observation;
    type Action = Action;
    const MAX_STEPS: usize = 3;
    fn state(&self) -> Observation {
        Observation(self.step % 2)
    }
    fn step(&mut self, action: Action) -> StepResult<Observation> {
        let reward = if action.0 == (self.step % 2) as i64 {
            1.
        } else {
            -1.
        };
        self.step += 1;
        StepResult {
            next_state: self.state(),
            reward,
            done: self.step == 3,
            truncated: self.truncate && self.step == 2,
        }
    }
    fn reset(&mut self) {
        self.step = 0;
    }
}
fn init() -> Bandit {
    Bandit {
        step: 0,
        truncate: false,
    }
}
fn device() -> Device<TestBackend> {
    Default::default()
}

fn shared_handoff<T: Clone + Send + 'static>(
    create: impl Fn(usize) -> T + Send + 'static,
    read: impl Fn(T, usize),
) {
    let (send, receive) = std::sync::mpsc::sync_channel(0);
    let barrier = Arc::new(Barrier::new(2));
    let producer_barrier = barrier.clone();
    let producer = std::thread::spawn(move || {
        for i in 0..256 {
            let tensor = create(i);
            send.send(tensor.clone()).unwrap();
            producer_barrier.wait();
            drop(tensor);
        }
    });
    for i in 0..256 {
        let tensor = receive.recv().unwrap();
        barrier.wait();
        read(tensor, i);
    }
    producer.join().unwrap();
}

#[test]
fn shared_float_read_survives_concurrent_producer_drop() {
    shared_handoff(
        |i| Tensor::<TestBackend, 1>::from_data([i as f32, -(i as f32)], &device()) + 2.,
        |tensor, i| {
            assert_eq!(tensor.into_data().convert::<f32>().to_vec::<f32>().unwrap(),
                vec![i as f32 + 2., -(i as f32) + 2.]);
        },
    );
}

#[test]
fn shared_int_read_survives_concurrent_producer_drop() {
    shared_handoff(
        |i| Tensor::<TestBackend, 1, Int>::from_data([i as i64, -(i as i64)], &device()) + 2,
        |tensor, i| {
            assert_eq!(tensor.into_data().convert::<i64>().to_vec::<i64>().unwrap(),
                vec![i as i64 + 2, -(i as i64) + 2]);
        },
    );
}

#[test]
fn shared_bool_read_survives_concurrent_producer_drop() {
    shared_handoff(
        |i| Tensor::<TestBackend, 1, Int>::from_data([i as i64, -(i as i64)], &device()).greater_elem(0),
        |tensor, i| {
            assert_eq!(tensor.into_data().convert::<bool>().to_vec::<bool>().unwrap(), vec![i > 0, false]);
        },
    );
}

#[test]
fn shared_quantized_read_survives_concurrent_producer_drop() {
    let create = |_| Tensor::<TestBackend, 1>::from_data([0., 1., -1., 0.5], &device())
        .quantize_dynamic(&Default::default());
    let expected = create(0).into_data();
    shared_handoff(create, |tensor, _| assert_eq!(tensor.into_data(), expected));
}

#[test]
fn shared_tensor_aliases_preserve_both_streams_and_chained_operations() {
    let original = Tensor::<TestBackend, 1>::from_data([1., 2., 3.], &device()) + 4.;
    let moved = original.clone();
    let returned = std::thread::spawn(move || {
        let alias = moved.clone();
        let chained = std::thread::spawn(move || alias * 3.).join().unwrap();
        (moved + 10., chained + 1.)
    }).join().unwrap();
    assert_eq!(original.into_data().convert::<f32>().to_vec::<f32>().unwrap(), vec![5., 6., 7.]);
    assert_eq!(returned.0.into_data().convert::<f32>().to_vec::<f32>().unwrap(), vec![15., 16., 17.]);
    assert_eq!(returned.1.into_data().convert::<f32>().to_vec::<f32>().unwrap(), vec![16., 19., 22.]);
}

#[derive(Debug)]
struct EnvironmentFailure(&'static str);
struct FailingBandit {
    inner: Bandit,
    point: Option<&'static str>,
}
impl Environment for FailingBandit {
    type State = Observation;
    type Action = Action;
    const MAX_STEPS: usize = 3;
    fn state(&self) -> Observation {
        if self.point == Some("state") {
            std::panic::panic_any(EnvironmentFailure("state"));
        }
        self.inner.state()
    }
    fn step(&mut self, action: Action) -> StepResult<Observation> {
        if self.point == Some("step") {
            std::panic::panic_any(EnvironmentFailure("step"));
        }
        self.inner.step(action)
    }
    fn reset(&mut self) { self.inner.reset(); }
}
fn worker_failure_is_propagated(multi: bool, episodes: bool) {
    for point in ["init", "state", "step"] {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let next = Arc::new(AtomicU64::new(0));
            let initialize = move || {
                let failing = next.fetch_add(1, Ordering::Relaxed) == 0;
                if failing && point == "init" {
                    std::panic::panic_any(EnvironmentFailure("init"));
                }
                FailingBandit { inner: init(), point: failing.then_some(point) }
            };
            fn run<L: AgentEnvLoop<TestBackend, FailingBandit, QPolicy>>(mut runner: L, episodes: bool) {
                let mut metrics = RLMetricsProcessor::<(), ActionContext>::new();
                let mut progress = Progress::new(0, 8);
                let interrupter = Interrupter::new();
                if episodes {
                    runner.run_episodes(3, &mut metrics, &interrupter, &mut progress);
                } else {
                    runner.run_steps(8, &mut metrics, &interrupter, &mut progress);
                }
            }
            if multi {
                run(MultiAgentEnvLoop::new(3, initialize, AsyncPolicy::new(3, QPolicy::new()),
                    false, false, &device()), episodes);
            } else {
                run(AgentEnvAsyncLoop::new(initialize, AsyncPolicy::new(1, QPolicy::new()),
                    AsyncAgentEnvLoopConfig { eval: false, deterministic: false, id: 0 }, &device()), episodes);
            }
        }));
        let error = outcome.expect_err("Original worker failure must reach caller");
        assert_eq!(error.downcast_ref::<EnvironmentFailure>().expect("Original panic payload").0, point);
    }
}
#[test]
fn single_worker_step_failure_reaches_caller() { worker_failure_is_propagated(false, false); }
#[test]
fn single_worker_episode_failure_reaches_caller() { worker_failure_is_propagated(false, true); }
#[test]
fn multi_worker_step_failure_reaches_caller_and_joins_peers() { worker_failure_is_propagated(true, false); }
#[test]
fn multi_worker_episode_failure_reaches_caller_and_joins_peers() { worker_failure_is_propagated(true, true); }

fn directory(label: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "ruda-rl-{label}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}
fn metrics() -> RLMetricsProcessor<RegressionOutput<TestBackend>, ActionContext> {
    RLMetricsProcessor::new()
        .metric_train(LossMetric::new())
        .metric_agent(ExplorationRateMetric::new())
        .metric_episode(CumulativeRewardMetric::new())
        .metric_episode(EpisodeLengthMetric::new())
}
fn checkpointer(path: &std::path::Path) -> RLCheckpointer<TestBackend, QLearner> {
    RLCheckpointer::new(
        FileCheckpointer::<TestBackend, <QState as PolicyState>::Record, _>::new(
            path.join("policy"),
            BinFileRecorder::<FullPrecisionSettings>::new(),
            device(),
        ),
        FileCheckpointer::<AD, <QLearner as PolicyLearner<TestBackend>>::Record, _>::new(
            path.join("learner"),
            BinFileRecorder::<FullPrecisionSettings>::new(),
            device(),
        ),
        2,
    )
}

#[test]
fn replay_wraparound_preserves_aligned_tensor_fields() {
    let mut replay = TransitionBuffer::new(4, &device());
    for i in 0..7 {
        replay.push(
            Tensor::<TestBackend, 2>::from_data([[i as f32, (i + 10) as f32]], &device()),
            Tensor::from_data([[(i + 1) as f32, (i + 11) as f32]], &device()),
            Tensor::<TestBackend, 2, Int>::from_data([[i as i64]], &device()),
            i as f32 + 0.25,
            i % 2 == 0,
        );
    }
    assert_eq!(
        (replay.len(), replay.capacity(), replay.is_empty()),
        (4, 4, false)
    );
    for _ in 0..8 {
        let batch = replay.sample(4);
        let states = batch
            .states
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        let next = batch
            .next_states
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        let actions = batch
            .actions
            .into_data()
            .convert::<i64>()
            .to_vec::<i64>()
            .unwrap();
        let rewards = batch
            .rewards
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        let dones = batch
            .dones
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        for i in 0..4 {
            let value = states[2 * i] as i64;
            assert!((3..7).contains(&value));
            assert_eq!(states[2 * i + 1], value as f32 + 10.);
            assert_eq!(next[2 * i], value as f32 + 1.);
            assert_eq!(actions[i], value);
            assert_eq!(rewards[i], value as f32 + 0.25);
            assert_eq!(dones[i], if value % 2 == 0 { 1. } else { 0. });
        }
    }
}
#[test]
fn async_policy_batches_modes_and_supports_forward_update_and_load() {
    let policy = QPolicy::new();
    let batches = policy.batches.clone();
    let mut asynchronous = AsyncPolicy::new(4, policy);
    asynchronous.increment_agents(4);
    let barrier = Arc::new(Barrier::new(4));
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let mut policy = asynchronous.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let input = Observation(i).to_observation(&device());
                barrier.wait();
                let (action, context) = policy.action(input, i % 2 == 0);
                assert_eq!(action.dims(), [1, 1]);
                assert_eq!(context.len(), 1);
                close(context[0].exploration, if i % 2 == 0 { 0. } else { 1. });
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
    let mut calls = batches.lock().unwrap().clone();
    calls.sort_unstable();
    assert_eq!(calls, vec![(2, false), (2, true)]);
    asynchronous.decrement_agents(4);
    let mut state = asynchronous.state();
    state.revision = 17;
    asynchronous.update(state);
    assert_eq!(asynchronous.state().revision, 17);
    let output = asynchronous.forward(Observation(0).to_observation(&device()));
    assert_eq!(output.dims(), [1, 2]);
    let mut state = asynchronous.state();
    state.revision = 23;
    asynchronous = asynchronous
        .load_record(state.into_record())
        .to_device(&device());
    assert_eq!(asynchronous.state().revision, 23);
}
#[test]
fn sync_async_multi_environment_steps_episodes_and_truncation() {
    let interrupter = Interrupter::new();
    let mut processor =
        RLMetricsProcessor::<(), ActionContext>::new().metric_episode(EpisodeLengthMetric::new());
    let mut progress = Progress::new(1, 100);
    let mut base = AgentEnvBaseLoop::new(init, QPolicy::new(), false, false, &device());
    let steps = base.run_steps(7, &mut processor, &interrupter, &mut progress);
    assert_eq!(
        steps.iter().map(|step| step.ep_len).collect::<Vec<_>>(),
        vec![1, 2, 3, 1, 2, 3, 1]
    );
    assert_eq!(progress.items_processed, 8);
    let mut asynchronous = AgentEnvAsyncLoop::new(
        init,
        AsyncPolicy::new(1, QPolicy::new()),
        AsyncAgentEnvLoopConfig {
            eval: true,
            deterministic: true,
            id: 9,
        },
        &device(),
    );
    let episodes = asynchronous.run_episodes(2, &mut processor, &interrupter, &mut progress);
    assert_eq!(episodes.len(), 2);
    assert!(episodes.iter().all(|episode| episode.timesteps.len() == 3
        && episode.timesteps[2].done
        && episode.timesteps[0].env_id == 9));
    assert_eq!(progress.items_processed, 8);
    let mut multi = MultiAgentEnvLoop::new(
        3,
        init,
        AsyncPolicy::new(3, QPolicy::new()),
        false,
        false,
        &device(),
    );
    let steps = multi.run_steps(11, &mut processor, &interrupter, &mut progress);
    assert_eq!(steps.len(), 11);
    assert_eq!(progress.items_processed, 19);
    assert!(steps.iter().all(|step| step.env_id < 3));
    let episodes = multi.run_episodes(2, &mut processor, &interrupter, &mut progress);
    assert_eq!(episodes.len(), 2);
    let truncated = || Bandit {
        step: 0,
        truncate: true,
    };
    let mut asynchronous = AgentEnvAsyncLoop::new(
        truncated,
        AsyncPolicy::new(1, QPolicy::new()),
        AsyncAgentEnvLoopConfig {
            eval: true,
            deterministic: true,
            id: 0,
        },
        &device(),
    );
    let episode = asynchronous
        .run_episodes(1, &mut processor, &interrupter, &mut progress)
        .remove(0);
    assert_eq!(episode.timesteps.len(), 2);
    assert!(!episode.timesteps[1].done);
    assert_eq!(
        episode.timesteps[1]
            .transition
            .done
            .clone()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap(),
        vec![1.]
    );
}
#[test]
fn off_policy_real_autodiff_learning_sync_async_lazy_metrics_and_evaluation() {
    for mode in 0..3 {
        let memory = InMemoryMetricLogger::new();
        let processor = metrics().with_logger(memory.clone());
        let config = OffPolicyConfig {
            num_envs: 2,
            autobatch_size: 2,
            replay_buffer_size: 64,
            train_interval: 4,
            train_steps: 2,
            eval_interval: 64,
            eval_episodes: 2,
            train_batch_size: 16,
            warmup_steps: 16,
        };
        let training = RLTraining::new(init, device())
            .off_policy(config)
            .num_steps(257);
        let training = match mode {
            0 => training.with_metrics(processor),
            1 => training.with_async_metrics(processor),
            _ => training.with_lazy_async_metrics(processor),
        };
        let result = training.launch(QLearner::new());
        assert!(result.error.is_none(), "{:?}", result.error);
        assert!(result.interrupted.is_none());
        assert_eq!(result.state.steps_processed, 257);
        assert_eq!(result.state.optimizer_steps, 122);
        assert_eq!(result.state.evaluations, 4);
        assert_eq!(result.learner.steps, 122);
        let mut policy = result.policy;
        for state in 0..2 {
            let (action, _) = policy.action(Observation(state).to_observation(&device()), true);
            assert_eq!(
                action.into_data().convert::<i64>().to_vec::<i64>().unwrap(),
                vec![state as i64]
            );
        }
        assert_eq!(memory.entries("Loss", 1, Split::Train).len(), 122);
        assert_eq!(memory.entries("Episode length", 1, Split::Valid).len(), 8);
        close(
            result
                .metrics
                .find_metric("Episode length", 1, Split::Valid)
                .unwrap(),
            3.,
        );
        assert!(
            !memory
                .entries("Exploration rate", 1, Split::Train)
                .is_empty()
        );
        assert!(
            !memory
                .entries("Exploration rate", 1, Split::Valid)
                .is_empty()
        );
    }
}
#[test]
fn off_policy_schedule_warmup_overshoot_file_metrics_and_checkpoint_resume() {
    let directory = directory("resume");
    let config = OffPolicyConfig {
        train_interval: 3,
        train_steps: 2,
        eval_interval: 5,
        train_batch_size: 2,
        warmup_steps: 4,
        ..Default::default()
    };
    let logger = SerializedLogger::new(&directory);
    let result = RLTraining::new(init, device())
        .off_policy(config.clone())
        .num_steps(11)
        .with_lazy_async_metrics(metrics().with_logger(logger))
        .with_checkpointer(checkpointer(&directory))
        .launch(QLearner::new());
    assert!(result.error.is_none(), "{:?}", result.error);
    assert_eq!(
        result.state,
        RLLoopState {
            steps_processed: 13,
            optimizer_steps: 8,
            evaluations: 2,
            last_checkpoint: Some(10)
        }
    );
    let mut saved = checkpointer(&directory);
    assert_eq!(saved.epochs().unwrap(), vec![5, 10]);
    let (policy, learner) = saved.restore_records(10).unwrap();
    assert_eq!(policy.1, 4);
    assert_eq!(learner.2, 6);
    let expected_weights = QNet::<TestBackend>::new()
        .load_record(policy.0)
        .weights
        .val()
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .unwrap();
    let restored = saved.load_checkpoint(QLearner::new(), 10).unwrap();
    assert_eq!(restored.steps, 6);
    assert_eq!(
        restored
            .net
            .valid()
            .weights
            .val()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap(),
        expected_weights
    );
    let resumed = RLTraining::new(init, device())
        .off_policy(config)
        .num_steps(20)
        .with_checkpointer(checkpointer(&directory))
        .checkpoint(10)
        .with_lazy_metrics(metrics())
        .launch(QLearner::new());
    assert!(resumed.error.is_none(), "{:?}", resumed.error);
    assert_eq!(resumed.state.steps_processed, 20);
    assert_eq!(resumed.learner.steps, 12);
    assert_eq!(checkpointer(&directory).epochs().unwrap(), vec![15, 20]);
    let summary = LearnerSummary::new(&directory, &["Loss", "Episode length"]).unwrap();
    assert_eq!(summary.epochs, 1);
    assert_eq!(
        summary
            .metrics
            .train
            .iter()
            .find(|metric| metric.name == "Loss")
            .unwrap()
            .entries
            .len(),
        1
    );
    std::fs::remove_dir_all(directory).unwrap();
}

struct StopProcessor {
    interrupter: Interrupter,
    stop_at: usize,
    steps: usize,
    ends: Arc<Mutex<usize>>,
}
impl RLEventProcessor<RegressionOutput<TestBackend>, ActionContext> for StopProcessor {
    fn process_train(
        &mut self,
        event: RLEvent<RegressionOutput<TestBackend>, ActionContext>,
    ) -> Result<(), TrainingError> {
        match event {
            RLEvent::EnvStep(_) => {
                self.steps += 1;
                if self.steps == self.stop_at {
                    self.interrupter.stop(Some("episode budget"));
                }
            }
            RLEvent::End(_) => *self.ends.lock().unwrap() += 1,
            _ => {}
        }
        Ok(())
    }
    fn process_valid(
        &mut self,
        _: AgentEvaluationEvent<ActionContext>,
    ) -> Result<(), TrainingError> {
        Ok(())
    }
}
#[test]
fn rl_interruption_and_real_logger_error_are_returned_with_end_event() {
    for before in [true, false] {
        let interrupter = Interrupter::new();
        if before {
            interrupter.stop(Some("before launch"));
        }
        let ends = Arc::new(Mutex::new(0));
        let processor = StopProcessor {
            interrupter: interrupter.clone(),
            stop_at: 3,
            steps: 0,
            ends: ends.clone(),
        };
        let result = RLTraining::new(init, device())
            .num_steps(20)
            .with_interrupter(interrupter)
            .with_event_processor(processor, Default::default())
            .launch(QLearner::new());
        assert!(result.error.is_none());
        assert!(result.interrupted.is_some());
        assert_eq!(result.state.steps_processed, if before { 1 } else { 4 });
        assert_eq!(*ends.lock().unwrap(), 1);
    }
    let path = directory("blocked-logger");
    std::fs::write(&path, "not a directory").unwrap();
    let result = RLTraining::new(init, device())
        .num_steps(32)
        .with_lazy_async_metrics(metrics().with_logger(SerializedLogger::new(&path)))
        .launch(QLearner::new());
    assert!(result.error.is_some());
    assert!(result.interrupted.is_none());
    std::fs::remove_file(path).unwrap();
    let result = RLTraining::new(init, device())
        .num_steps(10)
        .checkpoint(5)
        .launch(QLearner::new());
    assert!(result.error.unwrap().0.contains("requires a checkpointer"));
    assert_eq!(result.learner.steps, 0);
}

struct Deferred(usize);
impl ItemLazy for Deferred {
    type Synced = usize;
    fn sync(self) -> Result<usize, TrainingError> {
        Ok(self.0)
    }
}
struct Capture(Arc<Mutex<Vec<usize>>>);
impl RLEventProcessor<usize, usize> for Capture {
    fn process_train(&mut self, event: RLEvent<usize, usize>) -> Result<(), TrainingError> {
        match event {
            RLEvent::TrainStep(item) | RLEvent::EnvStep(item) => {
                self.0.lock().unwrap().push(item.item)
            }
            _ => {}
        }
        Ok(())
    }
    fn process_valid(&mut self, event: AgentEvaluationEvent<usize>) -> Result<(), TrainingError> {
        if let AgentEvaluationEvent::EnvStep(item) = event {
            self.0.lock().unwrap().push(item.item);
        }
        Ok(())
    }
}
#[test]
fn rl_lazy_worker_preserves_order_with_distinct_synced_types() {
    let capture = Arc::new(Mutex::new(Vec::new()));
    let mut processor =
        AsyncRLProcessor::<Deferred, Deferred>::with_lazy_capacity(Capture(capture.clone()), 2);
    processor
        .process_train(RLEvent::TrainStep(EvaluationItem::new(
            Deferred(1),
            Progress::new(1, 3),
            None,
        )))
        .unwrap();
    processor
        .process_train(RLEvent::EnvStep(EvaluationItem::new(
            Deferred(2),
            Progress::new(2, 3),
            None,
        )))
        .unwrap();
    processor
        .process_valid(AgentEvaluationEvent::EnvStep(EvaluationItem::new(
            Deferred(3),
            Progress::new(3, 3),
            None,
        )))
        .unwrap();
    processor.finish().unwrap();
    assert_eq!(*capture.lock().unwrap(), vec![1, 2, 3]);
}

#[test]
fn rl_checkpoint_composition_keeps_recent_and_best_actual_file_pairs() {
    use crate::metric::{Direction, EventStore, MetricEntry};
    let directory = directory("best-checkpoints");
    let store = EventStore::default();
    let strategy = ComposedCheckpointingStrategy::builder()
        .add(LastNCheckpointingStrategy::new(2))
        .add(MetricCheckpointingStrategy::new(
            &EpisodeLengthMetric::new(),
            Direction::Lowest,
            Split::Valid,
        ))
        .build();
    let mut saved = checkpointer(&directory).with_strategy(strategy);
    let mut learner = QLearner::new();
    let expected = [vec![1], vec![1, 2], vec![1, 2, 3], vec![3, 4], vec![4, 5]];
    for (index, value) in [3., 4., 5., 1., 2.].into_iter().enumerate() {
        let step = index + 1;
        store.record(
            step,
            Split::Valid,
            vec![MetricEntry {
                name: "Episode length".into(),
                value: Some(value),
            }],
        );
        learner.steps = step;
        assert!(
            saved
                .checkpoint(&learner.policy().state(), &learner, step, &store)
                .unwrap()
        );
        assert_eq!(saved.epochs().unwrap(), expected[index]);
    }
    let restored = saved.load_checkpoint(QLearner::new(), 4).unwrap();
    assert_eq!(restored.steps, 4);
    assert!(saved.restore_records(1).is_err());
    drop(saved);
    assert_eq!(checkpointer(&directory).epochs().unwrap(), vec![4, 5]);
    let mut saved = checkpointer(&directory);
    let blocked = directory.join("learner/epoch-6.bin");
    std::fs::create_dir(&blocked).unwrap();
    assert!(
        saved
            .checkpoint(&learner.policy().state(), &learner, 6, &store)
            .is_err()
    );
    assert_eq!(saved.epochs().unwrap(), vec![4, 5]);
    assert!(saved.restore_records(6).is_err());
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn rl_checkpoint_metric_selection_edges_and_training_store_integration() {
    use crate::metric::{Direction, EventStore, MetricEntry};
    use CheckpointingAction::{Delete, Save};
    let store = EventStore::default();
    let metric = EpisodeLengthMetric::new();
    let mut lowest = MetricCheckpointingStrategy::new(&metric, Direction::Lowest, Split::Valid);
    assert_eq!(lowest.checkpointing(4, &store), vec![Save]);
    assert_eq!(lowest.checkpointing(5, &store), vec![Delete(4), Save]);
    store.record(
        1,
        Split::Valid,
        vec![MetricEntry {
            name: "Episode length".into(),
            value: Some(2.),
        }],
    );
    store.record(
        3,
        Split::Valid,
        vec![MetricEntry {
            name: "Episode length".into(),
            value: Some(1.),
        }],
    );
    assert_eq!(lowest.checkpointing(3, &store), vec![Delete(5)]);
    store.record(
        2,
        Split::Valid,
        vec![MetricEntry {
            name: "Episode length".into(),
            value: Some(2.),
        }],
    );
    assert_eq!(lowest.checkpointing(3, &store), vec![Delete(1), Save]);
    let mut highest = MetricCheckpointingStrategy::new(&metric, Direction::Highest, Split::Valid);
    assert_eq!(highest.checkpointing(2, &store), vec![]);
    assert_eq!(highest.checkpointing(1, &store), vec![Save]);
    let mut last = LastNCheckpointingStrategy::new(2);
    assert_eq!(last.checkpointing(10, &store), vec![Save, Delete(8)]);
    let mut empty = ComposedCheckpointingStrategy::builder().build();
    assert!(empty.checkpointing(1, &store).is_empty());

    let directory = directory("metric-checkpoints");
    let result = RLTraining::new(init, device())
        .num_steps(13)
        .off_policy(OffPolicyConfig {
            train_interval: 3,
            train_batch_size: 2,
            eval_interval: 5,
            ..Default::default()
        })
        .with_lazy_async_metrics(metrics())
        .with_checkpointer(checkpointer(&directory).with_strategy(
            MetricCheckpointingStrategy::new(&metric, Direction::Lowest, Split::Valid),
        ))
        .launch(QLearner::new());
    assert!(result.error.is_none(), "{:?}", result.error);
    assert_eq!(result.state.evaluations, 2);
    assert_eq!(
        result
            .metrics
            .find_metric("Episode length", 1, Split::Valid),
        Some(3.)
    );
    assert_eq!(result.state.last_checkpoint, None);
    assert!(checkpointer(&directory).epochs().unwrap().is_empty());
    if directory.exists() {
        std::fs::remove_dir_all(directory).unwrap();
    }
}
