use super::{Batchable, Policy, PolicyState};
use ruda_model::tensor::backend::Backend;
use std::{
    sync::{
        Arc,
        mpsc::{self, Sender},
    },
    thread::JoinHandle,
};

struct ActionRequest<B: Backend, P: Policy<B>> {
    observation: P::Observation,
    deterministic: bool,
    reply: Sender<(P::Action, Vec<P::ActionContext>)>,
}
struct ForwardRequest<B: Backend, P: Policy<B>> {
    observation: P::Observation,
    reply: Sender<P::ActionDistribution>,
}
enum Message<B: Backend, P: Policy<B>> {
    Action(ActionRequest<B, P>),
    Forward(ForwardRequest<B, P>),
    Update(P::PolicyState),
    State(Sender<P::PolicyState>),
    Device(B::Device),
    Load(<P::PolicyState as PolicyState>::Record),
    Increment(usize),
    Decrement(usize),
    Stop,
}
struct Worker<B: Backend, P: Policy<B>> {
    sender: Option<Sender<Message<B, P>>>,
    handle: Option<JoinHandle<()>>,
}
impl<B: Backend, P: Policy<B>> Drop for Worker<B, P> {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
pub struct AsyncPolicy<B: Backend, P: Policy<B>> {
    worker: Arc<Worker<B, P>>,
}
impl<B: Backend, P: Policy<B>> Clone for AsyncPolicy<B, P> {
    fn clone(&self) -> Self {
        Self {
            worker: self.worker.clone(),
        }
    }
}
impl<B: Backend, P: Policy<B>> AsyncPolicy<B, P> {
    pub fn new(autobatch_size: usize, policy: P) -> Self {
        let (sender, receiver) = mpsc::channel();
        let handle = std::thread::Builder::new()
            .name("ruda-policy".into())
            .spawn(move || {
                let mut policy = policy;
                let mut active: usize = 0;
                let mut actions: Vec<ActionRequest<B, P>> = Vec::new();
                let mut forwards: Vec<ForwardRequest<B, P>> = Vec::new();
                while let Ok(message) = receiver.recv() {
                    match message {
                        Message::Stop => break,
                        Message::Action(request) => {
                            actions.push(request);
                            if actions.len() >= active.min(autobatch_size) {
                                flush_actions(&mut policy, &mut actions);
                            }
                        }
                        Message::Forward(request) => {
                            forwards.push(request);
                            if forwards.len() >= active.min(autobatch_size) {
                                flush_forwards(&mut policy, &mut forwards);
                            }
                        }
                        Message::Update(state) => {
                            flush_actions(&mut policy, &mut actions);
                            flush_forwards(&mut policy, &mut forwards);
                            policy.update(state);
                        }
                        Message::State(reply) => {
                            reply
                                .send(policy.state())
                                .expect("Policy state receiver is open");
                        }
                        Message::Device(device) => policy = policy.clone().to_device(&device),
                        Message::Load(record) => {
                            flush_actions(&mut policy, &mut actions);
                            flush_forwards(&mut policy, &mut forwards);
                            policy = policy.load_record(record);
                        }
                        Message::Increment(n) => active = active.wrapping_add(n),
                        Message::Decrement(n) => {
                            active = active.wrapping_sub(n);
                            if actions.len() >= active.min(autobatch_size) {
                                flush_actions(&mut policy, &mut actions);
                            }
                            if forwards.len() >= active.min(autobatch_size) {
                                flush_forwards(&mut policy, &mut forwards);
                            }
                        }
                    }
                }
            })
            .expect("Can start policy worker");
        Self {
            worker: Arc::new(Worker {
                sender: Some(sender),
                handle: Some(handle),
            }),
        }
    }
    fn send(&self, message: Message<B, P>) {
        self.worker
            .sender
            .as_ref()
            .unwrap()
            .send(message)
            .expect("Policy worker is open");
    }
    pub fn increment_agents(&self, n: usize) {
        self.send(Message::Increment(n));
    }
    pub fn decrement_agents(&self, n: usize) {
        self.send(Message::Decrement(n));
    }
    pub(super) fn stop_failed_workers(&self) {
        let _ = self.worker.sender.as_ref().unwrap().send(Message::Stop);
    }
}
fn flush_actions<B: Backend, P: Policy<B>>(policy: &mut P, pending: &mut Vec<ActionRequest<B, P>>) {
    if pending.is_empty() {
        return;
    }
    let requests = core::mem::take(pending);
    let first = requests[0].deterministic;
    for deterministic in [first, !first] {
        let (indices, observations): (Vec<_>, Vec<_>) = requests
            .iter()
            .enumerate()
            .filter(|(_, request)| request.deterministic == deterministic)
            .map(|(i, request)| (i, request.observation.clone()))
            .unzip();
        if observations.is_empty() {
            continue;
        }
        let (actions, contexts) = policy.action(P::Observation::batch(observations), deterministic);
        let actions = actions.unbatch();
        for (group_index, index) in indices.into_iter().enumerate() {
            requests[index]
                .reply
                .send((
                    actions[group_index].clone(),
                    vec![contexts[group_index].clone()],
                ))
                .expect("Action receiver is open");
        }
    }
}
fn flush_forwards<B: Backend, P: Policy<B>>(
    policy: &mut P,
    pending: &mut Vec<ForwardRequest<B, P>>,
) {
    if pending.is_empty() {
        return;
    }
    let requests = core::mem::take(pending);
    let observations = requests
        .iter()
        .map(|request| request.observation.clone())
        .collect();
    let outputs = policy
        .forward(P::Observation::batch(observations))
        .unbatch();
    for (i, request) in requests.into_iter().enumerate() {
        request
            .reply
            .send(outputs[i].clone())
            .expect("Forward receiver is open");
    }
}
impl<B: Backend, P: Policy<B>> Policy<B> for AsyncPolicy<B, P> {
    type Observation = P::Observation;
    type ActionDistribution = P::ActionDistribution;
    type Action = P::Action;
    type ActionContext = P::ActionContext;
    type PolicyState = P::PolicyState;
    fn forward(&mut self, observation: Self::Observation) -> Self::ActionDistribution {
        let (reply, receive) = mpsc::channel();
        self.send(Message::Forward(ForwardRequest { observation, reply }));
        receive
            .recv()
            .expect("Policy worker returns forward result")
    }
    fn action(
        &mut self,
        observation: Self::Observation,
        deterministic: bool,
    ) -> (Self::Action, Vec<Self::ActionContext>) {
        let (reply, receive) = mpsc::channel();
        self.send(Message::Action(ActionRequest {
            observation,
            deterministic,
            reply,
        }));
        receive.recv().expect("Policy worker returns action")
    }
    fn update(&mut self, state: Self::PolicyState) {
        self.send(Message::Update(state));
    }
    fn state(&self) -> Self::PolicyState {
        let (reply, receive) = mpsc::channel();
        self.send(Message::State(reply));
        receive.recv().expect("Policy worker returns state")
    }
    fn to_device(self, device: &B::Device) -> Self {
        self.send(Message::Device(device.clone()));
        self
    }
    fn load_record(self, record: <Self::PolicyState as PolicyState>::Record) -> Self {
        self.send(Message::Load(record));
        self
    }
}
