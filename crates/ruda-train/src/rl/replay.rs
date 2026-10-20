use ruda_model::tensor::{Distribution, Int, Tensor, backend::Backend};

#[derive(Clone)]
pub struct Transition<B: Backend, S, A> {
    pub state: S,
    pub next_state: S,
    pub action: A,
    pub reward: Tensor<B, 1>,
    pub done: Tensor<B, 1>,
}
pub struct TransitionBatch<B: Backend, S, A> {
    pub states: S,
    pub next_states: S,
    pub actions: A,
    pub rewards: Tensor<B, 2>,
    pub dones: Tensor<B, 2>,
}
pub trait SliceAccess<B: Backend>: Clone + Sized {
    fn zeros_like(sample: &Self, capacity: usize, device: &B::Device) -> Self;
    fn select(self, dim: usize, indices: Tensor<B, 1, Int>) -> Self;
    fn slice_assign_inplace(&mut self, index: usize, value: Self);
}
macro_rules! tensor_slice_access {
    ($($kind:ty),*) => {$ (
        impl<B: Backend> SliceAccess<B> for Tensor<B, 2, $kind> {
            fn zeros_like(sample: &Self, capacity: usize, device: &B::Device) -> Self {
                Tensor::zeros([capacity, sample.dims()[1]], device)
            }
            fn select(self, dim: usize, indices: Tensor<B, 1, Int>) -> Self {
                Tensor::select(self, dim, indices)
            }
            fn slice_assign_inplace(&mut self, index: usize, value: Self) {
                let width = self.dims()[1];
                self.inplace(|tensor| tensor.slice_assign([index..index+1, 0..width], value));
            }
        }
    )*};
}
tensor_slice_access!(ruda_model::tensor::Float, Int);

pub struct TransitionBuffer<B: Backend, S: SliceAccess<B>, A: SliceAccess<B>> {
    states: Option<S>,
    next_states: Option<S>,
    actions: Option<A>,
    rewards: Option<Tensor<B, 2>>,
    dones: Option<Tensor<B, 2>>,
    capacity: usize,
    write_head: usize,
    len: usize,
    device: B::Device,
}
impl<B: Backend, S: SliceAccess<B>, A: SliceAccess<B>> TransitionBuffer<B, S, A> {
    pub fn new(capacity: usize, device: &B::Device) -> Self {
        Self {
            states: None,
            next_states: None,
            actions: None,
            rewards: None,
            dones: None,
            capacity,
            write_head: 0,
            len: 0,
            device: device.clone(),
        }
    }
    pub fn push(&mut self, state: S, next_state: S, action: A, reward: f32, done: bool) {
        if self.states.is_none() {
            self.states = Some(S::zeros_like(&state, self.capacity, &self.device));
            self.next_states = Some(S::zeros_like(&next_state, self.capacity, &self.device));
            self.actions = Some(A::zeros_like(&action, self.capacity, &self.device));
            self.rewards = Some(Tensor::zeros([self.capacity, 1], &self.device));
            self.dones = Some(Tensor::zeros([self.capacity, 1], &self.device));
        }
        let i = self.write_head % self.capacity;
        self.states.as_mut().unwrap().slice_assign_inplace(i, state);
        self.next_states
            .as_mut()
            .unwrap()
            .slice_assign_inplace(i, next_state);
        self.actions
            .as_mut()
            .unwrap()
            .slice_assign_inplace(i, action);
        self.rewards.as_mut().unwrap().inplace(|tensor| {
            tensor.slice_assign(
                [i..i + 1, 0..1],
                Tensor::from_data([[reward]], &self.device),
            )
        });
        self.dones.as_mut().unwrap().inplace(|tensor| {
            tensor.slice_assign(
                [i..i + 1, 0..1],
                Tensor::from_data([[if done { 1f32 } else { 0f32 }]], &self.device),
            )
        });
        self.write_head += 1;
        self.len = (self.len + 1).min(self.capacity);
    }
    pub fn sample(&self, batch_size: usize) -> TransitionBatch<B, S, A> {
        assert!(batch_size <= self.len, "batch_size exceeds buffer length");
        let indices = Tensor::<B, 1>::random(
            [batch_size],
            Distribution::Uniform(0., self.len as f64),
            &self.device,
        )
        .int();
        TransitionBatch {
            states: self
                .states
                .as_ref()
                .unwrap()
                .clone()
                .select(0, indices.clone()),
            next_states: self
                .next_states
                .as_ref()
                .unwrap()
                .clone()
                .select(0, indices.clone()),
            actions: self
                .actions
                .as_ref()
                .unwrap()
                .clone()
                .select(0, indices.clone()),
            rewards: self
                .rewards
                .as_ref()
                .unwrap()
                .clone()
                .select(0, indices.clone()),
            dones: self.dones.as_ref().unwrap().clone().select(0, indices),
        }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}
