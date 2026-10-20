use crate::{
    Split,
    metric::{Direction, EventStore, Metric},
};
use std::collections::HashSet;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckpointingAction {
    Save,
    Delete(usize),
}
pub trait CheckpointingStrategy: Send {
    fn checkpointing(&mut self, step: usize, store: &EventStore) -> Vec<CheckpointingAction>;
}
impl CheckpointingStrategy for Box<dyn CheckpointingStrategy> {
    fn checkpointing(&mut self, step: usize, store: &EventStore) -> Vec<CheckpointingAction> {
        self.as_mut().checkpointing(step, store)
    }
}
pub struct LastNCheckpointingStrategy {
    num_keep: usize,
}
impl LastNCheckpointingStrategy {
    pub fn new(num_keep: usize) -> Self {
        Self { num_keep }
    }
}
impl CheckpointingStrategy for LastNCheckpointingStrategy {
    fn checkpointing(&mut self, step: usize, _: &EventStore) -> Vec<CheckpointingAction> {
        let mut actions = vec![CheckpointingAction::Save];
        if let Some(old) = step.checked_sub(self.num_keep)
            && old > 0
        {
            actions.push(CheckpointingAction::Delete(old));
        }
        actions
    }
}
pub struct MetricCheckpointingStrategy {
    current: Option<usize>,
    name: String,
    direction: Direction,
    split: Split,
}
impl MetricCheckpointingStrategy {
    pub fn new<M: Metric>(metric: &M, direction: Direction, split: Split) -> Self {
        Self {
            current: None,
            name: metric.name().into(),
            direction,
            split,
        }
    }
}
impl CheckpointingStrategy for MetricCheckpointingStrategy {
    fn checkpointing(&mut self, step: usize, store: &EventStore) -> Vec<CheckpointingAction> {
        let mut values = Vec::new();
        let mut epoch = 1;
        while let Some(value) = store.find_metric(&self.name, epoch, self.split) {
            values.push(value);
            epoch += 1;
        }
        let best = if values.is_empty() {
            step
        } else {
            let mut best = epoch;
            let mut best_value = match self.direction {
                Direction::Lowest => f64::MAX,
                Direction::Highest => f64::MIN,
            };
            for (index, value) in values.into_iter().enumerate() {
                let better = match self.direction {
                    Direction::Lowest => value < best_value,
                    Direction::Highest => value > best_value,
                };
                if better {
                    best = index + 1;
                    best_value = value;
                }
            }
            best
        };
        let mut actions = Vec::new();
        if let Some(previous) = self.current
            && previous != best
        {
            actions.push(CheckpointingAction::Delete(previous));
        }
        if best == step {
            actions.push(CheckpointingAction::Save);
        }
        self.current = Some(best);
        actions
    }
}
pub struct ComposedCheckpointingStrategy {
    strategies: Vec<Box<dyn CheckpointingStrategy>>,
    deleted: Vec<HashSet<usize>>,
}
#[derive(Default)]
pub struct ComposedCheckpointingStrategyBuilder {
    strategies: Vec<Box<dyn CheckpointingStrategy>>,
}
impl ComposedCheckpointingStrategyBuilder {
    #[allow(clippy::should_implement_trait)]
    pub fn add<S: CheckpointingStrategy + 'static>(mut self, strategy: S) -> Self {
        self.strategies.push(Box::new(strategy));
        self
    }
    pub fn build(self) -> ComposedCheckpointingStrategy {
        ComposedCheckpointingStrategy {
            deleted: self.strategies.iter().map(|_| HashSet::new()).collect(),
            strategies: self.strategies,
        }
    }
}
impl ComposedCheckpointingStrategy {
    pub fn builder() -> ComposedCheckpointingStrategyBuilder {
        ComposedCheckpointingStrategyBuilder::default()
    }
}
impl CheckpointingStrategy for ComposedCheckpointingStrategy {
    fn checkpointing(&mut self, step: usize, store: &EventStore) -> Vec<CheckpointingAction> {
        let mut save = false;
        let mut candidates = Vec::new();
        for (index, strategy) in self.strategies.iter_mut().enumerate() {
            let actions = strategy.checkpointing(step, store);
            if actions.is_empty() {
                self.deleted[index].insert(step);
            }
            for action in actions {
                match action {
                    CheckpointingAction::Save => save = true,
                    CheckpointingAction::Delete(old) => {
                        self.deleted[index].insert(old);
                        candidates.push(old);
                    }
                }
            }
        }
        let mut actions = Vec::new();
        if save {
            actions.push(CheckpointingAction::Save);
        }
        for old in candidates {
            if self.deleted.iter().all(|deleted| deleted.contains(&old)) {
                actions.push(CheckpointingAction::Delete(old));
                for deleted in &mut self.deleted {
                    deleted.remove(&old);
                }
            }
        }
        actions
    }
}
