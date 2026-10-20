use super::{CheckpointingAction, CheckpointingStrategy, Policy, PolicyLearner, PolicyState};
use crate::{Checkpointer, TrainingError, metric::EventStore};
use ruda_model::tensor::backend::Backend;

pub type RLPolicyRecord<B, L> =
    <<<L as PolicyLearner<B>>::InnerPolicy as Policy<B>>::PolicyState as PolicyState>::Record;
pub struct RLCheckpointer<B: Backend, L: PolicyLearner<B>> {
    policy: Box<dyn Checkpointer<RLPolicyRecord<B, L>>>,
    learner: Box<dyn Checkpointer<L::Record>>,
    keep: usize,
    strategy: Option<Box<dyn CheckpointingStrategy>>,
}
impl<B: Backend, L: PolicyLearner<B>> RLCheckpointer<B, L> {
    pub fn new<P, C>(policy: P, learner: C, keep: usize) -> Self
    where
        P: Checkpointer<RLPolicyRecord<B, L>> + 'static,
        C: Checkpointer<L::Record> + 'static,
    {
        assert!(keep > 0, "At least one RL checkpoint must be retained");
        Self {
            policy: Box::new(policy),
            learner: Box::new(learner),
            keep,
            strategy: None,
        }
    }
    pub fn with_strategy<S: CheckpointingStrategy + 'static>(mut self, strategy: S) -> Self {
        self.strategy = Some(Box::new(strategy));
        self
    }
    pub fn checkpoint(
        &mut self,
        policy: &<L::InnerPolicy as Policy<B>>::PolicyState,
        learner: &L,
        step: usize,
        store: &EventStore,
    ) -> Result<bool, TrainingError> {
        if let Some(strategy) = &mut self.strategy {
            let actions = strategy.checkpointing(step, store);
            let mut saved = false;
            for action in actions {
                match action {
                    CheckpointingAction::Save => {
                        self.policy.save(step, policy.clone().into_record())?;
                        self.learner.save(step, learner.record())?;
                        saved = true;
                    }
                    CheckpointingAction::Delete(old) => self.delete(old)?,
                }
            }
            return Ok(saved);
        }
        self.policy.save(step, policy.clone().into_record())?;
        self.learner.save(step, learner.record())?;
        let mut saved = self.epochs()?;
        saved.retain(|previous| *previous != step);
        saved.push(step);
        for old in saved.iter().take(saved.len().saturating_sub(self.keep)) {
            self.delete(*old)?;
        }
        Ok(true)
    }
    pub fn restore_records(
        &mut self,
        step: usize,
    ) -> Result<(RLPolicyRecord<B, L>, L::Record), TrainingError> {
        Ok((self.policy.restore(step)?, self.learner.restore(step)?))
    }
    pub fn load_checkpoint(&mut self, learner: L, step: usize) -> Result<L, TrainingError> {
        let (policy, record) = self.restore_records(step)?;
        let policy = learner.policy().load_record(policy);
        let mut learner = learner.load_record(record);
        learner.update_policy(policy);
        Ok(learner)
    }
    pub fn delete(&mut self, step: usize) -> Result<(), TrainingError> {
        self.policy.delete(step)?;
        self.learner.delete(step)
    }
    pub fn epochs(&self) -> Result<Vec<usize>, TrainingError> {
        let policies = self.policy.epochs()?;
        let mut learners = self.learner.epochs()?;
        learners.retain(|step| policies.contains(step));
        learners.sort_unstable();
        learners.dedup();
        Ok(learners)
    }
}
