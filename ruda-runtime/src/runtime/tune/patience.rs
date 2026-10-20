use alloc::vec::Vec;
use core::time::Duration;

use hashbrown::HashMap;

use crate::runtime::tune::sampler::TimeChange;
use crate::runtime::tune::{Batch, GroupId, Patience};

/// The [`PatienceTracker`] of every group with a [`Patience`] in a [`Batch`], and the groups
/// that brought each slot of the batch in.
#[derive(Debug)]
pub(crate) struct PatienceTable {
    groups: Vec<Vec<GroupId>>,
    trackers: HashMap<GroupId, PatienceTracker>,
}

impl PatienceTable {
    pub(crate) fn new(batch: &Batch) -> Self {
        Self {
            groups: batch
                .entries
                .iter()
                .map(|entry| entry.groups.clone())
                .collect(),
            trackers: batch
                .patience
                .iter()
                .map(|(group, patience)| (*group, PatienceTracker::new(*patience)))
                .collect(),
        }
    }

    /// Count the best time of the candidate at `slot` against each of the groups that brought
    /// it in.
    pub(crate) fn record(&mut self, slot: usize, time: Duration) {
        for group in &self.groups[slot] {
            if let Some(tracker) = self.trackers.get_mut(group) {
                tracker.record(time);
            }
        }
    }

    /// Whether the candidate at `slot` is skipped: every group that brought it in ran out of
    /// patience. One that a group without patience brought in, or that no group did, never is.
    pub(crate) fn skips(&self, slot: usize) -> bool {
        let groups = &self.groups[slot];
        !groups.is_empty()
            && groups.iter().all(|group| {
                self.trackers
                    .get(group)
                    .is_some_and(PatienceTracker::exhausted)
            })
    }
}

/// How one group fares against its [`Patience`]: how many of its members were measured, and
/// how many in a row failed to improve on its leader.
#[derive(Debug)]
struct PatienceTracker {
    patience: Patience,
    measured: usize,
    misses: usize,
    /// The last time that was an improvement.
    leader: Option<Duration>,
}

impl PatienceTracker {
    fn new(patience: Patience) -> Self {
        Self {
            patience,
            measured: 0,
            misses: 0,
            leader: None,
        }
    }

    /// Count one measured member by its best time.
    fn record(&mut self, time: Duration) {
        self.measured += 1;

        match self.leader.map(|leader| TimeChange::of(time, leader)) {
            None | Some(TimeChange::Improvement) => {
                self.leader = Some(time);
                self.misses = 0;
            }
            Some(TimeChange::Neutral | TimeChange::Regression) => self.misses += 1,
        }
    }

    /// Whether the group stops: one member is always measured, whatever the patience says.
    fn exhausted(&self) -> bool {
        self.measured >= self.patience.min_measured.max(1)
            && self.misses >= self.patience.max_misses
    }
}
