use alloc::vec::Vec;
use core::ops::Range;
use smallvec::SmallVec;

use super::{ErrorGraph, FailureId};

type Ranges = SmallVec<[Range<u64>; 1]>;

#[derive(Debug, Default)]
pub(crate) struct Taint {
    entries: SmallVec<[Tainted; 1]>,
}

#[derive(Debug)]
struct Tainted {
    failure: ErrorGraph,
    ranges: Ranges,
    unallocated: bool,
}

impl Drop for Tainted {
    fn drop(&mut self) {
        self.failure.untag();
    }
}

impl Taint {
    pub(crate) fn claim(&mut self, range: Range<u64>, failure: ErrorGraph) {
        if range.is_empty() {
            return;
        }
        self.entries.retain_mut(|entry| {
            if entry.failure.id() != failure.id() {
                subtract(&mut entry.ranges, &range);
            }
            entry.unallocated || !entry.ranges.is_empty()
        });
        if let Some(entry) = self.entries.iter_mut().find(|entry| entry.failure.id() == failure.id()) {
            add(&mut entry.ranges, range);
        } else {
            failure.tag();
            self.entries.push(Tainted { failure, ranges: Ranges::from_buf([range]), unallocated: false });
        }
    }

    pub(crate) fn claim_unallocated(&mut self, failure: ErrorGraph) {
        self.entries.clear();
        failure.tag();
        self.entries.push(Tainted { failure, ranges: Ranges::new(), unallocated: true });
    }

    pub(crate) fn clear_unallocated(&mut self) {
        self.entries.retain_mut(|entry| {
            entry.unallocated = false;
            !entry.ranges.is_empty()
        });
    }

    pub(crate) fn failures(&self, range: Range<u64>) -> Vec<ErrorGraph> {
        self.entries.iter()
            .filter(|entry| entry.unallocated || entry.ranges.iter().any(|held| overlaps(held, &range)))
            .map(|entry| entry.failure.clone())
            .collect()
    }

    pub(crate) fn written(&mut self, range: Range<u64>) {
        if range.is_empty() {
            return;
        }
        self.entries.retain_mut(|entry| {
            subtract(&mut entry.ranges, &range);
            entry.unallocated || !entry.ranges.is_empty()
        });
    }

    pub(crate) fn release(&mut self, range: Range<u64>, failure: FailureId) {
        self.entries.retain_mut(|entry| {
            if entry.failure.id() == failure {
                subtract(&mut entry.ranges, &range);
                entry.unallocated = false;
            }
            entry.unallocated || !entry.ranges.is_empty()
        });
    }
}

fn overlaps(a: &Range<u64>, b: &Range<u64>) -> bool {
    a.start.max(b.start) < a.end.min(b.end)
}

fn add(ranges: &mut Ranges, mut new: Range<u64>) {
    ranges.retain(|held| {
        let fuses = held.start <= new.end && new.start <= held.end;
        if fuses {
            new.start = new.start.min(held.start);
            new.end = new.end.max(held.end);
        }
        !fuses
    });
    let at = ranges.iter().position(|held| new.end < held.start).unwrap_or(ranges.len());
    ranges.insert(at, new);
}

fn subtract(ranges: &mut Ranges, cut: &Range<u64>) {
    let mut index = 0;
    while index < ranges.len() {
        let held = ranges[index].clone();
        if !overlaps(&held, cut) {
            index += 1;
            continue;
        }
        let left = held.start..cut.start.min(held.end);
        let right = cut.end.max(held.start)..held.end;
        match (left.is_empty(), right.is_empty()) {
            (true, true) => { ranges.remove(index); }
            (false, true) => { ranges[index] = left; index += 1; }
            (true, false) => { ranges[index] = right; index += 1; }
            (false, false) => {
                ranges[index] = left;
                ranges.insert(index + 1, right);
                index += 2;
            }
        }
    }
}
