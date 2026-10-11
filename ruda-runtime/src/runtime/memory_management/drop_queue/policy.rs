use ruda_core::bytes::Bytes;

/// Defines the thresholds that determine when a [`PendingDropQueue`] should be
/// flushed.
///
/// A flush is triggered when **either** limit is exceeded — whichever comes
/// first. Set a field to `u32::MAX` / `usize::MAX` to effectively disable it.
#[derive(Debug)]
pub struct FlushingPolicy {
    /// Flush when this many allocations have been staged.
    pub max_bytes_count: u32,
    /// Flush when the total staged size reaches this many bytes.
    pub max_bytes_size: u32,
}

impl Default for FlushingPolicy {
    fn default() -> Self {
        Self {
            max_bytes_count: 64,
            max_bytes_size: 64 * 1024 * 1024, // 64 MiB
        }
    }
}

/// Tracks staged allocations and evaluates them against a [`FlushingPolicy`].
#[derive(Default, Debug)]
pub(crate) struct FlushingPolicyState {
    bytes_count: u32,
    bytes_size: usize,
}

impl FlushingPolicyState {
    /// Record a newly staged [`Bytes`] allocation.
    pub(crate) fn register(&mut self, bytes: &Bytes) {
        self.bytes_count = self.bytes_count.saturating_add(1);
        self.bytes_size = self.bytes_size.saturating_add(bytes.len());
    }

    /// Reset all counters, typically called after a flush.
    pub(crate) fn reset(&mut self) {
        self.bytes_count = 0;
        self.bytes_size = 0;
    }

    /// Returns `true` if either threshold in `policy` has been reached.
    pub(crate) fn should_flush(&self, policy: &FlushingPolicy) -> bool {
        self.bytes_count >= policy.max_bytes_count
            || self.bytes_size >= policy.max_bytes_size as usize
    }
}

#[cfg(test)]
mod policy_tests {
    use std::vec;

    use super::*;

    fn policy() -> FlushingPolicy {
        FlushingPolicy {
            max_bytes_count: 4,
            max_bytes_size: 100,
        }
    }

    fn state() -> FlushingPolicyState {
        FlushingPolicyState {
            bytes_count: 0,
            bytes_size: 0,
        }
    }

    #[test]
    fn no_flush_when_below_both_thresholds() {
        let s = state();
        assert!(!s.should_flush(&policy()));
    }

    #[test]
    fn flush_when_count_threshold_reached() {
        let mut s = state();
        for _ in 0..4 {
            s.register(&Bytes::from_elems(vec![0u8]));
        }
        assert!(s.should_flush(&policy()));
    }

    #[test]
    fn flush_when_size_threshold_reached() {
        let mut s = state();
        s.register(&Bytes::from_elems(vec![0u8; 101]));
        assert!(s.should_flush(&policy()));
    }

    #[test]
    fn flush_triggered_by_whichever_limit_comes_first() {
        let mut s = state();
        // Only 2 allocations but already over the size limit.
        s.register(&Bytes::from_elems(vec![0u8; 60]));
        s.register(&Bytes::from_elems(vec![0u8; 60]));
        assert!(s.should_flush(&policy()));
    }

    #[test]
    fn reset_clears_state() {
        let mut s = state();
        for _ in 0..4 {
            s.register(&Bytes::from_elems(vec![0u8]));
        }
        assert!(s.should_flush(&policy()));
        s.reset();
        assert!(!s.should_flush(&policy()));
    }

    #[test]
    fn counters_do_not_wrap_before_flush() {
        let mut s = FlushingPolicyState {
            bytes_count: u32::MAX,
            bytes_size: u32::MAX as usize - 1,
        };
        s.register(&Bytes::from_elems(vec![0u8; 8]));
        assert_eq!(s.bytes_count, u32::MAX);
        assert_eq!(s.bytes_size, (u32::MAX as usize - 1).saturating_add(8));
        assert!(s.should_flush(&FlushingPolicy {
            max_bytes_count: u32::MAX,
            max_bytes_size: u32::MAX,
        }));
        s.bytes_size = usize::MAX;
        s.register(&Bytes::from_elems(vec![0u8; 8]));
        assert_eq!(s.bytes_size, usize::MAX);
    }
}
