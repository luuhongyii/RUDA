#[cfg(not(feature = "std"))]
use spin::{Mutex as MutexImported, MutexGuard};
#[cfg(feature = "std")]
use std::sync::{Mutex as MutexImported, MutexGuard};

#[cfg(not(feature = "std"))]
pub use spin::Lazy as LazyLock;
#[cfg(feature = "std")]
pub use std::sync::LazyLock;

#[cfg(target_has_atomic = "ptr")]
pub use alloc::sync::Arc;
#[cfg(not(target_has_atomic = "ptr"))]
pub use portable_atomic_util::Arc;

#[cfg(target_has_atomic = "ptr")]
pub use core::sync::atomic::{AtomicU8, AtomicU32, Ordering};
#[cfg(not(target_has_atomic = "ptr"))]
pub use portable_atomic::{AtomicU8, AtomicU32, Ordering};

/// A mutual exclusion primitive useful for protecting shared data
///
/// This mutex will block threads waiting for the lock to become available. The
/// mutex can also be statically initialized or created via a [`Mutex::new`]
///
/// [Mutex] wrapper to make `spin::Mutex` API compatible with `std::sync::Mutex` to swap
#[derive(Debug, Default)]
pub struct Mutex<T> {
    inner: MutexImported<T>,
}

impl<T> Mutex<T> {
    /// Creates a new mutex in an unlocked state ready for use.
    #[inline(always)]
    pub const fn new(value: T) -> Self {
        Self {
            inner: MutexImported::new(value),
        }
    }

    /// Locks the mutex blocking the current thread until it is able to do so.
    ///
    /// Locking cannot fail, so no `Result` is returned. A poisoned lock is
    /// recovered rather than reported: a panic under a guard must not turn
    /// every later lock into a panic, matching the unwind behavior of the spin
    /// implementation used off-std.
    #[inline(always)]
    pub fn lock(&self) -> MutexGuard<'_, T> {
        #[cfg(not(feature = "std"))]
        {
            self.inner.lock()
        }

        #[cfg(feature = "std")]
        {
            self.inner
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        }
    }

    /// Consumes the mutex, returning the value it protected. A poisoned lock
    /// is recovered, as in [`lock`](Self::lock).
    #[inline(always)]
    pub fn into_inner(self) -> T {
        #[cfg(not(feature = "std"))]
        {
            self.inner.into_inner()
        }

        #[cfg(feature = "std")]
        {
            self.inner
                .into_inner()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        }
    }
}

