//! Running a workload for the compilation and tuning it provokes, without
//! running the workload itself.
//!
//! Under a [`DryRun`] every launch is still expanded, compiled, validated and
//! cached, and is then dropped instead of reaching the device. A warm-up pass
//! then pays for compilation and tuning without also paying for the work that
//! provoked them, which is what makes producing a shippable environment
//! affordable.
//!
//! The launches autotune issues are the exception: they *are* the measurement,
//! so [`RealRun`] opts them back into executing.
//!
//! **Buffers are left as they were**, so anything read back during a dry run is
//! meaningless. It only suits a pass driven by the *shapes* it produces, which
//! is what keys the caches, and never one that branches on a computed value.
//!
//! The decision is made here, once, on the thread that issues the launch.
//! Servers receive the verdict as a [`LaunchMode`] argument rather than
//! deriving it: by the time a launch reaches a server thread, the context that
//! produced it is gone.

use ruda_core::stub::{AtomicUsize, Ordering};

/// What a server should do with a launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaunchMode {
    /// Compile if needed, then run it. The normal case.
    Execute,
    /// Compile if needed, cache the artifact, and drop the launch.
    ///
    /// A server honoring this must still do everything a first launch does
    /// short of dispatching — expand, compile, validate, populate its caches —
    /// or the pass buys nothing.
    Skip,
}

impl LaunchMode {
    /// Whether the launch should be dropped rather than run.
    pub fn is_skipped(self) -> bool {
        matches!(self, LaunchMode::Skip)
    }
}

/// What to do with a launch issued on this thread, right now.
pub fn launch_mode() -> LaunchMode {
    if !dry_run() || real_run::depth() > 0 {
        return LaunchMode::Execute;
    }

    LaunchMode::Skip
}

/// How many dry runs are open in this process.
///
/// A depth rather than a flag so overlapping guards compose: a swap-and-restore
/// would let one thread's guard end a dry run another thread is still inside,
/// and leave the process dry-running forever once that one dropped in turn.
static DRY_RUN: AtomicUsize = AtomicUsize::new(0);

/// Whether launches are currently compiled and dropped rather than run.
pub fn dry_run() -> bool {
    DRY_RUN.load(Ordering::Relaxed) > 0
}

/// Makes every launch a dry run for as long as it lives, on every thread and
/// every device.
///
/// Overlapping guards compose, so a pass that opens one while another is still
/// open leaves the mode on until the last of them drops.
///
/// The flag is read on the thread issuing a launch, with relaxed ordering, so a
/// launch another thread had already begun issuing may still execute. What is
/// guaranteed is the launches issued by the thread that opened the guard, and
/// every launch issued after other threads observe it.
///
/// This is the only way in: there is deliberately no configuration file or
/// environment variable for it. A dry run left on by accident turns the rest of
/// the process into launches that quietly do nothing and read back
/// uninitialized memory, so its lifetime belongs to a scope in the code that
/// wants it, not to an ambient default nothing in the process can see.
#[derive(Debug)]
pub struct DryRun {
    _private: (),
}

impl DryRun {
    /// Opens a dry run until the guard drops.
    #[allow(clippy::new_without_default, reason = "a guard is not a value")]
    pub fn new() -> Self {
        DRY_RUN.fetch_add(1, Ordering::Relaxed);
        Self { _private: () }
    }
}

impl Drop for DryRun {
    fn drop(&mut self) {
        DRY_RUN.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Makes the launches issued on this thread execute for real even inside a
/// [`DryRun`], for as long as it lives.
///
/// Autotune holds one: its launches are the measurement a dry run exists to
/// provoke, not the workload it exists to skip. Held across warm-up and samples
/// alike, since a candidate that was never warmed is a candidate measured on
/// its first, slowest run.
///
/// Thread-local, and the thread that matters is the one issuing the launches,
/// which is not always the one that asked for them: a task handed to
/// [`Client::exclusive`](crate::runtime::client::ComputeClient::exclusive) runs on
/// the device thread. The guard has to live inside that task, alongside the
/// launches it covers, not around the call that submits it.
#[derive(Debug)]
pub struct RealRun {
    _private: (),
}

impl RealRun {
    /// Opts this thread back into executing until the guard drops.
    #[allow(clippy::new_without_default, reason = "a guard is not a value")]
    pub fn new() -> Self {
        real_run::enter();
        Self { _private: () }
    }
}

impl Drop for RealRun {
    fn drop(&mut self) {
        real_run::exit();
    }
}

#[cfg(feature = "runtime-std")]
mod real_run {
    use core::cell::Cell;

    std::thread_local! {
        /// How many [`RealRun`](super::RealRun) guards are open on this thread.
        /// A depth rather than a flag: a tunable may itself dispatch through
        /// another tuner, and the inner one finishing must not un-mark the
        /// outer.
        static DEPTH: Cell<usize> = const { Cell::new(0) };
    }

    pub(super) fn depth() -> usize {
        DEPTH.with(|depth| depth.get())
    }

    pub(super) fn enter() {
        DEPTH.with(|depth| depth.set(depth.get() + 1));
    }

    pub(super) fn exit() {
        DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

#[cfg(not(feature = "runtime-std"))]
mod real_run {
    // No threads to be local to; this keeps the call sites uniform.
    pub(super) fn depth() -> usize {
        0
    }
    pub(super) fn enter() {}
    pub(super) fn exit() {}
}
