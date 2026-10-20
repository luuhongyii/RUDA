pub use ruda_core::profile::{ProfileDuration, ProfileTicks, TimingMethod};
use ruda_core::{backtrace::BackTrace, profile::Instant};
use crate::{Backend, ExecutionError};

/// Whether a profiled window dispatches remaining queued operations before closing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ProfileOptions { flush: bool }

impl ProfileOptions {
    /// Include pending operations rather than retaining normal lazy batching.
    pub fn flush(mut self) -> Self { self.flush = true; self }
    /// Whether closing the window dispatches its pending operations.
    pub fn flushes(&self) -> bool { self.flush }
}

/// Backend-local profiling window, including the queue on which it was opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ProfileToken {
    /// Backend-local window identifier.
    pub id: u64,
    /// Backend-local stream or queue identifier, retained across caller threads.
    #[serde(default)]
    pub opened_on: u64,
}

/// Measure a closure in system time between device synchronizations.
pub fn profile_system_time<B: Backend, O: Send + 'static>(device: &B::Device,
    func: impl FnOnce() -> O + Send) -> Result<(O, ProfileDuration), ExecutionError> {
    B::sync(device)?;
    let start = Instant::now();
    let output = func();
    B::sync(device)?;
    Ok((output, ProfileDuration::new_system_time(start, Instant::now())))
}

/// Measure using split windows, with system timing only when the backend opens none.
pub fn profile_with_tokens<B: Backend, O: Send + 'static>(device: &B::Device,
    options: ProfileOptions, func: impl FnOnce() -> O + Send) -> Result<(O, ProfileDuration), ExecutionError> {
    let token = match B::profile_start(device) {
        Ok(Some(token)) => token,
        Ok(None) => return profile_system_time::<B, O>(device, func),
        Err(error) => { func(); return Err(error); }
    };
    let mut window = OpenWindow::<B> { device, token: Some(token) };
    let output = func();
    let token = window.token.take().expect("open profiling window");
    let duration = B::profile_end(device, token, options)?;
    Ok((output, duration))
}

struct OpenWindow<'a, B: Backend> { device: &'a B::Device, token: Option<ProfileToken> }
impl<B: Backend> Drop for OpenWindow<'_, B> {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() { B::profile_abandon(self.device, token); }
    }
}

pub(crate) fn unsupported() -> ExecutionError {
    ExecutionError::Generic { reason: "profiling windows are not supported by this backend; use profile".into(),
        backtrace: BackTrace::capture() }
}
