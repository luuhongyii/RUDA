use ruda_tensor::{Backend, ExecutionError, ProfileDuration, ProfileOptions, ProfileToken, StreamId};
use crate::{FusionBackend, FusionRuntime, FusionServer};

pub(super) struct Window {
    stream: StreamId,
    token: Option<ProfileToken>,
    finish: Box<dyn Fn(ProfileToken, ProfileOptions) -> Result<ProfileDuration, ExecutionError> + Send>,
    abandon: Box<dyn Fn(ProfileToken) + Send>,
}

impl Window {
    fn new<B: Backend>(stream: StreamId, device: B::Device, token: ProfileToken) -> Self {
        let finish_device = device.clone();
        Self { stream, token: Some(token),
            finish: Box::new(move |token, options| B::profile_end(&finish_device, token, options)),
            abandon: Box::new(move |token| B::profile_abandon(&device, token)) }
    }
    fn close(mut self, options: ProfileOptions) -> Result<ProfileDuration, ExecutionError> {
        let token = self.token.take().expect("open inner profiling window");
        self.stream.executes(|| (self.finish)(token, options))
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            self.stream.executes(|| (self.abandon)(token));
        }
    }
}

impl<R: FusionRuntime> FusionServer<R> {
    pub(crate) fn profile_start<B: FusionBackend<FusionRuntime = R>>(&mut self, stream: StreamId,
        device: B::Device) -> Result<Option<ProfileToken>, ExecutionError> {
        let next = self.profile_counter.checked_add(1).ok_or_else(|| ExecutionError::WithContext {
            reason: "Fusion profiling token space exhausted".into(),
        })?;
        let Some(inner) = stream.executes(|| B::profile_start(&device))? else { return Ok(None); };
        let token = ProfileToken { id: self.profile_counter, opened_on: stream.value };
        self.profile_counter = next;
        self.profile_windows.insert(token.id, Window::new::<B>(stream, device, inner));
        Ok(Some(token))
    }

    pub(crate) fn profile_end(&mut self, token: ProfileToken, options: ProfileOptions)
        -> Result<ProfileDuration, ExecutionError> {
        if !self.profile_windows.get(&token.id).is_some_and(|window| window.stream.value == token.opened_on) {
            return Err(ExecutionError::WithContext { reason: "Unknown Fusion profiling window".into() });
        }
        let window = self.profile_windows.remove(&token.id).expect("checked open profiling window");
        if options.flushes() { self.drain_stream(window.stream); }
        window.close(options)
    }

    pub(crate) fn profile_abandon(&mut self, token: ProfileToken) {
        if self.profile_windows.get(&token.id).is_some_and(|window| window.stream.value == token.opened_on) {
            self.profile_windows.remove(&token.id);
        }
    }
}
