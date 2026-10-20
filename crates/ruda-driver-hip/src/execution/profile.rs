use std::{collections::HashMap, ops::Deref, sync::{Arc, Mutex}};
use ruda_hip_sys::{self as hip, hipEvent_t, hipStream_t};
use ruda_core::{backtrace::BackTrace, profile::{Duration, Instant, ProfileDuration, ProfileTicks}};
use ruda::runtime::server::{ProfileError, ProfilingToken, ServerError};

const ANCHOR_MAX_AGE: Duration = Duration::from_secs(1);
const UNREADABLE: Duration = Duration::from_secs(3600);

#[derive(Debug)]
struct DriverError { operation: &'static str, status: u32 }
impl core::fmt::Display for DriverError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} failed with HIP status {}", self.operation, self.status)
    }
}
fn checked(operation: &'static str, status: hip::hipError_t) -> Result<(), DriverError> {
    if status == hip::HIP_SUCCESS { Ok(()) } else { Err(DriverError { operation, status: status as u32 }) }
}

#[derive(Debug)]
struct Event { raw: usize }

impl Event {
    fn new() -> Result<Self, DriverError> {
        let mut raw = core::ptr::null_mut();
        checked("hipEventCreateWithFlags", unsafe { hip::hipEventCreateWithFlags(&mut raw, hip::hipEventDefault) })?;
        Ok(Self { raw: raw as usize })
    }
    fn record(&self, stream: hipStream_t) -> Result<(), DriverError> {
        checked("hipEventRecord", unsafe { hip::hipEventRecord(self.raw as hipEvent_t, stream) })
    }
    fn wait(&self) -> Result<(), DriverError> {
        checked("hipEventSynchronize", unsafe { hip::hipEventSynchronize(self.raw as hipEvent_t) })
    }
    fn elapsed(&self, end: &Self) -> Result<Duration, DriverError> {
        let mut milliseconds = 0.0f32;
        checked("hipEventElapsedTime", unsafe {
            hip::hipEventElapsedTime(&mut milliseconds, self.raw as hipEvent_t, end.raw as hipEvent_t)
        })?;
        Ok(Duration::from_secs_f64((milliseconds as f64 / 1000.0).max(0.0)))
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if let Err(error) = checked("hipEventDestroy", unsafe { hip::hipEventDestroy(self.raw as hipEvent_t) }) {
            log::warn!("HIP profiling event release failed: {error}");
        }
    }
}

#[derive(Clone, Debug, Default)]
struct EventPool { free: Arc<Mutex<Vec<Event>>> }
impl EventPool {
    fn acquire(&self) -> Result<Pooled, DriverError> {
        let event = match self.free.lock().unwrap().pop() { Some(event) => event, None => Event::new()? };
        Ok(Pooled { event: Some(event), pool: self.clone() })
    }
}

#[derive(Debug)]
struct Pooled { event: Option<Event>, pool: EventPool }
impl Deref for Pooled {
    type Target = Event;
    fn deref(&self) -> &Event { self.event.as_ref().expect("event is returned only on drop") }
}
impl Drop for Pooled {
    fn drop(&mut self) {
        if let Some(event) = self.event.take() { self.pool.free.lock().unwrap().push(event); }
    }
}

#[derive(Clone, Debug)]
struct Anchor { event: Arc<Pooled>, instant: Instant }
impl Anchor {
    fn take(stream: hipStream_t, pool: &EventPool) -> Result<Self, DriverError> {
        let event = pool.acquire()?;
        event.record(stream)?;
        event.wait()?;
        Ok(Self { event: Arc::new(event), instant: Instant::now() })
    }
}

#[derive(Debug)]
struct AnchorStream(usize);
impl Drop for AnchorStream {
    fn drop(&mut self) {
        if let Err(error) = checked("hipStreamDestroy", unsafe { hip::hipStreamDestroy(self.0 as hipStream_t) }) {
            log::warn!("HIP profiling anchor stream release failed: {error}");
        }
    }
}

#[derive(Debug)]
struct Anchoring { stream: AnchorStream, current: Anchor }
impl Anchoring {
    fn new(pool: &EventPool) -> Result<Self, DriverError> {
        let mut raw = core::ptr::null_mut();
        checked("hipStreamCreateWithFlags", unsafe { hip::hipStreamCreateWithFlags(&mut raw, hip::hipStreamNonBlocking) })?;
        let stream = AnchorStream(raw as usize);
        let current = Anchor::take(stream.0 as hipStream_t, pool)?;
        Ok(Self { stream, current })
    }
}

#[derive(Debug)]
struct Open { start: Pooled, anchor: Anchor }

#[derive(Default, Debug)]
pub(crate) struct EventProfiler {
    open: HashMap<ProfilingToken, Result<Open, ProfileError>>,
    counter: u64,
    pool: EventPool,
    anchoring: Option<Anchoring>,
}

impl EventProfiler {
    pub(crate) fn is_empty(&self) -> bool { self.open.is_empty() }

    pub(crate) fn start(&mut self, stream: hipStream_t) -> Result<ProfilingToken, ServerError> {
        let anchor = self.anchor().map_err(server_error)?;
        let start = self.pool.acquire().map_err(server_error)?;
        start.record(stream).map_err(server_error)?;
        let token = ProfilingToken { id: self.counter };
        self.counter += 1;
        self.open.insert(token, Ok(Open { start, anchor }));
        Ok(token)
    }

    pub(crate) fn stop(&mut self, stream: hipStream_t, token: ProfilingToken) -> Result<ProfileDuration, ProfileError> {
        let Open { start, anchor } = self.open.remove(&token)
            .ok_or_else(|| ProfileError::NotRegistered { backtrace: BackTrace::capture() })??;
        let end = self.pool.acquire().map_err(profile_error)?;
        end.record(stream).map_err(profile_error)?;
        Ok(ProfileDuration::new_device_time(async move {
            read(&start, &end, &anchor).unwrap_or_else(|error| {
                log::error!("HIP profiling readback failed ({error}); reporting unreadable-window sentinel {UNREADABLE:?}");
                let now = Instant::now();
                ProfileTicks::from_start_end(now, now + UNREADABLE)
            })
        }))
    }

    pub(crate) fn abandon(&mut self, token: ProfilingToken) { self.open.remove(&token); }

    pub(crate) fn error(&mut self, error: ProfileError) {
        for state in self.open.values_mut() { *state = Err(error.clone()); }
    }

    fn anchor(&mut self) -> Result<Anchor, DriverError> {
        if self.anchoring.is_none() { self.anchoring = Some(Anchoring::new(&self.pool)?); }
        let anchoring = self.anchoring.as_mut().expect("initialized anchor");
        if anchoring.current.instant.elapsed() > ANCHOR_MAX_AGE {
            anchoring.current = Anchor::take(anchoring.stream.0 as hipStream_t, &self.pool)?;
        }
        Ok(anchoring.current.clone())
    }
}

fn read(start: &Event, end: &Event, anchor: &Anchor) -> Result<ProfileTicks, DriverError> {
    start.wait()?;
    end.wait()?;
    let offset = anchor.event.elapsed(start)?;
    let span = start.elapsed(end)?;
    let instant = anchor.instant + offset;
    Ok(ProfileTicks::from_start_end(instant, instant + span))
}

fn server_error(error: DriverError) -> ServerError {
    super::status::server(error.operation, error.status as hip::hipError_t)
}
fn profile_error(error: DriverError) -> ProfileError { ProfileError::Server(Box::new(server_error(error))) }
