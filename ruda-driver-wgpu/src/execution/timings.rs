use std::sync::{Arc, atomic::{AtomicU32, Ordering}};

use ruda_core::profile::{Duration, Instant, ProfileDuration, ProfileTicks};
use ruda_kernel::dsl::{
    backtrace::BackTrace,
    server::{ProfileError, ProfilingToken},
};
use hashbrown::HashMap;
use wgpu::{QUERY_SIZE, QuerySet, QuerySetDescriptor, QueryType};

type QuerySetId = u64;

const PROFILE_START_INDEX: u32 = 0;
const PROFILE_END_INDEX: u32 = 1;

/// Shared per-device limit for Metal's timestamp counter buffers.
#[derive(Debug)]
pub struct TimestampQuerySetBudget { live: AtomicU32, max: u32 }

impl TimestampQuerySetBudget {
    pub fn metal() -> Self {
        let max = std::env::var("RUDA_MAX_METAL_TIMING_QUERY_SETS")
            .ok().and_then(|value| value.parse().ok()).unwrap_or(28);
        Self { live: AtomicU32::new(0), max }
    }
    pub fn unbounded() -> Self { Self { live: AtomicU32::new(0), max: u32::MAX } }
    pub(crate) fn try_acquire(&self) -> bool {
        let mut live = self.live.load(Ordering::Relaxed);
        loop {
            if live >= self.max { return false; }
            match self.live.compare_exchange_weak(live, live + 1, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => return true,
                Err(actual) => live = actual,
            }
        }
    }
    fn release(&self, count: u32) { self.live.fetch_sub(count, Ordering::Relaxed); }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimestampSampling { InCommandStream, AtStageBoundaries }
impl TimestampSampling {
    pub fn new(backend: wgpu::Backend) -> Self {
        if backend == wgpu::Backend::Metal { Self::AtStageBoundaries } else { Self::InCommandStream }
    }
    pub fn drains_before_window(self) -> bool { self == Self::AtStageBoundaries }
    pub fn resolves_after_completion(self) -> bool { self == Self::AtStageBoundaries }
}

#[derive(Debug)]
struct QuerySetAllocator {
    budget: Arc<TimestampQuerySetBudget>,
    held: u32,
    created: Vec<QuerySet>,
    reuse_idx: usize,
}

impl QuerySetAllocator {
    fn new(budget: Arc<TimestampQuerySetBudget>) -> Self {
        Self { budget, held: 1, created: Vec::new(), reuse_idx: 0 }
    }
    fn acquire(&mut self, device: &wgpu::Device) -> QuerySet {
        if (self.created.len() as u32) < self.held || self.budget.try_acquire() {
            if self.created.len() as u32 >= self.held { self.held += 1; }
            let query_set = device.create_query_set(&QuerySetDescriptor {
                label: Some("Ruda profile queries"), ty: QueryType::Timestamp, count: 2,
            });
            self.created.push(query_set.clone());
            query_set
        } else {
            let query_set = self.created[self.reuse_idx % self.created.len()].clone();
            self.reuse_idx = self.reuse_idx.wrapping_add(1);
            query_set
        }
    }
}
impl Drop for QuerySetAllocator {
    fn drop(&mut self) { self.budget.release(self.held); }
}

#[derive(Debug)]
pub struct TimestampReadback {
    pub commands: wgpu::CommandBuffer,
    pub map_buffer: wgpu::Buffer,
}

#[derive(Debug)]
/// Struct encapsulating how timings are captured on wgpu.
pub struct QueryProfiler {
    timestamps: HashMap<ProfilingToken, Result<Timestamp, ProfileError>>,
    init_tokens: Vec<ProfilingToken>,
    query_set_pool: Vec<QuerySet>,
    query_sets: HashMap<QuerySetId, QuerySetItem>,
    allocator: QuerySetAllocator,
    current: Option<u64>,
    counter_token: u64,
    counter_query_set: u64,
    cleanups: Vec<QuerySetId>,
    sampling: TimestampSampling,
    queue_period: f64,
    epoch_tick: u64,
    epoch_instant: Instant,
}

#[derive(Debug)]
pub struct Timestamp {
    start: Option<u64>,
    end: Option<u64>,
}

#[derive(Debug)]
struct QuerySetItem {
    query_set: QuerySet,
    // We only track references to the start query set
    num_ref: u32,
}

fn create_resolve_buffer(device: &wgpu::Device, count: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Ruda gpu -> cpu resolve buffer"),
        size: (QUERY_SIZE * count) as _,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn create_map_buffer(device: &wgpu::Device, count: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Ruda gpu -> cpu map buffer"),
        size: (QUERY_SIZE * count) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

// Measure a timestamp to align the CPU & GPU timelines.
#[cfg(feature = "profile-tracy")]
fn get_cur_timestamp(queue: &wgpu::Queue, device: &wgpu::Device) -> u64 {
    // Make sure no work is outstanding.

    use wgpu::BufferAddress;
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None, // Wait for most recent
            timeout: None,
        })
        .unwrap();

    // Resolve a timestamp for the query set.
    let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("Ruda gpu -> cpu sync query_set"),
        ty: wgpu::QueryType::Timestamp,
        count: 1,
    });

    let resolve_buffer = create_resolve_buffer(device, 1);
    let map_buffer = create_map_buffer(device, 1);

    let mut timestamp_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu-profiler gpu -> cpu query timestamp"),
    });
    // This compute pass is purely to get a timestamp.
    timestamp_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("Write timestamp pass"),
        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
            query_set: &query_set,
            beginning_of_pass_write_index: None,
            end_of_pass_write_index: Some(0),
        }),
    });
    timestamp_encoder.write_timestamp(&query_set, 0);
    timestamp_encoder.resolve_query_set(&query_set, 0..1, &resolve_buffer, 0);
    // Workaround for https://github.com/gfx-rs/wgpu/issues/6406
    // TODO when that bug is fixed, merge these encoders together again
    let mut copy_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu-profiler gpu -> cpu copy timestamp"),
    });
    copy_encoder.copy_buffer_to_buffer(
        &resolve_buffer,
        0,
        &map_buffer,
        0,
        Some(QUERY_SIZE as BufferAddress),
    );

    let commands = [timestamp_encoder.finish(), copy_encoder.finish()];

    queue.submit(commands);
    map_buffer.slice(..).map_async(wgpu::MapMode::Read, |_| ());

    device
        .poll(wgpu::PollType::Wait {
            submission_index: None, // Wait for most recent
            timeout: None,
        })
        .unwrap();

    let view = map_buffer.slice(..).get_mapped_range();
    u64::from_le_bytes((*view).try_into().unwrap())
}

impl QueryProfiler {
    pub fn new(queue: &wgpu::Queue, #[allow(unused)] device: &wgpu::Device,
        budget: Arc<TimestampQuerySetBudget>, sampling: TimestampSampling) -> Self {
        #[cfg(feature = "profile-tracy")]
        let sync_timestamps = get_cur_timestamp(queue, device);

        #[cfg(not(feature = "profile-tracy"))]
        let sync_timestamps = 0;

        // Measure CPU timestamp to go along GPU timestamp.
        // This can't be 100% correct as this includes the time to rendezvous the GPU timestamp.
        // Guesstimate by saying the rendesvouz time is twice the submission time.
        let epoch_instant = Instant::now();

        Self {
            cleanups: Vec::new(),
            sampling,
            counter_query_set: 0,
            counter_token: 0,
            query_sets: HashMap::new(),
            query_set_pool: Vec::new(),
            allocator: QuerySetAllocator::new(budget),
            current: None,
            timestamps: HashMap::new(),
            init_tokens: Vec::new(),
            queue_period: queue.get_timestamp_period() as f64,
            epoch_instant,
            epoch_tick: sync_timestamps,
        }
    }

    /// Start a new profiling using [device measurement](TimeMeasurement::Device).
    pub fn start_profile(&mut self) -> ProfilingToken {
        let token = ProfilingToken {
            id: self.counter_token,
        };
        self.counter_token += 1;
        self.init_tokens.push(token);
        self.timestamps.insert(
            token,
            Ok(Timestamp {
                start: None,
                end: None,
            }),
        );
        token
    }

    pub fn error(&mut self, error: ProfileError) {
        self.timestamps.iter_mut().for_each(|(_key, value)| {
            if let Ok(Timestamp { start: Some(start), .. }) = core::mem::replace(value, Err(error.clone())) {
                if let Some(query_set) = self.query_sets.get_mut(&start) {
                    query_set.num_ref -= 1;
                    if query_set.num_ref == 0 { self.cleanups.push(start); }
                }
            }
        });
    }

    pub fn abandon_profile(&mut self, token: ProfilingToken) {
        self.init_tokens.retain(|pending| *pending != token);
        if let Some(Ok(Timestamp { start: Some(start), .. })) = self.timestamps.remove(&token) {
            if let Some(query_set) = self.query_sets.get_mut(&start) {
                query_set.num_ref -= 1;
                if query_set.num_ref == 0 { self.cleanups.push(start); }
            }
        }
    }

    pub fn sampling(&self) -> TimestampSampling { self.sampling }

    /// Stop the profiling on a device.
    pub fn stop_profile_setup(
        &mut self,
        token: ProfilingToken,
        device: &wgpu::Device,
    ) -> Result<Option<TimestampReadback>, ProfileError> {
        self.init_tokens.retain(|pending| *pending != token);
        let timestamps =
            self.timestamps
                .remove(&token)
                .ok_or_else(|| ProfileError::NotRegistered {
                    backtrace: BackTrace::capture(),
                })?;
        let mut timestamps = timestamps?;
        let Timestamp { start, end } = &mut timestamps;

        *end = self.current;

        // TODO: We could optimize this by having a single handle for both `start` and `end`
        // when a single query_set is used, but it probably doesn't impact the real
        // performance all that much.
        let (Some(start), Some(end)) = (start, end) else {
            return Ok(None);
        };

        let query_set_error = || ProfileError::Unknown {
            reason: "Can't resolve the query sets".to_string(),
            backtrace: BackTrace::capture(),
        };

        let query_set_start = self.query_sets.get_mut(start).ok_or_else(query_set_error)?;

        query_set_start.num_ref -= 1;
        if query_set_start.num_ref == 0 {
            self.cleanups.push(*start);
        }

        // TODO: Could use a StagingBelt for a small speedup here.
        let resolve_start = create_resolve_buffer(device, 1);
        let resolve_end = create_resolve_buffer(device, 1);
        let map_buffer = create_map_buffer(device, 2);
        let query_set_start = self.query_sets.get(start).ok_or_else(query_set_error)?;

        let query_set_end = self.query_sets.get(end).ok_or_else(query_set_error)?;

        let size = QUERY_SIZE as u64;
        let mut readback = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Ruda profile readback"),
        });
        readback.resolve_query_set(&query_set_start.query_set, PROFILE_START_INDEX..PROFILE_START_INDEX + 1, &resolve_start, 0);
        readback.resolve_query_set(&query_set_end.query_set, PROFILE_END_INDEX..PROFILE_END_INDEX + 1, &resolve_end, 0);
        readback.copy_buffer_to_buffer(&resolve_start, 0, &map_buffer, 0, size);
        readback.copy_buffer_to_buffer(&resolve_end, 0, &map_buffer, size, size);
        Ok(Some(TimestampReadback { commands: readback.finish(), map_buffer }))
    }

    pub fn stop_profile(
        &self,
        map_buffer: Option<wgpu::Buffer>,
        poll_signal: Arc<()>,
    ) -> Result<ProfileDuration, ProfileError> {
        let Some(map_buffer) = map_buffer else {
            return Err(ProfileError::NotMeasured { backtrace: BackTrace::capture() });
        };
        let period = self.queue_period;
        let epoch_tick = self.epoch_tick;
        let epoch_instant = self.epoch_instant;
        let (sender, rec) = async_channel::bounded(1);
        map_buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            core::mem::drop(poll_signal);
            let _ = sender.try_send(result);
        });

        Ok(ProfileDuration::new_device_time_maybe(async move {
            rec.recv().await.expect("Unable to receive buffer slice result.")
                .expect("Failed to map buffer");
            let binding = map_buffer.slice(..).get_mapped_range();
            let data: &[u64] = bytemuck::try_cast_slice(&binding).unwrap();
            let (raw_start, raw_end) = (data[0], data[1]);
            drop(binding);
            map_buffer.unmap();
            if raw_start == 0 || raw_end == 0 || raw_end <= raw_start {
                return None;
            }
            let data_start = raw_start.saturating_sub(epoch_tick);
            let data_end = raw_end.saturating_sub(epoch_tick);
            let start_duration = Duration::from_nanos((data_start as f64 * period) as u64);
            let end_duration = Duration::from_nanos((data_end as f64 * period) as u64);
            Some(ProfileTicks::from_start_end(epoch_instant + start_duration, epoch_instant + end_duration))
        }))
    }

    /// Write a window's start once and update its end on every subsequent pass.
    pub fn register_profile_device(&mut self, device: &wgpu::Device)
        -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let (query_set_id, beginning_of_pass_write_index) = match self.init_query_set() {
            Some(info) => {
                self.new_query_set(info, device);
                (info.0, Some(PROFILE_START_INDEX))
            }
            None if self.timestamps.is_empty() => return None,
            None => (self.current?, None),
        };
        let item = self.query_sets.get(&query_set_id)?;
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &item.query_set,
            beginning_of_pass_write_index,
            end_of_pass_write_index: Some(PROFILE_END_INDEX),
        })
    }

    fn new_query_set(&mut self, query_set_info: (u64, u32), device: &wgpu::Device) {
        let (query_set_id, num_ref) = query_set_info;
        let query_set = match self.query_set_pool.pop() {
            Some(query_set) => query_set,
            None => self.allocator.acquire(device),
        };
        self.query_sets.insert(query_set_id, QuerySetItem { query_set, num_ref });
    }

    fn init_query_set(&mut self) -> Option<(QuerySetId, u32)> {
        let mut query_set_id = None;
        let mut count = 0;

        for token in self.init_tokens.drain(..) {
            if let Some(Ok(Timestamp { start, .. })) = &mut self.timestamps.get_mut(&token) {
                count += 1;
                let id = match query_set_id {
                    Some(id) => id,
                    None => {
                        let id = self.counter_query_set;
                        self.counter_query_set += 1;
                        self.current = Some(id);
                        query_set_id = Some(id);
                        id
                    }
                };

                *start = Some(id);
            }
        }

        // We only cleanup old query sets when creating a new one, since we don't know if we need
        // the end timing of the last query set.
        self.cleanup_query_sets();

        query_set_id.map(|v| (v, count))
    }

    fn cleanup_query_sets(&mut self) {
        let mut cleanups = core::mem::take(&mut self.cleanups);
        cleanups.retain(|key| {
            if Some(*key) == self.current { return true; }
            let removed = self.query_sets.remove(key).expect("Unknown query set cleaned up");
            self.query_set_pool.push(removed.query_set);
            false
        });
        self.cleanups = cleanups;
    }
}
