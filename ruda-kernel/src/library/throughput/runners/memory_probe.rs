use crate::dsl::prelude::*;
use ruda_core::ir::ElemType;
use ruda::runtime::{
    server::Handle,
    throughput::{DEFAULT_WORKING_SET_BYTES, MemorySpec, ThroughputError},
};

use crate::library::throughput::LaunchConfig;
use ruda_core::profile::Duration;

/// Windows the pool holds. What decides whether a rewritten line is still
/// resident is the pool's size against the last level cache, not the positions
/// the window walks, so two is a floor a probe that cannot read that cache size
/// can justify rather than a ratio known to be enough.
const POOL_WINDOWS: usize = 2;

/// The largest window one buffer is probed at: the default working set, or as
/// much of the device's allocation as leaves room for the pool around it.
pub(crate) fn window_cap(max_alloc: u64) -> u64 {
    (DEFAULT_WORKING_SET_BYTES.min(max_alloc / POOL_WINDOWS as u64)).max(1)
}

/// The buffer geometry and launch shape a memory probe uses to move a working
/// set of a given size in one pass.
///
/// Shared by the copy and read probes: both walk one line per thread per step
/// over buffers of identical size, and both measure a small working set by
/// moving a small *window* over a large buffer rather than by allocating a
/// small buffer.
///
/// The window is what makes a small working set mean anything. A small buffer
/// read over and over stays in cache after the first pass, so the probe would
/// report cache bandwidth for every size below the cache and the curve would
/// describe residency instead of size. Reading a fresh window each pass keeps
/// the data cold — by the time the window comes back around, a whole buffer of
/// traffic has evicted it — so what varies across the sweep is how much a pass
/// moves, which is the thing being measured.
///
/// # The pool is sized against the cache, not against the window
///
/// A store into a line a recent pass left in the last level cache never reaches
/// memory, so a pool the cache holds a large share of reports write traffic the
/// bus never carried: 747 GB/s across a 672 GB/s bus on an RTX 4070 Ti SUPER,
/// at every working set from 32 MiB up. More positions did not fix that — the
/// 32 MiB window already walked sixteen of them — and doubling the pool did.
#[derive(Clone, Copy, Debug)]
pub struct MemoryProbe {
    /// Lines of each buffer the probe walks: what the window is cold against.
    pub pool_lines: usize,
    /// Lines one pass moves through, per buffer. Always a fraction of
    /// [`pool_lines`](Self::pool_lines), the top of the sweep included.
    pub window_lines: usize,
    /// Bytes in each buffer, which is exactly the pool. The probe kernels take
    /// their wrap-around from `len()`, so a buffer longer than the pool would
    /// walk them past what [`prime`] wrote.
    pub buffer_bytes: usize,
    /// Cubes to dispatch, which is [`LaunchConfig::ruda_count`] unless the
    /// window is too small to give every thread a line.
    pub ruda_count: usize,
    /// Whether the probe kernels address lines as per-thread contiguous runs
    /// rather than coalesced across threads. Set on plane-1 runtimes, where a
    /// worker has no plane neighbour to coalesce with.
    pub blocked: bool,
}

/// All [`MemoryProbe::sized`] needs of a device, so the sizing can be exercised
/// without one.
#[derive(Clone, Copy)]
struct DeviceShape {
    /// Bytes the device will hand out in a single allocation.
    max_alloc: usize,
    /// Threads per cube, and cubes, of the probe's launch.
    ruda_dim: usize,
    ruda_count: usize,
}

impl MemoryProbe {
    /// Passes a launch carries for the window to come back to bytes a whole
    /// pool of traffic has since evicted, which is what keeps a window smaller
    /// than the cache cold.
    pub fn min_iterations(&self) -> usize {
        self.pool_lines.div_ceil(self.window_lines)
    }

    /// Sizes a probe moving `working_set` bytes per pass, split evenly across
    /// the buffers `access` touches.
    ///
    /// A blocked probe pins the launch to one cube: on the backend that
    /// addressing is for, a cube position is a loop wrapped around the whole
    /// kernel body, `n_iter` included, so more than one turns the window
    /// rotation between passes into a replay of the same narrow,
    /// blocked-addressed slice instead of a walk across the buffer, and cache
    /// serves it. A coalesced launch spreads one cube position's addresses
    /// across the full window regardless of cube count, so it has no such
    /// limit.
    pub fn new<R: Runtime>(client: &ComputeClient<R>, config: LaunchConfig, line_bytes: usize, spec: MemorySpec) -> Self {
        let blocked = config.plane_size == 1;
        let shape = DeviceShape {
            max_alloc: client.properties().memory.max_page_size as usize,
            ruda_dim: config.ruda_dim.num_elems() as usize,
            ruda_count: if blocked { 1 } else { config.ruda_count },
        };

        Self::sized(shape, line_bytes, spec, blocked)
    }

    /// The geometry itself, with the device reduced to its allocation limit and
    /// launch shape so the sizing can be exercised without one.
    ///
    /// The launch shrinks with the window instead of the window growing to fill
    /// the launch. A small window measured with the full dispatch would either
    /// hand many threads the same line or be padded back up to a large one, and
    /// neither is the small-kernel behaviour the curve exists to describe: a
    /// kernel that moves little has little in flight, and that is precisely
    /// what limits it.
    fn sized(shape: DeviceShape, line_bytes: usize, spec: MemorySpec, blocked: bool) -> Self {
        let buffers = spec.access.buffers() as usize;
        let window_bytes = (spec.bytes.min(usize::MAX as u64) as usize) / buffers;

        let cap_lines = (window_cap(shape.max_alloc as u64) as usize / line_bytes).max(1);
        let window_lines = (window_bytes / line_bytes).clamp(1, cap_lines);

        let pool_lines = cap_lines * POOL_WINDOWS;

        let ruda_count = (window_lines / shape.ruda_dim).clamp(1, shape.ruda_count);

        Self {
            pool_lines,
            window_lines,
            buffer_bytes: pool_lines * line_bytes,
            ruda_count,
            blocked,
        }
    }
}

/// Reserves a probe's buffers outside every pool: each is its own device
/// allocation, returned to the driver once the probe drops it.
///
/// A probe's gigabyte is no part of any workload. Held in a pool it would stay
/// reserved after the probe, and grow the pages an adaptive memory carves
/// every later allocation from. Dedicated, it exists exactly as long as the
/// probe does.
///
/// # Errors
///
/// [`ThroughputError::Allocation`] when the device has no room for them. A
/// reservation is only enqueued, and one that fails still leaves a handle, over
/// which a launch does nothing: the probe would time an empty pass and cache it
/// as the device's peak.
pub fn reserve<R: Runtime, const N: usize>(
    client: &ComputeClient<R>,
    bytes: [usize; N],
) -> Result<[Handle; N], ThroughputError> {
    client.memory_dedicated_allocation((), |_| {
        let handles = bytes.into_iter()
            .map(|bytes| client.try_empty(bytes).map_err(|_| ThroughputError::Allocation))
            .collect::<Result<alloc::vec::Vec<_>, _>>()?;
        handles.try_into().map_err(|_| ThroughputError::Allocation)
    })
}

/// Runs one pass and confirms it wrote `written`, before any pass is timed.
///
/// Samples propagate completion failures; buffer claims also catch failures
/// whose originating stream error has already been observed.
///
/// # Errors
///
/// [`ThroughputError::Launch`] when the pass did not run. The cause is logged
/// by the device where it happened.
pub fn verify<R: Runtime>(
    client: &ComputeClient<R>,
    sample: impl Fn(usize) -> Result<Duration, ThroughputError>,
    written: &Handle,
) -> Result<(), ThroughputError> {
    sample(1)?;
    synchronize(client, &[written])
}

pub(super) fn synchronize<R: Runtime>(
    client: &ComputeClient<R>,
    written: &[&Handle],
) -> Result<(), ThroughputError> {
    ruda_core::future::block_on(client.sync()).map_err(|_| ThroughputError::Launch)?;
    for handle in written {
        if (*handle).clone().binding().failure().is_some() {
            return Err(ThroughputError::Launch);
        }
    }
    Ok(())
}

/// Writes every line of `handle`, once, before it is handed to a probe that
/// only reads it.
///
/// A fresh allocation is backed by the same physical zero page until its
/// first write, so every unwritten line a read-only probe visits is served
/// from that one cached page rather than from DRAM, inflating its reported
/// bandwidth well past the device's real ceiling. Writing real data in first
/// gives each line its own page, the way a buffer a real kernel reads
/// already got one from whoever produced it.
pub fn prime<R: Runtime>(
    client: &ComputeClient<R>,
    handle: &Handle,
    pool_lines: usize,
    config: LaunchConfig,
    dtype: ElemType,
) -> Result<(), ThroughputError> {
    unsafe {
        prime_buffer::launch_unchecked(
            client,
            RudaCount::Static(config.ruda_count as u32, 1, 1),
            config.ruda_dim,
            config.vector_size,
            ArrayArg::from_raw_parts(handle.clone(), pool_lines * config.vector_size),
            pool_lines,
            dtype.into(),
        );
    }
    synchronize(client, &[handle])
}

#[ruda(launch_unchecked)]
fn prime_buffer<I: Numeric, N: Size>(
    output: &mut Array<Vector<I, N>>,
    len: usize,
    #[define(I)] _dtype: StorageType,
) {
    let stride = RUDA_DIM as usize * RUDA_COUNT;
    let steps = len.div_ceil(stride).max(1);

    for step in 0..steps {
        let idx = ABSOLUTE_POS + step * stride;
        if idx < len {
            output[idx] = Vector::<I, N>::empty();
        }
    }
}
