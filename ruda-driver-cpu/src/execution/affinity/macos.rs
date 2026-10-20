use std::{ffi::{CStr, c_void}, mem, ptr};

use super::{CoreId, ThreadAffinity};

pub(super) struct Platform;

impl ThreadAffinity for Platform {
    fn active_cpus() -> Vec<CoreId> {
        let count = sysctl(c"hw.logicalcpu")
            .and_then(|count| usize::try_from(count).ok())
            .filter(|count| *count > 0)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
        (0..count).map(CoreId).collect()
    }

    fn physical_core(cpu: CoreId) -> Option<CoreId> {
        let logical = usize::try_from(sysctl(c"hw.logicalcpu")?).ok()?;
        let physical = usize::try_from(sysctl(c"hw.physicalcpu")?).ok()?;
        let per_core = (logical / physical.max(1)).max(1);
        Some(CoreId(cpu.0 - cpu.0 % per_core))
    }

    fn pin_current(cpu: CoreId) {
        // Darwin provides affinity hints, not hard CPU pinning. Apple Silicon
        // may reject this policy; the worker still runs under the OS scheduler.
        let mut policy = libc::thread_affinity_policy_data_t {
            affinity_tag: cpu.0 as libc::integer_t + 1,
        };
        // SAFETY: The policy storage and count match THREAD_AFFINITY_POLICY,
        // and the Mach thread is the calling worker's pthread.
        unsafe {
            libc::thread_policy_set(
                libc::pthread_mach_thread_np(libc::pthread_self()),
                libc::THREAD_AFFINITY_POLICY as libc::thread_policy_flavor_t,
                &mut policy as *mut _ as libc::thread_policy_t,
                libc::THREAD_AFFINITY_POLICY_COUNT,
            );
        }
    }
}

fn sysctl(name: &CStr) -> Option<u64> {
    let mut value = 0u64;
    let mut size = mem::size_of::<u64>();
    // SAFETY: name is NUL-terminated; value has the advertised writable size.
    let result = unsafe {
        libc::sysctlbyname(
            name.as_ptr(), (&mut value as *mut u64).cast::<c_void>(),
            &mut size, ptr::null_mut(), 0,
        )
    };
    (result == 0 && (size == 4 || size == 8)).then_some(value)
}
