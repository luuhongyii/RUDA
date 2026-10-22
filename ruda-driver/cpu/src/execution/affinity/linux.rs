use std::mem;

use super::{CoreId, ThreadAffinity};

pub(super) struct Platform;

impl ThreadAffinity for Platform {
    fn active_cpus() -> Vec<CoreId> {
        // SAFETY: The affinity buffer is initialized and has the advertised size.
        let mut mask: libc::cpu_set_t = unsafe { mem::zeroed() };
        let result = unsafe {
            libc::sched_getaffinity(0, mem::size_of::<libc::cpu_set_t>(), &mut mask)
        };
        if result != 0 {
            return Vec::new();
        }
        (0..libc::CPU_SETSIZE as usize)
            .filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &mask) })
            .map(CoreId)
            .collect()
    }

    fn physical_core(cpu: CoreId) -> Option<CoreId> {
        let siblings = std::fs::read_to_string(format!(
            "/sys/devices/system/cpu/cpu{}/topology/thread_siblings_list", cpu.0,
        )).ok()?;
        siblings.split(['-', ',']).next()?.trim().parse().ok().map(CoreId)
    }

    fn pin_current(cpu: CoreId) {
        if cpu.0 >= libc::CPU_SETSIZE as usize {
            return;
        }
        // SAFETY: The checked CPU index fits the initialized set. PID 0 affects
        // only the calling worker, not the process or any other worker.
        unsafe {
            let mut mask: libc::cpu_set_t = mem::zeroed();
            libc::CPU_SET(cpu.0, &mut mask);
            libc::sched_setaffinity(0, mem::size_of::<libc::cpu_set_t>(), &mask);
        }
    }
}
