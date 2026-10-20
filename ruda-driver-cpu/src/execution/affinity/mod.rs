#[cfg(any(target_os = "linux", target_os = "android"))]
mod linux;
#[cfg(any(target_os = "linux", target_os = "android"))]
use linux::Platform;

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
use macos::Platform;

#[cfg(windows)]
mod windows;
#[cfg(windows)]
use windows::Platform;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CoreId(usize);

trait ThreadAffinity {
    fn active_cpus() -> Vec<CoreId>;
    fn physical_core(cpu: CoreId) -> Option<CoreId>;
    fn pin_current(cpu: CoreId);
}

pub(crate) fn active_cores() -> Vec<CoreId> {
    let mut primaries = Vec::new();
    let mut siblings = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for cpu in Platform::active_cpus() {
        if seen.insert(Platform::physical_core(cpu).unwrap_or(cpu)) {
            primaries.push(cpu);
        } else {
            siblings.push(cpu);
        }
    }
    primaries.extend(siblings);
    primaries
}

pub(crate) fn worker_count(cores: &[CoreId]) -> usize {
    if cores.is_empty() {
        std::thread::available_parallelism().map_or(1, |count| count.get())
    } else {
        cores.len()
    }
}

pub(crate) fn set_for_current(core: CoreId) {
    Platform::pin_current(core);
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    windows,
)))]
struct Platform;

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    windows,
)))]
impl ThreadAffinity for Platform {
    fn active_cpus() -> Vec<CoreId> { Vec::new() }
    fn physical_core(_cpu: CoreId) -> Option<CoreId> { None }
    fn pin_current(_cpu: CoreId) {}
}
