use std::{mem, ptr};

use winapi::shared::basetsd::{DWORD_PTR, KAFFINITY};
use winapi::shared::minwindef::DWORD;
use winapi::um::processthreadsapi::{GetCurrentProcess, GetCurrentThread};
use winapi::um::processtopologyapi::{GetThreadGroupAffinity, SetThreadGroupAffinity};
use winapi::um::sysinfoapi::GetLogicalProcessorInformationEx;
use winapi::um::winbase::GetProcessAffinityMask;
use winapi::um::winnt::{
    GROUP_AFFINITY, RelationProcessorCore, SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
};

use super::{CoreId, ThreadAffinity};

type Record = SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX;
const GROUP_STRIDE: usize = KAFFINITY::BITS as usize;

pub(super) struct Platform;

impl ThreadAffinity for Platform {
    fn active_cpus() -> Vec<CoreId> {
        let mut cpus: Vec<_> = match process_affinity() {
            Some(mask) => cpus_of(&mask).collect(),
            None => core_masks().iter().flat_map(cpus_of).collect(),
        };
        cpus.sort_unstable_by_key(|cpu| cpu.0);
        cpus.dedup();
        cpus
    }

    fn physical_core(cpu: CoreId) -> Option<CoreId> {
        core_masks().iter()
            .find(|mask| cpus_of(mask).any(|sibling| sibling == cpu))
            .and_then(|mask| cpus_of(mask).next())
    }

    fn pin_current(cpu: CoreId) {
        let affinity = GROUP_AFFINITY {
            Mask: 1 << (cpu.0 % GROUP_STRIDE),
            Group: (cpu.0 / GROUP_STRIDE) as u16,
            Reserved: [0; 3],
        };
        // SAFETY: affinity contains one valid group-local processor bit, and
        // the pseudo-handle refers only to this worker.
        unsafe { SetThreadGroupAffinity(GetCurrentThread(), &affinity, ptr::null_mut()) };
    }
}

fn process_affinity() -> Option<GROUP_AFFINITY> {
    let (mut process, mut system): (DWORD_PTR, DWORD_PTR) = (0, 0);
    // SAFETY: Both output masks point to initialized writable storage.
    let ok = unsafe { GetProcessAffinityMask(GetCurrentProcess(), &mut process, &mut system) };
    if ok == 0 || process == 0 {
        return None;
    }
    // SAFETY: The calling thread's group affinity fits the initialized output.
    let mut affinity: GROUP_AFFINITY = unsafe { mem::zeroed() };
    let ok = unsafe { GetThreadGroupAffinity(GetCurrentThread(), &mut affinity) };
    affinity.Mask = process;
    (ok != 0).then_some(affinity)
}

fn cpus_of(affinity: &GROUP_AFFINITY) -> impl Iterator<Item = CoreId> + use<> {
    let (mask, group) = (affinity.Mask, affinity.Group as usize);
    (0..GROUP_STRIDE)
        .filter(move |bit| mask >> bit & 1 == 1)
        .map(move |bit| CoreId(group * GROUP_STRIDE + bit))
}

fn core_masks() -> Vec<GROUP_AFFINITY> {
    let mut len: DWORD = 0;
    // SAFETY: A null buffer requests its required size.
    unsafe { GetLogicalProcessorInformationEx(RelationProcessorCore, ptr::null_mut(), &mut len) };
    if len == 0 { return Vec::new(); }
    let mut buffer = vec![0u64; (len as usize).div_ceil(mem::size_of::<u64>())];
    // SAFETY: The aligned buffer provides at least len writable bytes.
    let ok = unsafe {
        GetLogicalProcessorInformationEx(RelationProcessorCore, buffer.as_mut_ptr().cast(), &mut len)
    };
    if ok == 0 { return Vec::new(); }
    let bytes = bytemuck::cast_slice::<u64, u8>(&buffer);
    let bytes = &bytes[..(len as usize).min(bytes.len())];
    let header = mem::offset_of!(Record, u);
    let mut offset = 0;
    let mut masks = Vec::new();
    while offset + header <= bytes.len() {
        let start = offset + mem::offset_of!(Record, Size);
        let size = DWORD::from_ne_bytes(
            bytes[start..start + mem::size_of::<DWORD>()].try_into().unwrap(),
        ) as usize;
        if size < header || size > bytes.len() - offset { break; }
        // SAFETY: Copy only the available record prefix into aligned storage.
        // The core relationship has one group mask, including all SMT siblings.
        let mut record: Record = unsafe { mem::zeroed() };
        unsafe {
            ptr::copy_nonoverlapping(
                bytes[offset..].as_ptr(), ptr::from_mut(&mut record).cast::<u8>(),
                size.min(mem::size_of::<Record>()),
            );
        }
        if record.Relationship == RelationProcessorCore {
            let processor = unsafe { record.u.Processor() };
            if processor.GroupCount > 0 {
                masks.push(processor.GroupMask[0]);
            }
        }
        offset += size;
    }
    masks
}
