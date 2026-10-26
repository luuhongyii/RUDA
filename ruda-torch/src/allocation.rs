use super::*;
use ruda::runtime::memory_management::MemoryResourcePin;

const SLAB_BYTES: usize = 1024 * 1024;
const SMALL_BYTES: usize = 64 * 1024;
const MEDIUM_SLAB_BYTES: usize = 8 * 1024 * 1024;
const MEDIUM_BYTES: usize = 2 * 1024 * 1024;
const ALIGNMENT: usize = 256;

struct Slab {
    handle: Handle,
    address: u64,
    pin: Option<MemoryResourcePin>,
    next: usize,
    stream: u64,
}

thread_local! {
    static SMALL: RefCell<Option<Slab>> = const { RefCell::new(None) };
    static MEDIUM: RefCell<Option<Slab>> = const { RefCell::new(None) };
}

pub(super) fn allocate(bytes: usize) -> (Allocation, u64) {
    let physical = bytes.max(4);
    if async_dispatch_enabled() && physical <= MEDIUM_BYTES {
        let (pool, capacity) = if physical <= SMALL_BYTES { (&SMALL, SLAB_BYTES) }
            else { (&MEDIUM, MEDIUM_SLAB_BYTES) };
        return pool.with(|slot| {
            let mut slot = slot.borrow_mut();
            let stream = streams::current();
            let span = physical.div_ceil(ALIGNMENT) * ALIGNMENT;
            if slot.as_ref().is_none_or(|slab| slab.stream != stream || slab.next + span > capacity) {
                let c = client();
                let handle = c.empty(capacity);
                let resource = c.get_resource(handle.clone()).expect("RUDA CUDA slab allocation failed");
                *slot = Some(Slab { handle, address: resource.resource().ptr,
                    pin: resource.resource().address_pin(), next: 0, stream });
            }
            let slab = slot.as_mut().unwrap();
            let start = slab.next;
            slab.next += span;
            // Slices are never recycled in a live slab. Existing native binding
            // lifetimes and address leases retain its backing after rollover.
            let handle = slab.handle.clone().offset_start(start as u64)
                .offset_end((capacity - start - physical) as u64);
            (Allocation { handle, bytes, _pin: slab.pin.clone() }, slab.address + start as u64)
        });
    }
    let c = client();
    let handle = c.empty(physical);
    let resource = c.get_resource(handle.clone()).expect("RUDA CUDA allocation failed");
    if !async_dispatch_enabled() { sync(&c); }
    let address = resource.resource().ptr;
    (Allocation { handle, bytes, _pin: resource.resource().address_pin() }, address)
}
