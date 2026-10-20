use super::{Descriptor, View, element_bytes};

/// # Safety
/// The pointer must identify an aligned, initialized value that remains valid
/// and shared-borrowable for the caller-selected lifetime.
pub(super) unsafe fn reference<'a, T>(ptr: *const T) -> &'a T {
    assert!(!ptr.is_null() && ptr.is_aligned(), "invalid native ABI pointer");
    unsafe { &*ptr }
}

/// # Safety
/// For nonempty inputs, the caller owns one initialized, aligned allocation
/// covering all elements and keeps it alive and unmodified for the borrow.
/// These checks do not establish pointer provenance or allocation lifetime.
pub(super) unsafe fn slice<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if len == 0 { return &[]; }
    assert!(!ptr.is_null() && ptr.is_aligned(), "invalid native ABI slice");
    assert!(len.checked_mul(std::mem::size_of::<T>())
        .is_some_and(|bytes| bytes <= isize::MAX as usize), "native ABI slice size overflow");
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

// The C++ adapter owns the descriptor arrays and allocation for the entire call.
pub(super) unsafe fn view(desc: *const Descriptor) -> View {
    let desc = unsafe { reference(desc) };
    assert!(!desc.allocation.is_null());
    let allocation = unsafe { reference(desc.allocation) };
    let (shape, strides) = if desc.rank == 0 {
        (vec![1], vec![1])
    } else {
        assert!(!desc.shape.is_null() && !desc.strides.is_null());
        unsafe { (slice(desc.shape, desc.rank).to_vec(),
                  slice(desc.strides, desc.rank).to_vec()) }
    };
    let len = shape.iter().try_fold(1usize, |a, b| a.checked_mul(*b)).expect("shape overflow");
    let span = if len == 0 { 0 } else {
        shape.iter().zip(&strides).try_fold(1usize, |n, (&d, &s)|
            (d - 1).checked_mul(s).and_then(|v| n.checked_add(v))).expect("stride overflow")
    };
    let bytes = element_bytes(desc.dtype);
    assert_eq!(desc.offset_bytes % bytes, 0);
    assert!(span.checked_mul(bytes).and_then(|n| n.checked_add(desc.offset_bytes))
        .is_some_and(|n| n <= allocation.bytes), "tensor exceeds allocation");
    View { handle: allocation.handle.clone().offset_start(desc.offset_bytes as u64), shape, strides, len, dtype: desc.dtype }
}
