use super::*;

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_indexing_api_version() -> u32 { 1 }

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_advanced_index(
    source: *const Descriptor, indices: *const Descriptor, count: usize,
    sizes: *const i64, strides: *const i64, output: *const Descriptor,
) -> i32 {
    checked(|| {
        assert!(!source.is_null() && !output.is_null());
        let source_desc = unsafe { crate::abi::reference(source) };
        let source = unsafe { crate::abi::view(source_desc) };
        let output = unsafe { crate::abi::view(output) };
        assert_eq!(source.shape, output.shape);
        assert_eq!(source.dtype, output.dtype);
        if output.len == 0 { return; }
        if count == 0 { convert(&source, &output); return; }
        assert!(!indices.is_null() && !sizes.is_null() && !strides.is_null());
        let descriptors = unsafe { crate::abi::slice(indices, count) };
        let sizes = unsafe { crate::abi::slice(sizes, count) };
        let strides = unsafe { crate::abi::slice(strides, count) };
        let indices = descriptors.iter().map(|desc| unsafe { crate::abi::view(desc) }).collect::<Vec<_>>();
        let width = element_bytes(source.dtype);
        let mut span = width;
        for (&dim, &stride) in source.shape.iter().zip(&source.strides) {
            span = span.checked_add((dim - 1).checked_mul(stride).and_then(|n| n.checked_mul(width))
                .expect("advanced index span overflow")).expect("advanced index span overflow");
        }
        for (i, index) in indices.iter().enumerate() {
            assert_eq!(index.dtype, 4);
            assert_eq!(index.shape, output.shape);
            let size = usize::try_from(sizes[i]).expect("negative indexed size");
            let stride = usize::try_from(strides[i]).expect("negative indexed stride");
            assert!(size > 0);
            span = span.checked_add((size - 1).checked_mul(stride).expect("advanced index span overflow"))
                .expect("advanced index span overflow");
            primitives::check_indices(index, size, true);
        }
        let allocation = unsafe { crate::abi::reference(source_desc.allocation) };
        assert!(source_desc.offset_bytes.checked_add(span).is_some_and(|n| n <= allocation.bytes),
            "advanced index exceeds source allocation");
        let client = client();
        let addresses = View::packed(client.empty(output.len.checked_mul(8).expect("index allocation overflow")),
            output.shape.clone(), 4);
        let bytes = source.byte_view();
        let grid = u32::try_from(output.len.div_ceil(128)).expect("index grid overflow");
        for (i, index) in indices.iter().enumerate() {
            unsafe { kernels::advanced_index_offsets::launch::<CudaRuntime>(
                &client, RudaCount::Static(grid, 1, 1), RudaDim::new_1d(128),
                bytes.arg(), index.arg(), addresses.arg(), sizes[i], strides[i], i == 0); }
        }
        let out_bytes = output.byte_view();
        let grid = u32::try_from(out_bytes.len.div_ceil(128)).expect("index byte grid overflow");
        unsafe { kernels::advanced_index_copy::launch::<CudaRuntime>(
            &client, RudaCount::Static(grid, 1, 1), RudaDim::new_1d(128),
            bytes.arg(), addresses.arg(), out_bytes.arg()); }
        finish_dispatch(&client);
        LAUNCHES.fetch_add(count as u64 + 1, Ordering::Relaxed);
    })
}
