use super::*;

#[ruda(launch)]
fn advanced_index_fused(source:&Tensor<u8>,indices:&Sequence<Tensor<i64>>,out:&mut Tensor<u8>,
    #[comptime] sizes:Vec<i64>,#[comptime] strides:Vec<i64>,#[comptime] width:usize)
{
    let pos=ABSOLUTE_POS as usize;
    if pos<out.len()/width {
        let mut address=i64::cast_from(crate::kernels::offset(source,pos*width));
        #[unroll] for axis in 0usize..sizes.len() {
            let input=indices.index(axis);
            let mut value=input[crate::kernels::offset(input,pos)];
            if value<0i64 {value+=comptime!(sizes[axis]);}
            address+=value*comptime!(strides[axis]);
        }
        let destination=crate::kernels::offset(out,pos*width);
        #[unroll] for byte in 0usize..width {out[destination+byte]=source[usize::cast_from(address)+byte];}
    }
}

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
        let bytes = source.byte_view();
        let grid = u32::try_from(output.len.div_ceil(128)).expect("index grid overflow");
        let out_bytes = output.byte_view();
        let index_args=indices.iter().map(|index|index.arg()).collect::<SequenceArg<CudaRuntime,Tensor<i64>>>();
        unsafe { advanced_index_fused::launch::<CudaRuntime>(
            &client, RudaCount::Static(grid, 1, 1), RudaDim::new_1d(128),
            bytes.arg(), index_args, out_bytes.arg(), sizes.to_vec(), strides.to_vec(), width); }
        finish_dispatch(&client);
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
    })
}
