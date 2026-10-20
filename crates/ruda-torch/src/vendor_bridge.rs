use super::*;
use ruda_driver_cuda::interop;

type Callback = unsafe extern "C" fn(usize, *const usize, usize, usize) -> i32;

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_vendor_api_version() -> u32 { 1 }

// The adapter holds every Tensor and descriptor until the blocking call returns.
// Resource acquisition establishes allocation-origin stream dependencies; pins
// protect exported addresses against allocator relocation throughout the callback.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_vendor_scope(
    callback: Option<Callback>, opaque: usize, stream: u64,
    descriptors: *const Descriptor, count: usize,
) -> i32 {
    checked(|| {
        let callback = callback.expect("null vendor callback");
        assert!(count == 0 || !descriptors.is_null());
        let c = client();
        assert_eq!(stream, streams::current(), "vendor scope requires the selected RUDA stream");
        let views: Vec<_> = if count == 0 { Vec::new() } else {
            unsafe { crate::abi::slice(descriptors, count) }
                .iter().map(|d| unsafe { crate::abi::view(d) }).collect()
        };
        let resources: Vec<_> = views.iter().map(|v|
            c.get_resource(v.handle.clone()).expect("vendor allocation dependency failed")
        ).collect();
        let _pins: Vec<_> = resources.iter().map(|r| r.resource().address_pin()).collect();
        let pointers: Vec<usize> = resources.iter().map(|r| r.resource().ptr as usize).collect();
        unsafe {
            interop::native_scope(c.device_id(), stream, move |scope| {
                let code = callback(scope.stream, pointers.as_ptr(), pointers.len(), opaque);
                if code == 0 { Ok(()) } else { Err(format!("NVIDIA library callback failed: {code}")) }
            })
        }.expect("vendor CUDA scope failed");
    })
}
