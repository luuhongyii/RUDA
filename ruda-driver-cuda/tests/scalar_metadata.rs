use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda(launch, address_type = "dynamic")]
fn multiply_with_metadata(a: &Tensor<f32>, b: &Tensor<f32>, out: &mut Tensor<f32>, bias: f32) {
    let mut value = a[0] * b[0] + bias;
    if a.rank() > 0 {
        value += f32::cast_from(a.shape(0) + a.stride(0));
    }
    out[0] = value;
}

fn check(address: AddressType, ranked_input: bool) {
    let client = CudaRuntime::client(&CudaDevice::default());
    let a = client.create_from_slice(f32::as_bytes(&[2.0]));
    let b = client.create_from_slice(f32::as_bytes(&[4.0]));
    let output = client.empty(4);
    let shape = if ranked_input { vec![1] } else { vec![] };
    let strides = shape.clone();
    // SAFETY: Every buffer contains one f32 and exactly one thread is launched.
    // The rank-zero arguments have no dynamic shape/stride payload.
    unsafe {
        multiply_with_metadata::launch::<CudaRuntime>(
            &client, RudaCount::Static(1, 1, 1), RudaDim::new_1d(1), address,
            TensorArg::from_raw_parts(a, strides.into(), shape.into()),
            TensorArg::from_raw_parts(b, Vec::<usize>::new().into(), Vec::<usize>::new().into()),
            TensorArg::from_raw_parts(output.clone(), Vec::<usize>::new().into(), Vec::<usize>::new().into()),
            0.5f32,
        );
    }
    let bytes = client.read_one(output).unwrap();
    assert_eq!(f32::from_bytes(&bytes), &[if ranked_input { 10.5 } else { 8.5 }]);
}

#[test]
fn empty_dynamic_metadata_u32() { check(AddressType::U32, false); }

#[test]
fn empty_dynamic_metadata_u64() { check(AddressType::U64, false); }

#[test]
fn nonempty_dynamic_metadata_keeps_argument_order() {
    check(AddressType::U32, true);
    check(AddressType::U64, true);
}
