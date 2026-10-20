use super::*;
use rudnn::kda::{KdaInput, chunk_kda, chunk_kda_backward};

fn input(v: &[View], scale: f32) -> KdaInput<CudaRuntime> {
    KdaInput {
        query: primitives::tensor(&v[0]), key: primitives::tensor(&v[1]),
        value: primitives::tensor(&v[2]), beta: primitives::tensor(&v[3]),
        log_decay: primitives::tensor(&v[4]), initial_state: primitives::tensor(&v[5]),
        query_scale: scale,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_kda_api_version() -> u32 { 1 }

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_kda_forward(
    descriptors: *const Descriptor, count: usize, scale: f32, chunk: u32, retain: bool,
) -> i32 {
    checked(|| {
        assert!(!descriptors.is_null() && count == 9 && scale.is_finite() && chunk > 0);
        let ds = unsafe { crate::abi::slice(descriptors, count) };
        let v: Vec<_> = ds.iter().map(|d| unsafe { crate::abi::view(d) }).collect();
        let result = chunk_kda(input(&v, scale), chunk as usize, retain).expect("RUDA KDA forward failed");
        primitives::store(result.output, &v[6]);
        primitives::store(result.final_state, &v[7]);
        if let Some(boundaries) = result.boundaries { primitives::store(boundaries, &v[8]); }
        else { assert_eq!(v[8].len, 0); }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_kda_backward(
    descriptors: *const Descriptor, count: usize, scale: f32, chunk: u32,
) -> i32 {
    checked(|| {
        assert!(!descriptors.is_null() && count == 15 && scale.is_finite() && chunk > 0);
        let ds = unsafe { crate::abi::slice(descriptors, count) };
        let v: Vec<_> = ds.iter().map(|d| unsafe { crate::abi::view(d) }).collect();
        let gradients = chunk_kda_backward(input(&v, scale), primitives::tensor(&v[6]),
            primitives::tensor(&v[7]), primitives::tensor(&v[8]), chunk as usize)
            .expect("RUDA KDA backward failed");
        for (tensor, destination) in [gradients.query, gradients.key, gradients.value,
            gradients.beta, gradients.log_decay, gradients.initial_state].into_iter().zip(&v[9..])
        { primitives::store(tensor, destination); }
    })
}
