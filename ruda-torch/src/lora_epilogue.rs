//! FP32 adapter scaling followed by low-precision rounding and residual add.
use super::*;
use super::scientific::address;

#[ruda(launch)]
fn forward<S: Float + RudaElement>(
    base: &Tensor<S>, update: &Tensor<f32>, output: &mut Tensor<S>, scale: f32,
) {
    let pos = ABSOLUTE_POS as usize;
    if pos < output.len() {
        let scaled = update[address(update, pos)] * scale;
        let stored = S::cast_from(scaled);
        output[pos] = S::cast_from(f32::cast_from(base[address(base, pos)]) + f32::cast_from(stored));
    }
}

#[ruda(launch)]
fn backward<S: Float + RudaElement>(grad: &Tensor<S>, output: &mut Tensor<f32>, scale: f32) {
    let pos = ABSOLUTE_POS as usize;
    if pos < output.len() { output[pos] = f32::cast_from(grad[address(grad, pos)]) * scale; }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64], values: &[f64]) {
    assert!(ints.is_empty()); assert_eq!(values.len(), 1);
    assert_eq!(xs.len(), if op == 654 { 3 } else { 2 });
    let base = &xs[0]; let output = xs.last().unwrap();
    assert!(base.dtype == 1 || base.dtype == 2);
    assert!(xs.iter().all(|x| x.shape == base.shape));
    if op == 654 {
        assert_eq!(xs[1].dtype, 0); assert_eq!(output.dtype, base.dtype);
    } else { assert_eq!(output.dtype, 0); }
    let mut stride = 1usize;
    for (&size, &actual) in output.shape.iter().zip(&output.strides).rev() {
        assert!(size <= 1 || actual == stride, "LoRA epilogue output must be contiguous");
        stride = stride.checked_mul(size).expect("LoRA epilogue shape overflow");
    }
    assert!(base.len <= u32::MAX as usize, "LoRA epilogue grid overflow");
    if base.len == 0 { return; }
    let c = client(); let grid = RudaCount::Static(base.len.div_ceil(128) as u32, 1, 1);
    let scale = values[0] as f32;
    macro_rules! run { ($s:ty) => { unsafe {
        if op == 654 {
            forward::launch::<$s,CudaRuntime>(&c,grid,RudaDim::new_1d(128),
                base.arg(),xs[1].arg(),output.arg(),scale);
        } else {
            backward::launch::<$s,CudaRuntime>(&c,grid,RudaDim::new_1d(128),base.arg(),output.arg(),scale);
        }
    } }; }
    match base.dtype {1=>run!(f16),2=>run!(bf16),_=>unreachable!()}
    LAUNCHES.fetch_add(1,Ordering::Relaxed);
}
