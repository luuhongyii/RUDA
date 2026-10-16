use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn floating<A: Float + RudaElement, B: Float + RudaElement, S: Float + RudaElement, F: Float>(
    left: &Tensor<A>, right: &Tensor<B>, output: &mut Tensor<S>, alpha: u64,
    #[comptime] op: u32, #[comptime] scalar_left: bool, #[comptime] scalar_right: bool, #[comptime] code: u32,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < output.len() {
        let mut i = 0usize; let mut j = 0usize;
        if comptime!(!scalar_left) { i = address(left, p); }
        let a = F::cast_from(left[i]); let mut result = -a;
        if comptime!(op != 2) {
            if comptime!(!scalar_right) { j = address(right, p); }
            let b = F::cast_from(right[j]);
            if comptime!(op == 0) { result = fma(crate::floating::from_bits::<F>(alpha, code), b, a); }
            else { result = a * b; }
        }
        output[address(output, p)] = S::cast_from(result);
    }
}

#[ruda(launch)]
fn integral<S: Numeric + RudaElement>(
    left: &Tensor<S>, right: &Tensor<S>, output: &mut Tensor<S>, alpha: u64,
    #[comptime] op: u32, #[comptime] scalar_left: bool, #[comptime] scalar_right: bool, #[comptime] boolean: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < output.len() {
        let mut i = 0usize; let mut j = 0usize;
        if comptime!(!scalar_left) { i = address(left, p); }
        let a = u64::cast_from(left[i]); let mut result = 0u64 - a;
        if comptime!(op != 2) {
            if comptime!(!scalar_right) { j = address(right, p); }
            let b = u64::cast_from(right[j]);
            if comptime!(boolean) {
                if comptime!(op == 0) { result = u64::cast_from(a != 0u64 || alpha != 0u64 && b != 0u64); }
                else { result = u64::cast_from(a != 0u64 && b != 0u64); }
            } else if comptime!(op == 0) { result = a + alpha * b; }
            else { result = a * b; }
        }
        output[address(output, p)] = S::cast_from(result);
    }
}

pub(super) fn dispatch(xs: &[View], ints: &[i64]) {
    assert_eq!(xs.len(), 3); assert_eq!(ints.len(), 4);
    assert!((0..=2).contains(&ints[0]) && matches!(ints[1], 0 | 1) && matches!(ints[2], 0 | 1));
    let op = ints[0] as u32; let scalar_left = ints[1] != 0; let scalar_right = ints[2] != 0;
    assert!(ints[1] + ints[2] <= 1); assert!(op != 2 || !scalar_left && !scalar_right);
    let code = xs[2].dtype; assert!(code <= 9 && (op != 2 || code != 3));
    let math_code = if matches!(code, 1 | 2) { 0 } else { code };
    for (input, scalar) in [(&xs[0], scalar_left), (&xs[1], scalar_right)] {
        assert_eq!(input.dtype, if scalar { math_code } else { code });
        if scalar { assert_eq!(input.len, 1); } else { assert_eq!(input.shape, xs[2].shape); }
    }
    if xs[2].len == 0 { return; }
    let c = client(); let blocks = xs[2].len.div_ceil(256); let x = blocks.min(131072);
    let grid = RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1); let block = RudaDim::new_1d(256);
    let alpha = ints[3] as u64;
    macro_rules! real { ($s:ty, $f:ty) => { unsafe {
        if scalar_left {
            floating::launch::<$f, $s, $s, $f, CudaRuntime>(&c, grid, block, xs[0].arg(), xs[1].arg(), xs[2].arg(), alpha, op, true, false, math_code);
        } else if scalar_right {
            floating::launch::<$s, $f, $s, $f, CudaRuntime>(&c, grid, block, xs[0].arg(), xs[1].arg(), xs[2].arg(), alpha, op, false, true, math_code);
        } else {
            floating::launch::<$s, $s, $s, $f, CudaRuntime>(&c, grid, block, xs[0].arg(), xs[1].arg(), xs[2].arg(), alpha, op, false, false, math_code);
        }
    }}; }
    macro_rules! integer { ($s:ty) => { unsafe {
        integral::launch::<$s, CudaRuntime>(&c, grid, block, xs[0].arg(), xs[1].arg(), xs[2].arg(), alpha, op, scalar_left, scalar_right, code == 3);
    }}; }
    match code {
        0 => real!(f32, f32), 1 => real!(f16, f32), 2 => real!(bf16, f32), 9 => real!(f64, f64),
        3 | 8 => integer!(u8), 4 => integer!(i64), 5 => integer!(i32), 6 => integer!(i16), 7 => integer!(i8), _ => unreachable!(),
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed); finish_dispatch(&c);
}
