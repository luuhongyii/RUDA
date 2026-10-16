use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn floating<S: Float + RudaElement, F: Float + RudaElement, B: Float + RudaElement>(
    input: &Tensor<S>, lower: &Tensor<B>, upper: &Tensor<B>, output: &mut Tensor<S>,
    #[comptime] mode: u32, #[comptime] scalar: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < output.len() {
        let raw = input[address(input, p)]; let x = F::cast_from(raw);
        let mut lo = 0usize; let mut hi = 0usize;
        if comptime!(!scalar) { lo = address(lower, p); hi = address(upper, p); }
        let low = lower[lo]; let high = upper[hi];
        let y = F::cast_from(low); let z = F::cast_from(high);
        let mut result = x;
        if comptime!(mode == 0 || mode == 2 || mode == 5) { result = x.max(y); }
        else if comptime!(mode == 1 || mode == 3 || mode == 6) { result = x.min(y); }
        else { result = x.max(y).min(z); }
        let mut stored = S::cast_from(result);
        if comptime!(mode != 2 && mode != 3) {
            if comptime!(!scalar) {
                if comptime!(mode == 4) { if z != z { stored = S::cast_from(high); } }
                if y != y { stored = S::cast_from(low); }
            }
            if x != x { stored = raw; }
        }
        output[address(output, p)] = stored;
    }
}

#[ruda(launch)]
fn integral<I: Numeric + RudaElement>(
    input: &Tensor<I>, lower: &Tensor<I>, upper: &Tensor<I>, output: &mut Tensor<I>,
    #[comptime] mode: u32, #[comptime] scalar: bool, #[comptime] boolean: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < output.len() {
        let x = input[address(input, p)]; let mut lo = 0usize; let mut hi = 0usize;
        if comptime!(!scalar) { lo = address(lower, p); hi = address(upper, p); }
        let y = lower[lo]; let z = upper[hi]; let mut result = x;
        if comptime!(mode == 0 || mode == 2 || mode == 4 || mode == 5) { if result < y { result = y; } }
        if comptime!(mode == 1 || mode == 3 || mode == 6) { if result > y { result = y; } }
        if comptime!(mode == 4) { if result > z { result = z; } }
        if comptime!(boolean) {
            let a = x != I::cast_from(0u32); let b = y != I::cast_from(0u32);
            if comptime!(mode == 0 || mode == 2) { result = I::cast_from(a || b); }
            else { result = I::cast_from(a && b); }
        }
        output[address(output, p)] = result;
    }
}

pub(super) fn dispatch(xs: &[View], ints: &[i64]) {
    assert_eq!(xs.len(), 4); assert_eq!(ints.len(), 2);
    assert!((0..=6).contains(&ints[0]) && matches!(ints[1], 0 | 1));
    let mode = ints[0] as u32; let scalar = ints[1] != 0;
    assert!(if scalar { mode >= 4 } else { mode <= 4 });
    let input = &xs[0]; let output = &xs[3]; let code = input.dtype;
    assert!(code <= 9 && (code != 3 || mode < 4));
    assert_eq!(output.dtype, code); assert_eq!(output.shape, input.shape);
    let bound_code = if scalar && matches!(code, 1 | 2) { 0 } else { code };
    for bound in &xs[1..3] {
        assert_eq!(bound.dtype, bound_code);
        if scalar { assert_eq!(bound.len, 1); } else { assert_eq!(bound.shape, input.shape); }
    }
    if output.len == 0 { return; }
    let c = client(); let blocks = output.len.div_ceil(256); let x = blocks.min(131072);
    let grid = RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1); let block = RudaDim::new_1d(256);
    macro_rules! real { ($s:ty, $f:ty) => { unsafe {
        if scalar {
            floating::launch::<$s, $f, $f, CudaRuntime>(&c, grid, block, xs[0].arg(), xs[1].arg(), xs[2].arg(), xs[3].arg(), mode, scalar);
        } else {
            floating::launch::<$s, $f, $s, CudaRuntime>(&c, grid, block, xs[0].arg(), xs[1].arg(), xs[2].arg(), xs[3].arg(), mode, scalar);
        }
    }}; }
    macro_rules! integer { ($i:ty) => { unsafe {
        integral::launch::<$i, CudaRuntime>(&c, grid, block, xs[0].arg(), xs[1].arg(), xs[2].arg(), xs[3].arg(), mode, scalar, code == 3);
    }}; }
    match code {
        0 => real!(f32, f32), 1 => real!(f16, f32), 2 => real!(bf16, f32), 9 => real!(f64, f64),
        3 | 8 => integer!(u8), 4 => integer!(i64), 5 => integer!(i32), 6 => integer!(i16), 7 => integer!(i8), _ => unreachable!(),
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed); finish_dispatch(&c);
}
