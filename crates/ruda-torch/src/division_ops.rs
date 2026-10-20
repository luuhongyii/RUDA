use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn signed_zero<F: Float>(value: F, #[comptime] code: u32) -> F {
    let mut mask = 0x80000000u64;
    if comptime!(code == 9) { mask = 0x8000000000000000u64; }
    crate::floating::from_bits::<F>(crate::floating::bits::<F>(value, code) & mask, code)
}

#[ruda(launch)]
fn floating<S: Float + RudaElement, F: Float>(
    left: &Tensor<S>, right: &Tensor<S>, output: &mut Tensor<S>, scalar: u64, reciprocal: u64,
    #[comptime] op: u32, #[comptime] scalar_right: bool, #[comptime] code: u32,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < output.len() {
        let a = F::cast_from(left[address(left, p)]);
        let mut b = crate::floating::from_bits::<F>(scalar, code);
        if comptime!(!scalar_right) { b = F::cast_from(right[address(right, p)]); }
        let inv_b = crate::floating::from_bits::<F>(reciprocal, code);
        let zero = F::new(0.0); let one = F::new(1.0);
        let mut quotient = a / b;
        if comptime!(scalar_right) { quotient = a * inv_b; }
        let mut result = quotient;
        if comptime!(op == 1) {
            if comptime!(!scalar_right) { quotient = F::cast_from(S::cast_from(quotient)); }
            result = quotient.trunc();
        } else if comptime!(op == 2) {
            if b != zero {
                let rem = a % b;
                let mut div = (a - rem) / b;
                if comptime!(scalar_right) { div = (a - rem) * inv_b; }
                if rem != zero && ((b < zero) != (rem < zero)) { div -= one; }
                if div != zero {
                    result = F::cast_from(S::cast_from(div.floor()));
                    if div - result > F::new(0.5) { result = F::cast_from(S::cast_from(result + one)); }
                } else { result = signed_zero::<F>(quotient, code); }
            }
        } else if comptime!(op >= 3) {
            result = a % b;
            if comptime!(op == 3) {
                if result != zero && ((b < zero) != (result < zero)) { result += b; }
            }
        }
        output[address(output, p)] = S::cast_from(result);
    }
}

#[ruda(launch)]
fn integral<S: Numeric + RudaElement>(
    left: &Tensor<S>, right: &Tensor<S>, output: &mut Tensor<S>, minimum: i64,
    #[comptime] op: u32, #[comptime] signed: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < output.len() {
        let a = left[address(left, p)]; let b = right[address(right, p)];
        let zero = S::cast_from(0u32); let one = S::cast_from(1u32); let mut result = zero;
        if comptime!(op == 1 || op == 2 && !signed) { result = a / b; }
        else if comptime!(op == 2) {
            if b != zero {
                if a == S::cast_from(minimum) && b == zero - one { result = a; }
                else {
                    result = a / b;
                    if a % b != zero && ((a < zero) != (b < zero)) { result -= one; }
                }
            }
        } else {
            if comptime!(op == 3 && !signed) {
                if b == zero { result = S::cast_from(255u32); } else { result = a % b; }
            } else {
                result = a % b;
                if comptime!(op == 3) {
                    if result != zero && ((result < zero) != (b < zero)) { result += b; }
                }
            }
        }
        output[address(output, p)] = result;
    }
}

pub(super) fn dispatch(xs: &[View], ints: &[i64]) {
    assert_eq!(xs.len(), 3); assert_eq!(ints.len(), 4);
    assert!((0..=4).contains(&ints[0]) && matches!(ints[1], 0 | 1));
    let op = ints[0] as u32; let scalar_right = ints[1] != 0; let code = xs[2].dtype;
    assert!(code <= 9 && code != 3); assert_eq!(xs[0].dtype, code); assert_eq!(xs[1].dtype, code);
    assert_eq!(xs[0].shape, xs[2].shape); assert_eq!(xs[1].shape, xs[2].shape);
    let integral = (4..=8).contains(&code);
    assert!(!integral || op != 0 && !scalar_right);
    assert!(!scalar_right || op <= 2);
    if xs[2].len == 0 { return; }
    let c = client(); let blocks = xs[2].len.div_ceil(256); let x = blocks.min(131072);
    let grid = RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1); let block = RudaDim::new_1d(256);
    macro_rules! real { ($s:ty, $f:ty) => { unsafe {
        floating::launch::<$s, $f, CudaRuntime>(&c, grid, block, xs[0].arg(), xs[1].arg(), xs[2].arg(),
            ints[2] as u64, ints[3] as u64, op, scalar_right, if code == 9 { 9 } else { 0 });
    }}; }
    macro_rules! integer { ($s:ty) => { unsafe {
        integral::launch::<$s, CudaRuntime>(&c, grid, block, xs[0].arg(), xs[1].arg(), xs[2].arg(), <$s>::MIN as i64, op, code != 8);
    }}; }
    match code {
        0 => real!(f32, f32), 1 => real!(f16, f32), 2 => real!(bf16, f32), 9 => real!(f64, f64),
        4 => integer!(i64), 5 => integer!(i32), 6 => integer!(i16), 7 => integer!(i8), 8 => integer!(u8), _ => unreachable!(),
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed); finish_dispatch(&c);
}
