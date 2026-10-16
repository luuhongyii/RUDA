use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn truth<S: Numeric>(real: S, imaginary: S, #[comptime] code: u32) -> bool {
    let mut result = real != S::cast_from(0u32);
    if comptime!(code == 12 || code == 13) { result = (u8::cast_from(real) & 127u8) != 0u8; }
    if comptime!(code == 10 || code == 11 || code == 14) { result = result || imaginary != S::cast_from(0u32); }
    result
}

#[ruda(launch)]
fn predicate<S: Numeric + RudaElement>(
    left: &Tensor<S>, left_imaginary: &Tensor<S>, right: &Tensor<S>, right_imaginary: &Tensor<S>,
    output: &mut Tensor<u8>, #[comptime] op: u32, #[comptime] code: u32,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < output.len() {
        let mut a = left[address(left, p)]; let mut b = S::cast_from(0u32);
        let mut ai = S::cast_from(0u32); let mut bi = S::cast_from(0u32);
        if comptime!(op != 88) { b = right[address(right, p)]; }
        if comptime!(code == 10 || code == 11 || code == 14) {
            ai = left_imaginary[address(left_imaginary, p)];
            if comptime!(op != 88) { bi = right_imaginary[address(right_imaginary, p)]; }
        }
        if comptime!(code == 3) { a = S::cast_from(a != S::cast_from(0u32)); b = S::cast_from(b != S::cast_from(0u32)); }
        let mut result = false;
        if comptime!(op >= 85) {
            let x = truth(a, ai, code); let y = truth(b, bi, code);
            if comptime!(op == 85) { result = x && y; }
            else if comptime!(op == 86) { result = x || y; }
            else if comptime!(op == 87) { result = x != y; }
            else { result = !x; }
        } else if comptime!(op == 78 || op == 79) {
            result = a == b;
            if comptime!(code == 10 || code == 11 || code == 14) { result = result && ai == bi; }
            if comptime!(code == 12 || code == 13) {
                let x = u8::cast_from(a); let y = u8::cast_from(b); let ax = x & 127u8; let ay = y & 127u8;
                let mut nan_x = ax == 127u8; let mut nan_y = ay == 127u8;
                if comptime!(code == 13) { nan_x = ax > 124u8; nan_y = ay > 124u8; }
                result = !nan_x && !nan_y && (x == y || ax == 0u8 && ay == 0u8);
            }
            if comptime!(op == 79) { result = !result; }
        } else if comptime!(op == 80) { result = a < b; }
        else if comptime!(op == 81) { result = a <= b; }
        else if comptime!(op == 82) { result = a > b; }
        else { result = a >= b; }
        output[address(output, p)] = u8::cast_from(result);
    }
}

fn part(value: &View, imaginary: bool) -> View {
    if matches!(value.dtype, 10 | 11 | 14) {
        let result = scientific::component(value, imaginary); scientific::indexable(&result); result
    } else {
        View { handle: value.handle.clone(), shape: value.shape.clone(), strides: value.strides.clone(), len: value.len, dtype: value.dtype }
    }
}

pub(super) fn dispatch(xs: &[View], ints: &[i64]) {
    assert_eq!(xs.len(), 3); assert_eq!(ints.len(), 1);
    assert!(matches!(ints[0], 78..=83 | 85..=88)); let op = ints[0] as u32;
    let code = xs[0].dtype; assert!(code <= 14); assert_eq!(xs[1].dtype, code); assert_eq!(xs[2].dtype, 3);
    assert!(!(80..=83).contains(&op) || code <= 9, "ordered comparison requires a supported real dtype");
    assert_eq!(xs[0].shape, xs[1].shape); assert_eq!(xs[0].shape, xs[2].shape);
    if xs[2].len == 0 { return; }
    let left = part(&xs[0], false); let left_i = part(&xs[0], true);
    let right = part(&xs[1], false); let right_i = part(&xs[1], true);
    let c = client(); let blocks = xs[2].len.div_ceil(256); let x = blocks.min(131072);
    let grid = RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1); let block = RudaDim::new_1d(256);
    macro_rules! launch { ($s:ty) => { unsafe {
        predicate::launch::<$s, CudaRuntime>(&c, grid, block, left.arg(), left_i.arg(), right.arg(), right_i.arg(), xs[2].arg(), op, code);
    }}; }
    match left.dtype {
        0 => launch!(f32), 1 => launch!(f16), 2 => launch!(bf16), 3 | 8 | 12 | 13 => launch!(u8),
        4 => launch!(i64), 5 => launch!(i32), 6 => launch!(i16), 7 => launch!(i8), 9 => launch!(f64), _ => unreachable!(),
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed); finish_dispatch(&c);
}
