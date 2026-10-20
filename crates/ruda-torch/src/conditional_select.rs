use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn select<S: Numeric + RudaElement>(
    mask: &Tensor<u8>, left: &Tensor<S>, left_high: &Tensor<S>, right: &Tensor<S>, right_high: &Tensor<S>,
    output: &mut Tensor<S>, output_high: &mut Tensor<S>, #[comptime] wide: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < output.len() {
        let to = address(output, p);
        if mask[address(mask, p)] != 0u8 {
            output[to] = left[address(left, p)];
            if comptime!(wide) { output_high[address(output_high, p)] = left_high[address(left_high, p)]; }
        } else {
            output[to] = right[address(right, p)];
            if comptime!(wide) { output_high[address(output_high, p)] = right_high[address(right_high, p)]; }
        }
    }
}

#[ruda(launch)]
fn fill<S: Numeric + RudaElement>(
    input: &Tensor<S>, input_high: &Tensor<S>, mask: &Tensor<u8>, output: &mut Tensor<S>, output_high: &mut Tensor<S>,
    low: u64, high: u64, #[comptime] wide: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < output.len() {
        let to = address(output, p);
        if mask[address(mask, p)] != 0u8 {
            output[to] = S::cast_from(low);
            if comptime!(wide) { output_high[address(output_high, p)] = S::cast_from(high); }
        } else {
            output[to] = input[address(input, p)];
            if comptime!(wide) { output_high[address(output_high, p)] = input_high[address(input_high, p)]; }
        }
    }
}

fn word(value: &View, high: bool) -> View {
    if value.dtype == 11 {
        let result = scientific::component(value, high); scientific::indexable(&result); result
    } else {
        View { handle: value.handle.clone(), shape: value.shape.clone(), strides: value.strides.clone(), len: value.len, dtype: value.dtype }
    }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64]) {
    assert!(matches!(op, 482 | 483));
    assert_eq!(xs.len(), if op == 482 { 4 } else { 3 });
    assert_eq!(ints.len(), if op == 482 { 0 } else { 2 });
    let output = xs.last().unwrap(); let code = output.dtype;
    assert!(code <= 14);
    let mask_index = if op == 482 { 0 } else { 1 };
    let mask = &xs[mask_index]; assert_eq!(mask.dtype, 3);
    for (i, value) in xs.iter().enumerate() {
        assert_eq!(value.shape, output.shape);
        if i != mask_index { assert_eq!(value.dtype, code); }
    }
    if output.len == 0 { return; }
    let c = client(); let blocks = output.len.div_ceil(256); let x = blocks.min(131072);
    let grid = RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1); let block = RudaDim::new_1d(256);
    let source = &xs[if op == 482 { 1 } else { 0 }];
    let left = word(source, false); let left_high = word(source, true);
    let out = word(output, false); let out_high = word(output, true); let wide = code == 11;
    let other = &xs[2]; let right = word(other, false); let right_high = word(other, true);
    macro_rules! launch { ($s:ty) => { unsafe {
        if op == 482 {
            select::launch::<$s, CudaRuntime>(&c, grid, block, mask.arg(), left.arg(), left_high.arg(), right.arg(), right_high.arg(), out.arg(), out_high.arg(), wide);
        } else {
            fill::launch::<$s, CudaRuntime>(&c, grid, block, left.arg(), left_high.arg(), mask.arg(), out.arg(), out_high.arg(), ints[0] as u64, ints[1] as u64, wide);
        }
    }}; }
    match element_bytes(code) {
        1 => launch!(u8), 2 => launch!(u16), 4 => launch!(u32), 8 | 16 => launch!(u64), _ => unreachable!(),
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed); finish_dispatch(&c);
}
