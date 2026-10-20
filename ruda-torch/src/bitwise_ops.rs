use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn shift<I: Numeric + RudaElement>(
    a: &Tensor<I>, b: &Tensor<I>, out: &mut Tensor<I>,
    #[comptime] right: bool, #[comptime] width: u32, #[comptime] signed: bool,
) {
    let p = ABSOLUTE_POS as usize;
    if p < out.len() {
        let value = i64::cast_from(a[address(a, p)]);
        let count = i64::cast_from(b[address(b, p)]);
        let mut result = 0i64;
        if comptime!(right) {
            let mut limit = i64::cast_from(width);
            if comptime!(signed) { limit -= 1i64; }
            let mut amount = count;
            if count < 0i64 || count >= limit { amount = limit; }
            result = value >> amount;
        } else if count >= 0i64 && count < i64::cast_from(width) {
            result = i64::cast_from(u64::cast_from(value) << u64::cast_from(count));
        }
        out[address(out, p)] = I::cast_from(result);
    }
}

pub(super) fn dispatch(op: u32, xs: &[View]) {
    assert!(matches!(op, 397 | 398));
    assert_eq!(xs.len(), 3);
    let code = xs[0].dtype;
    assert!((4..=8).contains(&code));
    for value in &xs[1..] {
        assert_eq!(value.dtype, code);
        assert_eq!(value.shape, xs[0].shape);
    }
    if xs[2].len == 0 { return; }
    let c = client();
    let grid = RudaCount::Static(u32::try_from(xs[2].len.div_ceil(256)).expect("shift launch overflow"), 1, 1);
    let block = RudaDim::new_1d(256);
    macro_rules! launch { ($i:ty, $width:expr, $signed:expr) => { unsafe {
        shift::launch::<$i, CudaRuntime>(&c, grid, block,
            xs[0].arg(), xs[1].arg(), xs[2].arg(), op == 398, $width, $signed);
    }}; }
    match code {
        4 => launch!(i64, 64, true), 5 => launch!(i32, 32, true),
        6 => launch!(i16, 16, true), 7 => launch!(i8, 8, true),
        8 => launch!(u8, 8, false), _ => unreachable!(),
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    finish_dispatch(&c);
}
