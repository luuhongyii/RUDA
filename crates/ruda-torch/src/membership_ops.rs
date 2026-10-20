use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn membership<E: Numeric + RudaElement>(
    ar: &Tensor<E>, ai: &Tensor<E>, br: &Tensor<E>, bi: &Tensor<E>, out: &mut Tensor<u8>,
    #[comptime] complex: bool, #[comptime] invert: bool,
) {
    let p = ABSOLUTE_POS as usize;
    if p < out.len() {
        let value = ar[address(ar, p)];
        let mut imaginary = E::cast_from(0u32);
        if comptime!(complex) { imaginary = ai[address(ai, p)]; }
        let mut found = false;
        let mut index = 0usize;
        while index < br.len() && !found {
            let mut equal = value == br[address(br, index)];
            if comptime!(complex) { equal = equal && imaginary == bi[address(bi, index)]; }
            found = equal;
            index += 1usize;
        }
        out[address(out, p)] = u8::cast_from(found != invert);
    }
}

#[ruda(launch)]
fn adjacent_mask<E: Numeric + RudaElement>(
    values: &Tensor<E>, order: &Tensor<i64>, out: &mut Tensor<u8>, #[comptime] invert: bool,
) {
    let p = ABSOLUTE_POS as usize;
    if p < values.len() {
        let mut duplicate = false;
        if p + 1usize < values.len() {
            duplicate = values[address(values, p)] == values[address(values, p + 1usize)];
        }
        let destination = usize::cast_from(order[address(order, p)]);
        out[address(out, destination)] = u8::cast_from(duplicate != invert);
    }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64]) {
    assert!(matches!(op, 402 | 403));
    assert_eq!(ints.len(), if op == 402 { 2 } else { 1 });
    assert!(ints.iter().all(|&flag| matches!(flag, 0 | 1)));
    assert_eq!(xs.len(), if op == 402 { 5 } else { 3 });
    let code = xs[0].dtype;
    assert!(matches!(code, 0 | 1 | 2 | 4 | 5 | 6 | 7 | 8 | 9));
    let out = xs.last().unwrap();
    assert_eq!(out.dtype, 3);
    assert_eq!(out.shape, xs[0].shape);
    let complex = op == 402 && ints[1] != 0;
    if op == 402 {
        for value in &xs[1..4] { assert_eq!(value.dtype, code); }
        assert_eq!(xs[1].shape, xs[0].shape);
        assert_eq!(xs[2].shape, xs[3].shape);
        if complex { assert!(matches!(code, 0 | 1 | 9)); }
    } else {
        assert_eq!(xs[0].shape.len(), 1);
        assert_eq!(xs[1].dtype, 4);
        assert_eq!(xs[1].shape, xs[0].shape);
    }
    if out.len == 0 { return; }
    let c = client();
    let grid = RudaCount::Static(u32::try_from(out.len.div_ceil(256)).expect("membership launch overflow"), 1, 1);
    let block = RudaDim::new_1d(256);
    macro_rules! run { ($e:ty) => { unsafe {
        if op == 402 {
            membership::launch::<$e, CudaRuntime>(&c, grid, block,
                xs[0].arg(), xs[1].arg(), xs[2].arg(), xs[3].arg(), xs[4].arg(), complex, ints[0] != 0);
        } else {
            adjacent_mask::launch::<$e, CudaRuntime>(&c, grid, block,
                xs[0].arg(), xs[1].arg(), xs[2].arg(), ints[0] != 0);
        }
    }}; }
    match code {
        0 => run!(f32), 1 => run!(f16), 2 => run!(bf16), 4 => run!(i64),
        5 => run!(i32), 6 => run!(i16), 7 => run!(i8), 8 => run!(u8),
        9 => run!(f64), _ => unreachable!(),
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    finish_dispatch(&c);
}
