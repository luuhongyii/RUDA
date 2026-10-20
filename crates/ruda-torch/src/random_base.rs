use super::*;
use super::random_extra::{unit, word, word64};
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn fraction<F: Float>(index: u64, seed_low: u32, seed_high: u32, counter: u64, #[comptime] wide: bool) -> F {
    let mut result = F::new(0.0);
    if comptime!(wide) {
        result = F::cast_from(word64(index * 2u64, seed_low, seed_high, counter) >> 11u32)
            * F::new(1.0 / 9007199254740992.0);
    } else {
        result = F::cast_from(word(index, seed_low, seed_high, counter) >> 8u32)
            * F::new(1.0 / 16777216.0);
    }
    result
}

#[ruda]
fn normal<F: Float>(index: u64, seed_low: u32, seed_high: u32, counter: u64, #[comptime] wide: bool) -> F {
    let pair = index / 2u64;
    let mut radius_unit = F::new(0.0);
    let mut angle = F::new(0.0);
    if comptime!(wide) {
        radius_unit = unit::<F>(pair * 2u64, seed_low, seed_high, counter, true);
        angle = fraction::<F>(pair * 2u64 + 1u64, seed_low, seed_high, counter, true)
            * F::cast_from(6.28318530717958647692f64);
    } else {
        let u = word(pair * 2u64, seed_low, seed_high, counter);
        let v = word(pair * 2u64 + 1u64, seed_low, seed_high, counter);
        radius_unit = (F::cast_from(u >> 9u32) + F::new(1.0)) / F::new(8388609.0);
        angle = F::cast_from(v >> 8u32) * F::new(6.283185307179586f32 / 16777216.0f32);
    }
    let radius = (-F::new(2.0) * radius_unit.ln()).sqrt();
    let mut result = radius * angle.cos();
    if index % 2u64 == 1u64 { result = radius * angle.sin(); }
    result
}

#[ruda(launch)]
fn fill<S: Numeric + RudaElement, F: Float + RudaElement>(
    output: &mut Tensor<S>, seed_low: u32, seed_high: u32, counter: u64,
    first: F, second: F, range: F, #[comptime] op: u32, #[comptime] wide: bool,
) {
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let index = u64::cast_from(p);
        let mut value = F::new(0.0);
        if comptime!(op == 431) {
            value = first + second * normal::<F>(index, seed_low, seed_high, counter, wide);
        } else {
            let sample = fraction::<F>(index, seed_low, seed_high, counter, wide);
            if comptime!(op == 430) {
                value = F::cast_from(S::cast_from(first + range * sample));
                if value == second { value = first; }
            } else {
                if sample < first { value = F::new(1.0); }
            }
        }
        output[address(output, p)] = S::cast_from(value);
    }
}

#[ruda(launch)]
fn bernoulli<S: Numeric + RudaElement, F: Float + RudaElement>(
    probabilities: &Tensor<F>, output: &mut Tensor<S>, seed_low: u32, seed_high: u32, counter: u64,
) {
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let probability = probabilities[address(probabilities, p)];
        if !(probability >= F::new(0.0) && probability <= F::new(1.0)) {
            intrinsic!(|scope| { scope.register(ruda_core::ir::Branch::Unreachable); });
        }
        let sample = unit::<f32>(u64::cast_from(p), seed_low, seed_high, counter, false);
        output[address(output, p)] = S::cast_from(F::cast_from(sample) <= probability);
    }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64], values: &[f64]) {
    assert!((430..=433).contains(&op));
    assert_eq!(xs.len(), if op == 433 { 2 } else { 1 });
    assert_eq!(ints.len(), 4);
    assert!(ints.iter().all(|&n| (0..=u32::MAX as i64).contains(&n)));
    assert_eq!(values.len(), (433 - op) as usize);
    let output = xs.last().unwrap();
    let code = output.dtype;
    if op <= 431 { assert!(matches!(code, 0 | 1 | 2 | 9)); }
    else { assert!(code <= 9); }
    if op == 430 {
        assert!(values[0].is_finite() && values[1].is_finite());
        assert!(values[0] <= values[1] && values[2] >= 0.0);
    } else if op == 431 { assert!(values[1] >= 0.0); }
    else if op == 432 { assert!((0.0..=1.0).contains(&values[0])); }
    else {
        assert_eq!(xs[0].shape, output.shape);
        assert_eq!(xs[0].dtype, if code == 9 { 9 } else { 0 });
    }
    if output.len == 0 { return; }
    let wide = code == 9 && op != 433;
    let counter = ints[2] as u64 | ((ints[3] as u64) << 32);
    let blocks = if wide { output.len.div_ceil(2) } else { output.len.div_ceil(4) };
    counter.checked_add(blocks as u64).expect("RUDA random counter exhausted");
    let first = values.first().copied().unwrap_or(0.0);
    let second = values.get(1).copied().unwrap_or(0.0);
    let range = values.get(2).copied().unwrap_or(0.0);
    let c = client();
    let grid = RudaCount::Static(u32::try_from(output.len.div_ceil(256)).expect("random launch overflow"), 1, 1);
    let block = RudaDim::new_1d(256);
    macro_rules! run { ($s:ty, $f:ty) => { unsafe {
        if op == 433 {
            bernoulli::launch::<$s, $f, CudaRuntime>(&c, grid, block, xs[0].arg(), output.arg(),
                ints[0] as u32, ints[1] as u32, counter);
        } else {
            fill::launch::<$s, $f, CudaRuntime>(&c, grid, block, output.arg(),
                ints[0] as u32, ints[1] as u32, counter, first as $f, second as $f, range as $f, op, wide);
        }
    }}; }
    match code {
        0 => run!(f32, f32), 1 => run!(f16, f32), 2 => run!(bf16, f32),
        3 | 8 => run!(u8, f32), 4 => run!(i64, f32), 5 => run!(i32, f32),
        6 => run!(i16, f32), 7 => run!(i8, f32), 9 => run!(f64, f64), _ => unreachable!(),
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    finish_dispatch(&c);
}
