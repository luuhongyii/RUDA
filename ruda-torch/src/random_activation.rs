use super::*;
use super::random_extra::unit;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn dropout<S: Float + RudaElement, F: Float + RudaElement>(
    input: &Tensor<S>, output: &mut Tensor<S>, mask: &mut Tensor<u8>,
    seed_low: u32, seed_high: u32, counter: u64, keep: F,
) {
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let sample = unit::<f32>(u64::cast_from(p), seed_low, seed_high, counter, false);
        let retained = F::cast_from(sample) < keep;
        let scale = F::cast_from(1.0f64 / f64::cast_from(keep));
        let x = F::cast_from(input[address(input, p)]);
        output[address(output, p)] = S::cast_from(x * F::cast_from(retained) * scale);
        mask[address(mask, p)] = u8::cast_from(retained);
    }
}

#[ruda(launch)]
fn masked_scale<S: Float + RudaElement, F: Float + RudaElement>(
    input: &Tensor<S>, mask: &Tensor<u8>, output: &mut Tensor<S>, scale: F,
) {
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let x = F::cast_from(input[address(input, p)]);
        let multiplier = F::cast_from(mask[address(mask, p)]);
        output[address(output, p)] = S::cast_from(multiplier * x * scale);
    }
}

#[ruda(launch)]
fn rrelu<S: Float + RudaElement, F: Float + RudaElement>(
    input: &Tensor<S>, output: &mut Tensor<S>, noise: &mut Tensor<S>,
    seed_low: u32, seed_high: u32, counter: u64,
    lower: f64, range: f64, #[comptime] wide: bool,
) {
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let sample = S::cast_from(unit::<F>(u64::cast_from(p), seed_low, seed_high, counter, wide));
        let slope = S::cast_from(f64::cast_from(sample) * range + lower);
        let x = input[address(input, p)];
        let mut y = x;
        let mut multiplier = S::new(1.0);
        if F::cast_from(x) <= F::new(0.0) {
            y = S::cast_from(F::cast_from(x) * F::cast_from(slope));
            multiplier = slope;
        }
        output[address(output, p)] = y;
        noise[address(noise, p)] = multiplier;
    }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64], values: &[f64]) {
    assert!(matches!(op, 440..=442));
    assert_eq!(xs.len(), 3);
    assert_eq!(values.len(), if op == 442 { 2 } else { 1 });
    assert_eq!(ints.len(), if op == 441 { 0 } else { 4 });
    assert!(ints.iter().all(|&n| (0..=u32::MAX as i64).contains(&n)));
    let input = &xs[0];
    let output = &xs[if op == 441 { 2 } else { 1 }];
    assert!(matches!(input.dtype, 0 | 1 | 2 | 9));
    assert_eq!(input.dtype, output.dtype);
    for value in &xs[1..] { assert_eq!(value.shape, input.shape); }
    if op == 442 { assert_eq!(xs[2].dtype, input.dtype); }
    else { assert!(matches!(xs[if op == 441 { 1 } else { 2 }].dtype, 3 | 8)); }
    if input.len == 0 { return; }
    let wide = op == 442 && input.dtype == 9;
    let mut seed_low = 0u32;
    let mut seed_high = 0u32;
    let mut counter = 0u64;
    if op != 441 {
        seed_low = ints[0] as u32;
        seed_high = ints[1] as u32;
        counter = ints[2] as u64 | ((ints[3] as u64) << 32);
        let blocks = if wide { input.len.div_ceil(2) } else { input.len.div_ceil(4) };
        counter.checked_add(blocks as u64).expect("RUDA random counter exhausted");
    }
    let c = client();
    let grid = RudaCount::Static(u32::try_from(input.len.div_ceil(256)).expect("random activation launch overflow"), 1, 1);
    let block = RudaDim::new_1d(256);
    macro_rules! run { ($s:ty, $f:ty) => { unsafe {
        match op {
            440 => dropout::launch::<$s, $f, CudaRuntime>(&c, grid, block,
                input.arg(), output.arg(), xs[2].arg(), seed_low, seed_high, counter, values[0] as $f),
            441 => masked_scale::launch::<$s, $f, CudaRuntime>(&c, grid, block,
                input.arg(), xs[1].arg(), output.arg(), values[0] as $f),
            442 => rrelu::launch::<$s, $f, CudaRuntime>(&c, grid, block,
                input.arg(), output.arg(), xs[2].arg(), seed_low, seed_high, counter,
                values[0], values[1] - values[0], wide),
            _ => unreachable!(),
        }
    }}; }
    match input.dtype {
        0 => run!(f32, f32), 1 => run!(f16, f32), 2 => run!(bf16, f32),
        9 => run!(f64, f64), _ => unreachable!(),
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    finish_dispatch(&c);
}
