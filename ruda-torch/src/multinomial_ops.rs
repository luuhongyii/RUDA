// ATen multinomial sampling and block reduction; retained PyTorch license: bessel.rs.
use super::*;
use super::random_extra::unit;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn warp_sum<S: Float, F: Float>(input: F) -> F {
    let mut value = input;
    let mut offset = 16u32;
    while offset > 0 {
        value = F::cast_from(S::cast_from(value + plane_shuffle_down(value, offset)));
        offset /= 2;
    }
    value
}

#[ruda(launch)]
fn prepare<S: Float + RudaElement, F: Float + RudaElement>(
    input: &Tensor<S>, output: &mut Tensor<S>, seed_low: u32, seed_high: u32,
    counter: u64, #[comptime] scores: bool, #[comptime] wide: bool,
    #[comptime] threads: usize,
) {
    let lane = UNIT_POS as usize;
    let width = input.shape(1);
    let mut row = RUDA_POS_X as usize;
    let mut shared = SharedMemory::<F>::new(32usize);
    while row < input.shape(0) {
        let mut sum = F::new(0.0);
        let mut column = lane;
        while column < width {
            let index = row * width + column;
            let value = F::cast_from(input[address(input, index)]);
            if value < F::new(0.0) {
                intrinsic!(|scope| {scope.register(ruda_core::ir::Branch::Unreachable);});
            }
            if comptime!(scores) {
                if value != value || value.abs() == F::new(f32::INFINITY) {
                    intrinsic!(|scope| {scope.register(ruda_core::ir::Branch::Unreachable);});
                }
                let sample = unit::<F>(u64::cast_from(index), seed_low, seed_high, counter, wide);
                let mut epsilon = F::new(1.1920928955078125e-7);
                if comptime!(wide) { epsilon = F::new(2.220446049250313e-16); }
                let mut logarithm = sample.ln();
                if sample >= F::new(1.0) - epsilon / F::new(2.0) {
                    logarithm = -epsilon / F::new(2.0);
                }
                let exponential = F::cast_from(S::cast_from(-logarithm));
                output[address(output, index)] = S::cast_from(value / exponential);
            }
            sum = F::cast_from(S::cast_from(sum + value));
            column += threads;
        }
        sum = warp_sum::<S, F>(sum);
        sync_ruda();
        if lane % 32 == 0 { shared[lane / 32] = sum; }
        sync_ruda();
        sum = F::new(0.0);
        if lane < threads / 32 { sum = shared[lane]; }
        if lane < 32 { sum = warp_sum::<S, F>(sum); }
        if lane == 0 {
            if comptime!(scores) {
                if !(sum > F::new(0.0)) {
                    intrinsic!(|scope| {scope.register(ruda_core::ir::Branch::Unreachable);});
                }
            }
            shared[0] = sum;
        }
        sync_ruda();
        if comptime!(!scores) {
            let total = shared[0];
            let mut column = lane;
            while column < width {
                let index = row * width + column;
                let mut value = F::cast_from(input[address(input, index)]);
                if total > F::new(0.0) { value /= total; }
                output[address(output, index)] = S::cast_from(value);
                column += threads;
            }
        }
        sync_ruda();
        row += RUDA_COUNT_X as usize;
    }
}

#[ruda(launch)]
fn sample<S: Float + RudaElement>(
    probabilities: &Tensor<S>, cumulative: &Tensor<S>, output: &mut Tensor<i64>,
    seed_low: u32, seed_high: u32, counter: u64,
) {
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let row = p / output.shape(1);
        let width = probabilities.shape(1);
        let base = row * width;
        if !(cumulative[address(cumulative, base + width - 1)] > S::new(0.0)) {
            intrinsic!(|scope| {scope.register(ruda_core::ir::Branch::Unreachable);});
        }
        let target = S::cast_from(unit::<f32>(u64::cast_from(p), seed_low, seed_high, counter, false));
        let mut start = 0usize;
        let mut end = width;
        while start < end {
            let middle = start + (end - start) / 2;
            if cumulative[address(cumulative, base + middle)] < target { start = middle + 1; }
            else { end = middle; }
        }
        if start == width { start = width - 1; }
        while start > 0 && probabilities[address(probabilities, base + start)] == S::new(0.0) { start -= 1; }
        output[address(output, p)] = i64::cast_from(start);
    }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64]) {
    assert!((419..=421).contains(&op));
    assert_eq!(xs.len(), if op == 421 { 3 } else { 2 });
    assert_eq!(ints.len(), if op == 420 { 0 } else { 4 });
    assert!(ints.iter().all(|&n| (0..=u32::MAX as i64).contains(&n)));
    let source = &xs[0];
    assert_eq!(source.shape.len(), 2);
    assert!(matches!(source.dtype, 0 | 1 | 2 | 9));
    assert_eq!(xs[1].shape, source.shape);
    assert_eq!(xs[1].dtype, source.dtype);
    let rows = source.shape[0];
    let width = source.shape[1];
    assert!(width > 0 && width <= 1 << 24);
    if op == 421 {
        assert_eq!(xs[2].shape.len(), 2);
        assert_eq!(xs[2].shape[0], rows);
        assert!(xs[2].shape[1] > 0);
        assert_eq!(xs[2].dtype, 4);
    }
    if rows == 0 { return; }
    let c = client();
    let mut seed_low = 0u32;
    let mut seed_high = 0u32;
    let mut counter = 0u64;
    if op != 420 {
        seed_low = ints[0] as u32;
        seed_high = ints[1] as u32;
        counter = ints[2] as u64 | ((ints[3] as u64) << 32);
        let words = if op == 421 { xs[2].len as u64 }
            else { source.len as u64 * if source.dtype == 9 { 2 } else { 1 } };
        counter.checked_add(words.div_ceil(4)).expect("multinomial random counter exhausted");
    }
    let hardware = &c.properties().hardware;
    assert!(hardware.plane_size_min == 32 && hardware.plane_size_max == 32);
    let threads = (width.div_ceil(32) * 32).min(hardware.max_units_per_ruda.min(1024) as usize);
    assert!(threads > 0 && threads % 32 == 0);
    let blocks = rows.min(hardware.num_streaming_multiprocessors.expect("CUDA SM count required") as usize * 4);
    macro_rules! run { ($s:ty, $f:ty) => { unsafe {
        if op == 421 {
            sample::launch::<$s, CudaRuntime>(&c,
                RudaCount::Static(u32::try_from(xs[2].len.div_ceil(128)).expect("multinomial sample grid overflow"), 1, 1),
                RudaDim::new_1d(128), xs[0].arg(), xs[1].arg(), xs[2].arg(), seed_low, seed_high, counter);
        } else {
            prepare::launch::<$s, $f, CudaRuntime>(&c,
                RudaCount::Static(u32::try_from(blocks).expect("multinomial row grid overflow"), 1, 1),
                RudaDim::new_1d(threads as u32), xs[0].arg(), xs[1].arg(),
                seed_low, seed_high, counter, op == 419, source.dtype == 9, threads);
        }
    }}; }
    match source.dtype {
        0 => run!(f32, f32), 1 => run!(f16, f32), 2 => run!(bf16, f32), 9 => run!(f64, f64),
        _ => unreachable!(),
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    finish_dispatch(&c);
}
