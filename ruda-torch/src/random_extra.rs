// ATen distribution transformations; retained PyTorch license: bessel.rs.
use super::*;
use super::random::philox_word;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
pub(super) fn word(slot: u64, seed_low: u32, seed_high: u32, counter: u64) -> u32 {
    let block = counter + slot / 4u64;
    philox_word(u32::cast_from(block), u32::cast_from(block >> 32u32),
        seed_low, seed_high, u32::cast_from(slot % 4u64))
}

#[ruda]
pub(super) fn word64(slot: u64, seed_low: u32, seed_high: u32, counter: u64) -> u64 {
    let high = word(slot, seed_low, seed_high, counter);
    let low = word(slot + 1u64, seed_low, seed_high, counter);
    (u64::cast_from(high) << 32u32) | u64::cast_from(low)
}

#[ruda]
pub(super) fn unit<F: Float>(index: u64, seed_low: u32, seed_high: u32, counter: u64, #[comptime] wide: bool) -> F {
    let mut fraction = F::new(0.0);
    if comptime!(wide) {
        let bits = word64(index * 2u64, seed_low, seed_high, counter) >> 11u32;
        fraction = F::cast_from(bits) * F::new(1.0 / 9007199254740992.0);
    } else {
        let bits = word(index, seed_low, seed_high, counter) >> 8u32;
        fraction = F::cast_from(bits) * F::new(1.0 / 16777216.0);
    }
    F::new(1.0) - fraction
}

#[ruda(launch)]
fn discrete<S: Numeric + RudaElement>(
    out: &mut Tensor<S>, seed_low: u32, seed_high: u32, counter: u64,
    base: i64, range: u64, #[comptime] wide: bool, #[comptime] full: bool,
) {
    let p = ABSOLUTE_POS as usize;
    if p < out.len() {
        let index = u64::cast_from(p);
        let mut bits = 0u64;
        if comptime!(wide) { bits = word64(index * 2u64, seed_low, seed_high, counter); }
        else { bits = u64::cast_from(word(index, seed_low, seed_high, counter)); }
        if comptime!(!full) { bits = bits % range + u64::cast_from(base); }
        out[address(out, p)] = S::cast_from(i64::cast_from(bits));
    }
}

#[ruda(launch)]
fn distribution<S: Numeric + RudaElement, F: Float + RudaElement>(
    out: &mut Tensor<S>, seed_low: u32, seed_high: u32, counter: u64,
    first: F, second: F, #[comptime] op: u32, #[comptime] wide: bool,
) {
    let p = ABSOLUTE_POS as usize;
    if p < out.len() {
        let index = u64::cast_from(p);
        let mut sample = F::new(0.0);
        if comptime!(op != 418) { sample = unit::<F>(index, seed_low, seed_high, counter, wide); }
        let mut value = F::new(0.0);
        let mut epsilon = F::new(1.1920928955078125e-7);
        if comptime!(wide) { epsilon = F::new(2.220446049250313e-16); }
        if comptime!(op == 415) {
            let mut logarithm = sample.ln();
            if sample >= F::new(1.0) - epsilon / F::new(2.0) { logarithm = -epsilon / F::new(2.0); }
            value = (-F::new(1.0) / first) * logarithm;
        } else if comptime!(op == 416) {
            value = (sample.ln() / (-first).log1p()).ceil();
        } else if comptime!(op == 417) {
            let mut probability = sample;
            if comptime!(!wide) {
                if probability > F::new(1.0) - epsilon { probability = F::new(1.0) - epsilon; }
                if probability < epsilon { probability = epsilon; }
            }
            value = first + second * (F::cast_from(3.14159265358979323846f64) * (probability - F::new(0.5))).tan();
        } else {
            let mut radius_unit = F::new(0.0);
            let mut angle = F::new(0.0);
            let pair = index / 2u64;
            if comptime!(wide) {
                radius_unit = unit::<F>(pair * 2u64, seed_low, seed_high, counter, true);
                let angle_unit = F::cast_from(word64(pair * 4u64 + 2u64, seed_low, seed_high, counter) >> 11u32)
                    * F::new(1.0 / 9007199254740992.0);
                angle = angle_unit * F::cast_from(6.28318530717958647692f64);
            } else {
                let u = word(pair * 2u64, seed_low, seed_high, counter);
                let v = word(pair * 2u64 + 1u64, seed_low, seed_high, counter);
                radius_unit = (F::cast_from(u >> 9u32) + F::new(1.0)) / F::new(8388609.0);
                angle = F::cast_from(v >> 8u32) * F::new(6.283185307179586 / 16777216.0);
            }
            let radius = (-F::new(2.0) * radius_unit.ln()).sqrt();
            let mut normal = radius * angle.cos();
            if p % 2 == 1 { normal = radius * angle.sin(); }
            value = (first + second * normal).exp();
        }
        out[address(out, p)] = S::cast_from(value);
    }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64], values: &[f64]) {
    assert_eq!(xs.len(), 1);
    assert!((414..=418).contains(&op));
    assert_eq!(ints.len(), if op == 414 { 8 } else { 4 });
    assert_eq!(values.len(), if op == 414 { 0 } else { 2 });
    assert!(ints[..4].iter().all(|&n| (0..=u32::MAX as i64).contains(&n)));
    let out = &xs[0];
    let code = out.dtype;
    let mut range = 0u64;
    let mut base = 0i64;
    let mut wide = code == 9;
    let mut full = false;
    if op == 414 {
        assert!(code <= 9);
        assert!((0..=u32::MAX as i64).contains(&ints[5]) && (0..=u32::MAX as i64).contains(&ints[6]));
        assert!(matches!(ints[7], 0 | 1));
        base = ints[4];
        range = ints[5] as u64 | ((ints[6] as u64) << 32);
        wide = ints[7] != 0;
        full = range == 0;
        assert!(!full || (wide && matches!(code, 0 | 2 | 4 | 9)));
        assert!(wide || range <= (1u64 << 32));
    } else if op == 416 {
        assert!(matches!(code, 0 | 1 | 2 | 4..=9));
        assert!(values[0] > 0.0 && values[0] < 1.0);
    } else {
        assert!(matches!(code, 0 | 1 | 2 | 9));
        assert!(if op == 415 { values[0] > 0.0 } else { values[1] > 0.0 });
    }
    if out.len == 0 { return; }
    let counter = ints[2] as u64 | ((ints[3] as u64) << 32);
    let blocks = if wide { out.len.div_ceil(2) } else { out.len.div_ceil(4) };
    counter.checked_add(blocks as u64).expect("RUDA random counter exhausted");
    let c = client();
    let grid = RudaCount::Static(u32::try_from(out.len.div_ceil(256)).expect("random launch overflow"), 1, 1);
    let block = RudaDim::new_1d(256);
    macro_rules! run { ($s:ty, $f:ty) => { unsafe {
        if op == 414 {
            discrete::launch::<$s, CudaRuntime>(&c, grid, block, out.arg(),
                ints[0] as u32, ints[1] as u32, counter, base, range, wide, full);
        } else {
            distribution::launch::<$s, $f, CudaRuntime>(&c, grid, block, out.arg(),
                ints[0] as u32, ints[1] as u32, counter, values[0] as $f, values[1] as $f, op, wide);
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
