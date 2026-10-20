// ATen random-key permutation and duplicate-key shuffle; retained PyTorch license: bessel.rs.
use super::*;
use super::random_extra::{word, word64};
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn keys<I: Int + RudaElement>(
    out: &mut Tensor<I>, seed_low: u32, seed_high: u32, counter: u64,
    mask: u64, #[comptime] wide: bool,
) {
    let p = ABSOLUTE_POS as usize;
    if p < out.len() {
        let mut modulus = 0xffff_ffffu64;
        let mut minimum = 0x8000_0000u64;
        if comptime!(wide) {
            modulus = 0xffff_ffff_ffff_ffffu64;
            minimum = 0x8000_0000_0000_0000u64;
        }
        let random = word64(u64::cast_from(p) * 2u64, seed_low, seed_high, counter);
        out[address(out, p)] = I::cast_from((random % modulus + minimum) & mask);
    }
}

#[ruda(launch)]
fn islands<I: Int + RudaElement, S: Numeric + RudaElement>(
    keys: &Tensor<I>, output: &mut Tensor<S>, seed_low: u32, seed_high: u32, counter: u64,
) {
    let p = ABSOLUTE_POS as usize;
    if p < keys.len() {
        if p + 1 < keys.len() {
            let key = keys[address(keys, p)];
            let mut leader = key == keys[address(keys, p + 1)];
            if p > 0 { leader = leader && key != keys[address(keys, p - 1)]; }
            if leader {
                let mut end = p + 1;
                while end < keys.len() {
                    if keys[address(keys, end)] != key { break; }
                    end += 1;
                }
                let mut i = end - p - 1;
                while i > 0 {
                    let random = word(u64::cast_from(p + i), seed_low, seed_high, counter);
                    let other = usize::cast_from(random % u32::cast_from(i + 1));
                    if i != other {
                        let left = address(output, p + i);
                        let right = address(output, p + other);
                        let temporary = output[left];
                        output[left] = output[right];
                        output[right] = temporary;
                    }
                    i -= 1;
                }
            }
        }
    }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64]) {
    assert!(matches!(op, 422 | 423));
    assert_eq!(xs.len(), if op == 422 { 1 } else { 2 });
    assert_eq!(ints.len(), if op == 422 { 5 } else { 4 });
    assert!(ints[..4].iter().all(|&n| (0..=u32::MAX as i64).contains(&n)));
    let key = &xs[0];
    assert_eq!(key.shape.len(), 1);
    assert!(matches!(key.dtype, 4 | 5));
    let wide = key.dtype == 4;
    let mut mask = 0u64;
    if op == 422 {
        let bits = ints[4];
        assert!(if wide { (33..=64).contains(&bits) } else { (1..=32).contains(&bits) });
        mask = if bits == 64 { u64::MAX } else { (1u64 << bits as u32) - 1 };
    } else {
        assert_eq!(xs[1].shape, key.shape);
        assert!(matches!(xs[1].dtype, 0 | 1 | 4..=9));
    }
    if key.len == 0 { return; }
    let counter = ints[2] as u64 | ((ints[3] as u64) << 32);
    let words = key.len as u64 * if op == 422 { 2 } else { 1 };
    counter.checked_add(words.div_ceil(4)).expect("randperm random counter exhausted");
    let c = client();
    let grid = RudaCount::Static(u32::try_from(key.len.div_ceil(256)).expect("randperm launch overflow"), 1, 1);
    let block = RudaDim::new_1d(256);
    macro_rules! generate { ($i:ty) => { unsafe {
        keys::launch::<$i, CudaRuntime>(&c, grid, block, key.arg(),
            ints[0] as u32, ints[1] as u32, counter, mask, wide);
    }}; }
    macro_rules! shuffle { ($i:ty, $s:ty) => { unsafe {
        islands::launch::<$i, $s, CudaRuntime>(&c, grid, block, key.arg(), xs[1].arg(),
            ints[0] as u32, ints[1] as u32, counter);
    }}; }
    macro_rules! result { ($i:ty) => {
        match xs[1].dtype {
            0 => shuffle!($i, f32), 1 => shuffle!($i, f16), 4 => shuffle!($i, i64),
            5 => shuffle!($i, i32), 6 => shuffle!($i, i16), 7 => shuffle!($i, i8),
            8 => shuffle!($i, u8), 9 => shuffle!($i, f64), _ => unreachable!(),
        }
    }; }
    if op == 422 {
        if wide { generate!(i64); } else { generate!(i32); }
    } else if wide { result!(i64); } else { result!(i32); }
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    finish_dispatch(&c);
}
