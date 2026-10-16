// PTRS is adapted from ATen/native/Distributions.cpp and NumPy distributions.c.
// Copyright 2005 Robert Kern (robert.kern@gmail.com)
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use super::*;
use super::random::philox_word_full;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
pub(super) fn word(seed_low: u32, seed_high: u32, stream: u64, draw: &mut u64) -> u32 {
    if *draw == 0xffff_ffff_ffff_ffffu64 {
        intrinsic!(|scope| { scope.register(ruda_core::ir::Branch::Unreachable); });
    }
    let sequence = *draw / 4u64 + 1u64;
    let result = philox_word_full(u32::cast_from(stream), u32::cast_from(stream >> 32u32),
        u32::cast_from(sequence), u32::cast_from(sequence >> 32u32),
        seed_low, seed_high, u32::cast_from(*draw % 4u64));
    *draw += 1u64;
    result
}

#[ruda]
pub(super) fn uniform<F: Float>(seed_low: u32, seed_high: u32, stream: u64, draw: &mut u64,
    #[comptime] wide: bool) -> F {
    let high = word(seed_low, seed_high, stream, draw);
    let mut result = F::new(0.0);
    if comptime!(wide) {
        let low = word(seed_low, seed_high, stream, draw);
        let bits = ((u64::cast_from(high) << 32u32) | u64::cast_from(low)) >> 11u32;
        result = F::cast_from(bits) * F::new(1.0 / 9007199254740992.0);
    } else {
        result = F::cast_from(high >> 8u32) * F::new(1.0 / 16777216.0);
    }
    result
}

#[ruda]
fn sample(lambda: f64, seed_low: u32, seed_high: u32, stream: u64) -> u32 {
    let mut draw = 0u64;
    let mut result = 0u32;
    if lambda < 64.0 {
        let mut product = f32::cast_from(lambda).exp();
        let mut count = 0u32;
        loop {
            count += 1;
            product *= 1.0f32 - uniform::<f32>(seed_low, seed_high, stream, &mut draw, false);
            if product <= 1.0f32 { break; }
        }
        result = count - 1;
    } else if lambda > 4000.0 {
        let radius_unit = 1.0f64 - uniform::<f64>(seed_low, seed_high, stream, &mut draw, true);
        let angle_unit = uniform::<f64>(seed_low, seed_high, stream, &mut draw, true);
        let normal = (-2.0f64 * radius_unit.ln()).sqrt()
            * (6.28318530717958647692f64 * angle_unit).cos();
        result = u32::cast_from(lambda.sqrt() * normal + lambda + 0.5);
    } else {
        let rate = f64::cast_from(f32::cast_from(lambda));
        let log_rate = rate.ln();
        let b = 0.931f64 + 2.53f64 * rate.sqrt();
        let a = -0.059f64 + 0.02483f64 * b;
        let inverse_alpha = 1.1239f64 + 1.1328f64 / (b - 3.4f64);
        let vr = 0.9277f64 - 3.6224f64 / (b - 2.0f64);
        loop {
            let u = uniform::<f64>(seed_low, seed_high, stream, &mut draw, true) - 0.5f64;
            let v = uniform::<f64>(seed_low, seed_high, stream, &mut draw, true);
            let us = 0.5f64 - u.abs();
            if us != 0.0f64 {
                let k = ((2.0f64 * a / us + b) * u + rate + 0.43f64).floor();
                if us >= 0.07f64 && v <= vr {
                    result = u32::cast_from(k);
                    break;
                }
                if !(k < 0.0f64 || (us < 0.013f64 && v > us)) {
                    if v.ln() + inverse_alpha.ln() - (a / (us * us) + b).ln()
                        <= -rate + k * log_rate - super::gamma::lgamma(k + 1.0f64) {
                        result = u32::cast_from(k);
                        break;
                    }
                }
            }
        }
    }
    result
}

#[ruda(launch)]
fn poisson<S: Float + RudaElement>(
    rates: &Tensor<S>, output: &mut Tensor<S>, seed_low: u32, seed_high: u32, counter: u64,
) {
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let lambda = f64::cast_from(rates[address(rates, p)]);
        if lambda >= 0.0f64 {
            let value = sample(lambda, seed_low, seed_high, counter + u64::cast_from(p));
            output[address(output, p)] = S::cast_from(value);
        } else {
            intrinsic!(|scope| { scope.register(ruda_core::ir::Branch::Unreachable); });
        }
    }
}

pub(super) fn dispatch(xs: &[View], ints: &[i64]) {
    assert_eq!(xs.len(), 2);
    assert_eq!(ints.len(), 4);
    assert!(ints.iter().all(|&n| (0..=u32::MAX as i64).contains(&n)));
    let rates = &xs[0];
    let output = &xs[1];
    assert_eq!(rates.shape, output.shape);
    assert_eq!(rates.dtype, output.dtype);
    assert!(matches!(rates.dtype, 0 | 1 | 2 | 9));
    if output.len == 0 { return; }
    let counter = ints[2] as u64 | ((ints[3] as u64) << 32);
    counter.checked_add(output.len as u64).expect("poisson random counter exhausted");
    let c = client();
    let grid = RudaCount::Static(u32::try_from(output.len.div_ceil(256)).expect("poisson launch overflow"), 1, 1);
    let block = RudaDim::new_1d(256);
    macro_rules! run { ($s:ty) => { unsafe {
        poisson::launch::<$s, CudaRuntime>(&c, grid, block, rates.arg(), output.arg(),
            ints[0] as u32, ints[1] as u32, counter);
    }}; }
    match rates.dtype {
        0 => run!(f32), 1 => run!(f16), 2 => run!(bf16), 9 => run!(f64), _ => unreachable!(),
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    finish_dispatch(&c);
}
