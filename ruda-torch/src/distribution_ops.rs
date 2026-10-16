// Adapted from ATen/native/Distributions.h; PyTorch license retained in bessel.rs.
// Binomial inversion, BTRS and Stirling tail:
// Copyright 2019 The TensorFlow Authors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Gamma sampling:
// Copyright 2005 Robert Kern (robert.kern@gmail.com)
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use super::*;
use super::poisson_ops::{uniform, word};
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn stored<S: Float, F: Float>(value: F) -> F { F::cast_from(S::cast_from(value)) }

#[ruda]
fn binomial_uniform<F: Float>(seed_low: u32, seed_high: u32, stream: u64, draw: &mut u64) -> F {
    let bits = word(seed_low, seed_high, stream, draw) & 0x00ff_ffffu32;
    F::cast_from(f32::cast_from(bits) * (1.0f32 / 16777216.0f32))
}

#[ruda]
fn stirling_tail<F: Float>(k: F) -> F {
    let mut result = F::new(0.0);
    if k < F::new(10.0) {
        let values = Array::<F>::from_data(comptime!(STIRLING_TAIL.to_vec()));
        result = values[usize::cast_from(k)];
    } else {
        let kp1 = k + F::new(1.0);
        let square = f64::cast_from(kp1 * kp1);
        result = F::cast_from((1.0f64 / 12.0 - (1.0f64 / 360.0 - 1.0f64 / 1260.0 / square) / square)
            / f64::cast_from(kp1));
    }
    result
}

#[ruda]
fn binomial_lower<S: Float, F: Float>(count: F, prob: F, seed_low: u32, seed_high: u32,
    stream: u64, draw: &mut u64) -> F {
    let one = F::new(1.0);
    let np = stored::<S, F>(count * prob);
    let mut result = F::new(0.0);
    if np >= F::new(10.0) {
        let complement = stored::<S, F>(one - prob);
        let stddev = stored::<S, F>(np * complement).sqrt();
        let b = F::cast_from(1.15f64 + 2.53f64 * f64::cast_from(stddev));
        let a = F::cast_from(-0.0873f64 + 0.0248f64 * f64::cast_from(b) + 0.01f64 * f64::cast_from(prob));
        let c = F::cast_from(f64::cast_from(np) + 0.5f64);
        let vr = F::cast_from(0.92f64 - 4.2f64 / f64::cast_from(b));
        let r = stored::<S, F>(prob / complement);
        let alpha = F::cast_from((2.83f64 + 5.1f64 / f64::cast_from(b)) * f64::cast_from(stddev));
        let count1 = stored::<S, F>(count + one);
        let m = stored::<S, F>(count1 * prob).floor();
        loop {
            let u = binomial_uniform::<F>(seed_low, seed_high, stream, draw) - F::new(0.5);
            let v = binomial_uniform::<F>(seed_low, seed_high, stream, draw);
            let us = F::new(0.5) - u.abs();
            let k = stored::<S, F>(((F::new(2.0) * a / us + b) * u + c).floor());
            if !(k < F::new(0.0) || k > count) {
                if f64::cast_from(us) >= 0.07f64 && v <= vr {
                    result = k;
                    break;
                }
                let transformed = (v * alpha / (a / (us * us) + b)).ln();
                let count_k = stored::<S, F>(count - k);
                let count_k1 = stored::<S, F>(count_k + one);
                let k1 = stored::<S, F>(k + one);
                let count_m1 = count - m + one;
                let first = ((m + one) / (r * count_m1)).ln();
                let second = count1 * (count_m1 / count_k1).ln();
                let third = (r * count_k1 / k1).ln();
                let bound = F::cast_from(
                    (f64::cast_from(m) + 0.5f64) * f64::cast_from(first)
                    + f64::cast_from(second)
                    + (f64::cast_from(k) + 0.5f64) * f64::cast_from(third)
                    + f64::cast_from(stirling_tail::<F>(m)) + f64::cast_from(stirling_tail::<F>(count - m))
                    - f64::cast_from(stirling_tail::<F>(k)) - f64::cast_from(stirling_tail::<F>(count_k)));
                if transformed <= bound { result = k; break; }
            }
        }
    } else {
        let logarithm = (-prob).log1p();
        let mut sum = F::new(0.0);
        loop {
            let u = binomial_uniform::<F>(seed_low, seed_high, stream, draw);
            sum += (u.ln() / logarithm).ceil();
            if sum > count { break; }
            result = stored::<S, F>(result + one);
        }
    }
    result
}

#[ruda]
fn normal(seed_low: u32, seed_high: u32, stream: u64, draw: &mut u64,
    spare: &mut f32, has_spare: &mut bool) -> f32 {
    let mut result = *spare;
    if *has_spare { *has_spare = false; }
    else {
        let radius_unit = 1.0f32 - uniform::<f32>(seed_low, seed_high, stream, draw, false);
        let angle_unit = uniform::<f32>(seed_low, seed_high, stream, draw, false);
        let radius = (-2.0f32 * radius_unit.ln()).sqrt();
        let angle = 6.283185307179586f32 * angle_unit;
        result = radius * angle.cos();
        *spare = radius * angle.sin();
        *has_spare = true;
    }
    result
}

#[ruda]
fn gamma_sample<S: Float, F: Float>(input: F, seed_low: u32, seed_high: u32, stream: u64) -> F {
    let one = F::new(1.0);
    let mut alpha = input;
    let mut draw = 0u64;
    let mut scale = one;
    let mut result = F::new(0.0);
    if alpha != F::new(0.0) {
        if alpha < one {
            let u = F::cast_from(1.0f32 - uniform::<f32>(seed_low, seed_high, stream, &mut draw, false));
            scale *= (one - u).powf(one / alpha);
            alpha = stored::<S, F>(alpha + one);
        }
        let d = alpha - F::cast_from(1.0f32 / 3.0f32);
        let c = one / (F::new(9.0) * d).sqrt();
        let mut spare = 0.0f32;
        let mut has_spare = false;
        loop {
            let mut x = F::new(0.0);
            let mut y = F::new(0.0);
            loop {
                x = F::cast_from(normal(seed_low, seed_high, stream, &mut draw, &mut spare, &mut has_spare));
                y = one + c * x;
                if !(y <= F::new(0.0)) { break; }
            }
            let v = y * y * y;
            let u = one - F::cast_from(1.0f32 - uniform::<f32>(seed_low, seed_high, stream, &mut draw, false));
            let xx = x * x;
            if u < one - F::cast_from(0.0331f32) * xx * xx
                || u.ln() < F::new(0.5) * xx + d * (one - v + v.ln()) {
                result = stored::<S, F>(scale * d * v);
                break;
            }
        }
    }
    result
}

#[ruda(launch)]
fn binomial<S: Float + RudaElement, F: Float + RudaElement>(
    counts: &Tensor<S>, probabilities: &Tensor<S>, output: &mut Tensor<S>,
    seed_low: u32, seed_high: u32, counter: u64,
) {
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let n = F::cast_from(counts[address(counts, p)]);
        let probability = F::cast_from(probabilities[address(probabilities, p)]);
        let mut result = F::new(f32::NAN);
        if n <= F::new(0.0) || probability <= F::new(0.0) { result = F::new(0.0); }
        else if probability >= F::new(1.0) { result = n; }
        else if probability <= F::new(0.5) || probability > F::new(0.5) {
            let mut prob = probability;
            let reverse = probability > F::new(0.5);
            if reverse { prob = stored::<S, F>(F::new(1.0) - probability); }
            let mut draw = 0u64;
            result = binomial_lower::<S, F>(n, prob, seed_low, seed_high,
                counter + u64::cast_from(p), &mut draw);
            if reverse { result = stored::<S, F>(n - result); }
        }
        output[address(output, p)] = S::cast_from(result);
    }
}

#[ruda(launch)]
fn gamma<S: Float + RudaElement, F: Float + RudaElement>(
    alpha: &Tensor<S>, output: &mut Tensor<S>, seed_low: u32, seed_high: u32, counter: u64, minimum: F,
) {
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let a = F::cast_from(alpha[address(alpha, p)]);
        let mut result = gamma_sample::<S, F>(a, seed_low, seed_high, counter + u64::cast_from(p));
        if minimum > result { result = minimum; }
        output[address(output, p)] = S::cast_from(result);
    }
}

#[ruda(launch)]
fn dirichlet<S: Float + RudaElement, F: Float + RudaElement>(
    samples: &Tensor<S>, totals: &Tensor<S>, output: &mut Tensor<S>, minimum: F, maximum: F,
) {
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let value = F::cast_from(samples[address(samples, p)]);
        let total = F::cast_from(totals[address(totals, p)]);
        let mut result = stored::<S, F>(value / total);
        if minimum > result { result = minimum; }
        if maximum < result { result = maximum; }
        output[address(output, p)] = S::cast_from(result);
    }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64]) {
    assert!((425..=427).contains(&op));
    assert_eq!(xs.len(), if op == 426 { 2 } else { 3 });
    assert_eq!(ints.len(), if op == 427 { 0 } else { 4 });
    assert!(ints.iter().all(|&n| (0..=u32::MAX as i64).contains(&n)));
    let output = xs.last().unwrap();
    assert!(matches!(output.dtype, 0 | 1 | 2 | 9));
    assert!(xs.iter().all(|x| x.shape == output.shape && x.dtype == output.dtype));
    if output.len == 0 { return; }
    let mut seed_low = 0u32;
    let mut seed_high = 0u32;
    let mut counter = 0u64;
    if op != 427 {
        seed_low = ints[0] as u32;
        seed_high = ints[1] as u32;
        counter = ints[2] as u64 | ((ints[3] as u64) << 32);
        counter.checked_add(output.len as u64).expect("distribution random counter exhausted");
    }
    let (minimum, maximum): (f64, f64) = match output.dtype {
        1 => (6.103515625e-5, 0.9990234375),
        2 => (1.1754943508222875e-38, 0.9921875),
        0 => (1.1754943508222875e-38, 0.9999998807907104),
        9 => (2.2250738585072014e-308, 0.9999999999999998),
        _ => unreachable!(),
    };
    let c = client();
    let grid = RudaCount::Static(u32::try_from(output.len.div_ceil(256)).expect("distribution launch overflow"), 1, 1);
    let block = RudaDim::new_1d(256);
    macro_rules! run { ($s:ty, $f:ty) => { unsafe {
        if op == 425 {
            binomial::launch::<$s, $f, CudaRuntime>(&c, grid, block, xs[0].arg(), xs[1].arg(), output.arg(),
                seed_low, seed_high, counter);
        } else if op == 426 {
            gamma::launch::<$s, $f, CudaRuntime>(&c, grid, block, xs[0].arg(), output.arg(),
                seed_low, seed_high, counter, minimum as $f);
        } else {
            dirichlet::launch::<$s, $f, CudaRuntime>(&c, grid, block, xs[0].arg(), xs[1].arg(), output.arg(),
                minimum as $f, maximum as $f);
        }
    }}; }
    match output.dtype {
        0 => run!(f32, f32), 1 => run!(f16, f32), 2 => run!(bf16, f32), 9 => run!(f64, f64), _ => unreachable!(),
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    finish_dispatch(&c);
}

const STIRLING_TAIL: [f64; 10] = [
    0.0810614667953272, 0.0413406959554092, 0.0276779256849983, 0.02079067210376509,
    0.0166446911898211, 0.0138761288230707, 0.0118967099458917, 0.0104112652619720,
    0.00925546218271273, 0.00833056343336287,
];
