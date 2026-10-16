/*
 * Copyright 2008-2013 NVIDIA Corporation
 * Copyright 2013 Filipe RNC Maia
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*
 * Copyright (c) 2011 David Schultz <das@FreeBSD.ORG>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */
/*
 * Copyright (c) 2005 Bruce D. Evans and Steven G. Kargl
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice unmodified, this list of conditions, and the following
 *    disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

use super::*;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(fast_math = FastMath::ReducedPrecision.into())]
fn rsqrt<F: Float>(x: F) -> F { x.inverse_sqrt() }

#[ruda]
fn evaluate<F: Float>(x: F, #[comptime] op: u32) -> F {
    let mut result = F::new(0.0);
    if comptime!(op == 0) { result = crate::special::erfc::<F>(x, false); }
    else if comptime!(op == 1) { result = F::cast_from(crate::gamma::lgamma(f64::cast_from(x))); }
    else if comptime!(op == 2) { result = x.trunc(); }
    else if comptime!(op == 3) { result = x.floor(); }
    else if comptime!(op == 4) { result = x.ceil(); }
    else if comptime!(op == 5) { result = x.acos(); }
    else if comptime!(op == 6) { result = x.asin(); }
    else if comptime!(op == 7) { result = x.atan(); }
    else if comptime!(op == 8) { result = x.cosh(); }
    else if comptime!(op == 9) { result = x.tan(); }
    else if comptime!(op == 10) { result = x.sin(); }
    else if comptime!(op == 11) { result = x.sinh(); }
    else if comptime!(op == 12) { result = x.exp(); }
    else if comptime!(op == 13) { result = F::cast_from(crate::elementary::expm1(f64::cast_from(x))); }
    else if comptime!(op == 14) { result = x.tanh(); }
    else if comptime!(op == 15) { result = x.ln(); }
    else if comptime!(op == 16) { result = x.ln() * F::cast_from(0.434294481903251827651128918916605f64); }
    else if comptime!(op == 17) { result = x.ln() * F::cast_from(1.44269504088896340735992468100189f64); }
    else if comptime!(op == 18) { result = x.log1p(); }
    else if comptime!(op == 19) { result = x.cos(); }
    else if comptime!(op == 20) { result = x.sqrt(); }
    else if comptime!(op == 21) { result = x.erf(); }
    else if comptime!(op == 22) { result = F::new(1.0) / (F::new(1.0) + (-x).exp()); }
    else if comptime!(op == 23) { result = x.round(); }
    else if comptime!(op == 24) { result = x - x.trunc(); }
    else if comptime!(op == 25) { result = F::new(1.0) / x; }
    else if comptime!(op == 26) {
        if x > F::new(0.0) { result = F::new(1.0); }
        else if x < F::new(0.0) { result = -F::new(1.0); }
    }
    else if comptime!(op == 27) { result = rsqrt(x); }
    else if comptime!(op == 28) { result = -x; }
    else if comptime!(op == 29) { result = x.abs(); }
    result
}

#[ruda]
fn copy_sign<F: Float>(x: F, y: F, #[comptime] code: u32) -> F {
    let mut sign = 0x80000000u64;
    if comptime!(code == 9) { sign = 0x8000000000000000u64; }
    let a = crate::floating::bits(x, code);
    let b = crate::floating::bits(y, code);
    crate::floating::from_bits((a & !sign) | (b & sign), code)
}

#[ruda]
pub(super) fn complex_divide<F: Float>(a: F, b: F, c: F, d: F) -> (F, F) {
    let zero = F::new(0.0);
    let one = F::new(1.0);
    let mut real = zero;
    let mut imag = zero;
    if c.abs() >= d.abs() {
        if c.abs() == zero && d.abs() == zero {
            real = a / c.abs();
            imag = b / d.abs();
        } else {
            let ratio = d / c;
            let scale = one / (c + d * ratio);
            real = (a + b * ratio) * scale;
            imag = (b - a * ratio) * scale;
        }
    } else {
        let ratio = c / d;
        let scale = one / (d + c * ratio);
        real = (a * ratio + b) * scale;
        imag = (b * ratio - a) * scale;
    }
    (real, imag)
}

#[ruda]
fn scaled_complex_exp<F: Float>(x: F, y: F, #[comptime] code: u32, #[comptime] adjust: i32) -> (F, F) {
    let mut fraction = 23u32;
    let mut bias = 127i32;
    let mut k = 235i32;
    let mut reduction = F::cast_from(162.88958740f32);
    if comptime!(code == 9) {
        fraction = 52u32;
        bias = 1023i32;
        k = 1799i32;
        reduction = F::cast_from(1246.97177782734161156f64);
    }
    let value = (x - reduction).exp();
    let raw = crate::floating::bits(value, code);
    let exponent = i32::cast_from(raw >> fraction) - bias * 2i32 + k + adjust;
    let mantissa = crate::floating::from_bits::<F>(
        (raw & ((1u64 << fraction) - 1u64)) | (u64::cast_from(bias * 2i32) << fraction), code,
    );
    let first = exponent / 2i32;
    let second = exponent - first;
    let scale1 = crate::floating::from_bits::<F>(u64::cast_from(bias + first) << fraction, code);
    let scale2 = crate::floating::from_bits::<F>(u64::cast_from(bias + second) << fraction, code);
    (y.cos() * mantissa * scale1 * scale2, y.sin() * mantissa * scale1 * scale2)
}

#[ruda]
pub(super) fn complex_exp<F: Float>(x: F, y: F, #[comptime] code: u32) -> (F, F) {
    let zero = F::new(0.0);
    let inf = F::new(f32::INFINITY);
    let mut real = zero;
    let mut imag = zero;
    if y == zero { real = x.exp(); imag = y; }
    else if x == zero { real = y.cos(); imag = y.sin(); }
    else if y != y || y.abs() == inf {
        if x == -inf { real = zero; imag = zero; }
        else if x == inf { real = x; imag = y - y; }
        else { real = y - y; imag = y - y; }
    } else {
        let mut high = crate::floating::bits(x, code);
        let mut lower = 0x42b17218u64;
        let mut upper = 0x43400074u64;
        if comptime!(code == 9) { high >>= 32u32; lower = 0x40862e42u64; upper = 0x4096b8e4u64; }
        if high >= lower && high <= upper {
            let (a, b) = scaled_complex_exp(x, y, code, 0);
            real = a; imag = b;
        } else {
            let magnitude = x.exp();
            real = magnitude * y.cos(); imag = magnitude * y.sin();
        }
    }
    (real, imag)
}

#[ruda]
fn complex_hyperbolic<F: Float>(x: F, y: F, #[comptime] code: u32, #[comptime] cosh: bool) -> (F, F) {
    let zero = F::new(0.0);
    let one = F::new(1.0);
    let inf = F::new(f32::INFINITY);
    let ax = x.abs();
    let ay = y.abs();
    let mut real = F::new(f32::NAN);
    let mut imag = real;
    if ax < inf && ay < inf {
        let mut high = crate::floating::bits(ax, code);
        let mut small = 0x41100000u64;
        let mut lower = 0x42b17218u64;
        let mut upper = 0x4340b1e7u64;
        let mut huge = F::cast_from(1.70141183460469231731687303716e38f32);
        if comptime!(code == 9) {
            high >>= 32u32; small = 0x40360000u64; lower = 0x40862e42u64; upper = 0x4096bbaau64;
            huge = F::cast_from(8.98846567431157953864652595395e307f64);
        }
        if y == zero {
            if comptime!(cosh) { real = x.cosh(); imag = x * y; }
            else { real = x.sinh(); imag = y; }
        } else if high < small {
            if comptime!(cosh) { real = x.cosh() * y.cos(); imag = x.sinh() * y.sin(); }
            else { real = x.sinh() * y.cos(); imag = x.cosh() * y.sin(); }
        } else if high < lower {
            let magnitude = ax.exp() * F::new(0.5);
            if comptime!(cosh) { real = magnitude * y.cos(); imag = copy_sign(magnitude, x, code) * y.sin(); }
            else { real = copy_sign(magnitude, x, code) * y.cos(); imag = magnitude * y.sin(); }
        } else if high < upper {
            let (a, b) = scaled_complex_exp(ax, y, code, -1);
            if comptime!(cosh) { real = a; imag = b * copy_sign(one, x, code); }
            else { real = a * copy_sign(one, x, code); imag = b; }
        } else {
            let magnitude = huge * x;
            if comptime!(cosh) { real = magnitude * magnitude * y.cos(); imag = magnitude * y.sin(); }
            else { real = magnitude * y.cos(); imag = magnitude * magnitude * y.sin(); }
        }
    } else if x == zero && !(ay < inf) {
        let nan = y - y;
        if comptime!(cosh) { real = nan; imag = copy_sign(zero, x * nan, code); }
        else { real = copy_sign(zero, x * nan, code); imag = nan; }
    } else if y == zero && !(ax < inf) {
        if comptime!(cosh) {
            real = x * x;
            if ax == inf { imag = copy_sign(zero, x, code) * y; }
            else { imag = copy_sign(zero, (x + x) * y, code); }
        } else { real = x; imag = y; }
    } else if ax < inf && !(ay < inf) {
        real = y - y; imag = x * (y - y);
    } else if ax == inf {
        if !(ay < inf) { real = x * x; imag = x * (y - y); }
        else if comptime!(cosh) { real = x * x * y.cos(); imag = x * y.sin(); }
        else { real = x * y.cos(); imag = inf * y.sin(); }
    } else {
        real = x * x * (y - y); imag = (x + x) * (y - y);
    }
    (real, imag)
}

#[ruda]
fn evaluate_complex<F: Float>(x: F, y: F, #[comptime] op: u32, #[comptime] code: u32) -> (F, F) {
    let zero = F::new(0.0);
    let one = F::new(1.0);
    let mut real = zero;
    let mut imag = zero;
    if comptime!(op == 5) {
        let (a, b) = crate::scientific::inverse_hyperbolic(x, y, true);
        real = b.abs(); imag = -copy_sign(a, y, code);
        if x == zero && y != y { real = one.atan2(zero); }
    } else if comptime!(op == 6) {
        let (a, b) = crate::scientific::inverse_hyperbolic(-y, x, false);
        real = b; imag = -a;
        if y != y && x.abs() == F::new(f32::INFINITY) { imag = x; }
    } else if comptime!(op == 7) {
        let (a, b) = crate::scientific::complex_atanh(-y, x);
        real = b; imag = -a;
    } else if comptime!(op == 8) {
        let (a, b) = complex_hyperbolic(x, y, code, true);
        real = a; imag = b;
    } else if comptime!(op == 9) {
        let (a, b) = crate::scientific::complex_tanh(-y, x);
        real = b; imag = -a;
    } else if comptime!(op == 10) {
        let (a, b) = complex_hyperbolic(-y, x, code, false);
        real = b; imag = -a;
    } else if comptime!(op == 11) {
        let (a, b) = complex_hyperbolic(x, y, code, false);
        real = a; imag = b;
    } else if comptime!(op == 12) {
        let (a, b) = complex_exp(x, y, code);
        real = a; imag = b;
    } else if comptime!(op == 13) {
        let (a, b) = crate::elementary::evaluate(x, y, zero, zero, zero, 250, true);
        real = a; imag = b;
    } else if comptime!(op == 14) {
        let (a, b) = crate::scientific::complex_tanh(x, y);
        real = a; imag = b;
    } else if comptime!(op == 15 || op == 16 || op == 17) {
        let (a, b) = crate::scientific::complex_log(x, y);
        real = a; imag = b;
        if comptime!(op == 16) {
            let divisor = F::cast_from(2.30258509299404568402f64);
            real = a / divisor; imag = b / divisor;
        } else if comptime!(op == 17) {
            let (r, i) = complex_divide(a, b, F::cast_from(0.693147180559945309417232121458176f64), zero);
            real = r; imag = i;
        }
    } else if comptime!(op == 18) {
        let (a, b) = crate::elementary::complex_log1p(x, y);
        real = a; imag = b;
    } else if comptime!(op == 19) {
        let (a, b) = complex_hyperbolic(-y, x, code, true);
        real = a; imag = b;
    } else if comptime!(op == 20 || op == 27) {
        let (a, b) = crate::scientific::complex_sqrt(x, y);
        real = a; imag = b;
        if comptime!(op == 27) {
            let (r, i) = complex_divide(one, zero, a, b);
            real = r; imag = i;
        }
    } else if comptime!(op == 22) {
        let (a, b) = complex_exp(-x, -y, code);
        let (r, i) = complex_divide(one, zero, one + a, zero + b);
        real = r; imag = i;
    } else if comptime!(op == 25) {
        let (r, i) = complex_divide(one, zero, x, y);
        real = r; imag = i;
    } else if comptime!(op == 28) { real = -x; imag = -y; }
    (real, imag)
}

#[ruda(launch)]
fn complex<F: Float + RudaElement>(
    inputs: &Sequence<Tensor<F>>, outputs: &mut Sequence<Tensor<F>>, #[comptime] op: u32, #[comptime] code: u32,
) {
    let block = RUDA_POS_X as usize;
    let mut first = 0usize;
    #[unroll]
    for tensor in 0..inputs.len() {
        let input = inputs.index(tensor);
        let len = input.len() / 2usize;
        let blocks = len / 65536usize + usize::cast_from(len % 65536usize != 0);
        if block >= first && block < first + blocks {
            let output = outputs.index_mut(tensor);
            let base = (block - first) * 65536usize;
            let remaining = len - base;
            let mut offset = UNIT_POS as usize;
            while offset < remaining && offset < 65536usize {
                let p = (base + offset) * 2usize;
                let mut x = F::new(0.0);
                let mut y = F::new(0.0);
                if comptime!(op != 30) { x = input[p]; y = input[p + 1usize]; }
                let (real, imag) = evaluate_complex(x, y, op, code);
                output[p] = real;
                output[p + 1usize] = imag;
                offset += 512usize;
            }
        }
        first += blocks;
    }
}

#[ruda(launch)]
fn floating<S: Float + RudaElement, F: Float + RudaElement>(
    inputs: &Sequence<Tensor<S>>, outputs: &mut Sequence<Tensor<S>>, #[comptime] op: u32,
) {
    let block = RUDA_POS_X as usize;
    let mut first = 0usize;
    #[unroll]
    for tensor in 0..inputs.len() {
        let input = inputs.index(tensor);
        let blocks = input.len() / 65536usize + usize::cast_from(input.len() % 65536usize != 0);
        if block >= first && block < first + blocks {
            let output = outputs.index_mut(tensor);
            let base = (block - first) * 65536usize;
            let remaining = input.len() - base;
            let mut offset = UNIT_POS as usize;
            while offset < remaining && offset < 65536usize {
                let p = base + offset;
                let mut x = F::new(0.0);
                if comptime!(op != 30) { x = F::cast_from(input[p]); }
                output[p] = S::cast_from(evaluate(x, op));
                offset += 512usize;
            }
        }
        first += blocks;
    }
}

#[ruda(launch)]
fn integer<S: Numeric + RudaElement>(
    inputs: &Sequence<Tensor<S>>, outputs: &mut Sequence<Tensor<S>>, #[comptime] op: u32,
) {
    let block = RUDA_POS_X as usize;
    let mut first = 0usize;
    #[unroll]
    for tensor in 0..inputs.len() {
        let input = inputs.index(tensor);
        let blocks = input.len() / 65536usize + usize::cast_from(input.len() % 65536usize != 0);
        if block >= first && block < first + blocks {
            let output = outputs.index_mut(tensor);
            let base = (block - first) * 65536usize;
            let remaining = input.len() - base;
            let mut offset = UNIT_POS as usize;
            while offset < remaining && offset < 65536usize {
                let p = base + offset;
                let zero = S::cast_from(0u32);
                let mut value = zero;
                if comptime!(op != 30) {
                    value = input[p];
                    if comptime!(op == 28) { value = zero - value; }
                    else if value < zero { value = zero - value; }
                }
                output[p] = value;
                offset += 512usize;
            }
        }
        first += blocks;
    }
}

pub(super) fn dispatch(opcode: u32, xs: &[View], ints: &[i64]) {
    assert!(matches!(opcode, 449 | 450));
    let complex = opcode == 450;
    assert_eq!(ints.len(), 2);
    assert!((0..=30).contains(&ints[0]) && ints[1] > 0);
    let op = ints[0] as u32;
    let n = usize::try_from(ints[1]).expect("foreach tensor count overflow");
    assert_eq!(xs.len(), n.checked_mul(2).expect("foreach tensor count overflow"));
    let code = xs[0].dtype;
    assert!(code <= 9);
    if complex { assert!(matches!(code, 0 | 9) && matches!(op, 5..=20 | 22 | 25 | 27 | 28 | 30)); }
    if !matches!(code, 0 | 1 | 2 | 9) { assert!(op >= 28 && !(op == 28 && code == 3)); }
    for i in 0..n {
        assert_eq!(xs[i].dtype, code);
        assert_eq!(xs[n + i].dtype, code);
        assert_eq!(xs[i].shape, vec![xs[i].len]);
        assert_eq!(xs[n + i].shape, xs[i].shape);
        assert_eq!(xs[i].strides, vec![1usize]);
        assert_eq!(xs[n + i].strides, vec![1usize]);
        if complex { assert_eq!(xs[i].len % 2, 0); }
    }
    if xs[..n].iter().all(|x| x.len == 0) { return; }
    let c = client();
    let indices: Vec<_> = (0..n).filter(|&i| xs[i].len != 0).collect();
    for batch in indices.chunks(16) {
        let width = if complex { 2 } else { 1 };
        let blocks: usize = batch.iter().map(|&i| (xs[i].len / width).div_ceil(65536)).sum();
        let grid = RudaCount::Static(u32::try_from(blocks).expect("foreach grid overflow"), 1, 1);
        let block = RudaDim::new_1d(512);
        macro_rules! float { ($s:ty, $f:ty) => { unsafe {
            let inputs = batch.iter().map(|&i| xs[i].arg()).collect::<SequenceArg<CudaRuntime, Tensor<$s>>>();
            let outputs = batch.iter().map(|&i| xs[n + i].arg()).collect::<SequenceArg<CudaRuntime, Tensor<$s>>>();
            if complex { complex::launch::<$s, CudaRuntime>(&c, grid, block, inputs, outputs, op, code); }
            else { floating::launch::<$s, $f, CudaRuntime>(&c, grid, block, inputs, outputs, op); }
        }}; }
        macro_rules! integral { ($s:ty) => { unsafe {
            let inputs = batch.iter().map(|&i| xs[i].arg()).collect::<SequenceArg<CudaRuntime, Tensor<$s>>>();
            let outputs = batch.iter().map(|&i| xs[n + i].arg()).collect::<SequenceArg<CudaRuntime, Tensor<$s>>>();
            integer::launch::<$s, CudaRuntime>(&c, grid, block, inputs, outputs, op);
        }}; }
        match code {
            0 => float!(f32, f32), 1 => float!(f16, f32), 2 => float!(bf16, f32), 9 => float!(f64, f64),
            3 | 8 => integral!(u8), 4 => integral!(i64), 5 => integral!(i32),
            6 => integral!(i16), 7 => integral!(i8), _ => unreachable!(),
        }
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
    }
    finish_dispatch(&c);
}
