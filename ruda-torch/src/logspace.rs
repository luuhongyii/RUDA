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
 *
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

use super::*;
use super::scientific::{address, component, indexable};
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn real<S: Float + RudaElement, F: Float + RudaElement>(
    first: &Tensor<S>, last: &Tensor<S>, delta: &Tensor<S>, base: &Tensor<S>,
    out: &mut Tensor<S>,
) {
    let p = ABSOLUTE_POS as usize;
    if p < out.len() {
        let mut distance = p;
        if p >= out.len() / 2 { distance = out.len() - p - 1; }
        let index = F::cast_from(S::cast_from(distance));
        let product = F::cast_from(S::cast_from(F::cast_from(delta[0]) * index));
        let mut exponent = F::cast_from(S::cast_from(F::cast_from(first[0]) + product));
        if p >= out.len() / 2 {
            exponent = F::cast_from(S::cast_from(F::cast_from(last[0]) - product));
        }
        out[address(out, p)] = S::cast_from(F::cast_from(base[0]).powf(exponent));
    }
}

#[ruda(launch)]
fn integer<I: Numeric + RudaElement>(
    first: &Tensor<I>, last: &Tensor<I>, delta: &Tensor<f32>, base: &Tensor<f32>,
    out: &mut Tensor<I>,
) {
    let p = ABSOLUTE_POS as usize;
    if p < out.len() {
        let mut exponent = f32::cast_from(first[0]) + delta[0] * f32::cast_from(p);
        if p >= out.len() / 2 {
            exponent = f32::cast_from(last[0]) - delta[0] * f32::cast_from(out.len() - p - 1);
        }
        out[address(out, p)] = I::cast_from(base[0].powf(exponent));
    }
}

#[ruda]
fn complex_exp64(x: f64, y: f64) -> (f64, f64) {
    let xb = u64::reinterpret(x);
    let yb = u64::reinterpret(y);
    let magnitude = 0x7fff_ffff_ffff_ffffu64;
    if (yb & magnitude) == 0u64 { return (x.exp(), y); }
    if (xb & magnitude) == 0u64 { return (y.cos(), y.sin()); }
    if (yb & magnitude) >= 0x7ff0_0000_0000_0000u64 {
        if (xb & magnitude) != 0x7ff0_0000_0000_0000u64 { return (y - y, y - y); }
        if (xb >> 63u32) != 0u64 { return (0.0f64, 0.0f64); }
        return (x, y - y);
    }
    let high = xb >> 32u32;
    if high >= 0x4086_2e42u64 && high <= 0x4096_b8e4u64 {
        let reduced = (x - 1246.97177782734161156f64).exp();
        let bits = u64::reinterpret(reduced);
        let exponent = i32::cast_from(bits >> 52u32) - 2046i32 + 1799i32;
        let exp_x = f64::reinterpret((bits & 0x000f_ffff_ffff_ffffu64) | (2046u64 << 52u32));
        let half = exponent / 2i32;
        let scale1 = f64::reinterpret(u64::cast_from(1023i32 + half) << 52u32);
        let scale2 = f64::reinterpret(u64::cast_from(1023i32 + exponent - half) << 52u32);
        return (y.cos() * exp_x * scale1 * scale2, y.sin() * exp_x * scale1 * scale2);
    }
    let exp_x = x.exp();
    (exp_x * y.cos(), exp_x * y.sin())
}

#[ruda(launch)]
fn complex<F: Float + RudaElement>(
    first_r: &Tensor<F>, first_i: &Tensor<F>, last_r: &Tensor<F>, last_i: &Tensor<F>,
    delta_r: &Tensor<F>, delta_i: &Tensor<F>, base: &Tensor<F>,
    out_r: &mut Tensor<F>, out_i: &mut Tensor<F>, #[comptime] wide: bool,
) {
    let p = ABSOLUTE_POS as usize;
    if p < out_r.len() {
        let mut exponent_r = first_r[0] + delta_r[0] * F::cast_from(p);
        let mut exponent_i = first_i[0] + delta_i[0] * F::cast_from(p);
        if p >= out_r.len() / 2 {
            let index = F::cast_from(out_r.len() - p - 1);
            exponent_r = last_r[0] - delta_r[0] * index;
            exponent_i = last_i[0] - delta_i[0] * index;
        }
        let log_base = base[0].abs().ln();
        let angle = F::new(0.0).atan2(base[0]);
        let x = log_base * exponent_r - angle * exponent_i;
        let y = log_base * exponent_i + angle * exponent_r;
        let mut result_r = F::new(0.0);
        let mut result_i = F::new(0.0);
        if comptime!(wide) {
            let (r, i) = complex_exp64(f64::cast_from(x), f64::cast_from(y));
            result_r = F::cast_from(r);
            result_i = F::cast_from(i);
        } else {
            let exp_x = x.exp();
            result_r = exp_x * y.cos();
            result_i = exp_x * y.sin();
        }
        out_r[address(out_r, p)] = result_r;
        out_i[address(out_i, p)] = result_i;
    }
}

pub(super) fn dispatch(xs: &[View]) {
    assert_eq!(xs.len(), 5);
    let code = xs[4].dtype;
    assert!(matches!(code, 0 | 1 | 2 | 4..=11));
    let arithmetic_code = if (4..=8).contains(&code) { 0 } else { code };
    for (i, value) in xs[..4].iter().enumerate() {
        assert_eq!(value.len, 1);
        assert_eq!(value.dtype, if i < 2 { code } else { arithmetic_code });
    }
    let count = xs[4].len;
    if count == 0 { return; }
    let c = client();
    let grid = RudaCount::Static(u32::try_from(count.div_ceil(256)).expect("logspace launch overflow"), 1, 1);
    let block = RudaDim::new_1d(256);
    macro_rules! floating { ($s:ty, $f:ty) => { unsafe {
        real::launch::<$s, $f, CudaRuntime>(&c, grid, block,
            xs[0].arg(), xs[1].arg(), xs[2].arg(), xs[3].arg(), xs[4].arg());
    }}; }
    macro_rules! integral { ($i:ty) => { unsafe {
        integer::launch::<$i, CudaRuntime>(&c, grid, block,
            xs[0].arg(), xs[1].arg(), xs[2].arg(), xs[3].arg(), xs[4].arg());
    }}; }
    macro_rules! complex_run { ($f:ty, $wide:expr) => {{
        let first_r = component(&xs[0], false);
        let first_i = component(&xs[0], true);
        let last_r = component(&xs[1], false);
        let last_i = component(&xs[1], true);
        let delta_r = component(&xs[2], false);
        let delta_i = component(&xs[2], true);
        let base = component(&xs[3], false);
        let out_r = component(&xs[4], false);
        let out_i = component(&xs[4], true);
        for value in [&first_r, &first_i, &last_r, &last_i, &delta_r, &delta_i, &base, &out_r, &out_i] {
            indexable(value);
        }
        unsafe {
            complex::launch::<$f, CudaRuntime>(&c, grid, block,
                first_r.arg(), first_i.arg(), last_r.arg(), last_i.arg(),
                delta_r.arg(), delta_i.arg(), base.arg(), out_r.arg(), out_i.arg(), $wide);
        }
    }}; }
    match code {
        0 => floating!(f32, f32), 1 => floating!(f16, f32), 2 => floating!(bf16, f32),
        4 => integral!(i64), 5 => integral!(i32), 6 => integral!(i16),
        7 => integral!(i8), 8 => integral!(u8), 9 => floating!(f64, f64),
        10 => complex_run!(f32, false), 11 => complex_run!(f64, true), _ => unreachable!(),
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    finish_dispatch(&c);
}
