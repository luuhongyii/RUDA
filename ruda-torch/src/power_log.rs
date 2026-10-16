/* Adapted from Thrust/FreeBSD clog and clogf.
 * Copyright 2008-2021 NVIDIA Corporation; Copyright 2013 Filipe RNC Maia.
 * Licensed under the Apache License, Version 2.0, available at
 * http://www.apache.org/licenses/LICENSE-2.0 . Distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * Copyright (c) 2012 Stephen Montgomery-Smith <stephen@FreeBSD.ORG>.
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use super::*;

#[ruda]
fn trim<F: Float>(value: F, #[comptime] code: u32) -> F {
    let mut mask = 0xffff0000u64;
    if comptime!(code == 9) { mask = 0xfffffff800000000u64; }
    crate::floating::from_bits::<F>(crate::floating::bits::<F>(value, code) & mask, code)
}

#[ruda]
pub(super) fn logarithm<F: Float>(x: F, y: F, #[comptime] code: u32) -> F {
    let one = F::new(1.0); let half = F::new(0.5); let two = F::new(2.0);
    let mut ax = x.abs(); let mut ay = y.abs();
    if ax < ay { let t = ax; ax = ay; ay = t; }
    let e = F::cast_from(2.7182818284590452354f64);
    let mut high = F::cast_from(1e34f64); let mut tiny = F::cast_from(1e-19f64);
    let mut lower = F::cast_from(1e-6f64); let mut upper = F::cast_from(1e6f64);
    if comptime!(code == 9) {
        high = F::cast_from(5e307f64); tiny = F::cast_from(1e-150f64);
        lower = F::cast_from(1e-50f64); upper = F::cast_from(1e50f64);
    }
    let mut result = F::new(0.0);
    if x != x || y != y { result = (x*x + y*y).ln(); }
    else if ay > high { result = (x/e).hypot(y/e).ln() + one; }
    else if ax == one {
        if ay < tiny { result = (ay*half)*ay; } else { result = (ay*ay).log1p()*half; }
    } else if ax < lower || ay < lower || ax > upper || ay > upper { result = x.hypot(y).ln(); }
    else if ax >= one { result = ((ax-one)*(ax+one) + ay*ay).log1p()*half; }
    else if ax*ax + ay*ay <= F::cast_from(0.7f64) { result = (ax*ax + ay*ay).ln()*half; }
    else {
        let x0 = trim::<F>(ax, code); let y0 = trim::<F>(ay, code); ax -= x0; ay -= y0;
        let x1 = trim::<F>(ax, code); let y1 = trim::<F>(ay, code); let x2 = ax-x1; let y2 = ay-y1;
        let mut parts = Array::<F>::new(12usize);
        parts[0usize]=x0*x0; parts[1usize]=y0*y0; parts[2usize]=two*x0*x1; parts[3usize]=two*y0*y1;
        parts[4usize]=x1*x1; parts[5usize]=y1*y1; parts[6usize]=two*x0*x2; parts[7usize]=two*y0*y2;
        parts[8usize]=two*x1*x2; parts[9usize]=two*y1*y2; parts[10usize]=x2*x2; parts[11usize]=y2*y2;
        let mut sorted = false;
        while !sorted {
            sorted = true;
            for i in 0..11usize {
                if parts[i] < parts[i+1usize] {
                    let t = parts[i]; parts[i] = parts[i+1usize]; parts[i+1usize] = t; sorted = false;
                }
            }
        }
        let mut sum = -one;
        for i in 0..12usize { sum += parts[i]; }
        result = half*sum.log1p();
    }
    result
}
