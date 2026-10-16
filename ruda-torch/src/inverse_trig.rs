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
 * Copyright (c) 2012 Stephen Montgomery-Smith <stephen@FreeBSD.ORG>
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
 *
 * Copyright (c) 2011 David Schultz
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
use super::power_ops::copy_sign;

#[ruda]
fn difference<F: Float>(a: F, b: F, radius: F) -> F {
    let zero=F::new(0.0); let two=F::new(2.0); let mut result=zero;
    if b<zero {result=(radius-b)/two;}
    else if b==zero {result=a/two;}
    else {result=a*a/(radius+b)/two;}
    result
}

#[ruda]
fn hard_work<F: Float>(x: F, y: F, sign: F, #[comptime] cosine: bool, #[comptime] code: u32) -> (F,F) {
    let zero=F::new(0.0); let one=F::new(1.0); let two=F::new(2.0); let four=F::new(4.0);
    let epsilon=F::cast_from(comptime!(if code==9 {2.220446049250313080847263336181640625e-16f64}else{1.1920928955078125e-7f64}));
    let four_sqrt_min=F::cast_from(comptime!(if code==9 {5.966672584960165394632772e-154f64}else{4.336808689942017736029811e-19f64}));
    let plus=x.hypot(y+one); let minus=x.hypot(y-one); let mut a=(plus+minus)/two;
    if a<one {a=one;}
    let mut real=zero;
    if a<F::new(10.0) {
        if y==one && x<epsilon*epsilon/F::new(128.0) {real=x.sqrt();}
        else if x>=epsilon*(y-one).abs() {
            let am1=difference(x,one+y,plus)+difference(x,one-y,minus);
            real=(am1+(am1*(a+one)).sqrt()).log1p();
        } else if y<one {real=x/((one-y)*(one+y)).sqrt();}
        else {real=((y-one)+((y-one)*(y+one)).sqrt()).log1p();}
    } else {real=(a+(a*a-one).sqrt()).ln();}
    let mut new_y=y; let mut root=zero; let mut b=zero; let mut usable=false;
    if y<four_sqrt_min {root=a*(two/epsilon); new_y=y*(two/epsilon);}
    else {
        b=y/a; usable=true;
        if b>F::cast_from(0.6417f64) {
            usable=false;
            if y==one && x<epsilon/F::new(128.0) {root=x.sqrt()*((a+y)/two).sqrt();}
            else if x>=epsilon*(y-one).abs() {
                let amy=difference(x,y+one,plus)+difference(x,y-one,minus); root=(amy*(a+y)).sqrt();
            } else if y>one {
                root=x*(four/epsilon/epsilon)*y/((y+one)*(y-one)).sqrt(); new_y=y*(four/epsilon/epsilon);
            } else {root=((one-y)*(one+y)).sqrt();}
        }
    }
    let mut angle=zero;
    if comptime!(cosine) {
        if usable {angle=copy_sign(b,sign,code).acos();}
        else {angle=root.atan2(copy_sign(new_y,sign,code));}
    } else {
        if usable {angle=b.asin();} else {angle=new_y.atan2(root);}
    }
    (real,angle)
}

#[ruda]
fn large_log<F: Float>(x: F, y: F, #[comptime] code: u32) -> (F,F) {
    let zero=F::new(0.0); let one=F::new(1.0); let two=F::new(2.0);
    let mut ax=x.abs(); let mut ay=y.abs(); if ax<ay {let temporary=ax; ax=ay; ay=temporary;}
    let maximum=F::cast_from(comptime!(if code==9 {1.79769313486231570814527423731704357e308f64}else{3.40282346638528859811704183484516925e38f64}));
    let quarter_sqrt_max=F::cast_from(comptime!(if code==9 {5.966672584960165394632772e-154f64}else{2.305843009213693952e18f64}));
    let sqrt_min=F::cast_from(comptime!(if code==9 {1.491668146240041348658193e-154f64}else{1.084202172485504434007453e-19f64}));
    let mut real=zero;
    if ax>maximum/two {
        let e=F::cast_from(2.7182818284590452f64); real=(x/e).hypot(y/e).ln()+one;
    } else if ax>quarter_sqrt_max || ay<sqrt_min {real=x.hypot(y).ln();}
    else {real=(ax*ax+ay*ay).ln()/two;}
    (real,y.atan2(x))
}

#[ruda]
pub(super) fn asinh<F: Float>(x: F, y: F, #[comptime] code: u32) -> (F,F) {
    let zero=F::new(0.0); let one=F::new(1.0); let inf=F::new(f32::INFINITY);
    let reciprocal_epsilon=F::cast_from(comptime!(if code==9 {4503599627370496.0f64}else{8388608.0f64}));
    let sqrt_6_epsilon=F::cast_from(comptime!(if code==9 {3.6500241499888571e-8f64}else{8.4572793338e-4f64}));
    let ax=x.abs(); let ay=y.abs(); let mut real=zero; let mut imag=zero;
    if x!=x || y!=y {
        if ax==inf {real=x; imag=y+y;}
        else if ay==inf {real=y; imag=x+x;}
        else if y==zero {real=x+x; imag=y;}
        else {real=(x+zero)+(y+zero); imag=real;}
    } else if ax>reciprocal_epsilon || ay>reciprocal_epsilon {
        let mut xx=x; let mut yy=y;
        if copy_sign(one,x,code)<zero {xx=-x; yy=-y;}
        let (r,i)=large_log(xx,yy,code);
        real=copy_sign(r+F::cast_from(6.9314718055994531e-1f64),x,code); imag=copy_sign(i,y,code);
    } else if x==zero && y==zero {real=x; imag=y;}
    else if ax<sqrt_6_epsilon/F::new(4.0) && ay<sqrt_6_epsilon/F::new(4.0) {real=x; imag=y;}
    else {
        let (r,i)=hard_work(ax,ay,zero,false,code); real=copy_sign(r,x,code); imag=copy_sign(i,y,code);
    }
    (real,imag)
}

#[ruda]
pub(super) fn acos<F: Float>(x: F, y: F, #[comptime] code: u32) -> (F,F) {
    let zero=F::new(0.0); let one=F::new(1.0); let inf=F::new(f32::INFINITY);
    let reciprocal_epsilon=F::cast_from(comptime!(if code==9 {4503599627370496.0f64}else{8388608.0f64}));
    let sqrt_6_epsilon=F::cast_from(comptime!(if code==9 {3.6500241499888571e-8f64}else{8.4572793338e-4f64}));
    let pio2_hi=F::cast_from(1.5707963267948966f64); let pio2_lo=F::cast_from(6.1232339957367659e-17f64);
    let ax=x.abs(); let ay=y.abs(); let mut real=zero; let mut imag=zero;
    if x!=x || y!=y {
        if ax==inf {real=y+y; imag=-inf;}
        else if ay==inf {real=x+x; imag=-y;}
        else if x==zero {real=pio2_hi+pio2_lo; imag=y+y;}
        else {real=(x+zero)+(y+zero); imag=real;}
    } else if ax>reciprocal_epsilon || ay>reciprocal_epsilon {
        let (r,i)=large_log(x,y,code); real=i.abs(); imag=r+F::cast_from(6.9314718055994531e-1f64);
        if copy_sign(one,y,code)>zero {imag=-imag;}
    } else if x==one && y==zero {real=zero; imag=-y;}
    else if ax<sqrt_6_epsilon/F::new(4.0) && ay<sqrt_6_epsilon/F::new(4.0) {real=pio2_hi-(x-pio2_lo); imag=-y;}
    else {
        let (r,i)=hard_work(ay,ax,x,true,code); real=i; imag=r;
        if copy_sign(one,y,code)>zero {imag=-imag;}
    }
    (real,imag)
}

#[ruda]
pub(super) fn tanh<F: Float>(x: F, y: F, #[comptime] code: u32) -> (F,F) {
    let zero=F::new(0.0); let one=F::new(1.0); let inf=F::new(f32::INFINITY);
    let threshold=F::new(comptime!(if code==9 {22.0f32}else{11.0f32}));
    let mut real=zero; let mut imag=zero;
    if x!=x {real=x; imag=x*y; if y==zero {imag=y;}}
    else if x.abs()==inf {
        real=copy_sign(one,x,code); let mut sign=y;
        if y.abs()!=inf {sign=y.sin()*y.cos();} imag=copy_sign(zero,sign,code);
    } else if !(y.abs()<inf) {real=y-y; imag=real;}
    else if x.abs()>=threshold {
        let exp_mx=(-x.abs()).exp(); real=copy_sign(one,x,code); imag=F::new(4.0)*y.sin()*y.cos()*exp_mx*exp_mx;
    } else {
        let t=y.tan(); let beta=one+t*t; let s=x.sinh(); let rho=(one+s*s).sqrt(); let denominator=one+beta*s*s;
        real=(beta*rho*s)/denominator; imag=t/denominator;
    }
    (real,imag)
}
