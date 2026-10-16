// nextafter follows PyTorch c10/util/BFloat16-math.h and musl nextafter.
// Copyright (c) 2005-2020 Rich Felker, et al.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions: The above copyright notice and this
// permission notice shall be included in all copies or substantial portions of
// the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
// EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
// OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE. Retained PyTorch license: bessel.rs.

use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
pub(super) fn bits<F:Float>(x:F,#[comptime] code:u32)->u64 {
    let mut value=0u64;
    if comptime!(code==9) {value=u64::reinterpret(x);}
    else if comptime!(code==0) {value=u64::cast_from(u32::reinterpret(x));}
    else {value=u64::cast_from(u16::reinterpret(x));}
    value
}

#[ruda]
pub(super) fn from_bits<F:Float>(x:u64,#[comptime] code:u32)->F {
    let mut value=F::new(0.0);
    if comptime!(code==9) {value=F::reinterpret(x);}
    else if comptime!(code==0) {value=F::reinterpret(u32::cast_from(x));}
    else {value=F::reinterpret(u16::cast_from(x));}
    value
}

#[ruda]
pub(super) fn normalize(x:u64,#[comptime] fraction:u32,#[comptime] exponent_bits:u32)->(u64,i32) {
    let leading=1u64<<fraction;
    let mask=(1u64<<exponent_bits)-1u64;
    let mut exponent=i32::cast_from((x>>fraction)&mask);
    let mut significand=x&(leading-1u64);
    if exponent!=0 {significand|=leading;}
    else {
        exponent=1;
        while significand<leading {significand<<=1u32;exponent-=1;}
    }
    (significand,exponent)
}

#[ruda(launch)]
fn split<F:Float+RudaElement>(input:&Tensor<F>,mantissa:&mut Tensor<F>,exponents:&mut Tensor<i32>,#[comptime] code:u32,#[comptime] fraction:u32,#[comptime] exponent_bits:u32) {
    let p=ABSOLUTE_POS as usize;
    if p<mantissa.len() {
        let x=bits::<F>(input[address(input,p)],code);
        let sign=1u64<<(fraction+exponent_bits);let infinity=((1u64<<exponent_bits)-1u64)<<fraction;
        let magnitude=x&(sign-1u64);let mut result=x;let mut exponent=0i32;
        if magnitude!=0 && magnitude<infinity {
            let (significand,e)=normalize(x,fraction,exponent_bits);
            let bias=comptime!((1i32<<(exponent_bits-1))-1);
            exponent=e-bias+1;
            result=(x&sign)|(significand&((1u64<<fraction)-1u64))|(u64::cast_from(bias-1)<<fraction);
        }
        mantissa[address(mantissa,p)]=from_bits::<F>(result,code);
        exponents[address(exponents,p)]=exponent;
    }
}

#[ruda(launch)]
fn scale_binary<F:Float+RudaElement>(input:&Tensor<F>,exponents:&Tensor<i32>,out:&mut Tensor<F>,#[comptime] code:u32,#[comptime] fraction:u32,#[comptime] exponent_bits:u32) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let x=bits::<F>(input[address(input,p)],code);
        let sign=1u64<<(fraction+exponent_bits);let limit=(1u64<<exponent_bits)-1u64;
        let infinity=limit<<fraction;let magnitude=x&(sign-1u64);let mut result=x;
        if magnitude!=0 && magnitude<infinity {
            let (significand,e)=normalize(x,fraction,exponent_bits);
            let exponent=i64::cast_from(e)+i64::cast_from(exponents[address(exponents,p)]);
            result=x&sign;
            if exponent>=i64::cast_from(limit) {result|=infinity;}
            else if exponent>0 {result|=(u64::cast_from(exponent)<<fraction)|(significand&((1u64<<fraction)-1u64));}
            else {
                let shift=1i64-exponent;
                if shift<=i64::cast_from(fraction+1) {
                    let n=u64::cast_from(shift);let mut rounded=significand>>n;
                    let residual=significand&((1u64<<n)-1u64);let halfway=1u64<<(n-1);
                    if residual>halfway || (residual==halfway && (rounded&1u64)!=0) {rounded+=1u64;}
                    result|=rounded;
                }
            }
        }
        out[address(out,p)]=from_bits::<F>(result,code);
    }
}

#[ruda(launch)]
fn signed_neighbor<F:Float+RudaElement>(a:&Tensor<F>,b:&Tensor<F>,out:&mut Tensor<F>,#[comptime] code:u32,#[comptime] fraction:u32,#[comptime] exponent_bits:u32,#[comptime] next:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let x=bits::<F>(a[address(a,p)],code);let y=bits::<F>(b[address(b,p)],code);
        let sign=1u64<<(fraction+exponent_bits);let ax=x&(sign-1u64);let ay=y&(sign-1u64);
        let mut result=ax|(y&sign);
        if comptime!(next) {
            let infinity=((1u64<<exponent_bits)-1u64)<<fraction;result=x;
            if ax>infinity || ay>infinity {result=infinity|(1u64<<(fraction-1));}
            else if x!=y {
                if ax==0 {result=y;if ay!=0 {result=(y&sign)|1u64;}}
                else if ax>ay || ((x^y)&sign)!=0 {result=x-1u64;}
                else {result=x+1u64;}
            }
        }
        out[address(out,p)]=from_bits::<F>(result,code);
    }
}

#[ruda(launch)]
fn fraction_round<S:Float+RudaElement,F:Float+RudaElement>(input:&Tensor<S>,out:&mut Tensor<S>,factor:F,#[comptime] frac:bool,#[comptime] negative:bool,#[comptime] zero_decimals:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let x=F::cast_from(input[address(input,p)]);let scale=F::cast_from(S::cast_from(factor));let mut result=x;
        if comptime!(frac) {result=x-x.trunc();}
        else if comptime!(zero_decimals) {result=x.round();}
        else if comptime!(negative) {let quotient=F::cast_from(S::cast_from(x/scale));result=quotient.round()*scale;}
        else {let product=F::cast_from(S::cast_from(x*scale));result=product.round()/scale;}
        out[address(out,p)]=S::cast_from(result);
    }
}

#[ruda(launch)]
fn integral_power<I:Int+RudaElement>(a:&Tensor<I>,b:&Tensor<I>,out:&mut Tensor<I>) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let x=i64::cast_from(a[address(a,p)]);let exponent=i64::cast_from(b[address(b,p)]);
        let mut result=1u64;
        if exponent<0 {
            result=0u64;
            if x==1i64 {result=1u64;}
            else if x== -1i64 {result=1u64;if (exponent&1i64)!=0i64 {result=u64::cast_from(-1i64);}}
        } else {
            let mut n=u64::cast_from(exponent);let mut base=u64::cast_from(x);
            while n!=0u64 {
                if (n&1u64)!=0u64 {result*=base;}
                n>>=1u64;
                if n!=0u64 {base*=base;}
            }
        }
        out[address(out,p)]=I::cast_from(result);
    }
}

pub(super) fn dispatch(op:u32,xs:&[View],ints:&[i64]) {
    if op==264 {
        let a=&xs[0];let b=&xs[1];let out=&xs[2];
        assert_eq!(a.dtype,b.dtype);assert_eq!(a.dtype,out.dtype);
        assert_eq!(a.shape,b.shape);assert_eq!(a.shape,out.shape);
        if out.len==0 {return;}
        let work=RudaCount::Static(u32::try_from(out.len.div_ceil(128)).expect("integer power grid overflow"),1,1);
        macro_rules! run {($i:ty)=>{unsafe {integral_power::launch::<$i,CudaRuntime>(&client(),work,RudaDim::new_1d(128),a.arg(),b.arg(),out.arg());}}}
        match a.dtype {4=>run!(i64),5=>run!(i32),6=>run!(i16),7=>run!(i8),8=>run!(u8),_=>panic!("integer power requires integral non-bool storage")}
        LAUNCHES.fetch_add(1,Ordering::Relaxed);return;
    }
    let a=&xs[0];let out=&xs[if op>=262 {1}else{2}];
    assert!(matches!(a.dtype,0|1|2|9));assert_eq!(a.shape,out.shape);
    if op==258 {assert_eq!(a.dtype,xs[1].dtype);assert_eq!(out.dtype,5);assert_eq!(a.shape,xs[1].shape);}
    else {assert_eq!(a.dtype,out.dtype);}
    if (259..=261).contains(&op) {assert_eq!(a.shape,xs[1].shape);assert_eq!(xs[1].dtype,if op==259 {5}else{a.dtype});}
    if out.len==0 {return;}
    let work=RudaCount::Static(u32::try_from(out.len.div_ceil(128)).expect("floating grid overflow"),1,1);
    let dimensions=RudaDim::new_1d(128);
    let decimals=if op==263 {assert_eq!(ints.len(),1);ints[0]}else{0};
    let factor=10f64.powf(decimals.unsigned_abs() as f64);
    macro_rules! run {($s:ty,$f:ty,$fraction:expr,$exponent:expr)=>{unsafe {
        match op {
            258=>split::launch::<$s,CudaRuntime>(&client(),work,dimensions,a.arg(),xs[1].arg(),out.arg(),a.dtype,$fraction,$exponent),
            259=>scale_binary::launch::<$s,CudaRuntime>(&client(),work,dimensions,a.arg(),xs[1].arg(),out.arg(),a.dtype,$fraction,$exponent),
            260|261=>signed_neighbor::launch::<$s,CudaRuntime>(&client(),work,dimensions,a.arg(),xs[1].arg(),out.arg(),a.dtype,$fraction,$exponent,op==261),
            262|263=>fraction_round::launch::<$s,$f,CudaRuntime>(&client(),work,dimensions,a.arg(),out.arg(),factor as $f,op==262,decimals<0,decimals==0),
            _=>unreachable!(),
        }
    }}}
    match a.dtype {0=>run!(f32,f32,23,8),1=>run!(f16,f32,10,5),2=>run!(bf16,f32,7,8),9=>run!(f64,f64,52,11),_=>unreachable!()}
    LAUNCHES.fetch_add(1,Ordering::Relaxed);
}
