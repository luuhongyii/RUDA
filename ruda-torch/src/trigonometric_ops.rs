// Complex Jiterator functions adapt PyTorch's LLVM implementation.
// Part of the LLVM Project, Apache-2.0 WITH LLVM-exception.
// See the root LICENSE and https://llvm.org/LICENSE.txt.
/*
---- LLVM Exceptions to the Apache 2.0 License ----

As an exception, if, as a result of your compiling your source code, portions
of this Software are embedded into an Object form of such source code, you
may redistribute such embedded portions in such Object form without complying
with the conditions of Sections 4(a), 4(b) and 4(d) of the License.

In addition, if you combine or link compiled forms of this Software with
software that is licensed under the GPLv2 ("Combined Software") and if a
court of competent jurisdiction determines that the patent provision (Section
3), the indemnity provision (Section 9) or other Section of the License
conflicts with the conditions of the GPLv2, you may retroactively and
prospectively choose to deem waived or otherwise exclude such Section(s) of
the License, but only in their entirety and only with respect to the Combined
Software.
*/
use super::*;
use super::scientific::address;
use super::power_ops::copy_sign;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn jit_hyperbolic<F: Float>(x: F, y: F, #[comptime] cosine: bool) -> (F,F) {
    let zero=F::new(0.0); let inf=F::new(f32::INFINITY); let nan=F::new(f32::NAN);
    let mut real=zero; let mut imag=zero;
    if comptime!(cosine) {
        if x.abs()==inf && !(y.abs()<inf) {real=x.abs(); imag=nan;}
        else if x==zero && !(y.abs()<inf) {real=nan; imag=x;}
        else if x==zero && y==zero {real=F::new(1.0); imag=y;}
        else if y==zero && !(x.abs()<inf) {real=x.abs(); imag=y;}
        else {real=x.cosh()*y.cos(); imag=x.sinh()*y.sin();}
    } else {
        if x.abs()==inf && !(y.abs()<inf) {real=x; imag=nan;}
        else if x==zero && !(y.abs()<inf) {real=x; imag=nan;}
        else if y==zero && !(x.abs()<inf) {real=x; imag=y;}
        else {real=x.sinh()*y.cos(); imag=x.cosh()*y.sin();}
    }
    (real,imag)
}

#[ruda]
fn jit_atanh<F: Float>(x: F, y: F, #[comptime] code: u32) -> (F,F) {
    let zero=F::new(0.0); let one=F::new(1.0); let two=F::new(2.0); let inf=F::new(f32::INFINITY);
    let pi=F::cast_from(3.14159265358979323846264338327950288f64);
    let mut real=zero; let mut imag=zero;
    if y.abs()==inf {real=copy_sign::<F>(zero,x,code); imag=copy_sign::<F>(pi/two,y,code);}
    else if y!=y {
        if x.abs()==inf || x==zero {real=copy_sign::<F>(zero,x,code);} else {real=y;}
        imag=y;
    } else if x!=x {real=x; imag=x;}
    else if x.abs()==inf {real=copy_sign::<F>(zero,x,code); imag=copy_sign::<F>(pi/two,y,code);}
    else if x.abs()==one && y==zero {real=copy_sign::<F>(inf,x,code); imag=copy_sign::<F>(zero,y,code);}
    else {
        let (r,i)=crate::math_extra::complex_divide::<F>(one+x,y,one-x,-y,code,
            comptime!(if code==9 {52u32}else{23u32}),comptime!(if code==9 {11u32}else{8u32}));
        real=copy_sign::<F>(r.hypot(i).ln()/two,x,code); imag=copy_sign::<F>(i.atan2(r)/two,y,code);
    }
    (real,imag)
}

#[ruda(launch)]
fn real<S: Float + RudaElement, F: Float>(
    input: &Tensor<S>, output: &mut Tensor<S>, #[comptime] op: u32,
) {
    let p=(RUDA_POS_X as usize + RUDA_POS_Y as usize*RUDA_COUNT_X as usize)*256usize + UNIT_POS as usize;
    if p<output.len() {
        let x=F::cast_from(input[address(input,p)]); let mut value=F::new(0.0);
        if comptime!(op==0) {value=x.sin();}
        else if comptime!(op==1) {value=x.cos();}
        else if comptime!(op==2) {value=x.tan();}
        else if comptime!(op==3) {value=x.asin();}
        else if comptime!(op==4) {value=x.acos();}
        else if comptime!(op==5) {value=x.atan();}
        else if comptime!(op==6) {value=x.sinh();}
        else if comptime!(op==7) {value=x.cosh();}
        else if comptime!(op==8) {value=x.tanh();}
        else if comptime!(op==9) {value=x.asinh();}
        else if comptime!(op==10) {value=x.acosh();}
        else if comptime!(op==11) {value=x.atanh();}
        output[address(output,p)]=S::cast_from(value);
    }
}

#[ruda(launch)]
fn complex<S: Float + RudaElement, F: Float>(
    input_real: &Tensor<S>, input_imag: &Tensor<S>, output_real: &mut Tensor<S>, output_imag: &mut Tensor<S>,
    #[comptime] op: u32, #[comptime] code: u32,
) {
    let p=(RUDA_POS_X as usize + RUDA_POS_Y as usize*RUDA_COUNT_X as usize)*256usize + UNIT_POS as usize;
    if p<output_real.len() {
        let x=F::cast_from(input_real[address(input_real,p)]); let y=F::cast_from(input_imag[address(input_imag,p)]);
        let mut real=F::new(0.0); let mut imag=F::new(0.0);
        if comptime!(op==0) {let (r,i)=jit_hyperbolic::<F>(-y,x,false); real=i; imag=-r;}
        else if comptime!(op==1) {let (r,i)=jit_hyperbolic::<F>(-y,x,true); real=r; imag=i;}
        else if comptime!(op==2) {let (r,i)=crate::inverse_trig::tanh::<F>(-y,x,code); real=i; imag=-r;}
        else if comptime!(op==3) {let (r,i)=crate::inverse_trig::asinh::<F>(y,x,code); real=i; imag=r;}
        else if comptime!(op==4) {let (r,i)=crate::inverse_trig::acos::<F>(x,y,code); real=r; imag=i;}
        else if comptime!(op==5) {let (r,i)=jit_atanh::<F>(-y,x,code); real=i; imag=-r;}
        else if comptime!(op==6) {let (r,i)=jit_hyperbolic::<F>(x,y,false); real=r; imag=i;}
        else if comptime!(op==7) {let (r,i)=jit_hyperbolic::<F>(x,y,true); real=r; imag=i;}
        else if comptime!(op==8) {let (r,i)=crate::inverse_trig::tanh::<F>(x,y,code); real=r; imag=i;}
        else if comptime!(op==9) {let (r,i)=crate::inverse_trig::asinh::<F>(x,y,code); real=r; imag=i;}
        else if comptime!(op==10) {
            let (r,i)=crate::inverse_trig::acos::<F>(x,y,code);
            if r!=r && i!=i {real=i; imag=r;}
            else if r!=r {real=i.abs(); imag=r;}
            else if i!=i {real=i; imag=i;}
            else {real=i.abs(); imag=copy_sign::<F>(r,y,code);}
        } else if comptime!(op==11) {let (r,i)=jit_atanh::<F>(x,y,code); real=r; imag=i;}
        output_real[address(output_real,p)]=S::cast_from(real);
        output_imag[address(output_imag,p)]=S::cast_from(imag);
    }
}

pub(super) fn dispatch(xs:&[View],ints:&[i64]) {
    assert_eq!(xs.len(),2); assert_eq!(ints.len(),1); assert!((0..=11).contains(&ints[0]));
    let op=ints[0] as u32; let input=&xs[0]; let output=&xs[1]; let code=input.dtype;
    assert!(matches!(code,0|1|2|9|10|11|14)); assert_eq!(output.dtype,code); assert_eq!(input.shape,output.shape);
    if output.len==0 {return;}
    let c=client(); let blocks=output.len.div_ceil(256); let x=blocks.min(131072);
    let grid=RudaCount::Static(x as u32,blocks.div_ceil(x) as u32,1); let block=RudaDim::new_1d(256);
    if matches!(code,10|11|14) {
        let parts:Vec<View>=xs.iter().flat_map(|value|[scientific::component(value,false),scientific::component(value,true)]).collect();
        for value in &parts {scientific::indexable(value);}
        macro_rules! run {($s:ty,$f:ty)=>{unsafe {
            complex::launch::<$s,$f,CudaRuntime>(&c,grid,block,parts[0].arg(),parts[1].arg(),parts[2].arg(),parts[3].arg(),op,if code==11{9}else{0});
        }};}
        match code {10=>run!(f32,f32),11=>run!(f64,f64),14=>run!(f16,f32),_=>unreachable!()}
    } else {
        macro_rules! run {($s:ty,$f:ty)=>{unsafe {
            real::launch::<$s,$f,CudaRuntime>(&c,grid,block,input.arg(),output.arg(),op);
        }};}
        match code {0=>run!(f32,f32),1=>run!(f16,f32),2=>run!(bf16,f32),9=>run!(f64,f64),_=>unreachable!()}
    }
    LAUNCHES.fetch_add(1,Ordering::Relaxed); finish_dispatch(&c);
}
