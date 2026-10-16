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
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(fast_math = FastMath::ReducedPrecision.into())]
fn rsqrt<F: Float>(x: F) -> F { x.inverse_sqrt() }

#[ruda(launch)]
fn real<S: Float + RudaElement, F: Float>(
    base: &Tensor<S>, exponent: &Tensor<S>, output: &mut Tensor<S>, scalar: u64,
    #[comptime] mode: u32, #[comptime] special: u32, #[comptime] code: u32,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize)*256usize + UNIT_POS as usize;
    if p < output.len() {
        let a = F::cast_from(base[address(base,p)]); let mut b = crate::floating::from_bits::<F>(scalar,code);
        if comptime!(mode == 0) { b = F::cast_from(exponent[address(exponent,p)]); }
        let mut result = S::cast_from(0u32);
        if comptime!(special == 3) { result = S::cast_from(a.sqrt()); }
        else if comptime!(special == 4) { result = S::cast_from(rsqrt::<F>(a)); }
        else if comptime!(special == 5) { result = S::cast_from(F::new(1.0)/a); }
        else if comptime!(special == 6) { result = S::cast_from(a*a); }
        else if comptime!(special == 7) { result = S::cast_from(F::cast_from(S::cast_from(a*a))*a); }
        else if comptime!(special == 8) { result = S::cast_from(1.0f64/f64::cast_from(S::cast_from(a*a))); }
        else { result = S::cast_from(a.powf(b)); }
        output[address(output,p)] = result;
    }
}

#[ruda(launch)]
fn integral<S: Numeric + RudaElement>(
    base: &Tensor<S>, exponent: &Tensor<S>, output: &mut Tensor<S>, scalar: u64,
    #[comptime] mode: u32, #[comptime] special: u32, #[comptime] code: u32,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize)*256usize + UNIT_POS as usize;
    if p < output.len() {
        let a = base[address(base,p)]; let mut b = S::cast_from(scalar);
        if comptime!(mode == 0) { b = exponent[address(exponent,p)]; }
        let mut result = S::cast_from(0u32);
        if comptime!(special == 8) {
            let mut square = 0.0f64;
            if comptime!(code == 4 || code == 5) { square = f64::cast_from(a*a); }
            else { let value = i32::cast_from(a); square = f64::cast_from(value*value); }
            result = S::cast_from(1.0f64/square);
        } else {
            let mut value = 1u64; let zero = S::cast_from(0u32); let one = S::cast_from(1u32);
            if b < zero {
                value = 0u64;
                if a == one { value = 1u64; }
                else if a == zero-one { value = 1u64; if (i64::cast_from(b)&1i64) != 0i64 { value = u64::MAX; } }
            } else {
                let mut n = u64::cast_from(b); let mut factor = u64::cast_from(a);
                while n != 0u64 {
                    if (n&1u64) != 0u64 { value *= factor; }
                    n >>= 1u32; if n != 0u64 { factor *= factor; }
                }
            }
            result = S::cast_from(value);
        }
        output[address(output,p)] = result;
    }
}

#[ruda]
pub(super) fn copy_sign<F: Float>(value: F, sign: F, #[comptime] code: u32) -> F {
    let mut mask = 0x80000000u64; if comptime!(code == 9) { mask = 0x8000000000000000u64; }
    crate::floating::from_bits::<F>((crate::floating::bits::<F>(value,code)&!mask) | (crate::floating::bits::<F>(sign,code)&mask),code)
}

#[ruda]
pub(super) fn jit_exp<F: Float>(x: F, y: F, #[comptime] code: u32) -> (F,F) {
    let zero=F::new(0.0); let inf=F::new(f32::INFINITY); let mut angle=y;
    let mut real=zero; let mut imag=zero;
    if y==zero { real=x.exp(); imag=copy_sign::<F>(zero,y,code); }
    else if x==inf && !(y.abs()<inf) { real=x; imag=y; if y.abs()==inf { imag=F::new(f32::NAN); } }
    else {
        if x== -inf && !(y.abs()<inf) { angle=F::new(1.0); }
        let magnitude=x.exp(); real=magnitude*angle.cos(); imag=magnitude*angle.sin();
    }
    (real,imag)
}

#[ruda]
fn jit_sqrt<F: Float>(x: F,y: F,#[comptime] code:u32)->(F,F) {
    let zero=F::new(0.0); let inf=F::new(f32::INFINITY);
    let mut real=F::new(f32::NAN); let mut imag=F::new(f32::NAN);
    if y.abs()==inf {real=inf;imag=y;}
    else if x==inf {real=x;imag=copy_sign::<F>(zero,y,code);if y!=y {imag=y;}}
    else if x== -inf {real=zero;imag=copy_sign::<F>(x,y,code);if y!=y {real=y;}}
    else {
        let magnitude=x.hypot(y).sqrt(); let angle=y.atan2(x)/F::new(2.0);
        if magnitude==magnitude {
            if angle!=angle {real=angle;imag=angle;if magnitude==inf {real=magnitude;}}
            else if angle.abs()==inf {if magnitude==inf {real=magnitude;}}
            else {
                real=magnitude*angle.cos();imag=magnitude*angle.sin();
                if real!=real {real=zero;}if imag!=imag {imag=zero;}
            }
        }
    }
    (real,imag)
}

#[ruda(launch)]
fn complex<S:Float+RudaElement,F:Float>(
    ar:&Tensor<S>,ai:&Tensor<S>,br:&Tensor<S>,bi:&Tensor<S>,out_r:&mut Tensor<S>,out_i:&mut Tensor<S>,
    scalar_real:u64,scalar_imag:u64,#[comptime] mode:u32,#[comptime] special:u32,#[comptime] code:u32,#[comptime] half:bool,
) {
    let p=(RUDA_POS_X as usize + RUDA_POS_Y as usize*RUDA_COUNT_X as usize)*256usize + UNIT_POS as usize;
    if p<out_r.len() {
        let a=F::cast_from(ar[address(ar,p)]);let b=F::cast_from(ai[address(ai,p)]);
        let mut c=crate::floating::from_bits::<F>(scalar_real,code);let mut d=crate::floating::from_bits::<F>(scalar_imag,code);
        if comptime!(mode!=1) {c=F::cast_from(br[address(br,p)]);d=F::cast_from(bi[address(bi,p)]);}
        let zero=F::new(0.0);let one=F::new(1.0);let inf=F::new(f32::INFINITY);
        let mut real=zero;let mut imag=zero;
        if comptime!(special==3 || special==4) {
            let(r,i)=jit_sqrt::<F>(a,b,code);real=r;imag=i;
            if comptime!(special==4) {let(x,y)=crate::math_extra::complex_divide::<F>(one,zero,r,i,code,comptime!(if code==9{52u32}else{23u32}),comptime!(if code==9{11u32}else{8u32}));real=x;imag=y;}
        } else if comptime!(special==5) {
            if a!=a || b!=b || (a.abs()==inf && b.abs()==inf) {real=F::new(f32::NAN);imag=F::new(f32::NAN);}
            else if a.abs()!=inf && b.abs()!=inf {let(r,i)=crate::foreach_unary::complex_divide::<F>(one,zero,a,b);real=r;imag=i;}
        } else if comptime!(special==6) {real=a*a-b*b;imag=a*b+b*a;}
        else {
            let mut lr=zero;let mut li=zero;
            if comptime!(mode==2) {lr=crate::floating::from_bits::<F>(scalar_real,code);li=crate::floating::from_bits::<F>(scalar_imag,code);}
            else {
                li=b.atan2(a);
                if comptime!(half) {lr=a.hypot(b).ln();}else{lr=crate::power_log::logarithm::<F>(a,b,code);}
            }
            let mut pr=lr*c-li*d;let mut pi=lr*d+li*c;
            if comptime!(half) {
                if comptime!(mode==2) {let(r,i)=crate::math_extra::complex_multiply::<F>(lr,li,c,d,code);pr=r;pi=i;}
                else {let(r,i)=crate::math_extra::complex_multiply::<F>(c,d,lr,li,code);pr=r;pi=i;}
                let(r,i)=jit_exp::<F>(pr,pi,code);real=r;imag=i;
            } else {let(r,i)=crate::foreach_unary::complex_exp::<F>(pr,pi,code);real=r;imag=i;}
        }
        out_r[address(out_r,p)]=S::cast_from(real);out_i[address(out_i,p)]=S::cast_from(imag);
    }
}

pub(super) fn dispatch(xs:&[View],ints:&[i64]) {
    assert_eq!(xs.len(),3);assert_eq!(ints.len(),4);assert!((0..=2).contains(&ints[0]));
    assert!(ints[1]==0 || (3..=8).contains(&ints[1]));
    let mode=ints[0] as u32;let special=ints[1] as u32;let code=xs[2].dtype;
    assert!(code<=11 && code!=3 || code==14);
    for input in &xs[..2] {assert_eq!(input.dtype,code);assert_eq!(input.shape,xs[2].shape);}
    let is_complex=matches!(code,10|11|14);assert!(mode!=2 || is_complex);assert!(mode==1 || special==0);
    if xs[2].len==0 {return;}
    let c=client();let blocks=xs[2].len.div_ceil(256);let x=blocks.min(131072);
    let grid=RudaCount::Static(x as u32,blocks.div_ceil(x) as u32,1);let block=RudaDim::new_1d(256);
    if is_complex {
        let parts:Vec<View>=xs.iter().flat_map(|value|[scientific::component(value,false),scientific::component(value,true)]).collect();
        for value in &parts {scientific::indexable(value);}
        macro_rules! run {($s:ty,$f:ty)=>{unsafe{
            complex::launch::<$s,$f,CudaRuntime>(&c,grid,block,parts[0].arg(),parts[1].arg(),parts[2].arg(),parts[3].arg(),parts[4].arg(),parts[5].arg(),ints[2] as u64,ints[3] as u64,mode,special,if code==11{9}else{0},code==14);
        }};}
        match code {10=>run!(f32,f32),11=>run!(f64,f64),14=>run!(f16,f32),_=>unreachable!()}
    } else {
        macro_rules! floating {($s:ty,$f:ty)=>{unsafe{
            real::launch::<$s,$f,CudaRuntime>(&c,grid,block,xs[0].arg(),xs[1].arg(),xs[2].arg(),ints[2] as u64,mode,special,if code==9{9}else{0});
        }};}
        macro_rules! integer {($s:ty)=>{unsafe{
            integral::launch::<$s,CudaRuntime>(&c,grid,block,xs[0].arg(),xs[1].arg(),xs[2].arg(),ints[2] as u64,mode,special,code);
        }};}
        match code {0=>floating!(f32,f32),1=>floating!(f16,f32),2=>floating!(bf16,f32),9=>floating!(f64,f64),
            4=>integer!(i64),5=>integer!(i32),6=>integer!(i16),7=>integer!(i8),8=>integer!(u8),_=>unreachable!()}
    }
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);
}
