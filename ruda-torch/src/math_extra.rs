// ATen scalar/vector operations; PyTorch notice retained in bessel.rs.
// Complex arithmetic adapts PyTorch's LLVM Jiterator implementation.
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
use super::floating::{bits,from_bits,normalize};
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn heaviside<N:Numeric+RudaElement>(a:&Tensor<N>,b:&Tensor<N>,out:&mut Tensor<N>) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let x=a[address(a,p)];let mut value=N::cast_from(x>N::new(0));
        if x==N::new(0) {value=b[address(b,p)];}
        out[address(out,p)]=value;
    }
}

#[ruda]
fn unsigned_abs(x:i64)->u64 {
    let mut value=u64::cast_from(x);if x<0i64 {value=0u64-value;}value
}

#[ruda(launch)]
fn gcd_lcm<I:Numeric+RudaElement>(a:&Tensor<I>,b:&Tensor<I>,out:&mut Tensor<I>,#[comptime] lcm:bool,#[comptime] code:u32) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let x=unsigned_abs(i64::cast_from(a[address(a,p)]));let y=unsigned_abs(i64::cast_from(b[address(b,p)]));
        let mut u=x;let mut v=y;
        while u!=0u64 {let previous=u;u=v%u;v=previous;}
        let mut value=v;
        if comptime!(lcm) {
            if v!=0u64 {
                value=x/v*y;
                if comptime!(code==4 || code==5) {value=unsigned_abs(i64::cast_from(I::cast_from(value)));}
            }
        }
        out[address(out,p)]=I::cast_from(value);
    }
}

#[ruda]
fn copy_sign<F:Float>(x:F,y:F,#[comptime] code:u32)->F {
    let mut sign=1u64<<31u32;if comptime!(code==9) {sign=1u64<<63u32;}
    from_bits::<F>((bits(x,code)&(sign-1u64))|(bits(y,code)&sign),code)
}

#[ruda]
fn scale<F:Float>(x:F,power:i32,#[comptime] code:u32,#[comptime] fraction:u32,#[comptime] exponent_bits:u32)->F {
    let raw=bits(x,code);let sign=1u64<<(fraction+exponent_bits);let limit=(1u64<<exponent_bits)-1u64;
    let infinity=limit<<fraction;let magnitude=raw&(sign-1u64);let mut result=raw;
    if magnitude!=0u64 && magnitude<infinity {
        let (significand,e)=normalize(raw,fraction,exponent_bits);let exponent=i64::cast_from(e)+i64::cast_from(power);result=raw&sign;
        if exponent>=i64::cast_from(limit) {result|=infinity;}
        else if exponent>0i64 {result|=(u64::cast_from(exponent)<<fraction)|(significand&((1u64<<fraction)-1u64));}
        else {
            let shift=1i64-exponent;
            if shift<=i64::cast_from(fraction+1u32) {
                let n=u64::cast_from(shift);let mut rounded=significand>>n;
                let residual=significand&((1u64<<n)-1u64);let halfway=1u64<<(n-1u64);
                if residual>halfway || (residual==halfway && (rounded&1u64)!=0u64) {rounded+=1u64;}
                result|=rounded;
            }
        }
    }
    from_bits::<F>(result,code)
}

#[ruda]
pub(super) fn complex_divide<F:Float>(a:F,b:F,c0:F,d0:F,#[comptime] code:u32,#[comptime] fraction:u32,#[comptime] exponent_bits:u32)->(F,F) {
    let zero=F::new(0.0);let one=F::new(1.0);let inf=F::new(f32::INFINITY);let mut c=c0;let mut d=d0;
    let mut maximum=c.abs();if maximum!=maximum || d.abs()>maximum {maximum=d.abs();}
    let mut exponent=0i32;
    if maximum>zero && maximum<inf {
        let (_mantissa,e)=normalize(bits(maximum,code),fraction,exponent_bits);
        exponent=e-comptime!((1i32<<(exponent_bits-1u32))-1i32);
        c=scale(c,-exponent,code,fraction,exponent_bits);d=scale(d,-exponent,code,fraction,exponent_bits);
    }
    let denominator=c*c+d*d;
    let mut real=scale((a*c+b*d)/denominator,-exponent,code,fraction,exponent_bits);
    let mut imag=scale((b*c-a*d)/denominator,-exponent,code,fraction,exponent_bits);
    if real!=real && imag!=imag {
        if denominator==zero && (a==a || b==b) {let infinity=copy_sign(inf,c,code);real=infinity*a;imag=infinity*b;}
        else if (a.abs()==inf || b.abs()==inf) && c.abs()<inf && d.abs()<inf {
            let mut aa=zero;let mut bb=zero;if a.abs()==inf {aa=one;}if b.abs()==inf {bb=one;}
            aa=copy_sign(aa,a,code);bb=copy_sign(bb,b,code);real=inf*(aa*c+bb*d);imag=inf*(bb*c-aa*d);
        } else if maximum==inf && a.abs()<inf && b.abs()<inf {
            let mut cc=zero;let mut dd=zero;if c.abs()==inf {cc=one;}if d.abs()==inf {dd=one;}
            cc=copy_sign(cc,c,code);dd=copy_sign(dd,d,code);real=zero*(a*cc+b*dd);imag=zero*(b*cc-a*dd);
        }
    }
    (real,imag)
}

#[ruda]
pub(super) fn complex_multiply<F:Float>(a0:F,b0:F,c0:F,d0:F,#[comptime] code:u32)->(F,F) {
    let zero=F::new(0.0);let inf=F::new(f32::INFINITY);
    let mut a=a0;let mut b=b0;let mut c=c0;let mut d=d0;
    let ac=a*c;let bd=b*d;let ad=a*d;let bc=b*c;let mut real=ac-bd;let mut imag=ad+bc;
    if real!=real && imag!=imag {
        let mut recalc=false;
        if a.abs()==inf || b.abs()==inf {
            a=copy_sign(F::cast_from(a.abs()==inf),a,code);b=copy_sign(F::cast_from(b.abs()==inf),b,code);
            if c!=c {c=copy_sign(zero,c,code);}if d!=d {d=copy_sign(zero,d,code);}recalc=true;
        }
        if c.abs()==inf || d.abs()==inf {
            c=copy_sign(F::cast_from(c.abs()==inf),c,code);d=copy_sign(F::cast_from(d.abs()==inf),d,code);
            if a!=a {a=copy_sign(zero,a,code);}if b!=b {b=copy_sign(zero,b,code);}recalc=true;
        }
        if !recalc && (ac.abs()==inf || bd.abs()==inf || ad.abs()==inf || bc.abs()==inf) {
            if a!=a {a=copy_sign(zero,a,code);}if b!=b {b=copy_sign(zero,b,code);}
            if c!=c {c=copy_sign(zero,c,code);}if d!=d {d=copy_sign(zero,d,code);}recalc=true;
        }
        if recalc {real=inf*(a*c-b*d);imag=inf*(a*d+b*c);}
    }
    (real,imag)
}

#[ruda]
fn complex_sinc<F:Float>(x:F,y:F,#[comptime] code:u32,#[comptime] fraction:u32,#[comptime] exponent_bits:u32)->(F,F) {
    let zero=F::new(0.0);let one=F::new(1.0);let inf=F::new(f32::INFINITY);let pi=F::cast_from(3.14159265358979323846f64);
    let mut real=one;let mut imag=zero;
    if x!=zero || y!=zero {
        let mut pr=pi*x-zero*y;let mut im=pi*y+zero*x;
        if pr!=pr && im!=im && (x.abs()==inf || y.abs()==inf) {
            let mut a=zero;let mut b=zero;if x.abs()==inf {a=one;}if y.abs()==inf {b=one;}
            a=copy_sign(a,x,code);b=copy_sign(b,y,code);pr=inf*(pi*a-zero*b);im=inf*(pi*b+zero*a);
        }
        let hreal= -im;let himag=pr;let mut sr=F::new(f32::NAN);let mut si=F::new(f32::NAN);
        if (hreal.abs()==inf || hreal==zero) && !(himag.abs()<inf) {sr=hreal;}
        else if himag==zero && !(hreal.abs()<inf) {sr=hreal;si=himag;}
        else {sr=hreal.sinh()*himag.cos();si=hreal.cosh()*himag.sin();}
        let (r,i)=complex_divide::<F>(si,-sr,pr,im,code,fraction,exponent_bits);real=r;imag=i;
    }
    (real,imag)
}

#[ruda(launch)]
fn sinc<S:Float+RudaElement,F:Float+RudaElement>(a:&Tensor<S>,imaginary:&Tensor<S>,out:&mut Tensor<S>,out_i:&mut Tensor<S>,#[comptime] complex:bool,#[comptime] code:u32,#[comptime] fraction:u32,#[comptime] exponent_bits:u32) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let x=F::cast_from(a[address(a,p)]);let mut real=F::new(1.0);
        if comptime!(complex) {
            let (r,i)=complex_sinc::<F>(x,F::cast_from(imaginary[address(imaginary,p)]),code,fraction,exponent_bits);real=r;out_i[address(out_i,p)]=S::cast_from(i);
        } else if x!=F::new(0.0) {let product=F::cast_from(3.14159265358979323846f64)*x;real=product.sin()/product;}
        out[address(out,p)]=S::cast_from(real);
    }
}

#[ruda]
fn vector_base<N:Numeric>(a:&Tensor<N>,vector:usize,axis:usize)->usize {
    let mut remaining=vector;let mut result=0usize;let mut d=a.rank();
    while d>0usize {d-=1usize;if d!=axis {result+=(remaining%a.shape(d))*a.stride(d);remaining/=a.shape(d);}}
    result
}

#[ruda(launch)]
fn cross_float<S:Float+RudaElement,F:Float+RudaElement>(ar:&Tensor<S>,ai:&Tensor<S>,br:&Tensor<S>,bi:&Tensor<S>,out:&mut Tensor<S>,out_i:&mut Tensor<S>,axis:usize,#[comptime] complex:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len()/3usize {
        let abase=vector_base(ar,p,axis);let bbase=vector_base(br,p,axis);let obase=vector_base(out,p,axis);
        let mut aibase=0usize;let mut bibase=0usize;let mut oibase=0usize;
        if comptime!(complex) {aibase=vector_base(ai,p,axis);bibase=vector_base(bi,p,axis);oibase=vector_base(out_i,p,axis);}
        for c in 0usize..3usize {
            let j=(c+1usize)%3usize;let k=(c+2usize)%3usize;
            let a=F::cast_from(ar[abase+j*ar.stride(axis)]);let b=F::cast_from(br[bbase+k*br.stride(axis)]);
            let d=F::cast_from(ar[abase+k*ar.stride(axis)]);let e=F::cast_from(br[bbase+j*br.stride(axis)]);
            let mut left=S::cast_from(a*b);let mut right=S::cast_from(d*e);
            if comptime!(complex) {
                let u=F::cast_from(ai[aibase+j*ai.stride(axis)]);let v=F::cast_from(bi[bibase+k*bi.stride(axis)]);
                let w=F::cast_from(ai[aibase+k*ai.stride(axis)]);let z=F::cast_from(bi[bibase+j*bi.stride(axis)]);
                left=S::cast_from(a*b-u*v);right=S::cast_from(d*e-w*z);
                out_i[oibase+c*out_i.stride(axis)]=S::cast_from((a*v+u*b)-(d*z+w*e));
            }
            out[obase+c*out.stride(axis)]=S::cast_from(F::cast_from(left)-F::cast_from(right));
        }
    }
}

#[ruda(launch)]
fn cross_integer<I:Numeric+RudaElement>(a:&Tensor<I>,b:&Tensor<I>,out:&mut Tensor<I>,axis:usize) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len()/3usize {
        let abase=vector_base(a,p,axis);let bbase=vector_base(b,p,axis);let obase=vector_base(out,p,axis);
        for c in 0usize..3usize {
            let j=(c+1usize)%3usize;let k=(c+2usize)%3usize;
            let left=u64::cast_from(i64::cast_from(a[abase+j*a.stride(axis)]))*u64::cast_from(i64::cast_from(b[bbase+k*b.stride(axis)]));
            let right=u64::cast_from(i64::cast_from(a[abase+k*a.stride(axis)]))*u64::cast_from(i64::cast_from(b[bbase+j*b.stride(axis)]));
            out[obase+c*out.stride(axis)]=I::cast_from(left-right);
        }
    }
}

pub(super) fn dispatch(op:u32,xs:&[View],ints:&[i64]) {
    assert!((384..=388).contains(&op));let code=xs[0].dtype;
    let output=if op==388 {4}else{2};assert_eq!(xs.len(),if op==388 {6}else if op==387 {4}else{3});
    assert_eq!(ints.len(),if op==388 {2}else if op==387 {1}else{0});
    for x in &xs[1..] {assert_eq!(x.shape,xs[0].shape);assert_eq!(x.dtype,code);}
    let complex=if op>=387 {assert!(matches!(ints[ints.len()-1],0|1));ints[ints.len()-1]!=0}else{false};
    if complex {assert!(matches!(code,0|9));}
    let axis=if op==388 {usize::try_from(ints[0]).expect("negative cross axis")}else{0};
    if op==388 {assert_eq!(xs[0].shape[axis],3);assert_ne!(code,3);}
    if op==387 {assert!(matches!(code,0|1|2|9));}
    if op==385 || op==386 {assert!((4..=8).contains(&code));}
    let count=xs[output].len/if op==388 {3}else{1};if count==0 {return;}
    let c=client();let grid=RudaCount::Static(u32::try_from(count.div_ceil(256)).expect("math launch overflow"),1,1);let block=RudaDim::new_1d(256);
    macro_rules! integer {($i:ty)=>{unsafe {
        if op==384 {heaviside::launch::<$i,CudaRuntime>(&c,grid,block,xs[0].arg(),xs[1].arg(),xs[2].arg());}
        else if op==388 {cross_integer::launch::<$i,CudaRuntime>(&c,grid,block,xs[0].arg(),xs[2].arg(),xs[4].arg(),axis);}
        else {gcd_lcm::launch::<$i,CudaRuntime>(&c,grid,block,xs[0].arg(),xs[1].arg(),xs[2].arg(),op==386,code);}
    }};}
    macro_rules! floating {($s:ty,$f:ty,$code:expr,$fraction:expr,$bits:expr)=>{unsafe {
        if op==384 {heaviside::launch::<$s,CudaRuntime>(&c,grid,block,xs[0].arg(),xs[1].arg(),xs[2].arg());}
        else if op==387 {sinc::launch::<$s,$f,CudaRuntime>(&c,grid,block,xs[0].arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),complex,$code,$fraction,$bits);}
        else {cross_float::launch::<$s,$f,CudaRuntime>(&c,grid,block,xs[0].arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),xs[4].arg(),xs[5].arg(),axis,complex);}
    }};}
    match code {0=>floating!(f32,f32,0,23,8),1=>floating!(f16,f32,0,23,8),2=>floating!(bf16,f32,0,23,8),3|8=>integer!(u8),4=>integer!(i64),5=>integer!(i32),6=>integer!(i16),7=>integer!(i8),9=>floating!(f64,f64,9,52,11),_=>panic!("unsupported math dtype")}
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);
}
