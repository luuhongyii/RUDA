// expm1: adapted from OpenLibm src/s_expm1.c.
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.
// Complex expm1/logaddexp follow PyTorch c10/util/complex_math.h and
// ATen/native/cuda/LogAddExpKernel.cu; retained PyTorch license: bessel.rs.

use super::*;

#[ruda]
pub(super) fn expm1(input:f64)->f64 {
    let hx=(u64::reinterpret(input)>>32u32)&0x7fffffffu64;
    let mut result=input;
    if input!=input {result=input+input;}
    else if input>7.09782712893383973096e2f64 {result=f64::INFINITY;}
    else if input<0.0f64 && hx>=0x4043687au64 {result=-1.0f64;}
    else if hx>=0x3c900000u64 {
        let mut x=input;let mut k=0i32;let mut c=0.0f64;
        if hx>0x3fd62e42u64 {
            let mut hi=0.0f64;let mut lo=0.0f64;
            if hx<0x3ff0a2b2u64 {
                if x>0.0f64 {hi=x-6.93147180369123816490e-1f64;lo=1.90821492927058770002e-10f64;k=1;}
                else {hi=x+6.93147180369123816490e-1f64;lo=-1.90821492927058770002e-10f64;k=-1;}
            } else {
                let mut half=0.5f64;if x<0.0f64 {half=-half;}
                k=i32::cast_from(1.44269504088896338700f64*x+half);
                let t=f64::cast_from(k);
                hi=x-t*6.93147180369123816490e-1f64;lo=t*1.90821492927058770002e-10f64;
            }
            x=hi-lo;c=(hi-x)-lo;
        }
        let hfx=0.5f64*x;let hxs=x*hfx;
        let r1=1.0f64+hxs*(-3.33333333333331316428e-2f64+hxs*(1.58730158725481460165e-3f64+hxs*(-7.93650757867487942473e-5f64+hxs*(4.00821782732936239552e-6f64+hxs*(-2.01099218183624371326e-7f64)))));
        let t=3.0f64-r1*hfx;let mut e=hxs*((r1-t)/(6.0f64-x*t));
        if k==0 {result=x-(x*e-hxs);}
        else {
            let scale=f64::reinterpret(u64::cast_from(1023+k)<<52u32);
            e=(x*(e-c)-c)-hxs;
            if k == -1 {result=0.5f64*(x-e)-0.5f64;}
            else if k==1 {
                if x < -0.25f64 {result=-2.0f64*(e-(x+0.5f64));}
                else {result=1.0f64+2.0f64*(x-e);}
            } else if k <= -2 || k>56 {
                let mut y=1.0f64-(e-x);
                if k==1024 {y=(y*2.0f64)*8.98846567431157953865e307f64;}
                else {y*=scale;}
                result=y-1.0f64;
            } else {
                let inverse=f64::reinterpret(u64::cast_from(1023-k)<<52u32);
                if k<20 {result=((1.0f64-inverse)-(e-x))*scale;}
                else {result=(1.0f64+(x-(e+inverse)))*scale;}
            }
        }
    }
    result
}

#[ruda]
pub(super) fn complex_log1p<F:Float>(x:F,y:F)->(F,F) {
    let one=F::new(1.0);let theta=y.atan2(x+one);
    let mut r=(x+one).hypot(y).ln();
    if x.hypot(y)<F::new(0.5) {
        let t=x*(F::new(2.0)+x)+y*y;
        r=t.log1p()*F::new(0.5);
        if t==F::new(0.0) {r=x;}
    }
    (r,theta)
}

#[ruda]
fn complex_exp<F:Float>(x:F,y:F)->(F,F) {
    let zero=F::new(0.0);let inf=F::new(f32::INFINITY);
    let magnitude=x.exp();let mut r=magnitude*y.cos();let mut i=magnitude*y.sin();
    if y==zero {r=magnitude;i=y;}
    else if x == -inf && (y!=y || y.abs()==inf) {r=zero;i=zero;}
    else if x==inf && (y!=y || y.abs()==inf) {r=inf;i=F::new(f32::NAN);}
    (r,i)
}

#[ruda]
fn exp_inf<F:Float>(angle:F)->(F,F) {
    let inf=F::new(f32::INFINITY);let z=F::new(0.0);
    let mut r=inf;let mut i=F::new(f32::NAN);
    if angle==angle && angle.abs()!=inf {
        let cosine=angle.cos();let sine=angle.sin();
        r=inf*cosine;i=inf*sine;
        if cosine==z {r=z;}if sine==z {i=z;}
    }
    (r,i)
}

#[ruda]
fn logaddexp_complex<F:Float>(x:F,u:F,y:F,v:F)->(F,F) {
    let z=F::new(0.0);let inf=F::new(f32::INFINITY);
    let mut r=F::new(f32::NAN);let mut i=r;
    if x==x && u==u && y==y && v==v {
        let mut lo=y;let mut li=v;let mut hi=x;let mut hi_i=u;
        if x<y {lo=x;li=u;hi=y;hi_i=v;}
        if lo.abs()==inf && lo==hi {
            if lo<z {r=lo;i=li;}
            else {
                let (a,b)=exp_inf::<F>(li);let(c,d)=exp_inf::<F>(hi_i);
                let (rr,ii)=complex_log1p::<F>(a+c-F::new(1.0),b+d);r=rr;i=ii;
            }
        } else {
            let real=lo-hi;let imag=li-hi_i;
            let mut er=z;let mut ei=z;
            if real.abs()==inf {
                if real>z {let(a,b)=exp_inf::<F>(imag);er=a;ei=b;}
            } else {let magnitude=real.exp();er=magnitude*imag.cos();ei=magnitude*imag.sin();}
            let (rr,ii)=complex_log1p::<F>(er,ei);r=rr+hi;i=ii+hi_i;
        }
    }
    (r,i)
}

#[ruda]
pub(super) fn evaluate<F:Float>(x:F,u:F,y:F,v:F,eps:F,#[comptime] op:u32,#[comptime] complex:bool)->(F,F) {
    let z=F::new(0.0);let one=F::new(1.0);let inf=F::new(f32::INFINITY);
    let ln2=F::cast_from(0.693147180559945309417232121458176f64);
    let invln2=F::cast_from(1.44269504088896340735992468100189f64);
    let mut r=z;let mut i=z;
    if comptime!(op==249) {
        r=F::new(2.0).powf(x);
        if comptime!(complex) {let(a,b)=complex_exp::<F>(ln2*x,ln2*u);r=a;i=b;}
    } else if comptime!(op==250) {
        r=F::cast_from(expm1(f64::cast_from(x)));
        if comptime!(complex) {let sine=(u/F::new(2.0)).sin();r=r*u.cos()-F::new(2.0)*sine*sine;i=x.exp()*u.sin();}
    } else if comptime!(op==251 || op==252) {
        let mut factor=invln2;if comptime!(op==252) {factor=F::cast_from(0.434294481903251827651128918916605f64);}
        r=x.ln()*factor;
        if comptime!(complex) {let(a,b)=crate::scientific::complex_log::<F>(x,u);r=a*factor;i=b*factor;}
    } else if comptime!(op==253 || op==254) {
        if x.abs()==inf && x==y {r=x;}
        else {
            let mut maximum=x;if y>x {maximum=y;}
            if comptime!(op==253) {r=maximum+(-(x-y).abs()).exp().log1p();}
            else {r=maximum+F::new(2.0).powf(-(x-y).abs()).log1p()*invln2;}
        }
        if comptime!(complex) {let(a,b)=logaddexp_complex::<F>(x,u,y,v);r=a;i=b;}
    } else if comptime!(op==255) {
        let mut value=x;
        if !(eps<z) {if x<eps {value=eps;}else if x>one-eps {value=one-eps;}}
        r=(value/(one-value)).ln();
    } else if comptime!(op==256) {
        r=x/(y*(one-y));
        if eps<z {if y<z || y>one {r=F::new(f32::NAN);}}
        else if y<eps || y>one-eps {r=z;}
    } else if comptime!(op==257) {
        r=-inf;
        if x!=x {r=x;}else if x>z {r=-x*x.ln();}else if x==z {r=z;}
    }
    (r,i)
}
