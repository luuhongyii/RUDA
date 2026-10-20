// The erf/erfc rational coefficients are adapted from OpenLibm src/s_erf.c.
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.

use super::*;
use ruda_kernel::dsl as kernel_dsl;

#[ruda]
fn erf_small<F:Float>(x:F)->F {
    let s=x*x;
    let mut p=F::cast_from(-2.37630166566501626084e-5f64);
    p=fma(p,s,F::cast_from(-5.77027029648944159157e-3f64));
    p=fma(p,s,F::cast_from(-2.84817495755985104766e-2f64));
    p=fma(p,s,F::cast_from(-3.25042107247001499370e-1f64));
    p=fma(p,s,F::cast_from(1.28379167095512558561e-1f64));
    let mut q=F::cast_from(-3.96022827877536812320e-6f64);
    q=fma(q,s,F::cast_from(1.32494738004321644526e-4f64));
    q=fma(q,s,F::cast_from(5.08130628187576562776e-3f64));
    q=fma(q,s,F::cast_from(6.50222499887672944485e-2f64));
    q=fma(q,s,F::cast_from(3.97917223959155352819e-1f64));
    q=fma(q,s,F::cast_from(1.0f64));
    fma(x,p/q,x)
}

#[ruda]
fn erfc_positive<F:Float>(x:F,#[comptime] scaled:bool)->F {
    let one=F::cast_from(1.0f64);let mut result=F::new(f32::NAN);
    if x<F::cast_from(0.84375f64) {result=one-erf_small::<F>(x);}
    else if x<F::cast_from(1.25f64) {
        let s=x-one;
        let mut p=F::cast_from(-2.16637559486879084300e-3f64);
        p=fma(p,s,F::cast_from(3.54783043256182359371e-2f64));
        p=fma(p,s,F::cast_from(-1.10894694282396677476e-1f64));
        p=fma(p,s,F::cast_from(3.18346619901161753674e-1f64));
        p=fma(p,s,F::cast_from(-3.72207876035701323847e-1f64));
        p=fma(p,s,F::cast_from(4.14856118683748331666e-1f64));
        p=fma(p,s,F::cast_from(-2.36211856075265944077e-3f64));
        let mut q=F::cast_from(1.19844998467991074170e-2f64);
        q=fma(q,s,F::cast_from(1.36370839120290507362e-2f64));
        q=fma(q,s,F::cast_from(1.26171219808761642112e-1f64));
        q=fma(q,s,F::cast_from(7.18286544141962662868e-2f64));
        q=fma(q,s,F::cast_from(5.40397917702171048937e-1f64));
        q=fma(q,s,F::cast_from(1.06420880400844228286e-1f64));
        q=fma(q,s,one);
        result=F::cast_from(0.154937088489532470703125f64)-p/q;
    } else if x>=F::cast_from(28.0f64) {
        result=F::cast_from(0.0f64);
        if comptime!(scaled) {
            let reciprocal=one/x;let s=reciprocal*reciprocal;
            let mut p=F::cast_from(-1055.7421875f64);
            p=fma(p,s,F::cast_from(162.421875f64));p=fma(p,s,F::cast_from(-29.53125f64));
            p=fma(p,s,F::cast_from(6.5625f64));p=fma(p,s,F::cast_from(-1.875f64));
            p=fma(p,s,F::cast_from(0.75f64));p=fma(p,s,F::cast_from(-0.5f64));p=fma(p,s,one);
            result=(F::cast_from(0.5641895835477562869480794515607726f64)*reciprocal)*p;
        }
    } else if x==x {
        let reciprocal=one/x;let s=reciprocal*reciprocal;let mut p=F::cast_from(0.0f64);let mut q=one;
        if x<F::cast_from(2.857142857142857142857142857142857f64) {
            p=F::cast_from(-9.81432934416914548592f64);
            p=fma(p,s,F::cast_from(-8.12874355063065934246e1f64));
            p=fma(p,s,F::cast_from(-1.84605092906711035994e2f64));
            p=fma(p,s,F::cast_from(-1.62396669462573470355e2f64));
            p=fma(p,s,F::cast_from(-6.23753324503260060396e1f64));
            p=fma(p,s,F::cast_from(-1.05586262253232909814e1f64));
            p=fma(p,s,F::cast_from(-6.93858572707181764372e-1f64));
            p=fma(p,s,F::cast_from(-9.86494403484714822705e-3f64));
            q=F::cast_from(-6.04244152148580987438e-2f64);
            q=fma(q,s,F::cast_from(6.57024977031928170135f64));
            q=fma(q,s,F::cast_from(1.08635005541779435134e2f64));
            q=fma(q,s,F::cast_from(4.29008140027567833386e2f64));
            q=fma(q,s,F::cast_from(6.45387271733267880336e2f64));
            q=fma(q,s,F::cast_from(4.34565877475229228821e2f64));
            q=fma(q,s,F::cast_from(1.37657754143519042600e2f64));
            q=fma(q,s,F::cast_from(1.96512716674392571292e1f64));q=fma(q,s,one);
        } else {
            p=F::cast_from(-4.83519191608651397019e2f64);
            p=fma(p,s,F::cast_from(-1.02509513161107724954e3f64));
            p=fma(p,s,F::cast_from(-6.37566443368389627722e2f64));
            p=fma(p,s,F::cast_from(-1.60636384855821916062e2f64));
            p=fma(p,s,F::cast_from(-1.77579549177547519889e1f64));
            p=fma(p,s,F::cast_from(-7.99283237680523006574e-1f64));
            p=fma(p,s,F::cast_from(-9.86494292470009928597e-3f64));
            q=F::cast_from(-2.24409524465858183362e1f64);
            q=fma(q,s,F::cast_from(4.74528541206955367215e2f64));
            q=fma(q,s,F::cast_from(2.55305040643316442583e3f64));
            q=fma(q,s,F::cast_from(3.19985821950859553908e3f64));
            q=fma(q,s,F::cast_from(1.53672958608443695994e3f64));
            q=fma(q,s,F::cast_from(3.25792512996573918826e2f64));
            q=fma(q,s,F::cast_from(3.03380607434824582924e1f64));q=fma(q,s,one);
        }
        let correction=p/q-F::cast_from(0.5625f64);
        if comptime!(scaled) {result=correction.exp()*reciprocal;}
        else {
            let square=x*x;let residual=fma(-x,x,square);let e=(-F::cast_from(0.5f64)*square).exp();
            result=e*((e*(correction+residual).exp())*reciprocal);
        }
    }
    if comptime!(scaled) {if x<F::cast_from(1.25f64) {result*=(x*x).exp();}}
    result
}

#[ruda]
pub(super) fn erfc<F:Float>(x:F,#[comptime] scaled:bool)->F {
    let a=x.abs();let positive=erfc_positive::<F>(a,scaled);let mut result=positive;
    if x<F::cast_from(0.0f64) {
        if comptime!(scaled) {
            result=F::new(f32::INFINITY);
            if a<F::cast_from(27.0f64) {
                let square=a*a;let residual=fma(a,a,-square);let e=(F::cast_from(0.5f64)*square).exp();
                result=e*((F::cast_from(2.0f64)*e)*residual.exp())-positive;
            }
        } else {result=F::cast_from(2.0f64)-positive;}
    }
    result
}

#[ruda]
fn inverse_erfc_log<F:Float>(logq:F)->F {
    let mut x=(-logq).sqrt();let mut iteration:u32=0;
    while iteration<6u32 {
        let scaled=erfc_positive::<F>(x,true);
        let residual=fma(-x,x,scaled.ln()-logq);
        x+=F::cast_from(0.8862269254527580136490837416705726f64)*scaled*residual;
        iteration+=1u32;
    }
    x
}

#[ruda]
pub(super) fn erfinv<F:Float>(y:F)->F {
    let one=F::cast_from(1.0f64);let a=y.abs();let mut x=F::new(f32::NAN);
    if a==one {x=F::new(f32::INFINITY);}
    else if a<one {
        if a<=F::cast_from(0.7f64) {
            x=a;let mut iteration:u32=0;
            while iteration<4u32 {
                x-=(erf_small::<F>(x)-a)/(F::cast_from(1.1283791670955125738961589031215452f64)*(-x*x).exp());
                iteration+=1u32;
            }
        } else {x=inverse_erfc_log::<F>((one-a).ln());}
    }
    if y<F::cast_from(0.0f64) {x=-x;}
    if y==F::cast_from(0.0f64) {x=y;}
    x
}

#[ruda]
pub(super) fn ndtri<F:Float>(p:F)->F {
    let z=F::cast_from(0.0f64);let one=F::cast_from(1.0f64);let half=F::cast_from(0.5f64);let mut result=F::new(f32::NAN);
    if p==z {result=F::new(f32::NEG_INFINITY);}
    else if p==one {result=F::new(f32::INFINITY);}
    else if p>z && p<one {
        let mut tail=p;if p>half {tail=one-p;}
        let mut value=z;
        if tail>=F::cast_from(0.25f64) {value=erfinv::<F>(one-F::cast_from(2.0f64)*tail);}
        else {value=inverse_erfc_log::<F>((F::cast_from(2.0f64)*tail).ln());}
        result=F::cast_from(1.4142135623730950488016887242096981f64)*value;
        if p<half {result=-result;}
    }
    result
}

#[ruda]
pub(super) fn log_ndtr<F:Float>(x:F)->F {
    let t=x*F::cast_from(0.7071067811865475244008443621048490f64);let half=F::cast_from(0.5f64);let mut result=F::cast_from(0.0f64);
    if x<F::cast_from(-1.0f64) {result=(erfc_positive::<F>(-t,true)*half).ln()-t*t;}
    else {result=(-half*erfc::<F>(t,false)).log1p();}
    result
}
