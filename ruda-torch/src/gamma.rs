// lgamma: adapted from OpenLibm src/e_lgamma_r.c.
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunSoft, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.
// Digamma and Hurwitz zeta: Cephes formulas used by ATen/native/Math.h.
// Copyright (c) 2018, Steven Moshier. All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of the copyright holder nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

use super::*;
use ruda_kernel::dsl as kernel_dsl;

#[ruda]
pub(super) fn lgamma(input:f64)->f64 {
    let x=input.abs();let mut result=0.0f64;
    if x!=x || x==f64::INFINITY {result=x;}
    else if x==0.0 || (input<0.0 && x==x.floor()) {result=f64::INFINITY;}
    else if x<8.470329472543003e-22 {result=-x.ln();}
    else {
        if x==1.0 || x==2.0 {result=0.0;}
        else if x<2.0 {
            let mut y=0.0f64;let mut region:u32=0;
            if x<0.9000000953674316 {
                result=-x.ln();
                if x>=0.7315998077392578 {y=1.0-x;region=0;}
                else if x>=0.23163998126983643 {y=x-(1.46163214496836224576e+00-1.0);region=1;}
                else {y=x;region=2;}
            } else {
                if x>=1.7316312789916992 {y=2.0-x;region=0;}
                else if x>=1.2316322326660156 {y=x-1.46163214496836224576e+00;region=1;}
                else {y=x-1.0;region=2;}
            }
            if region==0 {
                let z=y*y;
                let p1=(7.72156649015328655494e-02+z*(6.73523010531292681824e-02+z*(7.38555086081402883957e-03+z*(1.19270763183362067845e-03+z*(2.20862790713908385557e-04+z*2.52144565451257326939e-05)))));
                let p2=z*(3.22467033424113591611e-01+z*(2.05808084325167332806e-02+z*(2.89051383673415629091e-03+z*(5.10069792153511336608e-04+z*(1.08011567247583939954e-04+z*4.48640949618915160150e-05)))));
                result+=(y*p1+p2)-0.5*y;
            } else if region==1 {
                let z=y*y;let w=z*y;
                let p1=(4.83836122723810047042e-01+w*(-3.27885410759859649565e-02+w*(6.10053870246291332635e-03+w*(-1.40346469989232843813e-03+w*3.15632070903625950361e-04))));
                let p2=(-1.47587722994593911752e-01+w*(1.79706750811820387126e-02+w*(-3.68452016781138256760e-03+w*(8.81081882437654011382e-04+w*-3.12754168375120860518e-04))));
                let p3=(6.46249402391333854778e-02+w*(-1.03142241298341437450e-02+w*(2.25964780900612472250e-03+w*(-5.38595305356740546715e-04+w*3.35529192635519073543e-04))));
                result+=-1.21486290535849611461e-01+(z*p1-(-3.63867699703950536541e-18-w*(p2+y*p3)));
            } else {
                let p=y*(-7.72156649015328655494e-02+y*(6.32827064025093366517e-01+y*(1.45492250137234768737e+00+y*(9.77717527963372745603e-01+y*(2.28963728064692451092e-01+y*1.33810918536787660377e-02)))));
                let q=1.0+y*(2.45597793713041134822e+00+y*(2.12848976379893395361e+00+y*(7.69285150456672783825e-01+y*(1.04222645593369134254e-01+y*3.21709242282423911810e-03))));
                result+=-0.5*y+p/q;
            }
        } else if x<8.0 {
            let integer=x.floor();let y=x-integer;
            let p=y*(-7.72156649015328655494e-02+y*(2.14982415960608852501e-01+y*(3.25778796408930981787e-01+y*(1.46350472652464452805e-01+y*(2.66422703033638609560e-02+y*(1.84028451407337715652e-03+y*3.19475326584100867617e-05))))));
            let q=1.0+y*(1.39200533467621045958e+00+y*(7.21935547567138069525e-01+y*(1.71933865632803078993e-01+y*(1.86459191715652901344e-02+y*(7.77942496381893596434e-04+y*7.32668430744625636189e-06)))));
            result=0.5*y+p/q;
            let mut z=1.0f64;let mut j=integer-1.0;
            while j>=2.0 {z*=y+j;j-=1.0;}
            result+=z.ln();
        } else if x<288230376151711744.0 {
            let z=1.0/x;let y=z*z;
            let w=4.18938533204672725052e-01+z*(8.33333333333329678849e-02+y*(-2.77777777728775536470e-03+y*(7.93650558643019558500e-04+y*(-5.95187557450339963135e-04+y*(8.36339918996282139126e-04+y*-1.63092934096575273989e-03)))));
            result=(x-0.5)*(x.ln()-1.0)+w;
        } else {result=x*(x.ln()-1.0);}
        if input<0.0 {
            let mut fraction=x-x.floor();if fraction>0.5 {fraction=1.0-fraction;}
            let mut sine=0.0f64;
            if fraction<0.25 {sine=(3.14159265358979323846*fraction).sin();}
            else {sine=(3.14159265358979323846*(0.5-fraction)).cos();}
            result=(3.14159265358979323846/(x*sine)).ln()-result;
        }
    }
    result
}

#[ruda]
pub(super) fn digamma<F:Float>(input:F)->F {
    let z=F::new(0.0);let one=F::new(1.0);let mut x=input;let mut result=z;
    if x==z {result=-one/x;}
    else if x<z && x==x.trunc() {result=F::new(f32::NAN);}
    else {
        if x<z {
            let xd=f64::cast_from(x);let fraction=xd-xd.trunc();
            result=F::cast_from(-3.14159265358979323846f64/(3.14159265358979323846f64*fraction).tan());
            x=one-x;
        }
        while x<F::new(10.0) {result-=one/x;x+=one;}
        if x==F::new(10.0) {result+=F::cast_from(2.25175258906672110764f64);}
        else {
            let mut correction=z;
            if x<F::cast_from(1.0e17f64) {
                let s=one/(x*x);
                let mut p=F::cast_from(8.33333333333333333333e-2f64);
                p=p*s-F::cast_from(2.10927960927960927961e-2f64);
                p=p*s+F::cast_from(7.57575757575757575758e-3f64);
                p=p*s-F::cast_from(4.16666666666666666667e-3f64);
                p=p*s+F::cast_from(3.96825396825396825397e-3f64);
                p=p*s-F::cast_from(8.33333333333333333333e-3f64);
                p=p*s+F::cast_from(8.33333333333333333333e-2f64);
                correction=s*p;
            }
            result=result+x.ln()-F::new(0.5)/x-correction;
        }
    }
    result
}

#[ruda]
fn trigamma<F:Float>(input:F)->F {
    let one=F::new(1.0);let pi=F::cast_from(3.14159265358979323846f64);
    let mut x=input;let mut sign=one;let mut result=F::new(0.0);
    if x<F::new(0.5) {sign=-one;let sine=(pi*x).sin();result-=(pi*pi)/(sine*sine);x=one-x;}
    let mut j:u32=0;
    while j<6 {result+=one/(x*x);x+=one;j+=1;}
    let s=one/(x*x);
    result+=(one+one/(F::new(2.0)*x)+s*(one/F::new(6.0)-s*(one/F::new(30.0)-s*(one/F::new(42.0)))))/x;
    sign*result
}

#[ruda]
fn hurwitz<F:Float>(power:F,input:F)->F {
    let one=F::new(1.0);let zero=F::new(0.0);let eps=F::cast_from(1.11022302462515654042e-16f64);
    let mut result=F::new(f32::NAN);
    if input<=zero && input==input.floor() {result=F::new(f32::INFINITY);}
    else if power>one {
        let mut a=input;let mut b=zero;let mut sum=input.powf(-power);let mut j:u32=0;let mut converged=false;
        while j<9 || a<=F::new(9.0) {
            j+=1;a+=one;b=a.powf(-power);sum+=b;
            if -eps*sum<b && b<eps*sum {converged=true;break;}
        }
        if !converged {
            let w=a;sum+=b*w/(power-one);sum-=F::new(0.5)*b;
            a=one;let mut k=zero;j=0;
            while j<12 {
                let mut denominator=F::cast_from(-7.1661652561756670113e18f64);
                if j==0 {denominator=F::new(12.0);}
                else if j==1 {denominator=F::new(-720.0);}
                else if j==2 {denominator=F::new(30240.0);}
                else if j==3 {denominator=F::new(-1209600.0);}
                else if j==4 {denominator=F::cast_from(47900160.0f64);}
                else if j==5 {denominator=F::cast_from(-1.8924375803183791606e9f64);}
                else if j==6 {denominator=F::cast_from(7.47242496e10f64);}
                else if j==7 {denominator=F::cast_from(-2.950130727918164224e12f64);}
                else if j==8 {denominator=F::cast_from(1.1646782814350067249e14f64);}
                else if j==9 {denominator=F::cast_from(-4.5979787224074726105e15f64);}
                else if j==10 {denominator=F::cast_from(1.8152105401943546773e17f64);}
                a*=power+k;b/=w;let term=a*b/denominator;sum+=term;
                if (term/sum).abs()<eps {break;}
                k+=one;a*=power+k;b/=w;k+=one;j+=1;
            }
        }
        result=sum;
    } else if power==one {result=F::new(f32::INFINITY);}
    result
}

#[ruda]
pub(super) fn polygamma<F:Float>(x:F,order:i64)->F {
    let mut result=F::new(0.0);
    if order==0 {result=digamma::<F>(x);}
    else if order==1 {result=trigamma::<F>(x);}
    else {
        let n=F::cast_from(order);let argument=n+F::new(1.0);
        let factor=F::cast_from(lgamma(f64::cast_from(argument))).exp();
        result=factor*hurwitz::<F>(argument,x);
        if order%2==0 {result=-result;}
    }
    result
}
