// PyTorch arithmetic; notices for reused LLVM helpers are in math_extra.rs.
use super::*;
use super::scientific::address;
use super::math_extra::{complex_divide,complex_multiply};
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn c10_divide<F:Float>(a:F,b:F,c:F,d:F)->(F,F) {
    let zero=F::new(0.0);let one=F::new(1.0);let mut real=zero;let mut imag=zero;
    let mut ac=c;let mut ad=d;if c<zero {ac=-c;}if d<zero {ad=-d;}
    if ac>=ad {
        if ac==zero && ad==zero {real=a/ac;imag=b/ad;}
        else {let ratio=d/c;let scale=one/(c+d*ratio);real=(a+b*ratio)*scale;imag=(b-a*ratio)*scale;}
    } else {let ratio=c/d;let scale=one/(d+c*ratio);real=(a*ratio+b)*scale;imag=(b*ratio-a)*scale;}
    (real,imag)
}

#[ruda(launch)]
fn binary<S:Float+RudaElement,F:Float+RudaElement>(ar:&Tensor<S>,ai:&Tensor<S>,br:&Tensor<S>,bi:&Tensor<S>,out_r:&mut Tensor<S>,out_i:&mut Tensor<S>,alpha_r:F,alpha_i:F,scalar_ar:F,scalar_ai:F,scalar_br:f64,scalar_bi:f64,#[comptime] op:u32,#[comptime] scalar_left:bool,#[comptime] scalar_right:bool,#[comptime] half:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<out_r.len() {
        let mut a=scalar_ar;let mut b=scalar_ai;
        let mut c=F::cast_from(scalar_br);let mut d=F::cast_from(scalar_bi);
        if comptime!(!scalar_left) {a=F::cast_from(ar[address(ar,p)]);b=F::cast_from(ai[address(ai,p)]);}
        if comptime!(!scalar_right) {c=F::cast_from(br[address(br,p)]);d=F::cast_from(bi[address(bi,p)]);}
        let mut real=F::new(0.0);let mut imag=F::new(0.0);
        if comptime!(op<=1) {
            real=a+(alpha_r*c-alpha_i*d);imag=b+(alpha_r*d+alpha_i*c);
        } else if comptime!(op==2 || op==4) {
            if comptime!(half && op==2) {let(r,i)=complex_multiply(a,b,c,d,0);real=r;imag=i;}
            else {real=a*c-b*d;imag=a*d+b*c;}
        } else if comptime!(op==3) {
            if comptime!(half) {let(r,i)=complex_divide(a,b,c,d,0,23,8);real=r;imag=i;}
            else if comptime!(scalar_right) {
                let(r,i)=c10_divide(1.0f64,0.0f64,scalar_br,scalar_bi);
                let cr=F::cast_from(r);let ci=F::cast_from(i);real=a*cr-b*ci;imag=a*ci+b*cr;
            } else {let(r,i)=c10_divide(a,b,c,d);real=r;imag=i;}
        } else {
            if comptime!(scalar_left) {a=F::cast_from(S::cast_from(a));b=F::cast_from(S::cast_from(b));}
            if comptime!(scalar_right) {c=F::cast_from(S::cast_from(c));d=F::cast_from(S::cast_from(d));}
            let equal=a==c && b==d;
            if comptime!(op==5) {real=F::cast_from(equal);}else {real=F::cast_from(!equal);}
        }
        out_r[address(out_r,p)]=S::cast_from(real);out_i[address(out_i,p)]=S::cast_from(imag);
    }
}

pub(super) fn dispatch(xs:&[View],ints:&[i64],values:&[f64]) {
    assert_eq!(xs.len(),6);assert_eq!(ints.len(),3);assert_eq!(values.len(),6);
    assert!((0..=6).contains(&ints[0]));assert!(matches!(ints[1],0|1) && matches!(ints[2],0|1));
    assert!(ints[1]+ints[2]<=1);let code=xs[0].dtype;assert!(matches!(code,0|1|9));
    for x in &xs[1..] {assert_eq!(x.dtype,code);assert_eq!(x.shape,xs[0].shape);}
    if xs[4].len==0 {return;}
    let c=client();let grid=RudaCount::Static(u32::try_from(xs[4].len.div_ceil(256)).expect("complex arithmetic grid overflow"),1,1);
    macro_rules! run {($s:ty,$f:ty)=>{unsafe {
        binary::launch::<$s,$f,CudaRuntime>(&c,grid,RudaDim::new_1d(256),xs[0].arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),xs[4].arg(),xs[5].arg(),
            values[0] as $f,values[1] as $f,values[2] as $f,values[3] as $f,values[4],values[5],ints[0] as u32,ints[1]!=0,ints[2]!=0,code==1);
    }};}
    match code {0=>run!(f32,f32),1=>run!(f16,f32),9=>run!(f64,f64),_=>unreachable!()}
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);
}
