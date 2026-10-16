// ATen complex construction and classification; PyTorch notice in bessel.rs.
use super::*;
use super::scientific::address;
use super::floating::bits;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn phase<S:Float+RudaElement,F:Float+RudaElement>(real:&Tensor<S>,imag:&Tensor<S>,out:&mut Tensor<S>,#[comptime] complex:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let x=F::cast_from(real[address(real,p)]);let mut value=F::new(0.0);
        if comptime!(complex) {value=F::cast_from(imag[address(imag,p)]).atan2(x);}
        else if x!=x {value=x;}
        else if x<F::new(0.0) {value=F::cast_from(3.14159265358979323846f64);}
        out[address(out,p)]=S::cast_from(value);
    }
}

#[ruda(launch)]
fn construct<S:Float+RudaElement,F:Float+RudaElement>(a:&Tensor<S>,b:&Tensor<S>,real:&mut Tensor<S>,imag:&mut Tensor<S>,#[comptime] polar:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<real.len() {
        let x=a[address(a,p)];let y=b[address(b,p)];let mut re=x;let mut im=y;
        if comptime!(polar) {
            let radius=F::cast_from(x);let theta=F::cast_from(y);
            re=S::cast_from(radius*theta.cos());im=S::cast_from(radius*theta.sin());
        }
        real[address(real,p)]=re;imag[address(imag,p)]=im;
    }
}

#[ruda(launch)]
fn replace_nonfinite<S:Float+RudaElement,F:Float+RudaElement>(a:&Tensor<S>,out:&mut Tensor<S>,nan:F,posinf:F,neginf:F) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let x=a[address(a,p)];let inf=S::new(f32::INFINITY);let mut value=x;
        if x!=x {value=S::cast_from(nan);}
        else if x==inf {value=S::cast_from(posinf);}
        else if x== -inf {value=S::cast_from(neginf);}
        out[address(out,p)]=value;
    }
}

#[ruda(launch)]
fn copy_integer<I:Numeric+RudaElement>(a:&Tensor<I>,out:&mut Tensor<I>) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {out[address(out,p)]=a[address(a,p)];}
}

#[ruda(launch)]
fn classify_float<S:Float+RudaElement>(a:&Tensor<S>,out:&mut Tensor<u8>,#[comptime] op:u32,#[comptime] code:u32,#[comptime] sign:u64,#[comptime] infinity:u64) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let raw=bits::<S>(a[address(a,p)],code);let mut value=false;
        if comptime!(op==393) {value=(raw&sign)!=0u64;}
        else if comptime!(op==394) {value=raw==infinity;}
        else {value=raw==(infinity|sign);}
        out[address(out,p)]=u8::cast_from(value);
    }
}

#[ruda(launch)]
fn classify_integer<I:Numeric+RudaElement>(a:&Tensor<I>,out:&mut Tensor<u8>,#[comptime] sign:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let mut value=false;
        if comptime!(sign) {value=a[address(a,p)]<I::cast_from(0i32);}
        out[address(out,p)]=u8::cast_from(value);
    }
}

pub(super) fn dispatch(op:u32,xs:&[View],ints:&[i64],values:&[f64]) {
    assert!((389..=395).contains(&op));
    assert_eq!(xs.len(),if op==389 {3}else if op==390 || op==391 {4}else{2});
    assert_eq!(ints.len(),usize::from(op==389));
    assert_eq!(values.len(),if op==392 {3}else{0});
    let code=xs[0].dtype;let output=if op<=391 {2}else{1};
    for x in &xs[1..] {assert_eq!(x.shape,xs[0].shape);}
    if op>=393 {assert_eq!(xs[1].dtype,3);assert!(code<=9);}
    else {for x in &xs[1..] {assert_eq!(x.dtype,code);}}
    if op==389 {assert!(matches!(code,0|1|9));assert!(matches!(ints[0],0|1));assert!(code!=1 || ints[0]==1);}
    if op==390 {assert!(matches!(code,0|1|9));}
    if op==391 {assert!(matches!(code,0|9));}
    let count=xs[output].len;if count==0 {return;}
    let c=client();let grid=RudaCount::Static(u32::try_from(count.div_ceil(256)).expect("complex/classification launch overflow"),1,1);
    let block=RudaDim::new_1d(256);
    if op==389 {
        macro_rules! angle {($s:ty,$f:ty)=>{unsafe {
            phase::launch::<$s,$f,CudaRuntime>(&c,grid,block,xs[0].arg(),xs[1].arg(),xs[2].arg(),ints[0]!=0);
        }};}
        match code {0=>angle!(f32,f32),1=>angle!(f16,f32),9=>angle!(f64,f64),_=>unreachable!()}
    } else {
        macro_rules! floating {($s:ty,$f:ty,$sign:expr,$inf:expr)=>{unsafe {
            match op {
                390|391=>construct::launch::<$s,$f,CudaRuntime>(&c,grid,block,xs[0].arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),op==391),
                392=>replace_nonfinite::launch::<$s,$f,CudaRuntime>(&c,grid,block,xs[0].arg(),xs[1].arg(),values[0] as $f,values[1] as $f,values[2] as $f),
                393..=395=>classify_float::launch::<$s,CudaRuntime>(&c,grid,block,xs[0].arg(),xs[1].arg(),op,code,$sign,$inf),
                _=>unreachable!(),
            }
        }};}
        macro_rules! integral {($i:ty)=>{unsafe {
            if op==392 {copy_integer::launch::<$i,CudaRuntime>(&c,grid,block,xs[0].arg(),xs[1].arg());}
            else {classify_integer::launch::<$i,CudaRuntime>(&c,grid,block,xs[0].arg(),xs[1].arg(),op==393);}
        }};}
        match code {
            0=>floating!(f32,f32,0x80000000u64,0x7f800000u64),
            1=>floating!(f16,f32,0x8000u64,0x7c00u64),
            2=>floating!(bf16,f32,0x8000u64,0x7f80u64),
            3|8=>integral!(u8),4=>integral!(i64),5=>integral!(i32),6=>integral!(i16),7=>integral!(i8),
            9=>floating!(f64,f64,0x8000000000000000u64,0x7ff0000000000000u64),
            _=>panic!("unsupported complex/classification storage dtype"),
        }
    }
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);
}
