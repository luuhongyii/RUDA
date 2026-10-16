use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn real<S: Float + RudaElement, F: Float>(
    input: &Tensor<S>, output: &mut Tensor<S>, #[comptime] op: u32,
) {
    let p=(RUDA_POS_X as usize + RUDA_POS_Y as usize*RUDA_COUNT_X as usize)*256usize + UNIT_POS as usize;
    if p<output.len() {
        let x=F::cast_from(input[address(input,p)]); let zero=F::new(0.0); let one=F::new(1.0);
        let mut value=zero;
        if comptime!(op==0) {value=x.exp();}
        else if comptime!(op==1) {value=x.ln();}
        else if comptime!(op==2) {value=x.log1p();}
        else if comptime!(op==3) {value=one/(one+(-x).exp());}
        else {
            let (r,_i)=crate::elementary::evaluate(x,zero,zero,zero,zero,comptime!(op+245u32),false);
            value=r;
        }
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
        let zero=F::new(0.0); let one=F::new(1.0); let mut real=zero; let mut imag=zero;
        if comptime!(op==0) {
            let (r,i)=crate::power_ops::jit_exp(x,y,code); real=r; imag=i;
        } else if comptime!(op==1 || op==6 || op==7) {
            real=x.hypot(y).ln(); imag=y.atan2(x);
            if comptime!(op==6) {let denominator=F::new(2.0).ln(); real/=denominator; imag/=denominator;}
            else if comptime!(op==7) {let denominator=F::new(10.0).ln(); real/=denominator; imag/=denominator;}
        } else if comptime!(op==2) {
            let (r,i)=crate::elementary::complex_log1p(x,y); real=r; imag=i;
        } else if comptime!(op==3) {
            let (r,i)=crate::power_ops::jit_exp(-x,-y,code);
            let (a,b)=crate::math_extra::complex_divide(one,zero,one+r,zero+i,code,
                comptime!(if code==9 {52u32}else{23u32}),comptime!(if code==9 {11u32}else{8u32}));
            real=a; imag=b;
        } else if comptime!(op==4) {
            let ln2=F::cast_from(0.693147180559945309417232121458176f64);
            let (r,i)=crate::power_ops::jit_exp(ln2*x,ln2*y,code); real=r; imag=i;
        } else if comptime!(op==5) {
            let (r,i)=crate::elementary::evaluate(x,y,zero,zero,zero,250,true); real=r; imag=i;
        }
        output_real[address(output_real,p)]=S::cast_from(real);
        output_imag[address(output_imag,p)]=S::cast_from(imag);
    }
}

pub(super) fn dispatch(xs:&[View],ints:&[i64]) {
    assert_eq!(xs.len(),2); assert_eq!(ints.len(),1); assert!((0..=7).contains(&ints[0]));
    let op=ints[0] as u32; let input=&xs[0]; let output=&xs[1]; let code=input.dtype;
    assert!(matches!(code,0|1|2|9|10|11|14)); assert_eq!(output.dtype,code); assert_eq!(input.shape,output.shape);
    assert!(code!=14 || matches!(op,0|1|3));
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

#[ruda(launch)]
fn backward_real<S: Float + RudaElement, F: Float>(
    grad: &Tensor<S>, saved: &Tensor<S>, output: &mut Tensor<S>, #[comptime] tanh: bool,
) {
    let p=(RUDA_POS_X as usize + RUDA_POS_Y as usize*RUDA_COUNT_X as usize)*256usize + UNIT_POS as usize;
    if p<output.len() {
        let a=F::cast_from(grad[address(grad,p)]); let b=F::cast_from(saved[address(saved,p)]); let one=F::new(1.0);
        let mut result=S::cast_from(0u32);
        if comptime!(tanh) {
            let square=F::cast_from(S::cast_from(b*b)); let difference=F::cast_from(S::cast_from(one-square));
            result=S::cast_from(a*difference);
        } else {
            let difference=F::cast_from(S::cast_from(one-b)); let product=F::cast_from(S::cast_from(a*difference));
            result=S::cast_from(product*b);
        }
        output[address(output,p)]=result;
    }
}

#[ruda(launch)]
fn backward_complex<S: Float + RudaElement, F: Float>(
    grad_real: &Tensor<S>, grad_imag: &Tensor<S>, saved_real: &Tensor<S>, saved_imag: &Tensor<S>,
    output_real: &mut Tensor<S>, output_imag: &mut Tensor<S>, #[comptime] tanh: bool, #[comptime] code: u32,
) {
    let p=(RUDA_POS_X as usize + RUDA_POS_Y as usize*RUDA_COUNT_X as usize)*256usize + UNIT_POS as usize;
    if p<output_real.len() {
        let a=F::cast_from(grad_real[address(grad_real,p)]); let ai=F::cast_from(grad_imag[address(grad_imag,p)]);
        let b=F::cast_from(saved_real[address(saved_real,p)]); let bi=F::cast_from(saved_imag[address(saved_imag,p)]);
        let zero=F::new(0.0); let one=F::new(1.0); let mut dr=zero; let mut di=zero;
        if comptime!(tanh) {
            let (r,i)=crate::math_extra::complex_multiply(b,bi,b,bi,code); dr=one-r; di=zero-i;
        } else {
            let (r,i)=crate::math_extra::complex_multiply(one-b,zero-bi,b,bi,code); dr=r; di=i;
        }
        let (r,i)=crate::math_extra::complex_multiply(a,ai,dr,-di,code);
        output_real[address(output_real,p)]=S::cast_from(r); output_imag[address(output_imag,p)]=S::cast_from(i);
    }
}

pub(super) fn dispatch_backward(xs:&[View],ints:&[i64]) {
    assert_eq!(xs.len(),3); assert_eq!(ints.len(),1); assert!(matches!(ints[0],0|1));
    let code=xs[2].dtype; assert!(matches!(code,0|1|2|9|10|11|14));
    for input in &xs[..2] {assert_eq!(input.dtype,code); assert_eq!(input.shape,xs[2].shape);}
    if xs[2].len==0 {return;}
    let c=client(); let blocks=xs[2].len.div_ceil(256); let x=blocks.min(131072);
    let grid=RudaCount::Static(x as u32,blocks.div_ceil(x) as u32,1); let block=RudaDim::new_1d(256); let tanh=ints[0]!=0;
    if matches!(code,10|11|14) {
        let parts:Vec<View>=xs.iter().flat_map(|value|[scientific::component(value,false),scientific::component(value,true)]).collect();
        for value in &parts {scientific::indexable(value);}
        macro_rules! run {($s:ty,$f:ty)=>{unsafe {
            backward_complex::launch::<$s,$f,CudaRuntime>(&c,grid,block,parts[0].arg(),parts[1].arg(),parts[2].arg(),parts[3].arg(),parts[4].arg(),parts[5].arg(),tanh,if code==11{9}else{0});
        }};}
        match code {10=>run!(f32,f32),11=>run!(f64,f64),14=>run!(f16,f32),_=>unreachable!()}
    } else {
        macro_rules! run {($s:ty,$f:ty)=>{unsafe {
            backward_real::launch::<$s,$f,CudaRuntime>(&c,grid,block,xs[0].arg(),xs[1].arg(),xs[2].arg(),tanh);
        }};}
        match code {0=>run!(f32,f32),1=>run!(f16,f32),2=>run!(bf16,f32),9=>run!(f64,f64),_=>unreachable!()}
    }
    LAUNCHES.fetch_add(1,Ordering::Relaxed); finish_dispatch(&c);
}
