use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn real<S: Float + RudaElement, F: Float>(
    left: &Tensor<S>, right: &Tensor<S>, output: &mut Tensor<S>, epsilon: F, #[comptime] op: u32,
) {
    let p=(RUDA_POS_X as usize + RUDA_POS_Y as usize*RUDA_COUNT_X as usize)*256usize + UNIT_POS as usize;
    if p<output.len() {
        let x=F::cast_from(left[address(left,p)]); let y=F::cast_from(right[address(right,p)]); let zero=F::new(0.0);
        let (value,_imag)=crate::elementary::evaluate(x,zero,y,zero,epsilon,comptime!(op+253u32),false);
        output[address(output,p)]=S::cast_from(value);
    }
}

#[ruda(launch)]
fn complex<S: Float + RudaElement, F: Float>(
    left_real: &Tensor<S>, left_imag: &Tensor<S>, right_real: &Tensor<S>, right_imag: &Tensor<S>,
    output_real: &mut Tensor<S>, output_imag: &mut Tensor<S>,
) {
    let p=(RUDA_POS_X as usize + RUDA_POS_Y as usize*RUDA_COUNT_X as usize)*256usize + UNIT_POS as usize;
    if p<output_real.len() {
        let x=F::cast_from(left_real[address(left_real,p)]); let u=F::cast_from(left_imag[address(left_imag,p)]);
        let y=F::cast_from(right_real[address(right_real,p)]); let v=F::cast_from(right_imag[address(right_imag,p)]);
        let (real,imag)=crate::elementary::evaluate(x,u,y,v,F::new(0.0),253,true);
        output_real[address(output_real,p)]=S::cast_from(real); output_imag[address(output_imag,p)]=S::cast_from(imag);
    }
}

pub(super) fn dispatch(xs:&[View],ints:&[i64],values:&[f64]) {
    assert_eq!(xs.len(),3); assert_eq!(ints.len(),1); assert!((0..=3).contains(&ints[0]));
    let op=ints[0] as u32; assert_eq!(values.len(),if op>=2 {1}else{0});
    let code=xs[2].dtype; assert!(matches!(code,0|1|2|9|10|11|14));
    if op!=0 {assert!(matches!(code,0|1|2|9));}
    for input in &xs[..2] {assert_eq!(input.dtype,code); assert_eq!(input.shape,xs[2].shape);}
    if xs[2].len==0 {return;}
    let epsilon=values.first().copied().unwrap_or(-1.0);
    let c=client(); let blocks=xs[2].len.div_ceil(256); let x=blocks.min(131072);
    let grid=RudaCount::Static(x as u32,blocks.div_ceil(x) as u32,1); let block=RudaDim::new_1d(256);
    if matches!(code,10|11|14) {
        let parts:Vec<View>=xs.iter().flat_map(|value|[scientific::component(value,false),scientific::component(value,true)]).collect();
        for value in &parts {scientific::indexable(value);}
        macro_rules! run {($s:ty,$f:ty)=>{unsafe {
            complex::launch::<$s,$f,CudaRuntime>(&c,grid,block,parts[0].arg(),parts[1].arg(),parts[2].arg(),parts[3].arg(),parts[4].arg(),parts[5].arg());
        }};}
        match code {10=>run!(f32,f32),11=>run!(f64,f64),14=>run!(f16,f32),_=>unreachable!()}
    } else {
        macro_rules! run {($s:ty,$f:ty)=>{unsafe {
            real::launch::<$s,$f,CudaRuntime>(&c,grid,block,xs[0].arg(),xs[1].arg(),xs[2].arg(),epsilon as $f,op);
        }};}
        match code {0=>run!(f32,f32),1=>run!(f16,f32),2=>run!(bf16,f32),9=>run!(f64,f64),_=>unreachable!()}
    }
    LAUNCHES.fetch_add(1,Ordering::Relaxed); finish_dispatch(&c);
}
