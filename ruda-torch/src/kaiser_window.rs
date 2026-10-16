use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn window<S: Float + RudaElement, F: Float + RudaElement>(
    input: &Tensor<S>, output: &mut Tensor<S>, inv_alpha: F, beta: F, inv_i0_beta: F,
    #[comptime] wide: bool,
) {
    let p=(RUDA_POS_X as usize + RUDA_POS_Y as usize*RUDA_COUNT_X as usize)*256usize + UNIT_POS as usize;
    if p<output.len() {
        let a=F::cast_from(input[address(input,p)]); let one=F::new(1.0); let zero=F::new(0.0);
        let x=a*inv_alpha-one; let mut y=one-x*x;
        if !(y>zero) {y=zero;}
        let value=crate::bessel::evaluate::<F>(beta*y.sqrt(),233u32,wide)*inv_i0_beta;
        output[address(output,p)]=S::cast_from(value);
    }
}

pub(super) fn dispatch(xs:&[View],values:&[f64]) {
    assert_eq!(xs.len(),2); assert_eq!(values.len(),3);
    let input=&xs[0]; let output=&xs[1]; let code=output.dtype;
    assert!(matches!(code,0|1|2|9)); assert_eq!(input.dtype,code); assert_eq!(input.shape,output.shape);
    if output.len==0 {return;}
    let c=client(); let blocks=output.len.div_ceil(256); let x=blocks.min(131072);
    let grid=RudaCount::Static(x as u32,blocks.div_ceil(x) as u32,1); let block=RudaDim::new_1d(256);
    macro_rules! run {($s:ty,$f:ty)=>{unsafe {
        window::launch::<$s,$f,CudaRuntime>(&c,grid,block,input.arg(),output.arg(),values[0] as $f,values[1] as $f,values[2] as $f,code==9);
    }};}
    match code {0=>run!(f32,f32),1=>run!(f16,f32),2=>run!(bf16,f32),9=>run!(f64,f64),_=>unreachable!()}
    LAUNCHES.fetch_add(1,Ordering::Relaxed); finish_dispatch(&c);
}
