use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn real<S: Float + RudaElement, F: Float>(
    input: &Tensor<S>, output: &mut Tensor<S>, #[comptime] op: u32,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize)*256usize + UNIT_POS as usize;
    if p < output.len() {
        let x = F::cast_from(input[address(input,p)]); let mut value = x;
        if comptime!(op == 0) { value = x.abs(); }
        else if comptime!(op == 1 || op == 2) {
            value = F::new(0.0);
            if x > F::new(0.0) { value = F::new(1.0); }
            else if x < F::new(0.0) { value = F::new(-1.0); }
        } else if comptime!(op == 3) { value = x.floor(); }
        else if comptime!(op == 4) { value = x.ceil(); }
        else if comptime!(op == 5) { value = x.trunc(); }
        else if comptime!(op == 6) { value = x.round(); }
        output[address(output,p)] = S::cast_from(value);
    }
}

#[ruda(launch)]
fn integral<S: Numeric + RudaElement>(
    input: &Tensor<S>, output: &mut Tensor<S>, #[comptime] op: u32, #[comptime] boolean: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize)*256usize + UNIT_POS as usize;
    if p < output.len() {
        let x = input[address(input,p)]; let zero = S::cast_from(0u32); let mut value = x;
        if comptime!(boolean) { value = S::cast_from(x != zero); }
        else if comptime!(op == 0) {
            if x < zero { value = S::cast_from(0u64-u64::cast_from(x)); }
        } else if comptime!(op == 1 || op == 2) {
            value = zero;
            if x > zero { value = S::cast_from(1u32); }
            else if x < zero { value = S::cast_from(0u64-1u64); }
        }
        output[address(output,p)] = value;
    }
}

#[ruda(launch)]
fn complex<S: Float + RudaElement, F: Float>(
    real: &Tensor<S>, imag: &Tensor<S>, output_real: &mut Tensor<S>, output_imag: &mut Tensor<S>,
    #[comptime] absolute: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize)*256usize + UNIT_POS as usize;
    if p < output_real.len() {
        let x = F::cast_from(real[address(real,p)]); let y = F::cast_from(imag[address(imag,p)]);
        let magnitude = x.hypot(y); let zero = F::new(0.0); let mut r = magnitude; let mut i = zero;
        if comptime!(!absolute) {
            r = zero;
            if x != zero || y != zero { r = x/magnitude; i = y/magnitude; }
            output_imag[address(output_imag,p)] = S::cast_from(i);
        }
        output_real[address(output_real,p)] = S::cast_from(r);
    }
}

pub(super) fn dispatch(xs: &[View], ints: &[i64]) {
    assert_eq!(xs.len(),2); assert_eq!(ints.len(),1); assert!((0..=6).contains(&ints[0]));
    let op = ints[0] as u32; let input = &xs[0]; let output = &xs[1]; let code = input.dtype;
    assert!(code<=11 || code==14); assert_eq!(input.shape,output.shape);
    let is_complex = matches!(code,10|11|14);
    let component_code = match code {10=>0,11=>9,14=>1,_=>code};
    assert!(!is_complex || op==0 || op==2); assert!(code!=3 || op<=2);
    assert_eq!(output.dtype,if is_complex && op==0 {component_code}else{code});
    if output.len==0 {return;}
    let c=client(); let blocks=output.len.div_ceil(256); let x=blocks.min(131072);
    let grid=RudaCount::Static(x as u32,blocks.div_ceil(x) as u32,1); let block=RudaDim::new_1d(256);
    if is_complex {
        let real=scientific::component(input,false); let imag=scientific::component(input,true);
        let output_parts=(op!=0).then(|| (scientific::component(output,false),scientific::component(output,true)));
        let (out_real,out_imag)=match &output_parts {Some((real,imag))=>(real,imag),None=>(output,output)};
        for value in [&real,&imag,out_real,out_imag] {scientific::indexable(value);}
        macro_rules! run {($s:ty,$f:ty)=>{unsafe {
            complex::launch::<$s,$f,CudaRuntime>(&c,grid,block,real.arg(),imag.arg(),out_real.arg(),out_imag.arg(),op==0);
        }};}
        match code {10=>run!(f32,f32),11=>run!(f64,f64),14=>run!(f16,f32),_=>unreachable!()}
    } else {
        macro_rules! floating {($s:ty,$f:ty)=>{unsafe {
            real::launch::<$s,$f,CudaRuntime>(&c,grid,block,input.arg(),output.arg(),op);
        }};}
        macro_rules! integer {($s:ty)=>{unsafe {
            integral::launch::<$s,CudaRuntime>(&c,grid,block,input.arg(),output.arg(),op,code==3);
        }};}
        match code {0=>floating!(f32,f32),1=>floating!(f16,f32),2=>floating!(bf16,f32),9=>floating!(f64,f64),
            3|8=>integer!(u8),4=>integer!(i64),5=>integer!(i32),6=>integer!(i16),7=>integer!(i8),_=>unreachable!()}
    }
    LAUNCHES.fetch_add(1,Ordering::Relaxed); finish_dispatch(&c);
}
