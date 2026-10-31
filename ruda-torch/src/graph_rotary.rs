use super::*;

#[ruda(launch)]
fn rotate<F:Float,V:Size>(input:&Tensor<F>,cos:&Tensor<F>,sin:&Tensor<F>,out:&mut Tensor<F>,
    #[comptime] shape:Vec<usize>,#[comptime] half:usize,#[comptime] subtract:bool)
{
    let p=ABSOLUTE_POS as usize;
    if p<out.len()/(2*V::value()) {
        let groups=comptime!(half/V::value());
        let base=(p/groups)*(2*half)+(p%groups)*V::value();
        let packed=input.slice(0usize,input.len()).into_vectorized().with_vector_size::<V>();
        let lower=packed[base/V::value()];let upper=packed[(base+half)/V::value()];
        let mut first=Vector::<F,V>::new(F::cast_from(0));
        let mut second=Vector::<F,V>::new(F::cast_from(0));
        #[unroll]
        for lane in 0usize..V::value() {
            let lo=base+lane;let hi=lo+half;
            let x=f32::cast_from(lower[lane]);let y=f32::cast_from(upper[lane]);
            let c0=f32::cast_from(cos[crate::graph_fusion::address(cos,lo,shape.clone())]);
            let c1=f32::cast_from(cos[crate::graph_fusion::address(cos,hi,shape.clone())]);
            let s0=f32::cast_from(sin[crate::graph_fusion::address(sin,lo,shape.clone())]);
            let s1=f32::cast_from(sin[crate::graph_fusion::address(sin,hi,shape.clone())]);
            let a=f32::cast_from(F::cast_from(x*c0));let b=f32::cast_from(F::cast_from((-y)*s0));
            let c=f32::cast_from(F::cast_from(y*c1));let d=f32::cast_from(F::cast_from(x*s1));
            if comptime!(subtract) {first[lane]=F::cast_from(a-b);second[lane]=F::cast_from(c-d);}
            else {first[lane]=F::cast_from(a+b);second[lane]=F::cast_from(c+d);}
        }
        let mut destination=out.slice_mut(0usize,out.len()).into_vectorized().with_vector_size::<V>();
        destination[base/V::value()]=first;destination[(base+half)/V::value()]=second;
    }
}

pub(super) fn prepare(client:&ComputeClient<CudaRuntime>,input:&View,cos:&View,sin:&View,out:&View,subtract:bool)
    ->PreparedKernel<CudaRuntime>
{
    let half=input.shape.last().unwrap()/2;
    let vector=if half%4==0 && [input,out].iter().all(|v|v.handle.offset_start.unwrap_or(0)%(4*element_bytes(v.dtype)) as u64==0) {4}else {1};
    let grid=RudaCount::Static(u32::try_from((out.len/(2*vector)).div_ceil(128)).expect("rotary grid overflow"),1,1);
    macro_rules! run {($f:ty,$v:ty)=>{unsafe {rotate::prepare::<$f,CudaRuntime>(client,grid,RudaDim::new_1d(128),<$v as Size>::value(),
        input.arg(),cos.arg(),sin.arg(),out.arg(),input.shape.clone(),half,subtract)}};}
    match (input.dtype,vector) {
        (0,4)=>run!(f32,Const<4>),(1,4)=>run!(f16,Const<4>),(2,4)=>run!(bf16,Const<4>),
        (0,_)=>run!(f32,Const<1>),(1,_)=>run!(f16,Const<1>),(2,_)=>run!(bf16,Const<1>),_=>unreachable!(),
    }
}
