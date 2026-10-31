use super::*;

#[ruda(launch)]
fn fill<F:Float,M:Numeric,V:Size>(input:&Tensor<F>,mask:&Tensor<M>,out:&mut Tensor<F>,value:f32,
    #[comptime] width:usize,#[comptime] groups:usize)
{
    let p=ABSOLUTE_POS as usize;
    if p<out.len()/V::value() {
        let position=p*V::value();let row=position/width;let group=(position%width)/(width/groups);
        let packed=input.slice(0usize,input.len()).into_vectorized().with_vector_size::<V>();
        let mut result=packed[p];
        if mask[row*groups+group]==M::cast_from(0) {result=Vector::<F,V>::new(F::cast_from(value));}
        let mut target=out.slice_mut(0usize,out.len()).into_vectorized().with_vector_size::<V>();target[p]=result;
    }
}

pub(super) fn prepare(client:&ComputeClient<CudaRuntime>,input:&View,mask:&View,out:&View,value:f32)->PreparedKernel<CudaRuntime> {
    let width=input.shape[1];let groups=mask.shape[1];
    let vector=if (width/groups)%4==0 && [input,out].iter().all(|v|v.handle.offset_start.unwrap_or(0)%(4*element_bytes(v.dtype)) as u64==0) {4}else {1};
    let grid=RudaCount::Static(u32::try_from((out.len/vector).div_ceil(128)).expect("group mask grid overflow"),1,1);
    macro_rules! launch {($f:ty,$m:ty,$v:ty)=>{unsafe {fill::prepare::<$f,$m,$v,CudaRuntime>(client,grid,RudaDim::new_1d(128),
        input.arg(),mask.arg(),out.arg(),value,width,groups)}};}
    macro_rules! sized {($f:ty,$m:ty)=>{if vector==4 {launch!($f,$m,Const<4>)}else {launch!($f,$m,Const<1>)}};}
    macro_rules! typed {($f:ty)=>{match mask.dtype {
        0=>sized!($f,f32),1=>sized!($f,f16),2=>sized!($f,bf16),3|8=>sized!($f,u8),
        4=>sized!($f,i64),5=>sized!($f,i32),6=>sized!($f,i16),7=>sized!($f,i8),_=>unreachable!(),
    }};}
    match input.dtype {0=>typed!(f32),1=>typed!(f16),2=>typed!(bf16),_=>unreachable!()}
}
