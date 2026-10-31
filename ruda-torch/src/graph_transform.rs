use super::*;

#[ruda]
fn source<E:Numeric>(input:&Tensor<E>,out:&Tensor<E>,position:usize,
    #[comptime] op:u32,#[comptime] axis:usize,#[comptime] start:usize,#[comptime] step:usize,
    #[comptime] rank:usize,#[comptime] input_rank:usize,#[comptime] flat:bool)->(usize,bool)
{
    let mut address=0usize;let mut inside=true;let mut rest=position;
    if comptime!(op==216 && flat) {
        address=position/step;
    } else {
        #[unroll] for reverse in 0usize..rank {
            let d=comptime!(rank-1-reverse);let coordinate=rest%out.shape(d);rest/=out.shape(d);
            if comptime!(op==215) {
                if comptime!(d+input_rank>=rank) {
                    let sd=comptime!(d+input_rank-rank);
                    address+=(coordinate%input.shape(sd))*input.stride(sd);
                }
            } else {
                let mut mapped=coordinate;
                if comptime!(d==axis) {
                    if comptime!(op==212) {mapped=start+coordinate*step;}
                    else if comptime!(op==216) {mapped=coordinate/step;}
                    else {
                        if coordinate<start {inside=false;}
                        else {
                            let delta=coordinate-start;
                            if delta%step!=0usize || delta/step>=input.shape(d) {inside=false;}
                            else {mapped=delta/step;}
                        }
                    }
                }
                address+=mapped*input.stride(d);
            }
        }
    }
    (address,inside)
}

#[ruda(launch)]
fn transform<E:Numeric,V:Size>(left:&Tensor<E>,part:&Tensor<E>,out:&mut Tensor<E>,
    #[comptime] op:u32,#[comptime] axis:usize,#[comptime] start:usize,#[comptime] step:usize,
    #[comptime] rank:usize,#[comptime] input_rank:usize,#[comptime] flat:bool)
{
    let p=ABSOLUTE_POS as usize;
    if p<out.len()/V::value() {
        let mut value=Vector::<E,V>::new(E::cast_from(0));
        if comptime!(op==214) {
            let location=source(part,out,p*V::value(),op,axis,start,step,rank,input_rank,flat);
            if location.1 {value=part.slice(0usize,part.len()).into_vectorized().with_vector_size::<V>()[location.0/V::value()];}
            else {value=left.slice(0usize,left.len()).into_vectorized().with_vector_size::<V>()[p];}
        } else {
            let location=source(left,out,p*V::value(),op,axis,start,step,rank,input_rank,flat);
            if location.1 {value=left.slice(0usize,left.len()).into_vectorized().with_vector_size::<V>()[location.0/V::value()];}
        }
        let mut packed=out.slice_mut(0usize,out.len()).into_vectorized().with_vector_size::<V>();packed[p]=value;
    }
}

pub(super) fn prepare(client:&ComputeClient<CudaRuntime>,left:&View,right:&View,out:&View,op:u32,scalar:f32)
    ->PreparedKernel<CudaRuntime>
{
    let code=scalar as usize;
    let flat=op==216 && scalar==-1.0;
    let input=if op==214 {right}else {left};
    let axis=if op<=214 {code&7}else if flat || op==215 {0}else {code};
    let start=if op<=214 {(code>>3)&2047}else {0};
    let step=if op<=214 {code>>14}else if op==215 {1}
        else if flat {out.len/input.len}else {out.shape[axis]/input.shape[axis]};
    let rank=out.shape.len();
    let aligned=[left,right,out].iter().all(|v|v.handle.offset_start.unwrap_or(0)%(4*element_bytes(v.dtype)) as u64==0);
    let contiguous_last=if op==215 {input.shape.last().unwrap()%4==0}
        else if flat {step==1}else {axis+1!=rank || step==1 && start%4==0};
    let vector=if aligned && contiguous_last && out.shape.last().unwrap()%4==0
        && input.shape.last().unwrap()%4==0 {4}else {1};
    let grid=RudaCount::Static(u32::try_from((out.len/vector).div_ceil(128)).expect("graph transform grid overflow"),1,1);
    macro_rules! run {($e:ty,$v:ty)=>{unsafe {transform::prepare::<$e,$v,CudaRuntime>(client,grid,RudaDim::new_1d(128),
        left.arg(),right.arg(),out.arg(),op,axis,start,step,rank,input.shape.len(),flat)}};}
    macro_rules! typed {($e:ty)=>{if vector==4 {run!($e,Const<4>)}else {run!($e,Const<1>)}};}
    match out.dtype {0=>typed!(f32),1=>typed!(f16),2=>typed!(bf16),3|8=>typed!(u8),
        4=>typed!(i64),5=>typed!(i32),6=>typed!(i16),7=>typed!(i8),_=>unreachable!()}
}
