use super::*;
use super::graph_contract::NodeSpec;
use super::graph_index::broadcast_index;

#[ruda(launch)]
fn full<E:Numeric+RudaElement>(out:&mut Tensor<E>,value:f32,#[comptime] boolean:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        if comptime!(boolean) {out[p]=E::cast_from(value!=0.0f32);}
        else {out[p]=E::cast_from(value);}
    }
}

#[ruda(launch)]
fn integer_range<E:Numeric+RudaElement>(out:&mut Tensor<E>,start:i64,step:i64) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {out[p]=E::cast_from(start+i64::cast_from(p)*step);}
}

#[ruda(launch)]
fn triangular<E:Numeric+RudaElement>(a:&Tensor<E>,out:&mut Tensor<E>,diagonal:i64,#[comptime] upper:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let width=out.shape(out.rank()-1);let height=out.shape(out.rank()-2);
        let offset=i64::cast_from(p%width)-i64::cast_from((p/width)%height);
        let mut keep=offset<=diagonal;
        if comptime!(upper) {keep=offset>=diagonal;}
        if keep {out[p]=a[p];} else {out[p]=E::cast_from(0);}
    }
}

#[ruda(launch)]
fn logical<E:Numeric+RudaElement>(a:&Tensor<E>,b:&Tensor<E>,out:&mut Tensor<u8>,#[comptime] operation:u32) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let x=a[broadcast_index(a,out,p)]!=E::cast_from(0);
        let mut result=false;
        if comptime!(operation==164) {result=!x;}
        else {
            let y=b[broadcast_index(b,out,p)]!=E::cast_from(0);
            if comptime!(operation==165) {result=x && y;}
            else if comptime!(operation==166) {result=x || y;}
            else {result=x!=y;}
        }
        out[p]=u8::cast_from(result);
    }
}

#[ruda(launch)]
fn bitwise<E:Numeric+RudaElement>(a:&Tensor<E>,b:&Tensor<E>,out:&mut Tensor<E>,#[comptime] operation:u32,#[comptime] boolean:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let x=i64::cast_from(a[broadcast_index(a,out,p)]);
        let y=i64::cast_from(b[broadcast_index(b,out,p)]);
        if comptime!(boolean) {
            let mut value=false;
            if comptime!(operation==168) {value=x==0i64;}
            else if comptime!(operation==169) {value=x!=0i64 && y!=0i64;}
            else if comptime!(operation==170) {value=x!=0i64 || y!=0i64;}
            else {value=(x!=0i64)!=(y!=0i64);}
            out[p]=E::cast_from(value);
        } else {
            let mut value=0i64;
            if comptime!(operation==168) {value=x ^ -1i64;}
            else if comptime!(operation==169) {value=x & y;}
            else if comptime!(operation==170) {value=x | y;}
            else {value=x ^ y;}
            out[p]=E::cast_from(value);
        }
    }
}

pub(super) fn prepare(client:&ComputeClient<CudaRuntime>,tensors:&[View],node:&NodeSpec,output:usize)
    ->Option<PreparedKernel<CudaRuntime>> {
    if !(160..=172).contains(&node.op) {return None;}
    let out=&tensors[output];
    let grid=RudaCount::Static(u32::try_from(out.len.div_ceil(128)).expect("factory grid overflow"),1,1);
    if node.op==172 {
        macro_rules! run {($e:ty)=>{kernels::arange_scalar::prepare::<$e,f32,CudaRuntime>(client,grid,RudaDim::new_1d(128),out.arg(),node.scalar,f32::from_bits(node.b))};}
        return Some(match out.dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),_=>unreachable!()});
    }
    if node.op>=162 {
        let a=&tensors[node.a as usize];let b=&tensors[node.b as usize];
        macro_rules! mask {($e:ty)=>{{unsafe {
            if node.op<=163 {triangular::prepare::<$e,CudaRuntime>(client,grid,RudaDim::new_1d(128),a.arg(),out.arg(),node.scalar as i64,node.op==163)}
            else if node.op<=167 {logical::prepare::<$e,CudaRuntime>(client,grid,RudaDim::new_1d(128),a.arg(),b.arg(),out.arg(),node.op)}
            else {bitwise::prepare::<$e,CudaRuntime>(client,grid,RudaDim::new_1d(128),a.arg(),b.arg(),out.arg(),node.op,a.dtype==3)}
        }}};}
        return Some(match a.dtype {0=>mask!(f32),1=>mask!(f16),2=>mask!(bf16),3|8=>mask!(u8),4=>mask!(i64),5=>mask!(i32),6=>mask!(i16),7=>mask!(i8),_=>unreachable!()});
    }
    macro_rules! run {($e:ty)=>{{unsafe {
        if node.op==160 {full::prepare::<$e,CudaRuntime>(client,grid,RudaDim::new_1d(128),out.arg(),node.scalar,out.dtype==3)}
        else {integer_range::prepare::<$e,CudaRuntime>(client,grid,RudaDim::new_1d(128),out.arg(),node.scalar as i64,node.b as i32 as i64)}
    }}};}
    Some(match out.dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),3|8=>run!(u8),4=>run!(i64),5=>run!(i32),6=>run!(i16),7=>run!(i8),_=>unreachable!()})
}
