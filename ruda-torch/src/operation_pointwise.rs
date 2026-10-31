use super::*;
use super::graph_index::broadcast_index;

#[ruda(launch)]
fn integer<E:Numeric+RudaElement,O:Numeric+RudaElement>(a:&Tensor<E>,b:&Tensor<E>,out:&mut Tensor<O>,
    invalid:&mut Tensor<Atomic<u32>>,rhs:i64,alpha:i64,#[comptime] operation:u32,#[comptime] mode:u32,
    #[comptime] boolean:bool)
{
    let p=ABSOLUTE_POS as usize;
    if p>=out.len() {terminate!();}
    let x=i64::cast_from(a[broadcast_index(a,out,p)]);
    let mut y=i64::cast_from(E::cast_from(rhs));
    if comptime!(mode==0) {y=i64::cast_from(b[broadcast_index(b,out,p)]);}
    let mut result=0i64;
    if comptime!(operation==0) {
        result=i64::cast_from(u64::cast_from(x)+u64::cast_from(alpha)*u64::cast_from(y));
        if comptime!(boolean) {result=i64::cast_from(x!=0i64 || (alpha!=0i64 && y!=0i64));}
    } else if comptime!(operation==1) {result=i64::cast_from(u64::cast_from(x)-u64::cast_from(alpha)*u64::cast_from(y));}
    else if comptime!(operation==2) {result=i64::cast_from(u64::cast_from(x)*u64::cast_from(y));}
    else if comptime!(operation>=3 && operation<=6) {
        if y==0i64 {invalid[0].store(1u32);out[p]=O::cast_from(0);terminate!();}
        let mut ax=u64::cast_from(x);let mut ay=u64::cast_from(y);
        if x<0i64 {ax=0u64-ax;}if y<0i64 {ay=0u64-ay;}
        let remainder=ax%ay;let mut quotient=ax/ay;
        if (x<0i64)!=(y<0i64) {quotient=0u64-quotient;}
        let mut rem=remainder;if x<0i64 {rem=0u64-rem;}
        if comptime!(operation<=4) {
            if comptime!(operation==3) {if remainder!=0u64 && (x<0i64)!=(y<0i64) {quotient-=1u64;}}
            result=i64::cast_from(quotient);
        } else {
            if comptime!(operation==6) {if remainder!=0u64 && (x<0i64)!=(y<0i64) {rem+=u64::cast_from(y);}}
            result=i64::cast_from(rem);
        }
    } else if comptime!(operation==7) {result=x&y;}
    else if comptime!(operation==8) {result=x|y;}
    else if comptime!(operation==9) {result=x^y;}
    else if comptime!(operation==10) {result=i64::cast_from(0u64-u64::cast_from(x));}
    else if comptime!(operation==11) {result=x;if x<0i64 {result=i64::cast_from(0u64-u64::cast_from(x));}}
    else if comptime!(operation==12) {if x>0i64 {result=1i64;}else if x<0i64 {result=-1i64;}}
    else if comptime!(operation==13) {
        result=x ^ -1i64;if comptime!(boolean) {result=i64::cast_from(x==0i64);}
    } else if comptime!(operation==14) {result=i64::cast_from(x==y);}
    else if comptime!(operation==15) {result=i64::cast_from(x!=y);}
    else if comptime!(operation==16) {result=i64::cast_from(x<y);}
    else if comptime!(operation==17) {result=i64::cast_from(x<=y);}
    else if comptime!(operation==18) {result=i64::cast_from(x>y);}
    else if comptime!(operation==19) {result=i64::cast_from(x>=y);}
    else if comptime!(operation==20) {result=x;if y<x {result=y;}}
    else if comptime!(operation==21) {result=x;if y>x {result=y;}}
    else if comptime!(operation==22) {result=i64::cast_from(u64::cast_from(y)-u64::cast_from(alpha)*u64::cast_from(x));}
    out[p]=O::cast_from(result);
}

#[ruda(launch)]
fn activation<F:Float+RudaElement>(input:&Tensor<F>,other:&Tensor<F>,out:&mut Tensor<F>,
    parameter:f32,limit:f32,#[comptime] operation:u32)
{
    let p=ABSOLUTE_POS as usize;
    if p>=out.len() {terminate!();}
    let x=f32::cast_from(input[broadcast_index(input,out,p)]);
    let y=f32::cast_from(other[broadcast_index(other,out,p)]);
    out[p]=F::cast_from(activation_value(x,y,parameter,limit,operation));
}

#[ruda]
pub(super) fn activation_value(x:f32,y:f32,parameter:f32,limit:f32,#[comptime] operation:u32)->f32 {
    let mut result=x;
    if comptime!(operation<=1) {
        let scaled=x*parameter;
        if comptime!(operation==0) {if !(scaled>limit) {result=scaled.exp().log1p()/parameter;}}
        else {
            let dy=y;result=dy;
            if !(scaled>limit) {let z=scaled.exp();result=dy*z/(z+1.0f32);}
        }
    } else if comptime!(operation>=2 && operation<=5) {
        if comptime!(operation==2 || operation==4) {
            let cdf=0.5f32*(1.0f32+(x*0.7071067811865476f32).erf());
            result=x*cdf;
            if comptime!(operation==4) {result=y*(cdf+x*((-0.5f32*x*x).exp()*0.3989422804014327f32));}
        } else {
            let square=x*x;let inner=0.7978845608028654f32*(x+0.044715f32*square*x);let t=inner.tanh();
            result=0.5f32*x*(1.0f32+t);
            if comptime!(operation==5) {
                let derivative=0.5f32*(1.0f32+t)+0.5f32*x*(1.0f32-t*t)*0.7978845608028654f32*(1.0f32+3.0f32*0.044715f32*square);
                result=y*derivative;
            }
        }
    } else if comptime!(operation==6 || operation==7) {
        if comptime!(operation==7) {result=y;}
        if !(x>0.0f32) {result*=parameter;}
    } else if comptime!(operation>=8 && operation<=10) {
        if x==x {
            if comptime!(operation==8 || operation==10) {result=result.max(parameter);}
            if comptime!(operation==9) {result=result.min(parameter);}
            if comptime!(operation==10) {result=result.min(limit);}
        }
    } else if comptime!(operation==11) {result=x.min(0.0f32)-(-x.abs()).exp().log1p();}
    else if comptime!(operation==12 || operation==13) {
        if x!=x {result=x;}else if y!=y {result=y;}
        else if comptime!(operation==12) {result=x.min(y);}else {result=x.max(y);}
    } else if comptime!(operation==14 || operation==15) {
        let activated=x.exp().log1p().tanh();result=x*activated;
        if comptime!(operation==15) {let sigmoid=1.0f32/(1.0f32+(-x).exp());result=y*(activated+x*sigmoid*(1.0f32-activated*activated));}
    }
    result
}

fn grid(n:usize)->RudaCount {RudaCount::Static(u32::try_from(n.div_ceil(128)).unwrap(),1,1)}

#[ruda(launch)]
fn activation_vector<F:Float+RudaElement,V:Size>(input:&Tensor<F>,other:&Tensor<F>,out:&mut Tensor<F>,
    parameter:f32,limit:f32,#[comptime] operation:u32,#[comptime] left_scalar:bool,#[comptime] right_scalar:bool)
{
    let p=ABSOLUTE_POS as usize;
    if p<out.len()/V::value() {
        let mut x=Vector::<F,V>::empty();
        let mut y=Vector::<F,V>::empty();
        if comptime!(left_scalar) {x=Vector::new(input[0usize]);}
        else {x=input.slice(0usize,input.len()).into_vectorized().with_vector_size::<V>()[p];}
        if comptime!(right_scalar) {y=Vector::new(other[0usize]);}
        else {y=other.slice(0usize,other.len()).into_vectorized().with_vector_size::<V>()[p];}
        let mut value=Vector::<F,V>::empty();
        #[unroll] for lane in 0usize..V::value() {
            value[lane]=F::cast_from(activation_value(f32::cast_from(x[lane]),f32::cast_from(y[lane]),parameter,limit,operation));
        }
        let mut target=out.slice_mut(0usize,out.len()).into_vectorized().with_vector_size::<V>();
        target[p]=value;
    }
}

pub(super) fn validate(values:&[View],kind:u32,ints:&[i64]) {
    assert!((2..=3).contains(&values.len()));let (out,inputs)=values.split_last().unwrap();
    let operation=ints[0];
    if kind==17 {
        assert!((0..=22).contains(&operation) && (0..=2).contains(&ints[1]) && (3..=8).contains(&ints[4]));
        assert!(inputs.iter().all(|v|(3..=8).contains(&v.dtype)));
        assert_eq!(inputs.len(),if ints[1]==0 {2}else {1});
        assert_eq!((10..=13).contains(&operation),ints[1]==2);
        assert_eq!(out.dtype,if (14..=19).contains(&operation) {3}else {ints[4] as u32});
        if ints[4]==3 {assert!(matches!(operation,0|2|7|8|9|13|14..=21));}
    } else {
        assert!((0..=15).contains(&operation));assert!(values.iter().all(|v|v.dtype<=2));
        assert_eq!(inputs.len(),if matches!(operation,1|4|5|7|12|13|15) {2}else {1});
        if !matches!(operation,12|13) {for v in values {assert_eq!(v.shape,out.shape);assert_eq!(v.dtype,out.dtype);}}
    }
    let rank=inputs.iter().map(|v|v.shape.len()).max().unwrap();assert_eq!(out.shape.len(),rank);
    let mut shape=vec![1;rank];
    for input in inputs {
        for (i,&dim) in input.shape.iter().enumerate() {
            let axis=rank-input.shape.len()+i;
            assert!(shape[axis]==1 || dim==1 || shape[axis]==dim,"pointwise broadcast mismatch");
            if shape[axis]==1 {shape[axis]=dim;}
        }
    }
    assert_eq!(out.shape,shape);
}
fn converted(c:&ComputeClient<CudaRuntime>,v:&View,dtype:u32,nodes:&mut Vec<PreparedKernel<CudaRuntime>>,scratch:&mut Vec<View>)->View {
    if v.dtype==dtype {return View::packed(v.handle.clone(),v.shape.clone(),dtype);}
    let target=View::packed(c.empty(v.len.max(1).checked_mul(element_bytes(dtype)).expect("pointwise workspace overflow")),v.shape.clone(),dtype);
    nodes.push(sequence_scan::cast(c,v,&target));
    scratch.push(View::packed(target.handle.clone(),target.shape.clone(),dtype));target
}

pub(super) fn prepare(c:&ComputeClient<CudaRuntime>,values:&[View],kind:u32,ints:&[i64],scalars:&[f32])
    ->(Vec<PreparedKernel<CudaRuntime>>,Vec<View>)
{
    let (out,inputs)=values.split_last().unwrap();let mut nodes=Vec::new();let mut scratch=Vec::new();
    if out.len==0 {return (nodes,scratch);}
    let operation=ints[0] as u32;
    if kind==17 {
        let checked=(3..=6).contains(&operation);
        let invalid=View::packed(c.empty(4),vec![1],5);
        if checked {
            let bytes=invalid.byte_view();nodes.push(kernels::fill_bytes::prepare::<CudaRuntime>(c,grid(bytes.len),RudaDim::new_1d(128),bytes.arg(),0u32,0u32));
        }
        scratch.push(invalid);
        let dtype=ints[4] as u32;
        let a=converted(c,&inputs[0],dtype,&mut nodes,&mut scratch);
        let b=if inputs.len()==2 {converted(c,&inputs[1],dtype,&mut nodes,&mut scratch)}else {View::packed(a.handle.clone(),a.shape.clone(),dtype)};
        macro_rules! run {($e:ty,$o:ty)=>{integer::prepare::<$e,$o,CudaRuntime>(c,grid(out.len),RudaDim::new_1d(128),
            a.arg(),b.arg(),out.arg(),scratch[0].arg(),ints[2],ints[3],operation,ints[1] as u32,dtype==3)};}
        macro_rules! data {($e:ty)=>{if (14..=19).contains(&operation) {run!($e,u8)}else {run!($e,$e)}};}
        nodes.push(match dtype {3|8=>data!(u8),4=>data!(i64),5=>data!(i32),6=>data!(i16),7=>data!(i8),_=>unreachable!()});
    } else {
        let a=converted(c,&inputs[0],out.dtype,&mut nodes,&mut scratch);
        let b=if inputs.len()==2 {converted(c,&inputs[1],out.dtype,&mut nodes,&mut scratch)}else {View::packed(a.handle.clone(),a.shape.clone(),out.dtype)};
        let vectorized=out.len%4==0 && [out,&a,&b].iter().all(|v|v.contiguous()
            && (v.len==1 || v.len==out.len && v.handle.offset_start.unwrap_or(0)%(4*element_bytes(v.dtype)) as u64==0));
        if vectorized {
            let views=[out,&a,&b].into_iter().filter(|v|v.len!=1).collect::<Vec<_>>();
            let width=super::row_cached::memory_vector_size(c,out.len,&views,8).max(4);
            macro_rules! vector {($f:ty,$v:ty)=>{activation_vector::prepare::<$f,CudaRuntime>(c,grid(out.len/width),RudaDim::new_1d(128),<$v as Size>::value(),
                a.arg(),b.arg(),out.arg(),scalars[0],scalars[1],operation,a.len==1,b.len==1)};}
            macro_rules! typed {($f:ty)=>{if width==8 {vector!($f,Const<8>)}else {vector!($f,Const<4>)}};}
            nodes.push(match out.dtype {0=>typed!(f32),1=>typed!(f16),2=>typed!(bf16),_=>unreachable!()});
            return (nodes,scratch);
        }
        macro_rules! run {($f:ty)=>{activation::prepare::<$f,CudaRuntime>(c,grid(out.len),RudaDim::new_1d(128),a.arg(),b.arg(),out.arg(),scalars[0],scalars[1],operation)};}
        nodes.push(match out.dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),_=>unreachable!()});
    }
    (nodes,scratch)
}

pub(super) fn checked_integer(kind:u32,ints:&[i64],values:&[View])->bool {
    kind==17 && (3..=6).contains(&ints[0]) && values.last().unwrap().len>0
}
pub(super) fn check(c:&ComputeClient<CudaRuntime>,scratch:&[View])->Result<(),String> {
    let bytes=c.read_one(scratch[0].handle.clone()).map_err(|e|e.to_string())?;DOWNLOAD.fetch_add(4,Ordering::Relaxed);
    if bytes.len()!=4 {return Err("invalid graph integer division status".into());}
    if u32::from_le_bytes(bytes[..4].try_into().unwrap())!=0 {return Err("integer division by zero".into());}
    Ok(())
}
