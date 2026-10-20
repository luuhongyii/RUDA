use super::*;

#[ruda(launch)]
fn tiles<E:Numeric+RudaElement,A:Numeric+RudaElement>(input:&Tensor<E>,out:&mut Tensor<A>,totals:&mut Tensor<A>,
    length:u32,inner:u32,blocks:u32,#[comptime] save_totals:bool)
{
    let task=RUDA_POS_X as usize; let lane=UNIT_POS as usize;
    let row=task/blocks as usize; let block=task%blocks as usize; let position=block*256+lane;
    let offset=(row/inner as usize*length as usize+position)*inner as usize+row%inner as usize;
    let mut shared=SharedMemory::<A>::new(256usize);
    let mut value=A::cast_from(0);
    if position<length as usize {value=A::cast_from(input[offset]);}
    shared[lane]=value;sync_ruda();
    let mut step=1usize;
    while step<256usize {
        let mut previous=A::cast_from(0);
        if lane>=step {previous=shared[lane-step];}
        sync_ruda();
        if lane>=step {shared[lane]=shared[lane]+previous;}
        sync_ruda();step*=2;
    }
    if position<length as usize {out[offset]=shared[lane];}
    if comptime!(save_totals) {if lane==255 {totals[task]=shared[lane];}}
}

#[ruda(launch)]
fn carry<A:Numeric+RudaElement>(out:&mut Tensor<A>,sums:&Tensor<A>,length:u32,inner:u32,blocks:u32) {
    let i=ABSOLUTE_POS as usize;
    if i<out.len() {
        let position=i/inner as usize%length as usize;let block=position/256;
        if block>0 {
            let row=i/(inner as usize*length as usize)*inner as usize+i%inner as usize;
            out[i]=sums[row*blocks as usize+block-1]+out[i];
        }
    }
}

#[ruda(launch)]
fn flip<E:Numeric+RudaElement>(input:&Tensor<E>,out:&mut Tensor<E>,#[comptime] mask:u32) {
    let i=ABSOLUTE_POS as usize;
    if i<out.len() {
        let mut rest=i;let mut d=input.rank();let mut source=0usize;
        while d>0 {
            d-=1;let mut coordinate=rest%input.shape(d);rest/=input.shape(d);
            if (mask >> u32::cast_from(d)) & 1u32 != 0 {coordinate=input.shape(d)-1-coordinate;}
            source+=coordinate*input.stride(d);
        }
        out[i]=input[source];
    }
}

#[ruda(launch)]
fn concatenate<E:Numeric+RudaElement>(input:&Tensor<E>,out:&mut Tensor<E>,inner:u32,width:u32,total:u32,start:u32) {
    let i=ABSOLUTE_POS as usize;
    if i<input.len() {
        let span=inner as usize*width as usize;
        out[i/span*(total as usize*inner as usize)+i%span+start as usize*inner as usize]=input[i];
    }
}

pub(super) fn prepare_cat(c:&ComputeClient<CudaRuntime>,values:&[View],axis:usize)->Vec<PreparedKernel<CudaRuntime>> {
    let (out,inputs)=values.split_last().expect("cat needs output");let rank=out.shape.len();assert!(axis<rank);
    let inner:usize=out.shape[axis+1..].iter().product();let mut start=0usize;let mut nodes=Vec::new();
    for input in inputs {
        assert_eq!(input.dtype,out.dtype);
        if input.shape==[0] {continue;}
        assert_eq!(input.shape.len(),rank);
        for d in 0..rank {if d!=axis {assert_eq!(input.shape[d],out.shape[d]);}}
        let width=input.shape[axis];assert!(start.checked_add(width).is_some_and(|end|end<=out.shape[axis]));
        if input.len>0 {
            macro_rules! run {($e:ty)=>{concatenate::prepare::<$e,CudaRuntime>(c,grid(input.len),RudaDim::new_1d(128),input.arg(),out.arg(),
                inner as u32,width as u32,out.shape[axis] as u32,start as u32)};}
            nodes.push(match out.dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),3|8|12|13=>run!(u8),4=>run!(i64),
                5=>run!(i32),6=>run!(i16),7=>run!(i8),9=>run!(f64),_=>unreachable!()});
        }
        start+=width;
    }
    assert_eq!(start,out.shape[axis]);nodes
}

fn grid(n:usize)->RudaCount {RudaCount::Static(u32::try_from(n.div_ceil(128)).expect("scan grid overflow"),1,1)}
fn allocate(c:&ComputeClient<CudaRuntime>,shape:Vec<usize>,dtype:u32)->View {
    let n=shape.iter().try_fold(1usize,|n,&d|n.checked_mul(d)).expect("scan workspace overflow");
    View::packed(c.empty(n.max(1).checked_mul(element_bytes(dtype)).expect("scan workspace bytes overflow")),shape,dtype)
}
pub(super) fn cast(c:&ComputeClient<CudaRuntime>,input:&View,out:&View)->PreparedKernel<CudaRuntime> {
    macro_rules! destination {($s:ty)=>{{
        macro_rules! run {($d:ty)=>{kernels::convert::prepare::<$s,$d,CudaRuntime>(c,grid(out.len),RudaDim::new_1d(128),input.arg(),out.arg())};}
        match out.dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),3|8=>run!(u8),4=>run!(i64),5=>run!(i32),6=>run!(i16),7=>run!(i8),9=>run!(f64),_=>unreachable!()}
    }};}
    match input.dtype {0=>destination!(f32),1=>destination!(f16),2=>destination!(bf16),3|8=>destination!(u8),
        4=>destination!(i64),5=>destination!(i32),6=>destination!(i16),7=>destination!(i8),9=>destination!(f64),_=>unreachable!()}
}

fn scan<E:Numeric+RudaElement,A:Numeric+RudaElement>(c:&ComputeClient<CudaRuntime>,input:&View,out:&View,
    axis:usize,accum_dtype:u32,nodes:&mut Vec<PreparedKernel<CudaRuntime>>,scratch:&mut Vec<View>)
{
    let length=input.shape[axis];let inner:usize=input.shape[axis+1..].iter().product();
    let lines=input.len/length;let blocks=length.div_ceil(256);
    let count=RudaCount::Static(u32::try_from(lines*blocks).expect("scan tile grid overflow"),1,1);
    if blocks==1 {
        nodes.push(tiles::prepare::<E,A,CudaRuntime>(c,count,RudaDim::new_1d(256),input.arg(),out.arg(),out.arg(),
            length as u32,inner as u32,1,false));return;
    }
    let totals=allocate(c,vec![lines,blocks],accum_dtype);let sums=allocate(c,totals.shape.clone(),accum_dtype);
    nodes.push(tiles::prepare::<E,A,CudaRuntime>(c,count,RudaDim::new_1d(256),input.arg(),out.arg(),totals.arg(),
        length as u32,inner as u32,blocks as u32,true));
    scan::<A,A>(c,&totals,&sums,1,accum_dtype,nodes,scratch);
    nodes.push(carry::prepare::<A,CudaRuntime>(c,grid(out.len),RudaDim::new_1d(128),out.arg(),sums.arg(),
        length as u32,inner as u32,blocks as u32));
    scratch.push(totals);scratch.push(sums);
}

pub(super) fn prepare(c:&ComputeClient<CudaRuntime>,input:&View,out:&View,kind:u32,parameter:i64)
    ->(Vec<PreparedKernel<CudaRuntime>>,Vec<View>)
{
    assert_eq!(input.shape,out.shape);let mut nodes=Vec::new();let mut scratch=Vec::new();
    if input.len==0 {return (nodes,scratch);}
    if kind==15 {
        assert_eq!(input.dtype,out.dtype);
        macro_rules! run {($e:ty)=>{flip::prepare::<$e,CudaRuntime>(c,grid(out.len),RudaDim::new_1d(128),input.arg(),out.arg(),parameter as u32)};}
        nodes.push(match input.dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),3|8|12|13=>run!(u8),4=>run!(i64),
            5=>run!(i32),6=>run!(i16),7=>run!(i8),9=>run!(f64),_=>unreachable!()});return (nodes,scratch);
    }
    let converted=if input.dtype!=out.dtype {
        let value=allocate(c,input.shape.clone(),out.dtype);nodes.push(cast(c,input,&value));Some(value)
    } else {None};
    let source=converted.as_ref().unwrap_or(input);
    let axis=usize::try_from(parameter).unwrap();assert!(axis<input.shape.len());
    let accumulation=if matches!(out.dtype,1|2) {Some(allocate(c,out.shape.clone(),0))} else {None};
    let result=accumulation.as_ref().unwrap_or(out);
    macro_rules! run {($e:ty,$a:ty,$dtype:expr)=>{scan::<$e,$a>(c,source,result,axis,$dtype,&mut nodes,&mut scratch)};}
    match out.dtype {0=>run!(f32,f32,0),1=>run!(f16,f32,0),2=>run!(bf16,f32,0),4=>run!(i64,i64,4),5=>run!(i32,i32,5),
        6=>run!(i16,i16,6),7=>run!(i8,i8,7),8=>run!(u8,u8,8),9=>run!(f64,f64,9),_=>unreachable!()}
    if let Some(value)=accumulation {nodes.push(cast(c,&value,out));scratch.push(value);}
    if let Some(value)=converted {scratch.push(value);}
    (nodes,scratch)
}
