use super::*;

#[ruda(launch)]
fn reduce_tile<E:Numeric+RudaElement>(input:&Tensor<E>,indices:&Tensor<u32>,out:&mut Tensor<E>,out_indices:&mut Tensor<u32>,
    length:u32,blocks:u32,#[comptime] operation:u32,#[comptime] mask:u32,#[comptime] rank:usize,#[comptime] first:bool)
{
    let task=RUDA_POS_X as usize;let lane=UNIT_POS as usize;
    let row=task/blocks as usize;let tile=task%blocks as usize;let position=tile*256+lane;
    let mut values=SharedMemory::<E>::new(256usize);let mut keys=SharedMemory::<u32>::new(256usize);
    let mut value=E::cast_from(0);let mut key=u32::MAX.runtime();
    if comptime!(operation==7 || operation==9) {value=E::cast_from(1);}
    if position<length as usize {
        if comptime!(first) {
            let mut outer=row;let mut inner=position;let mut offset=0usize;
            #[unroll]
            for axis in 0..rank {
                let d=comptime!(rank-axis-1);let mut coordinate=0usize;
                if comptime!((mask >> d)&1!=0) {coordinate=inner%input.shape(d);inner/=input.shape(d);}
                else {coordinate=outer%input.shape(d);outer/=input.shape(d);}
                offset+=coordinate*input.stride(d);
            }
            value=input[offset];key=u32::cast_from(position);
            if comptime!(operation==6 || operation==7) {value=E::cast_from(value!=E::cast_from(0));}
        } else {let offset=row*length as usize+position;value=input[offset];key=indices[offset];}
    }
    values[lane]=value;keys[lane]=key;sync_ruda();
    let mut stride=128usize;
    while stride>0 {
        if lane<stride {
            let a=values[lane];let b=values[lane+stride];let ai=keys[lane];let bi=keys[lane+stride];
            if comptime!(operation>=6) {
                if comptime!(operation==6) {values[lane]=E::cast_from(a!=E::cast_from(0) || b!=E::cast_from(0));}
                else if comptime!(operation==7) {values[lane]=E::cast_from(a!=E::cast_from(0) && b!=E::cast_from(0));}
                else if comptime!(operation==8) {values[lane]=E::cast_from(u64::cast_from(a)+u64::cast_from(b));}
                else {values[lane]=E::cast_from(u64::cast_from(a)*u64::cast_from(b));}
                if bi<ai {keys[lane]=bi;}
            } else {
                let mut better=b>a;
                if comptime!(operation==1 || operation==3 || operation==5) {better=b<a;}
                if a!=a {better=b!=b && bi<ai;}
                else if b!=b {better=true;}
                else if a==b {better=bi<ai;}
                if bi!=u32::MAX && (ai==u32::MAX || better) {values[lane]=b;keys[lane]=bi;}
            }
        }
        sync_ruda();stride/=2;
    }
    if lane==0 {out[task]=values[0];out_indices[task]=keys[0];}
}

#[ruda(launch)]
fn emit<E:Numeric+RudaElement,O:Numeric+RudaElement>(values:&Tensor<E>,indices:&Tensor<u32>,out:&mut Tensor<O>,#[comptime] index:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        if comptime!(index) {out[p]=O::cast_from(indices[p]);}
        else {out[p]=O::cast_from(values[p]);}
    }
}

fn extent(input:&View,mask:u32,reduced:bool)->usize {
    input.shape.iter().enumerate().filter(|(d,_)|((mask>>*d)&1!=0)==reduced)
        .try_fold(1usize,|n,(_,d)|n.checked_mul(*d)).filter(|&n|n<=u32::MAX as usize).expect("reduction indexing overflow")
}

pub(super) fn validate(values:&[View],ints:&[i64],scalars:&[f32]) {
    assert_eq!(ints.len(),3);assert!(scalars.is_empty());
    let operation=ints[0];assert!((0..=9).contains(&operation));
    assert_eq!(values.len(),if matches!(operation,4|5) {3}else {2});
    let input=&values[0];let rank=input.shape.len();assert!(rank<=8 && input.dtype<=9);
    assert!(ints[1]>=0 && ints[1]<(1i64<<rank) && matches!(ints[2],0|1));
    if ints[1]==0 {assert!(matches!(operation,6|7));}
    let mask=ints[1] as u32;
    if operation<=5 {assert!(extent(input,mask,true)>0,"extremum reduction has no identity for an empty axis");}
    if matches!(operation,2|3) {assert_ne!(input.dtype,3,"arg reductions do not support bool");}
    if operation>=8 {assert!((3..=8).contains(&input.dtype) && (4..=8).contains(&values[1].dtype));}
    let mut shape=Vec::new();
    for (d,&size) in input.shape.iter().enumerate() {
        if mask>>d&1==0 {shape.push(size);}else if ints[2]!=0 {shape.push(1);}
    }
    if shape.is_empty() {shape.push(1);}
    assert_eq!(values[1].shape,shape);
    if operation<8 {
        let dtype=if matches!(operation,2|3) {4}else if operation>=6 {if input.dtype==8 {8}else {3}}else {input.dtype};
        assert_eq!(values[1].dtype,dtype);
    }
    if matches!(operation,4|5) {assert_eq!(mask.count_ones(),1);assert_eq!(values[2].shape,shape);assert_eq!(values[2].dtype,4);}
    assert_eq!(extent(input,mask,false),values[1].len);
}

fn allocate(c:&ComputeClient<CudaRuntime>,shape:Vec<usize>,dtype:u32)->View {
    let len=shape.iter().try_fold(1usize,|n,&d|n.checked_mul(d)).expect("reduction workspace overflow");
    View::packed(c.empty(len.max(1).checked_mul(element_bytes(dtype)).expect("reduction workspace bytes overflow")),shape,dtype)
}

fn stages<E:Numeric+RudaElement>(c:&ComputeClient<CudaRuntime>,values:&[View],operation:u32,mask:u32,
    nodes:&mut Vec<PreparedKernel<CudaRuntime>>,scratch:&mut Vec<View>,input:View)
{
    let rows=values[1].len;let mut length=extent(&input,mask,true);let mut first=true;
    let mut current=input;let mut indices:Option<View>=None;
    loop {
        let blocks=length.div_ceil(256).max(1);let count=RudaCount::Static(u32::try_from(rows*blocks).expect("reduction grid overflow"),1,1);
        let next=allocate(c,vec![rows,blocks],current.dtype);let keys=allocate(c,vec![rows,blocks],5);
        nodes.push(reduce_tile::prepare::<E,CudaRuntime>(c,count,RudaDim::new_1d(256),current.arg(),
            indices.as_ref().unwrap_or(&keys).arg(),next.arg(),keys.arg(),length as u32,blocks as u32,operation,mask,current.shape.len(),first));
        current=View::packed(next.handle.clone(),next.shape.clone(),next.dtype);
        indices=Some(View::packed(keys.handle.clone(),keys.shape.clone(),keys.dtype));scratch.push(next);scratch.push(keys);
        if blocks==1 {break;}
        length=blocks;first=false;
    }
    let count=RudaCount::Static(u32::try_from(rows.div_ceil(128)).unwrap(),1,1);let keys=indices.as_ref().unwrap();
    let index=matches!(operation,2|3);
    macro_rules! write {($o:ty,$out:expr,$index:expr)=>{emit::prepare::<E,$o,CudaRuntime>(c,count.clone(),RudaDim::new_1d(128),current.arg(),keys.arg(),$out.arg(),$index)};}
    nodes.push(if index {write!(i64,values[1],true)}else if matches!(operation,6|7) {write!(u8,values[1],false)}else {write!(E,values[1],false)});
    if matches!(operation,4|5) {nodes.push(write!(i64,values[2],true));}
}

pub(super) fn prepare(c:&ComputeClient<CudaRuntime>,values:&[View],ints:&[i64])->(Vec<PreparedKernel<CudaRuntime>>,Vec<View>) {
    let mut nodes=Vec::new();let mut scratch=Vec::new();if values[1].len==0 {return (nodes,scratch);}
    let operation=ints[0] as u32;let mask=ints[1] as u32;
    let source=&values[0];let dtype=if operation>=8 {values[1].dtype}else {source.dtype};
    let input=if source.dtype==dtype {View::packed(source.handle.clone(),source.shape.clone(),dtype)}else {
        let converted=allocate(c,source.shape.clone(),dtype);
        if source.len>0 {nodes.push(sequence_scan::cast(c,source,&converted));}
        let result=View::packed(converted.handle.clone(),converted.shape.clone(),dtype);scratch.push(converted);result
    };
    macro_rules! run {($e:ty)=>{stages::<$e>(c,values,operation,mask,&mut nodes,&mut scratch,input)};}
    match dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),3|8=>run!(u8),4=>run!(i64),5=>run!(i32),6=>run!(i16),7=>run!(i8),9=>run!(f64),_=>unreachable!()}
    (nodes,scratch)
}
