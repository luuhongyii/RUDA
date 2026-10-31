use super::*;

#[ruda(launch)]
fn tiles<E:Numeric+RudaElement,A:Numeric+RudaElement>(input:&Tensor<E>,out:&mut Tensor<A>,totals:&mut Tensor<A>,
    length:u32,inner:u32,blocks:u32,#[comptime] save_totals:bool,#[comptime] integer:bool)
{
    let task=RUDA_POS_X as usize; let lane=UNIT_POS as usize;
    let row=task/blocks as usize; let block=task%blocks as usize; let position=block*256+lane;
    let offset=(row/inner as usize*length as usize+position)*inner as usize+row%inner as usize;
    let mut value=A::cast_from(0);
    if position<length as usize {value=A::cast_from(input[offset]);}
    if comptime!(integer) {
        let warp_lane=lane%32usize;let warp=lane/32usize;
        #[unroll] for stage in 0usize..5 {
            let distance=comptime!(1usize<<stage);
            let previous=plane_shuffle_up(value,distance as u32);
            if warp_lane>=distance {value=A::cast_from(u64::cast_from(previous)+u64::cast_from(value));}
        }
        let mut shared=SharedMemory::<A>::new(8usize);
        if warp_lane==31 {shared[warp]=value;}
        sync_ruda();
        if warp==0 {
            let mut prefix=A::cast_from(0);
            if warp_lane<8 {prefix=shared[warp_lane];}
            #[unroll] for stage in 0usize..3 {
                let distance=comptime!(1usize<<stage);
                let previous=plane_shuffle_up(prefix,distance as u32);
                if warp_lane>=distance {prefix=A::cast_from(u64::cast_from(previous)+u64::cast_from(prefix));}
            }
            if warp_lane<8 {shared[warp_lane]=prefix;}
        }
        sync_ruda();
        if warp>0 {value=A::cast_from(u64::cast_from(shared[warp-1])+u64::cast_from(value));}
        if position<length as usize {out[offset]=value;}
        if comptime!(save_totals) {if lane==255 {totals[task]=value;}}
    } else {
    let mut shared=SharedMemory::<A>::new(256usize);
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

#[ruda]
fn scan_registers<A:Numeric>(values:&mut Array<A>,#[comptime] integer:bool) {
    let lane=UNIT_POS as usize%32;
    #[unroll] for stage in 0usize..8 {
        let distance=comptime!(1usize<<stage);
        let mut next=Array::<A>::new(8usize);
        #[unroll] for item in 0usize..8 {
            let value=values[item];let mut previous=A::cast_from(0);
            if comptime!(stage<5) {
                let within=plane_shuffle_up(value,distance as u32);
                let mut before=A::cast_from(0);
                if comptime!(item>0) {before=values[comptime!(item-1)];}
                previous=plane_shuffle(before,((lane+32usize-distance)%32usize) as u32);
                if lane>=distance {previous=within;}
            } else if comptime!(item>=distance/32) {previous=values[comptime!(item-distance/32)];}
            let mut sum=value;
            if item*32usize+lane>=distance {
                if comptime!(integer) {sum=A::cast_from(u64::cast_from(value)+u64::cast_from(previous));}
                else {sum=value+previous;}
            }
            next[item]=sum;
        }
        #[unroll] for item in 0usize..8 {values[item]=next[item];}
    }
}

#[ruda(launch)]
fn register_tiles<E:Numeric+RudaElement,A:Numeric+RudaElement>(input:&Tensor<E>,out:&mut Tensor<A>,totals:&mut Tensor<A>,
    length:usize,inner:usize,blocks:usize,#[comptime] save_totals:bool,#[comptime] integer:bool)
{
    let task=RUDA_POS_X as usize*4usize+UNIT_POS as usize/32usize;let lane=UNIT_POS as usize%32usize;
    if task<input.len()/length*blocks {
        let row=task/blocks;let block=task%blocks;let mut values=Array::<A>::new(8usize);
        #[unroll] for item in 0usize..8 {
            let position=block*256usize+item*32usize+lane;let mut value=A::cast_from(0);
            if position<length {value=A::cast_from(input[(row/inner*length+position)*inner+row%inner]);}
            values[item]=value;
        }
        scan_registers(&mut values,integer);
        #[unroll] for item in 0usize..8 {
            let position=block*256usize+item*32usize+lane;
            if position<length {out[(row/inner*length+position)*inner+row%inner]=values[item];}
        }
        if comptime!(save_totals) {if lane==31 {totals[task]=values[7usize];}}
    }
}

#[ruda(launch)]
fn column_register_tiles<E:Numeric+RudaElement,A:Numeric+RudaElement>(input:&Tensor<E>,out:&mut Tensor<A>,totals:&mut Tensor<A>,
    length:usize,inner:usize,blocks:usize,#[comptime] save_totals:bool,#[comptime] integer:bool)
{
    let task=RUDA_POS_X as usize;let tid=UNIT_POS as usize;
    let columns=(inner+7usize)/8usize;let tile=task%blocks;let group=task/blocks;
    let outer=group/columns;let first_column=group%columns*8usize;
    let io_column=first_column+tid%8usize;
    let mut shared=SharedMemory::<A>::new(256usize*9usize);
    #[unroll] for item in 0usize..8 {
        let row=tid/8usize+item*32usize;let position=tile*256usize+row;let mut value=A::cast_from(0);
        if position<length && io_column<inner {value=A::cast_from(input[(outer*length+position)*inner+io_column]);}
        shared[row*9usize+tid%8usize]=value;
    }
    sync_ruda();
    let warp=tid/32usize;let lane=tid%32usize;let mut values=Array::<A>::new(8usize);
    #[unroll] for item in 0usize..8 {values[item]=shared[(item*32usize+lane)*9usize+warp];}
    sync_ruda();
    scan_registers(&mut values,integer);
    #[unroll] for item in 0usize..8 {shared[(item*32usize+lane)*9usize+warp]=values[item];}
    if comptime!(save_totals) {
        if lane==31 && first_column+warp<inner {totals[(outer*inner+first_column+warp)*blocks+tile]=values[7usize];}
    }
    sync_ruda();
    #[unroll] for item in 0usize..8 {
        let row=tid/8usize+item*32usize;let position=tile*256usize+row;
        if position<length && io_column<inner {out[(outer*length+position)*inner+io_column]=shared[row*9usize+tid%8usize];}
    }
}

#[ruda(launch)]
fn column_tiles<E:Numeric+RudaElement,A:Numeric+RudaElement>(input:&Tensor<E>,out:&mut Tensor<A>,totals:&mut Tensor<A>,
    length:usize,inner:usize,blocks:usize,#[comptime] save_totals:bool,#[comptime] integer:bool)
{
    let task=RUDA_POS_X as usize;let lane=UNIT_POS as usize;
    let columns=(inner+7usize)/8usize;let tile=task%blocks;let group=task/blocks;
    let outer=group/columns;let column=group%columns*8+lane%8;
    let mut shared=SharedMemory::<A>::new(2048usize);
    #[unroll] for j in 0usize..8 {
        let row=lane/8+j*32;let position=tile*256+row;let mut value=A::cast_from(0);
        if position<length && column<inner {value=A::cast_from(input[(outer*length+position)*inner+column]);}
        shared[row*8+lane%8]=value;
    }
    sync_ruda();
    #[unroll] for stage in 0usize..8 {
        let step=comptime!(1usize<<stage);let mut values=Array::<A>::new(8usize);
        #[unroll] for j in 0usize..8 {
            let row=lane/8+j*32;let at=row*8+lane%8;let mut value=shared[at];
            if row>=step {
                let previous=shared[at-step*8];
                if comptime!(integer) {value=A::cast_from(u64::cast_from(value)+u64::cast_from(previous));}
                else {value+=previous;}
            }
            values[j]=value;
        }
        sync_ruda();
        #[unroll] for j in 0usize..8 {shared[(lane/8+j*32)*8+lane%8]=values[j];}
        sync_ruda();
    }
    #[unroll] for j in 0usize..8 {
        let row=lane/8+j*32;let position=tile*256+row;
        if position<length && column<inner {out[(outer*length+position)*inner+column]=shared[row*8+lane%8];}
    }
    if comptime!(save_totals) {
        if lane<8 && column<inner {totals[(outer*inner+column)*blocks+tile]=shared[255*8+lane];}
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

#[ruda(launch)]
fn concatenate_many<E:Numeric+RudaElement,V:Size>(inputs:&Sequence<Tensor<E>>,out:&mut Tensor<E>,inner:usize,total:usize,
    #[comptime] starts:Vec<usize>)
{
    let p=ABSOLUTE_POS as usize;
    if p<out.len()/V::value() {
        let position=p*V::value();let outer=position/(total*inner);let column=position/inner%total;
        let mut value=Vector::<E,V>::empty();
        #[unroll] for i in 0usize..inputs.len() {
            if column>=comptime!(starts[i]) && column<comptime!(starts[i+1]) {
                let input=inputs.index(i);
                let source=(outer*comptime!(starts[i+1]-starts[i])+column-comptime!(starts[i]))*inner+position%inner;
                let packed=input.slice(0usize,input.len()).into_vectorized().with_vector_size::<V>();
                value=packed[source/V::value()];
            }
        }
        let mut packed=out.slice_mut(0usize,out.len()).into_vectorized().with_vector_size::<V>();packed[p]=value;
    }
}

fn prepare_cat_many(c:&ComputeClient<CudaRuntime>,values:&[View],axis:usize)->Option<Vec<PreparedKernel<CudaRuntime>>> {
    let (out,inputs)=values.split_last()?;
    if inputs.len()>16 || !out.contiguous() {return None;}
    let mut active=Vec::new();let mut starts=vec![0usize];let mut start=0usize;
    for input in inputs {
        assert_eq!(input.dtype,out.dtype);
        if input.shape==[0] {continue;}
        assert_eq!(input.shape.len(),out.shape.len());
        for d in 0..out.shape.len() {if d!=axis {assert_eq!(input.shape[d],out.shape[d]);}}
        if !input.contiguous() {return None;}
        start=start.checked_add(input.shape[axis]).expect("cat dimension overflow");
        if input.len>0 {active.push(input);starts.push(start);}
    }
    assert_eq!(start,out.shape[axis]);
    if out.len==0 {return Some(Vec::new());}
    let inner=out.shape[axis+1..].iter().product::<usize>();
    let vector=inner%4==0 && active.iter().copied().chain(std::iter::once(out))
        .all(|v|v.handle.offset_start.unwrap_or(0)%(4*element_bytes(v.dtype)) as u64==0);
    let width=if vector {4}else {1};
    macro_rules! data {($e:ty)=>{{
        macro_rules! run {($v:ty)=>{concatenate_many::prepare::<$e,$v,CudaRuntime>(c,grid(out.len/width),RudaDim::new_1d(128),
            active.iter().map(|v|v.arg()).collect::<SequenceArg<CudaRuntime,Tensor<$e>>>(),out.arg(),inner,out.shape[axis],starts)};}
        if vector {run!(Const<4>)}else {run!(Const<1>)}
    }};}
    Some(vec![match out.dtype {0=>data!(f32),1=>data!(f16),2=>data!(bf16),3|8|12|13=>data!(u8),4=>data!(i64),
        5=>data!(i32),6=>data!(i16),7=>data!(i8),9=>data!(f64),_=>unreachable!()}])
}

pub(super) fn prepare_cat(c:&ComputeClient<CudaRuntime>,values:&[View],axis:usize)->Vec<PreparedKernel<CudaRuntime>> {
    let (out,inputs)=values.split_last().expect("cat needs output");let rank=out.shape.len();assert!(axis<rank);
    if let Some(nodes)=prepare_cat_many(c,values,axis) {return nodes;}
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
fn reuse_or_allocate(c:&ComputeClient<CudaRuntime>,shape:Vec<usize>,dtype:u32,reuse:&[View],index:usize)->View {
    if let Some(value)=reuse.get(index) {
        assert_eq!(value.shape,shape);assert_eq!(value.dtype,dtype);
        View::packed(value.handle.clone(),shape,dtype)
    } else {allocate(c,shape,dtype)}
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
    axis:usize,accum_dtype:u32,nodes:&mut Vec<PreparedKernel<CudaRuntime>>,scratch:&mut Vec<View>,reuse:&[View])
{
    let length=input.shape[axis];let inner:usize=input.shape[axis+1..].iter().product();
    let lines=input.len/length;let blocks=length.div_ceil(256);
    let count=RudaCount::Static(u32::try_from(lines*blocks).expect("scan tile grid overflow"),1,1);
    let column_count=RudaCount::Static(u32::try_from((lines/inner)*inner.div_ceil(8)*blocks).expect("scan column grid overflow"),1,1);
    let columns=inner>=8 && length>=32;
    let h=&c.properties().hardware;
    let shuffle=h.plane_size_min==32 && h.plane_size_max==32 && c.features().plane.contains(ruda_core::ir::features::Plane::Ops);
    let register_count=RudaCount::Static(u32::try_from((lines*blocks).div_ceil(4)).expect("scan warp grid overflow"),1,1);
    macro_rules! tile {($totals:expr,$save:expr)=>{
        if shuffle && columns {column_register_tiles::prepare::<E,A,CudaRuntime>(c,column_count,RudaDim::new_1d(256),
            input.arg(),out.arg(),$totals.arg(),length,inner,blocks,$save,matches!(accum_dtype,4..=8))}
        else if shuffle {register_tiles::prepare::<E,A,CudaRuntime>(c,register_count,RudaDim::new_1d(128),
            input.arg(),out.arg(),$totals.arg(),length,inner,blocks,$save,matches!(accum_dtype,4..=8))}
        else if columns {column_tiles::prepare::<E,A,CudaRuntime>(c,column_count,RudaDim::new_1d(256),
            input.arg(),out.arg(),$totals.arg(),length,inner,blocks,$save,matches!(accum_dtype,4..=8))}
        else {tiles::prepare::<E,A,CudaRuntime>(c,count,RudaDim::new_1d(256),input.arg(),out.arg(),$totals.arg(),
            length as u32,inner as u32,blocks as u32,$save,matches!(accum_dtype,4..=8))}
    };}
    if blocks==1 {
        nodes.push(tile!(out,false));return;
    }
    let totals=reuse_or_allocate(c,vec![lines,blocks],accum_dtype,reuse,scratch.len());
    let sums=reuse_or_allocate(c,totals.shape.clone(),accum_dtype,reuse,scratch.len()+1);
    scratch.push(View::packed(totals.handle.clone(),totals.shape.clone(),totals.dtype));
    scratch.push(View::packed(sums.handle.clone(),sums.shape.clone(),sums.dtype));
    nodes.push(tile!(totals,true));
    scan::<A,A>(c,&totals,&sums,1,accum_dtype,nodes,scratch,reuse);
    nodes.push(carry::prepare::<A,CudaRuntime>(c,grid(out.len),RudaDim::new_1d(128),out.arg(),sums.arg(),
        length as u32,inner as u32,blocks as u32));
}

pub(super) fn prepare(c:&ComputeClient<CudaRuntime>,input:&View,out:&View,kind:u32,parameter:i64)
    ->(Vec<PreparedKernel<CudaRuntime>>,Vec<View>)
{
    prepare_reusing(c,input,out,kind,parameter,&[])
}

pub(super) fn prepare_reusing(c:&ComputeClient<CudaRuntime>,input:&View,out:&View,kind:u32,parameter:i64,reuse:&[View])
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
        let value=reuse_or_allocate(c,input.shape.clone(),out.dtype,reuse,scratch.len());nodes.push(cast(c,input,&value));
        scratch.push(View::packed(value.handle.clone(),value.shape.clone(),value.dtype));Some(value)
    } else {None};
    let source=converted.as_ref().unwrap_or(input);
    let axis=usize::try_from(parameter).unwrap();assert!(axis<input.shape.len());
    let accumulation=if matches!(out.dtype,1|2) {
        let value=reuse_or_allocate(c,out.shape.clone(),0,reuse,scratch.len());
        scratch.push(View::packed(value.handle.clone(),value.shape.clone(),value.dtype));Some(value)
    } else {None};
    let result=accumulation.as_ref().unwrap_or(out);
    macro_rules! run {($e:ty,$a:ty,$dtype:expr)=>{scan::<$e,$a>(c,source,result,axis,$dtype,&mut nodes,&mut scratch,reuse)};}
    match out.dtype {0=>run!(f32,f32,0),1=>run!(f16,f32,0),2=>run!(bf16,f32,0),4=>run!(i64,i64,4),5=>run!(i32,i32,5),
        6=>run!(i16,i16,6),7=>run!(i8,i8,7),8=>run!(u8,u8,8),9=>run!(f64,f64,9),_=>unreachable!()}
    if let Some(value)=accumulation {nodes.push(cast(c,&value,out));}
    (nodes,scratch)
}
