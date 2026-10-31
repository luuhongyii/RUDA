use super::*;
use super::graph_contract::NodeSpec;
use std::collections::BTreeMap;

#[ruda]
fn store_value<V:Size>(outputs32:&mut Sequence<Tensor<f32>>,outputs16:&mut Sequence<Tensor<f16>>,
    outputsbf:&mut Sequence<Tensor<bf16>>,position:usize,value:Vector<f32,V>,
    #[comptime] dtype:u32,#[comptime] slot:usize)
{
    if comptime!(dtype==0) {
        let output=outputs32.index_mut(slot);
        let mut packed=output.slice_mut(0usize,output.len()).into_vectorized().with_vector_size::<V>();
        packed[position/V::value()]=value;
    } else if comptime!(dtype==1) {
        let output=outputs16.index_mut(slot);
        let mut packed=output.slice_mut(0usize,output.len()).into_vectorized().with_vector_size::<V>();
        packed[position/V::value()]=Vector::cast_from(value);
    } else {
        let output=outputsbf.index_mut(slot);
        let mut packed=output.slice_mut(0usize,output.len()).into_vectorized().with_vector_size::<V>();
        packed[position/V::value()]=Vector::cast_from(value);
    }
}

#[ruda]
fn load_value<E:Numeric,V:Size>(input:&Tensor<E>,position:usize,#[comptime] shape:Vec<usize>,
    #[comptime] period:usize)->Vector<f32,V> {
    let mut value=Vector::<f32,V>::empty();
    if comptime!(period==1) {value=Vector::new(f32::cast_from(input[0usize]));}
    else if comptime!(period>1) {
        let packed=input.slice(0usize,input.len()).into_vectorized().with_vector_size::<V>();
        value=Vector::cast_from(packed[(position%period)/V::value()]);
    } else {
        #[unroll] for lane in 0usize..V::value() {
            value[lane]=f32::cast_from(input[crate::graph_fusion::address(input,position+lane,shape.clone())]);
        }
    }
    value
}

#[ruda(launch)]
fn fused_softmax<V:Size>(inputs32:&Sequence<Tensor<f32>>,inputs16:&Sequence<Tensor<f16>>,
    inputsbf:&Sequence<Tensor<bf16>>,masks:&Sequence<Tensor<u8>>,
    outputs32:&mut Sequence<Tensor<f32>>,outputs16:&mut Sequence<Tensor<f16>>,
    outputsbf:&mut Sequence<Tensor<bf16>>,rows:usize,
    #[comptime] shape:Vec<usize>,#[comptime] width:usize,#[comptime] threads:usize,#[comptime] periods:Vec<usize>,
    #[comptime] input_types:Vec<u32>,#[comptime] input_slots:Vec<usize>,
    #[comptime] codes:Vec<u32>,#[comptime] left:Vec<usize>,#[comptime] right:Vec<usize>,
    #[comptime] third:Vec<usize>,#[comptime] scalars:Vec<u32>,
    #[comptime] output_types:Vec<u32>,#[comptime] output_slots:Vec<usize>,#[comptime] stores:Vec<bool>,
    #[comptime] softmax:usize,#[comptime] source:usize,#[comptime] logarithmic:bool)
{
    let mut row=RUDA_POS_X as usize;
    let thread=UNIT_POS as usize%threads;
    if comptime!(threads==32) {row=row*4+UNIT_POS as usize/32;}
    if row<rows {
        let chunks=comptime!(width.div_ceil(threads*V::value()));
        let mut cached=Array::<Vector<f32,V>>::new(chunks);
        let mut scratch=SharedMemory::<f32>::new(2*(threads/32));
        let mut maximum=f32::new(f32::NEG_INFINITY);
        #[unroll] for j in 0usize..chunks {
            let column=(thread+j*threads)*V::value();
            let mut value=Vector::<f32,V>::new(f32::new(f32::NEG_INFINITY));
                if column<width {
                    let position=row*width+column;
                    let mut values=Array::<Vector<f32,V>>::new(input_types.len()+softmax);
                    #[unroll] for i in 0usize..input_types.len() {
                        let slot=comptime!(input_slots[i]);
                        if comptime!(input_types[i]==0) {
                            let input=inputs32.index(slot);
                            values[i]=load_value::<f32,V>(input,position,shape.clone(),comptime!(periods[i]));
                        } else if comptime!(input_types[i]==1) {
                            let input=inputs16.index(slot);
                            values[i]=load_value::<f16,V>(input,position,shape.clone(),comptime!(periods[i]));
                        } else if comptime!(input_types[i]==2) {
                            let input=inputsbf.index(slot);
                            values[i]=load_value::<bf16,V>(input,position,shape.clone(),comptime!(periods[i]));
                        } else {
                            let input=masks.index(slot);
                            values[i]=load_value::<u8,V>(input,position,shape.clone(),comptime!(periods[i]));
                        }
                    }
                    #[unroll] for i in 0usize..softmax {
                        let x=values[comptime!(left[i])];let y=values[comptime!(right[i])];
                        let scalar=f32::from_bits(comptime!(scalars[i]));let mut next=Vector::<f32,V>::empty();
                        #[unroll] for lane in 0usize..V::value() {
                            if comptime!(codes[i]==146) {
                                next[lane]=x[lane];if y[lane]!=0.0f32 {next[lane]=crate::graph_fusion::rounded(scalar,comptime!(output_types[i]));}
                            } else if comptime!(codes[i]==147) {
                                let other=values[comptime!(third[i])];next[lane]=other[lane];if x[lane]!=0.0f32 {next[lane]=y[lane];}
                            } else {
                                next[lane]=crate::graph_fusion::evaluate(x[lane],y[lane],scalar,comptime!(codes[i]),comptime!(output_types[i]));
                            }
                        }
                        values[input_types.len()+i]=next;
                        if comptime!(stores[i]) {
                            store_value(outputs32,outputs16,outputsbf,position,next,comptime!(output_types[i]),comptime!(output_slots[i]));
                        }
                    }
                    value=values[source];
                }
            cached[j]=value;
            #[unroll] for lane in 0usize..V::value() {
                if value[lane]>maximum || value[lane]!=value[lane] {maximum=value[lane];}
            }
        }
        maximum=crate::row_cached::block_reduce(maximum,&mut scratch,threads,true,0usize);
        let mut local=0.0f32;
        #[unroll] for j in 0usize..chunks {
            let column=(thread+j*threads)*V::value();
            if column<width {
                let previous=cached[j];let mut next=Vector::<f32,V>::empty();
                #[unroll] for lane in 0usize..V::value() {
                    let shifted=previous[lane]-maximum;
                    let exponential=crate::softmax::exp_fast(shifted);
                    local+=exponential;
                    if comptime!(logarithmic) {next[lane]=shifted;}
                    else {next[lane]=exponential;}
                }
                cached[j]=next;
            }
        }
        let total=crate::row_cached::block_reduce(local,&mut scratch,threads,false,1usize);
        let mut log_total=0.0f32;if comptime!(logarithmic) {log_total=total.ln();}
        #[unroll] for j in 0usize..chunks {
            let column=(thread+j*threads)*V::value();
                if column<width {
                    let position=row*width+column;let previous=cached[j];let mut value=Vector::<f32,V>::empty();
                    #[unroll] for lane in 0usize..V::value() {
                        if comptime!(logarithmic) {value[lane]=previous[lane]-log_total;}else {value[lane]=previous[lane]/total;}
                    }
                    #[unroll] for i in softmax..output_types.len() {
                        #[unroll] for lane in 0usize..V::value() {value[lane]=crate::graph_fusion::rounded(value[lane],comptime!(output_types[i]));}
                        if comptime!(stores[i]) {
                            store_value(outputs32,outputs16,outputsbf,position,value,comptime!(output_types[i]),comptime!(output_slots[i]));
                        }
                    }
                }
        }
    }
}

pub(super) fn prepare(c:&ComputeClient<CudaRuntime>,tensors:&[View],nodes:&[NodeSpec],inputs:usize,start:usize)
    ->Option<(usize,PreparedKernel<CudaRuntime>)>
{
    if std::env::var("RUDA_TORCH_SOFTMAX").is_ok_and(|value|value!="auto") {return None;}
    let first=&tensors[inputs+start];
    let width=*first.shape.last()?;
    if first.len==0 || first.dtype>2 || !first.contiguous() || !(32..=32768).contains(&width) {return None;}
    let mut softmax=start;
    loop {
        let n=nodes.get(softmax)?;let out=&tensors[inputs+softmax];
        if out.shape!=first.shape || out.dtype>2 || !out.contiguous() {return None;}
        if matches!(n.op,102|103) {break;}
        if softmax-start>=8 || !(super::graph_fusion::supported(n.op) || matches!(n.op,146|147)) {return None;}
        softmax+=1;
    }
    if nodes[softmax].scalar as usize+1!=first.shape.len() {return None;}
    let mut end=softmax+1;
    while end<nodes.len() && end-softmax<=4 && nodes[end].op==114 && nodes[end].a as usize==inputs+end-1
        && tensors[inputs+end].shape==first.shape && tensors[inputs+end].dtype<=2 && tensors[inputs+end].contiguous() {end+=1;}
    if end-start<2 {return None;}
    let mut external=Vec::new();let mut ids=BTreeMap::new();
    for n in &nodes[start..=softmax] {
        let mut operands=vec![n.a as usize,n.b as usize];
        if n.op==147 {operands.push(n.scalar as usize);}
        for id in operands {
            let v=tensors.get(id)?;
            if v.dtype>3 {return None;}
            if id<inputs+start && !ids.contains_key(&id) {ids.insert(id,external.len());external.push(id);}
        }
    }
    for i in start..softmax {ids.insert(inputs+i,external.len()+i-start);}
    let mut unique=Vec::<usize>::new();let mut slots=Vec::new();let mut output_slots=Vec::new();let mut counts=[0usize;3];
    for i in inputs+start..inputs+end {
        if external.iter().any(|&j|super::graph_fusion::overlaps(&tensors[i],&tensors[j])) {return None;}
        let previous=unique.iter().position(|&j|super::graph_fusion::overlaps(&tensors[i],&tensors[j]));
        if let Some(p)=previous {
            let old=&tensors[unique[p]];let new=&tensors[i];
            if old.dtype!=new.dtype || old.shape!=new.shape || old.handle.offset_start!=new.handle.offset_start {return None;}
            output_slots.push(slots[p]);
        } else {
            let dtype=tensors[i].dtype as usize;let slot=counts[dtype];counts[dtype]+=1;
            unique.push(i);slots.push(slot);output_slots.push(slot);
        }
    }
    let output_types:Vec<_>=(start..end).map(|i|tensors[inputs+i].dtype).collect();
    let stores=(0..end-start).map(|i|!(i+1..end-start).any(|j|output_types[i]==output_types[j] && output_slots[i]==output_slots[j])).collect();
    let input_types:Vec<_>=external.iter().map(|&i|tensors[i].dtype).collect();let mut counts=[0usize;4];
    let input_slots=input_types.iter().map(|&t|{let slot=counts[t as usize];counts[t as usize]+=1;slot}).collect();
    let codes=nodes[start..softmax].iter().map(|n|n.op).collect();
    let left=nodes[start..softmax].iter().map(|n|ids[&(n.a as usize)]).collect();
    let right=nodes[start..softmax].iter().map(|n|ids[&(n.b as usize)]).collect();
    let third=nodes[start..softmax].iter().map(|n|if n.op==147 {ids[&(n.scalar as usize)]}else {0}).collect();
    let scalars=nodes[start..softmax].iter().map(|n|n.scalar.to_bits()).collect();
    let source=ids[&(nodes[softmax].a as usize)];
    let rows=first.len/width;
    let threads=if width<=1024 {if rows>=1024 || width<512 {32}else {128}}
        else if width<=16384 {super::row_cached::threads(width)}else {1024};
    let views=external.iter().copied().filter(|&i|tensors[i].len!=1)
        .chain(inputs+start..inputs+end).map(|i|&tensors[i]).collect::<Vec<_>>();
    let vector=super::row_cached::memory_vector_size(c,width,&views,8);
    let periods:Vec<_>=external.iter().map(|&i|super::graph_fusion::vector_period(&tensors[i],&first.shape)
        .filter(|&period|period==1 || period%vector==0).unwrap_or(0)).collect();
    macro_rules! arguments {($indices:expr,$dtype:expr,$ty:ty)=>{
        ($indices).filter(|&i|tensors[i].dtype==$dtype).map(|i|tensors[i].arg()).collect::<SequenceArg<CudaRuntime,Tensor<$ty>>>()
    };}
    macro_rules! run {($v:ty)=>{unsafe {fused_softmax::prepare::<CudaRuntime>(c,super::row_cached::norm_grid(rows,threads),super::row_cached::norm_block(threads),<$v as Size>::value(),
        arguments!(external.iter().copied(),0,f32),arguments!(external.iter().copied(),1,f16),arguments!(external.iter().copied(),2,bf16),arguments!(external.iter().copied(),3,u8),
        arguments!(unique.iter().copied(),0,f32),arguments!(unique.iter().copied(),1,f16),arguments!(unique.iter().copied(),2,bf16),
        rows,first.shape.clone(),width,threads,periods,input_types,input_slots,codes,left,right,third,scalars,output_types,output_slots,stores,
        softmax-start,source,nodes[softmax].op==103)}};}
    let kernel=match vector {8=>run!(Const<8>),4=>run!(Const<4>),2=>run!(Const<2>),_=>run!(Const<1>)};
    Some((end-start,kernel))
}
