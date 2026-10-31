use super::*;
use super::graph_contract::{NodeSpec,NO_WEIGHT,RMS_NORM};
use std::collections::BTreeMap;

#[ruda]
fn load<E:Numeric,V:Size>(input:&Tensor<E>,position:usize,#[comptime] shape:Vec<usize>,#[comptime] period:usize)->Vector<f32,V> {
    let mut value=Vector::<f32,V>::empty();
    if comptime!(period==1) {value=Vector::new(f32::cast_from(input[0usize]));}
    else if comptime!(period!=0) {
        let packed=input.slice(0usize,input.len()).into_vectorized().with_vector_size::<V>();
        value=Vector::cast_from(packed[(position%period)/V::value()]);
    } else {
        #[unroll] for lane in 0usize..V::value() {
            value[lane]=f32::cast_from(input[crate::graph_fusion::address(input,position+lane,shape.clone())]);
        }
    }
    value
}

#[ruda]
fn read<V:Size>(inputs32:&Sequence<Tensor<f32>>,inputs16:&Sequence<Tensor<f16>>,inputsbf:&Sequence<Tensor<bf16>>,
    inputs8:&Sequence<Tensor<u8>>,position:usize,#[comptime] shape:Vec<usize>,#[comptime] dtype:u32,
    #[comptime] slot:usize,#[comptime] period:usize)->Vector<f32,V>
{
    let mut value=Vector::<f32,V>::empty();
    if comptime!(dtype==0) {value=load::<f32,V>(inputs32.index(slot),position,shape,period);}
    else if comptime!(dtype==1) {value=load::<f16,V>(inputs16.index(slot),position,shape,period);}
    else if comptime!(dtype==2) {value=load::<bf16,V>(inputsbf.index(slot),position,shape,period);}
    else {
        value=load::<u8,V>(inputs8.index(slot),position,shape,period);
        if comptime!(dtype==12 || dtype==13 || dtype==15) {
            #[unroll] for lane in 0usize..V::value() {value[lane]=crate::foreach_copy::decode_fp8(u32::cast_from(value[lane]),dtype);}
        }
    }
    value
}

#[ruda]
fn write<V:Size>(outputs32:&mut Sequence<Tensor<f32>>,outputs16:&mut Sequence<Tensor<f16>>,outputsbf:&mut Sequence<Tensor<bf16>>,
    outputs8:&mut Sequence<Tensor<u8>>,position:usize,value:Vector<f32,V>,#[comptime] dtype:u32,#[comptime] slot:usize)
{
    if comptime!(dtype==0) {
        let output=outputs32.index_mut(slot);let mut packed=output.slice_mut(0usize,output.len()).into_vectorized().with_vector_size::<V>();
        packed[position/V::value()]=value;
    } else if comptime!(dtype==1) {
        let output=outputs16.index_mut(slot);let mut packed=output.slice_mut(0usize,output.len()).into_vectorized().with_vector_size::<V>();
        packed[position/V::value()]=Vector::cast_from(value);
    } else if comptime!(dtype==2) {
        let output=outputsbf.index_mut(slot);let mut packed=output.slice_mut(0usize,output.len()).into_vectorized().with_vector_size::<V>();
        packed[position/V::value()]=Vector::cast_from(value);
    } else {
        let mut encoded=Vector::<u8,V>::empty();
        #[unroll] for lane in 0usize..V::value() {encoded[lane]=crate::foreach_copy::encode_fp8(value[lane],dtype);}
        let output=outputs8.index_mut(slot);let mut packed=output.slice_mut(0usize,output.len()).into_vectorized().with_vector_size::<V>();
        packed[position/V::value()]=encoded;
    }
}

#[ruda]
fn evaluate<V:Size>(values:&Array<Vector<f32,V>>,#[comptime] code:u32,#[comptime] left:usize,#[comptime] right:usize,
    #[comptime] third:usize,#[comptime] scalar:u32,#[comptime] dtype:u32)->Vector<f32,V>
{
    let x=values[left];let y=values[right];let mut result=Vector::<f32,V>::empty();
    #[unroll] for lane in 0usize..V::value() {
        if comptime!(code==147) {
            let other=values[third];let mut value=other[lane];if x[lane]!=0.0f32 {value=y[lane];}
            result[lane]=crate::graph_fusion::rounded(value,dtype);
        } else {result[lane]=crate::graph_fusion::evaluate(x[lane],y[lane],f32::from_bits(scalar),code,dtype);}
    }
    result
}

#[ruda(launch)]
fn fused<V:Size>(inputs32:&Sequence<Tensor<f32>>,inputs16:&Sequence<Tensor<f16>>,inputsbf:&Sequence<Tensor<bf16>>,
    inputs8:&Sequence<Tensor<u8>>,outputs32:&mut Sequence<Tensor<f32>>,outputs16:&mut Sequence<Tensor<f16>>,
    outputsbf:&mut Sequence<Tensor<bf16>>,outputs8:&mut Sequence<Tensor<u8>>,rows:usize,
    #[comptime] shape:Vec<usize>,#[comptime] reduced_shape:Vec<usize>,#[comptime] width:usize,#[comptime] threads:usize,
    #[comptime] input_types:Vec<u32>,#[comptime] input_slots:Vec<usize>,#[comptime] periods:Vec<usize>,
    #[comptime] preloads:Vec<bool>,#[comptime] postloads:Vec<bool>,
    #[comptime] codes:Vec<u32>,#[comptime] left:Vec<usize>,#[comptime] right:Vec<usize>,#[comptime] third:Vec<usize>,
    #[comptime] scalars:Vec<u32>,#[comptime] output_types:Vec<u32>,#[comptime] output_slots:Vec<usize>,#[comptime] stores:Vec<bool>,
    #[comptime] reduction:usize,#[comptime] source:usize,#[comptime] weight:usize,#[comptime] weighted:bool,
    #[comptime] rms:bool,#[comptime] mean:bool,#[comptime] product:bool,#[comptime] extremum:bool,#[comptime] minimum:bool,#[comptime] absolute:bool,
    #[comptime] rowwise:Vec<bool>,#[comptime] row_tail:bool)
{
    let mut row=RUDA_POS_X as usize;let thread=UNIT_POS as usize%threads;
    if comptime!(threads==32) {row=row*4+UNIT_POS as usize/32;}
    if row<rows {
        let chunks=comptime!(width.div_ceil(threads*V::value()));
        let mut cached=Array::<Vector<f32,V>>::new(chunks);
        let mut scratch=SharedMemory::<f32>::new(threads/32);
        let mut local=0.0f32;let mut sums=Vector::<f32,V>::new(0.0f32);
        let mut key=u32::MAX.runtime();
        #[unroll] for j in 0usize..chunks {
            let column=(thread+j*threads)*V::value();let position=row*width+column;
            let mut value=Vector::<f32,V>::new(0.0f32);
            let mut multiplied=Vector::<f32,V>::new(0.0f32);
            if column<width {
                let mut values=Array::<Vector<f32,V>>::new(input_types.len()+reduction);
                #[unroll] for i in 0usize..input_types.len() {
                    if comptime!(preloads[i]) {
                        values[i]=read::<V>(inputs32,inputs16,inputsbf,inputs8,position,shape.clone(),
                            comptime!(input_types[i]),comptime!(input_slots[i]),comptime!(periods[i]));
                    }
                }
                #[unroll] for i in 0usize..reduction {
                    let next=evaluate::<V>(&values,comptime!(codes[i]),comptime!(left[i]),comptime!(right[i]),
                        comptime!(third[i]),comptime!(scalars[i]),comptime!(output_types[i]));
                    values[input_types.len()+i]=next;
                    if comptime!(stores[i]) {write::<V>(outputs32,outputs16,outputsbf,outputs8,position,next,comptime!(output_types[i]),comptime!(output_slots[i]));}
                }
                value=values[source];
                if comptime!(product) {
                    let factor=values[comptime!(right[reduction])];
                    #[unroll] for lane in 0usize..V::value() {
                        multiplied[lane]=crate::graph_fusion::rounded(value[lane]*factor[lane],comptime!(output_types[reduction]));
                    }
                }
            }
            cached[j]=value;
            if comptime!(extremum) {
                if column<width {
                    #[unroll] for lane in 0usize..V::value() {
                        let mut term=value[lane];if comptime!(absolute) {term=term.abs();}
                        let combined=crate::operation_reduce::combine::<f32>(local,key,term,(column+lane) as u32,comptime!(if minimum {1u32}else {0u32}));
                        local=combined.0;key=combined.1;
                    }
                }
            } else if comptime!(rms) {#[unroll] for lane in 0usize..V::value() {local+=value[lane]*value[lane];}}
            else if comptime!(product) {sums+=multiplied;}
            else {sums+=value;}
        }
        if comptime!(!rms && !extremum) {#[unroll] for lane in 0usize..V::value() {local+=sums[lane];}}
        let mut reduced=0.0f32;
        if comptime!(extremum) {
            let partial=crate::operation_reduce::warp_reduce::<f32>(local,key,comptime!(if minimum {1u32}else {0u32}));
            let mut value=partial.0;
            if comptime!(threads>32) {
                let mut keys=SharedMemory::<u32>::new(threads/32);
                if thread%32==0 {scratch[thread/32]=partial.0;keys[thread/32]=partial.1;}
                sync_ruda();local=0.0f32;key=u32::MAX;
                if thread%32<threads/32 {local=scratch[thread%32];key=keys[thread%32];}
                let combined=crate::operation_reduce::warp_reduce::<f32>(local,key,comptime!(if minimum {1u32}else {0u32}));
                value=combined.0;
            }
            reduced=plane_shuffle(value,0u32);
        } else {reduced=crate::row_cached::block_reduce(local,&mut scratch,threads,false,0usize);}
        if comptime!(rms || mean) {reduced/=width as f32;}
        if comptime!(rms) {reduced=(reduced+f32::from_bits(comptime!(scalars[reduction]))).inverse_sqrt();}
        else {
            reduced=crate::graph_fusion::rounded(reduced,comptime!(output_types[reduction]));
            if comptime!(stores[reduction]) {
                if thread==0 {write::<Const<1>>(outputs32,outputs16,outputsbf,outputs8,row,Vector::new(reduced),comptime!(output_types[reduction]),comptime!(output_slots[reduction]));}
            }
        }
        if comptime!(row_tail) {if thread!=0 {terminate!();}}
        #[unroll] for j in 0usize..comptime!(if row_tail {1}else {chunks}) {
            let column=(thread+j*threads)*V::value();let position=row*width+column;
            if column<width {
                let mut values=Array::<Vector<f32,V>>::new(input_types.len()+codes.len());
                #[unroll] for i in 0usize..input_types.len() {
                    if comptime!(postloads[i]) {
                        if comptime!(row_tail) {
                            let scalar=read::<Const<1>>(inputs32,inputs16,inputsbf,inputs8,row,reduced_shape.clone(),
                                comptime!(input_types[i]),comptime!(input_slots[i]),0usize);
                            values[i]=Vector::new(scalar[0usize]);
                        } else {
                            values[i]=read::<V>(inputs32,inputs16,inputsbf,inputs8,position,shape.clone(),
                                comptime!(input_types[i]),comptime!(input_slots[i]),comptime!(periods[i]));
                        }
                    }
                }
                values[source]=cached[j];
                let mut result=Vector::<f32,V>::new(reduced);
                if comptime!(rms) {
                    let input=cached[j];
                    #[unroll] for lane in 0usize..V::value() {
                        let mut value=input[lane]*reduced;
                        if comptime!(weighted) {let factor=values[weight];value*=factor[lane];}
                        result[lane]=crate::graph_fusion::rounded(value,comptime!(output_types[reduction]));
                    }
                    if comptime!(stores[reduction]) {write::<V>(outputs32,outputs16,outputsbf,outputs8,position,result,comptime!(output_types[reduction]),comptime!(output_slots[reduction]));}
                }
                values[input_types.len()+reduction]=result;
                #[unroll] for i in reduction+1..codes.len() {
                    let next=evaluate::<V>(&values,comptime!(codes[i]),comptime!(left[i]),comptime!(right[i]),
                        comptime!(third[i]),comptime!(scalars[i]),comptime!(output_types[i]));
                    values[input_types.len()+i]=next;
                    if comptime!(stores[i]) {
                        if comptime!(rowwise[i]) {
                            if thread==0 && j==0 {write::<Const<1>>(outputs32,outputs16,outputsbf,outputs8,row,Vector::new(next[0usize]),comptime!(output_types[i]),comptime!(output_slots[i]));}
                        } else {write::<V>(outputs32,outputs16,outputsbf,outputs8,position,next,comptime!(output_types[i]),comptime!(output_slots[i]));}
                    }
                }
            }
        }
    }
}

fn operands(n:&NodeSpec)->Vec<usize> {
    let mut ids=vec![n.a as usize];if n.b!=NO_WEIGHT {ids.push(n.b as usize);}
    if n.op==147 {ids.push(n.scalar as usize);}ids
}

pub(super) fn prepare(c:&ComputeClient<CudaRuntime>,tensors:&[View],nodes:&[NodeSpec],inputs:usize,start:usize)
    ->Option<(usize,PreparedKernel<CudaRuntime>)>
{
    let reducible=|op|matches!(op,RMS_NORM|104|105|119|120|189..=192|203..=206|208|209);
    let first_node=nodes.get(start)?;
    let first=if reducible(first_node.op) {tensors.get(first_node.a as usize)?}else {tensors.get(inputs+start)?};
    let width=*first.shape.last()?;
    if first.len==0 || first.dtype>2 || !first.contiguous() || !super::row_cached::eligible(width) {return None;}
    let rows=first.len/width;let mut reduction=start;
    loop {
        let n=nodes.get(reduction)?;let out=&tensors[inputs+reduction];
        if out.dtype>2 || !out.contiguous() {return None;}
        if reducible(n.op) {
            if tensors[n.a as usize].shape!=first.shape {return None;}
            if n.op!=RMS_NORM && (n.scalar as u32!=1u32<<(first.shape.len()-1) || out.len!=rows) {return None;}
            break;
        }
        if reduction-start>=8 || out.shape!=first.shape || !super::graph_fusion::supported(n.op) {return None;}
        reduction+=1;
    }
    let normal=&nodes[reduction];let rms=normal.op==RMS_NORM;
    let product=matches!(normal.op,203..=206);
    if product {
        let rhs=&tensors[normal.b as usize];let rank=first.shape.len();
        if rhs.shape.len()>rank || rhs.shape.iter().enumerate().any(|(d,&size)|size!=1 && size!=first.shape[d+rank-rhs.shape.len()]) {return None;}
    }
    let reduced_shape=&tensors[inputs+reduction].shape;
    let broadcast=matches!(normal.op,RMS_NORM|104|105|189|190|203|205|208) || first.shape.len()==1;
    if rms && normal.b!=NO_WEIGHT && normal.b as usize>=inputs+start {return None;}
    let source_id=normal.a as usize;let mut end=reduction+1;
    while end<nodes.len() && end-reduction<=8 {
        let n=&nodes[end];let out=&tensors[inputs+end];let ids=operands(n);
        if !super::graph_fusion::supported(n.op) || !matches!(out.dtype,0..=2|12|13|15) || !out.contiguous()
            || !(broadcast && out.shape==first.shape || !rms && out.shape==*reduced_shape)
            || !ids.iter().any(|&id|id>=inputs+reduction)
            || ids.iter().any(|&id|id>=inputs+start && id<inputs+reduction && id!=source_id) {break;}
        if out.dtype>2 && (n.op!=114 || tensors[n.a as usize].dtype==out.dtype) {break;}
        end+=1;
    }
    if end-start<2 {return None;}
    let mut external=Vec::new();let mut ids=BTreeMap::new();
    for n in &nodes[start..end] {
        for id in operands(n) {
            let value=tensors.get(id)?;
            if !matches!(value.dtype,0..=3|12|13|15) {return None;}
            if id<inputs+start && !ids.contains_key(&id) {ids.insert(id,external.len());external.push(id);}
        }
    }
    if external.len()+end-start>24 {return None;}
    for i in start..end {ids.insert(inputs+i,external.len()+i-start);}
    let mut unique=Vec::<usize>::new();let mut slots=Vec::new();let mut output_slots=Vec::new();let mut counts=[0usize;4];
    for i in inputs+start..inputs+end {
        if external.iter().any(|&j|super::graph_fusion::overlaps(&tensors[i],&tensors[j])) {return None;}
        if let Some(p)=unique.iter().position(|&j|super::graph_fusion::overlaps(&tensors[i],&tensors[j])) {
            let old=&tensors[unique[p]];let new=&tensors[i];
            if old.dtype!=new.dtype || old.shape!=new.shape || old.handle.offset_start!=new.handle.offset_start {return None;}
            output_slots.push(slots[p]);
        } else {
            let dtype=tensors[i].dtype.min(3) as usize;let slot=counts[dtype];counts[dtype]+=1;
            unique.push(i);slots.push(slot);output_slots.push(slot);
        }
    }
    let input_types:Vec<_>=external.iter().map(|&i|tensors[i].dtype).collect();let mut counts=[0usize;4];
    let input_slots=input_types.iter().map(|&dtype|{let bucket=dtype.min(3) as usize;let slot=counts[bucket];counts[bucket]+=1;slot}).collect();
    let mut preloads=vec![false;external.len()];let mut postloads=preloads.clone();
    for n in &nodes[start..reduction] {for id in operands(n) {if id<inputs+start {preloads[ids[&id]]=true;}}}
    if source_id<inputs+start {preloads[ids[&source_id]]=true;}
    if product && (normal.b as usize)<inputs+start {preloads[ids[&(normal.b as usize)]]=true;}
    if rms && normal.b!=NO_WEIGHT {postloads[ids[&(normal.b as usize)]]=true;}
    for n in &nodes[reduction+1..end] {for id in operands(n) {if id<inputs+start && id!=source_id {postloads[ids[&id]]=true;}}}
    let output_types:Vec<_>=(start..end).map(|i|tensors[inputs+i].dtype).collect();
    let stores=(0..end-start).map(|i|!(i+1..end-start).any(|j|output_types[i]==output_types[j] && output_slots[i]==output_slots[j])).collect();
    let codes=nodes[start..end].iter().map(|n|n.op).collect();
    let left=nodes[start..end].iter().map(|n|ids[&(n.a as usize)]).collect();
    let right=nodes[start..end].iter().map(|n|{let operand=if n.b==NO_WEIGHT {n.a}else {n.b};ids[&(operand as usize)]}).collect();
    let third=nodes[start..end].iter().map(|n|if n.op==147 {ids[&(n.scalar as usize)]}else {0}).collect();
    let scalars=nodes[start..end].iter().map(|n|n.scalar.to_bits()).collect();
    let mut vector=super::row_cached::vector_size(width,&tensors[source_id],&tensors[inputs+start]);
    if product && tensors[normal.b as usize].handle.offset_start.unwrap_or(0)%(4*element_bytes(tensors[normal.b as usize].dtype)) as u64!=0 {vector=1;}
    let rowwise:Vec<_>=(start..end).map(|i|!rms && i>=reduction && tensors[inputs+i].shape==*reduced_shape).collect();
    let row_tail=!rms && rowwise[reduction-start+1..].iter().all(|&v|v);
    if vector==4 && (start..end).filter(|&i|!rowwise[i-start]).any(|i|tensors[inputs+i].handle.offset_start.unwrap_or(0)%(4*element_bytes(tensors[inputs+i].dtype)) as u64!=0) {return None;}
    let periods:Vec<_>=external.iter().map(|&i|super::graph_fusion::vector_period(&tensors[i],&first.shape).unwrap_or(0)).collect();
    if vector==4 && external.len()+end-start<=16 {
        let views=external.iter().copied().filter(|&i|tensors[i].len!=1)
            .chain((start..end).filter(|&i|!rowwise[i-start]).map(|i|inputs+i))
            .map(|i|&tensors[i]).collect::<Vec<_>>();
        let wide=super::row_cached::memory_vector_size(c,width,&views,8);
        if wide>vector && periods.iter().all(|&period|period<=1 || period%wide==0) {vector=wide;}
    }
    let extremum=matches!(normal.op,189..=192|208|209);
    let threads=if rms {super::row_cached::norm_threads(width,rows)}else if extremum {if width<=256 {32}else {256}}
        else if width<=128 || width<=1024 && rows>=128 {32}else if width<=4096 {128}else {256};
    let source=ids[&source_id];let weighted=rms && normal.b!=NO_WEIGHT;
    let weight=if weighted {ids[&(normal.b as usize)]}else {0};
    macro_rules! arguments {($indices:expr,$dtype:expr,$ty:ty)=>{
        ($indices).filter(|&i|tensors[i].dtype.min(3)==$dtype).map(|i|tensors[i].arg()).collect::<SequenceArg<CudaRuntime,Tensor<$ty>>>()
    };}
    macro_rules! run {($v:ty)=>{fused::prepare::<CudaRuntime>(c,super::row_cached::norm_grid(rows,threads),super::row_cached::norm_block(threads),<$v as Size>::value(),
        arguments!(external.iter().copied(),0,f32),arguments!(external.iter().copied(),1,f16),arguments!(external.iter().copied(),2,bf16),arguments!(external.iter().copied(),3,u8),
        arguments!(unique.iter().copied(),0,f32),arguments!(unique.iter().copied(),1,f16),arguments!(unique.iter().copied(),2,bf16),arguments!(unique.iter().copied(),3,u8),
        rows,first.shape.clone(),reduced_shape.clone(),width,threads,input_types,input_slots,periods,preloads,postloads,codes,left,right,third,scalars,output_types,output_slots,stores,
        reduction-start,source,weight,weighted,rms,matches!(normal.op,105|120|205|206),product,extremum,matches!(normal.op,190|192),matches!(normal.op,208|209),rowwise,row_tail)};}
    let kernel=match vector {8=>run!(Const<8>),4=>run!(Const<4>),_=>run!(Const<1>)};
    Some((end-start,kernel))
}
