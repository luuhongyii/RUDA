use super::*;
use super::graph_contract::NodeSpec;
use std::collections::BTreeMap;

#[ruda]
fn location(position:usize,#[comptime] terms:Vec<usize>,#[comptime] begin:usize,#[comptime] end:usize)->(usize,bool) {
    let mut rest=position;let mut address=0usize;let mut owner=true;
    #[unroll] for reverse in 0usize..comptime!((end-begin)/4) {
        let at=comptime!(end-4*(reverse+1));
        let coordinate=rest%comptime!(terms[at]);rest/=comptime!(terms[at]);
        let quotient=coordinate/comptime!(terms[at+3]);
        address+=(quotient%comptime!(terms[at+2]))*comptime!(terms[at+1]);
        if quotient>=comptime!(terms[at+2]) || coordinate%comptime!(terms[at+3])!=0usize {owner=false;}
    }
    (address,owner)
}

#[ruda]
fn read<V:Size>(inputs32:&Sequence<Tensor<f32>>,inputs16:&Sequence<Tensor<f16>>,inputsbf:&Sequence<Tensor<bf16>>,
    position:usize,#[comptime] dtype:u32,#[comptime] slot:usize,#[comptime] splat:bool)->Vector<f32,V>
{
    let mut value=Vector::<f32,V>::empty();
    if comptime!(dtype==0) {
        let input=inputs32.index(slot);
        if comptime!(splat) {value=Vector::new(input[position]);}
        else {value=input.slice(0usize,input.len()).into_vectorized().with_vector_size::<V>()[position/V::value()];}
    } else if comptime!(dtype==1) {
        let input=inputs16.index(slot);
        if comptime!(splat) {value=Vector::new(f32::cast_from(input[position]));}
        else {value=Vector::cast_from(input.slice(0usize,input.len()).into_vectorized().with_vector_size::<V>()[position/V::value()]);}
    } else {
        let input=inputsbf.index(slot);
        if comptime!(splat) {value=Vector::new(f32::cast_from(input[position]));}
        else {value=Vector::cast_from(input.slice(0usize,input.len()).into_vectorized().with_vector_size::<V>()[position/V::value()]);}
    }
    value
}

#[ruda(launch)]
fn chain<V:Size>(inputs32:&Sequence<Tensor<f32>>,inputs16:&Sequence<Tensor<f16>>,inputsbf:&Sequence<Tensor<bf16>>,
    outputs:&mut Sequence<Tensor<f32>>,length:usize,
    #[comptime] codes:Vec<u32>,#[comptime] scalars:Vec<u32>,
    #[comptime] terms:Vec<usize>,#[comptime] ranges:Vec<usize>,#[comptime] main:Vec<usize>,
    #[comptime] internal:Vec<bool>,#[comptime] types:Vec<u32>,#[comptime] slots:Vec<usize>,#[comptime] splats:Vec<bool>)
{
    let position=ABSOLUTE_POS as usize*V::value();
    if position<length {
        let mut positions=Array::<usize>::new(codes.len());let mut owners=Array::<u32>::new(codes.len());
        let mut rest=position;let mut owner=true;
        #[unroll] for reverse in 0usize..codes.len() {
            let i=comptime!(codes.len()-1-reverse);let map=comptime!(main[i]);
            positions[i]=rest;owners[i]=u32::cast_from(owner);
            let previous=location(rest,terms.clone(),comptime!(ranges[map]),comptime!(ranges[map+1]));
            rest=previous.0;owner=owner && previous.1;
        }
        let mut value=Vector::<f32,V>::new(0.0);
        #[unroll] for i in 0usize..codes.len() {
            let mut x=value;let mut y=value;
            if comptime!(!internal[2*i]) {
                let source=location(positions[i],terms.clone(),comptime!(ranges[2*i]),comptime!(ranges[2*i+1]));
                x=read::<V>(inputs32,inputs16,inputsbf,source.0,comptime!(types[2*i]),comptime!(slots[2*i]),comptime!(splats[2*i]));
            }
            if comptime!(!internal[2*i+1]) {
                let source=location(positions[i],terms.clone(),comptime!(ranges[2*i+1]),comptime!(ranges[2*i+2]));
                y=read::<V>(inputs32,inputs16,inputsbf,source.0,comptime!(types[2*i+1]),comptime!(slots[2*i+1]),comptime!(splats[2*i+1]));
            }
            #[unroll] for lane in 0usize..V::value() {
                value[lane]=super::graph_fusion::evaluate(x[lane],y[lane],f32::from_bits(comptime!(scalars[i])),comptime!(codes[i]),0);
            }
            if owners[i]!=0u32 {
                let output=outputs.index_mut(i);
                let mut packed=output.slice_mut(0usize,output.len()).into_vectorized().with_vector_size::<V>();
                packed[positions[i]/V::value()]=value;
            }
        }
    }
}

fn is_layout(op:u32)->bool {matches!(op,112|113|115|215|216)}
fn supported(op:u32)->bool {
    is_layout(op) || matches!(op,0..=4|8..=25|27|35..=40|43|45|101|108..=111|114|116..=118|121|129|174..=188)
}

fn mapping(input:&View,out:&View,node:&NodeSpec)->Option<Vec<usize>> {
    let mut result=Vec::new();
    if matches!(node.op,0|112|114) {
        if input.len!=out.len {return None;}
        return Some(vec![out.len,1,out.len,1]);
    }
    if node.op==216 && node.scalar==-1.0 {
        return Some(vec![out.len,1,input.len,out.len/input.len]);
    }
    if input.shape.len()>out.shape.len() {return None;}
    let padding=out.shape.len()-input.shape.len();
    for (d,&extent) in out.shape.iter().enumerate() {
        let axis=if node.op==113 {((node.scalar as usize)>>(3*d))&7}else {d.wrapping_sub(padding)};
        let old=if axis<input.shape.len() {input.shape[axis]}else {1};
        let stride=if axis<input.shape.len() {input.strides[axis]}else {0};
        let divisor=if node.op==216 && node.scalar as usize==d {extent/old}else {1};
        if node.op==113 || node.op==216 {
            if old.checked_mul(divisor)!=Some(extent) {return None;}
        } else if node.op==215 {
            if extent%old!=0 {return None;}
        } else if old!=1 && old!=extent {return None;}
        result.extend([extent,stride,old,divisor]);
    }
    Some(result)
}

fn vector_mapping(terms:&[usize])->Option<bool> {
    let active=terms.chunks_exact(4).filter(|t|t[0]>1).collect::<Vec<_>>();
    let last=*active.last()?;
    if last[0]%4!=0 {return None;}
    if last[1]==0 || last[2]==1 || last[3]%4==0 {return Some(true);}
    if last[1]==1 && last[2]%4==0 && last[3]==1
        && active[..active.len()-1].iter().all(|t|t[1]%4==0) {Some(false)}else {None}
}

pub(super) fn prepare(c:&ComputeClient<CudaRuntime>,tensors:&[View],nodes:&[NodeSpec],inputs:usize,start:usize)
    ->Option<(usize,PreparedKernel<CudaRuntime>)>
{
    let lower=inputs+start;let mut end=start;let mut layouts=0usize;let mut maps=Vec::new();
    let mut main=Vec::new();let mut internal=Vec::new();let mut sources=Vec::new();
    while end<nodes.len() && end-start<8 {
        let node=&nodes[end];let id=inputs+end;let out=&tensors[id];
        if !supported(node.op) || out.dtype!=0 || out.len==0 || !out.contiguous() {break;}
        let operands=[node.a as usize,node.b as usize];
        if operands.iter().any(|&source|source>=id || tensors[source].dtype>2 || tensors[source].len==0
            || !tensors[source].contiguous() || source>=lower && source!=id-1) {break;}
        let side=if end==start || operands[0]==id-1 {0}else if operands[1]==id-1 {1}else {break;};
        let mut pair=Vec::new();
        for &source in &operands {
            if let Some(map)=mapping(&tensors[source],out,node) {pair.push(map);}
        }
        if pair.len()!=2 {break;}
        if out.len>tensors[operands[side]].len.saturating_mul(32) {break;}
        if (lower..id).any(|previous|super::graph_fusion::overlaps(out,&tensors[previous]))
            || operands.iter().filter(|&&source|source<lower).any(|&source|
                (lower..=id).any(|output|super::graph_fusion::overlaps(&tensors[output],&tensors[source])))
            || sources.iter().any(|&source|super::graph_fusion::overlaps(out,&tensors[source])) {break;}
        for (operand,map) in operands.into_iter().zip(pair) {
            internal.push(end>start && operand==id-1);maps.push(map);sources.push(operand);
        }
        main.push((end-start)*2+side);layouts+=usize::from(is_layout(node.op));end+=1;
    }
    if end-start<2 || layouts==0 {return None;}
    let mut external=Vec::new();let mut ids=BTreeMap::new();let mut counts=[0usize;3];
    let mut types=Vec::new();let mut slots=Vec::new();
    for (&source,&inside) in sources.iter().zip(&internal) {
        let dtype=tensors[source].dtype;types.push(dtype);
        if inside {slots.push(0);continue;}
        let slot=*ids.entry(source).or_insert_with(|| {
            let slot=counts[dtype as usize];counts[dtype as usize]+=1;external.push(source);slot
        });
        slots.push(slot);
    }
    let vector_maps=maps.iter().map(|map|vector_mapping(map)).collect::<Option<Vec<_>>>();
    let vector=vector_maps.as_ref().is_some_and(|splats|
        main.iter().skip(1).all(|&map|!splats[map])
        && (lower..inputs+end).all(|id|tensors[id].len%4==0 && tensors[id].handle.offset_start.unwrap_or(0)%16==0)
        && sources.iter().zip(splats).zip(&internal).all(|((&source,&splat),&inside)|inside || splat
            || tensors[source].handle.offset_start.unwrap_or(0)%(4*element_bytes(tensors[source].dtype)) as u64==0));
    let mut ranges=vec![0];let mut terms=Vec::new();
    for map in maps {terms.extend(map);ranges.push(terms.len());}
    let codes=nodes[start..end].iter().map(|node|if is_layout(node.op) {0}else {node.op}).collect();
    let scalars=nodes[start..end].iter().map(|node|node.scalar.to_bits()).collect();
    let splats=if vector {vector_maps.unwrap()}else {vec![true;sources.len()]};
    let length=tensors[inputs+end-1].len;let width=if vector {4}else {1};
    let blocks=length.div_ceil(width*128);let h=&c.properties().hardware;
    if h.max_units_per_ruda<128 || h.max_ruda_dim.0<128 || blocks>h.max_ruda_count.0 as usize {return None;}
    macro_rules! arguments {($dtype:expr,$ty:ty)=>{external.iter().filter(|&&id|tensors[id].dtype==$dtype)
        .map(|&id|tensors[id].arg()).collect::<SequenceArg<CudaRuntime,Tensor<$ty>>>()};}
    macro_rules! launch {($v:ty)=>{unsafe {chain::prepare::<$v,CudaRuntime>(c,RudaCount::Static(blocks as u32,1,1),RudaDim::new_1d(128),
        arguments!(0,f32),arguments!(1,f16),arguments!(2,bf16),
        (lower..inputs+end).map(|id|tensors[id].arg()).collect::<SequenceArg<CudaRuntime,Tensor<f32>>>(),length,
        codes,scalars,terms,ranges,main,internal,types,slots,splats)}};}
    Some((end-start,if vector {launch!(Const<4>)}else {launch!(Const<1>)}))
}
