use super::*;
use super::graph_contract::{NodeSpec,matmul_layout};
use std::collections::BTreeSet;

#[ruda]
fn decode_value<E:Numeric,V:Size>(raw:Vector<E,V>,#[comptime] dtype:u32)->Vector<f32,V> {
    let mut value=Vector::<f32,V>::cast_from(raw);
    if comptime!(dtype==12 || dtype==13 || dtype==15) {
        #[unroll] for lane in 0usize..V::value() {
            value[lane]=crate::foreach_copy::decode_fp8(u32::cast_from(raw[lane]),dtype);
        }
    }
    value
}

#[ruda]
fn decode<E:Numeric,V:Size>(source:&Tensor<E>,position:usize,#[comptime] dtype:u32)->Vector<f32,V> {
    let packed=source.slice(0usize,source.len()).into_vectorized().with_vector_size::<V>();
    decode_value(packed[position/V::value()],dtype)
}

#[ruda]
fn scale<V:Size>(value:Vector<f32,V>,scales:&Tensor<f32>,row:usize,column:usize,
    #[comptime] columns:usize,#[comptime] block_rows:usize,#[comptime] block_columns:usize)->Vector<f32,V>
{
    let base=row/block_rows*comptime!(columns/block_columns);let mut result=value;
    if comptime!(block_columns%V::value()==0) {
        result*=Vector::new(scales[base+column/block_columns]);
    } else {
        #[unroll] for lane in 0usize..V::value() {result[lane]*=scales[base+(column+lane)/block_columns];}
    }
    result
}

#[ruda(launch)]
fn scaled_copy<E:Numeric,V:Size>(source:&Tensor<E>,scales:&Tensor<f32>,out:&mut Tensor<f32>,
    #[comptime] columns:usize,#[comptime] block_rows:usize,#[comptime] block_columns:usize,#[comptime] dtype:u32)
{
    let position=ABSOLUTE_POS as usize*V::value();
    if position<out.len() {
        let value=decode::<E,V>(source,position,dtype);
        let result=scale(value,scales,position/columns,position%columns,columns,block_rows,block_columns);
        let mut packed=out.slice_mut(0usize,out.len()).into_vectorized().with_vector_size::<V>();
        packed[position/V::value()]=result;
    }
}

pub(super) fn prepare(c:&ComputeClient<CudaRuntime>,source:&View,scales:&View,out:&View,block_rows:usize)
    ->PreparedKernel<CudaRuntime>
{
    let columns=source.shape[1];let block_columns=columns/(scales.len/(source.shape[0]/block_rows));
    let vector=super::row_cached::memory_vector_size(c,columns,&[source,out],8);
    let grid=RudaCount::Static(out.len.div_ceil(128*vector) as u32,1,1);
    macro_rules! launch {($e:ty,$v:ty)=>{scaled_copy::prepare::<$e,$v,CudaRuntime>(c,grid,RudaDim::new_1d(128),
        source.arg(),scales.arg(),out.arg(),columns,block_rows,block_columns,source.dtype)};}
    macro_rules! typed {($e:ty)=>{match vector {8=>launch!($e,Const<8>),4=>launch!($e,Const<4>),2=>launch!($e,Const<2>),_=>launch!($e,Const<1>)}};}
    match source.dtype {0=>typed!(f32),1=>typed!(f16),2=>typed!(bf16),12|13|15=>typed!(u8),_=>unreachable!()}
}

#[ruda]
fn stage<E:Numeric>(source:&Tensor<E>,scales:&Tensor<f32>,outputs:&mut Sequence<Tensor<f32>>,
    indices32:&Sequence<Tensor<i32>>,indices64:&Sequence<Tensor<i64>>,gathered:&mut Sequence<Tensor<E>>,
    invalid:&mut Tensor<Atomic<u32>>,
    shared:&mut SharedMemory<Vector<f32,Const<4>>>,row:usize,column:usize,bank:usize,owner:bool,
    batch_base:usize,
    #[comptime] rows:usize,#[comptime] columns:usize,#[comptime] tile_rows:usize,#[comptime] tile_columns:usize,
    #[comptime] block_rows:usize,#[comptime] block_columns:usize,#[comptime] dtype:u32,
    #[comptime] scaled:bool,#[comptime] store:bool,#[comptime] slot:usize,
    #[comptime] index_dtype:usize,#[comptime] index_slot:usize,#[comptime] index_bound:usize)
{
    let stride=comptime!(tile_columns/4+1);let size=comptime!(tile_rows*stride);
    #[unroll] for step in 0usize..comptime!((tile_rows*tile_columns/4).div_ceil(256)) {
        let p=UNIT_POS as usize+step*256usize;
        if p<tile_rows*tile_columns/4 {
            let r=row+p/comptime!(tile_columns/4);let col=column+p%comptime!(tile_columns/4)*4usize;
            let mut value=Vector::new(0.0f32);
            if r<rows && col<columns {
                if comptime!(index_dtype!=0) {
                    let mut index=0i64;
                    if comptime!(index_dtype==4) {index=indices64.index(index_slot)[r];}
                    else {index=i64::cast_from(indices32.index(index_slot)[r]);}
                    let mut raw=Vector::<E,Const<4>>::new(E::cast_from(0));
                    if index<0i64 || index>=i64::cast_from(index_bound) {invalid[0].store(1u32);}
                    else {
                        let packed=source.slice(0usize,source.len()).into_vectorized().with_vector_size::<Const<4>>();
                        raw=packed[(usize::cast_from(index)*columns+col)/4usize];
                    }
                    if owner {
                        let target=gathered.index_mut(0usize);
                        let mut packed=target.slice_mut(0usize,target.len()).into_vectorized().with_vector_size::<Const<4>>();
                        packed[(r*columns+col)/4usize]=raw;
                    }
                    value=decode_value(raw,dtype);
                } else {value=decode::<E,Const<4>>(source,batch_base+r*columns+col,dtype);}
                if comptime!(scaled) {
                    value=scale(value,scales,r,col,columns,block_rows,block_columns);
                }
                if comptime!(store) {
                    if owner {
                        let output=outputs.index_mut(slot);
                        let mut packed=output.slice_mut(0usize,output.len()).into_vectorized().with_vector_size::<Const<4>>();
                        packed[(batch_base+r*columns+col)/4usize]=value;
                    }
                }
            }
            shared[bank*size+p/comptime!(tile_columns/4)*stride+p%comptime!(tile_columns/4)]=value;
        }
    }
}

#[ruda]
fn load_tiles<A:Numeric,B:Numeric>(a:&Tensor<A>,b:&Tensor<B>,as_:&Tensor<f32>,bs:&Tensor<f32>,
    indices32:&Sequence<Tensor<i32>>,indices64:&Sequence<Tensor<i64>>,
    gathered_a:&mut Sequence<Tensor<A>>,gathered_b:&mut Sequence<Tensor<B>>,invalid:&mut Tensor<Atomic<u32>>,
    outputs:&mut Sequence<Tensor<f32>>,sa:&mut SharedMemory<Vector<f32,Const<4>>>,sb:&mut SharedMemory<Vector<f32,Const<4>>>,
    row:usize,col:usize,block:usize,bank:usize,batch_a:usize,batch_b:usize,owner_a:bool,owner_b:bool,
    #[comptime] m:usize,#[comptime] n:usize,#[comptime] k:usize,#[comptime] bm:usize,#[comptime] bn:usize,#[comptime] bk:usize,
    #[comptime] ta:bool,#[comptime] nt:bool,#[comptime] at:u32,#[comptime] bt:u32,
    #[comptime] ar:usize,#[comptime] ac:usize,#[comptime] br:usize,#[comptime] bc:usize,
    #[comptime] scaled_a:bool,#[comptime] scaled_b:bool,#[comptime] store_a:bool,#[comptime] store_b:bool,
    #[comptime] slot_a:usize,#[comptime] slot_b:usize,#[comptime] index_codes:Vec<usize>)
{
    if comptime!(ta) {
        stage(a,as_,outputs,indices32,indices64,gathered_a,invalid,sa,block*bk,row,bank,col==0usize && owner_a,batch_a*m*k,
            k,m,bk,bm,ar,ac,at,scaled_a,store_a,slot_a,comptime!(index_codes[0]),comptime!(index_codes[1]),comptime!(index_codes[2]));
    } else {
        stage(a,as_,outputs,indices32,indices64,gathered_a,invalid,sa,row,block*bk,bank,col==0usize && owner_a,batch_a*m*k,
            m,k,bm,bk,ar,ac,at,scaled_a,store_a,slot_a,comptime!(index_codes[0]),comptime!(index_codes[1]),comptime!(index_codes[2]));
    }
    if comptime!(nt) {
        stage(b,bs,outputs,indices32,indices64,gathered_b,invalid,sb,col,block*bk,bank,row==0usize && owner_b,batch_b*k*n,
            n,k,bn,bk,br,bc,bt,scaled_b,store_b,slot_b,comptime!(index_codes[3]),comptime!(index_codes[4]),comptime!(index_codes[5]));
    } else {
        stage(b,bs,outputs,indices32,indices64,gathered_b,invalid,sb,block*bk,col,bank,row==0usize && owner_b,batch_b*k*n,
            k,n,bk,bn,br,bc,bt,scaled_b,store_b,slot_b,comptime!(index_codes[3]),comptime!(index_codes[4]),comptime!(index_codes[5]));
    }
}

#[ruda(launch)]
fn tiled<A:Numeric,B:Numeric>(a:&Tensor<A>,b:&Tensor<B>,as_:&Tensor<f32>,bs:&Tensor<f32>,
    indices32:&Sequence<Tensor<i32>>,indices64:&Sequence<Tensor<i64>>,
    gathered_a:&mut Sequence<Tensor<A>>,gathered_b:&mut Sequence<Tensor<B>>,invalid:&mut Tensor<Atomic<u32>>,
    outputs:&mut Sequence<Tensor<f32>>,out:&mut Tensor<f32>,
    #[comptime] m:usize,#[comptime] n:usize,#[comptime] k:usize,#[comptime] bm:usize,#[comptime] bn:usize,#[comptime] bk:usize,
    #[comptime] ta:bool,#[comptime] nt:bool,#[comptime] at:u32,#[comptime] bt:u32,
    #[comptime] ar:usize,#[comptime] ac:usize,#[comptime] br:usize,#[comptime] bc:usize,
    #[comptime] scaled_a:bool,#[comptime] scaled_b:bool,#[comptime] store_a:bool,#[comptime] store_b:bool,
    #[comptime] slot_a:usize,#[comptime] slot_b:usize,#[comptime] vector:usize,
    #[comptime] span_a:usize,#[comptime] span_b:usize,#[comptime] repeats_a:usize,#[comptime] repeats_b:usize,
    #[comptime] index_codes:Vec<usize>)
{
    let batch=RUDA_POS_Y as usize;
    let batch_a=batch/(span_a*repeats_a)*span_a+batch%span_a;
    let batch_b=batch/(span_b*repeats_b)*span_b+batch%span_b;
    let owner_a=(batch/span_a)%repeats_a==0usize;
    let owner_b=(batch/span_b)%repeats_b==0usize;
    let tile=RUDA_POS_X as usize;
    let group=tile/comptime!(n.div_ceil(bn)*8)*8usize;
    let height=(comptime!(m.div_ceil(bm))-group).min(8usize);
    let within=tile%comptime!(n.div_ceil(bn)*8);
    let row=(group+within%height)*bm;
    let col=(within/height)*bn;
    let ty=UNIT_POS as usize/16usize;let tx=UNIT_POS as usize%16usize;
    let tm=comptime!(bm/16);let tn=comptime!(bn/16);
    let astage=comptime!(if ta {bk*(bm/4+1)}else {bm*(bk/4+1)});
    let bstage=comptime!(if nt {bn*(bk/4+1)}else {bk*(bn/4+1)});
    let mut sa=SharedMemory::<Vector<f32,Const<4>>>::new_aligned(astage*2,32usize);
    let mut sb=SharedMemory::<Vector<f32,Const<4>>>::new_aligned(bstage*2,32usize);
    let scalar_a=sa.to_slice().with_vector_size::<Const<1>>();let scalar_b=sb.to_slice().with_vector_size::<Const<1>>();
    let mut acc=Array::<f32>::new(tm*tn);
    #[unroll] for i in 0usize..(tm*tn) {acc[i]=0.0f32;}
    load_tiles(a,b,as_,bs,indices32,indices64,gathered_a,gathered_b,invalid,outputs,&mut sa,&mut sb,
        row,col,0usize,0usize,batch_a,batch_b,owner_a,owner_b,
        m,n,k,bm,bn,bk,ta,nt,at,bt,ar,ac,br,bc,scaled_a,scaled_b,store_a,store_b,slot_a,slot_b,index_codes);
    for block in 0usize..comptime!(k/bk) {
        let bank=block%2usize;sync_ruda();
        if block+1usize<k/bk {
            load_tiles(a,b,as_,bs,indices32,indices64,gathered_a,gathered_b,invalid,outputs,&mut sa,&mut sb,
                row,col,block+1usize,1usize-bank,batch_a,batch_b,owner_a,owner_b,
                m,n,k,bm,bn,bk,ta,nt,at,bt,ar,ac,br,bc,scaled_a,scaled_b,store_a,store_b,slot_a,slot_b,index_codes);
        }
        #[unroll] for inner in 0usize..bk {
            let mut av=Array::<f32>::new(tm);let mut bv=Array::<f32>::new(tn);
            #[unroll] for i in 0usize..tm {
                let r=ty*comptime!(tm.min(4))+comptime!(i%4+i/4*64);
                if comptime!(ta) {av[i]=scalar_a[bank*astage*4usize+inner*(bm+4usize)+r][0usize];}
                else {av[i]=scalar_a[bank*astage*4usize+r*(bk+4usize)+inner][0usize];}
            }
            #[unroll] for j in 0usize..tn {
                let c=tx*comptime!(tn.min(4))+comptime!(j%4+j/4*64);
                if comptime!(nt) {bv[j]=scalar_b[bank*bstage*4usize+c*(bk+4usize)+inner][0usize];}
                else {bv[j]=scalar_b[bank*bstage*4usize+inner*(bn+4usize)+c][0usize];}
            }
            #[unroll] for i in 0usize..tm {
                #[unroll] for j in 0usize..tn {acc[i*tn+j]=fma(av[i],bv[j],acc[i*tn+j]);}
            }
        }
    }
    if comptime!(vector==4) {store_result::<Const<4>>(out,&acc,batch*m*n,row,col,m,n,tm,tn);}
    else if comptime!(vector==2) {store_result::<Const<2>>(out,&acc,batch*m*n,row,col,m,n,tm,tn);}
    else {store_result::<Const<1>>(out,&acc,batch*m*n,row,col,m,n,tm,tn);}
}

#[ruda]
fn store_result<V:Size>(out:&mut Tensor<f32>,acc:&Array<f32>,base:usize,row:usize,col:usize,
    #[comptime] m:usize,#[comptime] n:usize,#[comptime] tm:usize,#[comptime] tn:usize)
{
    let ty=UNIT_POS as usize/16usize;let tx=UNIT_POS as usize%16usize;
    let mut packed=out.slice_mut(0usize,out.len()).into_vectorized().with_vector_size::<V>();
    #[unroll] for i in 0usize..tm {
        let r=row+ty*comptime!(tm.min(4))+comptime!(i%4+i/4*64);
        #[unroll] for j in 0usize..comptime!(tn/V::value()) {
            let c=col+tx*comptime!(tn.min(4))+comptime!((j*V::value())%4+(j*V::value())/4*64);
            if r<m && c<n {
                let mut value=Vector::<f32,V>::empty();
                #[unroll] for lane in 0usize..V::value() {value[lane]=acc[i*tn+j*V::value()+lane];}
                packed[(base+r*n+c)/V::value()]=value;
            }
        }
    }
}

struct Operand<'a> {
    data:&'a View,matrix:&'a View,scales:&'a View,block_rows:usize,block_columns:usize,
    scaled:bool,slot:Option<usize>,gathered:Option<&'a View>,indices:Option<&'a View>,
}

fn operand_layout(source:&View,axes:&[usize],repeat:usize,batches:usize)->Option<(bool,usize,usize)> {
    let rank=source.shape.len();
    if rank<2 || axes.len()!=rank || repeat>rank-2 || source.shape.contains(&0)
        || axes[..rank-2].iter().copied().ne(0..rank-2) {return None;}
    let transpose=axes[rank-2]==rank-1 && axes[rank-1]==rank-2;
    if !transpose && axes[rank-2..]!=[rank-2,rank-1] {return None;}
    let count=source.shape[..rank-2].iter().product::<usize>();
    if batches==0 || batches%count!=0 {return None;}
    Some((transpose,source.shape[repeat..rank-2].iter().product(),batches/count))
}

fn matrix_layout(node:&NodeSpec,a:&View,b:&View,out:&View)->Option<((bool,usize,usize),(bool,usize,usize),usize)> {
    if out.shape.len()<2 || a.shape.len()<2 || b.shape.len()<2 {return None;}
    let batches=out.shape[..out.shape.len()-2].iter().product::<usize>();
    if node.op==217 {
        let variants=super::graph_contract::bmm_layout_variants(a.shape.len());
        let code=node.scalar as usize;
        let (aa,ar)=super::graph_contract::bmm_layout_axes(a.shape.len(),code%variants)?;
        let (ba,br)=super::graph_contract::bmm_layout_axes(b.shape.len(),code/variants)?;
        return Some((operand_layout(a,&aa,ar,batches)?,operand_layout(b,&ba,br,batches)?,batches));
    }
    if node.op==207 {
        let code=node.scalar as u32;
        let aa=super::graph_contract::bmm_permutation(code&511)?;
        let ba=super::graph_contract::bmm_permutation(code>>9)?;
        return Some((operand_layout(a,&aa,1,batches)?,operand_layout(b,&ba,1,batches)?,batches));
    }
    if node.op==211 {
        let code=node.scalar as u32;
        let axes=|source:&View,row:usize,col:usize| {
            (0..source.shape.len()).filter(|&d|d!=row && d!=col).chain([row,col]).collect::<Vec<_>>()
        };
        let aa=axes(a,(code&7) as usize,((code>>3)&7) as usize);
        let ba=axes(b,((code>>6)&7) as usize,((code>>9)&7) as usize);
        return Some((operand_layout(a,&aa,a.shape.len()-2,batches)?,operand_layout(b,&ba,b.shape.len()-2,batches)?,batches));
    }
    let (rank,ta,nt)=matmul_layout(node.op)?;
    if a.shape.len()!=rank || b.shape.len()!=rank || out.shape.len()!=rank {return None;}
    Some(((ta,1,1),(nt,1,1),batches))
}

pub(super) fn prepare_matmul(c:&ComputeClient<CudaRuntime>,tensors:&[View],nodes:&[NodeSpec],inputs:usize,start:usize,
    invalid:Option<&View>)
    ->Option<(usize,PreparedKernel<CudaRuntime>)>
{
    let producer=|index:usize| {
        nodes.get(index).is_some_and(|n|n.op==218 || n.op==114 && tensors[inputs+index].dtype==0
            && matches!(tensors[n.a as usize].dtype,1|2|12|13|15)
            || matches!(n.op,134|136) && n.scalar==0.0 && tensors[n.a as usize].shape.len()==2
                && tensors[n.a as usize].dtype<=2 && tensors[n.b as usize].shape.len()<=1
                && tensors[inputs+index].shape.len()==2)
    };
    if !producer(start) {return None;}
    let mut end=start+1;while end-start<4 && producer(end) {end+=1;}
    let mm=nodes.get(end)?;
    if mm.a==mm.b {return None;}
    let lower=inputs+start;let upper=inputs+end;
    let conversions=(lower..upper).filter(|&id|matches!(nodes[id-inputs].op,114|218)).collect::<Vec<_>>();
    let mut consumed=BTreeSet::new();
    let mut operand=|id:usize|->Option<Operand<'_>> {
        if id>=upper {return None;}
        let matrix=&tensors[id];if matrix.dtype!=0 {return None;}
        let mut root=id;let mut slot=None;let mut scales=None;let mut block_rows=1;let mut block_columns=1;
        if root>=lower && matches!(nodes[root-inputs].op,114|218) {
            let node=&nodes[root-inputs];slot=conversions.iter().position(|&value|value==root);
            if node.op==218 {
                if node.b as usize>=lower {return None;}
                let source=&tensors[node.a as usize];let scale=&tensors[node.b as usize];
                block_rows=node.scalar as usize;
                block_columns=source.shape[1]/(scale.len/(source.shape[0]/block_rows));scales=Some(scale);
            }
            consumed.insert(root);root=node.a as usize;
        }
        let mut gathered=None;let mut indices=None;
        if root>=lower {
            let node=&nodes[root-inputs];
            if !matches!(node.op,134|136) || node.scalar!=0.0 || node.a as usize>=lower || node.b as usize>=lower {return None;}
            gathered=Some(&tensors[root]);indices=Some(&tensors[node.b as usize]);
            consumed.insert(root);root=node.a as usize;
        }
        let data=&tensors[root];
        Some(Operand{data,matrix,scales:scales.unwrap_or(data),block_rows,block_columns,scaled:scales.is_some(),slot,gathered,indices})
    };
    let a=operand(mm.a as usize)?;let b=operand(mm.b as usize)?;let out=&tensors[upper];
    if consumed.len()!=end-start || (a.indices.is_some() || b.indices.is_some()) && invalid.is_none()
        || a.gathered.zip(b.gathered).is_some_and(|(x,y)|super::graph_fusion::overlaps(x,y)) {return None;}
    let ((ta,span_a,repeats_a),(nt,span_b,repeats_b),batches)=matrix_layout(mm,a.matrix,b.matrix,out)?;
    let (m,k,n)=(out.shape[out.shape.len()-2],a.matrix.shape[a.matrix.shape.len()-if ta {2}else{1}],out.shape[out.shape.len()-1]);
    if out.dtype!=0 || batches==0 || m==0 || n==0 || k==0 || k%4!=0 {return None;}
    let outputs=&tensors[lower..=upper];
    if outputs.iter().enumerate().any(|(i,x)|outputs[..i].iter().any(|y|super::graph_fusion::overlaps(x,y))
        || [a.data,a.scales,b.data,b.scales].iter().copied().chain(a.indices).chain(b.indices)
            .any(|y|super::graph_fusion::overlaps(x,y))) {return None;}
    if [a.data,b.data].iter().any(|v|!v.contiguous() || v.shape[v.shape.len()-1]%4!=0
        || v.handle.offset_start.unwrap_or(0)%(4*element_bytes(v.dtype)) as u64!=0)
        || outputs[..outputs.len()-1].iter().any(|v|v.handle.offset_start.unwrap_or(0)%(4*element_bytes(v.dtype)) as u64!=0) {return None;}
    let h=&c.properties().hardware;
    let mut bm=if m%128==0 {128}else if m>=64 {64}else {32};
    let mut bn=if n%128==0 {128}else if n>=64 {64}else {32};
    let sms=h.num_streaming_multiprocessors.unwrap_or(108).max(1) as usize;
    while m.div_ceil(bm)*n.div_ceil(bn)*batches<sms && (bm>32 || bn>32) {
        if bm>=bn && bm>32 {bm/=2;}else {bn/=2;}
    }
    let bk=if k%8!=0 {4}else if m.min(n)<128 || bm.min(bn)<64 {8}else if k%32==0 {32}else if k%16==0 {16}else {8};
    let shared=(if ta {bk*(bm/4+1)}else {bm*(bk/4+1)})+(if nt {bn*(bk/4+1)}else {bk*(bn/4+1)});
    let blocks=m.div_ceil(bm)*n.div_ceil(bn);
    let offset=out.handle.offset_start.unwrap_or(0);
    let vector=if bn>=64 && n%4==0 && offset%16==0 {4}else if n%2==0 && offset%8==0 {2}else {1};
    if h.max_units_per_ruda<256 || h.max_ruda_dim.0<256 || shared*32>h.max_shared_memory_size
        || blocks>h.max_ruda_count.0 as usize || batches>h.max_ruda_count.1 as usize {return None;}
    let mut indices32=Vec::new();let mut indices64=Vec::new();let mut index_codes=Vec::new();
    for source in [&a,&b] {
        if let Some(ids)=source.indices {
            let bucket=if ids.dtype==4 {&mut indices64}else {&mut indices32};
            index_codes.extend([ids.dtype as usize,bucket.len(),source.data.shape[0]]);bucket.push(ids);
        } else {index_codes.extend([0,0,0]);}
    }
    macro_rules! launch {($a:ty,$b:ty)=>{tiled::prepare::<$a,$b,CudaRuntime>(c,RudaCount::Static(blocks as u32,batches as u32,1),RudaDim::new_1d(256),
        a.data.arg(),b.data.arg(),a.scales.arg(),b.scales.arg(),
        indices32.iter().map(|v|v.arg()).collect::<SequenceArg<CudaRuntime,Tensor<i32>>>(),
        indices64.iter().map(|v|v.arg()).collect::<SequenceArg<CudaRuntime,Tensor<i64>>>(),
        a.gathered.iter().map(|v|v.arg()).collect::<SequenceArg<CudaRuntime,Tensor<$a>>>(),
        b.gathered.iter().map(|v|v.arg()).collect::<SequenceArg<CudaRuntime,Tensor<$b>>>(),invalid.unwrap_or(out).arg(),
        conversions.iter().map(|&id|tensors[id].arg()).collect::<SequenceArg<CudaRuntime,Tensor<f32>>>(),out.arg(),
        m,n,k,bm,bn,bk,ta,nt,a.data.dtype,b.data.dtype,a.block_rows,a.block_columns,b.block_rows,b.block_columns,
        a.scaled,b.scaled,a.slot.is_some(),b.slot.is_some(),a.slot.unwrap_or(0),b.slot.unwrap_or(0),vector,
        span_a,span_b,repeats_a,repeats_b,index_codes)};}
    macro_rules! right {($a:ty)=>{match b.data.dtype {0=>launch!($a,f32),1=>launch!($a,f16),2=>launch!($a,bf16),12|13|15=>launch!($a,u8),_=>return None}};}
    let kernel=match a.data.dtype {0=>right!(f32),1=>right!(f16),2=>right!(bf16),12|13|15=>right!(u8),_=>return None};
    Some((end-start+1,kernel))
}
